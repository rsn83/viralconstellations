"""
Step 6: Evaluate at multiple horizons.

Usage:
  python scripts/06_evaluate.py
  python scripts/06_evaluate.py --config configs/colab_2022_test.yaml
"""

import sys, json, argparse, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()

import yaml, numpy as np, torch
from scipy.stats import pearsonr

from viralconstellations.model.model import (
    ConstellationTransformer, PopulationEncoder,
    LengthToGoHead, FrequencyRegressionHead, CooccurrenceRegressionHead,
    TrajectoryEmbeddingCache,
    generate_from_hidden, independence_baseline_generate,
    independence_cooccurrence, evaluate_cooccurrence,
)
from viralconstellations.model.trajectory import (
    build_trajectory_encoder, TransitionModel,
)
from viralconstellations.frontier.frontier import (
    frontier_coverage_benchmark, score_candidates_neural,
    compute_occupied, compute_frontier, compute_new_constellations,
    evaluate_ranking,
)
from viralconstellations.eval.metrics import all_metrics_categorical


def log(msg): print(msg, flush=True)

def load_mat(d, m):
    p = d / f"{m}.npy"; return np.load(p) if p.exists() else None
def load_pf(d, m):
    p = d / f"{m}_posfreq.npy"; return np.load(p) if p.exists() else None
def mplus(s, h):
    y,mo=int(s[:4]),int(s[5:7]); mo+=h; y+=(mo-1)//12; mo=(mo-1)%12+1
    return f"{y:04d}-{mo:02d}"
def mminus(s, k=1):
    y,mo=int(s[:4]),int(s[5:7]); mo-=k
    while mo<=0: mo+=12; y-=1
    return f"{y:04d}-{mo:02d}"


def main():
    cfg        = yaml.safe_load(open(ROOT / args.config))
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    ckpt_dir   = ROOT / cfg["train"]["checkpoint_dir"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("="*60)
    log(f"Evaluation | config: {args.config}")
    log(f"Device: {device}")
    log("="*60)

    ckpt  = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    P     = ckpt["n_positions"]
    max_h = ckpt["max_h"]
    T     = cfg["model"]["diffusion_T"]
    mode  = ckpt["traj_cfg"]["mode"]
    W     = ckpt["traj_cfg"]["window_size"]
    d     = ckpt["model_cfg"]["d_model"]
    eval_hs = [h for h in cfg["eval"]["horizons"] if h <= max_h]
    n_gen   = cfg["eval"]["n_generate"]

    log(f"Checkpoint epoch {ckpt['epoch']}  val={ckpt['best_val']:.5f}")
    log(f"Horizons to evaluate: {eval_hs}")

    # Rebuild models
    encoder     = PopulationEncoder(P, d, ckpt["model_cfg"]["phi_hidden"]).to(device)
    traj_enc    = build_trajectory_encoder(
        mode, d, ckpt["traj_cfg"]["gru_hidden"], W, P).to(device)
    transition  = TransitionModel(d).to(device)
    model       = ConstellationTransformer(
        P, d, ckpt["model_cfg"]["n_heads"], ckpt["model_cfg"]["n_layers"],
        0.0, T, max_h).to(device)
    length_head = LengthToGoHead(d, max_h).to(device)
    freq_head   = FrequencyRegressionHead(d, P).to(device)
    cooc_head   = CooccurrenceRegressionHead(d, P).to(device)

    model.load_state_dict(ckpt["model_state"])
    encoder.load_state_dict(ckpt["encoder_state"])
    traj_enc.load_state_dict(ckpt["traj_state"])
    transition.load_state_dict(ckpt["transition_state"])
    length_head.load_state_dict(ckpt["length_state"])
    freq_head.load_state_dict(ckpt["freq_state"])
    if "cooc_state" in ckpt:
        cooc_head.load_state_dict(ckpt["cooc_state"])
    for m in [model, encoder, traj_enc, transition, length_head, freq_head, cooc_head]:
        m.eval()

    # Context month and cache
    context_month = ckpt["train_months"][-1]
    all_avail     = sorted(p.stem for p in matrix_dir.glob("*.npy")
                           if "_posfreq" not in p.stem)
    ctx_idx       = all_avail.index(context_month)
    cache_months  = sorted(set(
        all_avail[max(0, ctx_idx-w)] for w in range(W+1)
    ))
    cache = TrajectoryEmbeddingCache(
        encoder,
        {m: load_mat(matrix_dir, m) for m in cache_months},
        {m: load_pf(matrix_dir, m)  for m in cache_months},
        device, cfg["model"]["deepsets_batch_size"],
    )
    cache.refresh()

    mat_ctx = load_mat(matrix_dir, context_month)
    pf_ctx  = load_pf(matrix_dir, context_month)
    log(f"\nContext: {context_month}  ({len(mat_ctx):,} seqs  "
        f"mean_muts={(mat_ctx>0).sum(1).mean():.1f})")

    # Precompute hidden states for all horizons at once
    log("Computing hidden state trajectory...")
    with torch.no_grad():
        window  = cache.get_window(context_month, W)
        h_t     = traj_enc(window)
        _, states = transition(h_t, max(eval_hs))

    results = {}

    for h in eval_hs:
        target   = mplus(context_month, h)
        real_mat = load_mat(matrix_dir, target)
        real_pf  = load_pf(matrix_dir, target)
        if real_mat is None:
            log(f"\nh={h}: {target} not available, skipping")
            continue

        h_state = states[min(h, len(states)-1)]

        log(f"\n{'='*60}")
        log(f"Horizon h={h}  ({context_month} → {target})")
        log(f"Real sequences: {len(real_mat):,}  "
            f"mean_muts={(real_mat>0).sum(1).mean():.1f}")

        with torch.no_grad():
            # Frequency prediction
            pred_pf = freq_head(h_state.unsqueeze(0)).squeeze(0).cpu().numpy()
            r_freq, _ = pearsonr(1-pred_pf[:,0], 1-real_pf[:,0])
            log(f"Freq Pearson r:      {r_freq:.4f}")

            # Co-occurrence prediction
            pred_coo  = cooc_head.predict_matrix(h_state)
            real_coo  = ((mat_real_bin := (real_mat>0).astype(np.float32)).T
                         @ mat_real_bin) / len(real_mat)
            indep_coo = independence_cooccurrence(real_pf)
            coo_eval  = evaluate_cooccurrence(pred_coo, real_coo, indep_coo, P)
            log(f"Co-occ Pearson r:   model={coo_eval['coo_pearson_r_model']:.4f}  "
                f"indep={coo_eval['coo_pearson_r_indep']:.4f}  "
                f"Δ={coo_eval['delta_coo_pearson_r']:+.4f}  "
                f"{'✓ model wins' if coo_eval['model_beats_indep'] else '✗ indep wins'}")

        # Frontier coverage — Hamming 1 and 2
        cov   = frontier_coverage_benchmark(mat_ctx, real_mat, P, hamming_r=1)
        cov2  = frontier_coverage_benchmark(mat_ctx, real_mat, P, hamming_r=2)
        log(f"Frontier coverage H=1: {cov['frontier_coverage']:.4f} "
            f"({cov['n_in_frontier']}/{cov['n_new']} new in F(O_t))")
        log(f"Frontier coverage H=2: {cov2['frontier_coverage']:.4f} "
            f"({cov2['n_in_frontier']}/{cov2['n_new']} within 2 mutations)")

        # Neural frontier scorer
        occupied_t   = compute_occupied(mat_ctx, top_k=200)
        frontier     = compute_frontier(occupied_t, P)
        _, new_in_th = compute_new_constellations(mat_ctx, real_mat)
        neural_met   = {}
        if frontier and new_in_th:
            candidates    = list(frontier.keys())
            neural_scores = score_candidates_neural(
                model, candidates, pred_pf, h_state, h, P, device
            )
            neural_ranked = sorted(zip(candidates, neural_scores), key=lambda x:-x[1])
            neural_met    = evaluate_ranking(neural_ranked, new_in_th)
            log(f"Frontier neural AP: {neural_met.get('AP',0):.4f}  "
                f"p@10={neural_met.get('precision@10',0):.4f}  "
                f"random={neural_met.get('random_baseline_P',0):.4f}")

        # Generative evaluation
        log(f"Generating {n_gen} sequences...")
        gen_model = generate_from_hidden(model, length_head, h_state, h,
                                         n_gen, P, T, device)
        gen_bl    = independence_baseline_generate(pf_ctx, n_gen)
        mk = dict(top_k=cfg["eval"]["top_k_sites_pairs"],
                  mmd_n_sub=cfg["eval"]["mmd_subsample"])
        m_model = all_metrics_categorical(gen_model, real_mat, **mk)
        m_bl    = all_metrics_categorical(gen_bl,    real_mat, **mk)
        delta   = m_model.get("pairwise_coo_r",0) - m_bl.get("pairwise_coo_r",0)
        log(f"Pairwise co-occ r:  model={m_model.get('pairwise_coo_r',0):.4f}  "
            f"baseline={m_bl.get('pairwise_coo_r',0):.4f}  Δ={delta:+.4f}")
        log(f"Mean mut count:     model={m_model.get('mean_mut_count',0):.1f}  "
            f"real={m_model.get('mean_mut_count_real',0):.1f}")

        results[f"h={h}"] = {
            "target_month":         target,
            "freq_pearson_r":       float(r_freq),
            "cooccurrence":         coo_eval,
            "frontier_coverage_H1": float(cov["frontier_coverage"]),
            "frontier_coverage_H2": float(cov2["frontier_coverage"]),
            "frontier_neural":    neural_met,
            "generative_model":   m_model,
            "generative_baseline":m_bl,
            "delta_pairwise_coo": float(delta),
        }

    out = {
        "context_month": context_month,
        "config":        args.config,
        "n_generated":   n_gen,
        "results":       results,
    }
    p = ROOT / "evaluation_results.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    log(f"\nWrote: {p}")

    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)
    for hk, r in results.items():
        log(f"\n{hk}:")
        log(f"  Freq Pearson r:       {r['freq_pearson_r']:.4f}")
        log(f"  Co-occ model beats indep: {r['cooccurrence']['model_beats_indep']}"
            f"  (Δ={r['cooccurrence']['delta_coo_pearson_r']:+.4f})")
        log(f"  Frontier coverage:    {r['frontier_coverage']:.4f}")
        if r["frontier_neural"]:
            log(f"  Neural frontier AP:   {r['frontier_neural'].get('AP',0):.4f}")
        log(f"  Generative Δ co-occ:  {r['delta_pairwise_coo']:+.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
