"""
Step 7: Walk-forward evaluation across all available windows.

For each context month c (every available month with enough history):
  - Frequency prediction: Pearson r between predicted and real posfreq
  - Frontier coverage: fraction of new constellations in F(O_c)
  - Frontier scoring: precision@k and AP for three scorers:
      random baseline, logistic regression, neural model

Generation is skipped by default (n_gen=0) since the length head
is untrained and would produce degenerate 1-mutation sequences.
Set n_gen > 0 only after the length head is properly trained.

Usage:
  python scripts/07_walk_forward_eval.py --config configs/colab_2022_test.yaml
"""

import sys, json, argparse, traceback
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--n_gen", type=int, default=0,
                    help="Sequences to generate per window (0 = skip generation)")
args = parser.parse_args()

import yaml, numpy as np, torch
from scipy.stats import pearsonr

from viralconstellations.model.model import (
    ConstellationTransformer, PopulationEncoder,
    LengthToGoHead, FrequencyRegressionHead, CooccurrenceRegressionHead,
    TrajectoryEmbeddingCache,
)
from viralconstellations.model.trajectory import (
    build_trajectory_encoder, TransitionModel,
)
from viralconstellations.frontier.frontier import (
    compute_occupied, compute_frontier,
    compute_new_constellations, frontier_coverage_benchmark,
    score_candidates_neural, LogisticFrontierScorer, evaluate_ranking,
)


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


@torch.no_grad()
def get_state_and_freq(freq_head, traj_enc, transition, cache,
                       ctx_month, h, mode, W, device):
    if mode == "gru":
        window = cache.get_window(ctx_month, W)
        h_t    = traj_enc(window)
        _, states = transition(h_t, h)
        h_state = states[min(h, len(states)-1)]
    else:
        h_state = traj_enc(
            cache.get_posfreq(ctx_month),
            cache.get_posfreq_prev(ctx_month)
        )
    pred_pf = freq_head(h_state.unsqueeze(0)).squeeze(0).cpu().numpy()
    return h_state, pred_pf


def main():
    cfg        = yaml.safe_load(open(ROOT / args.config))
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    ckpt_dir   = ROOT / cfg["train"]["checkpoint_dir"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gen      = args.n_gen

    log("="*60)
    log(f"Walk-forward evaluation | config: {args.config}")
    log(f"Device: {device}  n_gen={n_gen}")
    log("="*60)

    ckpt  = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    P     = ckpt["n_positions"]
    max_h = ckpt["max_h"]
    T     = cfg["model"]["diffusion_T"]
    mode  = ckpt["traj_cfg"]["mode"]
    W     = ckpt["traj_cfg"]["window_size"]
    d     = ckpt["model_cfg"]["d_model"]
    eval_horizons = [h for h in cfg["eval"]["horizons"] if h <= max_h]
    min_train     = 6

    log(f"P={P}  mode={mode}  W={W}  horizons={eval_horizons}")

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
    cooc_rank   = ckpt.get("cooc_rank", 16)
    cooc_head   = CooccurrenceRegressionHead(d, P, rank=cooc_rank).to(device)

    model.load_state_dict(ckpt["model_state"])
    encoder.load_state_dict(ckpt["encoder_state"])
    traj_enc.load_state_dict(ckpt["traj_state"])
    transition.load_state_dict(ckpt["transition_state"])
    length_head.load_state_dict(ckpt["length_state"])
    freq_head.load_state_dict(ckpt["freq_state"])
    if "cooc_state" in ckpt:
        cooc_head.load_state_dict(ckpt["cooc_state"])

    for m in [model, encoder, traj_enc, transition,
              length_head, freq_head, cooc_head]:
        m.eval()

    # Load all months
    all_months = sorted(
        p.stem for p in matrix_dir.glob("*.npy")
        if "_posfreq" not in p.stem
    )
    log(f"Available months: {len(all_months)}  "
        f"({all_months[0]} → {all_months[-1]})")

    all_mats  = {m: load_mat(matrix_dir, m)  for m in all_months}
    all_freqs = {m: load_pf(matrix_dir, m)   for m in all_months}

    cache = TrajectoryEmbeddingCache(
        encoder, all_mats, all_freqs, device,
        cfg["model"]["deepsets_batch_size"]
    )
    cache.refresh()

    # Progressive logistic regression scorer
    lr_scorer = LogisticFrontierScorer()
    lr_X, lr_y = [], []

    results_by_h = defaultdict(list)
    cutoffs = all_months[min_train : -max(eval_horizons)]
    log(f"Evaluating {len(cutoffs)} windows...")

    for c_idx, ctx_month in enumerate(cutoffs):
        mat_ctx = all_mats[ctx_month]
        pf_ctx  = all_freqs[ctx_month]
        prev_m  = mminus(ctx_month, 1)
        pf_prev = all_freqs.get(prev_m, None)
        if pf_prev is None:
            pf_prev = pf_ctx

        for h in eval_horizons:
            target  = mplus(ctx_month, h)
            if target not in all_mats or all_mats[target] is None:
                continue
            mat_th  = all_mats[target]
            real_pf = all_freqs[target]
            if real_pf is None:
                continue

            try:
                # Frequency prediction
                h_state, pred_pf = get_state_and_freq(
                    freq_head, traj_enc, transition, cache,
                    ctx_month, h, mode, W, device
                )
                r_freq, _ = pearsonr(1-pred_pf[:,0], 1-real_pf[:,0])

                # Frontier coverage H=1 and H=2
                cov1 = frontier_coverage_benchmark(mat_ctx, mat_th, P, hamming_r=1)
                cov2 = frontier_coverage_benchmark(mat_ctx, mat_th, P, hamming_r=2)

                # Three scorers
                occupied_t   = compute_occupied(mat_ctx, top_k=200)
                frontier     = compute_frontier(occupied_t, P)
                _, new_in_th = compute_new_constellations(mat_ctx, mat_th)

                neural_met = random_met = lr_met = {}

                if frontier and new_in_th:
                    candidates = list(frontier.keys())

                    # Random baseline
                    import random as rnd
                    shuffled = candidates.copy()
                    rnd.shuffle(shuffled)
                    random_met = evaluate_ranking(
                        [(c, 0.0) for c in shuffled], new_in_th
                    )

                    # Neural scorer
                    neural_scores = score_candidates_neural(
                        model, candidates, pred_pf, h_state, h, P, device
                    )
                    neural_ranked = sorted(
                        zip(candidates, neural_scores), key=lambda x: -x[1]
                    )
                    neural_met = evaluate_ranking(neural_ranked, new_in_th)

                    # Logistic regression
                    if lr_scorer.fitted:
                        lr_ranked = lr_scorer.score(
                            mat_ctx, pred_pf, pf_prev, P
                        )
                        if lr_ranked:
                            lr_met = evaluate_ranking(lr_ranked, new_in_th)

                results_by_h[h].append({
                    "context":              ctx_month,
                    "target":               target,
                    "freq_pearson_r":       float(r_freq),
                    "frontier_coverage_H1": float(cov1["frontier_coverage"]),
                    "frontier_coverage_H2": float(cov2["frontier_coverage"]),
                    "n_new":                cov1["n_new"],
                    "n_frontier":           cov1["n_frontier"],
                    "neural_AP":            float(neural_met.get("AP", 0)),
                    "neural_p10":           float(neural_met.get("precision@10", 0)),
                    "lr_AP":                float(lr_met.get("AP", 0)),
                    "random_AP":            float(random_met.get("random_baseline_P", 0)),
                })

                # Collect logistic regression training data (h=1)
                if h == 1:
                    X, y = lr_scorer.collect(
                        mat_ctx, mat_th, pred_pf, pf_prev, P
                    )
                    if len(X) > 0:
                        lr_X.append(X); lr_y.append(y)

            except Exception as e:
                log(f"  Error at {ctx_month} h={h}: {e}")
                traceback.print_exc()
                continue

        # Refit logistic scorer every 6 windows
        if c_idx > 0 and c_idx % 6 == 0 and lr_X:
            X_all = np.concatenate(lr_X)
            y_all = np.concatenate(lr_y)
            if y_all.sum() > 0:
                lr_scorer.fit(X_all, y_all)

        if c_idx % 10 == 0:
            log(f"  Window {c_idx+1}/{len(cutoffs)}  ctx={ctx_month}")

    # Summary
    log("\n" + "="*70)
    log("WALK-FORWARD RESULTS")
    log("="*70)

    summary = {}
    for h in eval_horizons:
        rows = results_by_h[h]
        if not rows: continue

        freq_rs = [r["freq_pearson_r"]       for r in rows]
        cov1s   = [r["frontier_coverage_H1"] for r in rows]
        cov2s   = [r["frontier_coverage_H2"] for r in rows]
        n_ap    = [r["neural_AP"]            for r in rows if r["neural_AP"] > 0]
        l_ap    = [r["lr_AP"]               for r in rows if r["lr_AP"] > 0]
        r_ap    = [r["random_AP"]            for r in rows if r["random_AP"] > 0]
        n_p10   = [r["neural_p10"]           for r in rows if r["neural_p10"] > 0]

        log(f"\nHorizon h={h}  ({len(rows)} windows):")
        log(f"  Freq Pearson r:         "
            f"{np.mean(freq_rs):.4f} ± {np.std(freq_rs):.4f}")
        log(f"  Frontier coverage H=1:  "
            f"{np.mean(cov1s):.4f} ± {np.std(cov1s):.4f}")
        log(f"  Frontier coverage H=2:  "
            f"{np.mean(cov2s):.4f} ± {np.std(cov2s):.4f}")
        log(f"  --- Frontier scoring ---")
        if r_ap:
            log(f"  Random baseline AP:     {np.mean(r_ap):.4f}")
        if l_ap:
            log(f"  Logistic regression AP: {np.mean(l_ap):.4f} ± {np.std(l_ap):.4f}")
        if n_ap:
            log(f"  Neural model AP:        {np.mean(n_ap):.4f} ± {np.std(n_ap):.4f}")
        if n_p10:
            log(f"  Neural precision@10:    {np.mean(n_p10):.4f}")

        summary[f"h={h}"] = {
            "n_windows":          len(rows),
            "freq_pearson_r":     {"mean": float(np.mean(freq_rs)),
                                   "std":  float(np.std(freq_rs))},
            "frontier_H1":        {"mean": float(np.mean(cov1s)),
                                   "std":  float(np.std(cov1s))},
            "frontier_H2":        {"mean": float(np.mean(cov2s)),
                                   "std":  float(np.std(cov2s))},
            "neural_AP":          {"mean": float(np.mean(n_ap))}  if n_ap else None,
            "logistic_AP":        {"mean": float(np.mean(l_ap))}  if l_ap else None,
            "random_AP":          {"mean": float(np.mean(r_ap))}  if r_ap else None,
        }

    # Logistic regression feature importances
    if lr_scorer.fitted:
        imps = lr_scorer.feature_importances()
        log("\nLogistic Regression Feature Importances:")
        for name, coef in sorted(imps.items(), key=lambda x: -abs(x[1])):
            log(f"  {name:<30} {coef:+.4f}")
        summary["lr_feature_importances"] = imps

    out_path = ROOT / "walk_forward_results.json"
    out_path.write_text(json.dumps({
        "n_windows":      len(cutoffs),
        "eval_horizons":  eval_horizons,
        "n_gen":          n_gen,
        "results":        summary,
    }, indent=2, default=str))
    log(f"\nWrote: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
