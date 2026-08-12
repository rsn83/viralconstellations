"""
Step 7: Walk-forward evaluation with frontier scoring.

Evaluation design (monthly walk-forward):
  For each cutoff month c from month MIN_TRAIN+1 to last available:
    1. Training data:  all months up to c
    2. Context month:  c (last training month)
    3. Target months:  c+1, c+2, c+3, c+6

  For each (c, h) pair where target exists:
    A. Frequency prediction:
       Use FrequencyRegressionHead → compare to real posfreq at c+h
       Metric: Pearson r between predicted and real mutation rates

    B. Frontier coverage benchmark:
       Compute F(O_c) → check what fraction of new constellations
       at c+h are in the frontier
       Metric: frontier_coverage (should be >0.8 if framing is valid)

    C. Frontier scoring:
       Train logistic regression on historical windows before c
       Score F(O_c) candidates → evaluate against new_in_th
       Metric: precision@10, precision@50, average_precision vs random baseline

    D. Co-occurrence prediction (generative model):
       Generate 500 sequences from h_{c+h} → compare to real
       Metric: pairwise co-occurrence Pearson r vs independence baseline

Results averaged across all windows, reported per horizon h.

Note: we use the SINGLE pre-trained model for all windows.
This is slightly optimistic for early windows (model saw later data)
but practical — full walk-forward retraining would require 70+ training runs.
For a proper paper, train on first half, evaluate on second half only.
"""

import sys, json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from viralconstellations.model.model import (
    ConstellationTransformer, PopulationEncoder,
    LengthToGoHead, FrequencyRegressionHead,
    TrajectoryEmbeddingCache,
    independence_baseline_generate, generate_from_hidden, N_RESIDUES,
)
from viralconstellations.model.trajectory import (
    build_trajectory_encoder, TransitionModel,
)
from viralconstellations.frontier.frontier import (
    compute_occupied, compute_frontier,
    compute_new_constellations, frontier_coverage_benchmark,
    LogisticFrontierScorer, extract_features, FEATURE_NAMES,
)
from viralconstellations.eval.metrics import all_metrics_categorical


def load_mat(d, m):
    p = d / f"{m}.npy"; return np.load(p) if p.exists() else None
def load_posfreq(d, m):
    p = d / f"{m}_posfreq.npy"; return np.load(p) if p.exists() else None
def month_plus(s, h):
    y, mo = int(s[:4]), int(s[5:7])
    mo += h; y += (mo-1)//12; mo = (mo-1)%12+1
    return f"{y:04d}-{mo:02d}"
def month_minus(s, k):
    y, mo = int(s[:4]), int(s[5:7])
    mo -= k
    while mo <= 0: mo += 12; y -= 1
    return f"{y:04d}-{mo:02d}"


@torch.no_grad()
def get_freq_prediction(freq_head, traj_enc, transition, encoder,
                        cache, context_month, h, mode, W, device):
    """Run filter → transition → freq head → return (P, 21) predicted posfreq."""
    if mode == "gru":
        window  = cache.get_window(context_month, W)
        h_t     = traj_enc(window)
        _, states = transition(h_t, h)
        h_state = states[min(h, len(states)-1)]
    else:
        pf_t    = cache.get_posfreq(context_month)
        pf_prev = cache.get_posfreq_prev(context_month)
        h_state = traj_enc(pf_t, pf_prev)

    pred_pf = freq_head(h_state.unsqueeze(0)).squeeze(0)   # (P, 21)
    return pred_pf.cpu().numpy(), h_state


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args   = parser.parse_args()
    cfg    = yaml.safe_load(open(ROOT / args.config))
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    ckpt_dir   = ROOT / cfg["train"]["checkpoint_dir"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt     = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    P        = ckpt["n_positions"]
    max_h    = ckpt["max_h"]
    T        = cfg["model"]["diffusion_T"]
    mode     = ckpt["traj_cfg"]["mode"]
    W        = ckpt["traj_cfg"]["window_size"]
    d        = ckpt["model_cfg"]["d_model"]

    # Walk-forward config
    eval_horizons = [h for h in cfg["eval"]["horizons"] if h <= max_h]
    step_months   = 1        # walk forward 1 month at a time
    min_train     = 6        # minimum months of training before first evaluation
    n_gen         = 500      # fewer sequences for speed (2000 too slow × 70 windows)

    print(f"Walk-forward evaluation")
    print(f"P={P}  mode={mode}  W={W}  horizons={eval_horizons}")
    print(f"Device: {device}")

    # Rebuild models
    encoder = PopulationEncoder(P, d, ckpt["model_cfg"]["phi_hidden"]).to(device)
    traj_enc = build_trajectory_encoder(
        mode, d, ckpt["traj_cfg"]["gru_hidden"], W, P
    ).to(device)
    transition = (TransitionModel(d).to(device)
                  if mode == "gru" and "transition_state" in ckpt else None)
    model = ConstellationTransformer(
        P, d, ckpt["model_cfg"]["n_heads"], ckpt["model_cfg"]["n_layers"],
        0.0, T, max_h
    ).to(device)
    length_head = LengthToGoHead(d, max_h).to(device)
    freq_head   = FrequencyRegressionHead(d, P).to(device)

    model.load_state_dict(ckpt["model_state"])
    encoder.load_state_dict(ckpt["encoder_state"])
    traj_enc.load_state_dict(ckpt["traj_state"])
    length_head.load_state_dict(ckpt["length_state"])
    freq_head.load_state_dict(ckpt["freq_state"])
    if transition and "transition_state" in ckpt:
        transition.load_state_dict(ckpt["transition_state"])
    for m in [model, encoder, traj_enc, length_head, freq_head]:
        m.eval()
    if transition: transition.eval()

    # All available months
    all_months = sorted(
        p.stem for p in matrix_dir.glob("*.npy") if "_posfreq" not in p.stem
    )
    print(f"Available months: {len(all_months)}  ({all_months[0]} → {all_months[-1]})")

    # Build full cache
    all_mats   = {m: load_mat(matrix_dir, m)     for m in all_months}
    all_freqs  = {m: load_posfreq(matrix_dir, m) for m in all_months}
    cache = TrajectoryEmbeddingCache(
        encoder, all_mats, all_freqs, device,
        cfg["model"]["deepsets_batch_size"],
    )
    cache.refresh()

    # ── Walk-forward loop ─────────────────────────────────────────────────
    # Collect results per horizon
    results_by_h = defaultdict(list)

    # Frontier scorer: train progressively
    scorer = LogisticFrontierScorer()
    scorer_X, scorer_y = [], []

    cutoff_months = all_months[min_train:-max(eval_horizons)]

    print(f"\nEvaluating {len(cutoff_months)} windows...")

    for c_idx, context_month in enumerate(cutoff_months):

        # ── Compute frontier coverage (no model needed) ───────────────────
        for h in eval_horizons:
            target_month = month_plus(context_month, h)
            if target_month not in all_mats or all_mats[target_month] is None:
                continue

            mat_t  = all_mats[context_month]
            mat_th = all_mats[target_month]

            # A. Frontier coverage benchmark
            coverage = frontier_coverage_benchmark(mat_t, mat_th, P)

            # B. Frequency prediction
            prev_month = month_minus(context_month, 1)
            prev_pf    = all_freqs.get(prev_month, all_freqs[context_month])
            if prev_pf is None:
                prev_pf = all_freqs[context_month]

            pred_pf_np, h_state = get_freq_prediction(
                freq_head, traj_enc, transition, encoder,
                cache, context_month, h, mode, W, device
            )

            real_pf    = all_freqs[target_month]
            pred_rate  = 1.0 - pred_pf_np[:, 0]
            real_rate  = 1.0 - real_pf[:, 0]
            r_freq, _  = pearsonr(pred_rate, real_rate)

            # C. Frontier scoring (only if scorer is fitted)
            frontier_metrics = {}
            if scorer.fitted:
                occupied_t  = compute_occupied(mat_t, top_k=200)
                frontier    = compute_frontier(occupied_t, P)
                _, new_in_th = compute_new_constellations(mat_t, mat_th)

                if frontier and new_in_th:
                    ranked = scorer.score(mat_t, pred_pf_np, prev_pf, P)
                    frontier_metrics = scorer.evaluate_predictions(ranked, new_in_th)

            # D. Co-occurrence (generative model, subset for speed)
            gen_model  = generate_from_hidden(
                model, length_head, h_state, h, n_gen, P, T, device
            )
            gen_bl     = independence_baseline_generate(
                all_freqs[context_month], n_gen
            )
            mk = dict(top_k=50, mmd_n_sub=200)
            m_model = all_metrics_categorical(gen_model,  mat_th, **mk)
            m_bl    = all_metrics_categorical(gen_bl,     mat_th, **mk)

            results_by_h[h].append({
                "context_month":      context_month,
                "target_month":       target_month,
                "freq_pearson_r":     float(r_freq),
                "frontier_coverage":  coverage["frontier_coverage"],
                "n_new":              coverage["n_new"],
                "n_in_frontier":      coverage["n_in_frontier"],
                "pairwise_coo_model": m_model.get("pairwise_coo_r", 0.0),
                "pairwise_coo_bl":    m_bl.get("pairwise_coo_r", 0.0),
                "delta_coo":          m_model.get("pairwise_coo_r",0) - m_bl.get("pairwise_coo_r",0),
                "frontier_scoring":   frontier_metrics,
            })

            # Collect training examples for frontier scorer
            # (train on windows before current, evaluate on current)
            if h == 1:   # collect for h=1 to train scorer
                pf_prev_np = prev_pf if isinstance(prev_pf, np.ndarray) else all_freqs[context_month]
                X, y = scorer.collect(
                    mat_t, mat_th, pred_pf_np, pf_prev_np, P
                )
                if len(X) > 0:
                    scorer_X.append(X)
                    scorer_y.append(y)

        # Refit scorer every 6 months with all data so far
        if c_idx > 0 and c_idx % 6 == 0 and scorer_X:
            X_all = np.concatenate(scorer_X)
            y_all = np.concatenate(scorer_y)
            print(f"\n  Refitting frontier scorer at {context_month}...")
            scorer.fit(X_all, y_all)

        if c_idx % 10 == 0:
            print(f"  Window {c_idx+1}/{len(cutoff_months)}  context={context_month}")

    # ── Aggregate results ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("WALK-FORWARD EVALUATION SUMMARY")
    print("="*70)

    summary = {}
    for h in eval_horizons:
        rows = results_by_h[h]
        if not rows:
            continue

        freq_rs      = [r["freq_pearson_r"]     for r in rows]
        coverages    = [r["frontier_coverage"]   for r in rows]
        delta_coos   = [r["delta_coo"]           for r in rows]
        coo_models   = [r["pairwise_coo_model"]  for r in rows]

        # Frontier scoring metrics (only where scorer was fitted)
        ap_scores = [r["frontier_scoring"].get("average_precision", None)
                     for r in rows if r["frontier_scoring"]]
        ap_scores = [x for x in ap_scores if x is not None]

        p10_scores = [r["frontier_scoring"].get("precision@10", None)
                      for r in rows if r["frontier_scoring"]]
        p10_scores = [x for x in p10_scores if x is not None]

        random_bl  = [r["frontier_scoring"].get("random_baseline_precision", None)
                      for r in rows if r["frontier_scoring"]]
        random_bl  = [x for x in random_bl if x is not None]

        print(f"\nHorizon h={h} ({len(rows)} windows):")
        print(f"  Freq Pearson r:          {np.mean(freq_rs):.4f} ± {np.std(freq_rs):.4f}")
        print(f"  Frontier coverage:        {np.mean(coverages):.4f} ± {np.std(coverages):.4f}")
        print(f"  Pairwise co-occ (model):  {np.mean(coo_models):.4f}")
        print(f"  Δ co-occ (model-baseline):{np.mean(delta_coos):+.4f} ± {np.std(delta_coos):.4f}")
        if ap_scores:
            print(f"  Frontier AP:              {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")
            print(f"  Frontier precision@10:    {np.mean(p10_scores):.4f} ± {np.std(p10_scores):.4f}")
            if random_bl:
                print(f"  Random baseline prec:     {np.mean(random_bl):.4f}")

        summary[f"h={h}"] = {
            "n_windows":             len(rows),
            "freq_pearson_r_mean":   float(np.mean(freq_rs)),
            "freq_pearson_r_std":    float(np.std(freq_rs)),
            "frontier_coverage_mean":float(np.mean(coverages)),
            "frontier_coverage_std": float(np.std(coverages)),
            "delta_coo_mean":        float(np.mean(delta_coos)),
            "delta_coo_std":         float(np.std(delta_coos)),
            "frontier_ap_mean":      float(np.mean(ap_scores)) if ap_scores else None,
            "frontier_p10_mean":     float(np.mean(p10_scores)) if p10_scores else None,
            "random_baseline_mean":  float(np.mean(random_bl)) if random_bl else None,
            "per_window":            rows,
        }

    # Final frontier scorer feature importances
    if scorer.fitted:
        print("\nFrontier Scorer Feature Importances:")
        for name, coef in zip(FEATURE_NAMES, scorer.model.coef_[0]):
            print(f"  {name:<30} {coef:+.4f}")

    out = {
        "n_total_windows":   len(cutoff_months),
        "eval_horizons":     eval_horizons,
        "n_generated_per_window": n_gen,
        "results_by_horizon": summary,
    }
    out_path = ROOT / "walk_forward_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
