"""
Step 6: Evaluate at multiple horizons.

For each horizon h:

  A) Frequency prediction (deterministic):
     Filter → Transition^h → FreqHead → posfreq_{t+h}
     Metric: Pearson r between predicted and real mutation rates per position.
     Also shows intermediate predictions at h=1,2,...,h-1.
     Answers: "did the model predict which mutations rise/fall?"

  B) Generative evaluation (sampled):
     Same hidden state → ConstellationTransformer → n_gen sequences
     Metric: pairwise co-occurrence Pearson r vs independence baseline.
     Answers: "did the model capture which mutations co-occur?"

Two baselines:
  Static independence: sample each position from month-t posfreq (ignores trajectory)
  Freq-regression:    sample each position from the model's PREDICTED posfreq
                      (uses trajectory but ignores co-occurrence)
  Model beats both → it learned co-occurrence beyond frequency trends.
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml, numpy as np, torch
from scipy.stats import pearsonr

from viralconstellations.model.model import (
    ConstellationTransformer, PopulationEncoder,
    LengthToGoHead, FrequencyRegressionHead,
    TrajectoryEmbeddingCache,
    independence_baseline_generate, generate_from_hidden,
)
from viralconstellations.model.trajectory import (
    build_trajectory_encoder, TransitionModel,
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


@torch.no_grad()
def compute_hidden_states(traj_enc, transition, cache, context_month, mode, W, max_h):
    """Returns list [h_t, h_{t+1}, ..., h_{t+max_h}]."""
    if mode == "gru":
        window  = cache.get_window(context_month, W)
        h_t     = traj_enc(window)
        _, states = transition(h_t, max_h)
    else:
        pf_t    = cache.get_posfreq(context_month)
        pf_prev = cache.get_posfreq_prev(context_month)
        h_t     = traj_enc(pf_t, pf_prev)
        states  = [h_t] * (max_h + 1)
    return states


def main():
    cfg        = yaml.safe_load(open(ROOT / "configs/default.yaml"))
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
    eval_hs  = [h for h in cfg["eval"]["horizons"] if h <= max_h]
    n_gen    = cfg["eval"]["n_generate"]

    print(f"Checkpoint epoch {ckpt['epoch']}  val={ckpt['best_val']:.5f}")
    print(f"Trajectory: {mode}  W={W}  horizons={eval_hs}")

    # Rebuild all components
    encoder = PopulationEncoder(P, d, ckpt["model_cfg"]["phi_hidden"]).to(device)
    traj_enc = build_trajectory_encoder(
        mode, d, ckpt["traj_cfg"]["gru_hidden"], W, P
    ).to(device)
    transition = (TransitionModel(d).to(device)
                  if mode == "gru" and "transition_state" in ckpt else None)
    model       = ConstellationTransformer(
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
    if transition: transition.load_state_dict(ckpt["transition_state"])
    for m in [model, encoder, traj_enc, length_head, freq_head]:
        m.eval()
    if transition: transition.eval()

    # Context month and cache
    context_month = ckpt["train_months"][-1]
    all_avail     = sorted(p.stem for p in matrix_dir.glob("*.npy")
                           if "_posfreq" not in p.stem)
    ctx_idx = all_avail.index(context_month)
    cache_months = sorted(set(
        all_avail[max(0, ctx_idx - w)] for w in range(W + 1)
    ))
    cache = TrajectoryEmbeddingCache(
        encoder,
        {m: load_mat(matrix_dir, m) for m in cache_months},
        {m: load_posfreq(matrix_dir, m) for m in cache_months},
        device, cfg["model"]["deepsets_batch_size"],
    )
    cache.refresh()

    mat_ctx  = load_mat(matrix_dir, context_month)
    pf_ctx   = load_posfreq(matrix_dir, context_month)
    print(f"\nContext: {context_month}  "
          f"({len(mat_ctx):,} seqs  "
          f"mean_muts={(mat_ctx>0).sum(1).mean():.1f})")

    # Compute full hidden state trajectory once
    states = compute_hidden_states(
        traj_enc, transition, cache, context_month, mode, W, max_h
    )

    results = {}
    for h in eval_hs:
        target   = month_plus(context_month, h)
        real_mat = load_mat(matrix_dir, target)
        real_pf  = load_posfreq(matrix_dir, target)
        if real_mat is None:
            print(f"\nh={h}: {target} not available, skipping.")
            continue

        h_state = states[min(h, len(states)-1)]   # (d_model,)

        print(f"\n{'='*62}")
        print(f"h={h}  {context_month} → {target}  "
              f"({len(real_mat):,} seqs  "
              f"mean_muts={(real_mat>0).sum(1).mean():.1f})")

        # ── A: Frequency prediction ───────────────────────────────────────
        with torch.no_grad():
            pred_pf = freq_head(h_state.unsqueeze(0)).squeeze(0).cpu().numpy()

        # Per-position mutation rate = 1 - P(reference)
        pred_rate = 1.0 - pred_pf[:, 0]
        real_rate = 1.0 - real_pf[:, 0]
        r_freq, _ = pearsonr(pred_rate, real_rate)
        mse_freq  = float(np.mean((pred_pf - real_pf)**2))

        print(f"  Freq Pearson r (h={h}): {r_freq:.4f}  MSE: {mse_freq:.6f}")

        # Intermediate step predictions
        for k in range(1, h):
            inter_target = month_plus(context_month, k)
            inter_pf = load_posfreq(matrix_dir, inter_target)
            if inter_pf is None: continue
            with torch.no_grad():
                pred_k = freq_head(
                    states[min(k,len(states)-1)].unsqueeze(0)
                ).squeeze(0).cpu().numpy()
            r_k, _ = pearsonr(1-pred_k[:,0], 1-inter_pf[:,0])
            print(f"  Freq Pearson r (h={k}): {r_k:.4f}")

        # ── B: Generative evaluation ──────────────────────────────────────
        print(f"  Generating {n_gen} sequences (model)...")
        gen_model = generate_from_hidden(
            model, length_head, h_state, h, n_gen, P, T, device
        )

        print(f"  Generating {n_gen} sequences (static independence baseline)...")
        gen_static = independence_baseline_generate(pf_ctx, n_gen)

        print(f"  Generating {n_gen} sequences (freq-regression baseline)...")
        gen_freq   = independence_baseline_generate(pred_pf, n_gen)

        mk = dict(top_k=cfg["eval"]["top_k_sites_pairs"],
                  mmd_n_sub=cfg["eval"]["mmd_subsample"])
        m_model  = all_metrics_categorical(gen_model,  real_mat, **mk)
        m_static = all_metrics_categorical(gen_static, real_mat, **mk)
        m_freq   = all_metrics_categorical(gen_freq,   real_mat, **mk)

        print(f"\n  {'Metric':<30} {'Model':>9} {'FreqBL':>9} {'StaticBL':>9}")
        print("  " + "-"*60)
        for label, key in [
            ("Pos-freq Pearson r",   "pos_freq_r"),
            ("Pairwise co-occ r",    "pairwise_coo_r"),
            ("MMD (lower=better)",   "mmd"),
            ("Frontier H=1",         "frontier_coverage_H1"),
            ("Mean mut count",       "mean_mut_count"),
        ]:
            mv = m_model.get(key, float("nan"))
            fv = m_freq.get(key, float("nan"))
            sv = m_static.get(key, float("nan"))
            print(f"  {label:<30} {mv:>9.3f} {fv:>9.3f} {sv:>9.3f}")

        delta = m_model.get("pairwise_coo_r",0) - m_static.get("pairwise_coo_r",0)
        print(f"\n  Δ co-occ r (model − static baseline): {delta:+.4f}")

        results[f"h={h}"] = {
            "target_month":          target,
            "n_real":                len(real_mat),
            "freq_pearson_r":        float(r_freq),
            "posfreq_mse":           float(mse_freq),
            "generative_model":      m_model,
            "baseline_freq_regression": m_freq,
            "baseline_static":       m_static,
            "delta_pairwise_coo_r":  float(delta),
            "model_beats_static_baseline": bool(delta > 0),
        }

    out = {
        "context_month": context_month,
        "traj_mode": mode, "window_size": W,
        "n_generated": n_gen,
        "results": results,
    }
    p = ROOT / "evaluation_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {p}")


if __name__ == "__main__":
    main()
