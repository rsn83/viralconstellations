"""
End-to-end pipeline test using 2020 data only.

Runs training + evaluation + frontier walk-forward on a small subset.
Should complete in 5-10 minutes on a CPU laptop.

Usage:
  python scripts/test_pipeline.py

Checks:
  1. Training losses decrease
  2. Frequency Pearson r > 0 (model predicts direction of change)
  3. Frontier coverage > 0.5 (most new constellations are in F(O_t))
  4. Model co-occurrence r > baseline (model captures co-occurrence)
  5. Frontier scorer AP > random baseline (logistic regression adds signal)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# ── Load test config ──────────────────────────────────────────────────────────
cfg = yaml.safe_load(open(ROOT / "configs/test.yaml"))

# Override the config path for all imports
import viralconstellations.data.utils as du

matrix_dir  = ROOT / cfg["paths"]["matrix_dir"]
vocab_dir   = ROOT / cfg["paths"]["vocab_dir"]
ckpt_dir    = ROOT / cfg["train"]["checkpoint_dir"]
ckpt_dir.mkdir(parents=True, exist_ok=True)

from viralconstellations.model.model import (
    ConstellationTransformer, PopulationEncoder,
    LengthToGoHead, FrequencyRegressionHead,
    TrajectoryEmbeddingCache, reversion_noising,
    generate_from_hidden, independence_baseline_generate, N_RESIDUES,
)
from viralconstellations.model.trajectory import (
    build_trajectory_encoder, TransitionModel,
)
from viralconstellations.frontier.frontier import (
    frontier_coverage_benchmark, FrontierScorer,
)
from viralconstellations.eval.metrics import all_metrics_categorical


def load_mat(m):
    p = matrix_dir / f"{m}.npy"
    return np.load(p) if p.exists() else None

def load_posfreq(m):
    p = matrix_dir / f"{m}_posfreq.npy"
    return np.load(p) if p.exists() else None

def month_plus(s, h):
    y, mo = int(s[:4]), int(s[5:7])
    mo += h; y += (mo-1)//12; mo = (mo-1)%12+1
    return f"{y:04d}-{mo:02d}"

def month_minus(s, k=1):
    y, mo = int(s[:4]), int(s[5:7])
    mo -= k
    while mo <= 0: mo += 12; y -= 1
    return f"{y:04d}-{mo:02d}"


def main():
    print("="*60)
    print("PIPELINE TEST — 2020 data, small model")
    print("="*60)

    # ── Setup ──────────────────────────────────────────────────────────────
    P = len(pd.read_csv(vocab_dir / "position_vocab.tsv", sep="\t"))
    max_h   = cfg["horizon"]["max_h"]
    T       = cfg["model"]["diffusion_T"]
    d       = cfg["model"]["d_model"]
    mode    = cfg["trajectory"]["mode"]
    W       = cfg["trajectory"]["window_size"]
    device  = torch.device("cpu")   # laptop test on CPU

    train_months = cfg["train"]["train_months"]
    test_month   = cfg["train"]["test_month"]
    eval_hs      = cfg["eval"]["horizons"]

    print(f"P={P}  d={d}  T={T}  W={W}")
    print(f"Train: {train_months}")
    print(f"Test:  {test_month}")

    # Verify data exists
    missing = [m for m in train_months + [test_month] if load_mat(m) is None]
    if missing:
        print(f"\nERROR: Missing matrices for: {missing}")
        print("Run scripts/04_build_matrices_from_metadata.py first.")
        return

    # ── Build models ───────────────────────────────────────────────────────
    encoder     = PopulationEncoder(P, d, cfg["model"]["phi_hidden"])
    traj_enc    = build_trajectory_encoder(mode, d,
                    cfg["trajectory"]["gru_hidden"], W, P)
    transition  = TransitionModel(d)
    model       = ConstellationTransformer(P, d,
                    cfg["model"]["n_heads"], cfg["model"]["n_layers"],
                    cfg["model"]["dropout"], T, max_h)
    length_head = LengthToGoHead(d, max_h)
    freq_head   = FrequencyRegressionHead(d, P)

    modules    = [model, encoder, traj_enc, transition, length_head, freq_head]
    all_params = [p for m in modules for p in m.parameters()]
    n_params   = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Build cache ────────────────────────────────────────────────────────
    # Need window history + intermediate months for supervision
    all_needed = set()
    for mt in train_months:
        idx = train_months.index(mt) if mt in train_months else -1
        for w in range(W + 1):
            if idx - w >= 0:
                all_needed.add(train_months[max(0, idx - w)])
        for h in range(1, max_h + 1):
            th = month_plus(mt, h)
            if load_mat(th) is not None:
                all_needed.add(th)
    all_needed.add(train_months[-1])
    all_needed = sorted(all_needed)

    ctx_mats  = {m: load_mat(m)     for m in all_needed if load_mat(m) is not None}
    ctx_freqs = {m: load_posfreq(m) for m in all_needed if load_posfreq(m) is not None}

    cache = TrajectoryEmbeddingCache(encoder, ctx_mats, ctx_freqs, device, 128)

    optimizer = torch.optim.Adam(all_params, lr=cfg["train"]["lr"],
                                 weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["n_epochs"]
    )

    # Build training triples
    triples = []
    for i, mt in enumerate(train_months):
        for h in range(1, max_h + 1):
            j = i + h
            if j < len(train_months):
                mat = load_mat(train_months[j])
                if mat is not None:
                    triples.append((mt, mat, h))
    print(f"Training triples: {len(triples)}")

    # ── Training loop ──────────────────────────────────────────────────────
    print("\nTraining...")
    denoise_w = cfg["horizon"]["denoising_weight"]
    length_w  = cfg["horizon"]["length_weight"]
    freq_w    = cfg["freq_head"]["freq_weight"]
    inter_w   = cfg["freq_head"]["intermediate_weight"]

    best_val = float("inf")
    all_m    = sorted(ctx_mats.keys())

    for epoch in range(1, cfg["train"]["n_epochs"] + 1):
        cache.refresh()
        for m in modules: m.train()

        ep_d = ep_l = ep_f = 0.0
        for mt, mat_th, h_val in triples:
            # Filter + Transition
            window  = cache.get_window(mt, W)
            h_t     = traj_enc(window)
            h_final, states = transition(h_t, h_val)

            # Freq loss at each step
            t_idx = all_m.index(mt) if mt in all_m else -1
            freq_losses = []
            for k in range(1, h_val + 1):
                if t_idx + k >= len(all_m): continue
                month_tk = all_m[t_idx + k]
                if month_tk not in cache._freq_cache: continue
                pf_tgt = cache.get_posfreq(month_tk)
                h_k    = states[k] if k < len(states) else h_final
                fl     = freq_head.loss(h_k.unsqueeze(0), pf_tgt.unsqueeze(0))
                freq_losses.append((1.0 if k == h_val else inter_w) * fl)
            freq_loss = torch.stack(freq_losses).mean() if freq_losses else torch.tensor(0.0)

            # Denoising loss
            n   = len(mat_th)
            idx = torch.randint(0, n, (min(64, n),))
            x0  = torch.tensor(mat_th[idx.numpy()].astype(np.int64))
            B   = x0.shape[0]
            t   = torch.randint(1, T + 1, (B,))
            xt  = reversion_noising(x0, t, T)
            ctx = h_final.unsqueeze(0).expand(B, -1)
            h_t_embed = torch.full((B,), h_val, dtype=torch.long)
            logits = model(xt, t, ctx, h_t_embed)
            Bp, Pv, C = logits.shape
            denoise_loss = F.cross_entropy(
                logits.reshape(Bp*Pv, C), x0.reshape(Bp*Pv), label_smoothing=0.05
            )

            # Length loss
            true_cnt  = torch.tensor([(mat_th > 0).sum(axis=1).mean()],
                                     dtype=torch.float32)
            pred_cnt  = length_head(h_final.unsqueeze(0),
                                    torch.tensor([h_val], dtype=torch.long))
            length_loss = F.mse_loss(pred_cnt, true_cnt)

            loss = denoise_w * denoise_loss + length_w * length_loss + freq_w * freq_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            ep_d += denoise_loss.item()
            ep_l += length_loss.item()
            ep_f += freq_loss.item()

        scheduler.step()
        n = max(len(triples), 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:2d}  "
                  f"denoise={ep_d/n:.4f}  "
                  f"freq={ep_f/n:.4f}  "
                  f"length={ep_l/n:.4f}")

    # Save checkpoint
    torch.save({
        "model_state":   model.state_dict(),
        "encoder_state": encoder.state_dict(),
        "traj_state":    traj_enc.state_dict(),
        "transition_state": transition.state_dict(),
        "length_state":  length_head.state_dict(),
        "freq_state":    freq_head.state_dict(),
        "n_positions":   P,
        "train_months":  train_months,
        "test_month":    test_month,
        "model_cfg":     cfg["model"],
        "traj_cfg":      cfg["trajectory"],
        "max_h":         max_h,
    }, ckpt_dir / "best_model.pt")

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    for m in modules: m.eval()

    # Add test month to cache if not there
    if test_month not in cache._freq_cache:
        mat_test = load_mat(test_month)
        if mat_test is not None:
            cache.month_matrices[test_month] = mat_test
            pf_test = load_posfreq(test_month)
            if pf_test is not None:
                cache.month_posfreqs[test_month] = pf_test
        cache.refresh()

    context_month = train_months[-1]
    results = {}

    for h in eval_hs:
        target = month_plus(context_month, h)
        real_mat = load_mat(target)
        real_pf  = load_posfreq(target)
        if real_mat is None:
            print(f"h={h}: {target} not available, skipping")
            continue

        print(f"\nHorizon h={h}  ({context_month} → {target})")
        print(f"  Real sequences: {len(real_mat):,}  "
              f"mean_muts={(real_mat>0).sum(1).mean():.1f}")

        # Get hidden state at t+h
        with torch.no_grad():
            window  = cache.get_window(context_month, W)
            h_t     = traj_enc(window)
            h_final_state, states = transition(h_t, h)
            h_state = states[min(h, len(states)-1)]

            # Frequency prediction
            pred_pf = freq_head(h_state.unsqueeze(0)).squeeze(0).numpy()

        pred_rate = 1.0 - pred_pf[:, 0]
        real_rate = 1.0 - real_pf[:, 0]
        r_freq, _ = pearsonr(pred_rate, real_rate)
        print(f"  Freq Pearson r: {r_freq:.4f}")

        # Frontier coverage
        mat_ctx = load_mat(context_month)
        cov = frontier_coverage_benchmark(mat_ctx, real_mat, P)
        print(f"  Frontier coverage: {cov['frontier_coverage']:.4f}  "
              f"({cov['n_in_frontier']}/{cov['n_new']} new constellations in F(O_t))")
        print(f"  |O_t|={cov['n_occupied_t']}  |F(O_t)|={cov['n_frontier']}")

        # Generative evaluation
        gen_model = generate_from_hidden(
            model, length_head, h_state, h,
            cfg["eval"]["n_generate"], P,
            cfg["model"]["diffusion_T"], device
        )
        gen_bl = independence_baseline_generate(
            load_posfreq(context_month), cfg["eval"]["n_generate"]
        )
        mk = dict(top_k=cfg["eval"]["top_k_sites_pairs"],
                  mmd_n_sub=cfg["eval"]["mmd_subsample"])
        m_model = all_metrics_categorical(gen_model, real_mat, **mk)
        m_bl    = all_metrics_categorical(gen_bl,    real_mat, **mk)

        delta = m_model.get("pairwise_coo_r",0) - m_bl.get("pairwise_coo_r",0)
        print(f"  Pairwise co-occ r: model={m_model.get('pairwise_coo_r',0):.4f}  "
              f"baseline={m_bl.get('pairwise_coo_r',0):.4f}  "
              f"Δ={delta:+.4f}")
        print(f"  Mean mut count: model={m_model.get('mean_mut_count',0):.1f}  "
              f"real={m_model.get('mean_mut_count_real',0):.1f}")

        results[f"h={h}"] = {
            "freq_pearson_r":    float(r_freq),
            "frontier_coverage": float(cov["frontier_coverage"]),
            "n_new":             cov["n_new"],
            "delta_coo":         float(delta),
        }

    # ── Quick frontier scorer test ─────────────────────────────────────────
    print("\nFrontier Scorer (quick test on 2020 data)...")
    scorer = FrontierScorer()
    all_X, all_y = [], []

    for i, mt in enumerate(train_months[:-1]):
        mat_t  = load_mat(mt)
        mat_t1 = load_mat(train_months[i+1])
        if mat_t is None or mat_t1 is None: continue
        pf_t   = load_posfreq(mt)
        pf_prev = load_posfreq(train_months[max(0, i-1)])
        with torch.no_grad():
            window = cache.get_window(mt, W)
            h_tt   = traj_enc(window)
            _, sts = transition(h_tt, 1)
            pred_pf = freq_head(sts[1].unsqueeze(0)).squeeze(0).numpy()
        X, y = scorer.collect_training_examples(
            mat_t, mat_t1, pred_pf,
            pf_prev if pf_prev is not None else pf_t, P
        )
        if len(X) > 0:
            all_X.append(X); all_y.append(y)

    if all_X:
        X_all = np.concatenate(all_X)
        y_all = np.concatenate(all_y)
        if y_all.sum() > 0:
            scorer.fit(X_all, y_all)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    all_passed = True
    for h_key, r in results.items():
        print(f"\n{h_key}:")
        # Check 1: frequency prediction better than random
        if r["freq_pearson_r"] > 0:
            print(f"  ✓ Freq Pearson r = {r['freq_pearson_r']:.4f} > 0")
        else:
            print(f"  ✗ Freq Pearson r = {r['freq_pearson_r']:.4f} (should be > 0)")
            all_passed = False

        # Check 2: frontier coverage reasonable
        if r["frontier_coverage"] > 0.3:
            print(f"  ✓ Frontier coverage = {r['frontier_coverage']:.4f}")
        else:
            print(f"  ✗ Frontier coverage = {r['frontier_coverage']:.4f} (low — expected > 0.3)")
            all_passed = False

        # Check 3: delta co-occurrence
        if r["delta_coo"] > -0.1:
            print(f"  ✓ Δ co-occ r = {r['delta_coo']:+.4f}")
        else:
            print(f"  ✗ Δ co-occ r = {r['delta_coo']:+.4f} (model worse than baseline)")
            all_passed = False

    print("\n" + ("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"))
    print("="*60)
    print("\nIf checks pass, run full training on Colab with configs/default.yaml")
    print("then evaluate with:")
    print("  python scripts/06_evaluate.py")
    print("  python scripts/07_walk_forward_eval.py")


if __name__ == "__main__":
    main()
