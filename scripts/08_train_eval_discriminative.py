"""
Step 8: Train and evaluate the discriminative frontier model.

Walk-forward design (temporally clean):
  For each test window t in the last 20 windows:
    - Train on ALL windows before t
    - Evaluate on window t
  This gives ~20 evaluation points with no data leakage.

Three scorers compared at each test window:
  1. Random baseline
  2. Logistic regression (hand features only)
  3. Neural discriminative (learned embedding + hand features)

The key question: does emb(c) from CandidateEncoder add value beyond
the 7 hand-crafted features that LogisticFrontierScorer uses?

Usage:
  python scripts/08_train_eval_discriminative.py --config configs/colab_2022_test.yaml
"""

import sys, json, argparse, traceback
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--n_test_windows", type=int, default=20,
                    help="Number of windows held out for evaluation")
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

import yaml, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

from viralconstellations.model.model import (
    PopulationEncoder, FrequencyRegressionHead,
    TrajectoryEmbeddingCache,
)
from viralconstellations.model.trajectory import (
    build_trajectory_encoder, TransitionModel,
)
from viralconstellations.model.discriminative import DiscriminativeModel
from viralconstellations.frontier.frontier import (
    compute_occupied, compute_frontier,
    compute_new_constellations,
    candidate_to_sequence, extract_features,
    LogisticFrontierScorer, evaluate_ranking,
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
def get_trajectory_state(traj_enc, transition, freq_head, cache,
                         ctx_month, h, mode, W, device):
    """Get frozen trajectory state and predicted posfreq."""
    if mode == "gru":
        window  = cache.get_window(ctx_month, W)
        h_t     = traj_enc(window)
        _, states = transition(h_t, h)
        h_state = states[min(h, len(states)-1)]
    else:
        h_state = traj_enc(
            cache.get_posfreq(ctx_month),
            cache.get_posfreq_prev(ctx_month)
        )
    pred_pf = freq_head(h_state.unsqueeze(0)).squeeze(0).cpu().numpy()
    return h_state.detach(), pred_pf


def build_window_examples(mat_ctx, mat_th, pred_pf, pf_prev, h_state,
                          P, device, top_k=200):
    """
    Build (candidate_seq, hand_features, h_state, label) for one window.
    Returns arrays ready for batched training.
    """
    occupied_t  = compute_occupied(mat_ctx, top_k=top_k)
    frontier    = compute_frontier(occupied_t, P)
    _, new_in_th = compute_new_constellations(mat_ctx, mat_th)

    if not frontier:
        return None

    candidates = list(frontier.keys())
    n = len(candidates)

    # Candidate sequences: fill predicted best residue at each mutated position
    seqs  = np.stack([
        candidate_to_sequence(c, pred_pf, P) for c in candidates
    ])  # (N, P) int8

    # Hand features: 7 interpretable features per candidate
    hand  = np.stack([
        extract_features(c, frontier[c], pred_pf, pf_prev, P)
        for c in candidates
    ])  # (N, 7) float32

    # Labels: 1 if candidate appeared in O_{t+h}, else 0
    labels = np.array([1 if c in new_in_th else 0 for c in candidates],
                      dtype=np.float32)

    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    return {
        "seqs":      seqs,
        "hand":      hand,
        "labels":    labels,
        "h_state":   h_state.cpu().numpy(),
        "n_pos":     n_pos,
        "n_neg":     n_neg,
        "candidates": candidates,
        "new_in_th": new_in_th,
    }


def train_discriminative(disc_model, examples_list, device,
                         n_epochs=20, lr=1e-3, batch_size=256):
    """Train discriminative model on collected window examples."""
    # Concatenate all windows
    all_seqs   = np.concatenate([e["seqs"]   for e in examples_list])
    all_hand   = np.concatenate([e["hand"]   for e in examples_list])
    all_labels = np.concatenate([e["labels"] for e in examples_list])
    all_hstate = np.concatenate([
        np.tile(e["h_state"], (len(e["seqs"]), 1))
        for e in examples_list
    ])

    n_pos = int(all_labels.sum())
    n_neg = int((1 - all_labels).sum())
    if n_pos == 0:
        log("  No positive examples — skipping training")
        return

    # Class weight for imbalanced data
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    log(f"  Training on {len(all_labels)} examples "
        f"({n_pos} positive, {n_neg} negative, "
        f"pos_weight={pos_weight.item():.1f})")

    optimizer = torch.optim.Adam(disc_model.parameters(), lr=lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    disc_model.train()

    for epoch in range(1, n_epochs + 1):
        perm = np.random.permutation(len(all_labels))
        ep_loss = 0.0; n_batches = 0

        for i in range(0, len(perm), batch_size):
            idx  = perm[i:i+batch_size]
            x    = torch.tensor(all_seqs[idx].astype(np.int64), device=device)
            hand = torch.tensor(all_hand[idx], device=device)
            h    = torch.tensor(all_hstate[idx], device=device)
            y    = torch.tensor(all_labels[idx], device=device)

            pred = disc_model(x, h, hand)
            loss = F.binary_cross_entropy(pred, y,
                                          weight=pos_weight * y + (1 - y))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(disc_model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item(); n_batches += 1

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            log(f"  Epoch {epoch:3d}  loss={ep_loss/max(n_batches,1):.4f}")


def eval_window(disc_model, lr_scorer, window_data, device):
    """Evaluate all three scorers on one test window."""
    candidates = window_data["candidates"]
    new_in_th  = window_data["new_in_th"]
    seqs       = window_data["seqs"]
    hand       = window_data["hand"]
    h_state    = torch.tensor(window_data["h_state"], device=device)

    if not candidates or not new_in_th:
        return {}

    import random as rnd
    shuffled = candidates.copy(); rnd.shuffle(shuffled)

    # Random baseline
    random_met = evaluate_ranking(
        [(c, 0.0) for c in shuffled], new_in_th
    )

    # Logistic regression (hand features only)
    lr_met = {}
    if lr_scorer.fitted:
        # Score using logistic scorer directly
        lr_scores = lr_scorer.model.predict_proba(
            lr_scorer.scaler.transform(hand)
        )[:, 1]
        lr_ranked = sorted(zip(candidates, lr_scores), key=lambda x: -x[1])
        lr_met = evaluate_ranking(lr_ranked, new_in_th)

    # Neural discriminative
    disc_model.eval()
    neural_scores = disc_model.score_candidates(seqs, hand, h_state, device)
    neural_ranked = sorted(zip(candidates, neural_scores.tolist()),
                           key=lambda x: -x[1])
    neural_met = evaluate_ranking(neural_ranked, new_in_th)

    return {
        "random":  random_met,
        "logistic": lr_met,
        "neural":  neural_met,
        "n_candidates": len(candidates),
        "n_positive":   len(new_in_th),
    }


def main():
    cfg        = yaml.safe_load(open(ROOT / args.config))
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    ckpt_dir   = ROOT / cfg["train"]["checkpoint_dir"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("="*60)
    log(f"Discriminative frontier model | config: {args.config}")
    log(f"Device: {device}  epochs={args.n_epochs}  "
        f"test_windows={args.n_test_windows}")
    log("="*60)

    # Load frozen trajectory encoder
    ckpt  = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    P     = ckpt["n_positions"]
    max_h = ckpt["max_h"]
    mode  = ckpt["traj_cfg"]["mode"]
    W     = ckpt["traj_cfg"]["window_size"]
    d     = ckpt["model_cfg"]["d_model"]
    eval_horizons = [h for h in cfg["eval"]["horizons"] if h <= max_h]

    log(f"P={P}  d={d}  mode={mode}  W={W}  horizons={eval_horizons}")

    encoder  = PopulationEncoder(P, d, ckpt["model_cfg"]["phi_hidden"]).to(device)
    traj_enc = build_trajectory_encoder(
        mode, d, ckpt["traj_cfg"]["gru_hidden"], W, P).to(device)
    transition = TransitionModel(d).to(device)
    freq_head  = FrequencyRegressionHead(d, P).to(device)

    encoder.load_state_dict(ckpt["encoder_state"])
    traj_enc.load_state_dict(ckpt["traj_state"])
    transition.load_state_dict(ckpt["transition_state"])
    freq_head.load_state_dict(ckpt["freq_state"])

    # Freeze trajectory encoder — it's already trained
    for m in [encoder, traj_enc, transition, freq_head]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    log("Trajectory encoder frozen (using pretrained weights)")

    # Load all months and build cache
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

    # Build window examples for all months
    log("\nBuilding window examples for all months...")
    all_windows = []
    min_train   = 6

    for ctx_month in all_months[min_train:-max(eval_horizons)]:
        mat_ctx = all_mats[ctx_month]
        pf_ctx  = all_freqs[ctx_month]
        prev_m  = mminus(ctx_month)
        pf_prev = all_freqs.get(prev_m, None) or pf_ctx

        for h in eval_horizons:
            target  = mplus(ctx_month, h)
            if target not in all_mats or all_mats[target] is None:
                continue
            mat_th  = all_mats[target]
            real_pf = all_freqs[target]
            if real_pf is None:
                continue

            try:
                h_state, pred_pf = get_trajectory_state(
                    traj_enc, transition, freq_head, cache,
                    ctx_month, h, mode, W, device
                )
                ex = build_window_examples(
                    mat_ctx, mat_th, pred_pf, pf_prev,
                    h_state, P, device
                )
                if ex is not None and ex["n_pos"] > 0:
                    ex["context"] = ctx_month
                    ex["target"]  = target
                    ex["h"]       = h
                    all_windows.append(ex)
            except Exception as e:
                continue

    log(f"Total windows with positives: {len(all_windows)}")
    total_pos = sum(w["n_pos"] for w in all_windows)
    total_neg = sum(w["n_neg"] for w in all_windows)
    log(f"Total examples: {total_pos} positive, {total_neg} negative")

    # Split train/test temporally
    n_test  = min(args.n_test_windows, len(all_windows) // 3)
    n_train = len(all_windows) - n_test
    train_windows = all_windows[:n_train]
    test_windows  = all_windows[n_train:]
    log(f"Train windows: {n_train}  Test windows: {n_test}")

    # Build discriminative model
    disc_model = DiscriminativeModel(P, d, phi_hidden=128).to(device)
    n_params = sum(p.numel() for p in disc_model.parameters() if p.requires_grad)
    log(f"Discriminative model parameters: {n_params:,}")

    # Train logistic regression on training windows
    lr_scorer = LogisticFrontierScorer()
    lr_X = np.concatenate([w["hand"] for w in train_windows])
    lr_y = np.concatenate([w["labels"] for w in train_windows])
    if lr_y.sum() > 0:
        lr_scorer.fit(lr_X, lr_y)

    # Train discriminative model on training windows
    log("\nTraining discriminative model...")
    train_discriminative(
        disc_model, train_windows, device,
        n_epochs=args.n_epochs, lr=args.lr
    )

    # Walk-forward evaluation on test windows
    log("\n" + "="*60)
    log("WALK-FORWARD EVALUATION ON TEST WINDOWS")
    log("="*60)

    results_by_h = defaultdict(list)
    for window in test_windows:
        r = eval_window(disc_model, lr_scorer, window, device)
        if r:
            r["context"] = window["context"]
            r["target"]  = window["target"]
            results_by_h[window["h"]].append(r)

    # Summary
    log("\n" + "="*60)
    log("RESULTS")
    log("="*60)
    log(f"{'Horizon':>8} {'Windows':>8} {'Random AP':>12} "
        f"{'Logistic AP':>12} {'Neural AP':>12} {'Neural>LR':>10}")
    log("-"*70)

    summary = {}
    for h in eval_horizons:
        rows = results_by_h.get(h, [])
        if not rows: continue

        r_ap = [r["random"].get("AP", 0)   for r in rows]
        l_ap = [r["logistic"].get("AP", 0) for r in rows]
        n_ap = [r["neural"].get("AP", 0)   for r in rows]
        n_p10= [r["neural"].get("precision@10", 0) for r in rows]
        l_p10= [r["logistic"].get("precision@10", 0) for r in rows]

        neural_beats_lr = float(np.mean(n_ap)) > float(np.mean(l_ap))
        log(f"  h={h}    {len(rows):>6}   "
            f"{np.mean(r_ap):>10.4f}   "
            f"{np.mean(l_ap):>10.4f}   "
            f"{np.mean(n_ap):>10.4f}   "
            f"{'✓' if neural_beats_lr else '✗':>9}")

        summary[f"h={h}"] = {
            "n_windows":          len(rows),
            "random_AP":          float(np.mean(r_ap)),
            "logistic_AP":        float(np.mean(l_ap)),
            "logistic_P10":       float(np.mean(l_p10)),
            "neural_AP":          float(np.mean(n_ap)),
            "neural_P10":         float(np.mean(n_p10)),
            "neural_beats_logistic": neural_beats_lr,
        }

    # Feature importances from logistic scorer
    if lr_scorer.fitted:
        imps = lr_scorer.feature_importances()
        log("\nLogistic feature importances (baseline):")
        for name, coef in sorted(imps.items(), key=lambda x: -abs(x[1])):
            log(f"  {name:<30} {coef:+.4f}")
        summary["lr_feature_importances"] = imps

    out_path = ROOT / "discriminative_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log(f"\nWrote: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
