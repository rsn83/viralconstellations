"""
Step 10: Check 2 -- is the near-zero co-occurrence coefficient in
LogisticFrontierScorer a real null result, or a collinearity artifact
with frequency features?

Reuses your existing frozen trajectory encoder + frontier machinery
exactly as in 08_train_eval_discriminative.py (compute_occupied,
compute_frontier, compute_new_constellations, extract_features,
LogisticFrontierScorer) -- but skips the discriminative model entirely,
since these diagnostics only need the 7 hand-crafted features + labels.

Feature names are imported directly from frontier.py's FEATURE_NAMES list,
so the columns below use your actual names (coo_support, max_parent_freq,
pred_freq_new_pos, etc.), not placeholders.

Usage:
  python scripts/10_check_cooccurrence_artifact.py --config configs/colab_2022_test.yaml
"""

import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--horizon", type=int, default=1,
                    help="Which forecast horizon h to build candidate features for")
parser.add_argument("--top_k", type=int, default=200)
parser.add_argument("--out_dir", default="outputs/checks")
args = parser.parse_args()

import yaml, numpy as np, torch
import pandas as pd

from viralconstellations.model.model import PopulationEncoder, FrequencyRegressionHead, TrajectoryEmbeddingCache
from viralconstellations.model.trajectory import build_trajectory_encoder, TransitionModel
from viralconstellations.frontier.frontier import (
    compute_occupied, compute_frontier, compute_new_constellations,
    extract_features, FEATURE_NAMES,
)
from viralconstellations.checks import (
    correlation_and_vif, held_out_ap_comparison, held_out_ap_with_residual,
    frequency_matched_stratification, plot_stratification,
)

# Taken directly from frontier.py's FEATURE_NAMES:
#   ["pred_freq_new_pos", "max_parent_freq", "log_n_parents",
#    "mean_parent_depth", "jaccard_best_parent", "freq_trend_new_pos",
#    "coo_support"]
# The two features your walk-forward run found dominant were parent
# frequency and predicted mutation frequency -- i.e. max_parent_freq and
# pred_freq_new_pos. The near-zero feature was coo_support.
FREQ_FEATURE_NAMES_GUESS = ["max_parent_freq", "pred_freq_new_pos"]
COOC_FEATURE_NAME_GUESS = "coo_support"


def log(msg): print(msg, flush=True)

def load_mat(d, m):
    p = d / f"{m}.npy"; return np.load(p) if p.exists() else None
def load_pf(d, m):
    p = d / f"{m}_posfreq.npy"; return np.load(p) if p.exists() else None
def mplus(s, h):
    y, mo = int(s[:4]), int(s[5:7]); mo += h; y += (mo - 1) // 12; mo = (mo - 1) % 12 + 1
    return f"{y:04d}-{mo:02d}"
def mminus(s, k=1):
    y, mo = int(s[:4]), int(s[5:7]); mo -= k
    while mo <= 0: mo += 12; y -= 1
    return f"{y:04d}-{mo:02d}"


@torch.no_grad()
def get_trajectory_state(traj_enc, transition, freq_head, cache, ctx_month, h, mode, W, device):
    if mode == "gru":
        window = cache.get_window(ctx_month, W)
        h_t = traj_enc(window)
        _, states = transition(h_t, h)
        h_state = states[min(h, len(states) - 1)]
    else:
        h_state = traj_enc(cache.get_posfreq(ctx_month), cache.get_posfreq_prev(ctx_month))
    pred_pf = freq_head(h_state.unsqueeze(0)).squeeze(0).cpu().numpy()
    return h_state.detach(), pred_pf


def build_feature_rows(mat_ctx, mat_th, pred_pf, pf_prev, P, top_k, window_idx):
    """Collect (hand_features, label) rows for one window -- no model scoring."""
    occupied_t = compute_occupied(mat_ctx, top_k=top_k)
    frontier = compute_frontier(occupied_t, P)
    _, new_in_th = compute_new_constellations(mat_ctx, mat_th)
    if not frontier:
        return None

    candidates = list(frontier.keys())
    hand = np.stack([
        extract_features(c, frontier[c], pred_pf, pf_prev, P) for c in candidates
    ])  # (N, 7)
    labels = np.array([1 if c in new_in_th else 0 for c in candidates], dtype=np.float32)
    windows = np.full(len(candidates), window_idx)
    return hand, labels, windows


def main():
    cfg = yaml.safe_load(open(ROOT / args.config))
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    ckpt_dir = ROOT / cfg["train"]["checkpoint_dir"]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(f"Loading frozen trajectory encoder from {ckpt_dir / 'best_model.pt'} ...")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    P = ckpt["n_positions"]
    mode = ckpt["traj_cfg"]["mode"]
    W = ckpt["traj_cfg"]["window_size"]
    d = ckpt["model_cfg"]["d_model"]

    encoder = PopulationEncoder(P, d, ckpt["model_cfg"]["phi_hidden"]).to(device)
    traj_enc = build_trajectory_encoder(mode, d, ckpt["traj_cfg"]["gru_hidden"], W, P).to(device)
    transition = TransitionModel(d).to(device)
    freq_head = FrequencyRegressionHead(d, P).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    traj_enc.load_state_dict(ckpt["traj_state"])
    transition.load_state_dict(ckpt["transition_state"])
    freq_head.load_state_dict(ckpt["freq_state"])
    for m in [encoder, traj_enc, transition, freq_head]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    all_months = sorted(p.stem for p in matrix_dir.glob("*.npy") if "_posfreq" not in p.stem)
    log(f"Available months: {len(all_months)}  ({all_months[0]} -> {all_months[-1]})")

    all_mats = {m: load_mat(matrix_dir, m) for m in all_months}
    all_freqs = {m: load_pf(matrix_dir, m) for m in all_months}
    cache = TrajectoryEmbeddingCache(encoder, all_mats, all_freqs, device,
                                      cfg["model"]["deepsets_batch_size"])
    cache.refresh()

    min_train = 6
    cutoffs = all_months[min_train:-args.horizon]
    log(f"Collecting features for {len(cutoffs)} windows at horizon h={args.horizon} ...")

    hand_list, label_list, window_list = [], [], []
    for w_idx, ctx_month in enumerate(cutoffs):
        target = mplus(ctx_month, args.horizon)
        if target not in all_mats or all_mats[target] is None:
            continue
        mat_ctx, mat_th = all_mats[ctx_month], all_mats[target]
        prev_m = mminus(ctx_month, 1)
        pf_prev = all_freqs.get(prev_m, all_freqs[ctx_month])

        try:
            h_state, pred_pf = get_trajectory_state(
                traj_enc, transition, freq_head, cache, ctx_month, args.horizon, mode, W, device
            )
            result = build_feature_rows(mat_ctx, mat_th, pred_pf, pf_prev, P, args.top_k, w_idx)
        except Exception as e:
            log(f"  Skipping {ctx_month}: {e}")
            continue
        if result is None:
            continue
        hand, labels, windows = result
        hand_list.append(hand); label_list.append(labels); window_list.append(windows)

        if w_idx % 10 == 0:
            log(f"  Window {w_idx+1}/{len(cutoffs)}  ctx={ctx_month}  "
                f"n_candidates={len(labels)}  n_pos={int(labels.sum())}")

    hand_all = np.concatenate(hand_list)
    label_all = np.concatenate(label_list)
    window_all = np.concatenate(window_list)
    log(f"\nTotal candidate rows: {len(label_all):,}  "
        f"positives={int(label_all.sum()):,}  ({label_all.mean()*100:.3f}%)")

    assert len(FEATURE_NAMES) == hand_all.shape[1], (
        f"frontier.py FEATURE_NAMES has {len(FEATURE_NAMES)} entries but collected "
        f"hand features have {hand_all.shape[1]} columns -- frontier.py may have "
        f"changed since this script was written; check extract_features()."
    )
    log(f"\nFeature names (from frontier.py): {FEATURE_NAMES}")
    log(f"Frequency group: {FREQ_FEATURE_NAMES_GUESS}")
    log(f"Co-occurrence feature: {COOC_FEATURE_NAME_GUESS}")

    df = pd.DataFrame(hand_all, columns=FEATURE_NAMES)
    df["label"] = label_all
    df["window"] = window_all

    freq_cols = FREQ_FEATURE_NAMES_GUESS
    cooc_col = COOC_FEATURE_NAME_GUESS

    log("\n" + "=" * 70); log("1. Correlation / VIF"); log("=" * 70)
    correlation_and_vif(df, freq_cols + [cooc_col])

    log("\n" + "=" * 70); log("2. Held-out AP comparison (raw co-occurrence)"); log("=" * 70)
    held_out_ap_comparison(df, freq_cols, cooc_col)

    log("\n" + "=" * 70); log("3. Residualized co-occurrence"); log("=" * 70)
    held_out_ap_with_residual(df, freq_cols, cooc_col)

    log("\n" + "=" * 70); log("4. Frequency-matched stratification"); log("=" * 70)
    result = frequency_matched_stratification(df, freq_cols[0], cooc_col)
    plot_stratification(result, str(out_dir / "cooc_stratification.png"))

    df.to_csv(out_dir / "check2_candidate_features.csv", index=False)
    log(f"\nWrote: {out_dir / 'check2_candidate_features.csv'}")


if __name__ == "__main__":
    main()