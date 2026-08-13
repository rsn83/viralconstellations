"""
Step 11: Check A driver -- builds the real candidate table + g_t graphs
from your repo's data and runs the raw co-occurrence diagnostics.

This mirrors 10_check_cooccurrence_artifact.py's data loading, then adds
raw g_t[i,j] as a feature and runs the four-way diagnostic on it, instead
of (or alongside) the frequency-product coo_support proxy.

Usage:
  python scripts/11_check_A_raw_gt.py --config configs/colab_2022_test.yaml
"""

import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--horizon", type=int, default=1)
parser.add_argument("--top_k", type=int, default=200)
args = parser.parse_args()

import yaml, numpy as np, torch
import pandas as pd

from viralconstellations.model.model import PopulationEncoder, FrequencyRegressionHead, TrajectoryEmbeddingCache
from viralconstellations.model.trajectory import build_trajectory_encoder, TransitionModel
from viralconstellations.frontier.frontier import (
    compute_occupied, compute_frontier, compute_new_constellations,
    extract_features, FEATURE_NAMES,
)
from checkA_and_C_graph_tests import build_cooccurrence_graph, add_raw_cooccurrence_feature, run_check_A

FREQ_FEATURE_NAMES = ["max_parent_freq", "pred_freq_new_pos"]


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


def main():
    cfg = yaml.safe_load(open(ROOT / args.config))
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    ckpt_dir = ROOT / cfg["train"]["checkpoint_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    all_mats = {m: load_mat(matrix_dir, m) for m in all_months}
    all_freqs = {m: load_pf(matrix_dir, m) for m in all_months}
    cache = TrajectoryEmbeddingCache(encoder, all_mats, all_freqs, device, cfg["model"]["deepsets_batch_size"])
    cache.refresh()

    min_train = 6
    cutoffs = all_months[min_train:-args.horizon]
    log(f"Collecting features for {len(cutoffs)} windows at horizon h={args.horizon} ...")

    hand_list, label_list, window_list, candidate_list, frontier_info_list, g_t_by_window = [], [], [], [], [], {}
    for w_idx, ctx_month in enumerate(cutoffs):
        target = mplus(ctx_month, args.horizon)
        if target not in all_mats or all_mats[target] is None:
            continue
        mat_ctx, mat_th = all_mats[ctx_month], all_mats[target]
        prev_m = mminus(ctx_month, 1)
        pf_prev = all_freqs.get(prev_m, all_freqs[ctx_month])

        try:
            h_state, pred_pf = get_trajectory_state(traj_enc, transition, freq_head, cache,
                                                      ctx_month, args.horizon, mode, W, device)
            occupied_t = compute_occupied(mat_ctx, top_k=args.top_k)
            frontier = compute_frontier(occupied_t, P)
            _, new_in_th = compute_new_constellations(mat_ctx, mat_th)
        except Exception as e:
            log(f"  Skipping {ctx_month}: {e}")
            continue
        if not frontier:
            continue

        candidates = list(frontier.keys())
        hand = np.stack([extract_features(c, frontier[c], pred_pf, pf_prev, P) for c in candidates])
        labels = np.array([1 if c in new_in_th else 0 for c in candidates], dtype=np.float32)

        hand_list.append(hand)
        label_list.append(labels)
        window_list.append(np.full(len(candidates), w_idx))
        candidate_list.extend(candidates)
        frontier_info_list.extend(frontier[c] for c in candidates)
        g_t_by_window[w_idx] = build_cooccurrence_graph(mat_ctx)

        if w_idx % 10 == 0:
            log(f"  Window {w_idx+1}/{len(cutoffs)}  ctx={ctx_month}  n_candidates={len(labels)}")

    hand_all = np.concatenate(hand_list)
    label_all = np.concatenate(label_list)
    window_all = np.concatenate(window_list)

    assert len(FEATURE_NAMES) == hand_all.shape[1], "FEATURE_NAMES mismatch -- check frontier.py"
    df = pd.DataFrame(hand_all, columns=FEATURE_NAMES)
    df["label"] = label_all
    df["window"] = window_all

    log("\nAdding new-edges-only raw g_t co-occurrence feature ...")
    df = add_raw_cooccurrence_feature(df, candidate_list, frontier_info_list, g_t_by_window, window_col="window")

    run_check_A(df, freq_cols=FREQ_FEATURE_NAMES)

    out_dir = ROOT / "outputs" / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "checkA_raw_gt_features.csv", index=False)
    log(f"\nWrote: {out_dir / 'checkA_raw_gt_features.csv'}")


if __name__ == "__main__":
    main()
