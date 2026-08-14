"""
Step 15: Train and evaluate GraphTemporalScorer on real (position,residue)
data, at h=1/3/6, with FULL pair-space AP (not sampled-negative AP) as the
reported metric, against three baselines and a leave-one-component-out
ablation.

Design decisions, all fixed, not re-litigated per run:
  - Windowing: W=6 input months -> encode ONCE -> decode h=1,3,6 from that
    SAME encoding (DySAT-style, no autoregressive feedback).
  - Training: negative-sampled BCE (standard, necessary given class
    imbalance).
  - Reported metric: FULL pair-space AP -- every possible (i,j) pair in
    that month scored and compared against the true future graph. This
    is what "predicted graph vs actual graph" means, computed properly.
  - Baselines, same eval, so numbers are directly comparable:
      1. random
      2. naive persistence (g_{t+h} = g_t)
      3. frequency-only logistic (edge exists iff both endpoints'
         marginal frequency product exceeds nothing -- i.e. a simple
         frequency-based edge score, fit per window like your existing
         LogisticFrontierScorer)
  - Ablation: full model vs no-GNN vs no-RNN vs no-edge-history.

Usage:
  python scripts/15_train_eval_graph_temporal.py --window 6 --epochs 20
"""

import sys, argparse, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=6)
parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n_neg_per_pos", type=int, default=50)
parser.add_argument("--eval_pair_batch", type=int, default=50000,
                    help="batch size for scoring the full N*(N-1)/2 pair space at eval time")
args = parser.parse_args()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

from viralconstellations.model.graph_temporal_scorer_v2 import (
    GraphTemporalScorer, NaivePersistenceBaseline,
    compute_context_profile, profile_similarity_matrix,
    compute_distinct_constellation_stats, background_overlap_matrix,
)


def log(msg): print(msg, flush=True)


def load_month(graphs_dir: Path, month: str):
    g_t = np.load(graphs_dir / f"{month}_g_t.npy")
    f_t = np.load(graphs_dir / f"{month}_f_t.npy")
    with open(graphs_dir / f"{month}_occupied.pkl", "rb") as fh:
        occupied = pickle.load(fh)
    n_seq = int((graphs_dir / f"{month}_n_seq.txt").read_text())
    return g_t, f_t, occupied, n_seq


def build_month_tensors(g_t_np, f_t_np, occupied, n_seq, N, prev_freq_np=None):
    g_t = torch.tensor(g_t_np, dtype=torch.float32)
    f_t = torch.tensor(f_t_np, dtype=torch.float32)

    freq = f_t / max(n_seq, 1)
    freq_trend = (freq - torch.tensor(prev_freq_np, dtype=torch.float32)) if prev_freq_np is not None else torch.zeros(N)
    degree = g_t.sum(dim=-1) / max(g_t.sum().item(), 1.0)
    node_feats = torch.stack([freq, freq_trend, degree], dim=-1)

    profiles = compute_context_profile(g_t, f_t)
    profile_sim = profile_similarity_matrix(profiles)

    G_distinct_np, F_distinct_np = compute_distinct_constellation_stats(occupied, N)
    G_distinct = torch.tensor(G_distinct_np, dtype=torch.float32)
    F_distinct = torch.tensor(F_distinct_np, dtype=torch.float32)
    background_overlap = background_overlap_matrix(G_distinct, F_distinct)

    struct = torch.zeros(N, N)  # placeholder until Check B provides real distances

    adj = {"cooc": g_t, "profile_sim": profile_sim,
           "background_overlap": background_overlap, "struct": struct}
    return node_feats, adj, freq.numpy(), g_t


def occupied_edge_set(occupied: dict) -> set:
    edges = set()
    for constellation in occupied.keys():
        nodes = sorted(constellation)
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                edges.add((nodes[a], nodes[b]))
    return edges


def all_pairs(N: int):
    iu, ju = np.triu_indices(N, k=1)
    return iu, ju


def full_pairspace_ap(scores: np.ndarray, edges_th: set, iu: np.ndarray, ju: np.ndarray) -> float:
    """AP over EVERY possible pair, not a sampled subset -- this is the
    'predicted graph vs actual graph' comparison."""
    labels = np.array([(int(i), int(j)) in edges_th for i, j in zip(iu, ju)], dtype=np.int32)
    if labels.sum() == 0:
        return float("nan")
    return average_precision_score(labels, scores)


def sample_training_pairs(edges_t: set, edges_th: set, N: int, n_neg_per_pos: int, rng):
    new_edges = list(edges_th - edges_t)
    if not new_edges:
        return [], []
    n_neg = min(len(new_edges) * n_neg_per_pos, N * (N - 1) // 2)
    neg_pairs, attempts = [], 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 20:
        i, j = rng.integers(0, N, size=2)
        attempts += 1
        if i == j:
            continue
        pair = (min(int(i), int(j)), max(int(i), int(j)))
        if pair in edges_t or pair in edges_th:
            continue
        neg_pairs.append(pair)
    return new_edges + neg_pairs, [1] * len(new_edges) + [0] * len(neg_pairs)


def build_edge_history(pairs, g_t_history, window):
    hist = np.zeros((len(pairs), window), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        for t, g in enumerate(g_t_history):
            hist[k, t] = np.log1p(g[i, j])
    return torch.tensor(hist, dtype=torch.float32)


@torch.no_grad()
def score_full_pairspace(model, node_feats_seq, adj_seq, g_t_history, iu, ju, batch_size, device, horizon):
    model.eval()
    scores = np.zeros(len(iu), dtype=np.float32)
    node_h = model.node_encoder(node_feats_seq, adj_seq)
    for start in range(0, len(iu), batch_size):
        end = min(start + batch_size, len(iu))
        pi = torch.tensor(iu[start:end], dtype=torch.long, device=device)
        pj = torch.tensor(ju[start:end], dtype=torch.long, device=device)
        h_i, h_j = node_h[pi], node_h[pj]
        parts = [h_i, h_j]
        if model.use_edge_history:
            eh = build_edge_history(list(zip(iu[start:end], ju[start:end])), g_t_history,
                                     len(g_t_history)).to(device)
            parts.append(model.edge_history_encoder(eh))
        if model.use_horizon_embed:
            hz = torch.full((end - start,), horizon, dtype=torch.long, device=device)
            parts.append(model.horizon_embed(hz))
        combined = torch.cat(parts, dim=-1)
        logits = model.decoder(combined).squeeze(-1)
        scores[start:end] = torch.sigmoid(logits).cpu().numpy()
    return scores


def frequency_baseline_scores(freq: np.ndarray, iu: np.ndarray, ju: np.ndarray) -> np.ndarray:
    """Simple frequency-product edge score -- the edge-level analog of
    your existing frequency-only feature set, as a fair baseline."""
    return freq[iu] * freq[ju]


def main():
    graphs_dir = ROOT / "data" / "processed" / "full_data_graphs_posres"
    index_df = pd.read_csv(graphs_dir / "index.tsv", sep="\t")
    months = sorted(index_df["month"].tolist())
    vocab_df = pd.read_csv(graphs_dir / "posres_vocab.tsv", sep="\t")
    N = len(vocab_df)
    log(f"N={N} (position,residue) nodes, {len(months)} months")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    W = args.window
    iu, ju = all_pairs(N)
    log(f"Full pair space: {len(iu):,} pairs")

    month_cache = {}
    def get_month(idx):
        if idx not in month_cache:
            m = months[idx]
            g_t_np, f_t_np, occ, n_seq = load_month(graphs_dir, m)
            prev_f = month_cache[idx - 1][2] if (idx - 1) in month_cache else None
            nf, adj, freq, g_t = build_month_tensors(g_t_np, f_t_np, occ, n_seq, N, prev_f)
            month_cache[idx] = (nf, adj, freq, g_t, occ, g_t_np)
        return month_cache[idx]

    configs = {
        "full_model":       dict(use_gnn=True,  use_rnn=True,  use_edge_history=True),
        "no_gnn":           dict(use_gnn=False, use_rnn=True,  use_edge_history=True),
        "no_rnn":           dict(use_gnn=True,  use_rnn=False, use_edge_history=True),
        "no_edge_history":  dict(use_gnn=True,  use_rnn=True,  use_edge_history=False),
    }

    results = {name: {h: [] for h in args.horizons} for name in configs}
    baseline_results = {"random": {h: [] for h in args.horizons},
                         "naive_persistence": {h: [] for h in args.horizons},
                         "frequency": {h: [] for h in args.horizons}}

    max_h = max(args.horizons)
    cutoffs = list(range(W, len(months) - max_h))
    log(f"Walk-forward: {len(cutoffs)} windows")

    models = {name: GraphTemporalScorer(
        node_feat_dim=3, hidden_dim=args.hidden_dim,
        relation_names=["cooc", "struct", "profile_sim", "background_overlap"],
        edge_history_window=W, **cfg,
    ).to(device) for name, cfg in configs.items()}
    optimizers = {name: torch.optim.Adam(m.parameters(), lr=args.lr) for name, m in models.items()}

    for w_idx, t_idx in enumerate(cutoffs):
        window_idxs = list(range(t_idx - W, t_idx))
        node_feats_seq, adj_seq, g_t_history = [], [], []
        for idx in window_idxs:
            nf, adj, freq, g_t, occ, g_t_np = get_month(idx)
            node_feats_seq.append(nf.to(device))
            adj_seq.append({k: v.to(device) for k, v in adj.items()})
            g_t_history.append(g_t_np)

        _, _, freq_t, g_t_t, occ_t, _ = get_month(t_idx - 1)
        edges_t = occupied_edge_set(occ_t)

        # MULTI-HORIZON TRAINING FIX: build one combined batch across ALL
        # requested horizons, weighted loss sum (Benechehab et al. 2024,
        # "A Multi-step Loss Function for Robust Learning of the Dynamics
        # in Model-Based RL": L = sum_h alpha_h * L_h). Each horizon's
        # pairs get a horizon_id so the shared decoder learns
        # horizon-dependent behavior, instead of training only on h=1 and
        # reusing those weights unchanged for h=3/h=6 (the earlier bug).
        combined_pairs, combined_labels, combined_horizon_ids = [], [], []
        for h in args.horizons:
            target_idx_h = t_idx - 1 + h
            if target_idx_h >= len(months):
                continue
            _, _, _, _, occ_th_h, _ = get_month(target_idx_h)
            edges_th_h = occupied_edge_set(occ_th_h)
            pairs_h, labels_h = sample_training_pairs(edges_t, edges_th_h, N, args.n_neg_per_pos, rng)
            combined_pairs.extend(pairs_h)
            combined_labels.extend(labels_h)
            combined_horizon_ids.extend([h] * len(pairs_h))

        if not combined_pairs:
            continue
        edge_hist = build_edge_history(combined_pairs, g_t_history, W).to(device)
        pair_i = torch.tensor([p[0] for p in combined_pairs], dtype=torch.long).to(device)
        pair_j = torch.tensor([p[1] for p in combined_pairs], dtype=torch.long).to(device)
        labels_t = torch.tensor(combined_labels, dtype=torch.float32).to(device)
        horizon_ids_t = torch.tensor(combined_horizon_ids, dtype=torch.long).to(device)

        for name, model in models.items():
            model.train()
            opt = optimizers[name]
            for _ in range(args.epochs):
                opt.zero_grad()
                logits = model(node_feats_seq, adj_seq, pair_i, pair_j,
                                edge_hist if model.use_edge_history else None,
                                horizon_ids_t if model.use_horizon_embed else None)
                loss = F.binary_cross_entropy_with_logits(logits, labels_t)
                loss.backward()
                opt.step()

        # evaluate at EACH horizon, full pair space, same encoding
        for h in args.horizons:
            target_idx = t_idx - 1 + h
            if target_idx >= len(months):
                continue
            _, _, _, _, occ_th, _ = get_month(target_idx)
            edges_th = occupied_edge_set(occ_th)

            ctx_month = months[t_idx - 1]

            for name, model in models.items():
                scores = score_full_pairspace(model, node_feats_seq, adj_seq, g_t_history,
                                               iu, ju, args.eval_pair_batch, device, horizon=h)
                ap = full_pairspace_ap(scores, edges_th, iu, ju)
                if not np.isnan(ap):
                    results[name][h].append((ctx_month, ap))

            rand_scores = rng.random(len(iu))
            ap_r = full_pairspace_ap(rand_scores, edges_th, iu, ju)
            if not np.isnan(ap_r):
                baseline_results["random"][h].append((ctx_month, ap_r))

            persist_scores = g_t_t[iu, ju]
            ap_p = full_pairspace_ap(persist_scores, edges_th, iu, ju)
            if not np.isnan(ap_p):
                baseline_results["naive_persistence"][h].append((ctx_month, ap_p))

            freq_scores = frequency_baseline_scores(freq_t, iu, ju)
            ap_f = full_pairspace_ap(freq_scores, edges_th, iu, ju)
            if not np.isnan(ap_f):
                baseline_results["frequency"][h].append((ctx_month, ap_f))

        if w_idx % 5 == 0:
            log(f"  window {w_idx+1}/{len(cutoffs)}  ctx_month={months[t_idx-1]}  loss={loss.item():.4f}")

    def era_of(month_str: str) -> str:
        year = int(month_str[:4])
        return "high_volume_2020_2022" if year <= 2022 else "low_volume_2023_2026"

    def summarize(pairs: list[tuple[str, float]]) -> tuple[float, float, int]:
        vals = [ap for _, ap in pairs]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    log("\n" + "=" * 70)
    log("RESULTS (full pair-space AP, mean +/- std across ALL windows)")
    log("=" * 70)
    for h in args.horizons:
        log(f"\n--- horizon h={h} ---")
        for name in ["random", "naive_persistence", "frequency"]:
            m, s, n = summarize(baseline_results[name][h])
            if n:
                log(f"  {name:<20} AP = {m:.4f} +/- {s:.4f}  (n={n})")
        for name in configs:
            m, s, n = summarize(results[name][h])
            if n:
                log(f"  {name:<20} AP = {m:.4f} +/- {s:.4f}  (n={n})")

    # ------------------------------------------------------------------
    # ERA BREAKDOWN: sequence volume swings ~15-20x across the timeline
    # (833k seqs in 2022-01 vs ~3k in 2025). Split results by data-volume
    # era, using the EXACT context month recorded alongside each AP value
    # (not an approximate window-index alignment), since GNN/RNN
    # performance may depend heavily on how much data was available per
    # window, not just on the architecture itself.
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("ERA BREAKDOWN (exact month-based split)")
    log("=" * 70)

    for h in args.horizons:
        log(f"\n--- horizon h={h} ---")
        for era in ["high_volume_2020_2022", "low_volume_2023_2026"]:
            log(f"\n  [{era}]")
            for name in ["random", "naive_persistence", "frequency"]:
                pairs = [(m, ap) for m, ap in baseline_results[name][h] if era_of(m) == era]
                mean, std, n = summarize(pairs)
                if n:
                    log(f"    {name:<20} AP = {mean:.4f} +/- {std:.4f}  (n={n})")
            for name in configs:
                pairs = [(m, ap) for m, ap in results[name][h] if era_of(m) == era]
                mean, std, n = summarize(pairs)
                if n:
                    log(f"    {name:<20} AP = {mean:.4f} +/- {std:.4f}  (n={n})")


if __name__ == "__main__":
    main()
