"""
scripts/18_variant_cycle_eval.py

Different control flow from 15_train_eval_graph_temporal.py's continuous
walk-forward. This does what you actually asked for:

  1. Bulk-train (combined multi-horizon, safe/leak-free -- nothing in
     this stretch is ever evaluated) on all months BEFORE the first
     variant's emergence window.
  2. FREEZE and evaluate (no training) just the emergence window.
  3. Bulk-train again, now covering everything up through just before
     the NEXT variant's emergence window (this legitimately includes
     the previous emergence window, since it's now in the past).
  4. Evaluate the next emergence window. Repeat.

This is much cheaper than the continuous retroactive walk-forward in
15_..., and matches standard practice ("train on 2020-2023, test on
specific held-out periods") rather than testing every single month.

DEFAULT VARIANT WINDOWS (searched, not from memory -- verify against
your own data before trusting them, since your months list may not
extend as far as some of these, and "emergence" dates are inherently
approximate/contested):
  Alpha   ~2020-11    Beta  ~2020-08    Gamma ~2021-01
  Delta   ~2021-05 (WHO designation end of May 2021)
  Omicron/BA.1 ~2021-11 (designated 24 Nov 2021)
  BA.2    ~2022-01     BA.4/BA.5 ~2022-05
  BA.2.86 ~2023-08 (first identified)   JN.1  ~2023-12 (became dominant)
  KP.3/KP.3.1.1 ~2024-06     LP.8.1 ~2025-01 (designated VUM)

Each window is (name, start_month, width_months) -- width defaults to
2 months (the emergence month + 1 following) unless overridden.

Usage:
  python scripts/18_variant_cycle_eval.py --window 6
  python scripts/18_variant_cycle_eval.py --window 6 \
      --variant_windows Alpha:2020-11:2 Delta:2021-05:2 Omicron:2021-11:2
"""

import sys, argparse, pickle, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_VARIANT_WINDOWS = [
    ("Alpha",   "2020-11", 2),
    ("Beta",    "2020-08", 2),
    ("Gamma",   "2021-01", 2),
    ("Delta",   "2021-05", 2),
    ("Omicron_BA1", "2021-11", 2),
    ("BA2",     "2022-01", 2),
    ("BA4_BA5", "2022-05", 2),
    ("BA286",   "2023-08", 2),
    ("JN1",     "2023-12", 2),
    ("KP3",     "2024-06", 2),
    ("LP81",    "2025-01", 2),
]

parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=6)
parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n_neg_per_pos", type=int, default=50)
parser.add_argument("--eval_pair_batch", type=int, default=50000)
parser.add_argument("--esm_cache_path", type=str, default="outputs/esm_cache.pkl")
parser.add_argument("--esm_adapter_dim", type=int, default=32)
parser.add_argument("--target_type", type=str, default="regression", choices=["binary", "regression"])
parser.add_argument("--variant_windows", type=str, nargs="+", default=None,
                     help="override DEFAULT_VARIANT_WINDOWS. Format name:start_month:width, "
                          "e.g. Alpha:2020-11:2. width is in months.")
args = parser.parse_args()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

from viralconstellations.model.graph_temporal_scorer_v2 import GraphTemporalScorer
from viralconstellations.model.esm_embeddings import ESMEmbeddingCache


def log(msg): print(msg, flush=True)


# ---------------------------------------------------------------------
# Helpers copied verbatim from 15_train_eval_graph_temporal.py -- same
# vetted logic, kept identical rather than imported, since scripts/
# isn't set up as an importable package.
# ---------------------------------------------------------------------
def load_month(graphs_dir: Path, month: str):
    g_t = np.load(graphs_dir / f"{month}_g_t.npy")
    f_t = np.load(graphs_dir / f"{month}_f_t.npy")
    with open(graphs_dir / f"{month}_occupied.pkl", "rb") as fh:
        occupied = pickle.load(fh)
    n_seq = int((graphs_dir / f"{month}_n_seq.txt").read_text())
    return g_t, f_t, occupied, n_seq


def build_month_tensors(g_t_np, f_t_np, occupied, n_seq, N, prev_freq_np=None):
    from viralconstellations.model.graph_temporal_scorer_v2 import (
        compute_context_profile, profile_similarity_matrix,
        compute_distinct_constellation_stats, background_overlap_matrix,
    )
    g_t = torch.tensor(g_t_np, dtype=torch.float32)
    f_t = torch.tensor(f_t_np, dtype=torch.float32)

    freq = f_t / max(n_seq, 1)
    freq_trend = (freq - torch.tensor(prev_freq_np, dtype=torch.float32)) if prev_freq_np is not None else torch.zeros(N)
    degree = g_t.sum(dim=-1) / max(g_t.sum().item(), 1.0)
    node_feats = torch.stack([freq, freq_trend, degree], dim=-1)

    # Frequency-normalized (not raw count) -- see 15_...py for rationale.
    g_t_freq = g_t / max(n_seq, 1)

    profiles = compute_context_profile(g_t, f_t)
    profile_sim = profile_similarity_matrix(profiles)

    G_distinct_np, F_distinct_np = compute_distinct_constellation_stats(occupied, N)
    G_distinct = torch.tensor(G_distinct_np, dtype=torch.float32)
    F_distinct = torch.tensor(F_distinct_np, dtype=torch.float32)
    background_overlap = background_overlap_matrix(G_distinct, F_distinct)

    struct = torch.zeros(N, N)

    adj = {"cooc": g_t_freq, "profile_sim": profile_sim,
           "background_overlap": background_overlap, "struct": struct}
    return node_feats, adj, freq.numpy(), g_t, g_t_freq.numpy()


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


def full_pairspace_ap(scores, edges_th, iu, ju):
    labels = np.array([(int(i), int(j)) in edges_th for i, j in zip(iu, ju)], dtype=np.int32)
    if labels.sum() == 0:
        return float("nan")
    return average_precision_score(labels, scores)


def sample_training_pairs(edges_t, edges_th, N, n_neg_per_pos, rng):
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


def sample_regression_pairs(edges_t, edges_th, g_th_np, N, n_neg_per_pos, rng):
    candidate = sorted(edges_t | edges_th)
    edge_union = edges_t | edges_th
    n_pos = max(len(candidate), 1)
    n_neg = min(n_pos * n_neg_per_pos, N * (N - 1) // 2)
    neg_pairs, attempts = [], 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 20:
        i, j = rng.integers(0, N, size=2)
        attempts += 1
        if i == j:
            continue
        pair = (min(int(i), int(j)), max(int(i), int(j)))
        if pair in edge_union:
            continue
        neg_pairs.append(pair)
    if not candidate and not neg_pairs:
        return [], []
    pos_labels = [float(g_th_np[i, j]) for (i, j) in candidate]
    neg_labels = [0.0] * len(neg_pairs)
    return candidate + neg_pairs, pos_labels + neg_labels


def build_edge_history(pairs, g_t_history, window):
    hist = np.zeros((len(pairs), window), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        for t, g in enumerate(g_t_history):
            hist[k, t] = np.log1p(g[i, j])
    return torch.tensor(hist, dtype=torch.float32)


@torch.no_grad()
def score_full_pairspace_full(model, node_feats_seq, adj_seq, g_t_history, iu, ju, batch_size,
                               device, horizon, esm_seq):
    model.eval()
    scores = np.zeros(len(iu), dtype=np.float32)
    node_h = model.node_encoder(node_feats_seq, adj_seq, esm_seq if model.use_esm_context else None)
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
        raw = model.decoder(combined).squeeze(-1)
        scores[start:end] = raw.cpu().numpy()
    return scores


def regression_metrics(scores, g_th_np, iu, ju):
    scores = np.asarray(scores, dtype=np.float64)
    true_log = np.log1p(g_th_np[iu, ju])
    mse = float(np.mean((scores - true_log) ** 2))
    rho, _ = spearmanr(scores, true_log)
    return {"mse_log1p": mse, "spearman": float(rho) if not np.isnan(rho) else float("nan")}


def frequency_baseline_scores(freq, iu, ju):
    return freq[iu] * freq[ju]


def parse_variant_windows(raw_list, months):
    """Returns list of (name, start_idx, end_idx) sorted by start_idx,
    clipped to what actually exists in `months`."""
    if raw_list is None:
        raw = DEFAULT_VARIANT_WINDOWS
    else:
        raw = []
        for entry in raw_list:
            name, start_month, width = entry.split(":")
            raw.append((name, start_month, int(width)))

    month_index = {m: i for i, m in enumerate(months)}
    windows = []
    for name, start_month, width in raw:
        if start_month not in month_index:
            log(f"  skipping {name}: {start_month} not in dataset range")
            continue
        start_idx = month_index[start_month]
        end_idx = min(start_idx + width - 1, len(months) - 1)
        windows.append((name, start_idx, end_idx))
    windows.sort(key=lambda w: w[1])
    return windows


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
    max_h = max(args.horizons)
    iu, ju = all_pairs(N)

    esm_cache = ESMEmbeddingCache(ROOT / args.esm_cache_path)
    log(f"Loaded ESM cache: {len(esm_cache.embeddings)} constellations, esm_dim={esm_cache.esm_dim}")

    month_cache = {}
    def get_month(idx):
        if idx not in month_cache:
            m = months[idx]
            g_t_np, f_t_np, occ, n_seq = load_month(graphs_dir, m)
            prev_f = month_cache[idx - 1][2] if (idx - 1) in month_cache else None
            nf, adj, freq, g_t, g_t_freq_np = build_month_tensors(g_t_np, f_t_np, occ, n_seq, N, prev_f)
            esm_emb = esm_cache.build_month_node_embeddings(occ, N)
            month_cache[idx] = (nf, adj, freq, g_t, occ, g_t_np, esm_emb, g_t_freq_np)
        return month_cache[idx]

    variant_windows = parse_variant_windows(args.variant_windows, months)
    log(f"\nVariant cycles ({len(variant_windows)}):")
    for name, s, e in variant_windows:
        log(f"  {name}: {months[s]} to {months[e]}")

    configs = {
        "full_model":       dict(use_gnn=True,  use_rnn=True,  use_edge_history=True,  use_esm_context=True),
        "no_gnn":           dict(use_gnn=False, use_rnn=True,  use_edge_history=True,  use_esm_context=True),
        "no_rnn":           dict(use_gnn=True,  use_rnn=False, use_edge_history=True,  use_esm_context=True),
        "no_edge_history":  dict(use_gnn=True,  use_rnn=True,  use_edge_history=False, use_esm_context=True),
        "no_esm_context":   dict(use_gnn=True,  use_rnn=True,  use_edge_history=True,  use_esm_context=False),
    }
    models = {name: GraphTemporalScorer(
        node_feat_dim=3, hidden_dim=args.hidden_dim,
        relation_names=["cooc", "struct", "profile_sim", "background_overlap"],
        edge_history_window=W, esm_dim=esm_cache.esm_dim, esm_adapter_dim=args.esm_adapter_dim, **cfg,
    ).to(device) for name, cfg in configs.items()}
    optimizers = {name: torch.optim.Adam(m.parameters(), lr=args.lr) for name, m in models.items()}

    def bulk_train_range(start_idx: int, end_idx_exclusive: int):
        """Combined multi-horizon training over every valid window whose
        input AND all horizon targets fall within [start_idx, end_idx_exclusive).
        Safe by construction: this range is never evaluated."""
        cutoffs = [t for t in range(max(W, start_idx), end_idx_exclusive)
                   if t - 1 + max_h < end_idx_exclusive]
        for t_idx in cutoffs:
            window_idxs = list(range(t_idx - W, t_idx))
            node_feats_seq, adj_seq, g_t_history, esm_seq = [], [], [], []
            for idx in window_idxs:
                nf, adj, freq, g_t, occ, g_t_np, esm_emb, g_t_freq_np = get_month(idx)
                node_feats_seq.append(nf.to(device))
                adj_seq.append({k: v.to(device) for k, v in adj.items()})
                g_t_history.append(g_t_freq_np)
                esm_seq.append(esm_emb.to(device))

            _, _, freq_t, g_t_t, occ_t, _, _, _ = get_month(t_idx - 1)
            edges_t = occupied_edge_set(occ_t)

            combined_pairs, combined_labels, combined_horizon_ids = [], [], []
            for h in args.horizons:
                target_idx_h = t_idx - 1 + h
                _, _, _, _, occ_th_h, g_th_np_h, _, _ = get_month(target_idx_h)
                edges_th_h = occupied_edge_set(occ_th_h)
                if args.target_type == "regression":
                    pairs_h, labels_h = sample_regression_pairs(
                        edges_t, edges_th_h, g_th_np_h, N, args.n_neg_per_pos, rng)
                else:
                    pairs_h, labels_h = sample_training_pairs(
                        edges_t, edges_th_h, N, args.n_neg_per_pos, rng)
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
                    raw_out = model(node_feats_seq, adj_seq, pair_i, pair_j,
                                    edge_hist if model.use_edge_history else None,
                                    horizon_ids_t if model.use_horizon_embed else None,
                                    esm_seq=esm_seq if model.use_esm_context else None)
                    if args.target_type == "regression":
                        loss = F.mse_loss(raw_out, torch.log1p(labels_t))
                    else:
                        loss = F.binary_cross_entropy_with_logits(raw_out, labels_t)
                    loss.backward()
                    opt.step()
        return len(cutoffs)

    results = {name: [] for name in configs}          # list of (variant, h, target_month, ap)
    results_reg = {name: [] for name in configs}       # list of (variant, h, target_month, mse, spearman)
    baseline_results = {"random": [], "naive_persistence": [], "frequency": []}
    baseline_results_reg = {"random": [], "naive_persistence": [], "frequency": []}

    def evaluate_range(variant_name: str, start_idx: int, end_idx: int):
        """FROZEN evaluation (no training) -- score every window whose
        context anchor falls in [start_idx, end_idx], at every horizon."""
        for ctx_idx in range(start_idx, end_idx + 1):
            t_idx = ctx_idx + 1  # window ending at ctx_idx, i.e. context anchor = months[ctx_idx]
            if t_idx - W < 0:
                continue
            window_idxs = list(range(t_idx - W, t_idx))
            node_feats_seq, adj_seq, g_t_history, esm_seq = [], [], [], []
            for idx in window_idxs:
                nf, adj, freq, g_t, occ, g_t_np, esm_emb, g_t_freq_np = get_month(idx)
                node_feats_seq.append(nf.to(device))
                adj_seq.append({k: v.to(device) for k, v in adj.items()})
                g_t_history.append(g_t_freq_np)
                esm_seq.append(esm_emb.to(device))
            _, _, freq_t, g_t_t, occ_t, _, _, _ = get_month(t_idx - 1)
            ctx_month = months[t_idx - 1]

            for h in args.horizons:
                target_idx = t_idx - 1 + h
                if target_idx >= len(months):
                    continue
                target_month = months[target_idx]
                _, _, _, _, occ_th, g_th_np, _, _ = get_month(target_idx)
                edges_th = occupied_edge_set(occ_th)

                for name, model in models.items():
                    scores = score_full_pairspace_full(model, node_feats_seq, adj_seq, g_t_history,
                                                        iu, ju, args.eval_pair_batch, device, h, esm_seq)
                    ap = full_pairspace_ap(scores, edges_th, iu, ju)
                    if not np.isnan(ap):
                        results[name].append((variant_name, h, target_month, ap))
                    if args.target_type == "regression":
                        rm = regression_metrics(scores, g_th_np, iu, ju)
                        results_reg[name].append((variant_name, h, target_month, rm["mse_log1p"], rm["spearman"]))

                rand_scores = rng.random(len(iu))
                ap_r = full_pairspace_ap(rand_scores, edges_th, iu, ju)
                if not np.isnan(ap_r):
                    baseline_results["random"].append((variant_name, h, target_month, ap_r))
                if args.target_type == "regression":
                    rm = regression_metrics(rand_scores, g_th_np, iu, ju)
                    baseline_results_reg["random"].append((variant_name, h, target_month, rm["mse_log1p"], rm["spearman"]))

                persist_scores = g_t_t[iu, ju].cpu().numpy()
                ap_p = full_pairspace_ap(persist_scores, edges_th, iu, ju)
                if not np.isnan(ap_p):
                    baseline_results["naive_persistence"].append((variant_name, h, target_month, ap_p))
                if args.target_type == "regression":
                    rm = regression_metrics(np.log1p(persist_scores), g_th_np, iu, ju)
                    baseline_results_reg["naive_persistence"].append((variant_name, h, target_month, rm["mse_log1p"], rm["spearman"]))

                freq_scores = frequency_baseline_scores(freq_t, iu, ju)
                ap_f = full_pairspace_ap(freq_scores, edges_th, iu, ju)
                if not np.isnan(ap_f):
                    baseline_results["frequency"].append((variant_name, h, target_month, ap_f))
                if args.target_type == "regression":
                    rm = regression_metrics(freq_scores, g_th_np, iu, ju)
                    baseline_results_reg["frequency"].append((variant_name, h, target_month, rm["mse_log1p"], rm["spearman"]))

    # ------------------------------------------------------------------
    # THE CYCLE: bulk-train up to each variant window, freeze + evaluate
    # that window, bulk-train through to the next (now including this
    # window as legitimate history), repeat.
    #
    # GUARD: W + max_h months of runway are required before ANY variant
    # window can produce even one valid training window (needs W months
    # of input AND the h=max_h target to land before the cutoff). Early
    # variants (e.g. Beta, Alpha, Gamma in the first year of data) may
    # not have that much history available yet. Evaluating with a
    # 0-window-trained (i.e. randomly initialized) model would produce
    # meaningless numbers silently -- skip evaluation in that case
    # instead, with a clear warning. Training still advances prev_train_end
    # regardless, so this window's real data still counts as legitimate
    # history for LATER cycles even if it couldn't be evaluated itself.
    # ------------------------------------------------------------------
    MIN_TRAIN_WINDOWS_TO_EVAL = 3
    MIN_TRAIN_WINDOWS_WARN = 10
    prev_train_end = 0
    for name, start_idx, end_idx in variant_windows:
        n_trained = bulk_train_range(prev_train_end, start_idx)
        log(f"\n[{name}] bulk-trained on {n_trained} windows (months {months[prev_train_end] if prev_train_end < len(months) else '?'} "
            f"to {months[start_idx - 1] if start_idx > 0 else '?'})")
        if n_trained < MIN_TRAIN_WINDOWS_TO_EVAL:
            log(f"[{name}] SKIPPED: only {n_trained} training windows available "
                f"(need W={W}+max_h={max_h}={W+max_h} months of runway before this "
                f"variant's start month; not enough history exists yet). Evaluating "
                f"an untrained model here would be meaningless. This variant's own "
                f"months still count as history for later cycles.")
        else:
            if n_trained < MIN_TRAIN_WINDOWS_WARN:
                log(f"[{name}] WARNING: only {n_trained} training windows -- results "
                    f"for this variant are low-confidence, treat with caution")
            evaluate_range(name, start_idx, end_idx)
            log(f"[{name}] evaluated months {months[start_idx]} to {months[end_idx]}")
        prev_train_end = end_idx + 1

    # ------------------------------------------------------------------
    # REPORTING: grouped by variant window, not by calendar era (that
    # split doesn't apply to this design).
    # ------------------------------------------------------------------
    def summarize(rows):
        vals = [r[-1] for r in rows]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    log("\n" + "=" * 70)
    log("RESULTS BY VARIANT WINDOW (full pair-space AP)")
    log("=" * 70)
    for vname, _, _ in variant_windows:
        log(f"\n--- {vname} ---")
        for h in args.horizons:
            log(f"  h={h}")
            for name in ["random", "naive_persistence", "frequency"]:
                rows = [r for r in baseline_results[name] if r[0] == vname and r[1] == h]
                m, s, n = summarize(rows)
                if n:
                    log(f"    {name:<20} AP = {m:.4f} +/- {s:.4f}  (n={n})")
            for name in configs:
                rows = [r for r in results[name] if r[0] == vname and r[1] == h]
                m, s, n = summarize(rows)
                if n:
                    log(f"    {name:<20} AP = {m:.4f} +/- {s:.4f}  (n={n})")

    if args.target_type == "regression":
        log("\n" + "=" * 70)
        log("REGRESSION METRICS BY VARIANT WINDOW")
        log("=" * 70)
        for vname, _, _ in variant_windows:
            log(f"\n--- {vname} ---")
            for h in args.horizons:
                log(f"  h={h}")
                for name in ["random", "naive_persistence", "frequency"]:
                    rows = [r for r in baseline_results_reg[name] if r[0] == vname and r[1] == h]
                    if rows:
                        mse = np.mean([r[3] for r in rows])
                        rho = np.mean([r[4] for r in rows if not np.isnan(r[4])])
                        log(f"    {name:<20} MSE={mse:.4f}  spearman={rho:.4f}  (n={len(rows)})")
                for name in configs:
                    rows = [r for r in results_reg[name] if r[0] == vname and r[1] == h]
                    if rows:
                        mse = np.mean([r[3] for r in rows])
                        rho = np.mean([r[4] for r in rows if not np.isnan(r[4])])
                        log(f"    {name:<20} MSE={mse:.4f}  spearman={rho:.4f}  (n={len(rows)})")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "18_variant_cycle_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "model", "variant", "horizon", "target_month", "ap", "mse_log1p", "spearman"])
        for source, res_dict, res_reg_dict in [("baseline", baseline_results, baseline_results_reg),
                                                 ("model", results, results_reg)]:
            for name in res_dict:
                reg_lookup = {(v, h, m): (mse, rho) for v, h, m, mse, rho in res_reg_dict[name]} \
                    if args.target_type == "regression" else {}
                for v, h, m, ap in res_dict[name]:
                    mse, rho = reg_lookup.get((v, h, m), ("", ""))
                    writer.writerow([source, name, v, h, m, ap, mse, rho])
    log(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
