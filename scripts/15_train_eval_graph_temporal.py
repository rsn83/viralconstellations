"""
Step 15: Train and evaluate GraphTemporalScorer on real (position,residue)
data, at h=1/3/6, with FULL pair-space evaluation against baselines and
a leave-one-component-out ablation.

--target_type binary (original): predict whether pair (i,j) newly
    co-occurs by t+h. Label = 0/1. Loss = BCE. Metric = AP.
--target_type regression (default): predict log1p(g_{t+h}[i,j]) --
    the actual co-occurrence COUNT, not just whether it's nonzero.
    This subsumes appearance (0 -> positive) AND growth/decline
    (positive -> different positive) AND extinction (positive -> 0)
    in one target, using information (g_t is already a real count
    matrix, not binary) that was previously being discarded down to
    0/1 for the appearance-only framing. Loss = MSE on log1p(count).
    Metrics = AP (still reported, using the raw predicted score to
    rank pairs -- monotonic transforms don't change AP, so this stays
    comparable to the binary run) AND Spearman correlation + MSE
    against the true log1p(count), which the binary framing cannot
    report at all since it never predicts magnitude.

Design decisions, all fixed, not re-litigated per run:
  - Windowing: W=6 input months -> encode ONCE -> decode h=1,3,6 from that
    SAME encoding (DySAT-style, no autoregressive feedback).
  - Reported AP: FULL pair-space -- every possible (i,j) pair in that
    month scored and compared against the true future graph.
  - Baselines, same eval, so numbers are directly comparable:
      1. random
      2. naive persistence (g_{t+h} = g_t)
      3. frequency-only (product of endpoint marginal frequencies)
  - Ablation: full model vs no-GNN vs no-RNN vs no-edge-history vs
    no-ESM-context.

Usage:
  python scripts/15_train_eval_graph_temporal.py --window 6 --epochs 20 --target_type regression
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
parser.add_argument("--esm_cache_path", type=str, default="outputs/esm_cache.pkl",
                     help="output of scripts/17_extract_esm_embeddings.py")
parser.add_argument("--esm_adapter_dim", type=int, default=32)
parser.add_argument("--target_type", type=str, default="regression", choices=["binary", "regression"],
                     help="binary: predict appearance only (0/1). regression: predict "
                          "log1p(count) -- appearance AND magnitude in one target.")
parser.add_argument("--pretrain_months", type=int, default=12,
                     help="number of initial months used ONLY for a warm-up training pass "
                          "(combined multi-horizon, no evaluation -- safe since nothing in "
                          "this period is ever graded). Live walk-forward evaluation/training "
                          "begins strictly after this, so no month used as a pretrain label "
                          "is ever later evaluated.")
args = parser.parse_args()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

from viralconstellations.model.graph_temporal_scorer_v2 import (
    GraphTemporalScorer, NaivePersistenceBaseline,
    compute_context_profile, profile_similarity_matrix,
    compute_distinct_constellation_stats, background_overlap_matrix,
)
from viralconstellations.model.esm_embeddings import ESMEmbeddingCache


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


def sample_regression_pairs(edges_t: set, edges_th: set, g_th_np: np.ndarray, N: int,
                             n_neg_per_pos: int, rng):
    """
    Regression version: candidates = every pair that is an edge at t
    OR at t+h (covers appearance, growth, decline, AND extinction --
    all in one set), labeled with the REAL count g_th_np[i,j] (0 for
    the extinction case). Plus a sampled set of true negatives (never
    an edge at either time) labeled 0, same negative-sampling ratio as
    the binary version, so training set size/imbalance is comparable.
    """
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


def regression_metrics(scores: np.ndarray, g_th_np: np.ndarray, iu: np.ndarray, ju: np.ndarray) -> dict:
    """scores: raw model output, trained to approximate log1p(count).
    Compared directly against the true log1p(count) over the full pair
    space -- same evaluation scope as full_pairspace_ap."""
    scores = np.asarray(scores, dtype=np.float64)  # guard against torch tensors sneaking in
    true_log = np.log1p(g_th_np[iu, ju])
    mse = float(np.mean((scores - true_log) ** 2))
    rho, _ = spearmanr(scores, true_log)
    return {"mse_log1p": mse, "spearman": float(rho) if not np.isnan(rho) else float("nan")}


def build_edge_history(pairs, g_t_history, window):
    hist = np.zeros((len(pairs), window), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        for t, g in enumerate(g_t_history):
            hist[k, t] = np.log1p(g[i, j])
    return torch.tensor(hist, dtype=torch.float32)


@torch.no_grad()
def score_full_pairspace(model, node_feats_seq, adj_seq, g_t_history, iu, ju, batch_size, device,
                          horizon, esm_seq=None):
    model.eval()
    scores = np.zeros(len(iu), dtype=np.float32)
    node_h = model.node_encoder(node_feats_seq, adj_seq,
                                 esm_seq if model.use_esm_context else None)
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
        # No sigmoid: AP is unaffected by monotonic transforms, and for
        # target_type=="regression" this raw value IS the prediction
        # (approximating log1p(count)), not a probability.
        scores[start:end] = raw.cpu().numpy()
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

    esm_cache = ESMEmbeddingCache(ROOT / args.esm_cache_path)
    log(f"Loaded ESM cache: {len(esm_cache.embeddings)} constellations, "
        f"esm_dim={esm_cache.esm_dim}")
    log(f"target_type={args.target_type}")

    month_cache = {}
    def get_month(idx):
        if idx not in month_cache:
            m = months[idx]
            g_t_np, f_t_np, occ, n_seq = load_month(graphs_dir, m)
            prev_f = month_cache[idx - 1][2] if (idx - 1) in month_cache else None
            nf, adj, freq, g_t = build_month_tensors(g_t_np, f_t_np, occ, n_seq, N, prev_f)
            esm_emb = esm_cache.build_month_node_embeddings(occ, N)  # (N, esm_dim), cheap
            month_cache[idx] = (nf, adj, freq, g_t, occ, g_t_np, esm_emb)
        return month_cache[idx]

    configs = {
        "full_model":       dict(use_gnn=True,  use_rnn=True,  use_edge_history=True,  use_esm_context=True),
        "no_gnn":           dict(use_gnn=False, use_rnn=True,  use_edge_history=True,  use_esm_context=True),
        "no_rnn":           dict(use_gnn=True,  use_rnn=False, use_edge_history=True,  use_esm_context=True),
        "no_edge_history":  dict(use_gnn=True,  use_rnn=True,  use_edge_history=False, use_esm_context=True),
        "no_esm_context":   dict(use_gnn=True,  use_rnn=True,  use_edge_history=True,  use_esm_context=False),
    }

    results = {name: {h: [] for h in args.horizons} for name in configs}
    baseline_results = {"random": {h: [] for h in args.horizons},
                         "naive_persistence": {h: [] for h in args.horizons},
                         "frequency": {h: [] for h in args.horizons}}
    # Only populated when target_type=="regression" -- (ctx_month, mse, spearman) tuples
    results_reg = {name: {h: [] for h in args.horizons} for name in configs}
    baseline_results_reg = {"random": {h: [] for h in args.horizons},
                             "naive_persistence": {h: [] for h in args.horizons},
                             "frequency": {h: [] for h in args.horizons}}

    max_h = max(args.horizons)

    # Pretrain cutoffs: windows whose input AND all target months stay
    # strictly within months[0:pretrain_months]. Safe to use ordinary
    # combined multi-horizon training here -- nothing in this range is
    # ever evaluated, so there is no future eval target to leak into.
    pretrain_cutoffs = [t for t in range(W, args.pretrain_months - max_h + 1) if t > 0]

    # Live cutoffs: start strictly at pretrain_months, so the first live
    # evaluation target is a month that was NEVER used as a pretrain
    # training label. Inputs may still reach back into the pretrain
    # period (that's fine -- past real data as features is not the
    # thing that needs guarding, only training LABELS).
    live_start = max(W, args.pretrain_months)
    cutoffs = list(range(live_start, len(months) - max_h))
    log(f"Pretrain windows: {len(pretrain_cutoffs)} (months 0-{args.pretrain_months-1}, train-only, no eval)")
    log(f"Live walk-forward windows: {len(cutoffs)}")

    models = {name: GraphTemporalScorer(
        node_feat_dim=3, hidden_dim=args.hidden_dim,
        relation_names=["cooc", "struct", "profile_sim", "background_overlap"],
        edge_history_window=W, esm_dim=esm_cache.esm_dim, esm_adapter_dim=args.esm_adapter_dim, **cfg,
    ).to(device) for name, cfg in configs.items()}
    optimizers = {name: torch.optim.Adam(m.parameters(), lr=args.lr) for name, m in models.items()}

    # ------------------------------------------------------------------
    # PRETRAIN PHASE: ordinary combined multi-horizon training (the
    # original, pre-fix approach), scoped ONLY to months that will never
    # be evaluated. This solves the cold-start problem -- without it,
    # the live phase's first ~6 windows would be undertrained by
    # necessity (nothing yet in the retroactive buffer to learn from).
    # Purely a warm-up: no results are recorded here.
    # ------------------------------------------------------------------
    for p_idx, t_idx in enumerate(pretrain_cutoffs):
        window_idxs = list(range(t_idx - W, t_idx))
        node_feats_seq, adj_seq, g_t_history, esm_seq = [], [], [], []
        for idx in window_idxs:
            nf, adj, freq, g_t, occ, g_t_np, esm_emb = get_month(idx)
            node_feats_seq.append(nf.to(device))
            adj_seq.append({k: v.to(device) for k, v in adj.items()})
            g_t_history.append(g_t_np)
            esm_seq.append(esm_emb.to(device))

        _, _, freq_t, g_t_t, occ_t, _, _ = get_month(t_idx - 1)
        edges_t = occupied_edge_set(occ_t)

        combined_pairs, combined_labels, combined_horizon_ids = [], [], []
        for h in args.horizons:
            target_idx_h = t_idx - 1 + h
            if target_idx_h >= len(months):
                continue
            _, _, _, _, occ_th_h, g_th_np_h, _ = get_month(target_idx_h)
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

        if p_idx % 5 == 0:
            log(f"  pretrain {p_idx+1}/{len(pretrain_cutoffs)}  ctx_month={months[t_idx-1]}  loss={loss.item():.4f}")

    log("Pretrain phase complete. Starting live walk-forward (evaluate-then-retroactively-train)...")

    # ------------------------------------------------------------------
    # LIVE PHASE -- how training and evaluation work now, in plain terms:
    #
    # At each window, we first EVALUATE the model's forecasts for
    # t+1/t+3/t+6 using ONLY weights that have never been trained on
    # ANY of those specific future months, under ANY horizon, from ANY
    # window (see proof in the comment below `retroactive train`).
    # Only AFTER recording that evaluation do we let the model learn
    # anything new.
    #
    # What it learns is NOT this window's own future guesses (that was
    # the leak). Instead: the month that JUST completed (today) was
    # exactly what some earlier window -- 1, 3, or 6 months back --
    # was trying to forecast at the time. Now that we actually know
    # what happened, we go back, re-run THAT earlier window's original
    # inputs through the CURRENT model, and grade/train it against
    # today's real outcome. A small rolling buffer keeps the last few
    # windows' input tensors around so this retroactive grading is
    # possible. This preserves training on all three horizons (so the
    # model still learns horizon-specific behavior) while guaranteeing
    # nothing is ever trained on before it has genuinely happened.
    # ------------------------------------------------------------------
    window_buffer = {}  # w_idx -> dict(node_feats_seq, adj_seq, esm_seq, g_t_history, edges_t)

    for w_idx, t_idx in enumerate(cutoffs):
        window_idxs = list(range(t_idx - W, t_idx))
        node_feats_seq, adj_seq, g_t_history, esm_seq = [], [], [], []
        for idx in window_idxs:
            nf, adj, freq, g_t, occ, g_t_np, esm_emb = get_month(idx)
            node_feats_seq.append(nf.to(device))
            adj_seq.append({k: v.to(device) for k, v in adj.items()})
            g_t_history.append(g_t_np)
            esm_seq.append(esm_emb.to(device))

        # "Today" = the last input month = this window's context anchor.
        # This IS the real outcome that has just become known.
        _, _, freq_t, g_t_t, occ_t, g_t_now_np, _ = get_month(t_idx - 1)
        edges_now = occupied_edge_set(occ_t)
        ctx_month = months[t_idx - 1]

        # ---- 1. EVALUATE FIRST, using weights not yet updated on today ----
        for h in args.horizons:
            target_idx = t_idx - 1 + h
            if target_idx >= len(months):
                continue
            _, _, _, _, occ_th, g_th_np, _ = get_month(target_idx)
            edges_th = occupied_edge_set(occ_th)

            for name, model in models.items():
                scores = score_full_pairspace(model, node_feats_seq, adj_seq, g_t_history,
                                               iu, ju, args.eval_pair_batch, device, horizon=h,
                                               esm_seq=esm_seq)
                ap = full_pairspace_ap(scores, edges_th, iu, ju)
                if not np.isnan(ap):
                    results[name][h].append((ctx_month, ap))
                if args.target_type == "regression":
                    rm = regression_metrics(scores, g_th_np, iu, ju)
                    results_reg[name][h].append((ctx_month, rm["mse_log1p"], rm["spearman"]))

            rand_scores = rng.random(len(iu))
            ap_r = full_pairspace_ap(rand_scores, edges_th, iu, ju)
            if not np.isnan(ap_r):
                baseline_results["random"][h].append((ctx_month, ap_r))
            if args.target_type == "regression":
                rm = regression_metrics(rand_scores, g_th_np, iu, ju)
                baseline_results_reg["random"][h].append((ctx_month, rm["mse_log1p"], rm["spearman"]))

            persist_scores = g_t_t[iu, ju].cpu().numpy()
            ap_p = full_pairspace_ap(persist_scores, edges_th, iu, ju)
            if not np.isnan(ap_p):
                baseline_results["naive_persistence"][h].append((ctx_month, ap_p))
            if args.target_type == "regression":
                rm = regression_metrics(np.log1p(persist_scores), g_th_np, iu, ju)
                baseline_results_reg["naive_persistence"][h].append((ctx_month, rm["mse_log1p"], rm["spearman"]))

            freq_scores = frequency_baseline_scores(freq_t, iu, ju)
            ap_f = full_pairspace_ap(freq_scores, edges_th, iu, ju)
            if not np.isnan(ap_f):
                baseline_results["frequency"][h].append((ctx_month, ap_f))
            if args.target_type == "regression":
                rm = regression_metrics(freq_scores, g_th_np, iu, ju)
                baseline_results_reg["frequency"][h].append((ctx_month, rm["mse_log1p"], rm["spearman"]))

        # ---- 2. Store this window's inputs for future retroactive training ----
        window_buffer[w_idx] = dict(
            node_feats_seq=node_feats_seq, adj_seq=adj_seq,
            esm_seq=esm_seq, g_t_history=g_t_history, edges_t=edges_now,
        )
        for old_key in [k for k in window_buffer if k < w_idx - max_h]:
            del window_buffer[old_key]  # bound buffer memory -- never looked at again

        # ---- 3. RETROACTIVE TRAIN: grade windows w_idx-1, w_idx-3, w_idx-6 ----
        # (whichever exist) against TODAY's now-known real outcome.
        # PROOF this cannot leak: at window w_idx-h, the model was only
        # ever asked to encode inputs -- no training happened using
        # today's month yet. The label used here (edges_now / g_t_now_np)
        # is attached to the CURRENT window's context month, which by
        # construction is exactly window (w_idx-h)'s h-step-ahead
        # target month. This is the FIRST time that label is used for
        # training, and it happens only after every window that could
        # possibly evaluate on this exact month has already done so
        # (any such window has window index <= w_idx, and all evaluation
        # above happens strictly before this step).
        last_loss = None
        for h in args.horizons:
            src_idx = w_idx - h
            if src_idx not in window_buffer:
                continue
            src = window_buffer[src_idx]
            if args.target_type == "regression":
                pairs, labels = sample_regression_pairs(
                    src["edges_t"], edges_now, g_t_now_np, N, args.n_neg_per_pos, rng)
            else:
                pairs, labels = sample_training_pairs(
                    src["edges_t"], edges_now, N, args.n_neg_per_pos, rng)
            if not pairs:
                continue

            edge_hist = build_edge_history(pairs, src["g_t_history"], W).to(device)
            pair_i = torch.tensor([p[0] for p in pairs], dtype=torch.long).to(device)
            pair_j = torch.tensor([p[1] for p in pairs], dtype=torch.long).to(device)
            labels_t = torch.tensor(labels, dtype=torch.float32).to(device)
            horizon_ids_t = torch.full((len(pairs),), h, dtype=torch.long).to(device)

            for name, model in models.items():
                model.train()
                opt = optimizers[name]
                for _ in range(args.epochs):
                    opt.zero_grad()
                    raw_out = model(src["node_feats_seq"], src["adj_seq"], pair_i, pair_j,
                                    edge_hist if model.use_edge_history else None,
                                    horizon_ids_t if model.use_horizon_embed else None,
                                    esm_seq=src["esm_seq"] if model.use_esm_context else None)
                    if args.target_type == "regression":
                        loss = F.mse_loss(raw_out, torch.log1p(labels_t))
                    else:
                        loss = F.binary_cross_entropy_with_logits(raw_out, labels_t)
                    loss.backward()
                    opt.step()
                last_loss = loss

        if w_idx % 5 == 0:
            loss_str = f"{last_loss.item():.4f}" if last_loss is not None else "n/a (buffer warming up)"
            log(f"  window {w_idx+1}/{len(cutoffs)}  ctx_month={ctx_month}  loss={loss_str}")

    esm_cache.report_coverage()

    def era_of(month_str: str) -> str:
        year = int(month_str[:4])
        return "high_volume_2020_2022" if year <= 2022 else "low_volume_2023_2026"

    def summarize(pairs: list[tuple[str, float]]) -> tuple[float, float, int]:
        vals = [ap for _, ap in pairs]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    def summarize_reg(triples: list[tuple[str, float, float]]) -> tuple[float, float, int]:
        """Mean Spearman correlation across windows (higher = better rank
        agreement with true magnitude), and mean MSE on log1p(count)."""
        if not triples:
            return float("nan"), float("nan"), 0
        mses = [t[1] for t in triples]
        rhos = [t[2] for t in triples if not np.isnan(t[2])]
        mean_rho = float(np.mean(rhos)) if rhos else float("nan")
        return float(np.mean(mses)), mean_rho, len(triples)

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

    if args.target_type == "regression":
        log("\n" + "=" * 70)
        log("REGRESSION METRICS (mean MSE on log1p(count), mean Spearman rho)")
        log("=" * 70)
        for h in args.horizons:
            log(f"\n--- horizon h={h} ---")
            for name in ["random", "naive_persistence", "frequency"]:
                mse, rho, n = summarize_reg(baseline_results_reg[name][h])
                if n:
                    log(f"  {name:<20} MSE={mse:.4f}  spearman={rho:.4f}  (n={n})")
            for name in configs:
                mse, rho, n = summarize_reg(results_reg[name][h])
                if n:
                    log(f"  {name:<20} MSE={mse:.4f}  spearman={rho:.4f}  (n={n})")

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

    if args.target_type == "regression":
        log("\n" + "=" * 70)
        log("REGRESSION METRICS ERA BREAKDOWN")
        log("=" * 70)
        for h in args.horizons:
            log(f"\n--- horizon h={h} ---")
            for era in ["high_volume_2020_2022", "low_volume_2023_2026"]:
                log(f"\n  [{era}]")
                for name in ["random", "naive_persistence", "frequency"]:
                    triples = [t for t in baseline_results_reg[name][h] if era_of(t[0]) == era]
                    mse, rho, n = summarize_reg(triples)
                    if n:
                        log(f"    {name:<20} MSE={mse:.4f}  spearman={rho:.4f}  (n={n})")
                for name in configs:
                    triples = [t for t in results_reg[name][h] if era_of(t[0]) == era]
                    mse, rho, n = summarize_reg(triples)
                    if n:
                        log(f"    {name:<20} MSE={mse:.4f}  spearman={rho:.4f}  (n={n})")


if __name__ == "__main__":
    main()
