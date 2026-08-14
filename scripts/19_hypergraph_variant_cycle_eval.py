"""
scripts/19_hypergraph_variant_cycle_eval.py

Hyperedge counterpart to 18_variant_cycle_eval.py -- same leak-free
cycle design (fresh model per variant, bulk-train from scratch, freeze,
evaluate), same CLI shape, same variant windows. The difference is
entirely in WHAT gets scored:

  18_variant_cycle_eval.py: candidate = a PAIR (i,j), built by
      decomposing each real constellation into C(k,2) pairs -- lossy,
      many-to-one (different constellations can share pairwise
      projections).
  19 (this file): candidate = the FULL constellation itself, taken
      directly from occ.keys() -- no decomposition, no information
      loss. Scored via HyperSAGNN self-attention (hypergraph_scorer.py),
      which naturally handles variable-size sets.

EVALUATION METHODOLOGY DIFFERS FROM SCRIPT 18, HONESTLY FLAGGED:
  Script 18 scores the FULL pair space (every possible pair) -- feasible
  because there are "only" ~N^2/2 pairs. There is no equivalent for
  hyperedges: the candidate space is 2^N, computationally impossible to
  enumerate. Standard practice in hyperedge-prediction literature
  (Benson et al. PNAS 2018 and others) is instead to score a SAMPLED
  candidate pool: real historical/future constellations (positives) +
  randomly sampled non-existent node sets (negatives), and compute AP
  over that pool. That's what this script does. AP numbers between 18
  and 19 are NOT directly comparable in absolute terms (different
  candidate pools, different base rates) -- what IS comparable is each
  script's OWN model vs ITS OWN baselines, and the general shape of
  results (does structure help, does it degrade with horizon, etc).

BASELINES, hyperedge-native versions of the same ideas as script 18:
  - random: random score
  - naive_persistence: predicted count = this exact constellation's
    count at the context month (0 if it wasn't occupied then)
  - frequency: product of each member node's marginal frequency --
    this is EXACTLY your project's original "frequency-weighted,
    independence-assumed baseline" (the one your whole research
    question is framed around beating), now finally implemented at
    the correct multi-way granularity instead of pairwise.

Usage (mirrors 18 exactly):
  python scripts/19_hypergraph_variant_cycle_eval.py --window 3 --horizons 1 2 3 \
      --variant_windows Alpha:2020-11:2 JN1:2023-12:2 LP81:2025-01:2
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
parser.add_argument("--n_neg_per_pos", type=int, default=10,
                     help="fewer than script 18's default -- hyperedge negative sampling "
                          "is more expensive per candidate (variable-size, attention), and "
                          "the candidate pool is already much smaller than full pair space")
parser.add_argument("--eval_batch", type=int, default=2000)
parser.add_argument("--esm_cache_path", type=str, default="outputs/esm_cache.pkl")
parser.add_argument("--struct_prior_path", type=str, default="outputs/structural_prior.pt",
                     help="output of scripts/20_build_structural_prior.py -- real PDB-derived "
                          "structural proximity, static every month")
parser.add_argument("--esm_adapter_dim", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--max_set_size", type=int, default=30,
                     help="cap on constellation size for attention/padding -- constellations "
                          "larger than this get truncated (rare; a warning is printed if it happens)")
parser.add_argument("--variant_windows", type=str, nargs="+", default=None)
args = parser.parse_args()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

from viralconstellations.model.hypergraph_scorer import HypergraphTemporalScorer, build_incidence
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
    """Node-level scalar features only -- no pairwise adjacency. The
    GNN step now uses true hypergraph convolution (build_incidence,
    hypergraph_scorer.py) instead of a pairwise adjacency dict."""
    g_t = torch.tensor(g_t_np, dtype=torch.float32)
    f_t = torch.tensor(f_t_np, dtype=torch.float32)

    freq = f_t / max(n_seq, 1)
    freq_trend = (freq - torch.tensor(prev_freq_np, dtype=torch.float32)) if prev_freq_np is not None else torch.zeros(N)
    degree = g_t.sum(dim=-1) / max(g_t.sum().item(), 1.0)  # still a useful scalar summary feature
    node_feats = torch.stack([freq, freq_trend, degree], dim=-1)
    return node_feats, freq.numpy()


def constellations_of(occ: dict) -> dict:
    """{frozenset(nodes): count} -- normalizes whatever raw key type
    occupied.pkl uses into a stable frozenset-keyed dict."""
    out = {}
    for c, v in occ.items():
        count = v if isinstance(v, (int, float)) else 1
        out[frozenset(c)] = count
    return out


def sample_hyperedge_candidates(constellations_t: dict, constellations_th: dict,
                                 N: int, n_neg_per_pos: int, rng, max_set_size: int):
    """
    Positives: union of constellations present at context OR target
    month (covers appearance, growth, decline, extinction -- same
    principle as sample_regression_pairs in script 18, just at the
    correct hyperedge granularity). Label = real count at target month
    (0 if it existed at context but vanished by target -- extinction).
    Negatives: randomly sampled node sets of similar size, not
    matching any real positive.
    """
    positive_sets = set(constellations_t.keys()) | set(constellations_th.keys())
    positive_sets = {c for c in positive_sets if 2 <= len(c) <= max_set_size}
    if not positive_sets:
        return [], []

    labels = [float(constellations_th.get(c, 0.0)) for c in positive_sets]
    sizes = [len(c) for c in positive_sets]

    n_neg = min(len(positive_sets) * n_neg_per_pos, 5000)
    neg_candidates, attempts = [], 0
    while len(neg_candidates) < n_neg and attempts < n_neg * 20:
        k = int(rng.choice(sizes))
        nodes = rng.choice(N, size=min(k, N), replace=False)
        fs = frozenset(nodes.tolist())
        attempts += 1
        if fs in positive_sets or len(fs) < 2:
            continue
        neg_candidates.append(fs)

    all_candidates = list(positive_sets) + neg_candidates
    all_labels = labels + [0.0] * len(neg_candidates)
    return all_candidates, all_labels


def pad_candidates(candidates: list, device, max_set_size: int):
    """Returns (member_indices, member_mask) padded tensors."""
    batch = len(candidates)
    max_len = min(max(len(c) for c in candidates), max_set_size)
    member_indices = torch.zeros(batch, max_len, dtype=torch.long)
    member_mask = torch.zeros(batch, max_len, dtype=torch.bool)
    for i, c in enumerate(candidates):
        nodes = list(c)[:max_set_size]
        member_indices[i, :len(nodes)] = torch.tensor(nodes, dtype=torch.long)
        member_mask[i, :len(nodes)] = True
    return member_indices.to(device), member_mask.to(device)


def full_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    return average_precision_score(labels, scores)


def regression_metrics(scores: np.ndarray, true_counts: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    true_log = np.log1p(true_counts)
    mse = float(np.mean((scores - true_log) ** 2))
    rho, _ = spearmanr(scores, true_log)
    return {"mse_log1p": mse, "spearman": float(rho) if not np.isnan(rho) else float("nan")}


def naive_persistence_scores(candidates, constellations_t: dict) -> np.ndarray:
    return np.array([constellations_t.get(c, 0.0) for c in candidates], dtype=np.float64)


def frequency_baseline_scores(candidates, freq: np.ndarray) -> np.ndarray:
    """Product of member marginal frequencies -- the multi-way version
    of the pairwise independence baseline. This is your project's
    original 'frequency-weighted, independence-assumed baseline',
    finally at the correct granularity."""
    out = np.zeros(len(candidates), dtype=np.float64)
    for i, c in enumerate(candidates):
        p = 1.0
        for node in c:
            p *= max(freq[node], 1e-12)
        out[i] = p
    return out


def parse_variant_windows(raw_list, months):
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

    esm_cache = ESMEmbeddingCache(ROOT / args.esm_cache_path)
    log(f"Loaded ESM cache: {len(esm_cache.embeddings)} constellations, esm_dim={esm_cache.esm_dim}")

    struct_prior_path = ROOT / args.struct_prior_path
    if struct_prior_path.exists():
        struct_data = torch.load(struct_prior_path, map_location=device)
        struct_adj = struct_data["struct"].to(device)
        n_resolved = int(struct_data["resolved_mask"].sum())
        log(f"Loaded structural prior: {struct_prior_path} "
            f"({n_resolved}/{N} nodes resolved, threshold={struct_data['contact_threshold']}A)")
    else:
        log(f"WARNING: {struct_prior_path} not found -- run scripts/20_build_structural_prior.py "
            f"first. Proceeding with struct_adj=None (use_struct effectively disabled).")
        struct_adj = None

    month_cache = {}
    def get_month(idx):
        if idx not in month_cache:
            m = months[idx]
            g_t_np, f_t_np, occ, n_seq = load_month(graphs_dir, m)
            prev_f = month_cache[idx - 1][1] if (idx - 1) in month_cache else None
            nf, freq = build_month_tensors(g_t_np, f_t_np, occ, n_seq, N, prev_f)
            incidence = build_incidence(occ, N, device)  # None if degenerate (0 constellations)
            esm_emb = esm_cache.build_month_node_embeddings(occ, N)
            constellations = constellations_of(occ)
            month_cache[idx] = (nf, freq, incidence, occ, esm_emb, constellations)
        return month_cache[idx]

    variant_windows = parse_variant_windows(args.variant_windows, months)
    log(f"\nVariant cycles ({len(variant_windows)}):")
    for name, s, e in variant_windows:
        log(f"  {name}: {months[s]} to {months[e]}")

    configs = {
        "full_model":       dict(use_gnn=True,  use_rnn=True,  use_esm_context=True,  use_struct=True),
        "no_gnn":           dict(use_gnn=False, use_rnn=True,  use_esm_context=True,  use_struct=True),
        "no_rnn":           dict(use_gnn=True,  use_rnn=False, use_esm_context=True,  use_struct=True),
        "no_esm_context":   dict(use_gnn=True,  use_rnn=True,  use_esm_context=False, use_struct=True),
        "no_struct":        dict(use_gnn=True,  use_rnn=True,  use_esm_context=True,  use_struct=False),
    }
    log(f"\nTraining {len(configs)} models per window every cycle: {list(configs.keys())}")
    # NOTE: no "no_edge_history" ablation here -- edge-history was a
    # PAIRWISE concept (a specific pair's own g_t trajectory). It
    # doesn't have a natural hyperedge analog, so it's dropped rather
    # than forced into a shape that doesn't fit.

    def fresh_models():
        m = {name: HypergraphTemporalScorer(
            node_feat_dim=3, hidden_dim=args.hidden_dim,
            esm_dim=esm_cache.esm_dim, esm_adapter_dim=args.esm_adapter_dim,
            dropout=args.dropout, **cfg,
        ).to(device) for name, cfg in configs.items()}
        o = {name: torch.optim.Adam(mm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
             for name, mm in m.items()}
        return m, o

    def bulk_train_range(models, optimizers, end_idx_exclusive: int):
        candidates_t = [t for t in range(W, end_idx_exclusive)
                        if t - 1 + max_h < end_idx_exclusive]
        log(f"    (bulk-training {len(candidates_t)} windows x {len(models)} models each...)")
        for c_idx, t_idx in enumerate(candidates_t):
            if c_idx % 5 == 0 or c_idx == len(candidates_t) - 1:
                log(f"      window {c_idx+1}/{len(candidates_t)}  (ctx_month={months[t_idx-1]})")
            window_idxs = list(range(t_idx - W, t_idx))
            node_feats_seq, incidence_seq, esm_seq = [], [], []
            for idx in window_idxs:
                nf, freq, incidence, occ, esm_emb, constellations = get_month(idx)
                node_feats_seq.append(nf.to(device))
                incidence_seq.append(incidence)  # already on device from build_incidence
                esm_seq.append(esm_emb.to(device))

            _, freq_t, _, occ_t, _, constellations_t = get_month(t_idx - 1)

            all_candidates, all_labels, all_horizon_ids = [], [], []
            for h in args.horizons:
                target_idx_h = t_idx - 1 + h
                _, _, _, occ_th_h, _, constellations_th = get_month(target_idx_h)
                cands, labs = sample_hyperedge_candidates(
                    constellations_t, constellations_th, N, args.n_neg_per_pos, rng, args.max_set_size)
                all_candidates.extend(cands)
                all_labels.extend(labs)
                all_horizon_ids.extend([h] * len(cands))

            if not all_candidates:
                continue
            member_indices, member_mask = pad_candidates(all_candidates, device, args.max_set_size)
            labels_t = torch.tensor(all_labels, dtype=torch.float32).to(device)
            horizon_ids_t = torch.tensor(all_horizon_ids, dtype=torch.long).to(device)

            for name, model in models.items():
                model.train()
                opt = optimizers[name]
                for _ in range(args.epochs):
                    opt.zero_grad()
                    raw_out = model(node_feats_seq, incidence_seq, member_indices, member_mask,
                                     struct_adj=struct_adj if model.node_encoder.use_struct else None,
                                     esm_seq=esm_seq if model.node_encoder.use_esm_context else None,
                                     horizon_ids=horizon_ids_t)
                    loss = F.mse_loss(raw_out, torch.log1p(labels_t))
                    loss.backward()
                    opt.step()
        return len(candidates_t)

    results = {name: [] for name in configs}
    results_reg = {name: [] for name in configs}
    baseline_results = {"random": [], "naive_persistence": [], "frequency": []}
    baseline_results_reg = {"random": [], "naive_persistence": [], "frequency": []}

    @torch.no_grad()
    def score_candidates(model, node_feats_seq, incidence_seq, esm_seq, member_indices, member_mask,
                          horizon_ids_t, batch_size):
        model.eval()
        n = member_indices.shape[0]
        scores = np.zeros(n, dtype=np.float32)
        node_h = model.node_encoder(
            node_feats_seq, incidence_seq,
            struct_adj=struct_adj if model.node_encoder.use_struct else None,
            esm_seq=esm_seq if model.node_encoder.use_esm_context else None,
        )
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mi, mm, hz = member_indices[start:end], member_mask[start:end], horizon_ids_t[start:end]
            member_embeds = node_h[mi]
            if model.use_horizon_embed:
                member_embeds = member_embeds + model.horizon_embed(hz).unsqueeze(1)
            s = model.hyperedge_scorer(member_embeds, mm)
            scores[start:end] = s.cpu().numpy()
        return scores

    def evaluate_range(models, variant_name: str, start_idx: int, end_idx: int):
        for ctx_idx in range(start_idx, end_idx + 1):
            t_idx = ctx_idx + 1
            if t_idx - W < 0:
                continue
            window_idxs = list(range(t_idx - W, t_idx))
            node_feats_seq, incidence_seq, esm_seq = [], [], []
            for idx in window_idxs:
                nf, freq, incidence, occ, esm_emb, constellations = get_month(idx)
                node_feats_seq.append(nf.to(device))
                incidence_seq.append(incidence)
                esm_seq.append(esm_emb.to(device))
            _, freq_t, _, occ_t, _, constellations_t = get_month(t_idx - 1)
            ctx_month = months[t_idx - 1]

            for h in args.horizons:
                target_idx = t_idx - 1 + h
                if target_idx >= len(months):
                    continue
                target_month = months[target_idx]
                _, _, _, occ_th, _, constellations_th = get_month(target_idx)

                candidates, labels = sample_hyperedge_candidates(
                    constellations_t, constellations_th, N, args.n_neg_per_pos, rng, args.max_set_size)
                if not candidates:
                    continue
                member_indices, member_mask = pad_candidates(candidates, device, args.max_set_size)
                labels_arr = np.array(labels, dtype=np.float64)
                binary_labels = (labels_arr > 0).astype(np.int32)
                horizon_ids_t = torch.full((len(candidates),), h, dtype=torch.long).to(device)

                for name, model in models.items():
                    scores = score_candidates(model, node_feats_seq, incidence_seq, esm_seq,
                                               member_indices, member_mask, horizon_ids_t,
                                               args.eval_batch)
                    ap = full_ap(scores, binary_labels)
                    if not np.isnan(ap):
                        results[name].append((variant_name, h, ctx_month, target_month, ap))
                    rm = regression_metrics(scores, labels_arr)
                    results_reg[name].append((variant_name, h, ctx_month, target_month,
                                               rm["mse_log1p"], rm["spearman"]))

                rand_scores = rng.random(len(candidates))
                ap_r = full_ap(rand_scores, binary_labels)
                if not np.isnan(ap_r):
                    baseline_results["random"].append((variant_name, h, ctx_month, target_month, ap_r))
                rm = regression_metrics(rand_scores, labels_arr)
                baseline_results_reg["random"].append((variant_name, h, ctx_month, target_month,
                                                         rm["mse_log1p"], rm["spearman"]))

                persist_scores = naive_persistence_scores(candidates, constellations_t)
                ap_p = full_ap(persist_scores, binary_labels)
                if not np.isnan(ap_p):
                    baseline_results["naive_persistence"].append((variant_name, h, ctx_month, target_month, ap_p))
                rm = regression_metrics(np.log1p(persist_scores), labels_arr)
                baseline_results_reg["naive_persistence"].append((variant_name, h, ctx_month, target_month,
                                                                    rm["mse_log1p"], rm["spearman"]))

                freq_scores = frequency_baseline_scores(candidates, freq_t)
                ap_f = full_ap(freq_scores, binary_labels)
                if not np.isnan(ap_f):
                    baseline_results["frequency"].append((variant_name, h, ctx_month, target_month, ap_f))
                rm = regression_metrics(freq_scores, labels_arr)
                baseline_results_reg["frequency"].append((variant_name, h, ctx_month, target_month,
                                                            rm["mse_log1p"], rm["spearman"]))

    MIN_TRAIN_WINDOWS_TO_EVAL = 3
    MIN_TRAIN_WINDOWS_WARN = 10
    for name, start_idx, end_idx in variant_windows:
        models, optimizers = fresh_models()
        n_trained = bulk_train_range(models, optimizers, start_idx)
        log(f"\n[{name}] bulk-trained FROM SCRATCH on {n_trained} windows "
            f"(all months before {months[start_idx]})")
        if n_trained < MIN_TRAIN_WINDOWS_TO_EVAL:
            log(f"[{name}] SKIPPED: only {n_trained} training windows available.")
        else:
            if n_trained < MIN_TRAIN_WINDOWS_WARN:
                log(f"[{name}] WARNING: only {n_trained} training windows -- low-confidence")
            evaluate_range(models, name, start_idx, end_idx)
            log(f"[{name}] evaluated months {months[start_idx]} to {months[end_idx]}")
            log(f"[{name}] results by target month:")
            for cfg_name in ["frequency", "full_model"]:
                src = baseline_results if cfg_name == "frequency" else results
                rows = [r for r in src[cfg_name] if r[0] == name]
                by_target = {}
                for _, h, ctx_month, target_month, ap in rows:
                    by_target.setdefault(target_month, []).append((h, ctx_month, ap))
                log(f"  [{cfg_name}]")
                for target_month in sorted(by_target):
                    preds = sorted(by_target[target_month], key=lambda p: p[0])
                    pred_str = ", ".join(f"h={h}(from {ctx})={ap:.4f}" for h, ctx, ap in preds)
                    flag = "  <-- predicted MORE THAN ONCE" if len(preds) > 1 else ""
                    log(f"    {target_month}: {pred_str}{flag}")

    def summarize(rows):
        vals = [r[-1] for r in rows]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    log("\n" + "=" * 70)
    log("RESULTS BY VARIANT WINDOW (candidate-pool AP -- NOT directly comparable")
    log("to script 18's full-pairspace AP, see module docstring)")
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
                    mse = np.mean([r[4] for r in rows])
                    rho = np.mean([r[5] for r in rows if not np.isnan(r[5])])
                    log(f"    {name:<20} MSE={mse:.4f}  spearman={rho:.4f}  (n={len(rows)})")
            for name in configs:
                rows = [r for r in results_reg[name] if r[0] == vname and r[1] == h]
                if rows:
                    mse = np.mean([r[4] for r in rows])
                    rho = np.mean([r[5] for r in rows if not np.isnan(r[5])])
                    log(f"    {name:<20} MSE={mse:.4f}  spearman={rho:.4f}  (n={len(rows)})")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "19_hypergraph_variant_cycle_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "model", "variant", "horizon", "ctx_month", "target_month",
                          "ap", "mse_log1p", "spearman"])
        for source, res_dict, res_reg_dict in [("baseline", baseline_results, baseline_results_reg),
                                                 ("model", results, results_reg)]:
            for name in res_dict:
                reg_lookup = {(v, h, cm, tm): (mse, rho) for v, h, cm, tm, mse, rho in res_reg_dict[name]}
                for v, h, cm, tm, ap in res_dict[name]:
                    mse, rho = reg_lookup.get((v, h, cm, tm), ("", ""))
                    writer.writerow([source, name, v, h, cm, tm, ap, mse, rho])
    log(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
