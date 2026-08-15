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
parser.add_argument("--n_neg_fixed", type=int, default=0,
                     help="if > 0, use this ABSOLUTE negative candidate count every window, "
                          "instead of scaling with real positive count (--n_neg_per_pos). "
                          "Makes candidate-pool difficulty comparable across cycles, since "
                          "real positive count varies a lot with how much viral diversity "
                          "existed that month. Recommended for cross-cycle comparison, e.g. "
                          "--n_neg_fixed 2000.")
parser.add_argument("--eval_batch", type=int, default=2000)
parser.add_argument("--esm_cache_path", type=str, default="outputs/esm_cache.pkl")
parser.add_argument("--struct_prior_path", type=str, default="outputs/structural_prior.pt",
                     help="output of scripts/20_build_structural_prior.py -- real PDB-derived "
                          "structural proximity, static every month")
parser.add_argument("--esm_adapter_dim", type=int, default=32)
parser.add_argument("--use_attention_esm_pool", action="store_true", default=False,
                     help="use learnable attention pooling (Ilse et al. 2018) over raw "
                          "per-constellation ESM embeddings instead of a fixed "
                          "count-weighted mean -- see AttentionPoolESMAdapter. NOT YET "
                          "VALIDATED to help -- test A/B before trusting.")
parser.add_argument("--esm_pool_k", type=int, default=8,
                     help="max carrier constellations sampled per node per month for "
                          "attention pooling (only used if --use_attention_esm_pool)")
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--max_set_size", type=int, default=30,
                     help="cap on constellation size for attention/padding -- constellations "
                          "larger than this get truncated (rare; a warning is printed if it happens)")
parser.add_argument("--variant_windows", type=str, nargs="+", default=None)
# ---- forecast-eval additions ----
parser.add_argument("--min_count", type=int, default=3,
                    help="drop constellations seen fewer than this many times in a month. "
                         "min_count=1 keeps singletons, which are largely sequencing "
                         "artefacts: they inflate turnover (68%% of H_t+1 'new' at mc1 vs "
                         "38%% at mc10) because they appear once and vanish. mc=3 keeps "
                         "frontier parent coverage (no_subset=0.086) while removing the "
                         "churn; mc=10 discards too many parents (no_subset=0.177).")
parser.add_argument("--start_month", type=str, default=None,
                    help="ignore months before this (e.g. 2021-01)")
parser.add_argument("--end_month", type=str, default=None,
                    help="ignore months after this. Sampling density collapses after "
                         "2024 and frontier coverage degrades with it (d1: 0.69 in 2022 "
                         "-> 0.28 in 2026), so late months measure surveillance, not "
                         "evolution. Recommend 2024-12.")
parser.add_argument("--ref_anchored_node_features", type=str, default=None,
                    help="path to outputs/esm_node_features_ref.pkl from script 21. "
                         "When set, the ESM channel becomes STATIC reference-anchored "
                         "per-mutation contrasts instead of the per-month pooled mean "
                         "over constellations containing each node. The pooled mean "
                         "estimates E[emb | m present], which varies with lineage "
                         "composition rather than with the mutation -- a lineage "
                         "descriptor, not a mutation descriptor. Sequence context is "
                         "preserved (ESM still attends over the whole spike); only the "
                         "population-averaging is removed.")
parser.add_argument("--train_on_frontier", action="store_true", default=True,
                    help="draw TRAINING negatives from the frontier, matching the "
                         "evaluation pool. Previously training negatives were random "
                         "node sets (rng.choice(N, k)) while evaluation used frontier "
                         "candidates -- the model was graded on a task it never saw. "
                         "Random sets are separable by node commonness alone, which is "
                         "exactly what the frequency baseline computes.")
parser.add_argument("--no_train_on_frontier", dest="train_on_frontier",
                    action="store_false")
parser.add_argument("--n_frontier_negs", type=int, default=4000,
                    help="frontier negatives sampled per training window per horizon")
parser.add_argument("--use_set_history", action="store_true", default=True,
                    help="give the scorer each candidate's own log1p count trajectory "
                         "over the window, as a residual head. Without it the score is "
                         "a function of member-node embeddings only and cannot express "
                         "persistence, which is why copy_forward wins.")
parser.add_argument("--no_set_history", dest="use_set_history", action="store_false")
parser.add_argument("--frontier_top_parents", type=int, default=0,
                    help="expand from the N most abundant circulating constellations. "
                         "0 = ALL of them (recommended). Capping this destroys coverage: "
                         "median_parents_at_d1 == 1, so each new constellation has exactly "
                         "ONE valid parent and it is usually not among the most abundant.")
parser.add_argument("--frontier_top_nodes", type=int, default=0,
                    help="only add mutations from the M most frequent active nodes. "
                         "0 = ALL active nodes. This is the SAFER cap: a new constellation "
                         "usually adds a mutation that is already circulating somewhere.")
parser.add_argument("--frontier_max", type=int, default=0,
                    help="uniformly subsample the frontier to at most this many candidates. "
                         "0 = no subsampling (recommended). Subsampling removes reachable "
                         "positives at random and lowers coverage proportionally.")
args = parser.parse_args()

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

from viralconstellations.model.hypergraph_scorer import HypergraphTemporalScorer, build_incidence
from viralconstellations.model.esm_embeddings import ESMEmbeddingCache


def log(msg): print(msg, flush=True)


class RefAnchoredNodeFeatures:
    """Drop-in replacement for ESMEmbeddingCache exposing the same interface.

    Returns the SAME (N, d) matrix for every month. The features are properties
    of a (position, residue) substitution against the reference spike, so they
    do not change as the population changes -- a vocabulary lookup, not a
    per-month aggregate. Everything that legitimately varies month to month
    (frequency, trend, degree, incidence, set history) already reaches the
    model through other channels.
    """

    def __init__(self, path: Path, N: int):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        X = np.asarray(blob["features"], dtype=np.float32)
        if X.shape[0] != N:
            raise ValueError(
                f"node feature matrix has {X.shape[0]} rows but the graphs use N={N} "
                f"nodes. Rebuild with script 21 against the SAME posres_vocab.tsv.")
        self._X = torch.tensor(X, dtype=torch.float32)
        self.esm_dim = int(X.shape[1])
        self.names = blob.get("names", [])
        self.meta = blob.get("meta", {})
        self.embeddings = {}   # only used for a log line in main()

    def build_month_node_embeddings(self, occ, N):
        return self._X

    def build_month_node_raw_embeddings(self, occ, N, K=8):
        # attention pooling has nothing to pool over: there is one static
        # vector per node, not K per-constellation vectors.
        raise NotImplementedError(
            "--use_attention_esm_pool is incompatible with "
            "--ref_anchored_node_features. Attention pooling reweights which "
            "constellations to average over; these features are not averaged "
            "over constellations at all.")


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
                                 N: int, n_neg_per_pos: int, rng, max_set_size: int,
                                 n_neg_fixed: int = 0):
    """
    Positives: union of constellations present at context OR target
    month (covers appearance, growth, decline, extinction -- same
    principle as sample_regression_pairs in script 18, just at the
    correct hyperedge granularity). Label = real count at target month
    (0 if it existed at context but vanished by target -- extinction).
    Negatives: randomly sampled node sets of similar size, not
    matching any real positive.

    n_neg_fixed: if > 0, use this ABSOLUTE negative count instead of
    scaling with the number of real positives (n_pos * n_neg_per_pos).
    This matters for cross-cycle comparability: real positive count
    varies a lot with how much viral diversity existed that month
    (much higher in 2023-2025 than 2020), so scaling negatives with it
    means candidate-pool difficulty -- and therefore what "AP" even
    means -- silently varies across cycles too. A fixed negative count
    makes task difficulty comparable by construction, so AP
    differences across cycles reflect the model, not the candidate
    pool's shifting positive rate.
    """
    positive_sets = set(constellations_t.keys()) | set(constellations_th.keys())
    positive_sets = {c for c in positive_sets if 2 <= len(c) <= max_set_size}
    if not positive_sets:
        return [], []

    labels = [float(constellations_th.get(c, 0.0)) for c in positive_sets]
    sizes = [len(c) for c in positive_sets]

    if n_neg_fixed > 0:
        n_neg = n_neg_fixed
    else:
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


def frontier_candidates(constellations_t: dict, freq_t, max_set_size: int, rng,
                        top_parents: int = 0, top_nodes: int = 0, n_max: int = 0):
    """F(O_t): constellations reachable by adding ONE mutation to something
    circulating at t, excluding anything already circulating.

    Justified empirically: script 22 measured that ~62% of newly-appearing
    constellations at t+1 are exactly one addition from a circulating set,
    with a MEDIAN OF EXACTLY ONE such parent, stable across abundance
    thresholds (0.554 / 0.622 / 0.582 at min_count 1 / 3 / 10) and degrading
    with horizon (0.62 / 0.51 / 0.27 at h = 1 / 3 / 6). Hence h=1.

    The unique-parent property is why top_parents must default to ALL: the one
    valid parent of a new constellation is usually a rare circulating set, not
    an abundant one. Capping parents at 300 collapsed measured coverage from
    ~62% to 3.3%.

    Cap `top_nodes` instead if the pool must be reduced -- restricting which
    mutation gets ADDED is far less destructive than restricting which set it
    is added TO.

    Returns (candidates, n_before_subsample).
    """
    parents = [(c, v) for c, v in constellations_t.items()
               if 1 <= len(c) < max_set_size]
    if not parents:
        return [], 0
    if top_parents and top_parents > 0:
        parents.sort(key=lambda kv: -kv[1])
        parents = parents[:top_parents]
    parents = [c for c, _ in parents]

    active = sorted({m for c in constellations_t.keys() for m in c})
    if top_nodes and top_nodes > 0 and len(active) > top_nodes:
        active = sorted(active, key=lambda m: -float(freq_t[m]))[:top_nodes]

    occupied = set(constellations_t.keys())
    out = set()
    for p in parents:
        for m in active:
            if m in p:
                continue
            cand = frozenset(set(p) | {m})
            if 2 <= len(cand) <= max_set_size and cand not in occupied:
                out.add(cand)

    n_raw = len(out)
    out = list(out)
    if n_max and n_max > 0 and len(out) > n_max:
        idx = rng.choice(len(out), size=n_max, replace=False)
        out = [out[i] for i in idx]
    return out, n_raw


def build_forecast_pool(constellations_t: dict, constellations_th: dict, freq_t,
                        max_set_size: int, rng, top_parents=0, top_nodes=0, n_max=0):
    """Candidate pool = everything circulating at t, PLUS the frontier.

    Built ONLY from month t. Nothing from t+h enters the pool construction --
    that was the defect in sample_hyperedge_candidates, which took
    `set(constellations_t) | set(constellations_th)`, i.e. built the candidate
    list partly from the answer.

    Returns (candidates, actual_counts_at_th, is_old_mask, stats).
    """
    occ_t = {c for c in constellations_t.keys() if 2 <= len(c) <= max_set_size}
    frontier_list, n_frontier_raw = frontier_candidates(
        constellations_t, freq_t, max_set_size, rng,
        top_parents=top_parents, top_nodes=top_nodes, n_max=n_max)
    frontier = set(frontier_list)
    new_th = {c for c in constellations_th.keys()
              if c not in occ_t and 2 <= len(c) <= max_set_size}

    cands = list(occ_t | frontier)
    if not cands:
        return [], None, None, {}

    actual = np.array([float(constellations_th.get(c, 0.0)) for c in cands])
    is_old = np.array([c in occ_t for c in cands], dtype=bool)

    stats = {
        "n_pool": len(cands),
        "n_old": int(is_old.sum()),
        "n_frontier": len(frontier),
        "n_frontier_raw": n_frontier_raw,
        "subsample_frac": (len(frontier) / n_frontier_raw) if n_frontier_raw else float("nan"),
        "n_new_total": len(new_th),
        "n_new_in_pool": len(new_th & frontier),
        "coverage": (len(new_th & frontier) / len(new_th)) if new_th else float("nan"),
    }
    return cands, actual, is_old, stats


def build_set_history(candidates, constellations_seq, device):
    """(B, W) tensor of log1p(count) for each candidate across the window months.

    Frontier candidates are all-zero: they have never been observed. That is
    correct and informative -- the scorer's 'has history' flag turns it into an
    explicit regime indicator rather than a silently-imputed value.
    """
    W = len(constellations_seq)
    hist = np.zeros((len(candidates), W), dtype=np.float32)
    for w, cons in enumerate(constellations_seq):
        if not cons:
            continue
        for i, c in enumerate(candidates):
            v = cons.get(c)
            if v:
                hist[i, w] = np.log1p(v)
    return torch.tensor(hist, dtype=torch.float32).to(device)


def sample_frontier_training_pool(constellations_t, constellations_th, freq_t,
                                  max_set_size, rng, n_negs,
                                  top_parents=0, top_nodes=0):
    """Training pool built the SAME WAY as the evaluation pool.

    positives : constellations circulating at t (label = count at t+h, which is
                0 for extinctions -- that is a real negative outcome, not a
                random set) PLUS newly-appearing constellations the frontier
                reaches.
    negatives : frontier candidates that did NOT appear at t+h. These are hard:
                real, plausible, built from currently-circulating mutations, so
                they cannot be separated by node commonness.

    Using the labels here is training, not leakage -- the leak was in the
    EVALUATION pool, which previously drew candidates from constellations_th.
    """
    occ_t = {c for c in constellations_t.keys() if 2 <= len(c) <= max_set_size}
    frontier, _ = frontier_candidates(constellations_t, freq_t, max_set_size, rng,
                                      top_parents=top_parents, top_nodes=top_nodes,
                                      n_max=0)
    frontier = list(frontier)
    new_th = {c for c in constellations_th.keys()
              if c not in occ_t and 2 <= len(c) <= max_set_size}

    fset = set(frontier)
    pos_new = list(new_th & fset)
    neg_pool = [c for c in frontier if c not in new_th]
    if len(neg_pool) > n_negs:
        idx = rng.choice(len(neg_pool), size=n_negs, replace=False)
        neg_pool = [neg_pool[i] for i in idx]

    cands = list(occ_t) + pos_new + neg_pool
    labels = [float(constellations_th.get(c, 0.0)) for c in cands]
    return cands, labels


def _prf(pred_mask, true_mask):
    tp = int((pred_mask & true_mask).sum())
    npred, ntrue = int(pred_mask.sum()), int(true_mask.sum())
    p = tp / npred if npred else float("nan")
    r = tp / ntrue if ntrue else float("nan")
    f = (2 * p * r / (p + r)) if (npred and ntrue and (p + r) > 0) else float("nan")
    return p, r, f, tp, npred, ntrue


def hypergraph_reconstruction(scores, actual, is_old, K, scores_are_log1p):
    """Top-K by score = the predicted hypergraph at t+h. Compare to the actual.

    Reported THREE ways, because a single aggregate F1 is dominated by
    persistence and moves for reasons unrelated to forecasting:
      all     -- the whole predicted hypergraph
      persist -- restricted to sets already circulating at t
      appear  -- restricted to sets NOT circulating at t (the open problem)

    K = |H_t|: predict as many constellations as currently exist. Rank-based,
    no threshold, no calibration assumption, and no leakage (K comes from the
    present, not the future).
    """
    scores = np.asarray(scores, dtype=np.float64)
    true_mask = actual > 0
    K = int(min(max(K, 1), len(scores)))
    order = np.argsort(-scores, kind="stable")[:K]
    pred_mask = np.zeros(len(scores), dtype=bool)
    pred_mask[order] = True

    out = {"K": K}
    for tag, sel in [("all", np.ones(len(scores), dtype=bool)),
                     ("persist", is_old), ("appear", ~is_old)]:
        p, r, f, tp, npred, ntrue = _prf(pred_mask & sel, true_mask & sel)
        out[f"{tag}_prec"], out[f"{tag}_rec"], out[f"{tag}_f1"] = p, r, f
        out[f"{tag}_tp"], out[f"{tag}_npred"], out[f"{tag}_ntrue"] = tp, npred, ntrue

    hit = pred_mask & true_mask
    if hit.sum() >= 3:
        a_log = np.log1p(actual[hit])
        rho, _ = spearmanr(scores[hit], a_log)
        out["w_spearman"] = float(rho) if not np.isnan(rho) else float("nan")
        out["w_mse_log1p"] = float(np.mean((scores[hit] - a_log) ** 2)) \
            if scores_are_log1p else float("nan")
    else:
        out["w_spearman"] = float("nan")
        out["w_mse_log1p"] = float("nan")
    return out


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
    _n_all = len(months)
    if args.start_month:
        months = [m for m in months if m >= args.start_month]
    if args.end_month:
        months = [m for m in months if m <= args.end_month]
    if len(months) != _n_all:
        log(f"month range restricted: {_n_all} -> {len(months)} months "
            f"({months[0]} .. {months[-1]})")
    vocab_df = pd.read_csv(graphs_dir / "posres_vocab.tsv", sep="\t")
    N = len(vocab_df)
    log(f"N={N} (position,residue) nodes, {len(months)} months")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    W = args.window
    max_h = max(args.horizons)

    if args.ref_anchored_node_features:
        if args.use_attention_esm_pool:
            raise SystemExit("--use_attention_esm_pool is incompatible with "
                             "--ref_anchored_node_features (nothing to pool over).")
        esm_cache = RefAnchoredNodeFeatures(ROOT / args.ref_anchored_node_features, N)
        log(f"ESM channel = REFERENCE-ANCHORED per-mutation contrasts "
            f"(static, dim={esm_cache.esm_dim})")
        log(f"  features: {esm_cache.names}")
        log(f"  meta: {esm_cache.meta}")
    else:
        esm_cache = ESMEmbeddingCache(ROOT / args.esm_cache_path)
        log(f"Loaded ESM cache: {len(esm_cache.embeddings)} constellations, "
            f"esm_dim={esm_cache.esm_dim}")
        log("  NOTE: this is the per-month POOLED mean over constellations "
            "containing each node -- confounded with lineage. See "
            "--ref_anchored_node_features.")

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
            if args.use_attention_esm_pool:
                esm_emb = esm_cache.build_month_node_raw_embeddings(occ, N, K=args.esm_pool_k)
            else:
                esm_emb = esm_cache.build_month_node_embeddings(occ, N)
            constellations = constellations_of(occ)
            if args.min_count > 1:
                constellations = {c: v for c, v in constellations.items()
                                  if v >= args.min_count}
            month_cache[idx] = (nf, freq, incidence, occ, esm_emb, constellations)
        return month_cache[idx]

    def esm_to_device(esm_emb):
        if args.use_attention_esm_pool:
            raw, mask = esm_emb
            return raw.to(device), mask.to(device)
        return esm_emb.to(device)

    variant_windows = parse_variant_windows(args.variant_windows, months)
    log(f"\nVariant cycles ({len(variant_windows)}):")
    for name, s, e in variant_windows:
        log(f"  {name}: {months[s]} to {months[e]}")

    _H = args.use_set_history
    configs = {
        "full_model":       dict(use_gnn=True,  use_rnn=True,  use_esm_context=True,  use_struct=True,  use_set_history=_H),
        "no_gnn":           dict(use_gnn=False, use_rnn=True,  use_esm_context=True,  use_struct=True,  use_set_history=_H),
        "no_rnn":           dict(use_gnn=True,  use_rnn=False, use_esm_context=True,  use_struct=True,  use_set_history=_H),
        "no_esm_context":   dict(use_gnn=True,  use_rnn=True,  use_esm_context=False, use_struct=True,  use_set_history=_H),
        "no_struct":        dict(use_gnn=True,  use_rnn=True,  use_esm_context=True,  use_struct=False, use_set_history=_H),
    }
    if _H:
        # direct test of how much performance is persistence vs. learned structure
        configs["no_set_history"] = dict(use_gnn=True, use_rnn=True, use_esm_context=True,
                                          use_struct=True, use_set_history=False)
    log(f"\nTraining {len(configs)} models per window every cycle: {list(configs.keys())}")
    # NOTE: no "no_edge_history" ablation here -- edge-history was a
    # PAIRWISE concept (a specific pair's own g_t trajectory). It
    # doesn't have a natural hyperedge analog, so it's dropped rather
    # than forced into a shape that doesn't fit.

    def fresh_models():
        m = {name: HypergraphTemporalScorer(
            node_feat_dim=3, hidden_dim=args.hidden_dim,
            esm_dim=esm_cache.esm_dim, esm_adapter_dim=args.esm_adapter_dim,
            dropout=args.dropout, use_attention_esm_pool=args.use_attention_esm_pool,
            history_len=args.window, **cfg,
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
            node_feats_seq, incidence_seq, esm_seq, constellations_seq = [], [], [], []
            for idx in window_idxs:
                nf, freq, incidence, occ, esm_emb, constellations = get_month(idx)
                node_feats_seq.append(nf.to(device))
                incidence_seq.append(incidence)  # already on device from build_incidence
                esm_seq.append(esm_to_device(esm_emb))
                constellations_seq.append(constellations)

            _, freq_t, _, occ_t, _, constellations_t = get_month(t_idx - 1)

            all_candidates, all_labels, all_horizon_ids = [], [], []
            for h in args.horizons:
                target_idx_h = t_idx - 1 + h
                _, _, _, occ_th_h, _, constellations_th = get_month(target_idx_h)
                if args.train_on_frontier:
                    cands, labs = sample_frontier_training_pool(
                        constellations_t, constellations_th, freq_t, args.max_set_size,
                        rng, args.n_frontier_negs,
                        top_parents=args.frontier_top_parents,
                        top_nodes=args.frontier_top_nodes)
                else:
                    cands, labs = sample_hyperedge_candidates(
                        constellations_t, constellations_th, N, args.n_neg_per_pos, rng,
                        args.max_set_size, n_neg_fixed=args.n_neg_fixed)
                all_candidates.extend(cands)
                all_labels.extend(labs)
                all_horizon_ids.extend([h] * len(cands))

            if not all_candidates:
                continue
            member_indices, member_mask = pad_candidates(all_candidates, device, args.max_set_size)
            labels_t = torch.tensor(all_labels, dtype=torch.float32).to(device)
            horizon_ids_t = torch.tensor(all_horizon_ids, dtype=torch.long).to(device)
            set_hist_t = build_set_history(all_candidates, constellations_seq, device) \
                if args.use_set_history else None

            for name, model in models.items():
                model.train()
                opt = optimizers[name]
                for _ in range(args.epochs):
                    opt.zero_grad()
                    raw_out = model(node_feats_seq, incidence_seq, member_indices, member_mask,
                                     struct_adj=struct_adj if model.node_encoder.use_struct else None,
                                     esm_seq=esm_seq if model.node_encoder.use_esm_context else None,
                                     horizon_ids=horizon_ids_t,
                                     set_history=set_hist_t if model.use_set_history else None)
                    loss = F.mse_loss(raw_out, torch.log1p(labels_t))
                    loss.backward()
                    opt.step()
        return len(candidates_t)

    results = {name: [] for name in configs}
    results_reg = {name: [] for name in configs}
    baseline_results = {"random": [], "naive_persistence": [], "frequency": []}
    baseline_results_reg = {"random": [], "naive_persistence": [], "frequency": []}
    recon_results = []   # hypergraph reconstruction: one dict per (config, cell)

    @torch.no_grad()
    def score_candidates(model, node_feats_seq, incidence_seq, esm_seq, member_indices, member_mask,
                          horizon_ids_t, batch_size, set_history=None):
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
            if model.use_set_history:
                sh = set_history[start:end]
                hh = (sh.abs().sum(dim=-1, keepdim=True) > 0).float()
                s = s + model.history_head(torch.cat([sh, hh], dim=-1)).squeeze(-1)
            scores[start:end] = s.cpu().numpy()
        return scores

    def evaluate_range(models, variant_name: str, start_idx: int, end_idx: int):
        for ctx_idx in range(start_idx, end_idx + 1):
            t_idx = ctx_idx + 1
            if t_idx - W < 0:
                continue
            window_idxs = list(range(t_idx - W, t_idx))
            node_feats_seq, incidence_seq, esm_seq, constellations_seq = [], [], [], []
            for idx in window_idxs:
                nf, freq, incidence, occ, esm_emb, constellations = get_month(idx)
                node_feats_seq.append(nf.to(device))
                incidence_seq.append(incidence)
                esm_seq.append(esm_to_device(esm_emb))
                constellations_seq.append(constellations)
            _, freq_t, _, occ_t, _, constellations_t = get_month(t_idx - 1)
            ctx_month = months[t_idx - 1]

            for h in args.horizons:
                target_idx = t_idx - 1 + h
                if target_idx >= len(months):
                    continue
                target_month = months[target_idx]
                _, _, _, occ_th, _, constellations_th = get_month(target_idx)

                candidates, labels = sample_hyperedge_candidates(
                    constellations_t, constellations_th, N, args.n_neg_per_pos, rng,
                    args.max_set_size, n_neg_fixed=args.n_neg_fixed)
                if not candidates:
                    continue
                member_indices, member_mask = pad_candidates(candidates, device, args.max_set_size)
                labels_arr = np.array(labels, dtype=np.float64)
                binary_labels = (labels_arr > 0).astype(np.int32)
                horizon_ids_t = torch.full((len(candidates),), h, dtype=torch.long).to(device)

                sh_old = build_set_history(candidates, constellations_seq, device) \
                    if args.use_set_history else None
                for name, model in models.items():
                    scores = score_candidates(model, node_feats_seq, incidence_seq, esm_seq,
                                               member_indices, member_mask, horizon_ids_t,
                                               args.eval_batch, set_history=sh_old)
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

                # ---------- HYPERGRAPH RECONSTRUCTION (leak-free pool) ----------
                cands_r, actual_r, is_old_r, stats_r = build_forecast_pool(
                    constellations_t, constellations_th, freq_t, args.max_set_size, rng,
                    top_parents=args.frontier_top_parents,
                    top_nodes=args.frontier_top_nodes, n_max=args.frontier_max)

                if cands_r and (actual_r > 0).sum() > 0:
                    mi_r, mm_r = pad_candidates(cands_r, device, args.max_set_size)
                    hz_r = torch.full((len(cands_r),), h, dtype=torch.long).to(device)
                    sh_r = build_set_history(cands_r, constellations_seq, device) \
                        if args.use_set_history else None
                    K = stats_r["n_old"]

                    def _rec(cfg, sc, is_log1p):
                        m = hypergraph_reconstruction(sc, actual_r, is_old_r, K, is_log1p)
                        m.update({"config": cfg, "variant": variant_name, "horizon": h,
                                  "ctx_month": ctx_month, "target_month": target_month,
                                  **stats_r})
                        recon_results.append(m)

                    for name_m, model_m in models.items():
                        sc = score_candidates(model_m, node_feats_seq, incidence_seq, esm_seq,
                                               mi_r, mm_r, hz_r, args.eval_batch,
                                               set_history=sh_r)
                        _rec(name_m, sc, True)

                    _rec("copy_forward",
                         np.array([np.log1p(constellations_t.get(c, 0.0)) for c in cands_r]), True)
                    _rec("frequency", frequency_baseline_scores(cands_r, freq_t), False)
                    _rec("random", rng.random(len(cands_r)), False)

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
            random_rows_by_h = {}
            for h in args.horizons:
                rh = [r for r in baseline_results["random"] if r[0] == name and r[1] == h]
                random_rows_by_h[h] = float(np.mean([r[-1] for r in rh])) if rh else float("nan")
            for cfg_name in ["frequency", "full_model"]:
                src = baseline_results if cfg_name == "frequency" else results
                rows = [r for r in src[cfg_name] if r[0] == name]
                by_target = {}
                for _, h, ctx_month, target_month, ap in rows:
                    by_target.setdefault(target_month, []).append((h, ctx_month, ap))
                log(f"  [{cfg_name}]")
                for target_month in sorted(by_target):
                    preds = sorted(by_target[target_month], key=lambda p: p[0])
                    pred_str = ", ".join(
                        f"h={h}(from {ctx})={ap:.4f}(lift={ap/random_rows_by_h[h]:.2f}x)"
                        if random_rows_by_h.get(h) else f"h={h}(from {ctx})={ap:.4f}"
                        for h, ctx, ap in preds
                    )
                    flag = "  <-- predicted MORE THAN ONCE" if len(preds) > 1 else ""
                    log(f"    {target_month}: {pred_str}{flag}")

            # ---------- HYPERGRAPH RECONSTRUCTION TABLE ----------
            rr = [r for r in recon_results if r["variant"] == name]
            if rr:
                covs = [r["coverage"] for r in rr if not np.isnan(r["coverage"])]
                log(f"\n[{name}] HYPERGRAPH RECONSTRUCTION at t+h  (top-K, K=|H_t|)")
                if covs:
                    sf = [r["subsample_frac"] for r in rr if not np.isnan(r.get("subsample_frac", np.nan))]
                    npool = np.mean([r["n_pool"] for r in rr])
                    log(f"  frontier reaches {np.mean(covs):.1%} of genuinely new "
                        f"constellations -- HARD CEILING on appear_rec")
                    log(f"  pool={npool:.0f}  frontier_raw={np.mean([r['n_frontier_raw'] for r in rr]):.0f}"
                        f"  kept={np.mean(sf) if sf else 1.0:.1%}"
                        f"  new_total={np.mean([r['n_new_total'] for r in rr]):.0f}")
                    if np.mean(covs) < 0.40:
                        log(f"  !! coverage far below the ~62% measured by script 22. "
                            f"If kept<100%, raise --frontier_max (0=off). Otherwise raise "
                            f"--frontier_top_parents (0=all) / --frontier_top_nodes (0=all).")
                log(f"  {'config':<18}{'h':>2} | {'F1':>6}{'P':>7}{'R':>7} | "
                    f"{'persF1':>7}{'persR':>7} | {'appF1':>7}{'appR':>7}{'appN':>6} | "
                    f"{'wRho':>6}{'wMSE':>8}")
                order = ["random", "frequency", "copy_forward"] + list(configs.keys())
                for h in sorted({r["horizon"] for r in rr}):
                    for cfg in order:
                        rows = [r for r in rr if r["config"] == cfg and r["horizon"] == h]
                        if not rows:
                            continue
                        def mu(k):
                            v = [r[k] for r in rows if not np.isnan(r[k])]
                            return float(np.mean(v)) if v else float("nan")
                        log(f"  {cfg:<18}{h:>2} | {mu('all_f1'):6.3f}{mu('all_prec'):7.3f}"
                            f"{mu('all_rec'):7.3f} | {mu('persist_f1'):7.3f}{mu('persist_rec'):7.3f}"
                            f" | {mu('appear_f1'):7.3f}{mu('appear_rec'):7.3f}"
                            f"{int(np.mean([r['appear_ntrue'] for r in rows])):6d} | "
                            f"{mu('w_spearman'):6.3f}{mu('w_mse_log1p'):8.3f}")
                log(f"  CHECK: copy_forward appR must be 0.000 -- it predicts H_t unchanged "
                    f"and cannot name anything new. Nonzero => is_old mask is broken.")

    def summarize(rows):
        vals = [r[-1] for r in rows]
        if not vals:
            return float("nan"), float("nan"), 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    log("\n" + "=" * 70)
    log("RESULTS BY VARIANT WINDOW (candidate-pool AP -- NOT directly comparable")
    log("to script 18's full-pairspace AP, see module docstring. 'lift' = AP / random's")
    log("AP for the SAME cell -- use this, not raw AP, to compare strength across cycles,")
    log("since candidate-pool difficulty (positive rate) varies a lot by how much real")
    log("viral diversity existed that month -- see --n_neg_fixed to control this directly.)")
    log("=" * 70)
    for vname, _, _ in variant_windows:
        log(f"\n--- {vname} ---")
        for h in args.horizons:
            log(f"  h={h}")
            random_rows = [r for r in baseline_results["random"] if r[0] == vname and r[1] == h]
            random_mean, _, _ = summarize(random_rows)
            for name in ["random", "naive_persistence", "frequency"]:
                rows = [r for r in baseline_results[name] if r[0] == vname and r[1] == h]
                m, s, n = summarize(rows)
                if n:
                    lift = m / random_mean if random_mean and not np.isnan(random_mean) else float("nan")
                    log(f"    {name:<20} AP = {m:.4f} +/- {s:.4f}  lift={lift:.2f}x  (n={n})")
            for name in configs:
                rows = [r for r in results[name] if r[0] == vname and r[1] == h]
                m, s, n = summarize(rows)
                if n:
                    lift = m / random_mean if random_mean and not np.isnan(random_mean) else float("nan")
                    log(f"    {name:<20} AP = {m:.4f} +/- {s:.4f}  lift={lift:.2f}x  (n={n})")

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

    if recon_results:
        csv_rec = out_dir / "19_hypergraph_reconstruction.csv"
        keys = sorted(recon_results[0].keys())
        with open(csv_rec, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys)
            wtr.writeheader()
            for r in recon_results:
                wtr.writerow(r)
        log(f"Wrote {csv_rec}")

        log("\n" + "=" * 70)
        log("HYPERGRAPH RECONSTRUCTION -- POOLED ACROSS VARIANTS")
        log("=" * 70)
        rdf = pd.DataFrame(recon_results)
        for h in sorted(rdf["horizon"].unique()):
            log(f"\n  h={h}")
            log(f"  {'config':<18} {'cells':>5} | {'F1':>6}{'P':>7}{'R':>7} | "
                f"{'persF1':>7} | {'appF1':>7}{'appR':>7} | {'wRho':>6}{'wMSE':>8}")
            sub = rdf[rdf["horizon"] == h]
            for cfg in ["random", "frequency", "copy_forward"] + list(configs.keys()):
                g = sub[sub["config"] == cfg]
                if not len(g):
                    continue
                log(f"  {cfg:<18} {len(g):>5} | {g['all_f1'].mean():6.3f}"
                    f"{g['all_prec'].mean():7.3f}{g['all_rec'].mean():7.3f} | "
                    f"{g['persist_f1'].mean():7.3f} | {g['appear_f1'].mean():7.3f}"
                    f"{g['appear_rec'].mean():7.3f} | {g['w_spearman'].mean():6.3f}"
                    f"{g['w_mse_log1p'].mean():8.3f}")
        log(f"\n  mean frontier coverage (recall ceiling on appear): "
            f"{rdf['coverage'].mean():.3f}")


if __name__ == "__main__":
    main()
