"""
Frontier Component.

Scientific claim: background structure predicts which new mutation
constellations will appear, better than independence assumption.

Three scorers compared at each walk-forward window:
  1. Random baseline       — rank F(O_t) candidates randomly
  2. Logistic regression   — hand-crafted features (interpretable ablation)
  3. Neural model scorer   — ConstellationTransformer log-likelihood (primary)

The neural scorer uses the SAME trained model that generates sequences
to score frontier candidates directly — no separate classifier.
Score = log P(candidate constellation | hidden state h_{t+h}, horizon h)
under the near-clean denoising step (t=1).

This makes the pipeline coherent: one model does both generation and scoring.
The logistic regression shows which hand-crafted features matter,
and comparing it to the neural scorer shows whether learned representations
add value beyond those features.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Set, FrozenSet, Tuple
from collections import Counter


Constellation = FrozenSet[int]   # set of mutated position indices


# ── Occupied constellations ───────────────────────────────────────────────────

def compute_occupied(
    mat:      np.ndarray,
    top_k:    int   = 500,
    min_freq: float = 0.001,
) -> Dict[Constellation, float]:
    """
    Extract distinct occupied constellations from a monthly matrix.
    Constellation = frozenset of mutated position indices (binary: mutated or not).
    Returns {constellation: frequency}.
    """
    n = len(mat)
    if n == 0:
        return {}
    counts: Counter = Counter()
    for i in range(n):
        pattern = frozenset(int(j) for j in range(mat.shape[1]) if mat[i, j] != 0)
        counts[pattern] += 1
    return {
        p: c / n
        for p, c in counts.most_common(top_k)
        if c / n >= min_freq
    }


# ── Frontier F(O_t) ───────────────────────────────────────────────────────────

def compute_frontier(
    occupied: Dict[Constellation, float],
    P:        int,
) -> Dict[Constellation, dict]:
    """
    F(O_t): all constellations reachable by adding one mutation to O_t.
    Excludes already-occupied constellations.
    Returns {candidate: {parents, parent_freqs, parent_depths, ...}}.
    """
    frontier: Dict[Constellation, dict] = {}
    for constellation, freq in occupied.items():
        depth = len(constellation)
        for j in range(P):
            if j in constellation:
                continue
            candidate = frozenset(constellation | {j})
            if candidate in occupied:
                continue
            if candidate not in frontier:
                frontier[candidate] = {
                    "parents": [], "parent_freqs": [],
                    "parent_depths": [], "n_parents": 0,
                    "max_parent_freq": 0.0,
                }
            d = frontier[candidate]
            d["parents"].append(constellation)
            d["parent_freqs"].append(freq)
            d["parent_depths"].append(depth)
            d["n_parents"] += 1
            d["max_parent_freq"] = max(d["max_parent_freq"], freq)
    return frontier


# ── New constellations ────────────────────────────────────────────────────────

def compute_new_constellations(
    mat_t:    np.ndarray,
    mat_th:   np.ndarray,
    min_freq: float = 0.001,
) -> Tuple[Set[Constellation], Set[Constellation]]:
    """
    Returns (occupied_t, new_in_th) where new_in_th = O_{t+h} minus O_t.
    These are the ground-truth targets for prediction.
    """
    occ_t  = set(compute_occupied(mat_t,  top_k=2000, min_freq=min_freq).keys())
    occ_th = set(compute_occupied(mat_th, top_k=2000, min_freq=min_freq).keys())
    return occ_t, occ_th - occ_t


# ── Frontier coverage benchmark ───────────────────────────────────────────────

def frontier_coverage_benchmark(
    mat_t:     np.ndarray,
    mat_th:    np.ndarray,
    P:         int,
    hamming_r: int   = 1,
    min_freq:  float = 0.001,
) -> dict:
    """
    What fraction of new constellations in month t+h are within
    hamming_r mutations of some occupied constellation in O_t?

    hamming_r=1: strict frontier (one mutation away) — original framing
    hamming_r=2: two mutations away — tests whether new constellations
                 arrive via two simultaneous mutations (recombination etc.)

    Coverage < 0.5 at hamming_r=1 but > 0.8 at hamming_r=2 means new
    constellations typically require two mutations, not one.
    """
    occupied_t, new_in_th = compute_new_constellations(mat_t, mat_th, min_freq)
    if not new_in_th:
        return {"frontier_coverage": 1.0, "n_new": 0,
                "n_in_frontier": 0, "n_occupied_t": len(occupied_t),
                "n_frontier": 0, "hamming_r": hamming_r}

    if hamming_r == 1:
        frontier     = compute_frontier({c: 1.0 for c in occupied_t}, P)
        frontier_set = set(frontier.keys())
        n_covered    = sum(1 for c in new_in_th if c in frontier_set)
    else:
        # For hamming_r > 1: check symmetric difference size
        occ_list  = list(occupied_t)
        n_covered = 0
        for c in new_in_th:
            for occ in occ_list:
                diff = len(c.symmetric_difference(occ))
                if diff <= hamming_r:
                    n_covered += 1
                    break
        frontier_set = set()  # not computed for hamming_r > 1

    return {
        "frontier_coverage": n_covered / len(new_in_th),
        "n_new":             len(new_in_th),
        "n_in_frontier":     n_covered,
        "n_occupied_t":      len(occupied_t),
        "n_frontier":        len(frontier_set),
        "hamming_r":         hamming_r,
    }


# ── Candidate → categorical sequence ─────────────────────────────────────────

def candidate_to_sequence(
    candidate:    Constellation,
    pred_posfreq: np.ndarray,   # (P, 21) predicted residue distribution
    P:            int,
) -> np.ndarray:
    """
    Convert a frontier candidate (frozenset of positions) to a categorical
    sequence (P,) int8 by filling in the most likely non-reference residue
    at each mutated position, as predicted by FrequencyRegressionHead.

    This is necessary to score candidates with the ConstellationTransformer,
    which operates on categorical sequences, not binary patterns.
    """
    seq = np.zeros(P, dtype=np.int8)
    for pos in candidate:
        if pos < P:
            non_ref_probs = pred_posfreq[pos, 1:]      # residues 1-20
            best_residue  = int(np.argmax(non_ref_probs)) + 1
            seq[pos]      = best_residue
    return seq


# ── Neural scorer: score candidates using ConstellationTransformer ────────────

@torch.no_grad()
def score_candidates_neural(
    model,                          # ConstellationTransformer
    candidates:    List[Constellation],
    pred_posfreq:  np.ndarray,     # (P, 21)
    h_state:       torch.Tensor,   # (d_model,) hidden state at t+h
    horizon:       int,
    P:             int,
    device:        torch.device,
    batch_size:    int = 256,
) -> List[float]:
    """
    Score each frontier candidate using the ConstellationTransformer.

    Score = log P(candidate | h_{t+h}, horizon) at near-clean step t=1.

    Computed as: Σ_j log P(x_j | candidate, t=1, h_{t+h}, horizon)
    where x_j is the predicted best residue at position j.

    Higher score → model thinks this constellation is more probable.

    This is coherent: the same model that generates sequences scores
    candidates. No separate classifier needed.
    Batched for efficiency — all candidates in a few forward passes.
    """
    model.eval()

    # Convert all candidates to categorical sequences
    seqs = np.stack([
        candidate_to_sequence(c, pred_posfreq, P) for c in candidates
    ])  # (n_candidates, P)

    scores = []
    for start in range(0, len(seqs), batch_size):
        batch = torch.tensor(
            seqs[start:start+batch_size].astype(np.int64), device=device
        )  # (B, P)
        B = batch.shape[0]

        t_tens = torch.full((B,), 1,       dtype=torch.long, device=device)
        h_tens = torch.full((B,), horizon, dtype=torch.long, device=device)
        ctx    = h_state.unsqueeze(0).expand(B, -1)

        logits   = model(batch, t_tens, ctx, h_tens)      # (B, P, 21)
        log_prob = F.log_softmax(logits, dim=-1)           # (B, P, 21)

        # Log probability of each position's actual residue
        target_lp = log_prob.gather(
            2, batch.unsqueeze(-1)
        ).squeeze(-1)                                      # (B, P)

        # Sum over all positions = total sequence log-likelihood
        seq_scores = target_lp.sum(dim=-1)                # (B,)
        scores.extend(seq_scores.cpu().tolist())

    return scores


# ── Logistic regression scorer: hand-crafted features ────────────────────────

FEATURE_NAMES = [
    "pred_freq_new_pos",      # predicted frequency of new position
    "max_parent_freq",         # frequency of most common parent
    "log_n_parents",           # number of parents reaching this candidate
    "mean_parent_depth",       # mean depth of parent constellations
    "jaccard_best_parent",     # Jaccard similarity to best parent
    "freq_trend_new_pos",      # change in frequency at new position
    "coo_support",             # mean pairwise co-occurrence support
]


def extract_features(
    candidate:    Constellation,
    info:         dict,
    pred_posfreq: np.ndarray,
    prev_posfreq: np.ndarray,
    P:            int,
) -> np.ndarray:
    parents       = info["parents"]
    parent_freqs  = info["parent_freqs"]
    parent_depths = info["parent_depths"]

    new_positions = list(set().union(*[candidate - p for p in parents]))

    feat1 = float(np.mean([
        1.0 - pred_posfreq[j, 0] for j in new_positions if j < P
    ])) if new_positions else 0.0

    feat2 = float(info["max_parent_freq"])
    feat3 = float(np.log1p(info["n_parents"]))
    feat4 = float(np.mean(parent_depths)) if parent_depths else 0.0

    best_parent = parents[int(np.argmax(parent_freqs))]
    feat5 = len(candidate & best_parent) / max(len(candidate | best_parent), 1)

    if new_positions:
        feat6 = float(np.mean([
            (1.0 - pred_posfreq[j, 0]) - (1.0 - prev_posfreq[j, 0])
            for j in new_positions if j < P
        ]))
    else:
        feat6 = 0.0

    positions = list(candidate)
    if len(positions) >= 2:
        pairs = [(positions[i], positions[k])
                 for i in range(len(positions))
                 for k in range(i+1, len(positions))]
        feat7 = float(np.mean([
            (1-pred_posfreq[i, 0] if i < P else 0) *
            (1-pred_posfreq[k, 0] if k < P else 0)
            for i, k in pairs
        ]))
    else:
        feat7 = 0.0

    return np.array([feat1, feat2, feat3, feat4, feat5, feat6, feat7],
                    dtype=np.float32)


class LogisticFrontierScorer:
    """
    Interpretable ablation baseline.
    Shows which hand-crafted features matter.
    Compared against neural scorer to quantify value of learned representations.
    """
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.model  = LogisticRegression(C=1.0, max_iter=1000,
                                         class_weight="balanced")
        self.scaler = StandardScaler()
        self.fitted = False

    def collect(self, mat_t, mat_th, pred_posfreq, prev_posfreq, P,
                top_k=200, min_freq=0.001):
        occupied_t  = compute_occupied(mat_t, top_k=top_k, min_freq=min_freq)
        _, new_in_th = compute_new_constellations(mat_t, mat_th, min_freq)
        frontier    = compute_frontier(occupied_t, P)
        if not frontier:
            return np.zeros((0, 7)), np.zeros(0)
        X = np.array([extract_features(c, frontier[c], pred_posfreq,
                                       prev_posfreq, P) for c in frontier])
        y = np.array([1 if c in new_in_th else 0 for c in frontier])
        return X, y

    def fit(self, X, y):
        if len(X) == 0 or y.sum() == 0:
            return False
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.fitted = True
        return True

    def score(self, mat_t, pred_posfreq, prev_posfreq, P, top_k=200):
        if not self.fitted:
            return []
        occupied_t = compute_occupied(mat_t, top_k=top_k)
        frontier   = compute_frontier(occupied_t, P)
        if not frontier:
            return []
        candidates = list(frontier.keys())
        X = np.array([extract_features(c, frontier[c], pred_posfreq,
                                       prev_posfreq, P) for c in candidates])
        probs = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        return sorted(zip(candidates, probs.tolist()), key=lambda x: -x[1])

    def feature_importances(self):
        if not self.fitted:
            return {}
        return dict(zip(FEATURE_NAMES, self.model.coef_[0].tolist()))


# ── Evaluation of ranked candidates ──────────────────────────────────────────

def evaluate_ranking(
    ranked:    List[Tuple[Constellation, float]],
    new_in_th: Set[Constellation],
    top_ks:    List[int] = [10, 20, 50, 100],
) -> dict:
    """
    Evaluate a ranked list of candidates against actual new constellations.
    Metrics: precision@k, recall@k, average precision, random baseline.
    """
    if not ranked or not new_in_th:
        return {}
    labels = [1 if c in new_in_th else 0 for c, _ in ranked]
    results = {}
    for k in top_ks:
        if k > len(labels):
            continue
        top_k = labels[:k]
        results[f"precision@{k}"] = float(sum(top_k) / k)
        results[f"recall@{k}"]    = float(sum(top_k) / len(new_in_th))
    ap, n_pos = 0.0, 0
    for i, lbl in enumerate(labels):
        if lbl == 1:
            n_pos += 1
            ap += n_pos / (i + 1)
    results["AP"]             = float(ap / max(len(new_in_th), 1))
    results["random_baseline_P"] = float(len(new_in_th) / max(len(ranked), 1))
    return results
