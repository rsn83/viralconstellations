"""
Check A: does raw g_t[i,j] co-occurrence predict appearance beyond frequency?
Check C: is the co-occurrence graph itself temporally predictable?

Both operate directly on your monthly (n_seq, P) matrices -- no PDB, no
graph neural network, no training. Add to src/viralconstellations/checks.py
or run standalone; designed to reuse the same interfaces as
09_check_extinction_vs_frequency.py / 10_check_cooccurrence_artifact.py.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

# reuse the same diagnostic functions already built and validated
from viralconstellations.checks import (
    correlation_and_vif, held_out_ap_comparison, held_out_ap_with_residual,
    frequency_matched_stratification, plot_stratification,
)


# ---------------------------------------------------------------------------
# Shared: build the co-occurrence graph g_t from a raw monthly matrix
# ---------------------------------------------------------------------------
def build_cooccurrence_graph(mat: np.ndarray, ref_value: int = 0) -> np.ndarray:
    """
    g_t[i, j] = number of sequences in this month's matrix where both
    position i and position j are mutated (mat != ref_value).
    Matches the encoding from 04_build_matrices_from_metadata.py.
    Diagonal g_t[i,i] = number of sequences where position i is mutated
    at all (its own marginal count) -- kept as a self-loop, consistent
    with how DNNTSP treats self-connections.
    """
    mutated = (mat != ref_value).astype(np.int32)  # (n_seq, P)
    g_t = mutated.T @ mutated                       # (P, P)
    return g_t


# ---------------------------------------------------------------------------
# Check A: raw g_t[i,j] as an 8th feature in the frontier scorer
# ---------------------------------------------------------------------------
def add_raw_cooccurrence_feature(
    df: pd.DataFrame,
    candidates: list,          # list of frozenset candidates, aligned to df rows
    frontier_infos: list,      # list of frontier.py's info dict (has "parents": [...]) per row
    g_t_by_window: dict,       # {window_id: g_t matrix (P,P)} for the CONTEXT month of each row
    window_col: str = "window",
) -> pd.DataFrame:
    """
    For each candidate row, compute raw co-occurrence support using ONLY
    the edges between the newly-added position(s) and each candidate's
    existing parent position(s) -- NOT averaged over all pairs in the
    candidate.

    Averaging over all pairs (an earlier version of this function) dilutes
    the signal: most pairs within a multi-mutation parent are already
    highly co-occurring by construction (they define an existing occupied
    lineage), which drowns out the one informative quantity -- does the
    NEW position connect well to what's already there. This isolates that.

    A candidate can have multiple parents (frontier.py's info["parents"]
    is a list) -- new_positions is computed the same way extract_features()
    does it, via set().union(*[candidate - p for p in parents]), for
    consistency with the rest of the pipeline.
    """
    raw_cooc_new_edges_mean = []
    raw_cooc_new_edges_max = []
    for idx, row in df.iterrows():
        w = row[window_col]
        g_t = g_t_by_window[w]
        candidate = candidates[idx]
        parents = frontier_infos[idx]["parents"]
        new_positions = list(set().union(*[candidate - p for p in parents]))
        parent_positions = list(set().union(*parents)) if parents else []

        if not new_positions or not parent_positions:
            raw_cooc_new_edges_mean.append(0.0)
            raw_cooc_new_edges_max.append(0.0)
            continue

        vals = [g_t[p, q] for p in new_positions for q in parent_positions
                if p != q and p < g_t.shape[0] and q < g_t.shape[0]]
        raw_cooc_new_edges_mean.append(float(np.mean(vals)) if vals else 0.0)
        raw_cooc_new_edges_max.append(float(np.max(vals)) if vals else 0.0)

    out = df.copy()
    out["raw_g_t_cooc_new_edges_mean"] = raw_cooc_new_edges_mean
    out["raw_g_t_cooc_new_edges_max"] = raw_cooc_new_edges_max
    return out


def run_check_A(df: pd.DataFrame, freq_cols: list[str]) -> None:
    """
    Run the same four diagnostics on the new-edges-only co-occurrence
    features (mean and max), instead of the diluted all-pairs average.
    `df` must already have 'raw_g_t_cooc_new_edges_mean' /
    '..._max' columns (see add_raw_cooccurrence_feature).
    """
    for col in ["raw_g_t_cooc_new_edges_mean", "raw_g_t_cooc_new_edges_max"]:
        print("=" * 70); print(f"CHECK A: {col}"); print("=" * 70)

        print("\n1. Correlation / VIF"); print("-" * 40)
        correlation_and_vif(df, freq_cols + [col])

        print("\n2. Held-out AP comparison"); print("-" * 40)
        held_out_ap_comparison(df, freq_cols, col)

        print("\n3. Residualized"); print("-" * 40)
        held_out_ap_with_residual(df, freq_cols, col)

        print("\n4. Frequency-matched stratification"); print("-" * 40)
        frequency_matched_stratification(df, freq_cols[0], col)
        print()


def run_check_A_alt_encodings(df: pd.DataFrame, freq_cols: list[str],
                               base_col: str = "raw_g_t_cooc_new_edges_mean") -> pd.DataFrame:
    """
    Test whether the raw stratification effect (large, consistent across
    bins) that the linear AP test missed is a functional-form problem:
    most candidates likely have base_col == 0 (the new position has never
    co-occurred with the parent at all), with a long tail of nonzero
    values. A linear model on the raw magnitude underfits a threshold
    effect. This tests two alternate encodings of the SAME underlying
    data against the same held-out AP comparison.
    """
    df = df.copy()

    n_zero = (df[base_col] == 0).sum()
    print(f"Zero-inflation check: {n_zero:,} / {len(df):,} rows "
          f"({100*n_zero/len(df):.1f}%) have {base_col} == 0\n")

    df["cooc_any_prior"] = (df[base_col] > 0).astype(float)
    df["cooc_log1p"] = np.log1p(df[base_col])

    for col in ["cooc_any_prior", "cooc_log1p"]:
        print("=" * 70); print(f"CHECK A (alt encoding): {col}"); print("=" * 70)

        print("\n1. Correlation / VIF"); print("-" * 40)
        correlation_and_vif(df, freq_cols + [col])

        print("\n2. Held-out AP comparison"); print("-" * 40)
        held_out_ap_comparison(df, freq_cols, col)

        print("\n3. Residualized"); print("-" * 40)
        held_out_ap_with_residual(df, freq_cols, col)

        print("\n4. Frequency-matched stratification"); print("-" * 40)
        frequency_matched_stratification(df, freq_cols[0], col)
        print()

    return df


# ---------------------------------------------------------------------------
# Check C: temporal persistence of the co-occurrence graph
# ---------------------------------------------------------------------------
def graph_persistence_check(
    monthly_mats: dict,   # {month_str: (n_seq, P) matrix}, sorted chronologically
    horizons: list[int] = (1, 3, 6),
    max_pairs_sampled: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    For each horizon h, correlate g_t[i,j] against g_{t+h}[i,j] across all
    (i,j) pairs and all valid windows t. Answers: does today's co-occurrence
    structure predict future co-occurrence structure at all?

    No model, no training -- plain Pearson correlation, pooled across
    positions and windows. Subsamples pairs for speed if the position
    count makes the full P*(P-1)/2 set too large.
    """
    from scipy.stats import pearsonr

    months = sorted(monthly_mats.keys())
    P = monthly_mats[months[0]].shape[1]
    rng = np.random.default_rng(seed)

    iu, ju = np.triu_indices(P, k=1)
    if len(iu) > max_pairs_sampled:
        sel = rng.choice(len(iu), size=max_pairs_sampled, replace=False)
        iu, ju = iu[sel], ju[sel]

    results = []
    for h in horizons:
        xs, ys = [], []
        for t_idx in range(len(months) - h):
            m_t, m_th = months[t_idx], months[t_idx + h]
            g_t = build_cooccurrence_graph(monthly_mats[m_t])
            g_th = build_cooccurrence_graph(monthly_mats[m_th])
            xs.append(g_t[iu, ju])
            ys.append(g_th[iu, ju])
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        # log1p to reduce dominance of a few huge-count pairs
        r, p = pearsonr(np.log1p(x_all), np.log1p(y_all))
        results.append({"horizon": h, "n_pairs": len(x_all), "pearson_r": r, "p_value": p})
        print(f"h={h}: Pearson r (log1p g_t vs log1p g_t+h) = {r:.4f}  (p={p:.4g}, n={len(x_all):,})")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Check C (residualized): is persistence real relational structure, or just
# marginal-frequency persistence wearing a pairwise costume?
# ---------------------------------------------------------------------------
def build_expected_cooccurrence(mat: np.ndarray, ref_value: int = 0) -> np.ndarray:
    """
    Expected g_t[i,j] under independence: if positions i and j were
    mutated independently at their observed marginal rates, how much
    co-occurrence would you expect by chance alone?

    expected[i,j] = f_i * f_j * n_seq, where f_i = P(position i mutated).
    This is the same independence-baseline logic as your original
    frequency-weighted baseline -- applied here to the graph itself.
    """
    n = mat.shape[0]
    mutated = (mat != ref_value).astype(np.float64)
    f = mutated.mean(axis=0)               # (P,) marginal mutation rate per position
    expected = np.outer(f, f) * n          # (P, P)
    return expected


def graph_persistence_check_residualized(
    monthly_mats: dict,
    horizons: list[int] = (1, 3, 6),
    max_pairs_sampled: int = 20000,
    seed: int = 0,
    epsilon: float = 1.0,
) -> pd.DataFrame:
    """
    Same as graph_persistence_check, but correlates the RESIDUAL
    (observed g_t - expected g_t under independence) across time,
    instead of raw g_t. If raw persistence was driven by marginal
    frequency alone, this residual correlation should collapse toward
    zero. If real pairwise/relational structure persists beyond what
    marginals explain, this should stay meaningfully positive.
    """
    from scipy.stats import pearsonr

    months = sorted(monthly_mats.keys())
    P = monthly_mats[months[0]].shape[1]
    rng = np.random.default_rng(seed)

    iu, ju = np.triu_indices(P, k=1)
    if len(iu) > max_pairs_sampled:
        sel = rng.choice(len(iu), size=max_pairs_sampled, replace=False)
        iu, ju = iu[sel], ju[sel]

    # precompute observed and expected graphs once per month
    observed, expected = {}, {}
    for m in months:
        observed[m] = build_cooccurrence_graph(monthly_mats[m])
        expected[m] = build_expected_cooccurrence(monthly_mats[m])

    results = []
    for h in horizons:
        xs_raw, ys_raw = [], []
        xs_resid, ys_resid = [], []
        for t_idx in range(len(months) - h):
            m_t, m_th = months[t_idx], months[t_idx + h]

            obs_t, exp_t = observed[m_t][iu, ju], expected[m_t][iu, ju]
            obs_th, exp_th = observed[m_th][iu, ju], expected[m_th][iu, ju]

            xs_raw.append(obs_t); ys_raw.append(obs_th)

            # residual: log ratio of observed to expected (a common way to
            # normalize count data by its null expectation; epsilon avoids
            # log(0) / division issues for zero-count pairs)
            resid_t = np.log((obs_t + epsilon) / (exp_t + epsilon))
            resid_th = np.log((obs_th + epsilon) / (exp_th + epsilon))
            xs_resid.append(resid_t); ys_resid.append(resid_th)

        x_raw, y_raw = np.concatenate(xs_raw), np.concatenate(ys_raw)
        x_res, y_res = np.concatenate(xs_resid), np.concatenate(ys_resid)

        r_raw, p_raw = pearsonr(np.log1p(x_raw), np.log1p(y_raw))
        r_res, p_res = pearsonr(x_res, y_res)

        print(f"h={h}:  raw r={r_raw:.4f} (p={p_raw:.4g})   "
              f"residualized r={r_res:.4f} (p={p_res:.4g})   n={len(x_raw):,}")

        results.append({
            "horizon": h, "n_pairs": len(x_raw),
            "raw_pearson_r": r_raw, "raw_p_value": p_raw,
            "residualized_pearson_r": r_res, "residualized_p_value": p_res,
        })

    result_df = pd.DataFrame(results)
    print("\nInterpretation:")
    for _, row in result_df.iterrows():
        drop = row["raw_pearson_r"] - row["residualized_pearson_r"]
        pct_drop = 100 * drop / row["raw_pearson_r"] if row["raw_pearson_r"] else 0
        print(f"  h={int(row['horizon'])}: residualizing dropped r by {drop:.4f} "
              f"({pct_drop:.1f}% of raw persistence explained by marginal frequency alone)")
    return result_df



if __name__ == "__main__":
    print("Running graph_persistence_check with SYNTHETIC data as a sanity check.\n")
    rng = np.random.default_rng(0)
    P = 30
    months = [f"2021-{m:02d}" for m in range(1, 13)]
    # synthetic data with real persistence: co-occurrence structure decays slowly
    base_mat = rng.integers(0, 2, size=(500, P))
    monthly_mats = {}
    mat = base_mat.copy()
    for m in months:
        # slowly perturb
        flip = rng.random(mat.shape) < 0.05
        mat = np.where(flip, 1 - mat, mat)
        monthly_mats[m] = mat.copy()

    print("--- Raw persistence ---")
    graph_persistence_check(monthly_mats, horizons=[1, 3, 6])

    print("\n--- Residualized persistence (controls for marginal frequency) ---")
    graph_persistence_check_residualized(monthly_mats, horizons=[1, 3, 6])
