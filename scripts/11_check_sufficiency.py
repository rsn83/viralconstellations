"""
Step 11 — Sufficient-statistic check for constellation expansion forecasting.

Claim being tested:
    H: Constellation frequency + short-horizon growth rate (a joint-structure
       representation) predicts future expansion at least as well as a
       position-marginal representation (mean per-position alt-allele
       frequency across the constellation's mutated positions), which
       discards co-occurrence structure.

Reads directly from this project's existing pipeline outputs:
    data/processed/mutation_matrices/YYYY-MM.npy          (n_seq, P) int8
    data/processed/mutation_matrices/YYYY-MM_posfreq.npy  (P, 21) float32
    data/processed/mutation_matrices/index.tsv
    data/vocab/position_vocab.tsv

No new data collection or parsing — this is a downstream analysis step
that consumes what 04_build_matrices_from_metadata.py already produced.

Output:
    data/processed/sufficiency_check/constellation_features.tsv
    Printed AUC comparison across three feature sets.

Usage:
    python scripts/11_check_sufficiency.py
"""

import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Load monthly matrices, derive constellation frequencies
# ---------------------------------------------------------------------------

def load_monthly_matrices(matrix_dir: Path, index_df: pd.DataFrame):
    """
    For each month, load the (n_seq, P) matrix and collapse identical rows
    into constellations. A constellation is the full mutation-pattern tuple
    across all P variable positions (0 = reference at that position).

    Returns a long DataFrame: time_bin, constellation_id, positions (tuple
    of nonzero (col, residue) pairs), count, n_total (sequences that month).
    """
    rows = []
    for _, r in index_df.iterrows():
        month = r["month"]
        mat_path = matrix_dir / f"{month}.npy"
        if not mat_path.exists():
            print(f"  [skip] missing {mat_path}")
            continue
        mat = np.load(mat_path)
        n_total = mat.shape[0]

        # collapse identical rows -> constellation counts
        # constellation id = tuple of the full row (hashable, comparable
        # across months since the position vocab is shared across the run)
        uniq, counts = np.unique(mat, axis=0, return_counts=True)
        for pattern, cnt in zip(uniq, counts):
            nonzero_cols = np.nonzero(pattern)[0]
            positions = tuple((int(c), int(pattern[c])) for c in nonzero_cols)
            if not positions:
                continue  # skip pure-reference rows; not a mutation constellation
            rows.append({
                "time_bin": month,
                "constellation_id": positions,   # hashable, identifies the pattern
                "positions": positions,
                "count": int(cnt),
                "n_total": n_total,
            })
    df = pd.DataFrame(rows)
    df["freq"] = df["count"] / df["n_total"]
    return df


# ---------------------------------------------------------------------------
# 2. Feature set A: constellation frequency + short-horizon growth rate
# ---------------------------------------------------------------------------

def build_freq_growth_features(df: pd.DataFrame):
    df = df.sort_values(["constellation_id", "time_bin"]).copy()
    # time_bin is a sortable "YYYY-MM" string; groupby preserves chronological
    # order given the prior sort, so shift(1)/(-1) below are valid provided
    # the underlying months are contiguous. If your data has gaps, add an
    # explicit month-index column and sort/shift on that instead.
    df["prev_freq"] = df.groupby("constellation_id")["freq"].shift(1)
    eps = 1e-6
    df["growth_rate"] = np.log((df["freq"] + eps) / (df["prev_freq"] + eps))
    df["growth_rate"] = df["growth_rate"].fillna(0.0)
    return df[["constellation_id", "time_bin", "freq", "growth_rate", "positions"]]


# ---------------------------------------------------------------------------
# 3. Feature set B: position-marginal alt-allele frequency (pipeline's own
#    posfreq.npy, not recomputed) — mean over the constellation's positions
# ---------------------------------------------------------------------------

def build_position_marginal_features(df: pd.DataFrame, matrix_dir: Path):
    posfreq_cache = {}

    def get_posfreq(month):
        if month not in posfreq_cache:
            path = matrix_dir / f"{month}_posfreq.npy"
            posfreq_cache[month] = np.load(path) if path.exists() else None
        return posfreq_cache[month]

    records = []
    for _, row in df.iterrows():
        posfreq = get_posfreq(row["time_bin"])
        if posfreq is None:
            marginal_mean = np.nan
        else:
            vals = []
            for col, residue in row["positions"]:
                # frequency of this exact residue at this position, that month
                vals.append(float(posfreq[col, residue]))
            marginal_mean = float(np.mean(vals)) if vals else 0.0
        records.append({
            "constellation_id": row["constellation_id"],
            "time_bin": row["time_bin"],
            "pos_marginal_mean": marginal_mean,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Labels: does this constellation expand next month?
# ---------------------------------------------------------------------------

def build_labels(freq_growth_df: pd.DataFrame, threshold: float = 0.15):
    df = freq_growth_df.sort_values(["constellation_id", "time_bin"]).copy()
    df["next_growth_rate"] = df.groupby("constellation_id")["growth_rate"].shift(-1)
    df["label_expands"] = (df["next_growth_rate"] > threshold).astype(int)
    return df.dropna(subset=["next_growth_rate"])


# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------

def evaluate_feature_set(X, y, label):
    if y.nunique() < 2 or len(X) < 20:
        print(f"{label}: insufficient data/label diversity to evaluate "
              f"(n={len(X)}, unique_labels={y.nunique()})")
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"{label}: AUC = {auc:.3f}  (n_train={len(X_train)}, n_test={len(X_test)})")
    return auc


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    out_dir = ROOT / "data/processed/sufficiency_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    MIN_COUNT = 10  # drop constellation-month rows below this raw count
    MIN_MONTHS_OBSERVED = 2  # constellation must appear in >=N months to
                              # have a defined growth rate at all

    index_df = pd.read_csv(matrix_dir / "index.tsv", sep="\t")
    print(f"Loaded index: {len(index_df)} months")

    print("\nCollapsing monthly matrices into constellation frequencies...")
    raw = load_monthly_matrices(matrix_dir, index_df)
    print(f"  {len(raw)} (constellation, month) rows across "
          f"{raw['constellation_id'].nunique()} distinct constellations "
          f"(pre-filter)")

    # --- Diagnostics: how sparse is this, really? ---
    appearances = raw.groupby("constellation_id").size()
    print("\nConstellation appearance-count distribution (months observed in):")
    print(appearances.value_counts().sort_index().head(10).to_string())
    print(f"  Constellations appearing in only 1 month: "
          f"{(appearances == 1).sum()} / {len(appearances)} "
          f"({(appearances == 1).mean():.1%})")

    print("\nRaw count distribution (sequences per constellation-month):")
    print(raw["count"].describe())

    # --- Filter: drop low-count rows (noise-dominated) and constellations
    #     that never appear in enough months to have a real trend ---
    before = len(raw)
    raw = raw[raw["count"] >= MIN_COUNT].copy()
    persistent = appearances[appearances >= MIN_MONTHS_OBSERVED].index
    raw = raw[raw["constellation_id"].isin(persistent)].copy()
    print(f"\nAfter filtering (count>={MIN_COUNT}, "
          f"months_observed>={MIN_MONTHS_OBSERVED}): "
          f"{len(raw)} rows (dropped {before - len(raw)}), "
          f"{raw['constellation_id'].nunique()} distinct constellations")

    freq_growth = build_freq_growth_features(raw)
    pos_marginal = build_position_marginal_features(raw, matrix_dir)
    labels = build_labels(freq_growth)

    merged = labels.merge(pos_marginal, on=["constellation_id", "time_bin"])
    merged.drop(columns=["positions"]).to_csv(
        out_dir / "constellation_features.tsv", sep="\t", index=False
    )
    print(f"\nWrote: {out_dir / 'constellation_features.tsv'}")

    print(f"\nRows available for evaluation: {len(merged)}")
    print("Label balance (label_expands):")
    print(merged["label_expands"].value_counts().to_string())

    y = merged["label_expands"]
    X_a = merged[["freq", "growth_rate"]]
    X_b = merged[["pos_marginal_mean"]]
    X_ab = merged[["freq", "growth_rate", "pos_marginal_mean"]]

    print("\n=== Sufficiency comparison ===")
    evaluate_feature_set(X_a, y, "A: freq + growth_rate (constellation-level, joint structure)")
    evaluate_feature_set(X_b, y, "B: position-marginal only (structure discarded)")
    evaluate_feature_set(X_ab, y, "A+B: combined")

    print(
        "\nInterpretation: if A >= A+B (within noise), freq+growth is close "
        "to a sufficient statistic for this task — position-marginal info "
        "adds little beyond it. If B is close to A, joint constellation "
        "structure may not be adding predictive value over single-position "
        "marginals, and the representation can likely be simplified."
    )


if __name__ == "__main__":
    main()
