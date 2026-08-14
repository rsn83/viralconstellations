"""
Step 13: Rerun the extinction-vs-frequency check (originally
09_check_extinction_vs_frequency.py) using FULL-data occupied
constellation counts instead of the 10k-subsampled M_t matrices.

Why: the subsampling ratio printout from 05_build_full_data_graphs.py
showed some months drew as little as 1.2% of the real population. Low-
frequency constellations are far more likely to randomly vanish from a
1-2% draw regardless of true extinction dynamics, which could have
artificially inflated the original frequency-vs-extinction result. This
reruns the exact same statistical test on full data to check whether the
original finding (coefficient -1.545, p~1e-86) holds up.

Usage:
  python scripts/13_check_extinction_full_data.py
"""

import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viralconstellations.checks import (
    extinction_rate_series, extinction_by_frequency_bin,
    clustered_logistic_freq_effect, plot_extinction_series, plot_extinction_by_bin,
)


def log(msg): print(msg, flush=True)


def load_full_occupied(graphs_dir: Path, month: str) -> dict:
    """Load {frozenset(col_indices): count} for one month from the full-data pass."""
    with open(graphs_dir / f"{month}_occupied.pkl", "rb") as f:
        return pickle.load(f)


def main():
    graphs_dir = ROOT / "data" / "processed" / "full_data_graphs"
    out_dir = ROOT / "outputs" / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = graphs_dir / "index.tsv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found -- run scripts/05_build_full_data_graphs.py first."
        )

    import pandas as pd
    index_df = pd.read_csv(index_path, sep="\t")
    months = sorted(index_df["month"].tolist())
    log(f"Loading full-data occupied constellations for {len(months)} months "
        f"({months[0]} -> {months[-1]})")

    monthly_freqs = [load_full_occupied(graphs_dir, m) for m in months]
    for m, freqs in zip(months, monthly_freqs):
        log(f"  {m}: {len(freqs):,} distinct constellations (full data)")

    # min_count needs to scale with real sequence volume now -- the original
    # threshold of 5 was calibrated for a ~10k-sequence sample. On full data,
    # a count of 5 out of 800,000 real sequences (2022-01) is a far rarer
    # relative frequency than 5 out of 10,000 was. Using an absolute count
    # threshold on full data risks flooding "occupied" with noise-level
    # detections. Two thresholds are run for comparison.
    for min_count in [5, 20]:
        log("\n" + "=" * 70)
        log(f"FULL-DATA extinction check, min_count={min_count}")
        log("=" * 70)

        rate_df = extinction_rate_series(monthly_freqs, min_count=min_count)
        log(f"Mean extinction rate: {rate_df['rate'].mean():.4f} "
            f"(std {rate_df['rate'].std():.4f}) across {len(rate_df)} windows")
        plot_extinction_series(
            rate_df, str(out_dir / f"extinction_rate_series_fulldata_min{min_count}.png")
        )

        df_long = extinction_by_frequency_bin(monthly_freqs, min_count=min_count, n_bins=10)
        plot_extinction_by_bin(
            df_long, n_bins=10,
            out_path=str(out_dir / f"extinction_by_freq_bin_fulldata_min{min_count}.png")
        )

        clustered_logistic_freq_effect(df_long)

        rate_df.to_csv(out_dir / f"extinction_rate_fulldata_min{min_count}.csv")
        df_long.to_csv(out_dir / f"extinction_bins_fulldata_min{min_count}.csv", index=False)


if __name__ == "__main__":
    main()
