"""
Step 9: Check 1 -- is constellation extinction from O_t memoryless
(i.i.d. thinning, what an ADD-THIN-style mechanism assumes) or
frequency/fitness-driven?

Reuses your exact matrix encoding from 04_build_matrices_from_metadata.py:
  mat[i, j] == 0        -> reference residue at position j
  mat[i, j] in 1..20    -> mutated (residue index) at position j
A "constellation" = frozenset of mutated position indices for a sequence
(matches how candidate_to_sequence / frontier.py treat candidates: the
identity is the set of mutated sites, not the specific residues).

Usage:
  python scripts/09_check_extinction_vs_frequency.py --config configs/colab_2022_test.yaml
"""

import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--min_count", type=int, default=5,
                    help="Minimum sequence count for a constellation to count as 'occupied'")
parser.add_argument("--n_freq_bins", type=int, default=10)
parser.add_argument("--out_dir", default="outputs/checks")
args = parser.parse_args()

import yaml
import numpy as np
from viralconstellations.checks import (
    extinction_rate_series, extinction_by_frequency_bin,
    clustered_logistic_freq_effect, plot_extinction_series, plot_extinction_by_bin,
)


def log(msg): print(msg, flush=True)


def constellations_from_matrix(mat: np.ndarray, ref_value: int = 0) -> dict:
    """
    Convert a monthly (n_seq, P) int8 matrix into {constellation: count},
    where constellation is a frozenset of mutated position (column) indices.
    Matches the encoding written by 04_build_matrices_from_metadata.py.
    """
    counts: dict[frozenset, int] = {}
    mutated = mat != ref_value  # (n_seq, P) boolean
    for row in mutated:
        sites = frozenset(np.nonzero(row)[0].tolist())
        if len(sites) == 0:
            continue
        counts[sites] = counts.get(sites, 0) + 1
    return counts


def main():
    cfg = yaml.safe_load(open(ROOT / args.config))
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_months = sorted(
        p.stem for p in matrix_dir.glob("*.npy") if "_posfreq" not in p.stem
    )
    log(f"Available months: {len(all_months)}  ({all_months[0]} -> {all_months[-1]})")

    log(f"Building per-month constellation counts (min_count={args.min_count} "
        f"applied downstream, this pass keeps raw counts)...")
    monthly_freqs = []
    for m in all_months:
        mat = np.load(matrix_dir / f"{m}.npy")
        monthly_freqs.append(constellations_from_matrix(mat))
        log(f"  {m}: {mat.shape[0]:,} sequences, "
            f"{len(monthly_freqs[-1]):,} distinct constellations")

    log("\n" + "=" * 70)
    log("Extinction rate over time")
    log("=" * 70)
    rate_df = extinction_rate_series(monthly_freqs, min_count=args.min_count)
    log(rate_df.to_string())
    plot_extinction_series(rate_df, str(out_dir / "extinction_rate_series.png"))

    log("\n" + "=" * 70)
    log("Extinction rate vs. parent frequency")
    log("=" * 70)
    df_long = extinction_by_frequency_bin(
        monthly_freqs, min_count=args.min_count, n_bins=args.n_freq_bins
    )
    plot_extinction_by_bin(
        df_long, n_bins=args.n_freq_bins, out_path=str(out_dir / "extinction_by_freq_bin.png")
    )

    log("\n" + "=" * 70)
    log("Clustered logistic regression: P(extinct) ~ log(freq_t)")
    log("=" * 70)
    clustered_logistic_freq_effect(df_long)

    summary_path = out_dir / "check1_summary.json"
    summary_path.write_text(json.dumps({
        "min_count": args.min_count,
        "n_months": len(all_months),
        "mean_extinction_rate": float(rate_df["rate"].mean()),
        "std_extinction_rate": float(rate_df["rate"].std()),
    }, indent=2))
    log(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()