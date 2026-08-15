"""
Step 16: Paired significance test for the era-breakdown result.

Motivation (from the era breakdown in step 15):
  - low_volume_2023_2026: full_model beats every ablation/baseline at
    every horizon, with small std -- looks like a real, consistent effect.
  - high_volume_2020_2022: full_model does NOT clearly beat the simple
    baselines, but std is huge (0.19-0.24 AP) -- could be a real null
    result, or could be a handful of extreme-turnover windows (e.g.
    Omicron emergence) dragging the mean around while most windows
    behave normally.

Comparing means with n=32-34 windows and std this large does not tell
you which of these is true. This script runs a PAIRED test (Wilcoxon
signed-rank on per-window AP differences, matched by context month) for
full_model vs each ablation and each baseline, separately within each
era and each horizon. Paired because the same window's difficulty
affects both models identically -- this is the right test, not an
unpaired comparison of two AP distributions.

Also reports:
  - median difference (robust to outlier windows, unlike the mean)
  - which specific windows are the extreme cases, so you can go look at
    what was happening in the underlying data that month (variant
    turnover, sequencing volume, etc.)

Usage:
  python scripts/16_check_paired_significance.py
  (reads outputs/15_results_raw.pkl, written by the modified step 15)
"""

import sys, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def log(msg):
    print(msg, flush=True)


def era_of(month_str: str) -> str:
    year = int(month_str[:4])
    return "high_volume_2020_2022" if year <= 2022 else "low_volume_2023_2026"


def to_month_dict(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """(ctx_month, ap) list -> {ctx_month: ap}. Last write wins if
    duplicate months exist (shouldn't happen in a clean walk-forward
    run, but guards against it silently swallowing a bug)."""
    return {m: ap for m, ap in pairs}


def paired_diffs(a: dict[str, float], b: dict[str, float], era: str | None = None):
    """Return (months, diffs) for months present in both a and b,
    optionally restricted to one era. diffs = a[m] - b[m]."""
    months = sorted(set(a) & set(b))
    if era is not None:
        months = [m for m in months if era_of(m) == era]
    diffs = np.array([a[m] - b[m] for m in months])
    return months, diffs


def run_paired_test(name_a: str, a: dict[str, float], name_b: str, b: dict[str, float],
                     era: str, h: int):
    months, diffs = paired_diffs(a, b, era)
    n = len(diffs)
    if n < 3:
        log(f"    {name_a} vs {name_b}: n={n}, too few paired windows to test")
        return None

    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    n_pos = int((diffs > 0).sum())   # windows where a beat b
    n_neg = int((diffs < 0).sum())

    try:
        stat, p = wilcoxon(diffs)
    except ValueError:
        # all-zero differences, or too few non-zero diffs
        stat, p = float("nan"), float("nan")

    sig_flag = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
    log(f"    {name_a:<14} vs {name_b:<20} mean_diff={mean_diff:+.4f}  "
        f"median_diff={median_diff:+.4f}  wins/losses={n_pos}/{n_neg}  "
        f"(n={n})  p={p:.4f} {sig_flag}")

    # flag the worst outlier windows for this comparison -- these are
    # the ones worth going and looking at in the underlying data.
    order = np.argsort(diffs)
    worst = [(months[i], float(diffs[i])) for i in order[:3]]
    best = [(months[i], float(diffs[i])) for i in order[-3:][::-1]]
    return {
        "name_a": name_a, "name_b": name_b, "era": era, "h": h,
        "n": n, "mean_diff": mean_diff, "median_diff": median_diff,
        "n_pos": n_pos, "n_neg": n_neg, "wilcoxon_stat": stat, "p": p,
        "worst_for_a": worst, "best_for_a": best,
    }


def main():
    raw_path = ROOT / "outputs" / "15_results_raw.pkl"
    if not raw_path.exists():
        log(f"ERROR: {raw_path} not found. Re-run the (patched) step 15 first "
            f"to generate it -- no retraining needed beyond that one run.")
        sys.exit(1)

    with open(raw_path, "rb") as fh:
        raw = pickle.load(fh)
    results = raw["results"]              # {model_name: {h: [(month, ap), ...]}}
    baseline_results = raw["baseline_results"]
    horizons = raw["horizons"]

    all_models = {**baseline_results, **results}
    full = {h: to_month_dict(results["full_model"][h]) for h in horizons}

    comparisons = ["no_gnn", "no_rnn", "no_edge_history", "frequency", "naive_persistence"]

    all_rows = []
    for h in horizons:
        log("\n" + "=" * 78)
        log(f"HORIZON h={h}  (full_model vs each competitor, paired by context month)")
        log("=" * 78)
        for era in ["high_volume_2020_2022", "low_volume_2023_2026"]:
            log(f"\n  [{era}]")
            for other in comparisons:
                other_dict = to_month_dict(all_models[other][h])
                row = run_paired_test("full_model", full[h], other, other_dict, era, h)
                if row is not None:
                    all_rows.append(row)

    log("\n" + "=" * 78)
    log("SUMMARY TABLE")
    log("=" * 78)
    df = pd.DataFrame(all_rows)[["h", "era", "name_b", "n", "mean_diff", "median_diff",
                                  "n_pos", "n_neg", "p"]]
    log(df.to_string(index=False))

    out_csv = ROOT / "outputs" / "16_paired_significance.csv"
    df.to_csv(out_csv, index=False)
    log(f"\nWrote {out_csv}")

    # Call out the worst-case windows for the highest-variance comparison
    # (full_model vs frequency, high-volume era, h=1) as a starting point
    # for the "what's happening in those specific months" check.
    log("\n" + "=" * 78)
    log("WORST WINDOWS FOR full_model vs frequency, high_volume era, h=1")
    log("(these are the months to go inspect for turnover / data quality)")
    log("=" * 78)
    target = [r for r in all_rows if r["name_b"] == "frequency"
              and r["era"] == "high_volume_2020_2022" and r["h"] == 1]
    if target:
        for m, d in target[0]["worst_for_a"]:
            log(f"  {m}: full_model - frequency = {d:+.4f}")


if __name__ == "__main__":
    main()
