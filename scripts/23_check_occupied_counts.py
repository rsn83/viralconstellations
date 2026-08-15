#!/usr/bin/env python
"""
23_check_occupied_counts.py

Ten-second check: are the values in `{month}_occupied.pkl` real per-month
sequence counts, and does --min_count in script 22 actually filter anything?

This matters because script 22's headline number -- ~47% of each month's
hypergraph consists of never-before-seen constellations -- is only meaningful
if singleton constellations (often sequencing artefacts) can be filtered out
and the number survives. If the filter is a no-op, that test never ran.

Usage
-----
  python scripts/23_check_occupied_counts.py
  python scripts/23_check_occupied_counts.py --months 2021-06 2022-06 2023-06
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(m):
    print(m, flush=True)


def constellations_of(occ: dict) -> dict:
    """EXACTLY the normalisation script 19 and script 22 use."""
    out = {}
    for c, v in occ.items():
        count = v if isinstance(v, (int, float)) else 1
        out[frozenset(c)] = count
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--months", nargs="*", default=None,
                    help="specific months to inspect; default = 5 spread across the range")
    args = ap.parse_args()

    graphs_dir = Path(args.graphs_dir)
    index_df = pd.read_csv(graphs_dir / "index.tsv", sep="\t")
    all_months = sorted(index_df["month"].tolist())

    if args.months:
        months = args.months
    else:
        idx = np.linspace(0, len(all_months) - 1, 5).astype(int)
        months = [all_months[i] for i in idx]

    log(f"dataset spans {all_months[0]} .. {all_months[-1]}  ({len(all_months)} months)")
    log(f"inspecting: {months}\n")

    # ---------------- 1. what is actually in the pickle ----------------
    log("=" * 74)
    log("1. RAW VALUE TYPES")
    log("=" * 74)
    m0 = months[len(months) // 2]
    with open(graphs_dir / f"{m0}_occupied.pkl", "rb") as fh:
        occ_raw = pickle.load(fh)

    log(f"  month {m0}: {len(occ_raw)} entries, container = {type(occ_raw).__name__}")
    types = Counter(type(v).__name__ for v in occ_raw.values())
    log(f"  value types: {dict(types)}")
    ktypes = Counter(type(k).__name__ for k in occ_raw.keys())
    log(f"  key types  : {dict(ktypes)}")

    numeric = all(isinstance(v, (int, float)) for v in occ_raw.values())
    log(f"  all values numeric? {numeric}")
    if not numeric:
        log("  ^^ NOT NUMERIC. constellations_of() collapses these to count=1 for")
        log("     every constellation, so --min_count > 1 empties the dict and")
        log("     --min_count in {0,1} is a no-op. The singleton test never ran.")
        ex = list(occ_raw.items())[:2]
        for k, v in ex:
            log(f"     example: key={type(k).__name__} value={repr(v)[:120]}")
    print()

    # ---------------- 2. count distribution ----------------
    log("=" * 74)
    log("2. COUNT DISTRIBUTION (after constellations_of normalisation)")
    log("=" * 74)
    log(f"  {'month':<10} {'n_sets':>8} {'min':>6} {'p25':>7} {'med':>7} {'p75':>8} "
        f"{'max':>9} {'%==1':>7}")
    for m in months:
        with open(graphs_dir / f"{m}_occupied.pkl", "rb") as fh:
            H = constellations_of(pickle.load(fh))
        v = np.array(list(H.values()), dtype=float)
        if v.size == 0:
            log(f"  {m:<10}  EMPTY")
            continue
        log(f"  {m:<10} {len(v):>8} {v.min():>6.0f} {np.percentile(v,25):>7.0f} "
            f"{np.median(v):>7.0f} {np.percentile(v,75):>8.0f} {v.max():>9.0f} "
            f"{(v==1).mean():>7.1%}")
    print()

    # ---------------- 3. does min_count filter? ----------------
    log("=" * 74)
    log("3. EFFECT OF --min_count  (this is the actual check)")
    log("=" * 74)
    log(f"  {'month':<10} {'raw':>8} {'>=1':>8} {'>=2':>8} {'>=3':>8} {'>=5':>8} "
        f"{'>=10':>8}")
    any_effect = False
    for m in months:
        with open(graphs_dir / f"{m}_occupied.pkl", "rb") as fh:
            H = constellations_of(pickle.load(fh))
        v = np.array(list(H.values()), dtype=float)
        cells = [int((v >= k).sum()) for k in (1, 2, 3, 5, 10)]
        if cells[0] != cells[2]:
            any_effect = True
        log(f"  {m:<10} {len(v):>8} " + " ".join(f"{c:>8}" for c in cells))
    print()

    log("=" * 74)
    log("VERDICT")
    log("=" * 74)
    if not numeric:
        log("  Values are NOT counts. --min_count cannot work as intended.")
        log("  -> The ~47% new-set fraction has NOT been tested against singleton")
        log("     noise. Find where occupied.pkl is written (likely script 05b)")
        log("     and store the real per-month sequence count per constellation.")
    elif not any_effect:
        log("  Values are numeric but --min_count 3 removes nothing, i.e. every")
        log("  constellation already has count >= 3. Check section 2: if '%==1' is")
        log("  0.0% everywhere, some upstream step already filtered singletons --")
        log("  in which case the 47% figure is ALREADY singleton-free and stands.")
    else:
        log("  --min_count works. Section 3 shows how many sets each threshold")
        log("  removes. Since script 22's numbers barely moved between min_count")
        log("  1 and 3, the ~47% new-set fraction is robust to singleton noise.")
    log("")
    log("  Either way, re-run script 22 with SEPARATE output paths so the runs")
    log("  stop overwriting each other:")
    log("    python scripts/22_set_dynamics_stats.py --min_count 1 \\")
    log("        --out outputs/22_sd_mc1.csv")
    log("    python scripts/22_set_dynamics_stats.py --min_count 3 \\")
    log("        --out outputs/22_sd_mc3.csv")
    log("    python scripts/22_set_dynamics_stats.py --min_count 3 --weight_by_count \\")
    log("        --out outputs/22_sd_mc3_wt.csv")


if __name__ == "__main__":
    main()
