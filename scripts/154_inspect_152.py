#!/usr/bin/env python3
"""
Pull per-month values out of 152's horizontal.json.

The pooled means hid a lot: j_var ranged 0.47-8.44 and co_var 0.83-7.73 across
months. This prints the per-month series so outlier months can be identified,
and flags whether they coincide with lineage turnover (when the core collapses
because no mutation clears theta).

USAGE
    python scripts/154_inspect_152.py --json results_152/horizontal.json
    python scripts/154_inspect_152.py --json results_152/horizontal.json --theta 0.5
"""

import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="results_152/horizontal.json")
    p.add_argument("--theta", default=None,
                   help="which theta to show; default: all present")
    p.add_argument("--top", type=int, default=8,
                   help="how many outlier months to list")
    a = p.parse_args()

    d = json.load(open(a.json))
    keys = [a.theta] if a.theta else sorted(d.keys(), key=float)

    for k in keys:
        recs = d.get(k)
        if not recs:
            print(f"theta={k}: no records")
            continue

        print(f"\n{'='*78}\ntheta={k}   ({len(recs)} months)\n{'='*78}")
        print(f"{'month':<10}{'core':>6}{'nonCore':>9}{'setSize':>9}"
              f"{'residSz':>9}{'j_tail':>9}{'j_var':>8}{'co_var':>8}")
        for r in recs:
            print(f"{r['month']:<10}{r['core_size']:>6}"
                  f"{r['non_core_size']:>9}{r['median_set_size']:>9.0f}"
                  f"{r['median_residual_size']:>9.1f}"
                  f"{r['j_tail_ratio']:>9.2f}{r['j_var_ratio']:>8.2f}"
                  f"{r['co_var_ratio']:>8.2f}")

        # Where does the core collapse? During turnover no mutation clears
        # theta, so core_size drops and both competing backbones fall into the
        # residual -- which is exactly where clustering should appear.
        cs = np.array([r["core_size"] for r in recs], dtype=float)
        jt = np.array([r["j_tail_ratio"] for r in recs], dtype=float)
        months = [r["month"] for r in recs]

        print(f"\n-- months with the largest j_tail_ratio (structure) --")
        for i in np.argsort(-jt)[:a.top]:
            print(f"   {months[i]}  j_tail {jt[i]:.2f}  core {int(cs[i])}")

        print(f"\n-- months with the smallest core (turnover candidates) --")
        for i in np.argsort(cs)[:a.top]:
            print(f"   {months[i]}  core {int(cs[i])}  j_tail {jt[i]:.2f}")

        ok = np.isfinite(jt) & np.isfinite(cs)
        if ok.sum() > 3:
            rho = np.corrcoef(cs[ok], jt[ok])[0, 1]
            print(f"\n-- correlation(core_size, j_tail_ratio) = {rho:+.2f}")
            print("   negative => structure appears when the core collapses,")
            print("   i.e. during lineage turnover, which would mean one")
            print("   global core is the wrong abstraction (several cores).")


if __name__ == "__main__":
    main()
