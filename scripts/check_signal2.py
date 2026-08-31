#!/usr/bin/env python3
"""
check_signal2.py -- Does co-occurrence history predict new variants?

Fast version: only exact match check (O(1) per variant via set lookup).
Combo check removed -- too slow for 32k variants x 1000s window variants.

Usage:
  python scripts/check_signal2.py \
    --events data/processed/events_v3.tsv \
    --train-frac 0.7 \
    --horizons 1 2 3 6 \
    --window 3
"""
import argparse
import numpy as np
from collections import defaultdict

def load_events(path):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if ln == 0 and not parts[0][:4].isdigit():
                continue
            date, muts = parts[0].strip(), parts[1].strip()
            cnt = float(parts[2]) if len(parts) > 2 else 1.0
            s = frozenset(int(x) for x in muts.split(',') if x)
            if s:
                rows.append((date, s, cnt))
    rows.sort(key=lambda r: r[0])
    days = sorted({r[0] for r in rows})
    print(f"events {len(rows):,}  days {len(days)}  ({days[0]}..{days[-1]})")
    return rows, days

def monthly_variants(rows):
    by_month = defaultdict(lambda: defaultdict(float))
    for date, s, cnt in rows:
        ym = date[:7]
        by_month[ym][s] += cnt
    months = sorted(by_month.keys())
    var_mass = {}
    for ym in months:
        tot = sum(by_month[ym].values()) or 1.0
        var_mass[ym] = {s: v/tot for s, v in by_month[ym].items()}
    return var_mass, months

def run(a):
    rows, days = load_events(a.events)
    var_mass, months = monthly_variants(rows)

    n_train = int(len(months) * a.train_frac)
    train_months = months[:n_train]
    test_months  = months[n_train:]
    print(f"train {len(train_months)} months ({train_months[0]}..{train_months[-1]})")
    print(f"test  {len(test_months)} months ({test_months[0]}..{test_months[-1]})")
    print()

    for h in a.horizons:
        exact_num = exact_den = 0.0
        # mutation pair co-occurrence: for each new variant, check if its
        # most common mutation pair co-occurred in window
        pair_num = pair_den = 0.0
        n_obs = 0

        for t in test_months:
            t_idx = months.index(t)
            if t_idx < a.window or t_idx + h >= len(months):
                continue

            # build lookup: all variants seen in window (O(1) lookup)
            window_vars = set()
            # build mutation pair lookup from window
            window_pairs = set()
            for wm in months[t_idx - a.window: t_idx]:
                for v in var_mass[wm]:
                    window_vars.add(v)
                    mlist = sorted(v)
                    for i in range(len(mlist)):
                        for j in range(i+1, min(i+5, len(mlist))):
                            window_pairs.add((mlist[i], mlist[j]))

            current = set(var_mass.get(t, {}).keys())
            future_month = months[t_idx + h]
            new_vars = {v: w for v, w in var_mass.get(future_month, {}).items()
                        if v not in current}
            if not new_vars:
                continue

            for variant, weight in new_vars.items():
                n_obs += 1

                # exact match O(1)
                exact_num += weight * float(variant in window_vars)
                exact_den += weight

                # do any adjacent mutation pairs in this variant
                # appear as co-occurring pairs in the window?
                mlist = sorted(variant)
                found = 0; total_pairs = 0
                for i in range(len(mlist)):
                    for j in range(i+1, min(i+5, len(mlist))):
                        total_pairs += 1
                        if (mlist[i], mlist[j]) in window_pairs:
                            found += 1
                frac = found / total_pairs if total_pairs > 0 else 0.0
                pair_num += weight * frac
                pair_den += weight

        print(f"h={h} months  |  {n_obs} new-variant observations")
        print(f"  exact match in window:          "
              f"{exact_num/exact_den:.3f}  (was this exact set seen before?)")
        print(f"  adjacent pair co-occurrence:    "
              f"{pair_num/pair_den:.3f}  (do adjacent mutations co-occur in window?)")
        print(f"  (pair measure: 0=no pairs known  1=all pairs already co-occurred)")
        print()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events',     required=True)
    p.add_argument('--train-frac', type=float, default=0.7, dest='train_frac')
    p.add_argument('--horizons',   type=int, nargs='+', default=[1, 2, 3, 6])
    p.add_argument('--window',     type=int, default=3)
    run(p.parse_args())

if __name__ == '__main__':
    main()
