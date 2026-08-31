#!/usr/bin/env python3
"""
check_signal2.py -- Does co-occurrence history predict new variants?

For each new variant at T+h, check whether its specific mutation combination
was already forming in the W months before T -- even at low frequency.

Three measures:
  1. exact_seen: variant appeared at least once in window before T
  2. near_seen:  variant is within radius-1 of something in window
  3. combo_seen: at least half of its mutations co-occurred in some
                 single variant in the window

Result interpretation:
  High values --> combinations were already forming, signal exists
  Low values  --> combinations are genuinely novel, hard to predict

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
        exact_num = near_num = combo_num = 0.0
        exact_den = near_den = combo_den = 0.0
        n_obs = 0

        for t in test_months:
            t_idx = months.index(t)
            if t_idx < a.window or t_idx + h >= len(months):
                continue

            # all variants seen in window before T
            window_months = months[t_idx - a.window: t_idx]
            window_vars = set()
            for wm in window_months:
                window_vars |= set(var_mass[wm].keys())

            # variants present at T itself
            current = set(var_mass.get(t, {}).keys())

            # future new variants
            future_month = months[t_idx + h]
            future_vars  = var_mass.get(future_month, {})
            new_vars = {v: w for v, w in future_vars.items() if v not in current}
            if not new_vars:
                continue

            for variant, weight in new_vars.items():
                n_obs += 1
                w = weight

                # 1. exact: did this exact set appear in the window?
                exact = float(variant in window_vars)
                exact_num += w * exact
                exact_den += w

                # 2. skipped (slow) -- use script 159 for radius-1

                # 3. combo: do at least half of variant's mutations
                #    co-occur in some single window variant?
                best_overlap = 0.0
                for wv in window_vars:
                    overlap = len(variant & wv) / len(variant)
                    if overlap > best_overlap:
                        best_overlap = overlap
                combo = float(best_overlap >= 0.5)
                combo_num += w * combo
                combo_den += w

        print(f"h={h} months  |  {n_obs} new-variant observations")
        print(f"  exact match in window:        "
              f"{exact_num/exact_den:.3f}  (was this exact set seen before?)")
        print(f"  >50% mutations co-occurred:   "
              f"{combo_num/combo_den:.3f}  (partial combination existed?)")
        print()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events',     required=True)
    p.add_argument('--train-frac', type=float, default=0.7, dest='train_frac')
    p.add_argument('--horizons',   type=int, nargs='+', default=[1, 2, 3, 6])
    p.add_argument('--window',     type=int, default=3,
                   help='months of history to measure co-occurrence over')
    run(p.parse_args())

if __name__ == '__main__':
    main()
