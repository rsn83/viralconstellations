#!/usr/bin/env python3
"""
check_signal.py -- Does mutation acceleration predict new variants?

Result: weighted fraction of new-variant mutations that were accelerating
  > 0.5  signal exists
  ~ 0.5  random
  < 0.5  anti-signal

Usage:
  python scripts/check_signal.py \
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

def monthly_mass(rows):
    by_month = defaultdict(lambda: defaultdict(float))
    for date, s, cnt in rows:
        ym = date[:7]
        for m in s:
            by_month[ym][m] += cnt
    months = sorted(by_month.keys())
    freq = {}
    for ym in months:
        tot = sum(by_month[ym].values()) or 1.0
        freq[ym] = {m: v/tot for m, v in by_month[ym].items()}
    return freq, months

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
    freq, months = monthly_mass(rows)
    var_mass, _  = monthly_variants(rows)

    n_train = int(len(months) * a.train_frac)
    train_months = months[:n_train]
    test_months  = months[n_train:]
    print(f"train {len(train_months)} months ({train_months[0]}..{train_months[-1]})")
    print(f"test  {len(test_months)} months ({test_months[0]}..{test_months[-1]})")
    print()

    for h in a.horizons:
        results = []
        for t in test_months:
            t_idx = months.index(t)
            if t_idx < a.window or t_idx + h >= len(months):
                continue
            window_months = months[t_idx - a.window: t_idx]
            all_muts = set()
            for wm in window_months:
                all_muts |= set(freq[wm].keys())
            slopes = {}
            for m in all_muts:
                flist = [freq[wm].get(m, 0.0) for wm in window_months]
                if max(flist) > 0:
                    slopes[m] = float(np.polyfit(range(len(flist)), flist, 1)[0])
            avg_slope = float(np.mean(list(slopes.values()))) if slopes else 0.0
            current = set(var_mass.get(t, {}).keys())
            future_month = months[t_idx + h]
            future_vars  = var_mass.get(future_month, {})
            new_vars = {v: w for v, w in future_vars.items() if v not in current}
            if not new_vars:
                continue
            for variant, weight in new_vars.items():
                muts = list(variant)
                mut_slopes   = [slopes.get(m, 0.0) for m in muts]
                frac_above   = float(np.mean([s > avg_slope for s in mut_slopes]))
                results.append({'t': t, 'future': future_month,
                                'weight': weight, 'frac_accelerating': frac_above,
                                'n_muts': len(muts)})

        if not results:
            print(f"h={h}: no test results"); continue

        weights    = np.array([r['weight'] for r in results])
        fracs      = np.array([r['frac_accelerating'] for r in results])
        weighted   = float(np.average(fracs, weights=weights))
        unweighted = float(np.mean(fracs))
        top = sorted(results, key=lambda r: -r['weight'])[:5]

        print(f"h={h} months  |  {len(results)} new-variant observations")
        print(f"  weighted frac accelerating:   {weighted:.3f}"
              f"  (0.5=random  >0.5=signal  <0.5=anti-signal)")
        print(f"  unweighted frac accelerating: {unweighted:.3f}")
        print(f"  top 5 new variants by mass:")
        for r in top:
            print(f"    {r['t']}->+{h}m  mass={r['weight']:.4f}"
                  f"  frac_accel={r['frac_accelerating']:.2f}"
                  f"  n_muts={r['n_muts']}")
        print()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events',     required=True)
    p.add_argument('--train-frac', type=float, default=0.7, dest='train_frac')
    p.add_argument('--horizons',   type=int, nargs='+', default=[1, 2, 3, 6])
    p.add_argument('--window',     type=int, default=3,
                   help='months of history to compute slope over')
    run(p.parse_args())

if __name__ == '__main__':
    main()
