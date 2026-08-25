#!/usr/bin/env python3
"""
107_set_distance_by_horizon.py

Model-free. Asks whether the targets get further away as the horizon grows.

The origin decomposition -- 55% one mutation from a set present this month,
15% returning, 7% two steps, 20% unreachable -- was only ever computed at
h = 1. Every horizon result since has been a log-likelihood, which mixes the
difficulty of the target with the model's ability to learn a rate.

This separates them. For each test month and each horizon h, take the sets that
are new at the test month and were not present h months earlier, and measure how
far each sits from the sets that WERE present then.

  distance(S, month t) = min over sets P present at t of |S \\ P|
                         i.e. the fewest mutations that must be ADDED to some
                         set already circulating to produce S

Reported per test month, weighted two ways:
  by distinct set   -- how many new set-types are reachable
  by sequences      -- how much of the month those sets account for

WHY PER TEST MONTH AND NOT POOLED
  The horizon sweep gave BA.5 decaying (+4.07, +1.21, -0.31) and BQ.1.1 holding
  (+3.97, +4.20, +2.05). Pooling would average over exactly the difference we
  are trying to see.

  If BA.5's targets at h=3 are far from the training window while BQ.1.1's are
  not, that is the mechanism, measured without fitting anything. If the distance
  distributions look the same, the difference is in the RATE being learnable,
  not in the targets being further away -- which points somewhere else entirely.

Usage:
  python 107_set_distance_by_horizon.py \
      --data-dir data/processed/full_data_graphs_withdel \
      --tests 2022-06,2022-11,2024-01 --horizons 1,2,3,6
"""
import argparse, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np


def ym_add(ym, k):
    y, m = map(int, ym.split("-")); m += k
    y += (m - 1) // 12; m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def load_month(data_dir, ym):
    p = Path(data_dir) / f"{ym}_occupied.pkl"
    if not p.exists(): return None
    obj = pickle.load(open(p, "rb"))
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and isinstance(vals[0], (int, np.integer)):
            return [(frozenset(k), int(v)) for k, v in obj.items()]
        return [(frozenset(v), 1) for v in vals]
    items = list(obj)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:
        return [(frozenset(s), int(c)) for s, c in items]
    return [(frozenset(s), 1) for s in items]


from itertools import combinations


def min_additions(S, pool, cap=3):
    """Fewest mutations that must be ADDED to some set in `pool` to reach S.

    Exact for distances 0, 1 and 2 by enumerating the subsets of S obtained by
    removing that many elements and testing membership -- O(|S|^2) hash lookups,
    independent of how large the pool is. Anything further is reported as cap+.
    Enumerating distance 3 would be C(|S|,3) per set, which at |S| ~ 40 and
    25,000 new sets is not worth the time for a bucket we only need as a
    remainder.
    """
    if S in pool: return 0
    els = tuple(S)
    for n in els:
        if S - {n} in pool: return 1
    if cap >= 2 and len(els) <= 60:
        for a, b in combinations(els, 2):
            if S - {a, b} in pool: return 2
    return cap + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--horizons", default="1,2,3,6")
    ap.add_argument("--cap", type=int, default=4,
                    help="distances above this are reported as unreachable")
    args = ap.parse_args()

    tests = [t.strip() for t in args.tests.split(",")]
    hs = [int(h) for h in args.horizons.split(",")]

    for te in tests:
        rec_te = load_month(args.data_dir, te)
        if rec_te is None:
            print(f"{te}: no data, skipped"); continue
        te_sets = {s: c for s, c in rec_te}
        te_tot = sum(te_sets.values())

        print("\n" + "=" * 78)
        print(f"TEST MONTH {te}    {len(te_sets):,} distinct sets, "
              f"{te_tot:,} sequences")
        print("=" * 78)
        res = {}
        for h in hs:
            src = ym_add(te, -h)
            rec_src = load_month(args.data_dir, src)
            if rec_src is None: continue
            pool = {s_ for s_, _ in rec_src}
            new = [(s_, c) for s_, c in te_sets.items() if s_ not in pool]
            b = defaultdict(int); bs = defaultdict(int)
            for s_, c in new:
                d = min_additions(s_, pool, cap=args.cap)
                b[d] += 1; bs[d] += c
            res[h] = (src, len(new), sum(c for _, c in new), b, bs)
            print(f"    {src} done", flush=True)

        print(f"\n  BY DISTINCT SET  -- how many new set-types are reachable")
        print(f"  {'h':>3}{'origin':>10}{'new sets':>11}{'1 mut':>9}{'2 mut':>9}"
              f"{'3+':>9}")
        for h in hs:
            if h not in res: continue
            src, n_new, _, b, _ = res[h]
            t = max(n_new, 1)
            print(f"  {h:>3}{src:>10}{n_new:>11,}"
                  f"{b.get(1,0)/t:>9.1%}{b.get(2,0)/t:>9.1%}"
                  f"{b.get(args.cap+1,0)/t:>9.1%}")

        print(f"\n  BY SEQUENCE  -- how much of the month those sets account for")
        print(f"  {'h':>3}{'origin':>10}{'new share':>12}{'1 mut':>9}{'2 mut':>9}"
              f"{'3+':>9}")
        for h in hs:
            if h not in res: continue
            src, _, seq_new, _, bs = res[h]
            t = max(seq_new, 1)
            print(f"  {h:>3}{src:>10}{seq_new/te_tot:>12.1%}"
                  f"{bs.get(1,0)/t:>9.1%}{bs.get(2,0)/t:>9.1%}"
                  f"{bs.get(args.cap+1,0)/t:>9.1%}")

    print("""

HOW TO READ
  'distance' is the fewest mutations that must be added to a set already
  circulating h months earlier to produce the new set. Distance 1 means the new
  set is one mutation from something already there; 4+ means it is not
  reachable by small edits from anything in that month.

  '3+' is everything not reachable by adding one or two mutations.
  If the distribution shifts right as h grows, targets genuinely get further
  away and the horizon decay has a data-level cause -- no model could fix it.

  If it barely shifts, the targets are equally close at every horizon and the
  decay is about the RATE being unlearnable from an earlier window, not about
  the targets. That points at the fitting, not the data.

  Compare 2022-06 with 2022-11. Drift decayed on the first (+4.07, +1.21,
  -0.31) and held on the second (+3.97, +4.20, +2.05). If their distance
  profiles differ in the same direction, that is the mechanism.
""")


if __name__ == "__main__":
    main()
