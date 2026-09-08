#!/usr/bin/env python3
"""
No model.  Just counts.

For each time window T+1, take the constellations that were NOT seen in any
window <= T, and ask how they relate to what already existed:

  * are they supersets of something seen before?  (pure additions)
  * if so, how many mutations were added?
  * if not, do they involve losses/reversions, or are they far from
    everything?
  * how frequent was the closest prior constellation?

This decides what a generative operator has to be able to do, and explains
whether the 30% recall came from parent cutoffs, multi-mutation jumps, or
non-superset emergence.
"""

import argparse
import csv
import datetime as dt
import sys
from collections import Counter, defaultdict

import numpy as np


def load_windows(path, window_days, stride_days, min_seqs):
    dates, cons = [], []
    with open(path) as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        i_d, i_c = h.index("date"), h.index("constellation")
        for row in r:
            if len(row) <= i_c:
                continue
            dates.append(dt.date.fromisoformat(row[i_d]))
            cons.append(frozenset(row[i_c].split(",")))

    order = np.argsort(dates)
    dates = [dates[i] for i in order]
    cons = [cons[i] for i in order]

    out, start, j = [], dates[0], 0
    while start <= dates[-1]:
        end = start + dt.timedelta(days=window_days)
        c = Counter()
        k = j
        while k < len(dates) and dates[k] < end:
            if dates[k] >= start:
                c[cons[k]] += 1
            k += 1
        if sum(c.values()) >= min_seqs:
            out.append(c)
        start += dt.timedelta(days=stride_days)
        while j < len(dates) and dates[j] < start:
            j += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--stride-days", type=int, default=30)
    ap.add_argument("--min-seqs", type=int, default=50)
    ap.add_argument("--max-prior", type=int, default=40000,
                    help="cap on prior constellations scanned per novel one")
    args = ap.parse_args()

    W = load_windows(args.data, args.window_days,
                     args.stride_days, args.min_seqs)
    print(f"{len(W)} windows", file=sys.stderr)

    seen = {}                    # constellation -> best frequency ever seen
    add_hist = Counter()         # how many mutations added
    kind = Counter()             # superset / mixed / far
    parent_rank = []             # frequency rank of the closest parent
    min_dist_nonsup = []

    for t, cur in enumerate(W):
        if t > 0:
            tot = sum(cur.values())
            # prior pool, most frequent first
            prior = sorted(seen.items(), key=lambda kv: -kv[1])[:args.max_prior]
            prior_sets = [p for p, _ in prior]

            for x, k in cur.items():
                if x in seen:
                    continue
                w = k / tot
                best_sup, best_sup_i = None, None
                best_any, best_any_i = None, None
                for i, p in enumerate(prior_sets):
                    d = len(x ^ p)
                    if best_any is None or d < best_any:
                        best_any, best_any_i = d, i
                    if p <= x:
                        add = len(x) - len(p)
                        if best_sup is None or add < best_sup:
                            best_sup, best_sup_i = add, i
                    if best_sup == 1:
                        break

                if best_sup is not None:
                    kind["superset"] += k
                    add_hist[min(best_sup, 10)] += k
                    parent_rank.append(best_sup_i)
                else:
                    lost = best_any is not None
                    kind["non_superset"] += k
                    if lost:
                        min_dist_nonsup.append(best_any)

        for x, k in cur.items():
            f = k / max(sum(cur.values()), 1)
            if f > seen.get(x, 0):
                seen[x] = f

        if t % 5 == 0:
            print(f"  window {t}: pool={len(seen):,}", file=sys.stderr)

    tot_nov = sum(kind.values())
    print("\n=== how novel constellations arise (weighted by sequences) ===")
    for k, v in kind.most_common():
        print(f"{k:<15} {v:>10,}  {100*v/tot_nov:5.1f}%")

    print("\n=== mutations added, when a prior subset exists ===")
    s = sum(add_hist.values())
    for n in sorted(add_hist):
        lbl = f"{n}" if n < 10 else "10+"
        print(f"  +{lbl:<3} {add_hist[n]:>10,}  {100*add_hist[n]/s:5.1f}%")

    if parent_rank:
        pr = np.array(parent_rank)
        print("\n=== frequency rank of closest prior parent ===")
        for q in (50, 75, 90, 95, 99):
            print(f"  p{q}: {np.percentile(pr, q):,.0f}")
        print(f"  fraction within top 300: {(pr < 300).mean():.3f}")
        print(f"  fraction within top 5000: {(pr < 5000).mean():.3f}")

    if min_dist_nonsup:
        md = np.array(min_dist_nonsup)
        print("\n=== non-supersets: symmetric-difference to nearest prior ===")
        for q in (10, 50, 90):
            print(f"  p{q}: {np.percentile(md, q):.0f}")


if __name__ == "__main__":
    main()
