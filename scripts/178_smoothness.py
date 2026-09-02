#!/usr/bin/env python3
"""
178 -- IS THE ATTACHMENT DISTRIBUTION SMOOTH IN THE BACKGROUND?

WHY THIS RUN
------------
Every estimator in 171-176 assumes, without testing it, that backgrounds
which are similar have similar attachment distributions. The Jaccard backoff
IS that assumption: it predicts p(D|S) by averaging over training
backgrounds with Jaccard >= tau to S.

If the assumption holds, generalization to unseen backgrounds is possible
and a sample-complexity bound follows from a Lipschitz-plus-covering-number
argument. If it fails, observing attachments to S says nothing about S', no
such bound exists, and the 8/8-month result in 171/173 is carried by
something other than similarity.

THE CONDITION
-------------
    TV( p(.|S), p(.|S') )  <=  L * d(S, S')

with d = Jaccard distance and TV the total variation distance between the
two attachment distributions. This script estimates whether such an L exists
and what it is.

THE NOISE FLOOR -- WITHOUT THIS THE PLOT IS UNINTERPRETABLE
-----------------------------------------------------------
Two backgrounds at distance 0 would still show TV > 0, because each
attachment distribution is estimated from finitely many observations.
Splitting a single background's attachment events into two halves and
measuring TV between them gives that floor. Any TV at distance d only counts
as structure if it rises above it.

WHAT WOULD FALSIFY SMOOTHNESS
    - TV flat in d (already at the ceiling everywhere): similar backgrounds
      are no more alike than dissimilar ones. Backoff is doing nothing and
      the 171 result comes from elsewhere.
    - TV at small d indistinguishable from the noise floor across the whole
      range: nothing is measurable at this sample size.

WHAT WOULD SUPPORT IT
    - TV rising with d, clearly above the noise floor at small d, with a
      bounded slope. The slope is L.

This measures ONE metric (Jaccard). A negative result rules out Jaccard, not
every possible background metric.

USAGE
    python scripts/178_smoothness.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --train-window 12 --out results/smoothness.json

GIT
    git add scripts/178_smoothness.py
    git commit -m "178: is p(D|S) Lipschitz in background distance"
    git push
"""

import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict

import numpy as np


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tv(p, q):
    """Total variation distance between two Counters treated as pmfs."""
    sp, sq = sum(p.values()), sum(q.values())
    if sp <= 0 or sq <= 0:
        return None
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) / sp - q.get(k, 0) / sq) for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--train-window", type=int, default=12)
    ap.add_argument("--min-events", type=int, default=8,
                    help="minimum attachment events for a background to be "
                         "included; below this the pmf is too noisy")
    ap.add_argument("--max-pairs", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)

    L = load_ladder(args.ladder)
    print("loading ...")
    monthly = L.load_events(args.events)
    pops = {m: L.population(monthly[m]) for m in sorted(monthly)}
    pops = {m: p for m, p in pops.items() if p}
    months = sorted(pops)

    tr_end = L.TRAIN_END[:7]
    train_months = [m for m in months if m <= tr_end]
    if args.train_window > 0:
        train_months = train_months[-args.train_window:]
    print(f"  train {len(train_months)}m")

    # attachment events per background, kept as a LIST so it can be split
    events = defaultdict(list)
    for a in range(len(train_months) - 1):
        pT, pN = pops.get(train_months[a], {}), pops.get(train_months[a + 1], {})
        if not pT or not pN:
            continue
        by_size = defaultdict(list)
        for S in pT:
            by_size[len(S)].append(S)
        for Sn in pN:
            for S in by_size.get(len(Sn) - 1, ()):
                if S < Sn:
                    events[S].append(next(iter(Sn - S)))

    bgs = [S for S, e in events.items() if len(e) >= args.min_events]
    print(f"  {len(events):,} backgrounds with attachments | "
          f"{len(bgs):,} with >= {args.min_events} events")
    if len(bgs) < 20:
        print("  TOO FEW BACKGROUNDS -- lower --min-events or widen window")
        return

    pmf = {S: Counter(events[S]) for S in bgs}
    supp = {S: len(pmf[S]) for S in bgs}
    print(f"  median distinct additions per background "
          f"{int(np.median(list(supp.values())))}")

    # ---- noise floor: split each background's own events in two -----------
    floor = []
    for S in bgs:
        e = events[S][:]
        rng.shuffle(e)
        h = len(e) // 2
        if h < 2:
            continue
        t = tv(Counter(e[:h]), Counter(e[h:]))
        if t is not None:
            floor.append(t)
    floor_m = float(np.mean(floor)) if floor else float("nan")
    print(f"\n  NOISE FLOOR (same background, split halves)  TV = "
          f"{floor_m:.3f}  n={len(floor)}")
    print("  Any TV at distance d only counts as structure above this.")

    # ---- pairs -------------------------------------------------------------
    pairs = []
    n = len(bgs)
    tried = 0
    while len(pairs) < args.max_pairs and tried < args.max_pairs * 6:
        tried += 1
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        A, B = bgs[i], bgs[j]
        d = 1.0 - len(A & B) / len(A | B)
        t = tv(pmf[A], pmf[B])
        if t is not None:
            pairs.append((d, t))
    if not pairs:
        print("  NO PAIRS")
        return

    D = np.array([p[0] for p in pairs])
    T = np.array([p[1] for p in pairs])

    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.01]
    print(f"\n  {len(pairs):,} background pairs")
    print(f"\n  {'jaccard dist':>16s} {'n':>8s} {'mean TV':>9s} "
          f"{'above floor':>12s}")
    print("  " + "-" * 50)
    binned = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (D >= a) & (D < b)
        if m.sum() < 5:
            continue
        mt = float(T[m].mean())
        binned.append({"lo": a, "hi": b, "n": int(m.sum()), "mean_tv": mt})
        print(f"  {a:6.2f}-{b:<9.2f} {int(m.sum()):8d} {mt:9.3f} "
              f"{mt - floor_m:12.3f}")

    # slope over the informative range
    if len(binned) >= 2:
        xs = np.array([(b["lo"] + b["hi"]) / 2 for b in binned])
        ys = np.array([b["mean_tv"] for b in binned])
        slope = float(np.polyfit(xs, ys, 1)[0])
        rng_tv = float(ys.max() - ys.min())
    else:
        slope = rng_tv = float("nan")

    print(f"\n  slope (empirical L)          {slope:.3f}")
    print(f"  TV range across distance     {rng_tv:.3f}")
    print(f"  TV at largest distance       {binned[-1]['mean_tv']:.3f}")

    print("\n  READING THIS")
    print("  TV rises with distance and is clearly above the floor at small")
    print("  d  ->  smoothness holds under Jaccard; slope is L; a covering-")
    print("         number bound is available in principle.")
    print("  TV flat, or at the ceiling everywhere  ->  Jaccard similarity")
    print("         does not track attachment similarity, and the backoff in")
    print("         171/173 is not working the way it is assumed to.")
    print("  TV near the floor everywhere  ->  sample size too small to say.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"train_window": args.train_window,
                       "n_backgrounds": len(bgs), "n_pairs": len(pairs),
                       "noise_floor_tv": floor_m, "slope": slope,
                       "tv_range": rng_tv, "binned": binned}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
