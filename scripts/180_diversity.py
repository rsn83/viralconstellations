#!/usr/bin/env python3
"""
180 -- IS THERE ENOUGH BACKGROUND DIVERSITY TO CONDITION ON?

WHY THIS RUN, AND WHY IT SHOULD HAVE COME FIRST
-----------------------------------------------
179 tried to test whether attachment depends on WHICH background, using a
control that scores each background with a dissimilar background's profile.
The control could not fire: a dissimilar background (Jaccard <= 0.3) existed
for only 6% of backgrounds in the 2024-07..2025-02 test window, so the
control silently fell back to the global attachment baseline and the test
was vacuous.

That is a fact about the DATA, not the control. If essentially every
circulating constellation resembles every other, then "which background"
carries little information by construction, and no estimator or architecture
recovers signal that is not there.

This script measures that diversity directly, per month, across all 77
months. It should have been run before 171.

THE DESIGN QUESTION IT DECIDES
------------------------------
    diversity HIGH in some months, LOW in others
        -> background conditioning is informative during regime transitions
           and uninformative during clonal sweeps. The model must represent a
           HETEROGENEOUS population -- e.g. incidence-matrix representations
           as in HGDHE (AAAI 2023), which express which mutations belong to
           which of the currently circulating hyperedges. Conditioning should
           be gated on diversity.

    diversity LOW throughout
        -> background identity never carried signal in this representation.
           Drop background conditioning. Model attachment-frequency dynamics
           instead, which is what A1 in 179 already does (+0.059 recall@100
           over population frequency). Much simpler object.

WHAT IS MEASURED
    n_bg            backgrounds present (mass >= threshold)
    mean_jaccard    mean pairwise Jaccard among backgrounds
    frac_dissim     fraction of pairs at Jaccard <= 0.3
                    (this is the quantity that made 179's control fail)
    eff_clusters    exp(entropy of mass distribution) -- effective number of
                    distinct constellations carrying the population
    n_components    connected components of the graph linking backgrounds at
                    Jaccard > 0.5, i.e. how many distinct lineage groups are
                    circulating

frac_dissim is the operative number. 179 needed it above ~0.3 to work.

This is a property of the population, computed with no model and no
train/test split, so it is not subject to the leakage and metric issues that
have affected earlier scripts.

USAGE
    python scripts/180_diversity.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --out results/diversity.json

GIT
    git add scripts/180_diversity.py
    git commit -m "180: background diversity per month -- is conditioning possible at all"
    git push
"""

import argparse
import importlib.util
import json
import random

import numpy as np


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def components(bgs, thresh=0.5):
    """Connected components of the graph joining backgrounds at Jaccard >
    thresh. Counts distinct circulating lineage groups."""
    n = len(bgs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if len(bgs[i] & bgs[j]) / len(bgs[i] | bgs[j]) > thresh:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    return len({find(i) for i in range(n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--max-bg", type=int, default=200,
                    help="heaviest N backgrounds per month, matching 175/179")
    ap.add_argument("--dissim", type=float, default=0.3,
                    help="Jaccard threshold defining a dissimilar pair; 0.3 "
                         "is the value 179's control used")
    ap.add_argument("--max-pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    L = load_ladder(args.ladder)
    print("loading ...")
    monthly = L.load_events(args.events)
    pops = {m: L.population(monthly[m]) for m in sorted(monthly)}
    pops = {m: p for m, p in pops.items() if p}
    months = sorted(pops)
    print(f"  {len(months)} months")

    rows = []
    for m in months:
        items = sorted(pops[m].items(), key=lambda kv: -kv[1])[:args.max_bg]
        bgs = [S for S, _ in items]
        w = np.array([x for _, x in items], dtype=np.float64)
        if len(bgs) < 3:
            continue
        w = w / w.sum()
        ent = float(-(w * np.log(w + 1e-12)).sum())

        pairs = []
        n = len(bgs)
        if n * (n - 1) // 2 <= args.max_pairs:
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append(len(bgs[i] & bgs[j]) / len(bgs[i] | bgs[j]))
        else:
            for _ in range(args.max_pairs):
                i, j = rng.randrange(n), rng.randrange(n)
                if i == j:
                    continue
                pairs.append(len(bgs[i] & bgs[j]) / len(bgs[i] | bgs[j]))
        if not pairs:
            continue
        P = np.array(pairs)

        rows.append({
            "month": m,
            "n_bg": len(bgs),
            "median_size": int(np.median([len(S) for S in bgs])),
            "mean_jaccard": float(P.mean()),
            "frac_dissim": float((P <= args.dissim).mean()),
            "eff_clusters": float(np.exp(ent)),
            "n_components": components(bgs[:120]),
        })

    if not rows:
        print("  NO MONTHS")
        return

    print(f"\n  {'month':9s} {'n_bg':>6s} {'|S|':>5s} {'meanJ':>7s} "
          f"{'frac<=%.1f' % args.dissim:>9s} {'eff_cl':>8s} {'ncomp':>6s}")
    print("  " + "-" * 56)
    for r in rows:
        print(f"  {r['month']:9s} {r['n_bg']:6d} {r['median_size']:5d} "
              f"{r['mean_jaccard']:7.3f} {r['frac_dissim']:9.3f} "
              f"{r['eff_clusters']:8.1f} {r['n_components']:6d}")

    fd = np.array([r["frac_dissim"] for r in rows])
    print(f"\n  frac_dissim: min {fd.min():.3f}  median {np.median(fd):.3f}"
          f"  max {fd.max():.3f}")
    hi = [r["month"] for r in rows if r["frac_dissim"] > 0.3]
    print(f"  months with frac_dissim > 0.3: {len(hi)}/{len(rows)}")
    if hi:
        print(f"    {', '.join(hi[:15])}" + (" ..." if len(hi) > 15 else ""))

    print("\n  READING THIS")
    print("  frac_dissim varies -> background conditioning is informative in")
    print("     some regimes only; the model must represent a heterogeneous")
    print("     population and gate conditioning on diversity.")
    print("  frac_dissim low throughout -> background identity carries little")
    print("     information in this representation; model attachment-frequency")
    print("     dynamics instead (A1 in 179) and drop the conditioning.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"max_bg": args.max_bg, "dissim": args.dissim,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
