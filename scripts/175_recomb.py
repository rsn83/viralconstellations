#!/usr/bin/env python3
"""
175 -- DOES RECOMBINATION EXPLAIN THE UNREACHABLE NEW CONSTELLATIONS?

THE GAP THIS TARGETS
--------------------
171/173 measured that only ~54% of genuinely new constellations are within
one substitution of anything present in the population. The other 46% are
unreachable by local edit, and no refinement of p(D|S) can reach them,
because the parent is not in the candidate pool.

Three explanations are possible:
    (a) they are two-or-more-step edits          -> radius 2, 3 below
    (b) they are RECOMBINANTS of two co-circulating parents
    (c) they descend from unsampled intermediates -> unreachable in principle

This script measures (a) and (b). Whatever is left is (c).

WHY RECOMBINATION IS A DIFFERENT KIND OF OPERATOR
-------------------------------------------------
Substitution and deletion are already set operations: because sequences are
reference-encoded, both are just "a token is present". That is why radius-r
enumeration works for them at all.

Recombination is NOT a set operation. It is defined on sequence COORDINATES:

    child = parent_A[positions <= b]  U  parent_B[positions > b]

for some breakpoint b. It needs mutations ORDERED BY POSITION, which is
exactly analogous to a profile HMM needing an alignment path to define
insert/delete. Position order is the machinery that makes the operator
definable; without it recombination cannot be expressed.

XBB is the documented case: it arose from two co-circulating parents and sits
~30 mutations from either, so no radius-r ball around the population contains
it.

THE TEST
--------
S' is recombination-reachable from population P if there exist parents
A, B in P and a breakpoint b with

    A agrees with S' on all positions <= b        AND
    B agrees with S' on all positions >  b

Computed efficiently: let fd(A) be the FIRST position where A and S' differ,
and ld(B) the LAST position where B and S' differ. A works as a prefix donor
for any b < fd(A); B works as a suffix donor for any b >= ld(B). So S' is
reachable iff

    max_A fd(A)  >  min_B ld(B)

One pass over parents, no breakpoint loop.

NOTE: this measures REACHABILITY, not ancestry. Finding a (A, B, b) that
composes S' does not mean recombination produced it -- two parents may
compose a set that arose by ordinary substitution. This is a ceiling on what
a recombination-aware proposal could cover, exactly as 159 is a ceiling for
local perturbation.

USAGE
    python scripts/175_recomb.py \
        --events data/processed/events_v3.tsv \
        --vocab  data/processed/vocab_v3.tsv \
        --ladder scripts/171_ladder.py \
        --test-end 2025-02 --out results/recomb.json

GIT
    git add scripts/175_recomb.py
    git commit -m "175: recombination reachability -- what explains the unreachable 46%"
    git push
"""

import argparse
import importlib.util
import json
import re
from collections import defaultdict

import numpy as np


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_positions(path):
    """vocab id -> spike position.

    Lines look like:  0 <TAB> S:A1015G <TAB> 2021-05-03
    Deletions look like S:DEL144 or S:DEL144-145 -- take the first number.
    """
    pos = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            vid, name = parts[0].strip(), parts[1]
            m = re.search(r"(\d+)", name.split(":")[-1])
            if m:
                pos[vid] = int(m.group(1))
    return pos


def reach_recomb(Sn, parents, pos, max_parents=200):
    """Is Sn composable as prefix(A) + suffix(B) for parents A, B?

    Returns (reachable, n_prefix_donors, n_suffix_donors).
    """
    best_fd = -1          # largest "first difference" over prefix donors
    best_ld = 10 ** 9     # smallest "last difference" over suffix donors
    npre = nsuf = 0
    INF = 10 ** 9
    for S in parents[:max_parents]:
        diff = S ^ Sn                       # symmetric difference
        if not diff:
            continue                        # identical, not a new set
        dpos = [pos[m] for m in diff if m in pos]
        if not dpos:
            continue
        fd, ld = min(dpos), max(dpos)
        if fd > best_fd:
            best_fd = fd
        if ld < best_ld:
            best_ld = ld
    # a breakpoint b exists iff some prefix donor agrees further right than
    # some suffix donor starts agreeing
    for S in parents[:max_parents]:
        diff = S ^ Sn
        dpos = [pos[m] for m in diff if m in pos]
        if not dpos:
            continue
        if min(dpos) == best_fd:
            npre += 1
        if max(dpos) == best_ld:
            nsuf += 1
    return (best_fd > best_ld), npre, nsuf


def min_radius(Sn, parents, cap=4):
    """Smallest Hamming distance from Sn to any parent, capped."""
    best = cap + 1
    for S in parents:
        d = len(S ^ Sn)
        if d < best:
            best = d
            if best <= 1:
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--vocab", default="data/processed/vocab_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-bg", type=int, default=200)
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    L = load_ladder(args.ladder)
    pos = load_positions(args.vocab)
    print(f"loading ...\n  {len(pos):,} mutations with positions")

    monthly = L.load_events(args.events)
    pops = {m: L.population(monthly[m]) for m in sorted(monthly)}
    pops = {m: p for m, p in pops.items() if p}
    months = sorted(pops)

    tr_end = L.TRAIN_END[:7]
    te_end = args.test_end or L.TEST_END[:7]
    test_months = [m for m in months if tr_end < m <= te_end]

    seen_ever = set()
    for m in [x for x in months if x <= tr_end]:
        seen_ever |= set(pops[m])

    rows = []
    for m in test_months:
        t_ix = months.index(m)
        nxt = t_ix + args.horizon
        if nxt >= len(months):
            break
        pT, pN = pops[m], pops[months[nxt]]
        new = L.new_constellations(pT, pN, seen_ever)
        seen_ever |= set(pT)
        if not new:
            continue

        parents = [S for S, _ in
                   sorted(pT.items(), key=lambda kv: -kv[1])[:args.max_bg]]
        if not parents:
            continue

        c = defaultdict(int)
        for Sn in new:
            r = min_radius(Sn, parents)
            if r <= 1:
                c["r1"] += 1
            elif r <= 2:
                c["r2"] += 1
            elif r <= 3:
                c["r3"] += 1
            else:
                c["far"] += 1
            if r > 2:
                ok, _, _ = reach_recomb(Sn, parents, pos, args.max_bg)
                if ok:
                    c["recomb_beyond_r2"] += 1
        n = len(new)
        rows.append({
            "month": m, "n_new": n,
            "r1": c["r1"] / n, "r2": c["r2"] / n, "r3": c["r3"] / n,
            "far": c["far"] / n,
            "recomb_beyond_r2": c["recomb_beyond_r2"] / n,
        })

    if not rows:
        print("  NO EVALUABLE MONTHS")
        return

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    r1, r2, r3, far = avg("r1"), avg("r2"), avg("r3"), avg("far")
    rec = avg("recomb_beyond_r2")

    print(f"\n  months           {len(rows)}")
    print(f"  new constellations/month  "
          f"{int(np.mean([r['n_new'] for r in rows]))}")
    print("\n  WHERE DO NEW CONSTELLATIONS COME FROM?")
    print(f"    within 1 mutation          {r1:.3f}")
    print(f"    within 2                   {r2:.3f}   (cum {r1 + r2:.3f})")
    print(f"    within 3                   {r3:.3f}   (cum {r1 + r2 + r3:.3f})")
    print(f"    further than 3             {far:.3f}")
    print(f"\n    of those beyond radius 2, recombination-reachable: "
          f"{rec:.3f} of all new")
    print(f"    LOCAL + RECOMBINATION covers {r1 + r2 + rec:.3f}")
    print(f"    unexplained (unsampled intermediates?) "
          f"{1 - r1 - r2 - rec:.3f}")
    print("\n  reachability, not ancestry: composability does not prove")
    print("  recombination produced the set.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"test_end": te_end, "n_months": len(rows),
                       "r1": r1, "r2": r2, "r3": r3, "far": far,
                       "recomb_beyond_r2": rec,
                       "local_plus_recomb": r1 + r2 + rec,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
