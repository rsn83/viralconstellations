#!/usr/bin/env python
"""
83_where_new_sets_come_from.py

The observation this follows from
---------------------------------
Each month brings 30-50 mutations that were absent the month before. But the
number that were never seen in ANY earlier month falls from 45 in 2020-04 to
1-4 from 2023 onward. So the alphabet closes around 2021, and after that the
"new" mutations are almost all ones that had dropped out and come back.

If the alphabet is closed and recycling, then new sets should be derivable from
the past. This script measures how far back you have to look.

What it accounts for
--------------------
For every set present at month t+1 and absent at month t, it asks in order:

  1. Is it one mutation added to a set present at month t?
  2. If not, is it one mutation added to a set present at t-1, t-2, ... t-k?
     (reported as the smallest number of months you must look back)
  3. If not, is it two mutations added to a set present at t?
  4. Is it a set that existed at some earlier month, disappeared, and came back?
  5. Otherwise: unaccounted for within this reach.

Every new set falls into exactly one of these, so the shares sum to 1.

It also splits the mutations entering each month into never-seen and returning,
and for the returning ones reports how many months they were gone and whether
they came back on a similar genetic background or a different one.

Usage
-----
python scripts/83_where_new_sets_come_from.py --min_count 3 --end_month 2024-12
python scripts/83_where_new_sets_come_from.py --self_test
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")


def load_months(data_dir, min_count, start_month=None, end_month=None):
    files = []
    for fn in os.listdir(data_dir):
        m = MONTH_RE.search(fn)
        if m:
            files.append((m.group(1), os.path.join(data_dir, fn)))
    files.sort()
    out = []
    for month, path in files:
        if start_month and month < start_month:
            continue
        if end_month and month > end_month:
            continue
        with open(path, "rb") as f:
            occ = pickle.load(f)
        occ = {k: v for k, v in occ.items() if v >= min_count}
        if occ:
            out.append((month, occ))
    return out


def sample_month(occ, n_target, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    if counts.sum() < n_target:
        return None
    draws = rng.multinomial(n_target, counts / counts.sum())
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws)[0]}


def one_step_source(s, pool):
    """Is s one mutation added to some set in pool? Returns the mutation or None."""
    for x in s:
        if frozenset(s - {x}) in pool:
            return x
    return None


def two_step(s, pool):
    """Is s two mutations added to some set in pool?"""
    xs = list(s)
    for i in range(len(xs)):
        a = frozenset(s - {xs[i]})
        if a in pool:
            return True
        for j in range(i + 1, len(xs)):
            if frozenset(s - {xs[i], xs[j]}) in pool:
                return True
    return False


def classify(s, pools, ever_sets, max_back):
    """
    pools[0] is month t, pools[1] is t-1, and so on.
    Returns a category and, for the look-back case, how far back.
    """
    if s in ever_sets:
        return "seen_before_and_returned", 0
    if one_step_source(s, pools[0]) is not None:
        return "one_step_from_this_month", 0
    for k in range(1, min(max_back, len(pools))):
        if one_step_source(s, pools[k]) is not None:
            return "one_step_from_an_earlier_month", k
    if two_step(s, pools[0]):
        return "two_steps_from_this_month", 0
    return "unaccounted", -1


def self_test():
    print("checking the accounting")

    now = {frozenset({1, 2, 3}), frozenset({7, 8})}
    older = {frozenset({4, 5})}

    # one mutation added to a set present now
    c, k = classify(frozenset({1, 2, 3, 9}), [now, older], set(), 6)
    assert c == "one_step_from_this_month" and k == 0
    print("  one mutation added to a current set              ok")

    # one mutation added to a set that was present a month ago
    c, k = classify(frozenset({4, 5, 6}), [now, older], set(), 6)
    assert c == "one_step_from_an_earlier_month" and k == 1, (c, k)
    print("  one mutation added to a set from a month back    ok")

    # two mutations added to a current set
    c, k = classify(frozenset({1, 2, 3, 9, 10}), [now, older], set(), 6)
    assert c == "two_steps_from_this_month", c
    print("  two mutations added to a current set             ok")

    # a set that existed before, went away, and came back
    c, k = classify(frozenset({99, 98}), [now, older], {frozenset({99, 98})}, 6)
    assert c == "seen_before_and_returned", c
    print("  a set that existed earlier and returned          ok")

    # nothing within reach
    c, k = classify(frozenset({50, 51, 52, 53, 54}), [now, older], set(), 6)
    assert c == "unaccounted", c
    print("  nothing within reach                             ok")

    # the returning check must take priority over the one-step check, so a
    # returning set is never miscounted as newly built
    c, _ = classify(frozenset({1, 2, 3, 9}), [now, older],
                    {frozenset({1, 2, 3, 9})}, 6)
    assert c == "seen_before_and_returned", c
    print("  a returning set is not counted as newly built    ok")
    print("all checks passed\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--n_per_month", type=int, default=5000)
    ap.add_argument("--max_back", type=int, default=12)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    kept = []
    for m, o in months:
        s = sample_month(o, args.n_per_month, rng)
        if s:
            kept.append((m, s))
    names = [m for m, _ in kept]
    samp = {m: s for m, s in kept}
    print(f"months used: {len(names)} ({names[0]} to {names[-1]}), "
          f"each sampled to {args.n_per_month:,} sequences\n")

    CATS = ["one_step_from_this_month", "one_step_from_an_earlier_month",
            "two_steps_from_this_month", "seen_before_and_returned",
            "unaccounted"]

    rows, mut_rows, back_hist = [], [], defaultdict(int)
    ever_sets, ever_mut = set(), set()
    last_seen_mut = {}
    last_set_of_mut = {}

    for i in range(len(names) - 1):
        a, b = samp[names[i]], samp[names[i + 1]]
        pools = [set(samp[names[i - k]]) for k in range(0, min(i + 1,
                                                              args.max_back))]
        mut_a = set()
        for s in a:
            mut_a |= set(s)
        ever_sets |= set(a)
        ever_mut |= mut_a
        for mm in mut_a:
            last_seen_mut[mm] = i
            for s in a:
                if mm in s:
                    last_set_of_mut[mm] = s
                    break

        new_sets = [s for s in b if s not in a]
        if not new_sets:
            continue
        counts = defaultdict(int)
        wts = defaultdict(float)
        tot_w = sum(b[s] for s in new_sets)
        for s in new_sets:
            c, k = classify(s, pools, ever_sets, args.max_back)
            counts[c] += 1
            wts[c] += b[s]
            if c == "one_step_from_an_earlier_month":
                back_hist[k] += 1
        row = {"month": names[i + 1], "new_sets": len(new_sets)}
        for c in CATS:
            row[c] = counts[c] / len(new_sets)
        for c in CATS:
            row["w_" + c] = wts[c] / tot_w if tot_w else np.nan
        rows.append(row)

        # mutations entering this month
        mut_b = set()
        for s in b:
            mut_b |= set(s)
        entering = mut_b - mut_a
        never = entering - ever_mut
        returning = entering & ever_mut
        gaps, same_bg = [], []
        for mm in returning:
            gaps.append(i - last_seen_mut[mm])
            prev = last_set_of_mut.get(mm)
            if prev is not None:
                now_set = next((s for s in b if mm in s), None)
                if now_set is not None:
                    same_bg.append(len(now_set ^ prev))
        mut_rows.append({
            "month": names[i + 1],
            "mutations_entering": len(entering),
            "never_seen_before": len(never),
            "returning": len(returning),
            "median_months_gone": float(np.median(gaps)) if gaps else np.nan,
            "median_distance_to_last_background":
                float(np.median(same_bg)) if same_bg else np.nan,
        })

    df = pd.DataFrame(rows)
    dm = pd.DataFrame(mut_rows)
    df.to_csv(f"{args.out_dir}/83_new_set_sources.csv", index=False)
    dm.to_csv(f"{args.out_dir}/83_mutation_entry.csv", index=False)

    print("=" * 104)
    print("WHERE EACH NEW SET COMES FROM (share of that month's new sets)")
    print("=" * 104)
    show = df[["month", "new_sets"] + CATS].copy()
    show.columns = ["month", "new sets", "one step from this month",
                    "one step from an earlier month",
                    "two steps from this month",
                    "existed before, returned", "unaccounted"]
    print(show.round(3).to_string(index=False))

    print("\naverages over all months, by count of sets:")
    for c in CATS:
        print(f"  {c:34s} {df[c].mean():.3f}")
    print("\naverages weighted by how many sequences carry them:")
    for c in CATS:
        print(f"  {c:34s} {df['w_' + c].mean():.3f}")

    if back_hist:
        print("\nfor sets reached only by looking back, how far back:")
        tot = sum(back_hist.values())
        cum = 0
        for k in sorted(back_hist):
            cum += back_hist[k]
            print(f"  {k} month(s) back: {back_hist[k]:5d}   "
                  f"cumulative {cum/tot:.3f}")

    print("\n" + "=" * 104)
    print("MUTATIONS ENTERING EACH MONTH")
    print("=" * 104)
    print(dm.round(2).to_string(index=False))
    print("\naverages:")
    print(f"  mutations entering per month : "
          f"{dm['mutations_entering'].mean():.1f}")
    print(f"  never seen before            : "
          f"{dm['never_seen_before'].mean():.1f}")
    print(f"  returning                    : {dm['returning'].mean():.1f}")
    print(f"  months gone, median          : "
          f"{dm['median_months_gone'].median():.1f}")
    print(f"  distance from the set it was last on to the set it")
    print(f"    returns on, median          : "
          f"{dm['median_distance_to_last_background'].median():.1f}")

    early = dm[dm["month"] < "2022-01"]
    late = dm[dm["month"] >= "2022-01"]
    print(f"\n  before 2022: {early['never_seen_before'].mean():.1f} never seen, "
          f"{early['returning'].mean():.1f} returning per month")
    print(f"  2022 onward: {late['never_seen_before'].mean():.1f} never seen, "
          f"{late['returning'].mean():.1f} returning per month")

    print(f"\nwrote 2 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
