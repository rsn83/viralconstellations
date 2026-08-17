#!/usr/bin/env python
"""
81_basic_facts.py

Plain description of the data. No models, no prediction, no ranking metrics.
Everything here is counting.

The script answers, in order:

  1. How many sequences per month, and how many months are usable.
  2. After sampling every usable month down to the same number of sequences:
     how many distinct sets, how many distinct mutations, how many mutations
     per sequence.
  3. Between each pair of consecutive months:
       - how many sets carry over, how many disappear, how many are new
       - the same, weighted by how many sequences carry each set
       - the same for individual mutations
  4. What the new sets are made of:
       - do they use only mutations already circulating?
       - are they one mutation away from a set already circulating?
       - do any contain a mutation never seen in any earlier month?
  5. The copy-forward benchmark stated plainly: if you simply assert that next
     month looks exactly like this month, how right are you -- counted two ways.

The two ways in point 5 differ a lot and that difference is the point.
Counting distinct sets, most of next month is new. Counting sequences, most of
next month is old. Both are true: the sets that carry over are the common ones.

Sampling
--------
Months with fewer than the target number of sequences are dropped. Every other
month is sampled down to exactly that number, by drawing sequences at random in
proportion to how common each set is. This is done so that months can be
compared: without it, a month with 700,000 sequences shows many more distinct
sets than a month with 6,000, for reasons that have nothing to do with the virus.
Raw numbers are reported alongside so the effect of the sampling is visible.

Usage
-----
python scripts/81_basic_facts.py --min_count 3 --end_month 2024-12
python scripts/81_basic_facts.py --self_test
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
    """Draw n_target sequences, in proportion to how common each set is."""
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    if counts.sum() < n_target:
        return None
    draws = rng.multinomial(n_target, counts / counts.sum())
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws)[0]}


def mutations_of(occ):
    v = set()
    for s in occ:
        v |= set(s)
    return v


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def self_test():
    print("checking the counting is right")

    a = {frozenset({1, 2}): 60, frozenset({3}): 40}
    b = {frozenset({1, 2}): 90, frozenset({4}): 10}

    # sets: one of two carries over
    assert jaccard(set(a), set(b)) == 1 / 3
    print("  set overlap: 1 shared, 3 in total -> 0.333          ok")

    # weighted by sequences: 90 of month b's 100 sequences are in a set that
    # existed in month a
    w = sum(v for k, v in b.items() if k in a) / sum(b.values())
    assert abs(w - 0.9) < 1e-9
    print("  same pair weighted by sequences -> 0.900            ok")
    print("     (this gap between 0.333 and 0.900 is the whole point)")

    # mutations
    assert mutations_of(a) == {1, 2, 3}
    assert jaccard(mutations_of(a), mutations_of(b)) == 2 / 4
    print("  mutation overlap: 2 shared of 4 -> 0.500            ok")

    # sampling keeps proportions
    rng = np.random.default_rng(0)
    big = {frozenset({1}): 9000, frozenset({2}): 1000}
    s = sample_month(big, 1000, rng)
    share = s[frozenset({1})] / sum(s.values())
    assert 0.86 < share < 0.94, share
    print(f"  sampling preserves proportions ({share:.2f} vs 0.90)     ok")
    assert sample_month(big, 20000, rng) is None
    print("  a month with too few sequences is dropped           ok")
    print("all checks passed\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--n_per_month", type=int, default=5000)
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

    # ---- 1. what we start with, and what we keep -------------------------
    print("=" * 78)
    print("1. THE DATA")
    print("=" * 78)
    raw_n = {m: int(sum(o.values())) for m, o in months}
    print(f"months on disk: {len(months)}  ({months[0][0]} to {months[-1][0]})")
    print(f"sequences per month: smallest {min(raw_n.values()):,}, "
          f"largest {max(raw_n.values()):,}")
    print(f"total sequences: {sum(raw_n.values()):,}")

    kept, dropped = [], []
    for m, o in months:
        s = sample_month(o, args.n_per_month, rng)
        (kept if s else dropped).append((m, s))
    names = [m for m, _ in kept]
    samp = {m: s for m, s in kept}
    raw = {m: o for m, o in months}
    print(f"\nmonths with at least {args.n_per_month:,} sequences: {len(kept)}")
    if dropped:
        print(f"months dropped: {[m for m, _ in dropped]}")
    print(f"kept: {names[0]} to {names[-1]}")
    print(f"each month sampled down to exactly {args.n_per_month:,} sequences")

    # ---- 2. what each month looks like ------------------------------------
    print("\n" + "=" * 78)
    print("2. EACH MONTH ON ITS OWN")
    print("=" * 78)
    rows = []
    for m in names:
        s, o = samp[m], raw[m]
        w = np.array(list(s.values()), dtype=float)
        sizes = np.array([len(k) for k in s], dtype=float)
        rows.append({
            "month": m,
            "sequences_before_sampling": raw_n[m],
            "distinct_sets_before": len(o),
            "distinct_sets_after": len(s),
            "distinct_mutations_before": len(mutations_of(o)),
            "distinct_mutations_after": len(mutations_of(s)),
            "mutations_per_sequence": float((sizes * w).sum() / w.sum()),
        })
    d2 = pd.DataFrame(rows)
    d2.to_csv(f"{args.out_dir}/81_per_month.csv", index=False)
    print(d2.to_string(index=False))

    print("\nafter sampling every month to the same size:")
    print(f"  distinct sets per month: {d2['distinct_sets_after'].min()} "
          f"to {d2['distinct_sets_after'].max()}")
    print(f"  distinct mutations per month: "
          f"{d2['distinct_mutations_after'].min()} to "
          f"{d2['distinct_mutations_after'].max()}")
    print(f"  mutations carried per sequence: "
          f"{d2['mutations_per_sequence'].iloc[0]:.1f} at the start, "
          f"{d2['mutations_per_sequence'].iloc[-1]:.1f} at the end")
    print("\n  before sampling, distinct mutations per month ranged "
          f"{d2['distinct_mutations_before'].min()} to "
          f"{d2['distinct_mutations_before'].max()},")
    print("  which mostly tracks how much sequencing was done that month")

    # ---- 3. month to month -------------------------------------------------
    print("\n" + "=" * 78)
    print("3. FROM ONE MONTH TO THE NEXT")
    print("=" * 78)
    rows = []
    ever_mut = set()
    for i in range(len(names) - 1):
        a, b = samp[names[i]], samp[names[i + 1]]
        A, B = set(a), set(b)
        ma, mb = mutations_of(a), mutations_of(b)
        ever_mut |= ma
        tot_b = sum(b.values())
        carried_seqs = sum(v for k, v in b.items() if k in A)
        rows.append({
            "month": names[i], "next": names[i + 1],
            "sets_carried_over": len(A & B),
            "sets_disappeared": len(A - B),
            "sets_new": len(B - A),
            "set_overlap": jaccard(A, B),
            "share_of_next_month_sequences_in_old_sets":
                carried_seqs / tot_b,
            "mutations_carried_over": len(ma & mb),
            "mutations_disappeared": len(ma - mb),
            "mutations_new": len(mb - ma),
            "mutation_overlap": jaccard(ma, mb),
            "mutations_never_seen_before": len(mb - ever_mut),
        })
    d3 = pd.DataFrame(rows)
    d3.to_csv(f"{args.out_dir}/81_month_to_month.csv", index=False)
    show = ["month", "next", "sets_carried_over", "sets_disappeared",
            "sets_new", "set_overlap",
            "share_of_next_month_sequences_in_old_sets",
            "mutations_new", "mutations_never_seen_before",
            "mutation_overlap"]
    print(d3[show].round(3).to_string(index=False))

    print("\naverages:")
    print(f"  sets that carry over          : "
          f"{d3['sets_carried_over'].mean():.0f}")
    print(f"  sets that disappear           : "
          f"{d3['sets_disappeared'].mean():.0f}")
    print(f"  sets that are new             : {d3['sets_new'].mean():.0f}")
    print(f"  new mutations per month       : {d3['mutations_new'].mean():.1f}, "
          f"of which never seen before: "
          f"{d3['mutations_never_seen_before'].mean():.1f}")

    # ---- 4. what the new sets are made of ----------------------------------
    print("\n" + "=" * 78)
    print("4. WHAT THE NEW SETS ARE MADE OF")
    print("=" * 78)
    rows = []
    ever_mut = set()
    for i in range(len(names) - 1):
        a, b = samp[names[i]], samp[names[i + 1]]
        A = set(a)
        ma = mutations_of(a)
        ever_mut |= ma
        new = [s for s in b if s not in A]
        if not new:
            continue
        only_old, one_step, has_novel = 0, 0, 0
        seqs_new = sum(b[s] for s in new)
        seqs_one_step = 0
        for s in new:
            if s <= ma:
                only_old += 1
            if s - ever_mut:
                has_novel += 1
            if any(frozenset(s - {x}) in A for x in s):
                one_step += 1
                seqs_one_step += b[s]
        rows.append({
            "month": names[i + 1], "new_sets": len(new),
            "share_using_only_circulating_mutations": only_old / len(new),
            "share_one_mutation_from_an_existing_set": one_step / len(new),
            "share_containing_a_never_seen_mutation": has_novel / len(new),
            "share_of_new_sequences_one_step_away":
                seqs_one_step / seqs_new if seqs_new else np.nan,
        })
    d4 = pd.DataFrame(rows)
    d4.to_csv(f"{args.out_dir}/81_new_sets.csv", index=False)
    print(d4.round(3).to_string(index=False))
    print("\naverages:")
    for c, lab in [
        ("share_using_only_circulating_mutations",
         "new sets built only from mutations already circulating"),
        ("share_one_mutation_from_an_existing_set",
         "new sets that are one mutation added to an existing set"),
        ("share_containing_a_never_seen_mutation",
         "new sets containing a mutation never seen in any earlier month"),
    ]:
        print(f"  {lab:62s} {d4[c].mean():.3f}")

    # ---- 5. the copy-forward benchmark -------------------------------------
    print("\n" + "=" * 78)
    print("5. WHAT HAPPENS IF YOU JUST COPY THIS MONTH ONTO NEXT MONTH")
    print("=" * 78)
    print(f"  counting distinct sets, overlap        : "
          f"{d3['set_overlap'].mean():.3f}")
    print(f"  counting sequences, share of next month")
    print(f"    that sits in a set we already had    : "
          f"{d3['share_of_next_month_sequences_in_old_sets'].mean():.3f}")
    print(f"  counting distinct mutations, overlap   : "
          f"{d3['mutation_overlap'].mean():.3f}")
    print("\n  These two views of the same thing disagree, and the disagreement")
    print("  is the reason this problem is hard to make progress on.")
    print("  Counting distinct sets, a large share of next month is new.")
    print("  Counting sequences, most of next month sits in sets we already")
    print("  had -- because the sets that carry over are the common ones and")
    print("  the new ones are mostly rare.")
    print("\n  So any measurement that weights by sequences makes copying look")
    print("  almost right, and any model has to earn its improvement on the")
    print("  small, rare part.")

    print(f"\nwrote 4 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
