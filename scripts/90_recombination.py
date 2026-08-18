#!/usr/bin/env python
"""
90_recombination.py

The question
------------
Script 83 found that 20% of new constellations each month cannot be reached
by adding one or two mutations to anything circulating. We called these
"unaccounted." This script asks: are they recombinants?

A recombinant set c has two parents A and B if there exists a breakpoint
position p (an amino-acid position in spike) such that:
  - every mutation in c at positions < p is present in parent A
  - every mutation in c at positions >= p is present in parent B
  - both A and B are circulating this month

This is tested directly from spike amino-acid mutation sets using the
position information in posres_vocab.tsv.

Three groups are tested per month:
  - unaccounted sets (the 20% from script 83)
  - one-step sets (the 55%, as a control)
  - returned sets (the 15%, as a second control)

If recombination explains a large share of unaccounted sets but very few
one-step or returned sets, the signal is specific to the unreachable fraction.

The XBB transition (2023-01 -> 2023-02) is the key case. XBB is known to
be a recombinant of BJ.1 and BM.1.1.1 with a breakpoint at spike position
~486 (genomic position 22,920). If the test recovers this, it validates the
approach.

Note
----
This is an amino-acid level approximation. The true recombination breakpoint
is in the nucleotide sequence. At the amino-acid level, a breakpoint between
spike positions 486 and 490 corresponds to the known XBB breakpoint. False
positives are possible when two parents happen to share mutations on both
sides of a position boundary by chance. The control groups measure this rate.

Usage
-----
python scripts/90_recombination.py --min_count 3 --end_month 2024-12
python scripts/90_recombination.py --self_test
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

TRANSITIONS = {
    "2021-01": "Alpha",
    "2021-06": "Delta",
    "2021-12": "Omicron_BA1",
    "2022-03": "BA2",
    "2022-06": "BA5",
    "2023-02": "XBB",
    "2023-12": "JN1",
}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_vocab(data_dir):
    """Returns {node_idx: aa_pos} for position lookup."""
    path = os.path.join(data_dir, "posres_vocab.tsv")
    idx2pos = {}
    with open(path) as f:
        header = f.readline().strip().split("\t")
        cols = {c.lower(): i for i, c in enumerate(header)}
        id_col = next(cols[c] for c in ("node_idx", "id", "node") if c in cols)
        pos_col = next(cols[c] for c in ("aa_pos", "pos", "position")
                       if c in cols)
        for line in f:
            parts = line.strip().split("\t")
            nid = int(parts[id_col])
            pos = int(parts[pos_col])
            idx2pos[nid] = pos
    return idx2pos


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


# ----------------------------------------------------------------------------
# recombination test
# ----------------------------------------------------------------------------

def is_recombinant(c, circulating_sets, idx2pos, min_each_side=2):
    """
    Test whether set c could be a recombinant of two circulating parents.

    For each possible breakpoint (a position value between consecutive
    positions in c), split c into left and right halves and check whether
    some circulating set contains the left half and some circulating set
    contains the right half.

    min_each_side: minimum number of mutations required on each side of
    the breakpoint for the test to be meaningful. This prevents trivially
    splitting off one mutation and calling it recombination.

    Returns (is_recomb, breakpoint, left_parent_size, right_parent_size)
    """
    if len(c) < 2 * min_each_side:
        return False, None, None, None

    # sort mutations by position
    muts_by_pos = sorted(c, key=lambda m: idx2pos.get(m, 0))
    positions = [idx2pos.get(m, 0) for m in muts_by_pos]

    # build a lookup: for each circulating set, its mutation set
    circ = list(circulating_sets)

    # try each breakpoint between consecutive positions
    for split in range(min_each_side, len(muts_by_pos) - min_each_side + 1):
        left = frozenset(muts_by_pos[:split])
        right = frozenset(muts_by_pos[split:])
        bp = positions[split]

        left_found = any(left <= s for s in circ)
        if not left_found:
            continue
        right_found = any(right <= s for s in circ)
        if right_found:
            # find sizes of matching parents for reporting
            lp = min((len(s) for s in circ if left <= s), default=None)
            rp = min((len(s) for s in circ if right <= s), default=None)
            return True, bp, lp, rp

    return False, None, None, None


def classify_new_sets(occ_t, occ_next, ever_sets, ever_mut, idx2pos,
                      min_each_side=2):
    """
    For each set present in occ_next but absent in occ_t, classify it as:
      one_step: one mutation added to a set in occ_t
      two_step: two mutations added to a set in occ_t
      returned: was in ever_sets (appeared in an earlier month)
      unaccounted: none of the above

    Then test each group for recombination.
    """
    A = set(occ_t.keys())
    B = set(occ_next.keys())
    new_sets = [c for c in B if c not in A]

    mut_t = set()
    for s in occ_t:
        mut_t |= set(s)

    results = []
    for c in new_sets:
        # classify
        if c in ever_sets:
            group = "returned"
        elif any(frozenset(c - {m}) in A for m in c):
            group = "one_step"
        elif any(frozenset(c - {m}) in A
                 for m in c
                 for _ in [None]  # dummy loop
                 ) or any(
                frozenset(c - {m1, m2}) in A
                for m1 in c for m2 in c if m1 != m2
            ):
            group = "two_step"
        else:
            group = "unaccounted"

        # test recombination
        is_rec, bp, lp, rp = is_recombinant(c, A, idx2pos, min_each_side)

        results.append({
            "set_size": len(c),
            "group": group,
            "is_recombinant": is_rec,
            "breakpoint_pos": bp,
            "left_parent_size": lp,
            "right_parent_size": rp,
            "n_mut_in_vocab": sum(1 for m in c if m in mut_t),
            "count": occ_next[c],
        })

    return results


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    print("self-test")

    # vocabulary: positions 1-10
    idx2pos = {i: i for i in range(1, 11)}

    # parent A has mutations 1-5, parent B has mutations 6-10
    A = frozenset({1, 2, 3, 4, 5})
    B = frozenset({6, 7, 8, 9, 10})
    circulating = {A, B, frozenset({1, 2}), frozenset({8, 9, 10})}

    # recombinant: left half from A, right half from B
    c = frozenset({1, 2, 3, 7, 8, 9})
    is_rec, bp, lp, rp = is_recombinant(c, circulating, idx2pos,
                                         min_each_side=2)
    assert is_rec, f"expected recombinant, got {is_rec}"
    print(f"  known recombinant detected, breakpoint at position {bp}  ok")

    # non-recombinant: mutations scattered with no clean split
    c2 = frozenset({1, 3, 5, 7, 9})
    is_rec2, _, _, _ = is_recombinant(c2, {frozenset({1, 2, 3, 4, 5, 6})},
                                       idx2pos, min_each_side=2)
    # this should NOT be detected since right half {7,9} has no parent
    assert not is_rec2, f"false positive: {is_rec2}"
    print("  non-recombinant correctly rejected                 ok")

    # too small: fewer than 2*min_each_side mutations
    c3 = frozenset({1, 2, 3})
    is_rec3, _, _, _ = is_recombinant(c3, circulating, idx2pos,
                                       min_each_side=2)
    assert not is_rec3
    print("  set too small correctly skipped                    ok")

    print("all checks passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--min_each_side", type=int, default=2,
                    help="minimum mutations on each side of breakpoint")
    ap.add_argument("--max_sets", type=int, default=800,
                    help="top sets per month to use as circulating parents")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)

    idx2pos = load_vocab(args.data_dir)
    print(f"loaded vocabulary: {len(idx2pos)} positions")

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    print(f"loaded {len(names)} months: {names[0]} .. {names[-1]}\n")

    ever_sets, ever_mut = set(), set()
    all_rows = []

    for i in range(len(names) - 1):
        m_t, m_n = names[i], names[i + 1]
        occ_t = {c: v for c, v in
                 sorted(occ_by[m_t].items(),
                        key=lambda kv: -kv[1])[:args.max_sets]}
        occ_n = occ_by[m_n]

        results = classify_new_sets(occ_t, occ_n, ever_sets, ever_mut,
                                    idx2pos, args.min_each_side)

        for r in results:
            r["month"] = m_t
            r["next_month"] = m_n
            r["is_transition"] = m_n in TRANSITIONS
            r["variant"] = TRANSITIONS.get(m_n, "")
        all_rows.extend(results)

        # update history
        ever_sets |= set(occ_t.keys())
        for s in occ_t:
            ever_mut |= set(s)

        # summary for this month
        if results:
            df_m = pd.DataFrame(results)
            for grp in ["one_step", "two_step", "returned", "unaccounted"]:
                sub = df_m[df_m["group"] == grp]
                if len(sub) == 0:
                    continue
                rec_rate = sub["is_recombinant"].mean()
                tag = f"  ** {TRANSITIONS.get(m_n,'')}" if m_n in TRANSITIONS else ""
                print(f"  {m_t}->{m_n} {grp:12s}: "
                      f"n={len(sub):4d} recomb={rec_rate:.3f}{tag}")

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{args.out_dir}/90_recombination.csv", index=False)

    print("\n" + "=" * 80)
    print("RECOMBINATION RATE BY GROUP, ALL MONTHS")
    print("=" * 80)
    summ = df.groupby("group").agg(
        n=("is_recombinant", "count"),
        recomb_rate=("is_recombinant", "mean"),
        mean_set_size=("set_size", "mean"),
    ).reset_index()
    print(summ.round(3).to_string(index=False))

    print("\n" + "=" * 80)
    print("RECOMBINATION RATE AT TRANSITION MONTHS")
    print("=" * 80)
    trans = df[df["is_transition"]]
    if len(trans):
        t2 = trans.groupby(["next_month", "variant", "group"]).agg(
            n=("is_recombinant", "count"),
            recomb_rate=("is_recombinant", "mean"),
        ).reset_index()
        print(t2.round(3).to_string(index=False))

    print("\n" + "=" * 80)
    print("XBB TRANSITION (2023-01 -> 2023-02) DETAIL")
    print("=" * 80)
    xbb = df[(df["next_month"] == "2023-02") & df["is_recombinant"]]
    if len(xbb):
        print(f"  {len(xbb)} recombinant sets detected")
        print(f"  breakpoint positions: "
              f"{sorted(xbb['breakpoint_pos'].dropna().unique().tolist())}")
        print(f"  mean set size of recombinants: "
              f"{xbb['set_size'].mean():.1f}")
        print(f"  groups: "
              f"{xbb['group'].value_counts().to_dict()}")
        print(f"\n  known XBB recombination breakpoint: spike position ~486")
        bps = xbb['breakpoint_pos'].dropna().values
        near_486 = sum(1 for b in bps if 480 <= b <= 495)
        print(f"  detected breakpoints near 486 (480-495): {near_486} / {len(bps)}")
    else:
        print("  no recombinants detected at XBB transition")

    print("\n" + "=" * 80)
    print("READING")
    print("=" * 80)
    print("""
  recomb_rate for unaccounted >> recomb_rate for one_step/returned:
     recombination explains a specific fraction of the unreachable sets.
     The 20% is not one homogeneous group -- it splits into recombinants
     (recoverable by a recombination model) and cryptic arrivals (not
     recoverable from the observed data).

  XBB breakpoint near position 486:
     validates the test. XBB is known to recombine at genomic position
     22,920, which corresponds to spike amino-acid position ~486.
     If the test recovers this, it confirms the spike-level approximation
     is detecting real recombination events.

  recomb_rate similar across groups:
     no specific recombination signal. All groups are equally explained
     (or not explained) by recombination. The unaccounted fraction is
     genuinely cryptic -- not recombination, not single-step mutation.
""")

    print(f"\nwrote outputs/90_recombination.csv")


if __name__ == "__main__":
    main()
