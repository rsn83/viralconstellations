#!/usr/bin/env python
"""
82_size_decomposition.py

Average mutations per sequence rises from 1.07 to 55.47 across the series, while
about 90% of each month's sequences carry a set that was present the month
before. This script splits each month pair into three groups of sets to show
where the rise comes from.

For each pair of consecutive months, sets are divided into:

  carried   present in both months
  lost      present in the earlier month, absent in the later one
  new       absent in the earlier month, present in the later one

For each group the script reports the average number of mutations per set, and
the share of sequences in that group -- so a group's contribution to the monthly
average is size x weight.

Two weights are reported for carried sets, because they can change frequency:
their share of the earlier month's sequences and their share of the later
month's.

The identity that holds exactly:

  average size in month t     = size(carried, weighted in t) x weight_t(carried)
                              + size(lost)                  x weight_t(lost)

  average size in month t+1   = size(carried, weighted in t+1) x weight_{t+1}(carried)
                              + size(new)                     x weight_{t+1}(new)

The difference between those two lines is the monthly change, and the script
splits that difference into three parts:

  from sets being replaced   (lost sets leaving, new sets arriving)
  from carried sets shifting weight among themselves
  residual

Both months are sampled to the same number of sequences first, drawn in
proportion to how common each set is.

Usage
-----
python scripts/82_size_decomposition.py --min_count 3 --end_month 2024-12
python scripts/82_size_decomposition.py --self_test
"""

import argparse
import os
import pickle
import re

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


def weighted_size(occ, keys):
    """Average set size over the sequences carrying those sets."""
    w = np.array([occ[k] for k in keys], dtype=float)
    if w.sum() == 0:
        return np.nan
    s = np.array([len(k) for k in keys], dtype=float)
    return float((s * w).sum() / w.sum())


def decompose(a, b):
    """a, b are month dicts {set: count}, both sampled to the same total."""
    A, B = set(a), set(b)
    carried, lost, new = A & B, A - B, B - A
    ta, tb = float(sum(a.values())), float(sum(b.values()))

    wa_car = sum(a[k] for k in carried) / ta
    wa_lost = sum(a[k] for k in lost) / ta
    wb_car = sum(b[k] for k in carried) / tb
    wb_new = sum(b[k] for k in new) / tb

    sa_car = weighted_size(a, carried) if carried else np.nan
    sb_car = weighted_size(b, carried) if carried else np.nan
    s_lost = weighted_size(a, lost) if lost else np.nan
    s_new = weighted_size(b, new) if new else np.nan

    mean_a = weighted_size(a, A)
    mean_b = weighted_size(b, B)

    # split the change into: what the carried sets did on their own, and what
    # replacing lost sets with new ones did
    carried_effect = wb_car * (sb_car if carried else 0.0) - \
        wa_car * (sa_car if carried else 0.0)
    replace_effect = wb_new * (s_new if new else 0.0) - \
        wa_lost * (s_lost if lost else 0.0)

    return {
        "n_carried": len(carried), "n_lost": len(lost), "n_new": len(new),
        "weight_carried_before": wa_car, "weight_lost": wa_lost,
        "weight_carried_after": wb_car, "weight_new": wb_new,
        "size_carried_before": sa_car, "size_carried_after": sb_car,
        "size_lost": s_lost, "size_new": s_new,
        "mean_size_before": mean_a, "mean_size_after": mean_b,
        "change": mean_b - mean_a,
        "from_carried_sets": carried_effect,
        "from_replacement": replace_effect,
    }


def self_test():
    print("checking the decomposition")

    # everything carries over unchanged: no change, no contribution
    a = {frozenset({1, 2}): 80, frozenset({3}): 20}
    r = decompose(a, dict(a))
    assert abs(r["change"]) < 1e-12
    assert r["n_new"] == 0 and r["n_lost"] == 0
    assert abs(r["weight_carried_before"] - 1.0) < 1e-12
    print("  identical months -> change 0                     ok")

    # a small set is replaced by a large one
    b = {frozenset({1, 2}): 80, frozenset({4, 5, 6, 7}): 20}
    r = decompose(a, b)
    assert r["n_lost"] == 1 and r["n_new"] == 1
    assert abs(r["size_lost"] - 1.0) < 1e-9
    assert abs(r["size_new"] - 4.0) < 1e-9
    # before: (2*80 + 1*20)/100 = 1.8 ; after: (2*80 + 4*20)/100 = 2.4
    assert abs(r["mean_size_before"] - 1.8) < 1e-9
    assert abs(r["mean_size_after"] - 2.4) < 1e-9
    assert abs(r["change"] - 0.6) < 1e-9
    assert abs(r["from_replacement"] - 0.6) < 1e-9
    assert abs(r["from_carried_sets"]) < 1e-9
    print("  a small set replaced by a large one:")
    print("     change 0.600, all of it from replacement       ok")

    # carried sets keep the same membership but shift weight toward the larger
    c = {frozenset({1, 2}): 20, frozenset({3}): 80}
    r = decompose(a, c)
    assert r["n_new"] == 0 and r["n_lost"] == 0
    assert abs(r["change"] - (1.2 - 1.8)) < 1e-9
    assert abs(r["from_carried_sets"] - (1.2 - 1.8)) < 1e-9
    assert abs(r["from_replacement"]) < 1e-9
    print("  carried sets shifting weight among themselves:")
    print("     change -0.600, all of it from carried sets     ok")

    # the two parts must add to the total change
    d = {frozenset({1, 2}): 50, frozenset({9, 9, 8}): 50}
    r = decompose(a, d)
    assert abs(r["from_carried_sets"] + r["from_replacement"]
               - r["change"]) < 1e-9
    print("  the two parts add up to the total change         ok")
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
    kept = []
    for m, o in months:
        s = sample_month(o, args.n_per_month, rng)
        if s:
            kept.append((m, s))
    names = [m for m, _ in kept]
    samp = {m: s for m, s in kept}
    print(f"months used: {len(names)}  ({names[0]} to {names[-1]}), "
          f"each sampled to {args.n_per_month:,} sequences")

    rows = []
    for i in range(len(names) - 1):
        r = decompose(samp[names[i]], samp[names[i + 1]])
        rows.append({"month": names[i], "next": names[i + 1], **r})
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/82_size_decomposition.csv", index=False)

    cols = ["month", "next", "mean_size_before", "mean_size_after", "change",
            "n_carried", "size_carried_before", "size_carried_after",
            "weight_carried_after",
            "n_lost", "size_lost", "weight_lost",
            "n_new", "size_new", "weight_new",
            "from_carried_sets", "from_replacement"]
    print("\n" + "=" * 78)
    print("AVERAGE SET SIZE, SPLIT BY WHAT HAPPENED TO EACH SET")
    print("=" * 78)
    print("size = average mutations per set, over the sequences carrying it")
    print("weight = share of that month's sequences\n")
    print(df[cols].round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("TOTALS")
    print("=" * 78)
    print(f"average size, first month : {df['mean_size_before'].iloc[0]:.2f}")
    print(f"average size, last month  : {df['mean_size_after'].iloc[-1]:.2f}")
    print(f"total rise                : "
          f"{df['mean_size_after'].iloc[-1] - df['mean_size_before'].iloc[0]:.2f}")
    print(f"sum of monthly changes    : {df['change'].sum():.2f}")
    print(f"  of which from carried sets shifting weight : "
          f"{df['from_carried_sets'].sum():.2f}")
    print(f"  of which from lost sets replaced by new    : "
          f"{df['from_replacement'].sum():.2f}")

    print("\nlargest monthly rises:")
    top = df.nlargest(8, "change")[
        ["month", "next", "change", "from_carried_sets", "from_replacement",
         "size_lost", "size_new", "weight_new"]]
    print(top.round(3).to_string(index=False))

    print("\naverages over all pairs:")
    for c in ["size_carried_before", "size_carried_after", "size_lost",
              "size_new", "weight_carried_after", "weight_lost", "weight_new"]:
        print(f"  {c:24s} {df[c].mean():.3f}")

    print(f"\nwrote outputs/82_size_decomposition.csv")


if __name__ == "__main__":
    main()
