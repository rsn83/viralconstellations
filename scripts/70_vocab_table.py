#!/usr/bin/env python
"""
70_vocab_table.py

A plain table: per month, how big the mutation vocabulary is and how it changed
from the previous month.

  stayed   in the vocabulary at t-1 and still at t
  dropped  in the vocabulary at t-1 but not at t
  new      in the vocabulary at t but not at t-1
  first    of those new ones, how many had never been seen in ANY earlier month
           (the rest are returning after an absence)

Two versions of every row, because they answer different questions:

  RAW              every sequence in the month. Vocabulary size correlates
                   +0.815 with how many sequences were collected, so raw counts
                   partly measure surveillance effort rather than the virus.
  DEPTH-CONTROLLED every month subsampled to the same number of sequences
                   (default 5000) before counting, averaged over replicates. A
                   label counts as present if it is detected in at least half
                   the replicates. Months with fewer sequences than the depth
                   are blank.

Usage
-----
python scripts/70_vocab_table.py --min_count 3 --end_month 2024-12
python scripts/70_vocab_table.py --self_test
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


def raw_vocab(occ):
    v = set()
    for cs in occ:
        v |= set(cs)
    return v


def rarefied_vocab(occ, depth, min_count, reps, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    if counts.sum() < depth:
        return None
    seen = defaultdict(int)
    for _ in range(reps):
        draws = rng.multinomial(depth, counts / counts.sum())
        for i in np.flatnonzero(draws >= min_count):
            for l in keys[i]:
                seen[l] += 1
    return {l for l, c in seen.items() if c >= reps / 2}


def transitions(vocabs, months):
    """stayed / dropped / new / first, given a list of vocabulary sets."""
    rows = []
    ever = set()
    prev = None
    for m, v in zip(months, vocabs):
        if v is None:
            rows.append({"month": m, "vocab": np.nan, "stayed": np.nan,
                         "dropped": np.nan, "new": np.nan, "first": np.nan,
                         "jaccard": np.nan})
            continue
        if prev is None:
            rows.append({"month": m, "vocab": len(v), "stayed": np.nan,
                         "dropped": np.nan, "new": np.nan,
                         "first": len(v - ever), "jaccard": np.nan})
        else:
            stayed = v & prev
            new = v - prev
            dropped = prev - v
            union = v | prev
            rows.append({
                "month": m, "vocab": len(v), "stayed": len(stayed),
                "dropped": len(dropped), "new": len(new),
                "first": len(new - ever),
                "jaccard": len(stayed) / len(union) if union else np.nan,
            })
        ever |= v
        prev = v
    return rows


def self_test():
    print("self-test")
    months = ["m1", "m2", "m3", "m4"]
    v = [{1, 2, 3}, {2, 3, 4}, {2, 3, 4}, {1, 9}]
    r = pd.DataFrame(transitions(v, months))
    assert r.loc[1, "stayed"] == 2 and r.loc[1, "dropped"] == 1 \
        and r.loc[1, "new"] == 1 and r.loc[1, "first"] == 1
    print("  stayed/dropped/new counted correctly            ok")
    assert r.loc[2, "stayed"] == 3 and r.loc[2, "new"] == 0
    print("  identical months -> no change                   ok")
    # label 1 returns at m4 after being absent: new, but NOT first-ever
    assert r.loc[3, "new"] == 2 and r.loc[3, "first"] == 1
    print("  returning label counts as new, not first-ever   ok")
    # stayed + dropped must equal the previous month's vocabulary
    for i in range(1, 4):
        assert r.loc[i, "stayed"] + r.loc[i, "dropped"] == r.loc[i - 1, "vocab"]
    # stayed + new must equal this month's vocabulary
    for i in range(1, 4):
        assert r.loc[i, "stayed"] + r.loc[i, "new"] == r.loc[i, "vocab"]
    print("  the counts add up in both directions            ok")
    # a month with no data must not break the chain
    r2 = pd.DataFrame(transitions([{1, 2}, None, {1, 2, 3}], ["a", "b", "c"]))
    assert np.isnan(r2.loc[1, "vocab"]) and r2.loc[2, "new"] == 1
    print("  a missing month is skipped, not fatal           ok")
    print("all tests passed\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--depth", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=20)
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
    names = [m for m, _ in months]
    print(f"loaded {len(names)} months: {names[0]} .. {names[-1]}")

    n_seqs = [int(sum(o.values())) for _, o in months]
    raw = [raw_vocab(o) for _, o in months]
    rare = [rarefied_vocab(o, args.depth, args.min_count, args.reps, rng)
            for _, o in months]

    rdf = pd.DataFrame(transitions(raw, names)).add_prefix("raw_")
    qdf = pd.DataFrame(transitions(rare, names)).add_prefix("rare_")
    df = pd.concat([pd.DataFrame({"month": names, "n_seqs": n_seqs}),
                    rdf.drop(columns=["raw_month"]),
                    qdf.drop(columns=["rare_month"])], axis=1)
    df.to_csv(f"{args.out_dir}/70_vocab_table.csv", index=False)

    print("\n" + "=" * 96)
    print("RAW  (all sequences; partly measures how much sequencing happened)")
    print("=" * 96)
    print(df[["month", "n_seqs", "raw_vocab", "raw_stayed", "raw_dropped",
              "raw_new", "raw_first", "raw_jaccard"]]
          .to_string(index=False, na_rep="-"))

    print("\n" + "=" * 96)
    print(f"DEPTH-CONTROLLED  (every month subsampled to {args.depth} sequences)")
    print("=" * 96)
    print(df[["month", "rare_vocab", "rare_stayed", "rare_dropped",
              "rare_new", "rare_first", "rare_jaccard"]]
          .to_string(index=False, na_rep="-"))

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    for pref, lab in [("raw", "raw"), ("rare", f"at depth {args.depth}")]:
        s = df[[f"{pref}_vocab", f"{pref}_stayed", f"{pref}_dropped",
                f"{pref}_new", f"{pref}_first", f"{pref}_jaccard"]].dropna()
        if not len(s):
            continue
        print(f"\n{lab}:")
        print(f"  vocabulary size   mean {s[f'{pref}_vocab'].mean():6.1f}   "
              f"min {s[f'{pref}_vocab'].min():.0f}   "
              f"max {s[f'{pref}_vocab'].max():.0f}")
        print(f"  stayed per month  mean {s[f'{pref}_stayed'].mean():6.1f}")
        print(f"  dropped per month mean {s[f'{pref}_dropped'].mean():6.1f}")
        print(f"  new per month     mean {s[f'{pref}_new'].mean():6.1f}   "
              f"of which never seen before "
              f"{s[f'{pref}_first'].mean():.1f}")
        print(f"  Jaccard with previous month  mean "
              f"{s[f'{pref}_jaccard'].mean():.3f}")
        net = s[f"{pref}_new"].mean() - s[f"{pref}_dropped"].mean()
        print(f"  net change per month {net:+.1f}")

    def safe_corr(frame, col):
        d = frame[["n_seqs", col]].dropna()
        if len(d) < 3 or d["n_seqs"].std() == 0 or d[col].std() == 0:
            return np.nan          # undefined when either series is constant
        return float(np.corrcoef(d["n_seqs"], d[col])[0, 1])

    r_raw = safe_corr(df, "raw_vocab")
    r_rare = safe_corr(df, "rare_vocab")
    print(f"\ncorr(sequences collected, vocabulary size): "
          f"raw {r_raw:+.3f}   depth-controlled {r_rare:+.3f}")
    print("  the raw figure largely reflects sequencing effort; the")
    print("  depth-controlled one is the comparable series.")

    print(f"\nwrote outputs/70_vocab_table.csv")


if __name__ == "__main__":
    main()
