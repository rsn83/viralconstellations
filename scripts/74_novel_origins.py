#!/usr/bin/env python
"""
74_novel_origins.py

Question
--------
About 10 mutations per month have never been seen in ANY earlier month. They
carry only ~0.06% of sequence mass, so they are rare -- but every future variant
starts as one of them. WHICH BACKGROUND CARRIES THEM WHEN THEY FIRST APPEAR?

Three possibilities, and they say different things about mechanism:

  the dominant background
      novelty appears where the population is. Mutation is proportional to
      opportunity, and there is nothing to predict beyond abundance.
  a minority background
      novelty is concentrated in specific lineages -- a challenger, a diverging
      sublineage -- which would mean where you look matters.
  spread across many backgrounds
      the same mutation arising independently on different backgrounds, i.e.
      recurrence. That is evidence the mutation is favoured, since independent
      origins imply repeated selection rather than one lucky event.

Separating novel from returning
-------------------------------
Most "new" vocabulary is not new. Script 51 found 76.6% of labels drop out and
return, with a 29% one-month return rate, so a label absent at t and present at
t+1 was usually already there below the detection threshold. This script
therefore splits the two explicitly:

  FIRST-EVER  never present in any month up to t
  RETURNING   present at some earlier month, absent at t, back at t+1

and reports them separately. Only the first group is genuine novelty.

What is measured for each first-ever mutation
---------------------------------------------
  n_carriers        how many distinct constellations carry it in its first month
  carrier_mass      total sequence share of those constellations
  host_rank         population rank of its most common carrier (1 = dominant)
  host_share        that carrier's share of the month
  host_size         how many mutations the carrier already had
  dist_to_dominant  edit distance from the carrier to the month's modal set
  n_backgrounds     how many DISTINCT backgrounds carry it, after collapsing
                    carriers that are within `--near` edits of each other.
                    Greater than one is evidence of independent recurrence.

Baseline for comparison
-----------------------
The same statistics for RETURNING labels, and for a random sample of labels
already established. If first-ever mutations sit on the same backgrounds with
the same ranks as established ones, novelty is simply proportional to abundance.

Everything is causal: a label's novelty at month t is judged only against months
before t.

Outputs
-------
outputs/74_novel_events.csv    one row per first-ever mutation appearance
outputs/74_returning.csv       the same for returning labels
outputs/74_summary.csv         pooled comparison

Usage
-----
python scripts/74_novel_origins.py --min_count 3 --end_month 2024-12
python scripts/74_novel_origins.py --self_test
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

KNOWN = {
    "2021-01": "Alpha", "2021-06": "Delta", "2022-01": "BA.1",
    "2022-03": "BA.2", "2022-06": "BA.5", "2023-02": "XBB", "2023-12": "JN.1",
}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

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


def load_vocab(path):
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = {c.lower(): c for c in df.columns}
    idc = next((cols[c] for c in ("node_idx", "node", "id", "idx")
                if c in cols), None)
    pc = next((cols[c] for c in ("aa_pos", "pos", "position") if c in cols), None)
    rc = next((cols[c] for c in ("residue", "res", "aa") if c in cols), None)
    if pc is None or rc is None:
        return {}
    out = {}
    for i, row in enumerate(df.itertuples(index=False)):
        d = dict(zip(df.columns, row))
        key = int(d[idc]) if idc else i
        out[key] = f"{str(d[pc]).strip()}{str(d[rc]).strip()}"
    return out


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------

def collapse_backgrounds(carriers, near):
    """
    Group carrier constellations that are within `near` edits of each other,
    single-linkage. Two carriers 30 edits apart are different backgrounds; two
    that differ by one mutation are the same background sampled twice.
    Returns the number of distinct groups.
    """
    n = len(carriers)
    if n <= 1:
        return n
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if len(carriers[i] ^ carriers[j]) <= near:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    return len({find(i) for i in range(n)})


def describe(label, occ, near):
    """Statistics for one label in the month it appears."""
    tot = float(sum(occ.values()))
    items = sorted(occ.items(), key=lambda kv: -kv[1])
    ranks = {c: r + 1 for r, (c, _) in enumerate(items)}
    modal = items[0][0]

    carriers = [(c, w) for c, w in occ.items() if label in c]
    if not carriers:
        return None
    carriers.sort(key=lambda cw: -cw[1])
    host, host_w = carriers[0]
    return {
        "n_carriers": len(carriers),
        "carrier_mass": float(sum(w for _, w in carriers) / tot),
        "host_rank": ranks[host],
        "host_share": float(host_w / tot),
        "host_size": len(host),
        "dist_to_dominant": len(host ^ modal),
        "n_backgrounds": collapse_backgrounds([c for c, _ in carriers], near),
        "host_is_dominant": int(host == modal),
    }


def self_test():
    print("self-test")

    # two carriers one edit apart are ONE background; 30 apart are TWO
    a = frozenset({1, 2, 3, 4})
    b = frozenset({1, 2, 3, 4, 5})
    c = frozenset(range(100, 130))
    assert collapse_backgrounds([a, b], near=3) == 1
    assert collapse_backgrounds([a, c], near=3) == 2
    assert collapse_backgrounds([a, b, c], near=3) == 2
    print("  background collapsing groups near, splits far    ok")

    # single linkage must chain: a-b close, b-d close, a-d far -> still one
    d = frozenset({1, 2, 3, 4, 5, 6})
    assert collapse_backgrounds([a, b, d], near=1) == 1
    print("  single linkage chains through intermediates      ok")

    # describe must find the dominant carrier and rank it correctly
    occ = {frozenset({1, 2, 3}): 1000,          # dominant, no novel label
           frozenset({1, 2, 3, 99}): 50,        # rank 2, carries 99
           frozenset({7, 8, 9, 99}): 10}        # rank 3, also carries 99
    r = describe(99, occ, near=3)
    assert r["n_carriers"] == 2 and r["host_rank"] == 2
    assert r["host_is_dominant"] == 0
    assert r["n_backgrounds"] == 2, r
    assert r["dist_to_dominant"] == 1, r
    print("  describe finds host, rank and background count   ok")

    # a label on the dominant set must report rank 1
    occ2 = {frozenset({1, 2, 3, 99}): 1000, frozenset({1, 2, 3}): 10}
    r2 = describe(99, occ2, near=3)
    assert r2["host_rank"] == 1 and r2["host_is_dominant"] == 1
    print("  a label on the dominant set reports rank 1       ok")

    # absent label -> None, not a crash
    assert describe(12345, occ, near=3) is None
    print("  absent label returns None                        ok")
    print("all tests passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--near", type=int, default=3,
                    help="carriers within this many edits are one background")
    ap.add_argument("--max_carriers", type=int, default=400,
                    help="cap on carriers used for the background collapse")
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    names_map = load_vocab(args.vocab or
                           os.path.join(args.data_dir, "posres_vocab.tsv"))

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    print(f"loaded {len(names)} months: {names[0]} .. {names[-1]}")

    vocab = {}
    for m in names:
        v = set()
        for cs in occ_by[m]:
            v |= set(cs)
        vocab[m] = v

    novel, returning, established = [], [], []
    ever = set()
    prev = None
    for t, m in enumerate(names):
        occ = occ_by[m]
        v = vocab[m]
        first_ever = v - ever
        ret = (v - prev) & ever if prev is not None else set()
        est = v & prev if prev is not None else set()

        for lab in sorted(first_ever, key=str):
            r = describe(lab, occ, args.near)
            if r:
                novel.append({"month": m, "label": names_map.get(lab, str(lab)),
                              "kind": "first_ever",
                              "variant_note": KNOWN.get(m, ""), **r})
        for lab in sorted(ret, key=str):
            r = describe(lab, occ, args.near)
            if r:
                returning.append({"month": m,
                                  "label": names_map.get(lab, str(lab)),
                                  "kind": "returning", **r})
        # a size-matched random sample of established labels, as the reference
        if est:
            k = min(len(first_ever) if first_ever else 5, len(est))
            for lab in rng.choice(sorted(est, key=str), size=k, replace=False):
                r = describe(lab, occ, args.near)
                if r:
                    established.append({"month": m,
                                        "label": names_map.get(lab, str(lab)),
                                        "kind": "established", **r})
        ever |= v
        prev = v

    ndf = pd.DataFrame(novel)
    rdf = pd.DataFrame(returning)
    edf = pd.DataFrame(established)
    ndf.to_csv(f"{args.out_dir}/74_novel_events.csv", index=False)
    rdf.to_csv(f"{args.out_dir}/74_returning.csv", index=False)

    print(f"\nfirst-ever appearances : {len(ndf)}  "
          f"({len(ndf)/max(len(names)-1,1):.1f} per month)")
    print(f"returning appearances  : {len(rdf)}  "
          f"({len(rdf)/max(len(names)-1,1):.1f} per month)")
    print("  most 'new' vocabulary is returning, not novel -- 76.6% of labels")
    print("  drop out and come back (script 51). Only the first group is new.")

    cols = ["n_carriers", "carrier_mass", "host_rank", "host_share",
            "host_size", "dist_to_dominant", "n_backgrounds",
            "host_is_dominant"]
    print("\n" + "=" * 88)
    print("WHERE NOVELTY APPEARS  (medians, with the two reference groups)")
    print("=" * 88)
    comp = []
    for lab, d in (("first_ever", ndf), ("returning", rdf),
                   ("established", edf)):
        if not len(d):
            continue
        row = {"kind": lab, "n": len(d)}
        for c in cols:
            row[f"med_{c}"] = float(d[c].median())
        row["mean_host_is_dominant"] = float(d["host_is_dominant"].mean())
        row["share_multi_background"] = float((d["n_backgrounds"] > 1).mean())
        comp.append(row)
    cdf = pd.DataFrame(comp)
    cdf.to_csv(f"{args.out_dir}/74_summary.csv", index=False)
    print(cdf.round(4).to_string(index=False))

    if len(ndf):
        print("\n--- reading the first-ever row ---")
        print(f"  host_rank median {ndf['host_rank'].median():.0f} of "
              f"{np.mean([len(occ_by[m]) for m in names]):.0f} sets per month")
        print(f"  appears on the dominant set: "
              f"{ndf['host_is_dominant'].mean():.3f} of the time")
        print(f"  on more than one background: "
              f"{(ndf['n_backgrounds'] > 1).mean():.3f} of the time")
        print(f"  distance from carrier to the month's modal set: median "
              f"{ndf['dist_to_dominant'].median():.0f}")
        print("\n  host_rank near 1 and host_is_dominant high -> novelty appears")
        print("     where the population already is, and there is nothing to")
        print("     predict beyond abundance.")
        print("  host_rank large and dist_to_dominant large -> novelty is")
        print("     concentrated in minority backgrounds, so where you look")
        print("     matters.")
        print("  share_multi_background high -> the same mutation arises")
        print("     independently on separate backgrounds, which implies")
        print("     repeated selection rather than one lucky event.")
        print("  compare every figure against the 'established' row: if they")
        print("     match, novelty is simply proportional to abundance.")

        print("\n--- first-ever mutations by month (most recent 20) ---")
        g = ndf.groupby("month").agg(
            n=("label", "count"),
            labels=("label", lambda s: ", ".join(list(s)[:8])),
            med_host_rank=("host_rank", "median"),
            on_dominant=("host_is_dominant", "mean"),
        ).reset_index().tail(20)
        print(g.round(3).to_string(index=False))

    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
