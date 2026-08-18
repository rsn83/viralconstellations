#!/usr/bin/env python
"""
85_cluster_description.py

A post-hoc description of the monthly cluster structure. Nothing here is a
forecast and nothing is scored against a baseline.

What is claimed
---------------
Only that grouping each month's constellations by mutation-set similarity,
using that month's sequences alone, produces groups whose sizes and frequencies
line up with the known variant history -- and that the diversity of those groups
moves in a characteristic way around a transition.

What is NOT claimed
-------------------
That this predicts anything. Script 78 showed that following a cluster from one
month to the next is ambiguous: about a quarter of month-to-month best matches
are not mutual, and where they are ambiguous two defensible matches imply growth
rates differing by a factor of twenty. So nothing here tracks a cluster over
time or reports its growth.

Known variant months are printed for reading only. They are never used to
cluster, to choose a threshold, or to compute anything.

Each month is clustered on its own sequences alone, so no month can see any
other month, past or future.

Reported per month
------------------
  number of clusters
  the largest cluster's share of sequences
  the number of clusters needed to cover half the sequences
  Shannon entropy over the cluster frequencies, in bits
  the mean pairwise edit distance between two random sequences
  the largest cluster's size, internal spread and consensus

Reported per cluster per month, in the CSV
------------------------------------------
  share of sequences, number of constellations, mean set size,
  internal mean pairwise distance, consensus size

Sensitivity
-----------
Everything is recomputed at four thresholds so it is visible which statements
depend on the choice and which do not.

Usage
-----
python scripts/85_cluster_description.py --min_count 3 --end_month 2024-12
python scripts/85_cluster_description.py --self_test
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

KNOWN = {
    "2021-01": "Alpha", "2021-06": "Delta", "2022-01": "BA.1",
    "2022-03": "BA.2", "2022-06": "BA.5", "2023-02": "XBB", "2023-12": "JN.1",
}


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


def edit_matrix(sets):
    labs = sorted({l for s in sets for l in s}, key=str)
    idx = {l: i for i, l in enumerate(labs)}
    A = np.zeros((len(sets), len(labs)), dtype=np.float32)
    for i, s in enumerate(sets):
        for l in s:
            A[i, idx[l]] = 1.0
    sz = A.sum(1)
    D = np.maximum(sz[:, None] + sz[None, :] - 2.0 * (A @ A.T), 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def describe_month(occ, threshold, max_sets, rng, n_pairs=4000):
    items = sorted(occ.items(), key=lambda kv: -kv[1])[:max_sets]
    sets = [c for c, _ in items]
    w = np.array([v for _, v in items], dtype=float)
    total = w.sum()
    p_sets = w / total

    if len(sets) < 2:
        D = np.zeros((1, 1))
        labels = np.array([1])
    else:
        D = edit_matrix(sets)
        labels = fcluster(linkage(squareform(D, checks=False),
                                  method="average"),
                          t=threshold, criterion="distance")

    # mean pairwise distance between two randomly drawn sequences
    if len(sets) > 1:
        i = rng.choice(len(sets), size=n_pairs, p=p_sets)
        j = rng.choice(len(sets), size=n_pairs, p=p_sets)
        mpd = float(D[i, j].mean())
    else:
        mpd = 0.0

    clusters = []
    for cid in np.unique(labels):
        mem = np.flatnonzero(labels == cid)
        mw = w[mem]
        share = float(mw.sum() / total)
        cnt = defaultdict(float)
        for k in mem:
            for l in sets[k]:
                cnt[l] += w[k]
        half = mw.sum() / 2.0
        consensus = frozenset(l for l, v in cnt.items() if v > half)
        if mem.size > 1:
            sub = D[np.ix_(mem, mem)]
            iu = np.triu_indices(mem.size, 1)
            ww = np.outer(mw, mw)[iu]
            internal = float((sub[iu] * ww).sum() / ww.sum())
        else:
            internal = 0.0
        clusters.append({
            "share": share,
            "n_constellations": int(mem.size),
            "mean_set_size": float(np.average([len(sets[k]) for k in mem],
                                              weights=mw)),
            "internal_spread": internal,
            "consensus_size": len(consensus),
        })
    clusters.sort(key=lambda d: -d["share"])

    shares = np.array([c["share"] for c in clusters])
    cum = np.cumsum(shares)
    return {
        "n_clusters": len(clusters),
        "largest_share": float(shares[0]),
        "clusters_to_cover_half": int(np.searchsorted(cum, 0.5) + 1),
        "entropy_bits": entropy_bits(shares),
        "mean_pairwise_distance": mpd,
        "largest_set_size": clusters[0]["mean_set_size"],
        "largest_internal_spread": clusters[0]["internal_spread"],
        "largest_consensus_size": clusters[0]["consensus_size"],
        "clusters": clusters,
    }


def self_test():
    print("checking the description")
    rng = np.random.default_rng(0)

    # two clearly separated groups of equal weight
    occ = {}
    for i in range(6):
        occ[frozenset({1, 2, 3, 10 + i})] = 100
    for i in range(6):
        occ[frozenset({50, 51, 52, 60 + i})] = 100
    d = describe_month(occ, 4.0, 100, rng)
    assert d["n_clusters"] == 2
    assert abs(d["largest_share"] - 0.5) < 1e-9
    assert abs(d["entropy_bits"] - 1.0) < 1e-9
    print("  two equal groups: 2 clusters, share 0.5, entropy 1 bit  ok")

    # one dominant group: entropy near zero, one cluster covers half
    occ2 = dict(occ)
    for i in range(6):
        occ2[frozenset({1, 2, 3, 10 + i})] = 2000
    d2 = describe_month(occ2, 4.0, 100, rng)
    assert d2["largest_share"] > 0.9
    assert d2["entropy_bits"] < 0.6
    assert d2["clusters_to_cover_half"] == 1
    print(f"  one dominant group: share {d2['largest_share']:.2f}, "
          f"entropy {d2['entropy_bits']:.2f} bits        ok")

    # mean pairwise distance is larger when two distant groups coexist
    assert d["mean_pairwise_distance"] > d2["mean_pairwise_distance"]
    print(f"  two coexisting groups give a larger mean pairwise")
    print(f"     distance ({d['mean_pairwise_distance']:.1f} vs "
          f"{d2['mean_pairwise_distance']:.1f})                       ok")

    # a wide threshold merges everything
    d3 = describe_month(occ, 100.0, 100, rng)
    assert d3["n_clusters"] == 1 and d3["entropy_bits"] == 0.0
    print("  a wide threshold merges everything                      ok")

    # entropy of four equal groups is 2 bits
    assert abs(entropy_bits([0.25] * 4) - 2.0) < 1e-12
    print("  entropy: four equal groups -> 2 bits                    ok")
    print("all checks passed\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--n_per_month", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--thresholds", default="4,5,6,7")
    ap.add_argument("--max_sets", type=int, default=600)
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
    print(f"months used: {len(kept)} ({kept[0][0]} to {kept[-1][0]}), "
          f"each sampled to {args.n_per_month:,} sequences")
    print(f"clustering threshold: {args.threshold} edits, average linkage,")
    print("each month clustered on its own sequences alone\n")

    rows, per_cluster = [], []
    for m, s in kept:
        d = describe_month(s, args.threshold, args.max_sets, rng)
        rows.append({
            "month": m,
            "variant_note": KNOWN.get(m, ""),
            "n_clusters": d["n_clusters"],
            "largest_share": d["largest_share"],
            "clusters_to_cover_half": d["clusters_to_cover_half"],
            "entropy_bits": d["entropy_bits"],
            "mean_pairwise_distance": d["mean_pairwise_distance"],
            "largest_set_size": d["largest_set_size"],
            "largest_internal_spread": d["largest_internal_spread"],
            "largest_consensus_size": d["largest_consensus_size"],
        })
        for r, c in enumerate(d["clusters"]):
            per_cluster.append({"month": m, "rank": r + 1, **c})

    df = pd.DataFrame(rows)
    dc = pd.DataFrame(per_cluster)
    df.to_csv(f"{args.out_dir}/85_monthly.csv", index=False)
    dc.to_csv(f"{args.out_dir}/85_per_cluster.csv", index=False)

    print("=" * 118)
    print("MONTHLY CLUSTER STRUCTURE")
    print("=" * 118)
    print(df.round(3).to_string(index=False))

    print("\n" + "=" * 118)
    print("THE FIVE LARGEST CLUSTERS, EVERY THIRD MONTH")
    print("=" * 118)
    sel = dc[dc["month"].isin(sorted(dc["month"].unique())[::3])]
    sel = sel[sel["rank"] <= 5]
    print(sel.round(3).to_string(index=False))

    print("\n" + "=" * 118)
    print("MONTHS AROUND EACH KNOWN VARIANT")
    print("=" * 118)
    idx = {m: i for i, m in enumerate(df["month"])}
    for vm, vn in KNOWN.items():
        if vm not in idx:
            continue
        lo, hi = max(0, idx[vm] - 3), min(len(df), idx[vm] + 4)
        w = df.iloc[lo:hi][["month", "n_clusters", "largest_share",
                            "entropy_bits", "mean_pairwise_distance",
                            "largest_set_size"]]
        print(f"\n{vn} ({vm}):")
        print(w.round(3).to_string(index=False))

    print("\n" + "=" * 118)
    print("SENSITIVITY TO THE THRESHOLD")
    print("=" * 118)
    trows = []
    for thr in [float(x) for x in args.thresholds.split(",")]:
        vals = [describe_month(s, thr, args.max_sets, rng) for _, s in kept]
        nc = np.array([v["n_clusters"] for v in vals], dtype=float)
        en = np.array([v["entropy_bits"] for v in vals])
        ls = np.array([v["largest_share"] for v in vals])
        trows.append({
            "threshold": thr,
            "mean_clusters": nc.mean(),
            "min_clusters": int(nc.min()), "max_clusters": int(nc.max()),
            "mean_entropy_bits": en.mean(),
            "mean_largest_share": ls.mean(),
        })
    print(pd.DataFrame(trows).round(3).to_string(index=False))

    print("\nnote: cluster counts and entropies shift with the threshold, so any")
    print("statement about their absolute value depends on that choice. Whether")
    print("they rise or fall around a given month does not.")
    print("\nreminder: nothing here follows a cluster from one month to the next.")
    print("Script 78 measured that correspondence and found it ambiguous in")
    print("about a quarter of months, with growth rates differing twentyfold")
    print("between two defensible matches.")

    print(f"\nwrote 2 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
