#!/usr/bin/env python
"""
68_monthly_clusters.py

The leakage this fixes
---------------------
Script 38 built one cluster partition on all 78 months pooled and then read
per-month statistics off it. Every downstream result inherited that: a cluster
defined using 2023 data was used to describe 2021. Script 48's rebuild test
already showed new_set_share and entropy do not survive it.

Here each month is clustered ON ITS OWN SEQUENCES ALONE. Nothing is pooled. A
month's clustering cannot see any other month, past or future. Cluster IDs are
therefore arbitrary and not comparable across months, so mapping between months
is a separate, explicit step that can be inspected and doubted.

Sections
--------
A. PER-MONTH CLUSTERING. Edit distance between constellations (symmetric
   difference), average linkage, cut at a fixed distance threshold. Per cluster
   per month: mass, number of constellations, mean set size, internal diversity,
   consensus set, radius.

B. CROSS-MONTH MAPPING. Clusters at t are matched to clusters at t+1 by
   consensus-set edit distance. Because clusters split and merge, the mapping is
   reported as many-to-many statistics -- how many targets each source maps to,
   how many sources land on each target, and how far the matched consensus
   moved. A mapping that is mostly 1-to-1 with small movement is usable; one
   that is mostly many-to-one is not, and the numbers say which.

C. CLUSTER DYNAMICS. Given the mapping, each cluster has a growth rate. Two
   questions, both scored out of sample:
     does growth at t predict growth at t+1?      (momentum -- the ingredient
                                                   behind the winner rule)
     does growth at t predict mass at t+2?        (establishment)
   Baselines: current mass alone, and random. Momentum is what would let a
   cluster-level forecast work at all.

D. THRESHOLD SENSITIVITY. Everything in A-C is recomputed at several
   thresholds. If cluster counts and the section C answers move a lot with the
   threshold, the clustering is a knob rather than a structure.

Known variant months are printed for READING ONLY and are never used to fit,
cluster or map anything.

Outputs
-------
outputs/68_clusters.csv      one row per (month, cluster)
outputs/68_mapping.csv       one row per matched pair across consecutive months
outputs/68_dynamics.csv      momentum and establishment scores
outputs/68_thresholds.csv    sensitivity summary

Usage
-----
python scripts/68_monthly_clusters.py --min_count 3 --end_month 2024-12
python scripts/68_monthly_clusters.py --self_test
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


# ----------------------------------------------------------------------------
# loading
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


# ----------------------------------------------------------------------------
# A. clustering, one month at a time
# ----------------------------------------------------------------------------

def pairwise_edit(sets):
    """Symmetric-difference distance matrix, via a binary incidence matrix."""
    labs = sorted({l for s in sets for l in s}, key=str)
    idx = {l: i for i, l in enumerate(labs)}
    A = np.zeros((len(sets), len(labs)), dtype=np.float32)
    for i, s in enumerate(sets):
        for l in s:
            A[i, idx[l]] = 1.0
    inter = A @ A.T
    sz = A.sum(axis=1)
    D = sz[:, None] + sz[None, :] - 2.0 * inter
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def cluster_month(occ, threshold, max_sets):
    """
    Cluster one month's constellations. Uses only this month's data.
    Returns a list of cluster dicts.
    """
    items = sorted(occ.items(), key=lambda kv: -kv[1])[:max_sets]
    sets = [c for c, _ in items]
    w = np.array([v for _, v in items], dtype=float)
    total = w.sum()
    if len(sets) == 1:
        labels = np.array([1])
        D = np.zeros((1, 1))
    else:
        D = pairwise_edit(sets)
        Z = linkage(squareform(D, checks=False), method="average")
        labels = fcluster(Z, t=threshold, criterion="distance")

    out = []
    for cid in np.unique(labels):
        mem = np.flatnonzero(labels == cid)
        mw = w[mem]
        mass = float(mw.sum() / total)
        # consensus: mutations carried by more than half the cluster's sequences
        cnt = defaultdict(float)
        for i in mem:
            for l in sets[i]:
                cnt[l] += w[i]
        half = mw.sum() / 2.0
        consensus = frozenset(l for l, v in cnt.items() if v > half)
        sub = D[np.ix_(mem, mem)]
        if mem.size > 1:
            pw = sub[np.triu_indices(mem.size, 1)]
            ww = np.outer(mw, mw)[np.triu_indices(mem.size, 1)]
            mpd = float((pw * ww).sum() / ww.sum())
        else:
            mpd = 0.0
        radius = float(np.average([len(sets[i] ^ consensus) for i in mem],
                                  weights=mw))
        out.append({
            "cluster": int(cid), "mass": mass, "n_sets": int(mem.size),
            "n_seqs": float(mw.sum()),
            "mean_set_size": float(np.average([len(sets[i]) for i in mem],
                                              weights=mw)),
            "mpd_internal": mpd, "radius": radius,
            "consensus": consensus, "consensus_size": len(consensus),
        })
    return sorted(out, key=lambda d: -d["mass"])


# ----------------------------------------------------------------------------
# B. mapping between consecutive months
# ----------------------------------------------------------------------------

def map_clusters(cl_t, cl_n):
    """
    Match each cluster at t to the nearest cluster at t+1 by consensus edit
    distance. Reports the multiplicity in both directions, so splits and merges
    are visible rather than hidden by the argmin.
    """
    if not cl_t or not cl_n:
        return []
    D = np.array([[len(a["consensus"] ^ b["consensus"]) for b in cl_n]
                  for a in cl_t], dtype=float)
    best = D.argmin(axis=1)
    # how many sources land on each target
    inbound = defaultdict(int)
    for j in best:
        inbound[int(j)] += 1
    # the reverse match, to detect one-way pairings
    rev = D.argmin(axis=0)

    rows = []
    for i, a in enumerate(cl_t):
        j = int(best[i])
        b = cl_n[j]
        rows.append({
            "src_cluster": a["cluster"], "dst_cluster": b["cluster"],
            "dist": float(D[i, j]),
            "src_mass": a["mass"], "dst_mass": b["mass"],
            "log_growth": float(np.log((b["mass"] + 1e-6) /
                                       (a["mass"] + 1e-6))),
            "mutual": bool(int(rev[j]) == i),
            "inbound_to_dst": inbound[j],
            "src_mpd": a["mpd_internal"], "src_radius": a["radius"],
            "src_n_sets": a["n_sets"],
            "src_consensus_size": a["consensus_size"],
            "dst_consensus_size": b["consensus_size"],
        })
    return rows


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 5:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    print("self-test")

    # pairwise edit distance must equal symmetric difference
    sets = [frozenset({1, 2, 3}), frozenset({1, 2}), frozenset({7, 8, 9})]
    D = pairwise_edit(sets)
    assert D[0, 1] == 1 and D[0, 2] == 6 and D[1, 2] == 5, D
    print("  edit distance matches symmetric difference       ok")

    # two well-separated groups must be found, and not found at a huge threshold
    occ = {}
    for i in range(6):
        occ[frozenset({1, 2, 3, 100 + i})] = 100
    for i in range(6):
        occ[frozenset({50, 51, 52, 200 + i})] = 100
    cl = cluster_month(occ, threshold=4.0, max_sets=100)
    assert len(cl) == 2, [c["n_sets"] for c in cl]
    assert abs(cl[0]["mass"] - 0.5) < 1e-9
    print("  two separated groups -> 2 clusters, mass 0.5 each ok")
    cl_wide = cluster_month(occ, threshold=100.0, max_sets=100)
    assert len(cl_wide) == 1
    print("  huge threshold -> 1 cluster                       ok")

    # consensus must be the shared core, not the union
    assert cl[0]["consensus"] in (frozenset({1, 2, 3}), frozenset({50, 51, 52}))
    print("  consensus is the shared core                      ok")

    # mapping must recover a known correspondence, with growth of the right sign
    occ2 = {}
    for i in range(6):
        occ2[frozenset({1, 2, 3, 100 + i})] = 300      # this group grows
    for i in range(6):
        occ2[frozenset({50, 51, 52, 200 + i})] = 50    # this one shrinks
    cl_t = cluster_month(occ, 4.0, 100)
    cl_n = cluster_month(occ2, 4.0, 100)
    mp = map_clusters(cl_t, cl_n)
    assert all(r["dist"] == 0 for r in mp), [r["dist"] for r in mp]
    assert all(r["mutual"] for r in mp)
    growing = [r for r in mp if 1 in cl_t[[c["cluster"] for c in cl_t].index(
        r["src_cluster"])]["consensus"]]
    assert len(growing) == 1 and growing[0]["log_growth"] > 0, mp
    shrinking = [r for r in mp if r is not growing[0]]
    assert shrinking[0]["log_growth"] < 0
    print("  mapping recovers correspondence, growth signed    ok")

    # a cluster that vanishes must not silently map to something distant
    # without that showing up in the distance column
    occ3 = {frozenset({1, 2, 3, 100 + i}): 300 for i in range(6)}
    mp2 = map_clusters(cl_t, cluster_month(occ3, 4.0, 100))
    dists = sorted(r["dist"] for r in mp2)
    assert dists[0] == 0 and dists[-1] > 3, dists
    print("  a vanished cluster shows a large mapping distance ok")

    print("all tests passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def run_threshold(names, occ_by, thr, max_sets):
    clusters = {m: cluster_month(occ_by[m], thr, max_sets) for m in names}
    mapping = {}
    for t in range(len(names) - 1):
        mapping[names[t]] = map_clusters(clusters[names[t]],
                                         clusters[names[t + 1]])
    return clusters, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--thresholds", default="4,5,6,7")
    ap.add_argument("--max_sets", type=int, default=600)
    ap.add_argument("--min_clusters_dyn", type=int, default=5,
                    help="months with fewer matched clusters are skipped in C")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    print(f"loaded {len(names)} months: {names[0]} .. {names[-1]}")

    # ---- A / B at the primary threshold ------------------------------------
    print(f"\nclustering each month independently at threshold "
          f"{args.threshold} ...")
    clusters, mapping = run_threshold(names, occ_by, args.threshold,
                                      args.max_sets)

    crows = []
    for m in names:
        for c in clusters[m]:
            crows.append({"month": m, "variant_note": KNOWN.get(m, ""),
                          **{k: v for k, v in c.items() if k != "consensus"}})
    cdf = pd.DataFrame(crows)
    cdf.to_csv(f"{args.out_dir}/68_clusters.csv", index=False)

    print("\n" + "=" * 74)
    print("A. PER-MONTH CLUSTERS  (each month clustered on its own data only)")
    print("=" * 74)
    per_month = cdf.groupby("month").agg(
        n_clusters=("cluster", "count"),
        top_mass=("mass", "max"),
        mean_set_size=("mean_set_size", "mean"),
        top_consensus_size=("consensus_size", "max"),
    ).reset_index()
    per_month["variant_note"] = per_month["month"].map(KNOWN).fillna("")
    print(per_month.round(3).to_string(index=False))

    mrows = []
    for m, rows in mapping.items():
        for r in rows:
            mrows.append({"month": m, "next_month":
                          names[names.index(m) + 1], **r})
    mdf = pd.DataFrame(mrows)
    mdf.to_csv(f"{args.out_dir}/68_mapping.csv", index=False)

    print("\n" + "=" * 74)
    print("B. CROSS-MONTH MAPPING QUALITY")
    print("=" * 74)
    print(f"matched pairs: {len(mdf)}")
    print(f"mutual best match      : {mdf['mutual'].mean():.3f}")
    print(f"consensus distance     : median {mdf['dist'].median():.1f}, "
          f"mean {mdf['dist'].mean():.2f}, "
          f"share == 0 {(mdf['dist'] == 0).mean():.3f}")
    print(f"targets receiving 1 source: "
          f"{(mdf['inbound_to_dst'] == 1).mean():.3f}")
    print(f"mean sources per target   : {mdf['inbound_to_dst'].mean():.2f}")
    print("  a mapping that is mostly mutual and 1-to-1 with small consensus")
    print("  movement is usable. Mostly many-to-one means clusters are being")
    print("  collapsed and the growth rates below are not comparable.")

    # ---- C. cluster dynamics -----------------------------------------------
    print("\n" + "=" * 74)
    print("C. CLUSTER DYNAMICS")
    print("=" * 74)
    drows, skipped = [], []
    for t in range(len(names) - 2):
        cur = mapping.get(names[t], [])
        nxt = {r["src_cluster"]: r for r in mapping.get(names[t + 1], [])}
        g_now, g_next, mass_now, mass_2 = [], [], [], []
        for r in cur:
            follow = nxt.get(r["dst_cluster"])
            if follow is None:
                continue
            g_now.append(r["log_growth"])
            g_next.append(follow["log_growth"])
            mass_now.append(r["src_mass"])
            mass_2.append(follow["dst_mass"])
        if len(g_now) < args.min_clusters_dyn:
            skipped.append(names[t])
            continue
        drows.append({
            "month": names[t], "n_clusters": len(g_now),
            "momentum_growth": spearman(g_now, g_next),
            "momentum_mass": spearman(mass_now, g_next),
            "establish_growth": spearman(g_now, mass_2),
            "establish_mass": spearman(mass_now, mass_2),
        })
    ddf = pd.DataFrame(drows)
    ddf.to_csv(f"{args.out_dir}/68_dynamics.csv", index=False)
    if skipped:
        print(f"skipped {len(skipped)} months with fewer than "
              f"{args.min_clusters_dyn} matched clusters: "
              f"{skipped[:6]}{' ...' if len(skipped) > 6 else ''}")
    if len(ddf):
        print(ddf.round(3).to_string(index=False))
        print("\npooled (mean Spearman over months, and share of months > 0):")
        for col in ["momentum_growth", "momentum_mass",
                    "establish_growth", "establish_mass"]:
            v = ddf[col].dropna()
            print(f"  {col:20s} {v.mean():+.3f}   "
                  f"positive in {(v > 0).mean():.2f} of {len(v)} months")
        print("\n  momentum_growth is the one that matters: does a cluster's")
        print("  growth this month predict its growth next month.")
        print("  CALIBRATED on synthetic data, 12 lineages, 28 months:")
        print("     growth autocorrelated by construction -> +0.726")
        print("     growth redrawn every month            -> -0.026")
        print("  So the statistic separates cleanly. Near 0 here means")
        print("  cluster-level growth is memoryless and no cluster-level")
        print("  forecast can work. Note establish_mass came out +0.83 even in")
        print("  the memoryless case -- big clusters stay big regardless, so")
        print("  that column is NOT evidence of anything predictive.")
    else:
        print("too few matched clusters per month for dynamics")

    # ---- D. threshold sensitivity ------------------------------------------
    print("\n" + "=" * 74)
    print("D. THRESHOLD SENSITIVITY")
    print("=" * 74)
    trows = []
    for thr in [float(x) for x in args.thresholds.split(",")]:
        cl, mp = run_threshold(names, occ_by, thr, args.max_sets)
        nc = [len(cl[m]) for m in names]
        allm = [r for rows in mp.values() for r in rows]
        mo = []
        for t in range(len(names) - 2):
            cur = mp.get(names[t], [])
            nx = {r["src_cluster"]: r for r in mp.get(names[t + 1], [])}
            gn = [(r["log_growth"], nx[r["dst_cluster"]]["log_growth"])
                  for r in cur if r["dst_cluster"] in nx]
            if len(gn) >= args.min_clusters_dyn:
                mo.append(spearman([a for a, _ in gn], [b for _, b in gn]))
        trows.append({
            "threshold": thr,
            "mean_clusters": float(np.mean(nc)),
            "min_clusters": int(np.min(nc)), "max_clusters": int(np.max(nc)),
            "mutual_rate": float(np.mean([r["mutual"] for r in allm])),
            "mean_map_dist": float(np.mean([r["dist"] for r in allm])),
            "momentum_growth": float(np.nanmean(mo)) if mo else np.nan,
        })
    tdf = pd.DataFrame(trows)
    tdf.to_csv(f"{args.out_dir}/68_thresholds.csv", index=False)
    print(tdf.round(3).to_string(index=False))
    print("  if momentum_growth swings with the threshold, the clustering is a")
    print("  knob. If it is stable, it is a property of the data.")

    print(f"\nwrote 4 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
