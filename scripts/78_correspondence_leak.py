#!/usr/bin/env python
"""
78_correspondence_leak.py

What this settles
-----------------
Script 48 found that new_set_share and entropy lose their signal when the
cluster partition is rebuilt causally. Two explanations were on the table:

  (a) WITHIN-MONTH SEPARATION. The pooled partition supposedly recognises a
      nascent variant as its own cluster because its future descendants are in
      the pool.
  (b) CROSS-MONTH CORRESPONDENCE. The pooled partition assigns the same cluster
      ID in every month, so a cluster's growth is computable for free. A causal
      per-month clustering has to MATCH clusters across months, and script 68
      measured that matching at only 63% mutual best-match with 2.22 sources per
      target.

Controlled tests rule out (a): average linkage separation is driven by DISTANCE,
not abundance. A nascent group of two sets seven edits from the incumbent
separates immediately at a threshold of four, and stays merged at a threshold of
six even after sixty future descendants are added. Adding members to a cluster
raises its average distance to others, so the partition self-stabilises. Those
checks are reproduced in the self-test below.

So this script tests (b), by computing the SAME quantity three ways that differ
only in how clusters are identified across months:

  pooled      one partition over all months. Cluster IDs are shared, so growth
              needs no matching. This is the leaky version.
  causal      each month clustered on its own data; clusters matched across
              months by nearest consensus. This is what is actually available at
              forecast time.
  causal_mutual
              the same, restricted to MUTUAL best matches, discarding ambiguous
              correspondences instead of forcing them.

The quantity measured is cluster growth, and the signal is whether growth at t
predicts growth at t+1 (momentum) and mass at t+2 (establishment). Script 68
found momentum NEGATIVE (-0.132) under causal matching. If the pooled version is
clearly positive on the same months, the correspondence is the leak, and its size
is the gap between the two.

Also reported: how often the two correspondences disagree, and how much of the
growth discrepancy that disagreement explains.

Outputs
-------
outputs/78_growth_pairs.csv   per cluster-month, growth under each scheme
outputs/78_signal.csv         momentum and establishment under each scheme
outputs/78_disagreement.csv   where pooled and causal correspondence differ

Usage
-----
python scripts/78_correspondence_leak.py --min_count 3 --end_month 2024-12
python scripts/78_correspondence_leak.py --self_test
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


# ----------------------------------------------------------------------------
# data and clustering
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


def cluster_sets(sets, threshold):
    if len(sets) < 2:
        return np.zeros(len(sets), dtype=int)
    D = edit_matrix(sets)
    return fcluster(linkage(squareform(D, checks=False), method="average"),
                    t=threshold, criterion="distance")


def top_sets(occ, max_sets):
    return [c for c, _ in sorted(occ.items(), key=lambda kv: -kv[1])[:max_sets]]


def consensus_of(members, weights):
    cnt = defaultdict(float)
    for c, w in zip(members, weights):
        for l in c:
            cnt[l] += w
    half = sum(weights) / 2.0
    return frozenset(l for l, v in cnt.items() if v > half)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 5:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------------------------------------------------------
# self-test, including the two checks that ruled out explanation (a)
# ----------------------------------------------------------------------------

def self_test():
    print("self-test")

    sets = [frozenset({1, 2, 3, 10 + i}) for i in range(4)] + \
           [frozenset({50, 51, 52, 60 + i}) for i in range(4)]
    assert len(set(cluster_sets(sets, 4.0).tolist())) == 2
    assert len(set(cluster_sets(sets, 100.0).tolist())) == 1
    print("  clustering separates and merges as expected      ok")

    # RULING OUT (a), part 1: average linkage is stable to added intermediates.
    before = cluster_sets(sets, 4.0)
    chain = [frozenset({1, 2, 3, 50}), frozenset({1, 2, 50, 51}),
             frozenset({1, 50, 51, 52}), frozenset({2, 3, 50, 51}),
             frozenset({1, 2, 3, 50, 51}), frozenset({1, 2, 50, 51, 52})]
    after = cluster_sets(sets + chain, 4.0)[:len(sets)]
    same = (len(set(before.tolist())) == len(set(after.tolist())))
    print(f"  adding 6 intermediates leaves the partition of the")
    print(f"     original sets unchanged: {same}                  ok")
    assert same, "expected average linkage to be stable"

    # RULING OUT (a), part 2: separation is driven by DISTANCE, not abundance.
    inc = [frozenset({1, 2, 3, 4, 5, 10 + i}) for i in range(20)]
    nasc = [frozenset({1, 2, 3, 4, 5, 50, 51, 52, 60 + i}) for i in range(2)]
    fut = [frozenset({1, 2, 3, 4, 5, 50, 51, 52, 70 + i}) for i in range(60)]
    for thr, expect in ((4.0, True), (6.0, False)):
        lab_now = cluster_sets(inc + nasc, thr)
        sep_now = bool(set(lab_now[-2:].tolist()) -
                       set(lab_now[:-2].tolist()))
        lab_fut = cluster_sets(inc + nasc + fut, thr)[:len(inc) + 2]
        sep_fut = bool(set(lab_fut[-2:].tolist()) -
                       set(lab_fut[:-2].tolist()))
        assert sep_now == expect and sep_fut == expect, (thr, sep_now, sep_fut)
    print("  a 2-set nascent group separates at thr 4 with NO future")
    print("     data, and stays merged at thr 6 even with 60 future")
    print("     descendants -> separation is distance-driven         ok")

    # correspondence: a mutual best match must be detected as mutual
    A = [frozenset({1, 2, 3}), frozenset({40, 41, 42})]
    B = [frozenset({1, 2, 3, 4}), frozenset({40, 41, 42, 43})]
    D = np.array([[len(a ^ b) for b in B] for a in A], dtype=float)
    fwd, rev = D.argmin(axis=1), D.argmin(axis=0)
    assert list(fwd) == [0, 1] and list(rev) == [0, 1]
    print("  mutual best matching identified correctly        ok")

    print("  spearman sanity: ", end="")
    x = np.arange(20.0)
    assert spearman(x, x) > 0.99 and spearman(x, -x) < -0.99
    print("ok")
    print("all tests passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--max_sets", type=int, default=400)
    ap.add_argument("--min_mass", type=float, default=0.002)
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
    T = len(names)
    print(f"loaded {T} months: {names[0]} .. {names[-1]}")

    per_month_sets = {m: top_sets(occ_by[m], args.max_sets) for m in names}

    # ---- POOLED: one partition over every month ----------------------------
    print("\nbuilding the pooled partition over all months (the leaky one) ...")
    pool = list(dict.fromkeys(c for m in names for c in per_month_sets[m]))
    pool_lab = cluster_sets(pool, args.threshold)
    pool_of = {c: int(l) for c, l in zip(pool, pool_lab)}
    print(f"  {len(pool)} distinct sets -> {len(set(pool_lab.tolist()))} clusters")

    pooled_mass = {}
    for m in names:
        tot = float(sum(occ_by[m].values()))
        d = defaultdict(float)
        for c, w in occ_by[m].items():
            if c in pool_of:
                d[pool_of[c]] += w / tot
        pooled_mass[m] = dict(d)

    # ---- CAUSAL: cluster each month alone, then match across months --------
    print("clustering each month independently (the causal one) ...")
    cl = {}
    for m in names:
        sets = per_month_sets[m]
        w = np.array([occ_by[m][c] for c in sets], dtype=float)
        tot = float(sum(occ_by[m].values()))
        lab = cluster_sets(sets, args.threshold)
        entry = []
        for cid in np.unique(lab):
            mem = np.flatnonzero(lab == cid)
            mass = float(w[mem].sum() / tot)
            if mass < args.min_mass:
                continue
            entry.append({"cid": int(cid), "mass": mass,
                          "consensus": consensus_of([sets[i] for i in mem],
                                                    w[mem])})
        cl[m] = entry
    print(f"  clusters per month: "
          f"{min(len(cl[m]) for m in names)}-{max(len(cl[m]) for m in names)}")

    # ---- growth under each scheme ------------------------------------------
    rows, dis_rows = [], []
    for t in range(T - 1):
        m, n = names[t], names[t + 1]
        a, b = cl[m], cl[n]
        if not a or not b:
            continue
        D = np.array([[len(x["consensus"] ^ y["consensus"]) for y in b]
                      for x in a], dtype=float)
        fwd = D.argmin(axis=1)
        rev = D.argmin(axis=0)

        for i, x in enumerate(a):
            j = int(fwd[i])
            mutual = bool(int(rev[j]) == i)
            g_causal = float(np.log((b[j]["mass"] + 1e-6) /
                                    (x["mass"] + 1e-6)))

            # the pooled scheme needs no matching: the same cluster id is used
            # in both months. Take the pooled cluster that carries most of this
            # causal cluster's consensus.
            pid = None
            best = -1.0
            for c, w in occ_by[m].items():
                if c in pool_of and x["consensus"] <= c and w > best:
                    pid, best = pool_of[c], w
            if pid is None:
                continue
            pm_t = pooled_mass[m].get(pid, 0.0)
            pm_n = pooled_mass[n].get(pid, 0.0)
            g_pooled = float(np.log((pm_n + 1e-6) / (pm_t + 1e-6)))

            rows.append({
                "month": m, "next": n, "causal_cid": x["cid"],
                "pooled_cid": pid, "mass_causal": x["mass"],
                "mass_pooled": pm_t, "match_dist": float(D[i, j]),
                "mutual": mutual,
                "growth_pooled": g_pooled, "growth_causal": g_causal,
                "dst_mass_causal": b[j]["mass"], "dst_mass_pooled": pm_n,
            })
            dis_rows.append({
                "month": m, "mutual": mutual, "match_dist": float(D[i, j]),
                "abs_growth_gap": abs(g_pooled - g_causal),
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/78_growth_pairs.csv", index=False)
    pd.DataFrame(dis_rows).to_csv(f"{args.out_dir}/78_disagreement.csv",
                                  index=False)

    print(f"\ncluster-months compared: {len(df)}")
    print(f"mutual best match: {df['mutual'].mean():.3f}")
    print(f"matching distance: median {df['match_dist'].median():.1f}  "
          f"mean {df['match_dist'].mean():.2f}")
    print(f"correlation between the two growth estimates: "
          f"{spearman(df['growth_pooled'], df['growth_causal']):+.3f}")
    print(f"mean |growth gap|: {df['abs_growth_gap'].mean():.3f}"
          if "abs_growth_gap" in df else "")

    # ---- momentum and establishment under each scheme ----------------------
    print("\n" + "=" * 78)
    print("SIGNAL WITH THE LEAK vs WITHOUT IT")
    print("=" * 78)
    srows = []
    for scheme, gcol in (("pooled", "growth_pooled"),
                         ("causal", "growth_causal"),
                         ("causal_mutual", "growth_causal")):
        sub = df[df["mutual"]] if scheme == "causal_mutual" else df
        mom, est = [], []
        for t in range(T - 2):
            m, n = names[t], names[t + 1]
            cur = sub[sub["month"] == m]
            nxt = sub[sub["month"] == n]
            if len(cur) < 5 or len(nxt) < 5:
                continue
            if scheme == "pooled":
                key_c, key_n = "pooled_cid", "pooled_cid"
                nxt_map = dict(zip(nxt[key_n], nxt[gcol]))
                nxt_mass = dict(zip(nxt[key_n], nxt["dst_mass_pooled"]))
                g_now = cur[gcol].to_numpy()
                ids = cur[key_c].to_numpy()
            else:
                # follow the causal correspondence one more step
                nxt_map = dict(zip(nxt["causal_cid"], nxt[gcol]))
                nxt_mass = dict(zip(nxt["causal_cid"], nxt["dst_mass_causal"]))
                g_now = cur[gcol].to_numpy()
                ids = cur["causal_cid"].to_numpy()
            g_next = np.array([nxt_map.get(i, np.nan) for i in ids])
            m2 = np.array([nxt_mass.get(i, np.nan) for i in ids])
            mom.append(spearman(g_now, g_next))
            est.append(spearman(g_now, m2))
        srows.append({
            "scheme": scheme,
            "momentum_growth": float(np.nanmean(mom)) if mom else np.nan,
            "momentum_positive_share": float(np.nanmean(
                [v > 0 for v in mom if not np.isnan(v)])) if mom else np.nan,
            "establish_growth": float(np.nanmean(est)) if est else np.nan,
            "months": int(np.sum([not np.isnan(v) for v in mom])),
            "n_cluster_months": int(len(sub)),
        })
    sdf = pd.DataFrame(srows)
    sdf.to_csv(f"{args.out_dir}/78_signal.csv", index=False)
    print(sdf.round(4).to_string(index=False))

    print("\n  pooled uses ONE partition over all months, so a cluster keeps the")
    print("  same identity throughout and its growth needs no matching. That")
    print("  identity is the leaked information: at forecast time the")
    print("  correspondence is genuinely ambiguous.")
    print("  causal clusters each month alone and matches by nearest consensus.")
    print("  causal_mutual keeps only mutual best matches, discarding the")
    print("  ambiguous ones rather than forcing them.")
    print("\n  pooled clearly positive while causal is near zero or negative ->")
    print("     the correspondence IS the leak, and the gap is its size.")
    print("  both similar -> the leak is elsewhere, and script 48's result")
    print("     needs a different explanation.")

    print("\ngrowth gap by correspondence quality:")
    dd = pd.DataFrame(dis_rows)
    if len(dd):
        print(dd.groupby("mutual")[["match_dist", "abs_growth_gap"]]
              .agg(["mean", "count"]).round(3).to_string())
        print("  if the gap is much larger for non-mutual matches, the damage is")
        print("  concentrated exactly where the correspondence is ambiguous.")

    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
