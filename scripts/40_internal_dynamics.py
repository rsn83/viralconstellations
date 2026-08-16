#!/usr/bin/env python
"""
40_internal_dynamics.py

Does a cluster's INTERNAL behaviour predict its EXTERNAL growth?

THE QUESTION
------------
A cluster is a group of near-identical constellations. Internally it can be
static (the same few sets, unchanged) or diversifying (gaining constellations,
gaining mutations, throwing off new variants of itself). The hypothesis is that
a diversifying cluster is one on the way up, and a static one is not.

WHY THIS AND NOT SCRIPT 39
--------------------------
Script 39 asked which clusters RISE -- crossing a frequency threshold -- and
found only 9 distinct risers in four years, most test months carrying exactly
one positive. At that sample size an AUC is a single rank and the per-month
figures swung 0.227 to 1.000 on noise. Its headline was also narrow: recent
growth alone scored 0.839 and a twelve-feature model 0.8399, a gain of 0.0009.
Internal features were among those twelve and added nothing.

But that tested the rare OUTCOME on ~20 events. This tests the MECHANISM on
every cluster-month -- thousands of observations -- by regressing next month's
frequency growth on this month's internal state. If diversification matters, it
should show here even though it did not show there.

WHAT IS MEASURED
----------------
For every (cluster, month) with enough sequences:

  internal, at month t
    n_sets           distinct constellations in the cluster
    d_n_sets         log change in that count since t-1
    mean_size        mean number of mutations per constellation
    d_mean_size      change in mean size since t-1
    max_size         largest constellation
    set_entropy      Shannon entropy over the within-cluster count
                     distribution -- high means mass spread across many
                     sublineages, low means one dominant sequence
    new_set_share    fraction of the cluster's sequences sitting in
                     constellations first seen this month
    mean_pair_dist   mean edit distance between the cluster's constellations,
                     i.e. how spread out it is

  external, the target
    g_next           log frequency change from t to t+1

CONTROLS THAT MATTER
--------------------
1. Current growth (g_prev) is the baseline. Growth is autocorrelated, and any
   internal feature correlated with growth will look predictive unless g_prev
   is partialled out. The script reports raw correlation AND correlation after
   regressing out g_prev and current frequency.
2. Sequencing depth inflates n_sets directly -- deeper months resolve more
   sublineages. Depth is therefore included as a control, and the per-era
   breakdown shows whether any effect is confined to the dense period.
3. Cluster identity: a few large clusters dominate the cluster-months, so
   correlations are also reported within-cluster (demeaned per cluster).

Usage
-----
  python scripts/40_internal_dynamics.py
  python scripts/40_internal_dynamics.py --thresh 4 --min_cluster_seqs 100
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(m):
    print(m, flush=True)


def constellations_of(occ):
    out = {}
    for c, v in occ.items():
        out[frozenset(c)] = v if isinstance(v, (int, float)) else 1
    return out


def cluster_sets(sets, thresh, metric, block=400):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    nodes = sorted({m for s in sets for m in s})
    idx = {m: i for i, m in enumerate(nodes)}
    M = np.zeros((len(sets), len(nodes)), dtype=np.float32)
    for i, s in enumerate(sets):
        for m in s:
            M[i, idx[m]] = 1
    size = M.sum(1)
    n = len(sets)
    D = np.empty((n, n), dtype=np.float32)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)
            inter = M[i0:i1] @ M[j0:j1].T
            si, sj = size[i0:i1][:, None], size[j0:j1][None, :]
            D[i0:i1, j0:j1] = (1.0 - inter / np.maximum(si + sj - inter, 1e-9)
                               if metric == "jaccard" else si + sj - 2.0 * inter)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=thresh, criterion="distance") - 1, D


def partial_corr(x, y, Z):
    """Spearman correlation of x and y after linearly removing controls Z."""
    from scipy.stats import rankdata
    xr, yr = rankdata(x), rankdata(y)
    Zr = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])]
                         + [np.ones(len(x))])
    bx, *_ = np.linalg.lstsq(Zr, xr, rcond=None)
    by, *_ = np.linalg.lstsq(Zr, yr, rcond=None)
    rx, ry = xr - Zr @ bx, yr - Zr @ by
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float((rx * ry).mean() / (sx * sy))


INTERNAL = ["n_sets", "d_n_sets", "mean_size", "d_mean_size", "max_size",
            "set_entropy", "new_set_share", "mean_pair_dist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--min_seqs", type=int, default=5000)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=40)
    ap.add_argument("--max_sets", type=int, default=6000)
    ap.add_argument("--metric", default="edit", choices=["edit", "jaccard"])
    ap.add_argument("--thresh", type=float, default=5.0)
    ap.add_argument("--min_cluster_seqs", type=int, default=50,
                    help="a cluster-month needs this many sequences for its "
                         "internal statistics to be meaningful")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "40_internal.csv"))
    args = ap.parse_args()

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    months = [m for m in months if m <= args.end_month]

    per_month, total = {}, Counter()
    for mo in months:
        with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
            raw = constellations_of(pickle.load(fh))
        f = {c: v for c, v in raw.items()
             if v >= args.min_count and 2 <= len(c) <= args.max_set_size}
        if sum(f.values()) < args.min_seqs:
            continue
        per_month[mo] = f
        for c, v in f.items():
            total[c] += v
    mos = sorted(per_month)
    sets = [c for c, _ in total.most_common(args.max_sets)]
    lab, D = cluster_sets(sets, args.thresh, args.metric)
    c2k = {c: int(lab[i]) for i, c in enumerate(sets)}
    s2i = {c: i for i, c in enumerate(sets)}
    K = int(lab.max()) + 1
    log(f"{len(mos)} months, {K} clusters at {args.metric} thresh={args.thresh}\n")

    seen = set()
    rows = []
    prev = {}
    for i, mo in enumerate(mos):
        f = per_month[mo]
        tot = sum(f.values())
        new_now = {c for c in f if c not in seen}
        by_k = defaultdict(list)
        for c, v in f.items():
            k = c2k.get(c)
            if k is not None:
                by_k[k].append((c, v))
        seen |= set(f)

        cur = {}
        for k, items in by_k.items():
            n_seq = sum(v for _, v in items)
            if n_seq < args.min_cluster_seqs:
                continue
            counts = np.array([v for _, v in items], float)
            p = counts / counts.sum()
            ent = float(-(p * np.log(p + 1e-12)).sum())
            sizes = np.array([len(c) for c, _ in items], float)
            idxs = [s2i[c] for c, _ in items if c in s2i]
            if len(idxs) >= 2:
                sub = D[np.ix_(idxs, idxs)]
                mpd = float(sub[np.triu_indices(len(idxs), 1)].mean())
            else:
                mpd = 0.0
            new_seq = sum(v for c, v in items if c in new_now)
            cur[k] = dict(freq=n_seq / tot, n_sets=float(len(items)),
                          mean_size=float(sizes.mean()), max_size=float(sizes.max()),
                          set_entropy=ent, new_set_share=new_seq / n_seq,
                          mean_pair_dist=mpd, n_seq=n_seq)

        for k, c in cur.items():
            p = prev.get(k)
            c["d_n_sets"] = (np.log1p(c["n_sets"]) - np.log1p(p["n_sets"])) if p else 0.0
            c["d_mean_size"] = (c["mean_size"] - p["mean_size"]) if p else 0.0
            c["g_prev"] = float(np.log((c["freq"] + 1e-9) / (p["freq"] + 1e-9))) if p else 0.0
            c.update(cluster=k, month=mo, month_i=i, depth=tot)
        prev = cur

        # attach next month's growth once we know it
        if i > 0:
            for r in rows:
                if r["month_i"] == i - 1 and r["cluster"] in cur:
                    r["g_next"] = float(np.log(
                        (cur[r["cluster"]]["freq"] + 1e-9) / (r["freq"] + 1e-9)))
        rows.extend(cur.values())

    df = pd.DataFrame([r for r in rows if "g_next" in r])
    if len(df) < 100:
        raise SystemExit(f"only {len(df)} cluster-months; lower --min_cluster_seqs")
    df["year"] = df.month.str[:4]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log("=" * 78)
    log(f"{len(df)} cluster-months, {df.cluster.nunique()} clusters, "
        f"{df.month.nunique()} months")
    log("=" * 78)

    Z = df[["g_prev", "freq", "depth"]].to_numpy(float)
    log(f"\n  {'feature':<16}{'raw':>9}{'partial':>10}{'within-clu':>12}")
    log("  (raw = Spearman with next-month growth; partial = after removing")
    log("   current growth, current frequency and sequencing depth;")
    log("   within-clu = partial, on values demeaned per cluster)")
    res = []
    for c in INTERNAL:
        x = df[c].to_numpy(float)
        y = df.g_next.to_numpy(float)
        raw = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        par = partial_corr(x, y, Z)
        dm = df.groupby("cluster")[[c, "g_next"]].transform(lambda v: v - v.mean())
        Zd = df.groupby("cluster")[["g_prev", "freq", "depth"]].transform(
            lambda v: v - v.mean()).to_numpy(float)
        wit = partial_corr(dm[c].to_numpy(float), dm.g_next.to_numpy(float), Zd)
        res.append((c, raw, par, wit))
        log(f"  {c:<16}{raw:>9.3f}{par:>10.3f}{wit:>12.3f}")

    # the baseline it has to beat
    braw = float(df.g_prev.corr(df.g_next, method="spearman"))
    log(f"\n  {'g_prev (baseline)':<16}{braw:>9.3f}")
    log("  Growth is autocorrelated, so this is the number any internal feature")
    log("  must add to -- which is what the partial column removes.")

    log("\n  BY YEAR (partial correlation)")
    log(f"  {'year':>6}{'n':>7}{'depth':>10}" +
        "".join(f"{c[:9]:>10}" for c in INTERNAL))
    for y, g in df.groupby("year"):
        if len(g) < 40:
            continue
        Zy = g[["g_prev", "freq", "depth"]].to_numpy(float)
        vals = [partial_corr(g[c].to_numpy(float), g.g_next.to_numpy(float), Zy)
                for c in INTERNAL]
        log(f"  {y:>6}{len(g):>7}{g.depth.median():>10.0f}" +
            "".join(f"{v:>10.3f}" for v in vals))

    log("\n" + "-" * 78)
    log("READ")
    log("-" * 78)
    best = max(res, key=lambda t: abs(t[2]) if not np.isnan(t[2]) else 0)
    log(f"  strongest partial correlation: {best[0]} at {best[2]:+.3f}")
    if abs(best[2]) > 0.15:
        log("  Internal state carries information about next-month growth beyond")
        log("  current growth, frequency and depth. Check the per-year row: if it")
        log("  holds in more than one era it is not a period artefact.")
    elif abs(best[2]) < 0.05:
        log("  Internal state adds essentially nothing once current growth is")
        log("  controlled for. Consistent with script 39, where twelve features")
        log("  beat single-feature growth by 0.0009 -- but measured here on")
        log("  thousands of cluster-months rather than 20 rise events, so it is")
        log("  a much stronger negative.")
    else:
        log("  Weak. Read the within-cluster column and the per-year row before")
        log("  concluding: a small effect that holds within clusters and across")
        log("  eras is more credible than a larger one that does not.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
