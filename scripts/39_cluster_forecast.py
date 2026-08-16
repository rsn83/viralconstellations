#!/usr/bin/env python
"""
39_cluster_forecast.py

Two questions about the clusters from script 38, in order:

  1. HOW MANY forecastable events are there?
  2. Given that many, is which-cluster-rises predictable?

Question 1 comes first because it can kill question 2. Script 27 counted
establishment events at the exact-constellation level and found 95 usable ones;
at cluster level there will be far fewer, and a handful of events cannot
support a learned model however good the features are.

WHY CLUSTER LEVEL
-----------------
Script 38 established an unsupervised variant definition: agglomerative
clustering on edit distance over mutation sets recovers Alpha (2021-01),
Delta (2021-06), Omicron BA.1 (2022-01), BA.2 (2022-03), BA.5 (2022-06) and
XBB (2023-02) at the correct months, order-independently and stably across
thresholds 3-7.

That matters because forecasting a CLUSTER's growth means predicting the rise
of something already circulating at low frequency -- not generating a set that
does not exist. Generation is the wall every earlier approach hit: attachment
+0.02 over marginal frequency (script 24), growth additive with no persistent
deviation (script 32), position choice +0.000 (script 33). Predicting growth of
an existing cluster sidesteps it, and it is what tfpscanner, HELEN and the
published AI risk models actually do.

THE EVENT
---------
A cluster is a RISER at month t if its frequency is at or below `low` at t and
exceeds `high` within `horizon` months. Frequencies, not counts: sequencing
depth swings ~20x across the study period.

THE PREDICTION TASK
-------------------
At month t, consider every cluster present but below `low`. Rank them by how
likely they are to exceed `high` by t+horizon. Walk-forward: each test month is
scored by a model fitted only on strictly earlier months.

  BASELINES
    current_freq  rank by frequency now. The bar -- a cluster already larger is
                  more likely to grow, and beating this is the whole question.
    recent_growth rank by log frequency change over the last month.

  FEATURES (all observable at t)
    frequency now, and log growth over 1 / 2 / 3 months
    months since the cluster first appeared, months present in the window
    number of distinct constellations in the cluster, and its growth
    mean and max set size, and whether sets are still gaining mutations
    the cluster's share of the newest constellations that month

Usage
-----
  python scripts/39_cluster_forecast.py
  python scripts/39_cluster_forecast.py --thresh 4 --low 0.02 --high 0.20
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


def membership(sets):
    nodes = sorted({m for s in sets for m in s})
    idx = {m: i for i, m in enumerate(nodes)}
    M = np.zeros((len(sets), len(nodes)), dtype=np.float32)
    for i, s in enumerate(sets):
        for m in s:
            M[i, idx[m]] = 1
    return M, M.sum(1)


def cluster_sets(sets, thresh, metric, block=400):
    """Order-independent agglomerative clustering, as validated in script 38."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    M, size = membership(sets)
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
    return fcluster(Z, t=thresh, criterion="distance") - 1, size


def auc(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s), dtype=float)
    r[o] = np.arange(1, len(s) + 1)
    ss, rs = s[o], r[o]
    k = 0
    while k < len(ss):
        j = k
        while j + 1 < len(ss) and ss[j + 1] == ss[k]:
            j += 1
        if j > k:
            rs[k:j + 1] = rs[k:j + 1].mean()
        k = j + 1
    r[o] = rs
    p, n = int(y.sum()), int((~y).sum())
    if p == 0 or n == 0:
        return float("nan")
    return (r[y].sum() - p * (p + 1) / 2) / (p * n)


FEATS = ["freq", "g1", "g2", "g3", "age", "months_present",
         "n_sets", "n_sets_g1", "mean_size", "max_size", "size_g1", "new_share"]


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
    ap.add_argument("--low", type=float, nargs="+", default=[0.01, 0.02, 0.05],
                    help="cluster frequency must be AT OR BELOW this at t")
    ap.add_argument("--high", type=float, nargs="+", default=[0.10, 0.25, 0.50],
                    help="and EXCEED this within the horizon")
    ap.add_argument("--horizon", type=int, nargs="+", default=[2, 3, 6])
    ap.add_argument("--min_train_months", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "39_cluster_forecast.csv"))
    args = ap.parse_args()

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    months = [m for m in months if m <= args.end_month]

    # ---- pool and cluster once, so clusters are fixed across time ----
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
    lab, size = cluster_sets(sets, args.thresh, args.metric)
    c2k = {c: int(lab[i]) for i, c in enumerate(sets)}
    K = int(lab.max()) + 1
    log(f"{len(mos)} months, {K} clusters at {args.metric} thresh={args.thresh}\n")

    # ---- per-cluster monthly series ----
    freq = defaultdict(dict)      # k -> month -> frequency
    nsets = defaultdict(dict)
    msize = defaultdict(dict)
    xsize = defaultdict(dict)
    newshare = defaultdict(dict)
    seen_before = set()
    for mo in mos:
        f = per_month[mo]
        tot = sum(f.values())
        agg = defaultdict(lambda: [0.0, 0, 0.0, 0, 0.0])
        new_now = {c for c in f if c not in seen_before}
        n_new = max(len(new_now), 1)
        for c, v in f.items():
            k = c2k.get(c)
            if k is None:
                continue
            a = agg[k]
            a[0] += v
            a[1] += 1
            a[2] += len(c)
            a[3] = max(a[3], len(c))
            a[4] += 1.0 if c in new_now else 0.0
        seen_before |= set(f)
        for k, a in agg.items():
            freq[k][mo] = a[0] / tot
            nsets[k][mo] = a[1]
            msize[k][mo] = a[2] / a[1]
            xsize[k][mo] = a[3]
            newshare[k][mo] = a[4] / n_new

    first_seen = {k: min(freq[k]) for k in freq}
    midx = {m: i for i, m in enumerate(mos)}

    def feats(k, mo):
        i = midx[mo]
        f0 = freq[k].get(mo, 0.0)
        if f0 <= 0:
            return None

        def g(lag):
            if i - lag < 0:
                return 0.0
            p = freq[k].get(mos[i - lag], 0.0)
            return float(np.log((f0 + 1e-9) / (p + 1e-9)))
        ns = nsets[k].get(mo, 0)
        ns1 = nsets[k].get(mos[i - 1], 0) if i >= 1 else ns
        ms = msize[k].get(mo, 0.0)
        ms1 = msize[k].get(mos[i - 1], ms) if i >= 1 else ms
        return [f0, g(1), g(2), g(3),
                float(i - midx[first_seen[k]]),
                float(sum(1 for m in mos[:i + 1] if m in freq[k])),
                float(ns), float(np.log1p(ns) - np.log1p(ns1)),
                ms, float(xsize[k].get(mo, 0)), ms - ms1,
                newshare[k].get(mo, 0.0)]

    # ---- Q1: count events over the threshold grid ----
    log("=" * 78)
    log("Q1  HOW MANY FORECASTABLE EVENTS?")
    log("=" * 78)
    log(f"  {'low':>6}{'high':>7}{'h':>3}{'cands':>8}{'risers':>8}{'rate':>8}"
        f"{'uniq_k':>8}{'months':>8}")
    grid = []
    for lo in args.low:
        for hi in args.high:
            if hi <= lo:
                continue
            for h in args.horizon:
                cand = ris = 0
                ks, mset = set(), set()
                for i, mo in enumerate(mos):
                    if i + h >= len(mos):
                        continue
                    for k in freq:
                        f0 = freq[k].get(mo, 0.0)
                        if not (0 < f0 <= lo):
                            continue
                        cand += 1
                        fut = max((freq[k].get(mos[j], 0.0)
                                   for j in range(i + 1, i + h + 1)), default=0.0)
                        if fut > hi:
                            ris += 1
                            ks.add(k)
                            mset.add(mo)
                grid.append(dict(low=lo, high=hi, horizon=h, cands=cand,
                                 risers=ris, rate=ris / max(cand, 1),
                                 uniq_clusters=len(ks), months=len(mset)))
                log(f"  {lo:>6.2f}{hi:>7.2f}{h:>3}{cand:>8}{ris:>8}"
                    f"{ris/max(cand,1):>8.3f}{len(ks):>8}{len(mset):>8}")

    g = pd.DataFrame(grid)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(args.out, index=False)

    log("\n  `uniq_clusters` is the number that bounds what a model can learn --")
    log("  the same cluster rising in consecutive months is ONE event, counted")
    log("  several times. Fewer than ~10 distinct risers is a case study, not a")
    log("  learning problem, whatever `risers` says.")

    # ---- Q2: is it predictable, at the most permissive usable setting ----
    ok = g[(g.uniq_clusters >= 5) & (g.risers >= 15)]
    if not len(ok):
        log("\n" + "=" * 78)
        log("Q2  SKIPPED -- no setting has enough distinct risers to model.")
        log("=" * 78)
        log("  This is the answer, not a failure of the script. At cluster level")
        log("  the events are variant transitions, and there are only a handful")
        log("  in four years. Report them as case studies.")
        return
    best = ok.sort_values("risers", ascending=False).iloc[0]
    lo, hi, h = best.low, best.high, int(best.horizon)
    log("\n" + "=" * 78)
    log(f"Q2  IS IT PREDICTABLE?  (low={lo}, high={hi}, horizon={h})")
    log("=" * 78)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise SystemExit("needs scikit-learn")

    data = {}
    for i, mo in enumerate(mos):
        if i + h >= len(mos):
            continue
        X, y = [], []
        for k in freq:
            f0 = freq[k].get(mo, 0.0)
            if not (0 < f0 <= lo):
                continue
            fv = feats(k, mo)
            if fv is None:
                continue
            fut = max((freq[k].get(mos[j], 0.0)
                       for j in range(i + 1, i + h + 1)), default=0.0)
            X.append(fv)
            y.append(fut > hi)
        if len(y) >= 5:
            data[i] = (np.array(X, float), np.array(y, bool), mo)

    idxs = sorted(data)
    rows = []
    for pos, ti in enumerate(idxs):
        if pos < args.min_train_months:
            continue
        Xtr = np.vstack([data[j][0] for j in idxs[:pos]])
        ytr = np.concatenate([data[j][1] for j in idxs[:pos]])
        X, y, mo = data[ti]
        if ytr.sum() < 5 or y.sum() == 0 or y.all():
            continue
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=3000, class_weight="balanced")
        lr.fit(sc.transform(Xtr), ytr)
        s_lr = lr.predict_proba(sc.transform(X))[:, 1]
        rows.append(dict(month=mo, n=len(y), n_pos=int(y.sum()),
                         auc_freq=auc(y, X[:, FEATS.index("freq")]),
                         auc_growth=auc(y, X[:, FEATS.index("g1")]),
                         auc_lr=auc(y, s_lr)))
        log(f"  {mo}  n={len(y):3d} pos={int(y.sum()):2d} | "
            f"freq {rows[-1]['auc_freq']:.3f} | growth {rows[-1]['auc_growth']:.3f}"
            f" | logreg {rows[-1]['auc_lr']:.3f}")

    if not rows:
        log("  no test months with both classes present.")
        return
    d = pd.DataFrame(rows)
    log("\n  " + "-" * 60)
    log(f"  over {len(d)} test months, {int(d.n_pos.sum())} rise events")
    for c, lbl in [("auc_freq", "current frequency"),
                   ("auc_growth", "recent growth"), ("auc_lr", "logreg")]:
        log(f"    {lbl:<20}{d[c].mean():.4f}")
    gain = d.auc_lr.mean() - d.auc_freq.mean()
    log(f"\n  gain over frequency: {gain:+.4f}  "
        f"(beats it in {(d.auc_lr > d.auc_freq).sum()}/{len(d)} months)")
    log("\n  With this many events the mean is fragile -- the per-month win count")
    log("  is the more trustworthy figure, and even that rests on few positives.")


if __name__ == "__main__":
    main()
