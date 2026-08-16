#!/usr/bin/env python
"""
38_cluster_constellations.py

Group constellations into variant-like clusters, then check whether those
clusters behave like real lineages. CPU, a few minutes. numpy + pandas only.

WHY CLUSTERS AND NOT SETS
------------------------
A Pango lineage is a clade whose members carry similar but not identical
mutation profiles -- sublineages, private mutations, sequencing noise. One
variant therefore corresponds to MANY exact constellations. That mismatch has
been distorting everything measured so far: set counts swing 121 -> 5,984 ->
27 across months largely because sequencing depth changes how much sublineage
structure is resolved, not because the virus changed that much.

Clustering collapses that. The unit becomes a group of near-identical
constellations, which is closer to what "a variant" means and much less
sensitive to depth.

It also changes the problem in a way that matters. Forecasting a cluster's
growth is predicting the rise of something ALREADY PRESENT in small numbers,
not generating something absent. That sidesteps the wall every previous
approach hit -- attachment +0.02 (script 24), growth additive (script 32),
position choice +0.000 (script 33) -- and it is what tfpscanner, HELEN and
the published AI risk models actually do.

HOW CLUSTERS ARE BUILT
----------------------
Pooled across all months, so clusters are FIXED and comparable over time.
Clustering each month separately would reintroduce the correspondence problem
this is meant to avoid.

Single-linkage connected components on the graph joining constellations with
Jaccard distance below --thresh. Single linkage is deliberate: it chains
along one-mutation steps, which is how sublineages actually relate.

THE CHECKS
----------
Clusters are not validated against Pango labels here (you do not have them),
so the checks are internal and behavioural:

  1. SIZE       do a handful of clusters carry most sequences, as lineages do,
                or does everything collapse into one giant component?
  2. COHERENCE  within-cluster Jaccard should be far below between-cluster.
  3. TEMPORAL   a real lineage occupies a contiguous stretch of months and
                rises then falls. Clusters scattered across the whole series
                are an artefact.
  4. SWEEPS     the dominant cluster should change at the known transitions
                (Alpha ~2021-01, Delta ~2021-06, Omicron ~2021-12, BA.2
                ~2022-03). This is the one external check available.

Threshold choice is the main free parameter and check 4 is how to set it.
Sweep --thresh and keep the value where the dominance timeline matches.

Usage
-----
  python scripts/38_cluster_constellations.py
  python scripts/38_cluster_constellations.py --thresh 0.05 0.10 0.15 0.20
  python scripts/38_cluster_constellations.py --thresh 0.10 --detail
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


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _pairdist(M, size, metric, i0, i1, j0, j1):
    Mi, Mj = M[i0:i1], M[j0:j1]
    inter = Mi @ Mj.T
    si, sj = size[i0:i1][:, None], size[j0:j1][None, :]
    if metric == "jaccard":
        return 1.0 - inter / np.maximum(si + sj - inter, 1e-9)
    return si + sj - 2.0 * inter


def cluster(sets, thresh, metric="jaccard", linkage="single", block=400):
    """Agglomerative clustering at distance <= thresh.

    single   -- connected components. Fast, but chains: A~B and B~C puts A and
                C together however far apart they are. On this data that merged
                pre-Alpha, Alpha and Delta at edit-2, because each consecutive
                lineage is within 2 mutations of the last.
    average  -- a point joins a cluster only if its MEAN distance to that
                cluster's members is within thresh. Breaks chains.
    complete -- MAX distance within thresh. Strictest; clusters have bounded
                diameter.

    average/complete are done greedily in abundance order (sets are passed
    most-abundant-first), which is O(n * k) rather than the O(n^2 log n) of a
    full agglomerative merge and is adequate here because real lineages are
    seeded by abundant members.
    """
    nodes = sorted({m for s in sets for m in s})
    idx = {m: i for i, m in enumerate(nodes)}
    M = np.zeros((len(sets), len(nodes)), dtype=np.float32)
    for i, s in enumerate(sets):
        for m in s:
            M[i, idx[m]] = 1
    size = M.sum(1)
    n = len(sets)

    if linkage == "single":
        dsu = DSU(n)
        for i0 in range(0, n, block):
            for j0 in range(i0, n, block):
                D = _pairdist(M, size, metric, i0, min(i0 + block, n),
                              j0, min(j0 + block, n))
                ii, jj = np.where(D <= thresh) if metric == "edit" \
                    else np.where(D < thresh)
                for u, v in zip(ii, jj):
                    gu, gv = i0 + u, j0 + v
                    if gu != gv:
                        dsu.union(gu, gv)
        lab = np.array([dsu.find(i) for i in range(n)])
    else:
        # greedy: walk sets in abundance order, join the first cluster whose
        # mean (or max) distance is within thresh, else start a new one
        lab = np.full(n, -1, dtype=int)
        members = []                      # list of index arrays, one per cluster
        for i in range(n):
            best, bestd = -1, np.inf
            for k, mem in enumerate(members):
                inter = M[mem] @ M[i]
                if metric == "jaccard":
                    d = 1.0 - inter / np.maximum(size[mem] + size[i] - inter, 1e-9)
                else:
                    d = size[mem] + size[i] - 2.0 * inter
                dd = float(d.mean()) if linkage == "average" else float(d.max())
                if dd <= thresh and dd < bestd:
                    best, bestd = k, dd
            if best < 0:
                members.append(np.array([i]))
                lab[i] = len(members) - 1
            else:
                members[best] = np.append(members[best], i)
                lab[i] = best

    remap = {r: k for k, r in enumerate(sorted(set(lab.tolist())))}
    return np.array([remap[x] for x in lab.tolist()]), M, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--min_seqs", type=int, default=5000)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=40)
    ap.add_argument("--max_sets", type=int, default=6000,
                    help="cap on distinct constellations pooled across months "
                         "(most abundant kept), for tractability")
    ap.add_argument("--linkage", default="single",
                    choices=["single", "average", "complete"],
                    help="single: connected components. Chains along one-mutation "
                         "steps, which is how sublineages relate -- but that is "
                         "exactly why it merged pre-Alpha, Alpha and Delta into one "
                         "cluster at edit-2: each consecutive pair is within 2, so "
                         "the whole corridor links up. average/complete break those "
                         "chains by requiring a candidate to be close to the cluster "
                         "as a whole, not just to one member.")
    ap.add_argument("--metric", default="jaccard", choices=["jaccard", "edit"],
                    help="jaccard: 1 - |A&B|/|A|B|. Scale-dependent -- Jaccard "
                         "0.15 means 4 differing mutations on a size-25 set but "
                         "only 1 on a size-7 set, so a fixed threshold means "
                         "different things in the Delta era (small sets) and the "
                         "Omicron era (~30 mutations). That is why jaccard=0.15 "
                         "resolves Delta but merges everything after BA.2, while "
                         "0.05 resolves XBB but shatters Delta into pieces that "
                         "trade dominance month to month.\n"
                         "edit: |A symmetric-difference B|, the raw number of "
                         "differing mutations. Scale-free, so one threshold "
                         "means the same thing in both eras.")
    ap.add_argument("--thresh", type=float, nargs="+", default=None,
                    help="defaults to [0.05,0.10,0.15,0.25] for jaccard, "
                         "[1,2,3,5] for edit")
    ap.add_argument("--shuffle_seed", type=int, default=None,
                    help="permute the constellation ordering before clustering. "
                         "average/complete linkage are GREEDY in abundance order, "
                         "so clusters are seeded by abundant members and the result "
                         "can in principle depend on that ordering. Rerun with "
                         "different seeds: if the dominance switch MONTHS survive, "
                         "the six transitions are not an artefact of the ordering. "
                         "Cluster IDs will differ between runs -- only the months "
                         "and the sequence of switches are comparable.")
    ap.add_argument("--detail", action="store_true",
                    help="print the dominance timeline for the first threshold")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "38_clusters.csv"))
    args = ap.parse_args()
    if args.thresh is None:
        args.thresh = [0.05, 0.10, 0.15, 0.25] if args.metric == "jaccard" \
            else [1.0, 2.0, 3.0, 5.0]

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    months = [m for m in months if m <= args.end_month]

    # ---- pool constellations across months ----
    per_month = {}
    total = Counter()
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
    if not per_month:
        raise SystemExit("no usable months")

    sets = [c for c, _ in total.most_common(args.max_sets)]
    if args.shuffle_seed is not None:
        rs = np.random.default_rng(args.shuffle_seed)
        order = rs.permutation(len(sets))
        sets = [sets[i] for i in order]
        log(f"ordering shuffled with seed {args.shuffle_seed} "
            f"(greedy linkage seeds clusters in input order)")
    keep = set(sets)
    log(f"{len(per_month)} months, {len(total)} distinct constellations, "
        f"clustering the top {len(sets)}\n")

    rows = []
    for th in args.thresh:
        lab, M, size = cluster(sets, th, args.metric, args.linkage)
        K = lab.max() + 1

        # sequence mass per cluster per month
        mass = defaultdict(Counter)
        c2k = {c: lab[i] for i, c in enumerate(sets)}
        for mo, f in per_month.items():
            for c, v in f.items():
                if c in keep:
                    mass[mo][c2k[c]] += v

        # cluster sizes by total sequences
        tot_k = Counter()
        for mo in mass:
            for k, v in mass[mo].items():
                tot_k[k] += v
        grand = sum(tot_k.values())
        top = tot_k.most_common()
        share_top1 = top[0][1] / grand
        share_top10 = sum(v for _, v in top[:10]) / grand
        n_big = sum(1 for _, v in top if v / grand > 0.01)

        # coherence: mean within-cluster vs between-cluster Jaccard, sampled
        rng = np.random.default_rng(0)
        wi, be = [], []
        for _ in range(3000):
            i, j = rng.integers(0, len(sets), 2)
            if i == j:
                continue
            inter = float(M[i] @ M[j])
            d = (1.0 - inter / max(size[i] + size[j] - inter, 1e-9)
                 if args.metric == "jaccard" else size[i] + size[j] - 2.0 * inter)
            (wi if lab[i] == lab[j] else be).append(d)
        coh_w = float(np.mean(wi)) if wi else np.nan
        coh_b = float(np.mean(be)) if be else np.nan

        # temporal contiguity: months a cluster is present / span it covers
        contig = []
        for k, _ in top[:30]:
            ms = sorted(mo for mo in mass if mass[mo].get(k, 0) > 0)
            if len(ms) < 2:
                continue
            span = months.index(ms[-1]) - months.index(ms[0]) + 1
            contig.append(len(ms) / span)
        contiguity = float(np.mean(contig)) if contig else np.nan

        # dominance timeline
        dom = {mo: max(mass[mo].items(), key=lambda kv: kv[1])[0]
               for mo in sorted(mass) if mass[mo]}
        switches = sum(1 for a, b in zip(list(dom.values())[:-1], list(dom.values())[1:])
                       if a != b)

        sw_months = [mo for prev, (mo, k) in
                     zip([None] + list(dom.values())[:-1], dom.items())
                     if prev is not None and k != prev]
        rows.append(dict(metric=args.metric, linkage=args.linkage, thresh=th,
                         switch_months="|".join(sw_months), n_clusters=K, n_big=n_big,
                         share_top1=share_top1, share_top10=share_top10,
                         within_jac=coh_w, between_jac=coh_b,
                         contiguity=contiguity, dom_switches=switches))
        log(f"  thresh={th:<5} clusters={K:<6} >1%={n_big:<4} "
            f"top1={share_top1:.1%} top10={share_top10:.1%}  "
            f"within={coh_w:.2f} between={coh_b:.2f}  "
            f"contiguity={contiguity:.2f}  switches={switches}")
        log(f"           switch months: {' '.join(sw_months) if sw_months else '(none)'}")

        if args.detail and th == args.thresh[0]:
            log(f"\n  DOMINANCE TIMELINE (thresh={th})")
            log(f"  {'month':<10}{'dom':>6}{'share':>8}{'n_clust':>9}{'seqs':>10}")
            prev = None
            for mo in sorted(mass):
                tt = sum(mass[mo].values())
                k, v = max(mass[mo].items(), key=lambda kv: kv[1])
                mark = "  <-- SWITCH" if prev is not None and k != prev else ""
                log(f"  {mo:<10}{k:>6}{v/tt:>8.1%}{len(mass[mo]):>9}{tt:>10}{mark}")
                prev = k
            log("\n  Known transitions to check against: Alpha ~2021-01,")
            log("  Delta ~2021-06, Omicron ~2021-12, BA.2 ~2022-03.")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log("\n" + "-" * 74)
    log("READ")
    log("-" * 74)
    log("  Pick the threshold where:")
    log("   - a handful of clusters carry most sequences (top10 high, >1% count")
    log("     in the tens, not hundreds or one)")
    log("   - within-cluster Jaccard is far below between-cluster")
    log("   - contiguity near 1.0 (clusters occupy a continuous stretch of")
    log("     months, as lineages do, rather than reappearing at random)")
    log("   - dominance switches are FEW and land near the known transitions")
    log("")
    log("  Too small a threshold: everything is its own cluster, nothing is")
    log("  grouped. Too large: one giant component, top1 near 100%. The useful")
    log("  range sits between, and the dominance timeline is how to find it.")
    log("")
    log("  Rerun with --detail once a threshold looks right, and check the")
    log("  switch months against Alpha / Delta / Omicron / BA.2. With")
    log("  --metric edit the threshold is a COUNT of differing mutations, so")
    log("  the same value means the same thing in the small-set Delta era and")
    log("  the ~30-mutation Omicron era -- which is what the jaccard version")
    log("  could not do.")
    log("  switch months against Alpha / Delta / Omicron / BA.2 before building")
    log("  anything on these clusters.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
