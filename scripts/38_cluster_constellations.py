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


def cluster(sets, thresh, block=400):
    """Single-linkage connected components at Jaccard distance < thresh.

    Blocked so the full N x N Jaccard matrix is never materialised -- with
    ~20k distinct constellations that would be 3 GB.
    """
    nodes = sorted({m for s in sets for m in s})
    idx = {m: i for i, m in enumerate(nodes)}
    M = np.zeros((len(sets), len(nodes)), dtype=np.float32)
    for i, s in enumerate(sets):
        for m in s:
            M[i, idx[m]] = 1
    size = M.sum(1)

    dsu = DSU(len(sets))
    n = len(sets)
    for i0 in range(0, n, block):
        Mi = M[i0:i0 + block]
        si = size[i0:i0 + block][:, None]
        for j0 in range(i0, n, block):
            Mj = M[j0:j0 + block]
            sj = size[j0:j0 + block][None, :]
            inter = Mi @ Mj.T
            jac = 1.0 - inter / np.maximum(si + sj - inter, 1e-9)
            ii, jj = np.where(jac < thresh)
            for u, v in zip(ii, jj):
                gu, gv = i0 + u, j0 + v
                if gu != gv:
                    dsu.union(gu, gv)
    lab = np.array([dsu.find(i) for i in range(n)])
    # relabel densely
    remap = {r: k for k, r in enumerate(sorted(set(lab)))}
    return np.array([remap[x] for x in lab]), M, size


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
    ap.add_argument("--thresh", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.25])
    ap.add_argument("--detail", action="store_true",
                    help="print the dominance timeline for the first threshold")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "38_clusters.csv"))
    args = ap.parse_args()

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
    keep = set(sets)
    log(f"{len(per_month)} months, {len(total)} distinct constellations, "
        f"clustering the top {len(sets)}\n")

    rows = []
    for th in args.thresh:
        lab, M, size = cluster(sets, th)
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
            jac = 1.0 - inter / max(size[i] + size[j] - inter, 1e-9)
            (wi if lab[i] == lab[j] else be).append(jac)
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

        rows.append(dict(thresh=th, n_clusters=K, n_big=n_big,
                         share_top1=share_top1, share_top10=share_top10,
                         within_jac=coh_w, between_jac=coh_b,
                         contiguity=contiguity, dom_switches=switches))
        log(f"  thresh={th:<5} clusters={K:<6} >1%={n_big:<4} "
            f"top1={share_top1:.1%} top10={share_top10:.1%}  "
            f"within={coh_w:.3f} between={coh_b:.3f}  "
            f"contiguity={contiguity:.2f}  dominance switches={switches}")

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
    log("  switch months against Alpha / Delta / Omicron / BA.2 before building")
    log("  anything on these clusters.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
