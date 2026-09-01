#!/usr/bin/env python3
"""
159_reachability.py -- how much of the future population is reachable from the
present one? No model, no training.

WHY THIS FIRST
--------------
The forecast run showed only 1.8% of the mass observed 12 months ahead sits on
variants present in the candidate support, so every method was charged the same
floor penalty for the other 98% and the comparison measured nothing. Before any
further modelling, the question is whether the target is reachable at all.

For each origin T and horizon h, this reports:

  mass_existing      share of the mass at T+h carried by variants already
                     present at T                          (persistence ceiling)
  mass_within_r      share carried by variants within Hamming distance r of
                     SOME variant present at T             (generation ceiling)
  median_distance    typical distance from a variant at T+h to the nearest
                     variant at T, weighted by mass

If mass_within_2 is high, candidate generation by local perturbation can work
and the problem is a search-budget one. If it is low, future variants are not
local edits of the present population and local generation is the wrong idea --
which would invalidate the enumeration approach rather than its parameters.

Distances are computed on a sparse incidence matrix, so this runs in minutes.

USAGE
    python scripts/159_reachability.py \
        --events data/processed/events_v3.tsv \
        --origins 2023-01-01 2023-07-01 2024-01-01 \
        --horizons 3 6 12 --top-src 5000
"""

import argparse
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


def load(path):
    by_day = defaultdict(lambda: defaultdict(float))
    V = 0
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            ids = tuple(int(x) for x in p[1].split(",") if x != "")
            if not ids:
                continue
            w = float(p[2]) if len(p) > 2 else 1.0
            by_day[p[0]][ids] += w
            V = max(V, max(ids) + 1)
    return by_day, V


def window_mass(by_day, days, lo, hi):
    """Aggregate normalised mass over a date window."""
    agg = defaultdict(float)
    for d in days:
        if lo <= d <= hi:
            for ids, w in by_day[d].items():
                agg[ids] += w
    tot = sum(agg.values())
    return {k: v / tot for k, v in agg.items()} if tot else {}


def to_csr(variants, V):
    rows, cols = [], []
    for i, s in enumerate(variants):
        for m in s:
            rows.append(i); cols.append(m)
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(max(len(variants), 1), V))


def nearest_distances(tgt, src, V, block=2000):
    """Hamming distance from each target variant to the nearest source one."""
    if not src or not tgt:
        return np.full(len(tgt), 1e9)
    S = to_csr(src, V)
    s_sz = np.array([len(x) for x in src], dtype=np.float32)
    out = np.empty(len(tgt), dtype=np.float32)
    for i in range(0, len(tgt), block):
        chunk = tgt[i:i + block]
        Q = to_csr(chunk, V)
        q_sz = np.array([len(x) for x in chunk], dtype=np.float32)
        inter = np.asarray((Q @ S.T).todense())
        d = q_sz[:, None] + s_sz[None, :] - 2.0 * inter
        out[i:i + block] = d.min(axis=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--origins", nargs="+", required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 12])
    p.add_argument("--src-window", type=int, default=30, dest="src_window",
                   help="days of history defining the present population")
    p.add_argument("--tgt-window", type=int, default=30, dest="tgt_window")
    p.add_argument("--top-src", type=int, default=5000, dest="top_src",
                   help="keep this many source variants, heaviest first")
    p.add_argument("--top-tgt", type=int, default=5000, dest="top_tgt")
    p.add_argument("--radii", type=int, nargs="+", default=[0, 1, 2, 3, 5, 10])
    a = p.parse_args()

    by_day, V = load(a.events)
    days = sorted(by_day)
    print(f"{len(days)} days, V={V}")

    def shift(d, months):
        y, m, dd = (int(x) for x in d.split("-"))
        m += months
        y += (m - 1) // 12; m = (m - 1) % 12 + 1
        return f"{y:04d}-{m:02d}-{min(dd, 28):02d}"

    def minus_days(d, n):
        import datetime
        y, m, dd = (int(x) for x in d.split("-"))
        return str(datetime.date(y, m, dd) - datetime.timedelta(days=n))

    print(f"\n{'origin':<12}{'h':>4}{'srcV':>7}{'tgtV':>7}"
          f"{'existing':>10}" +
          "".join(f"{'<=' + str(r):>8}" for r in a.radii if r > 0) +
          f"{'med d':>8}")
    print("-" * (40 + 8 * len([r for r in a.radii if r > 0]) + 8))

    rows = []
    for T in a.origins:
        src = window_mass(by_day, days, minus_days(T, a.src_window), T)
        if not src:
            print(f"{T:<12}  no source data")
            continue
        src_v = [k for k, _ in sorted(src.items(), key=lambda kv: -kv[1])
                 ][:a.top_src]
        src_set = set(src_v)
        for h in a.horizons:
            t1 = shift(T, h)
            import datetime as _dt
            _y, _m, _d = (int(x) for x in t1.split("-"))
            t2 = str(_dt.date(_y, _m, _d)
                     + _dt.timedelta(days=a.tgt_window))
            tgt = window_mass(by_day, days, t1, t2)
            if not tgt:
                continue
            tgt_items = sorted(tgt.items(), key=lambda kv: -kv[1])[:a.top_tgt]
            tgt_v = [k for k, _ in tgt_items]
            w = np.array([v for _, v in tgt_items], dtype=float)
            w = w / w.sum()

            m_exist = float(sum(wi for k, wi in zip(tgt_v, w)
                                if k in src_set))
            d = nearest_distances([set(x) for x in tgt_v],
                                  [set(x) for x in src_v], V)
            order = np.argsort(d)
            cw = np.cumsum(w[order])
            med = float(d[order][np.searchsorted(cw, 0.5)]) \
                if len(cw) else float("nan")
            cells = []
            for r in a.radii:
                if r == 0:
                    continue
                cells.append(float(w[d <= r].sum()))
            print(f"{T:<12}{h:>4}{len(src_v):>7}{len(tgt_v):>7}"
                  f"{m_exist:>10.3f}" +
                  "".join(f"{c:>8.3f}" for c in cells) + f"{med:>8.1f}")
            rows.append((T, h, m_exist, cells, med))

    print("\nColumns are the share of FUTURE MASS that is:")
    print("  existing : carried by a variant already present at T")
    print("  <=r      : within Hamming distance r of some variant at T")
    print("  med d    : mass-weighted median distance to the nearest variant")
    print("\nThese are ceilings. No model can score mass it cannot reach, so")
    print("<=2 is the most a local-perturbation generator could ever cover.")


if __name__ == "__main__":
    main()
