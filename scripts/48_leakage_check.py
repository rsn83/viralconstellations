#!/usr/bin/env python
"""
48_leakage_check.py

Test whether the winner rule is exploiting future cluster definitions.
Rebuilds cluster partition from scratch for 3,4,5,6 variant cutoffs.
"""
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
gd = ROOT / "data" / "processed" / "full_data_graphs_posres"
months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())

CUTOFFS = {
    3: "2021-05",
    4: "2021-11",
    5: "2022-02",
    6: "2024-12",
}


def load_and_cluster(end_month, min_count=3, min_seqs=5000,
                     thresh=5.0, max_sets=6000):
    per_month = {}
    total = Counter()
    for mo in months:
        if mo > end_month:
            continue
        with open(gd / f"{mo}_occupied.pkl", "rb") as f:
            raw = pickle.load(f)
        occ = {frozenset(c): v for c, v in raw.items()
               if isinstance(v, (int, float)) and v >= min_count
               and 2 <= len(frozenset(c)) <= 40}
        if sum(occ.values()) < min_seqs:
            continue
        per_month[mo] = occ
        for c, v in occ.items():
            total[c] += v

    sets = [c for c, _ in total.most_common(max_sets)]
    if len(sets) < 10:
        return None, None

    nodes = sorted({m for s in sets for m in s})
    idx = {m: i for i, m in enumerate(nodes)}
    M = np.zeros((len(sets), len(nodes)), dtype=np.float32)
    for i, s in enumerate(sets):
        for m in s:
            M[i, idx[m]] = 1
    sz = M.sum(1)
    n = len(sets)
    D = np.zeros((n, n), dtype=np.float32)
    block = 400
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)
            inter = M[i0:i1] @ M[j0:j1].T
            si = sz[i0:i1][:, None]
            sj = sz[j0:j1][None, :]
            D[i0:i1, j0:j1] = si + sj - 2 * inter
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    lab = fcluster(linkage(squareform(D, checks=False), method="average"),
                   t=thresh, criterion="distance") - 1
    c2k = {c: int(lab[i]) for i, c in enumerate(sets)}
    return per_month, c2k


print("Leakage check: winner rule across variant cutoffs")
print("Cluster partition rebuilt from scratch at each cutoff")
print()
header = ("n_variants", "end_month", "n_clusters", "n_switches", "correct")
print(f"  {header[0]:>12} {header[1]:>12} {header[2]:>12} "
      f"{header[3]:>12} {header[4]:>8}")
print("  " + "-" * 60)

for n_var, end_mo in CUTOFFS.items():
    per_month, c2k = load_and_cluster(end_mo)
    if per_month is None:
        print(f"  {n_var:>12} {end_mo:>12} failed")
        continue

    mos = sorted(per_month.keys())
    midx = {m: i for i, m in enumerate(mos)}

    clust_freq = {}
    for mo in mos:
        tot = sum(per_month[mo].values())
        by_k = Counter()
        for c, v in per_month[mo].items():
            k = c2k.get(c)
            if k is not None:
                by_k[k] += v
        clust_freq[mo] = {k: v / tot for k, v in by_k.items()}

    switches_in_data = []
    prev_dom = None
    for mo in mos:
        cf = clust_freq[mo]
        if not cf:
            continue
        dom = max(cf, key=cf.get)
        if prev_dom is not None and dom != prev_dom:
            switches_in_data.append((mo, dom))
        prev_dom = dom

    correct = 0
    total_sw = 0
    details = []
    for sw_mo, true_winner in switches_in_data:
        sw_i = midx.get(sw_mo)
        if sw_i is None or sw_i < 2:
            continue
        mo_prev = mos[sw_i - 2]
        mo_curr = mos[sw_i - 1]
        cf_prev = clust_freq.get(mo_prev, {})
        cf_curr = clust_freq.get(mo_curr, {})
        dom_curr = max(cf_curr, key=cf_curr.get) if cf_curr else None
        scores = {}
        for k in set(cf_prev) & set(cf_curr):
            if k == dom_curr:
                continue
            f0 = cf_prev.get(k, 0)
            f1 = cf_curr.get(k, 0)
            if f0 > 0 and f1 > 0:
                scores[k] = np.log(f1 / f0) + np.log(f1 + 1e-6)
        if not scores:
            continue
        pred = max(scores, key=scores.get)
        ok = pred == true_winner
        correct += ok
        total_sw += 1
        details.append((sw_mo, true_winner, pred, ok))

    n_clusters = len(set(c2k.values()))
    print(f"  {n_var:>12} {end_mo:>12} {n_clusters:>12} "
          f"{total_sw:>12} {correct}/{total_sw}")
    for sw_mo, true_w, pred_w, ok in details:
        print(f"               {sw_mo}  true={true_w}  pred={pred_w}  "
              f"{'OK' if ok else 'MISS'}")
    print()
