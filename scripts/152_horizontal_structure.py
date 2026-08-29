#!/usr/bin/env python3
"""
152_horizontal_structure.py -- is there structure across mutation sets within a
month, beyond the shared core?

QUESTION
--------
In late 2022 nearly every circulating mutation set shares a large Omicron core,
so any similarity measure between sets is high for trivial reasons. The real
question is whether structure REMAINS once the core is removed: do sets still
cluster, or are they a shared core plus independent noise?

This matters because Jaccard attention in 143_v2 mixes information across sets
within a month. If the only cross-set signal is the core, that attention is
modelling nothing that uniform pooling would miss.

MEASUREMENT
-----------
For each month:
  1. core = mutations present in >= theta of the sets
  2. strip core columns -> residual matrix R (sets x non-core mutations)
  3. statistics on R: mean pairwise Jaccard, mean size-2 co-occurrence count

NULL
----
Degree-preserving randomisation (curveball swaps) on R: row sums (set sizes) and
column sums (mutation frequencies) are both held fixed, so only WHICH mutation
sits in WHICH set is destroyed. Preserving column sums is essential -- with free
column sums, frequent mutations co-occur trivially and any test passes.

WHAT IS AND IS NOT CLAIMED
--------------------------
Claimed:     residual co-occurrence exceeds what independent assortment at the
             same margins would produce.
Not claimed: the structure is phylogenetic, causal, or predictive. Clustering
             consistent with descent is not evidence of descent.

USAGE
    python scripts/152_horizontal_structure.py \
        --data data/processed/full_data_graphs_withdel \
        --months 2020-06:2024-12 --theta 0.9 --n-null 20 --out results_152
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module


def _load_150(script_dir):
    """Reuse 150's loader so both scripts read the data identically."""
    import importlib.util
    path = os.path.join(script_dir, "150_covering_null.py")
    spec = importlib.util.spec_from_file_location("m150", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# STATISTICS
# ----------------------------------------------------------------------

def pair_stats(R, max_pairs=20000, max_cols=400, rng=None):
    """Statistics on cross-set similarity.

    Against a degree-preserving null the MEANS are invariant by construction --
    fixing both margins fixes mean co-occurrence almost exactly. Only the spread
    and the tail carry signal, so those are the primary statistics. Benson et
    al.'s mean co-occurrence test has power only because their null leaves
    element frequencies free.
    """
    rng = rng or np.random.default_rng(0)
    n = R.shape[0]
    if n < 2:
        return {k: float("nan") for k in
                ("j_mean", "j_var", "j_tail", "co_mean", "co_var")}

    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    ok = i != j
    A, B = R[i[ok]], R[j[ok]]
    inter = np.logical_and(A, B).sum(1)
    union = np.logical_or(A, B).sum(1)
    v = union > 0
    J = inter[v] / union[v] if v.sum() else np.array([np.nan])

    colsum = R.sum(0)
    keep = np.argsort(-colsum)[:max_cols]
    S = R[:, keep].astype(np.float32)
    co = S.T @ S
    iu = np.triu_indices(co.shape[0], k=1)
    cov = co[iu]

    return {
        "j_mean": float(np.nanmean(J)),
        "j_var": float(np.nanvar(J)),
        "j_tail": float(np.nanmean(J > 0.15)),
        "co_mean": float(cov.mean()),
        "co_var": float(cov.var()),
    }


# ----------------------------------------------------------------------
# DEGREE-PRESERVING NULL (curveball)
# ----------------------------------------------------------------------

def curveball(R, n_trades=None, rng=None):
    """Randomise a binary matrix holding both row and column sums fixed.

    Repeatedly picks two rows and swaps a random subset of the mutations unique
    to each. Row sums are preserved by construction; column sums are preserved
    because each swap moves a mutation out of one row and into the other.
    """
    rng = rng or np.random.default_rng(0)
    n, m = R.shape
    rows = [set(np.flatnonzero(r).tolist()) for r in R]
    n_trades = n_trades or 5 * n

    for _ in range(n_trades):
        a, b = rng.integers(0, n, 2)
        if a == b:
            continue
        A, B = rows[a], rows[b]
        only_a = A - B
        only_b = B - A
        if not only_a or not only_b:
            continue
        pool = list(only_a | only_b)
        rng.shuffle(pool)
        k = len(only_a)
        new_a = set(pool[:k])
        new_b = set(pool[k:])
        shared = A & B
        rows[a] = shared | new_a
        rows[b] = shared | new_b

    out = np.zeros_like(R)
    for i, s in enumerate(rows):
        if s:
            out[i, list(s)] = True
    return out


# ----------------------------------------------------------------------

def analyse_month(month, theta, n_null, rng, V):
    sets = month["sets"]
    X = np.zeros((len(sets), V), dtype=bool)
    for i, s in enumerate(sets):
        if s:
            X[i, list(s)] = True

    prevalence = X.mean(0)
    core = np.flatnonzero(prevalence >= theta)
    non_core = np.flatnonzero((prevalence < theta) & (prevalence > 0))
    if non_core.size < 2:
        return None
    R = X[:, non_core]
    keep_rows = R.sum(1) > 0
    R = R[keep_rows]
    if R.shape[0] < 10:
        return None

    real = pair_stats(R, rng=rng)
    nulls = [pair_stats(curveball(R, rng=rng), rng=rng) for _ in range(n_null)]

    out = {
        "month": month["month"],
        "n_sets": int(R.shape[0]),
        "core_size": int(core.size),
        "non_core_size": int(non_core.size),
        "median_set_size": float(np.median(X.sum(1))),
        "median_residual_size": float(np.median(R.sum(1))),
    }
    for k, v in real.items():
        nv = [d[k] for d in nulls]
        out[f"{k}_real"] = v
        out[f"{k}_null"] = float(np.nanmean(nv)) if nv else float("nan")
        out[f"{k}_null_sd"] = float(np.nanstd(nv)) if nv else float("nan")
        denom = out[f"{k}_null"]
        out[f"{k}_ratio"] = (v / denom) if denom else float("nan")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--months", default="2020-06:2024-12")
    p.add_argument("--top", type=int, default=500)
    p.add_argument("--theta", type=float, default=0.9,
                   help="prevalence above which a mutation counts as core")
    p.add_argument("--theta-sweep", default=None,
                   help="comma-separated thetas, e.g. 0.5,0.7,0.9,0.99")
    p.add_argument("--n-null", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results_152")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    m150 = _load_150(os.path.dirname(os.path.abspath(__file__)))
    months = m150.load_months(a.data, month_range=a.months, top=a.top)
    V = 1 + max(max(s) for mo in months for s in mo["sets"] if s)
    rng = np.random.default_rng(a.seed)

    thetas = ([float(x) for x in a.theta_sweep.split(",")]
              if a.theta_sweep else [a.theta])

    all_out = {}
    for th in thetas:
        recs = []
        print(f"\ntheta={th}")
        for mo in months:
            r = analyse_month(mo, th, a.n_null, rng, V)
            if r is None:
                continue
            recs.append(r)
            print(f"  {r['month']}  core={r['core_size']:>4} "
                  f"resid={r['median_residual_size']:>5.1f}  "
                  f"Jvar x{r['j_var_ratio']:.2f}  "
                  f"Jtail x{r['j_tail_ratio']:.2f}  "
                  f"CoVar x{r['co_var_ratio']:.2f}", flush=True)
        all_out[str(th)] = recs
        if recs:
            for k in ("j_var", "j_tail", "co_var", "j_mean", "co_mean"):
                v = np.array([r[f"{k}_ratio"] for r in recs], dtype=float)
                print(f"  -- theta={th}  {k:<8} mean ratio "
                      f"{np.nanmean(v):.2f}  min {np.nanmin(v):.2f}  "
                      f"max {np.nanmax(v):.2f}")

    with open(os.path.join(a.out, "horizontal.json"), "w") as f:
        json.dump(all_out, f)
    print(f"\nwrote {a.out}/horizontal.json")


if __name__ == "__main__":
    main()
