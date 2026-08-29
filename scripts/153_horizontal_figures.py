#!/usr/bin/env python3
"""
153_horizontal_figures.py -- show the horizontal-structure claim directly.

The claim is about the TAIL of the cross-set similarity distribution: after the
shared core is removed, some pairs of mutation sets are far more similar than
independent assortment at the same margins would produce. A ratio hides that;
the distribution shows it.

Panels:
  A  Jaccard between pairs of mutation sets, BEFORE core removal -- real vs null.
     High similarity here is mostly the shared core.
  B  Jaccard AFTER core removal -- real vs null, log y. Separation in the right
     tail is the claim.
  C  Tail ratio over time, from 152's output.

USAGE
    python scripts/153_horizontal_figures.py \
        --data data/processed/full_data_graphs_withdel \
        --month 2022-12 --json results_152/horizontal.json --out figures/
"""

import argparse
import json
import os
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_REAL = "#2a6fb5"
C_NULL = "#c44"


def load_mod(script_dir, name):
    path = os.path.join(script_dir, name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jaccard_sample(R, n_pairs, rng):
    n = R.shape[0]
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    ok = i != j
    A, B = R[i[ok]], R[j[ok]]
    inter = np.logical_and(A, B).sum(1)
    union = np.logical_or(A, B).sum(1)
    v = union > 0
    return inter[v] / union[v]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--month", default="2022-12")
    p.add_argument("--theta", type=float, default=0.9)
    p.add_argument("--n-null", type=int, default=5)
    p.add_argument("--n-pairs", type=int, default=40000)
    p.add_argument("--json", default=None, help="results_152/horizontal.json")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    here = os.path.dirname(os.path.abspath(__file__))
    m150 = load_mod(here, "150_covering_null.py")
    m152 = load_mod(here, "152_horizontal_structure.py")
    rng = np.random.default_rng(0)

    months = m150.load_months(a.data, month_range=f"{a.month}:{a.month}",
                              top=500)
    if not months:
        raise SystemExit(f"month {a.month} not found")
    mo = months[0]
    V = 1 + max(max(s) for s in mo["sets"] if s)

    X = np.zeros((len(mo["sets"]), V), dtype=bool)
    for i, s in enumerate(mo["sets"]):
        if s:
            X[i, list(s)] = True
    prev = X.mean(0)
    core = np.flatnonzero(prev >= a.theta)
    non_core = np.flatnonzero((prev < a.theta) & (prev > 0))
    R = X[:, non_core]
    R = R[R.sum(1) > 0]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    bins = np.linspace(0, 1, 41)

    # ---- A: before core removal -------------------------------------------
    Xn = X[:, prev > 0]
    Xn = Xn[Xn.sum(1) > 0]
    j_raw = jaccard_sample(Xn, a.n_pairs, rng)
    j_raw_null = np.concatenate([
        jaccard_sample(m152.curveball(Xn, rng=rng), a.n_pairs // a.n_null, rng)
        for _ in range(a.n_null)])
    ax[0].hist(j_raw, bins=bins, density=True, alpha=.6, color=C_REAL,
               label="observed")
    ax[0].hist(j_raw_null, bins=bins, density=True, alpha=.6, color=C_NULL,
               label="null (margins fixed)")
    ax[0].set_xlabel("Jaccard between two mutation sets")
    ax[0].set_ylabel("density")
    ax[0].set_title(f"A. All mutations ({a.month})\n"
                    f"shared core inflates both")
    ax[0].legend(fontsize=8)

    # ---- B: after core removal, log y -------------------------------------
    j_res = jaccard_sample(R, a.n_pairs, rng)
    j_res_null = np.concatenate([
        jaccard_sample(m152.curveball(R, rng=rng), a.n_pairs // a.n_null, rng)
        for _ in range(a.n_null)])
    ax[1].hist(j_res, bins=bins, density=True, alpha=.6, color=C_REAL,
               label="observed")
    ax[1].hist(j_res_null, bins=bins, density=True, alpha=.6, color=C_NULL,
               label="null (margins fixed)")
    ax[1].axvline(0.15, color="k", ls=":", lw=1)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Jaccard after removing the shared core")
    ax[1].set_ylabel("density (log)")
    tail_r = (j_res > 0.15).mean()
    tail_n = (j_res_null > 0.15).mean()
    ratio = tail_r / tail_n if tail_n > 0 else float("inf")
    ax[1].set_title(f"B. Core removed: right tail is the claim\n"
                    f"pairs above 0.15  {tail_r:.3f} vs {tail_n:.3f} "
                    f"({ratio:.1f}x)")
    ax[1].legend(fontsize=8)

    # ---- C: tail ratio over time ------------------------------------------
    if a.json and os.path.exists(a.json):
        d = json.load(open(a.json))
        key = str(a.theta) if str(a.theta) in d else sorted(d)[0]
        recs = d[key]
        xs = np.arange(len(recs))
        yr = np.array([r["j_tail_ratio"] for r in recs], dtype=float)
        ax[2].plot(xs, yr, marker=".", color=C_REAL)
        ax[2].axhline(1.0, color="k", lw=1, ls="--")
        step = max(1, len(recs) // 6)
        ax[2].set_xticks(xs[::step])
        ax[2].set_xticklabels([recs[i]["month"] for i in
                               range(0, len(recs), step)],
                              rotation=45, ha="right", fontsize=7)
        ax[2].set_ylabel("tail ratio (observed / null)")
        ax[2].set_title(f"C. Structure over time (theta={key})\n"
                        "1.0 = no structure beyond the core")
    else:
        ax[2].text(.5, .5, "run 152 and pass --json", ha="center",
                   transform=ax[2].transAxes)
        ax[2].set_axis_off()

    fig.tight_layout()
    out = os.path.join(a.out, f"horizontal_{a.month}.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    print(f"\ncore size {core.size}, non-core {non_core.size}, "
          f"sets {R.shape[0]}")
    print(f"pairs above 0.15 after core removal: observed {tail_r:.4f}, "
          f"null {tail_n:.4f}, ratio {ratio:.1f}x")


if __name__ == "__main__":
    main()
