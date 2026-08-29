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


def month_curves(mo, theta, n_null, n_pairs, grid, rng, m152):
    """Survival curves for one month, before and after core removal."""
    V = 1 + max(max(s) for s in mo["sets"] if s)
    X = np.zeros((len(mo["sets"]), V), dtype=bool)
    for i, s in enumerate(mo["sets"]):
        if s:
            X[i, list(s)] = True
    prev = X.mean(0)
    core = np.flatnonzero(prev >= theta)
    non_core = np.flatnonzero((prev < theta) & (prev > 0))
    if non_core.size < 2:
        return None

    def surv(j):
        return np.array([(j > x).mean() for x in grid])

    Xn = X[:, prev > 0]
    Xn = Xn[Xn.sum(1) > 0]
    R = X[:, non_core]
    R = R[R.sum(1) > 0]
    if R.shape[0] < 10 or Xn.shape[0] < 10:
        return None

    per_null = max(1, n_pairs // max(1, n_null))
    out = {"month": mo["month"], "core_size": int(core.size),
           "median_residual_size": float(np.median(R.sum(1)))}
    for tag, M in (("raw", Xn), ("res", R)):
        j = jaccard_sample(M, n_pairs, rng)
        jn = np.concatenate([
            jaccard_sample(m152.curveball(M, rng=rng), per_null, rng)
            for _ in range(n_null)])
        out[f"{tag}_obs"] = surv(j)
        out[f"{tag}_null"] = surv(jn)
        out[f"{tag}_max"] = float(max(j.max(initial=0), jn.max(initial=0)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--months", default="2020-06:2024-12",
                   help="all months are pooled; no single month is privileged")
    p.add_argument("--theta", type=float, default=0.9)
    p.add_argument("--n-null", type=int, default=5)
    p.add_argument("--n-pairs", type=int, default=20000)
    p.add_argument("--json", default=None, help="results_152/horizontal.json")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    here = os.path.dirname(os.path.abspath(__file__))
    m150 = load_mod(here, "150_covering_null.py")
    m152 = load_mod(here, "152_horizontal_structure.py")
    rng = np.random.default_rng(0)

    months = m150.load_months(a.data, month_range=a.months, top=500)
    if not months:
        raise SystemExit("no months loaded")

    grid = np.linspace(0, 1, 201)
    curves = []
    for mo in months:
        c = month_curves(mo, a.theta, a.n_null, a.n_pairs, grid, rng, m152)
        if c:
            curves.append(c)
            print(f"  {c['month']}  core={c['core_size']:>4}  "
                  f"resid_size={c['median_residual_size']:.1f}", flush=True)
    if not curves:
        raise SystemExit("no usable months")
    print(f"pooled over {len(curves)} months")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    def panel(axis, tag, title, xlabel):
        """Faint per-month curves behind the pooled mean of both arms."""
        for c in curves:
            axis.plot(grid, c[f"{tag}_obs"], color=C_REAL, lw=.4, alpha=.18)
            axis.plot(grid, c[f"{tag}_null"], color=C_NULL, lw=.4, alpha=.18)
        obs = np.nanmean([c[f"{tag}_obs"] for c in curves], axis=0)
        nul = np.nanmean([c[f"{tag}_null"] for c in curves], axis=0)
        axis.plot(grid, obs, color=C_REAL, lw=2.4, label="observed (pooled)")
        axis.plot(grid, nul, color=C_NULL, lw=2.4, label="null (margins fixed)")
        axis.set_yscale("log")
        xmax = min(1.0, max(c[f"{tag}_max"] for c in curves) * 1.1)
        axis.set_xlim(0, xmax)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("fraction of variant pairs above x")
        axis.set_title(title)
        axis.legend(fontsize=8, loc="lower left")
        return obs, nul, xmax

    panel(ax[0], "raw",
          "A. All mutations\ncurves coincide: similarity is the shared core",
          "Jaccard between two variants, x")

    obs, nul, xmax = panel(
        ax[1], "res",
        "B. Core removed\ngap between the curves is the structure",
        "Jaccard after removing the shared core, x")

    ins = ax[1].inset_axes([0.58, 0.62, 0.38, 0.33])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_curve = np.where(nul > 0, obs / nul, np.nan)
    ins.plot(grid, ratio_curve, color="k", lw=1)
    ins.axhline(1, color="grey", ls="--", lw=.8)
    ins.set_yscale("log")
    ins.set_xlim(0, xmax)
    ins.set_title("observed / null", fontsize=7)
    ins.tick_params(labelsize=6)

    # ---- C: structure over time -------------------------------------------
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
    out = os.path.join(a.out, "horizontal_pooled.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    for x in (0.05, 0.10, 0.15, 0.20, 0.30):
        i = int(np.argmin(np.abs(grid - x)))
        r = obs[i] / nul[i] if nul[i] > 0 else float("inf")
        print(f"  pairs above {x:.2f}: observed {obs[i]:.4f}  "
              f"null {nul[i]:.4f}  ratio {r:.1f}x")


if __name__ == "__main__":
    main()
