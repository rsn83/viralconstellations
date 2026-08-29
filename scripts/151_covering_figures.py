#!/usr/bin/env python3
"""
151_covering_figures.py -- assemble 150's outputs into a data-analysis figure.

Reads records_h*.json from the results directories produced by 150 and builds a
six-panel figure in the style of a data-analysis section: each panel answers one
question, and each has its matched null where a null applies.

USAGE
    python scripts/151_covering_figures.py \
        --h1 results_150_h1/records_h1.json \
        --h3 results_150_h3/records_h3.json \
        --h6 results_150_timing/records_h6.json \
        --h6-wide results_150_top5000/records_h6.json \
        --h6-2021 results_150_2021/records_h6.json \
        --out figures/
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_REAL = "#2a6fb5"
C_NULL = "#c44"
C_ALT = "#7a4"


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def arr(recs, key):
    return np.array([r[key] for r in recs], dtype=float)


def ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n_boot, len(x))).mean(axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h1"); p.add_argument("--h3"); p.add_argument("--h6")
    p.add_argument("--h6-wide"); p.add_argument("--h6-2021")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    R = {h: load(getattr(a, f"h{h}")) for h in (1, 3, 6)}
    R_wide, R_2021 = load(a.h6_wide), load(a.h6_2021)
    if R[6] is None:
        raise SystemExit("need at least --h6")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

    # ---- A. residual distribution by horizon -------------------------------
    hs = [h for h in (1, 3, 6) if R[h]]
    maxr = int(max(arr(R[h], "n_residual").max() for h in hs))
    bins = np.arange(-0.5, min(maxr, 8) + 1.5)
    for h, c in zip(hs, [C_REAL, C_ALT, C_NULL]):
        v = arr(R[h], "n_residual")
        ax[0, 0].hist(v, bins=bins, density=True, histtype="step", lw=2,
                      color=c, label=f"h={h}")
    ax[0, 0].set_xlabel("mutations not covered by best single haplotype")
    ax[0, 0].set_ylabel("fraction of novel constellations")
    ax[0, 0].set_title("A. Residual after one parent")
    ax[0, 0].legend()

    # ---- B. residual vs horizon: the accumulation rate ----------------------
    xs, ms, los, his = [], [], [], []
    for h in hs:
        v = arr(R[h], "n_residual")
        lo, hi = ci(v)
        xs.append(h); ms.append(v.mean()); los.append(lo); his.append(hi)
    ax[0, 1].errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                      marker="o", color=C_REAL, capsize=3)
    if len(xs) > 1:
        slope = np.polyfit(xs, ms, 1)[0]
        gx = np.linspace(0, max(xs), 10)
        ax[0, 1].plot(gx, np.polyval(np.polyfit(xs, ms, 1), gx),
                      ls="--", color="grey",
                      label=f"{slope:.2f} mutations / month")
        ax[0, 1].legend()
    ax[0, 1].set_xlabel("forecast horizon h (months)")
    ax[0, 1].set_ylabel("mean residual (mutations)")
    ax[0, 1].set_title("B. Residual grows linearly with horizon")

    # ---- C. minimum sources, real vs matched null (h=6) --------------------
    kr, kn = arr(R[6], "k_real"), arr(R[6], "k_null_mean")
    b2 = np.arange(0.5, max(8, kr.max(), kn.max()) + 1.5)
    ax[0, 2].hist(kr, bins=b2, density=True, alpha=.65, color=C_REAL,
                  label="real")
    ax[0, 2].hist(kn, bins=b2, density=True, alpha=.65, color=C_NULL,
                  label="rank-matched null")
    ax[0, 2].set_xlabel("minimum number of source haplotypes")
    ax[0, 2].set_ylabel("fraction")
    ax[0, 2].set_title("C. Real vs null (h=6)")
    ax[0, 2].legend()

    # ---- D. residual vs constellation size ---------------------------------
    # Flat = residual is absolute, not proportional to background size.
    groups = [("2022-23", R[6], C_REAL)]
    if R_2021:
        groups.append(("2021", R_2021, C_ALT))
    for label, recs, c in groups:
        sz, rs = arr(recs, "size"), arr(recs, "n_residual")
        edges = np.unique(np.percentile(sz, [0, 25, 50, 75, 100]))
        cx, cy, ce = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (sz >= lo) & (sz <= hi)
            if sel.sum() < 5:
                continue
            cx.append(sz[sel].mean()); cy.append(rs[sel].mean())
            ce.append(rs[sel].std() / np.sqrt(sel.sum()))
        ax[1, 0].errorbar(cx, cy, yerr=ce, marker="o", color=c, label=label,
                          capsize=3)
    ax[1, 0].set_xlabel("constellation size (mutations)")
    ax[1, 0].set_ylabel("mean residual")
    ax[1, 0].set_title("D. Residual is absolute, not proportional")
    ax[1, 0].legend()

    # ---- E. fraction covered by the single best haplotype ------------------
    for label, recs, c in groups:
        v = arr(recs, "best_single_frac")
        ax[1, 1].hist(v, bins=np.linspace(0.5, 1.0, 26), density=True,
                      histtype="step", lw=2, color=c, label=label)
    ax[1, 1].set_xlabel("fraction of C covered by best single haplotype")
    ax[1, 1].set_ylabel("density")
    ax[1, 1].set_title("E. One haplotype covers nearly all of C")
    ax[1, 1].legend()

    # ---- F. robustness across configurations -------------------------------
    cfg = [("h=1", R[1]), ("h=3", R[3]), ("h=6", R[6]),
           ("h=6\ntop-5000", R_wide), ("h=6\n2021", R_2021)]
    cfg = [(k, v) for k, v in cfg if v]
    xs = np.arange(len(cfg))
    ms, los, his = [], [], []
    for _, recs in cfg:
        v = arr(recs, "n_residual")
        lo, hi = ci(v)
        ms.append(v.mean()); los.append(lo); his.append(hi)
    ax[1, 2].bar(xs, ms, color=C_REAL, alpha=.75)
    ax[1, 2].errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                      fmt="none", ecolor="k", capsize=3)
    ax[1, 2].set_xticks(xs)
    ax[1, 2].set_xticklabels([k for k, _ in cfg], fontsize=8)
    ax[1, 2].set_ylabel("mean residual")
    ax[1, 2].set_title("F. Robustness across configurations")

    fig.tight_layout()
    out = os.path.join(a.out, "covering_summary.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # ---- table for the slide ----------------------------------------------
    print(f"\n{'config':<16}{'n':>6}{'resid':>8}{'k_real':>8}"
          f"{'k_null':>8}{'1-src':>8}")
    for k, recs in cfg:
        v = arr(recs, "n_residual")
        print(f"{k.replace(chr(10), ' '):<16}{len(recs):>6}{v.mean():>8.2f}"
              f"{arr(recs,'k_real').mean():>8.2f}"
              f"{arr(recs,'k_null_mean').mean():>8.2f}"
              f"{(arr(recs,'k_real')==1).mean():>8.2f}")


if __name__ == "__main__":
    main()
