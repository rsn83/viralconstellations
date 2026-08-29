#!/usr/bin/env python3
"""
151_covering_figures.py -- assemble 150's outputs into a data-analysis figure.

TERMINOLOGY
-----------
variant            a set of spike mutations (150 calls this a constellation)
new variant        a variant appearing at t+h that was absent in every month <= t
nearest variant    among variants present at time <= t, the one sharing the most
                   mutations with a given new variant
residual           mutations in a new variant that are absent from its nearest
                   variant

Each new variant is measured against its own nearest variant. There is no single
reference variant per month.

CONVENTIONS
-----------
- y axes are fractions of new variants, never kernel densities
- horizons are compared only on calendar months where every horizon is defined,
  so h=1 and h=6 means are not computed over different periods
- time is shown continuously; no era boundaries are imposed

USAGE
    python scripts/151_covering_figures.py \
        --h1 results_150_h1/records_h1.json \
        --h3 results_150_h3/records_h3.json \
        --h6 results_150_h6/records_h6.json \
        --h6-wide results_150_top5000/records_h6.json \
        --out figures/
"""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_OBS = "#2a6fb5"
C_NULL = "#c44"
C_H = {1: "#2a6fb5", 3: "#e08a1e", 6: "#8b3a8b"}


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def arr(recs, key):
    return np.array([r[key] for r in recs], dtype=float)


def ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n_boot, x.size)).mean(axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def match_windows(R):
    """Restrict every horizon to the calendar months all horizons share.

    Without this, h=1 averages over later months than h=6 (t+h must land inside
    the range), so part of any horizon trend would be calendar drift in variant
    size rather than horizon.
    """
    present = [set(r["t"] for r in recs) for recs in R.values() if recs]
    if not present:
        return R, set()
    common = set.intersection(*present)
    return ({h: ([r for r in recs if r["t"] in common] if recs else None)
             for h, recs in R.items()}, common)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h1"); p.add_argument("--h3"); p.add_argument("--h6")
    p.add_argument("--h6-wide")
    p.add_argument("--no-match", action="store_true",
                   help="do not restrict horizons to shared calendar windows")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    R_all = OrderedDict((h, load(getattr(a, f"h{h}"))) for h in (1, 3, 6))
    R_wide = load(a.h6_wide)
    if R_all[6] is None:
        raise SystemExit("need at least --h6")

    if a.no_match:
        R, common = R_all, set()
    else:
        R, common = match_windows(R_all)
        print(f"matched windows ({len(common)}): {sorted(common)}")

    hs = [h for h in (1, 3, 6) if R[h]]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

    # ---- A. residual distribution by horizon -------------------------------
    maxr = int(max(arr(R[h], "n_residual").max() for h in hs))
    edges = np.arange(-0.5, min(maxr, 8) + 1.5)
    centers = (edges[:-1] + edges[1:]) / 2
    width = 0.8 / len(hs)
    for i, h in enumerate(hs):
        v = arr(R[h], "n_residual")
        frac = np.histogram(v, bins=edges)[0] / len(v)   # fraction, not density
        ax[0, 0].bar(centers + (i - (len(hs) - 1) / 2) * width, frac,
                     width=width, color=C_H[h], label=f"h={h}")
    ax[0, 0].set_xlabel("mutations not in the nearest existing variant")
    ax[0, 0].set_ylabel("fraction of new variants")
    ax[0, 0].set_title("A. Most new variants differ by 0-1 mutations")
    ax[0, 0].legend()

    # ---- B. residual vs horizon, matched windows ---------------------------
    xs, ms, los, his = [], [], [], []
    for h in hs:
        v = arr(R[h], "n_residual")
        lo, hi = ci(v)
        xs.append(h); ms.append(v.mean()); los.append(lo); his.append(hi)
    ax[0, 1].errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                      marker="o", color=C_OBS, capsize=3, lw=2)
    if len(xs) > 1:
        coef = np.polyfit(xs, ms, 1)
        gx = np.linspace(0, max(xs), 10)
        ax[0, 1].plot(gx, np.polyval(coef, gx), ls="--", color="grey",
                      label=f"slope {coef[0]:.2f} mutations/month")
        ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_xlabel("forecast horizon h (months)")
    ax[0, 1].set_ylabel("mean residual (mutations)")
    ax[0, 1].set_title("B. Residual grows with horizon\n"
                       "same calendar windows at every h")

    # ---- C. minimum source variants, observed vs null ----------------------
    kr, kn = arr(R[6], "k_real"), arr(R[6], "k_null_mean")
    edges2 = np.arange(0.5, max(8, kr.max(), kn.max()) + 1.5)
    c2 = (edges2[:-1] + edges2[1:]) / 2
    fr = np.histogram(kr, bins=edges2)[0] / len(kr)
    fn = np.histogram(kn, bins=edges2)[0] / len(kn)
    ax[0, 2].bar(c2 - 0.2, fr, width=0.4, color=C_OBS, label="observed")
    ax[0, 2].bar(c2 + 0.2, fn, width=0.4, color=C_NULL,
                 label="rank-matched null")
    ax[0, 2].set_xlabel("minimum number of existing variants needed")
    ax[0, 2].set_ylabel("fraction of new variants")
    ax[0, 2].set_title("C. Fewer sources than chance (h=6)\n"
                       "but per A, the extra source supplies ~1 mutation")
    ax[0, 2].legend(fontsize=8)

    # ---- D. residual over calendar time, continuous ------------------------
    # one shared month axis, so lines from different horizons are aligned
    all_months = sorted({r["t"] for h in hs for r in R_all[h]})
    pos = {m: i for i, m in enumerate(all_months)}
    for h in hs:
        recs = R_all[h]
        months = sorted(set(r["t"] for r in recs))
        mu = [np.mean([r["n_residual"] for r in recs if r["t"] == m])
              for m in months]
        ax[1, 0].plot([pos[m] for m in months], mu, marker=".", color=C_H[h],
                      label=f"h={h}")
    step = max(1, len(all_months) // 6)
    ax[1, 0].set_xticks(range(0, len(all_months), step))
    ax[1, 0].set_xticklabels([all_months[i] for i in
                              range(0, len(all_months), step)],
                             rotation=45, ha="right", fontsize=7)
    ax[1, 0].set_ylabel("mean residual (mutations)")
    ax[1, 0].set_title("D. Residual over time (no era boundaries)")
    ax[1, 0].legend(fontsize=8)

    # ---- E. residual vs variant size, coloured continuously by time --------
    recs = R_all[6]
    sz, rs = arr(recs, "size"), arr(recs, "n_residual")
    months = sorted(set(r["t"] for r in recs))
    tix = np.array([months.index(r["t"]) for r in recs], dtype=float)
    jitter = np.random.default_rng(0).normal(0, .06, rs.size)
    sc = ax[1, 1].scatter(sz, rs + jitter, c=tix, cmap="viridis", s=8, alpha=.6)
    cb = fig.colorbar(sc, ax=ax[1, 1])
    cb.set_ticks([0, len(months) - 1])
    cb.set_ticklabels([months[0], months[-1]], fontsize=7)
    edges3 = np.unique(np.percentile(sz, [0, 20, 40, 60, 80, 100]))
    bx, by = [], []
    for lo, hi in zip(edges3[:-1], edges3[1:]):
        sel = (sz >= lo) & (sz <= hi)
        if sel.sum() >= 5:
            bx.append(sz[sel].mean()); by.append(rs[sel].mean())
    ax[1, 1].plot(bx, by, color="k", lw=2, marker="o", ms=4)
    ax[1, 1].set_xlabel("variant size (mutations)")
    ax[1, 1].set_ylabel("residual (mutations, jittered)")
    ax[1, 1].set_title("E. Residual is absolute, not proportional to size")

    # ---- F. robustness -----------------------------------------------------
    cfg = [(f"h={h}", R[h]) for h in hs]
    if R_wide:
        cfg.append(("h=6\ntop-5000", R_wide))
    xs = np.arange(len(cfg))
    ms, los, his = [], [], []
    for _, recs in cfg:
        v = arr(recs, "n_residual")
        lo, hi = ci(v)
        ms.append(v.mean()); los.append(lo); his.append(hi)
    ax[1, 2].bar(xs, ms, color=C_OBS, alpha=.8)
    ax[1, 2].errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                      fmt="none", ecolor="k", capsize=3)
    ax[1, 2].set_xticks(xs)
    ax[1, 2].set_xticklabels([k for k, _ in cfg], fontsize=8)
    ax[1, 2].set_ylabel("mean residual (mutations)")
    ax[1, 2].set_title("F. Robustness across configurations")

    fig.tight_layout()
    out = os.path.join(a.out, "covering_summary.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    print(f"\n{'config':<16}{'n':>6}{'resid':>8}{'k_obs':>8}"
          f"{'k_null':>8}{'1-src':>8}")
    for k, recs in cfg:
        v = arr(recs, "n_residual")
        print(f"{k.replace(chr(10), ' '):<16}{len(recs):>6}{v.mean():>8.2f}"
              f"{arr(recs,'k_real').mean():>8.2f}"
              f"{arr(recs,'k_null_mean').mean():>8.2f}"
              f"{(arr(recs,'k_real')==1).mean():>8.2f}")


if __name__ == "__main__":
    main()
