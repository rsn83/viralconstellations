#!/usr/bin/env python
"""
25_analyse_source_dependence.py

Reads outputs/24_source_dependence.csv and answers the two questions that
decide whether the +0.0195 mean AUC gain is worth building an edit model for.

  Q1  Is the gain UNIFORM or CONCENTRATED in particular months?
      Uniform +0.02 everywhere = a weak general effect; the edit model would
      win by very little. Concentrated in high-turnover months = a regime
      effect, and background dependence mattering exactly when the population
      is diversifying is a more interesting claim than a flat small edge.

  Q2  Is the gain SUPPRESSED BY SPARSE DATA?
      The source-affinity term is a leave-one-out centroid over the sources
      that added each mutation. With few adopters it is noisy and biased
      toward zero, so 0.0195 may be a floor rather than the true effect.
      If gain rises with n_additions, the dense months are the honest estimate.

Usage
-----
  python scripts/25_analyse_source_dependence.py
  python scripts/25_analyse_source_dependence.py --csv outputs/24_source_dependence.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "24_source_dependence.csv"))
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    d["yr"] = d["month_t"].astype(str).str[:4]
    n = len(d)
    g = d["auc_gain"]

    log("=" * 78)
    log(f"OVERALL  ({n} month pairs)")
    log("=" * 78)
    log(f"  mean gain      {g.mean():+.4f}")
    log(f"  median gain    {g.median():+.4f}")
    log(f"  sd             {g.std():.4f}")
    log(f"  positive in    {(g > 0).sum()}/{n}  ({(g > 0).mean():.1%})")
    # sign test against a coin flip -- is "positive most months" itself unlikely?
    k = int((g > 0).sum())
    try:
        from scipy.stats import binomtest
        p = binomtest(k, n, 0.5, alternative="greater").pvalue
        log(f"  sign test p    {p:.2e}   (H0: gain is symmetric around 0)")
    except Exception:
        pass
    log(f"  range          {g.min():+.4f} .. {g.max():+.4f}")

    log("\n" + "=" * 78)
    log("Q1  IS THE GAIN UNIFORM OR CONCENTRATED?")
    log("=" * 78)
    log("\n  by year:")
    by_yr = d.groupby("yr").agg(
        pairs=("auc_gain", "size"),
        gain=("auc_gain", "mean"),
        adds=("n_additions", "mean"),
        srcs=("n_sources", "mean"),
        top10=("top10_share", "mean"),
        spawn=("frac_sources_spawn", "mean"),
    ).round(4)
    log(by_yr.to_string())

    log("\n  gain quintiles (are the top months carrying the average?):")
    d["q"] = pd.qcut(d["auc_gain"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    log(d.groupby("q", observed=True).agg(
        pairs=("auc_gain", "size"),
        gain=("auc_gain", "mean"),
        adds=("n_additions", "mean"),
        srcs=("n_sources", "mean"),
        pair_rate=("pair_rate", "mean"),
        top10=("top10_share", "mean"),
    ).round(4).to_string())

    top = d.nlargest(8, "auc_gain")[
        ["month_t", "auc_gain", "n_additions", "n_sources", "top10_share"]]
    log("\n  highest-gain months:")
    log(top.round(4).to_string(index=False))

    # concentration: what share of total gain comes from the top 20% of months?
    srt = np.sort(g.values)[::-1]
    k20 = max(1, int(0.2 * n))
    share = srt[:k20][srt[:k20] > 0].sum() / max(srt[srt > 0].sum(), 1e-9)
    log(f"\n  top 20% of months contribute {share:.1%} of all positive gain")
    log("  (20% would be perfectly uniform; >50% means concentrated)")

    log("\n" + "=" * 78)
    log("Q2  IS THE GAIN SUPPRESSED BY SPARSE DATA?")
    log("=" * 78)
    cols = ["n_additions", "n_sources", "n_valid_pairs", "pair_rate",
            "n_distinct_muts_added", "top10_share", "frac_sources_spawn",
            "auc_marginal"]
    cols = [c for c in cols if c in d.columns]
    log("\n  Spearman correlation of auc_gain with:")
    for c in cols:
        r = d[["auc_gain", c]].corr(method="spearman").iloc[0, 1]
        log(f"    {c:<24} {r:+.3f}")

    log("\n  gain by data volume (n_additions quintile):")
    d["vol"] = pd.qcut(d["n_additions"], 5,
                       labels=["V1 sparse", "V2", "V3", "V4", "V5 dense"])
    vol = d.groupby("vol", observed=True).agg(
        pairs=("auc_gain", "size"),
        gain=("auc_gain", "mean"),
        adds=("n_additions", "mean"),
        srcs=("n_sources", "mean"),
    ).round(4)
    log(vol.to_string())

    dense = d[d["n_additions"] >= d["n_additions"].quantile(0.6)]["auc_gain"]
    log(f"\n  gain on the densest 40% of months: {dense.mean():+.4f} "
        f"(n={len(dense)}, positive in {(dense > 0).sum()})")

    log("\n" + "=" * 78)
    log("READ")
    log("=" * 78)
    mg, dg = g.mean(), dense.mean()
    conc = share
    vol_r = d[["auc_gain", "n_additions"]].corr(method="spearman").iloc[0, 1]

    log(f"  overall {mg:+.4f} | densest 40% {dg:+.4f} | "
        f"gain~volume rho {vol_r:+.3f} | top-20% share {conc:.1%}")
    log("")
    if dg > mg + 0.005 and vol_r > 0.2:
        log("  Gain rises with data volume -> 0.0195 is a FLOOR, depressed by the")
        log("  leave-one-out affinity being noisy in sparse months. Use the dense")
        log("  estimate as the honest effect size.")
    elif vol_r < -0.2:
        log("  Gain FALLS with volume. That is the wrong direction for a real")
        log("  effect and suggests the affinity term is picking up small-sample")
        log("  structure. Treat 0.0195 as an overestimate.")
    else:
        log("  Gain is roughly flat in data volume, so 0.0195 is not a sparsity")
        log("  artefact in either direction.")
    log("")
    if conc > 0.5:
        log("  Gain is CONCENTRATED in a minority of months -> regime effect.")
        log("  Background dependence appears to matter in specific periods rather")
        log("  than uniformly. Check the highest-gain months above against known")
        log("  variant transitions before claiming this.")
    else:
        log("  Gain is spread fairly evenly across months -> a weak GENERAL effect,")
        log("  not a regime one. An edit model would win consistently but by little.")
    log("")
    log("  Either way: background dependence is measurable and second-order")
    log("  relative to marginal frequency. That is a reportable finding on the")
    log("  central hypothesis, arrived at by direct measurement rather than")
    log("  inferred from a model underperforming.")


if __name__ == "__main__":
    main()
