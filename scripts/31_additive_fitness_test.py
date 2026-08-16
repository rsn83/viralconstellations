#!/usr/bin/env python
"""
31_additive_fitness_test.py

CPU, no model training, runs in minutes. Pure arithmetic on frequencies.

THE QUESTION
------------
Does a constellation's growth rate equal the SUM of its member mutations'
growth rates, or does the combination behave differently?

Additive fitness is the assumption behind every existing method in this space
-- Luksza & Lassig give a clade a fitness that is a sum of per-mutation terms
and project frequencies by exp(fitness x dt). If growth is additive, that
assumption holds and the epistasis story has nothing to add at the level that
matters for forecasting.

WHY THIS AND NOT SCRIPT 24
--------------------------
Script 24 measured whether the source set's composition predicts WHICH mutation
attaches -- an attachment question, within one month. It found +0.0195 AUC over
marginal frequency: real (37/38 months, p=1.4e-10) but small.

That is NOT the same quantity as epistasis in the fitness sense, which is about
GROWTH RATE over time. A pair of mutations could attach at exactly the marginal
rate and still grow much faster together than apart. Nothing so far has measured
that, and it is closer to the project's actual claim.

MEASUREMENT
-----------
For each constellation c present at t and t+1 with enough sequences:

  g_obs(c)  = log( freq(c, t+1) / freq(c, t) )          observed growth
  g_add(c)  = sum over m in c of  g(m) - (|c|-1) * gbar  additive prediction
              where g(m) = log( freq(m,t+1) / freq(m,t) )

  The (|c|-1)*gbar correction removes the mechanical inflation from summing
  |c| terms that each already include the population-wide drift; gbar is the
  count-weighted mean per-mutation growth that month.

  residual = g_obs - g_add

If fitness is additive, residual is noise: centered near zero, uncorrelated
with anything about the constellation, and g_add predicts g_obs well.

If it is not, residuals are structured -- they should correlate with set size,
persist for the same constellation across consecutive months (the same
combination is consistently better or worse than the sum of its parts), and
g_add should predict g_obs poorly.

FOUR CHECKS
-----------
  1. R^2 of g_add against g_obs, and against a naive baseline (predict gbar).
     Additive fitness makes a real prediction; does it beat predicting the mean?
  2. Residual autocorrelation: for constellations seen in consecutive month
     pairs, does residual(t) predict residual(t+1)? Noise does not persist;
     a real combination effect does. THIS IS THE DECISIVE CHECK.
  3. Residual vs set size: additive error should not grow systematically with
     the number of mutations if the model is right.
  4. Per-era breakdown, since sampling density confounds everything else in
     this project and must be checked here too.

Usage
-----
  python scripts/31_additive_fitness_test.py
  python scripts/31_additive_fitness_test.py --min_seqs 5000 --min_count 10
"""

import argparse
import pickle
from collections import Counter
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--min_count", type=int, default=10,
                    help="a constellation needs at least this many sequences in BOTH "
                         "months for its growth rate to be meaningful. Ratios of small "
                         "counts are dominated by sampling noise.")
    ap.add_argument("--min_seqs", type=int, default=5000,
                    help="skip months below this sequencing depth")
    ap.add_argument("--end_month", type=str, default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "31_additive_fitness.csv"))
    args = ap.parse_args()

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    months = [m for m in months if m <= args.end_month]

    cache = {}

    def H(mo):
        if mo not in cache:
            with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
                raw = constellations_of(pickle.load(fh))
            filt = {c: v for c, v in raw.items() if 1 <= len(c) <= args.max_set_size}
            tot = sum(filt.values())
            cf = {c: v / max(tot, 1) for c, v in filt.items()}
            mc = Counter()
            for c, v in filt.items():
                for m in c:
                    mc[m] += v
            mf = {m: v / max(tot, 1) for m, v in mc.items()}
            cache[mo] = (filt, cf, mf, tot)
        return cache[mo]

    log(f"{len(months)} months, min_count={args.min_count}, "
        f"min_seqs={args.min_seqs}\n")

    rows = []
    for i in range(len(months) - 1):
        mt, mt1 = months[i], months[i + 1]
        Ht, cft, mft, tot_t = H(mt)
        Ht1, cft1, mft1, tot_t1 = H(mt1)
        if tot_t < args.min_seqs or tot_t1 < args.min_seqs:
            continue

        # per-mutation log growth, only for mutations with enough support
        gm = {}
        for m, f0 in mft.items():
            f1 = mft1.get(m, 0.0)
            if f0 > 0 and f1 > 0:
                gm[m] = np.log(f1 / f0)
        if len(gm) < 20:
            continue
        # count-weighted mean per-mutation growth: the population-wide drift
        w = np.array([mft[m] for m in gm])
        gbar = float(np.average([gm[m] for m in gm], weights=w))

        for c, v0 in Ht.items():
            if v0 < args.min_count or len(c) < 2:
                continue
            v1 = Ht1.get(c, 0)
            if v1 < args.min_count:
                continue
            if not all(m in gm for m in c):
                continue
            g_obs = np.log(cft1[c] / cft[c])
            g_add = sum(gm[m] for m in c) - (len(c) - 1) * gbar
            rows.append(dict(month_t=mt, month_t1=mt1, year=mt[:4],
                             key="|".join(str(x) for x in sorted(c)),
                             size=len(c), count_t=v0, count_t1=v1,
                             n_seqs_t=tot_t, gbar=gbar,
                             g_obs=g_obs, g_add=g_add,
                             residual=g_obs - g_add))
        if rows and rows[-1]["month_t"] == mt:
            n = sum(1 for r in rows if r["month_t"] == mt)
            log(f"  {mt} -> {mt1}  n_seqs={tot_t:>8}  constellations={n:>5}")

    if len(rows) < 50:
        raise SystemExit(f"only {len(rows)} usable observations; "
                         f"lower --min_count or --min_seqs")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log(f"\n{'=' * 76}")
    log(f"{len(df)} constellation-month observations, "
        f"{df['key'].nunique()} distinct constellations, "
        f"{df['month_t'].nunique()} month pairs")
    log("=" * 76)

    # ---- CHECK 1: does additive fitness predict observed growth? ----
    ss_res = float(((df.g_obs - df.g_add) ** 2).sum())
    ss_tot = float(((df.g_obs - df.g_obs.mean()) ** 2).sum())
    r2_add = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    ss_naive = float(((df.g_obs - df.gbar) ** 2).sum())
    r2_naive = 1 - ss_naive / ss_tot if ss_tot > 0 else np.nan
    rho = float(df[["g_obs", "g_add"]].corr(method="spearman").iloc[0, 1])

    log("\n1. DOES ADDITIVE FITNESS PREDICT GROWTH?")
    log(f"   R^2 of g_add                 {r2_add:+.4f}")
    log(f"   R^2 of predicting gbar       {r2_naive:+.4f}   (naive baseline)")
    log(f"   Spearman(g_obs, g_add)       {rho:+.4f}")
    log(f"   residual mean {df.residual.mean():+.4f}  sd {df.residual.std():.4f}")
    log(f"   sd of g_obs   {df.g_obs.std():.4f}")

    # ---- CHECK 2: do residuals persist for the same constellation? ----
    log("\n2. DO RESIDUALS PERSIST?  (the decisive check)")
    log("   Noise does not repeat. If the SAME constellation is consistently")
    log("   above or below its additive prediction in consecutive months, the")
    log("   deviation is a property of the combination, not sampling error.")
    d = df.sort_values(["key", "month_t"]).copy()
    d["prev_res"] = d.groupby("key")["residual"].shift(1)
    d["prev_month"] = d.groupby("key")["month_t"].shift(1)
    idx = {m: k for k, m in enumerate(months)}
    d["consec"] = [
        (isinstance(p, str) and idx.get(m, -99) - idx.get(p, 99) == 1)
        for m, p in zip(d.month_t, d.prev_month)]
    pair = d[d.consec & d.prev_res.notna()]
    if len(pair) >= 30:
        ac_p = float(pair[["residual", "prev_res"]].corr().iloc[0, 1])
        ac_s = float(pair[["residual", "prev_res"]].corr(method="spearman").iloc[0, 1])
        log(f"   n consecutive pairs          {len(pair)}")
        log(f"   autocorr (Pearson)           {ac_p:+.4f}")
        log(f"   autocorr (Spearman)          {ac_s:+.4f}")
    else:
        ac_s = np.nan
        log(f"   only {len(pair)} consecutive pairs -- too few")

    # ---- CHECK 3: does additive error grow with set size? ----
    log("\n3. RESIDUAL vs SET SIZE")
    log(f"   {'size':>6}{'n':>7}{'mean_res':>10}{'sd_res':>9}{'mean|res|':>11}")
    df["szbin"] = pd.cut(df["size"], [1, 3, 6, 10, 15, 20, 100],
                         labels=["2-3", "4-6", "7-10", "11-15", "16-20", "21+"])
    for b, g in df.groupby("szbin", observed=True):
        log(f"   {str(b):>6}{len(g):>7}{g.residual.mean():>10.3f}"
            f"{g.residual.std():>9.3f}{g.residual.abs().mean():>11.3f}")
    sz_rho = float(df[["size", "residual"]].corr(method="spearman").iloc[0, 1])
    sz_abs = float(df[["size", "residual"]].assign(a=df.residual.abs())[
        ["size", "a"]].corr(method="spearman").iloc[0, 1])
    log(f"   Spearman(size, residual)     {sz_rho:+.4f}")
    log(f"   Spearman(size, |residual|)   {sz_abs:+.4f}")

    # ---- CHECK 4: per era ----
    log("\n4. BY YEAR  (sampling density confounds everything else in this project)")
    log(f"   {'year':>6}{'n':>7}{'med_seqs':>10}{'R2_add':>9}{'sd_res':>9}")
    for y, g in df.groupby("year"):
        sr = float(((g.g_obs - g.g_add) ** 2).sum())
        st = float(((g.g_obs - g.g_obs.mean()) ** 2).sum())
        log(f"   {y:>6}{len(g):>7}{g.n_seqs_t.median():>10.0f}"
            f"{1 - sr/st if st > 0 else np.nan:>9.3f}{g.residual.std():>9.3f}")

    # ---- READ ----
    log("\n" + "-" * 76)
    log("READ")
    log("-" * 76)
    if r2_add < 0:
        log(f"   R^2 of {r2_add:.3f} is NEGATIVE -- additive fitness predicts growth")
        log("   WORSE than predicting the population mean. The additive assumption")
        log("   is not merely incomplete here, it is actively misleading.")
    elif r2_add > r2_naive + 0.05:
        log(f"   Additive fitness explains real variance ({r2_add:.3f} vs naive")
        log(f"   {r2_naive:.3f}). It is a working model of growth on this data.")
    else:
        log(f"   Additive fitness ({r2_add:.3f}) barely beats predicting the mean")
        log(f"   ({r2_naive:.3f}). Per-mutation growth rates carry little")
        log("   information about constellation growth either way.")
    log("")
    if not np.isnan(ac_s):
        if ac_s > 0.2:
            log(f"   Residual autocorrelation {ac_s:+.3f} -- deviations from additivity")
            log("   PERSIST for the same constellation across months. That is not")
            log("   sampling noise: the combination is consistently better or worse")
            log("   than the sum of its parts. This is the epistasis signal at the")
            log("   level that matters for forecasting, and it is what script 24's")
            log("   attachment measurement could not see.")
        elif ac_s < 0.05:
            log(f"   Residual autocorrelation {ac_s:+.3f} -- deviations do NOT persist.")
            log("   They are noise. Constellation growth is additive over member")
            log("   mutations to within measurement error, and there is no")
            log("   combination-level fitness effect to model. Combined with script")
            log("   24, that is a clean negative on the central hypothesis.")
        else:
            log(f"   Residual autocorrelation {ac_s:+.3f} -- weak. Some persistence,")
            log("   not much. Check whether it concentrates in particular size bins")
            log("   or eras before building on it.")
    log(f"\n   wrote {args.out}")


if __name__ == "__main__":
    main()
