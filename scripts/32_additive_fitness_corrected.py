#!/usr/bin/env python
"""
32_additive_fitness_corrected.py

Corrects two problems in script 31 and re-tests. CPU, minutes.

WHAT SCRIPT 31 FOUND
--------------------
  R^2 of additive fitness  -45.36   (worse than predicting the mean)
  residual autocorrelation +0.514   (synthetic reference: +0.002 if additive)

WHAT WAS WRONG WITH IT
----------------------
(1) The additive prediction imposed unit coefficients:
        g_add = sum_m g(m) - (|c|-1) * gbar
    That is a guess at the normalisation, not an estimate. The diagnostics show
    it undercorrects badly at large sets -- mean residual runs from -0.17 at
    size 2-3 to -3.86 at size 16-20, and residual sd (6.19) is 6.7x the sd of
    the quantity being predicted (0.93). So R^2 = -45 is partly arithmetic,
    not biology. Fixed here by FITTING the additive model:
        g_obs ~ a * sum_m g(m) + b * |c| + c0
    per month. If additive fitness works at all, the best-fit version of it is
    the fair test. This can only help the additive model, so any remaining
    failure is real.

(2) Residual autocorrelation can be inflated by ANY per-constellation constant,
    including a size-dependent bias, because a constellation keeps its size.
    Fixed here by demeaning residuals within size bins and within months before
    recomputing, and by adding two shuffle controls that destroy constellation
    identity while preserving everything else.

FOUR TESTS
----------
  A  Fitted additive model vs naive baselines. Does additive fitness, at its
     best, explain constellation growth?
  B  Autocorrelation of raw / size-demeaned / month-and-size-demeaned residuals.
     If it survives demeaning, it is a property of the specific combination.
  C  Shuffle controls. Reassign residuals to random constellations within a
     month, and permute constellation labels across months. Both should
     destroy the autocorrelation if it is real.
  D  Is the persistent deviation PREDICTIVE? Split each constellation's history
     in half, take mean residual from the first half, and see whether it
     predicts residuals in the second half. Out-of-sample, and this is what
     matters for forecasting -- persistence you can only see in hindsight is
     not useful.

Usage
-----
  python scripts/32_additive_fitness_corrected.py
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


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument("--min_seqs", type=int, default=5000)
    ap.add_argument("--end_month", type=str, default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "32_additive_corrected.csv"))
    args = ap.parse_args()

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    months = [m for m in months if m <= args.end_month]
    midx = {m: i for i, m in enumerate(months)}

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

    rows = []
    for i in range(len(months) - 1):
        mt, mt1 = months[i], months[i + 1]
        Ht, cft, mft, tot_t = H(mt)
        Ht1, cft1, mft1, tot_t1 = H(mt1)
        if tot_t < args.min_seqs or tot_t1 < args.min_seqs:
            continue
        gm = {m: np.log(mft1[m] / f0) for m, f0 in mft.items()
              if f0 > 0 and mft1.get(m, 0.0) > 0}
        if len(gm) < 20:
            continue
        for c, v0 in Ht.items():
            if v0 < args.min_count or len(c) < 2:
                continue
            v1 = Ht1.get(c, 0)
            if v1 < args.min_count or not all(m in gm for m in c):
                continue
            rows.append(dict(month_t=mt, year=mt[:4],
                             key="|".join(str(x) for x in sorted(c)),
                             size=len(c), n_seqs_t=tot_t,
                             g_obs=float(np.log(cft1[c] / cft[c])),
                             sum_gm=float(sum(gm[m] for m in c))))

    if len(rows) < 200:
        raise SystemExit(f"only {len(rows)} observations")
    df = pd.DataFrame(rows)

    # ---------- A: FITTED additive model, per month ----------
    # g_obs ~ a * sum_gm + b * size + intercept, fitted separately each month.
    # Fitting can only HELP the additive model, so any remaining failure is real.
    preds = np.full(len(df), np.nan)
    coefs = []
    for mo, g in df.groupby("month_t"):
        if len(g) < 15:
            continue
        X = np.column_stack([g.sum_gm.values, g["size"].values, np.ones(len(g))])
        beta, *_ = np.linalg.lstsq(X, g.g_obs.values, rcond=None)
        preds[g.index] = X @ beta
        coefs.append(dict(month=mo, n=len(g), a=beta[0], b=beta[1], c0=beta[2]))
    df["g_add_fit"] = preds
    df = df[df.g_add_fit.notna()].copy()
    df["residual"] = df.g_obs - df.g_add_fit
    df.to_csv(args.out, index=False)

    log("=" * 78)
    log(f"{len(df)} observations, {df['key'].nunique()} constellations, "
        f"{df['month_t'].nunique()} month pairs")
    log("=" * 78)

    log("\nA. DOES ADDITIVE FITNESS EXPLAIN GROWTH (best-fit version)?")
    log(f"   R2, fitted additive (a*sum_gm + b*size + c)   {r2(df.g_obs, df.g_add_fit):+.4f}")
    # naive: per-month mean
    mm = df.groupby("month_t")["g_obs"].transform("mean")
    log(f"   R2, per-month mean only                        {r2(df.g_obs, mm):+.4f}")
    # size-only
    so = df.groupby(["month_t", "size"])["g_obs"].transform("mean")
    log(f"   R2, per-month-and-size mean                    {r2(df.g_obs, so):+.4f}")
    log(f"   Spearman(g_obs, sum_gm)                        "
        f"{df[['g_obs','sum_gm']].corr(method='spearman').iloc[0,1]:+.4f}")
    cf = pd.DataFrame(coefs)
    log(f"   fitted coefficient on sum_gm: median {cf.a.median():+.4f}  "
        f"IQR [{cf.a.quantile(.25):+.3f}, {cf.a.quantile(.75):+.3f}]")
    log("   (unit coefficient = 1.0 is what script 31 imposed)")
    log(f"   residual mean {df.residual.mean():+.4f}  sd {df.residual.std():.4f}  "
        f"vs sd(g_obs) {df.g_obs.std():.4f}")

    # ---------- B: autocorrelation, three demeanings ----------
    log("\nB. RESIDUAL AUTOCORRELATION (consecutive months, same constellation)")
    df["r_size"] = df.residual - df.groupby("size")["residual"].transform("mean")
    df["r_ms"] = df.residual - df.groupby(["month_t", "size"])["residual"].transform("mean")

    def autocorr(frame, col):
        d = frame.sort_values(["key", "month_t"]).copy()
        d["prev"] = d.groupby("key")[col].shift(1)
        d["pm"] = d.groupby("key")["month_t"].shift(1)
        ok = [isinstance(p, str) and midx.get(m, -99) - midx.get(p, 99) == 1
              for m, p in zip(d.month_t, d.pm)]
        d = d[np.array(ok) & d.prev.notna()]
        if len(d) < 30:
            return np.nan, np.nan, len(d)
        return (float(d[[col, "prev"]].corr().iloc[0, 1]),
                float(d[[col, "prev"]].corr(method="spearman").iloc[0, 1]), len(d))

    for col, lbl in [("residual", "raw"), ("r_size", "size-demeaned"),
                     ("r_ms", "month+size demeaned")]:
        p, s, n = autocorr(df, col)
        log(f"   {lbl:<22} n={n:>6}  Pearson {p:+.4f}  Spearman {s:+.4f}")

    # ---------- C: shuffle controls ----------
    log("\nC. SHUFFLE CONTROLS  (should destroy the autocorrelation)")
    rng = np.random.default_rng(0)
    sh = df.copy()
    sh["r_ms"] = sh.groupby("month_t")["r_ms"].transform(
        lambda x: rng.permutation(x.values))
    _, s1, n1 = autocorr(sh, "r_ms")
    log(f"   residuals shuffled within month   n={n1:>6}  Spearman {s1:+.4f}")

    sh2 = df.copy()
    keys = sh2["key"].unique()
    perm = dict(zip(keys, rng.permutation(keys)))
    sh2["key"] = [perm[k] for k in sh2["key"]]
    _, s2, n2 = autocorr(sh2, "r_ms")
    log(f"   constellation labels permuted     n={n2:>6}  Spearman {s2:+.4f}")

    # ---------- D: is it predictive out of sample? ----------
    log("\nD. IS THE DEVIATION PREDICTIVE?  (first half -> second half)")
    log("   Persistence visible only in hindsight is not useful for forecasting.")
    parts = []
    for k, g in df.sort_values("month_t").groupby("key"):
        if len(g) < 4:
            continue
        h = len(g) // 2
        parts.append(dict(key=k, n=len(g),
                          first=g.r_ms.iloc[:h].mean(),
                          second=g.r_ms.iloc[h:].mean()))
    if len(parts) >= 30:
        pp = pd.DataFrame(parts)
        log(f"   n constellations with >=4 months: {len(pp)}")
        log(f"   corr(first-half mean, second-half mean)  "
            f"Pearson {pp[['first','second']].corr().iloc[0,1]:+.4f}  "
            f"Spearman {pp[['first','second']].corr(method='spearman').iloc[0,1]:+.4f}")
        oos = r2(pp.second, pp.first)
        log(f"   R2 of using first-half mean to predict second half: {oos:+.4f}")
    else:
        oos = np.nan
        log(f"   only {len(parts)} constellations with >=4 months")

    # ---------- READ ----------
    _, s_ms, _ = autocorr(df, "r_ms")
    log("\n" + "-" * 78)
    log("READ")
    log("-" * 78)
    ra = r2(df.g_obs, df.g_add_fit)
    log(f"   Fitted additive R2 = {ra:+.3f} vs per-month-and-size mean "
        f"{r2(df.g_obs, so):+.3f}")
    if ra < r2(df.g_obs, so):
        log("   Even at its best fit, additive fitness does not beat simply")
        log("   predicting the mean growth for that month and set size. That is a")
        log("   direct hit on the additive assumption, and it is not an artefact")
        log("   of normalisation this time -- the coefficients were estimated.")
    else:
        log("   Fitted additive fitness does explain real variance. Script 31's")
        log("   R2 of -45 was largely an artefact of imposing unit coefficients.")
    log("")
    log(f"   Month+size demeaned autocorrelation: {s_ms:+.4f}")
    log(f"   Shuffle controls: within-month {s1:+.4f}, label-permuted {s2:+.4f}")
    if s_ms > 0.2 and abs(s1) < 0.1 and abs(s2) < 0.1:
        log("   Survives demeaning AND both shuffles. The deviation from additivity")
        log("   is a genuine property of the specific combination.")
        if not np.isnan(oos) and oos > 0.1:
            log(f"   And it is PREDICTIVE out of sample (R2 {oos:+.3f}): a")
            log("   constellation's past deviation forecasts its future deviation.")
            log("   That is a usable forecasting signal, not just a description.")
        else:
            log("   But it does NOT transfer out of sample, so it describes the")
            log("   past without helping forecast. Weaker claim.")
    elif abs(s1) > 0.1 or abs(s2) > 0.1:
        log("   A SHUFFLE CONTROL ALSO SHOWS AUTOCORRELATION. The measurement is")
        log("   picking up structure that survives destroying constellation")
        log("   identity -- do not trust the main number until that is understood.")
    else:
        log("   Autocorrelation does not survive demeaning. Script 31's +0.51 was")
        log("   driven by per-size bias in the imposed normalisation, not by")
        log("   combination-specific effects.")
    log(f"\n   wrote {args.out}")


if __name__ == "__main__":
    main()
