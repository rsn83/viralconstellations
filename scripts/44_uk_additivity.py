#!/usr/bin/env python
"""
44_uk_additivity.py

UK-only replication of script 32's additive fitness test.

WHY UK-ONLY
-----------
Script 32 tested whether a constellation's growth rate equals the sum of its
member mutations' growth rates, on the global monthly aggregates. Result:
R² = 0.44, fitted coefficient 0.23 (not 1.0 as imposed in script 31),
residual autocorrelation +0.080 after month+size demeaning -- not significant,
not persistent out-of-sample.

Two confounds could suppress a real epistasis signal in global data:

  LINEAGE MIXING: global counts aggregate sequences from different lineages in
  different countries. A mutation pair that grows together in one lineage looks
  independent when averaged with another lineage where one is absent. UK
  sequenced one population with one epidemiological history, so lineage mixing
  is minimised.

  SAMPLING HETEROGENEITY: global depth swings ~20x, so growth rates in sparse
  months are noisy. UK is dense and consistent (15k-300k per month) throughout.

If the additivity null survives UK-only, it is real. If deviations persist
there, the global test was aggregation destroying a signal that exists.

DATA
----
Built fresh from data/raw/metadata.tsv.zst, UK sequences only, QC-passed,
2020-09 to 2022-04. aaSubstitutions parsed to frozensets, no dependence on
the global occupied.pkl files.

THE TEST
--------
Identical to script 32:
  g_obs(c) = log(freq(c, t+1) / freq(c, t))        observed growth
  g_add(c) = fitted regression on sum_m g(m) and |c|  per month
  residual  = g_obs - g_add

  A. Does additive fitness explain growth? R2 vs per-month-and-size mean.
  B. Do residuals persist? Autocorrelation, with month+size demeaning.
  C. Shuffle controls: within-month and label-permuted.
  D. Out-of-sample: first-half mean predicts second-half mean?

Usage
-----
  python scripts/44_uk_additivity.py
  python scripts/44_uk_additivity.py --min_count 5 --min_seqs 10000
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(m):
    print(m, flush=True)


def parse_aa(s):
    """Parse aaSubstitutions string to a frozenset of spike substitutions.
    GISAID format: 'S:N501Y,S:E484K,ORF1a:K856R,...'
    We keep only spike (S:) substitutions, strip the 'S:' prefix,
    and return frozenset of strings like 'N501Y'.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    muts = frozenset(x[2:] for x in s.split(',') if x.startswith('S:'))
    return muts if muts else None


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata",
                    default=str(ROOT / "data" / "raw" / "metadata.tsv.zst"))
    ap.add_argument("--country", default="United Kingdom")
    ap.add_argument("--start_month", default="2020-09")
    ap.add_argument("--end_month", default="2022-04")
    ap.add_argument("--min_count", type=int, default=3,
                    help="min sequences for a constellation to be counted")
    ap.add_argument("--min_seqs", type=int, default=5000,
                    help="skip months below this depth")
    ap.add_argument("--max_set_size", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "44_uk_additivity.csv"))
    args = ap.parse_args()

    # ---- load and parse ----
    log(f"loading {args.metadata} for {args.country} ...")
    d = pd.read_csv(args.metadata, sep="\t",
                    usecols=["date", "country", "aaSubstitutions",
                             "QC_overall_status"])
    d = d[(d.country == args.country) &
          (d.QC_overall_status == "good")].copy()
    d["month"] = d.date.astype(str).str[:7]
    d = d[(d.month >= args.start_month) & (d.month <= args.end_month)]
    d = d.dropna(subset=["aaSubstitutions"])
    d["cons"] = d.aaSubstitutions.map(parse_aa)
    d = d.dropna(subset=["cons"])
    d = d[d.cons.map(lambda c: 1 <= len(c) <= args.max_set_size)]
    log(f"{len(d):,} sequences after filtering\n")

    # ---- build monthly counts ----
    months = sorted(d.month.unique())
    H = {}
    for mo in months:
        sub = d[d.month == mo]
        cnt = Counter(sub.cons)
        H[mo] = {c: v for c, v in cnt.items() if v >= args.min_count}
    months = [mo for mo in months if sum(H[mo].values()) >= args.min_seqs]
    log(f"{len(months)} usable months: {months[0]} .. {months[-1]}\n")

    midx = {m: i for i, m in enumerate(months)}

    # ---- same additivity test as script 32 ----
    rows = []
    for i in range(len(months) - 1):
        mt, mt1 = months[i], months[i + 1]
        Ht, Ht1 = H[mt], H[mt1]
        tot_t = sum(Ht.values())
        tot_t1 = sum(Ht1.values())
        cft = {c: v / tot_t for c, v in Ht.items()}
        cft1 = {c: v / tot_t1 for c, v in Ht1.items()}

        # per-mutation log growth
        mut_t = Counter(); mut_t1 = Counter()
        for c, v in Ht.items():
            for m in c: mut_t[m] += v
        for c, v in Ht1.items():
            for m in c: mut_t1[m] += v
        gm = {}
        for m in mut_t:
            f0 = mut_t[m] / tot_t
            f1 = mut_t1.get(m, 0) / tot_t1
            if f0 > 0 and f1 > 0:
                gm[m] = np.log(f1 / f0)
        if len(gm) < 10:
            continue

        for c, v0 in Ht.items():
            if v0 < args.min_count or len(c) < 2:
                continue
            v1 = Ht1.get(c, 0)
            if v1 < args.min_count:
                continue
            if not all(m in gm for m in c):
                continue
            g_obs = np.log(cft1[c] / cft[c])
            rows.append(dict(month_t=mt, month_t1=mt1, year=mt[:4],
                             key="|".join(str(x) for x in sorted(c)),
                             size=len(c), count_t=v0, count_t1=v1,
                             n_seqs_t=tot_t,
                             g_obs=g_obs,
                             sum_gm=float(sum(gm[m] for m in c))))
        if rows and rows[-1]["month_t"] == mt:
            n = sum(1 for r in rows if r["month_t"] == mt)
            log(f"  {mt} -> {mt1}  n_seqs={tot_t:>9}  constellations={n:>5}")

    if len(rows) < 200:
        raise SystemExit(f"only {len(rows)} observations; try lower --min_count")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log(f"\n{len(df)} constellation-month observations, "
        f"{df['key'].nunique()} distinct, {df.month_t.nunique()} month pairs\n")

    # ---- A: fitted additive model ----
    preds = np.full(len(df), np.nan)
    for mo, g in df.groupby("month_t"):
        if len(g) < 15:
            continue
        X = np.column_stack([g.sum_gm.values, g["size"].values, np.ones(len(g))])
        beta, *_ = np.linalg.lstsq(X, g.g_obs.values, rcond=None)
        preds[g.index] = X @ beta
    df["g_add_fit"] = preds
    df = df[df.g_add_fit.notna()].copy()
    df["residual"] = df.g_obs - df.g_add_fit
    mm = df.groupby("month_t")["g_obs"].transform("mean")
    so = df.groupby(["month_t", "size"])["g_obs"].transform("mean")

    log("=" * 74)
    log("A. DOES ADDITIVE FITNESS EXPLAIN GROWTH?")
    log("=" * 74)
    log(f"   R2, fitted additive        {r2(df.g_obs, df.g_add_fit):+.4f}")
    log(f"   R2, per-month mean         {r2(df.g_obs, mm):+.4f}")
    log(f"   R2, per-month-and-size     {r2(df.g_obs, so):+.4f}")
    log(f"   Spearman(g_obs, sum_gm)    "
        f"{df[['g_obs','sum_gm']].corr(method='spearman').iloc[0,1]:+.4f}")
    log(f"   residual mean {df.residual.mean():+.4f}  sd {df.residual.std():.4f}")

    # ---- B: autocorrelation, demeaned ----
    log("\nB. RESIDUAL AUTOCORRELATION (the decisive check)")
    df["r_ms"] = df.residual - df.groupby(["month_t", "size"])["residual"].transform("mean")
    d2 = df.sort_values(["key", "month_t"]).copy()
    d2["prev"] = d2.groupby("key")["r_ms"].shift(1)
    d2["pm"] = d2.groupby("key")["month_t"].shift(1)
    ok = [isinstance(p, str) and midx.get(m, -99) - midx.get(p, 99) == 1
          for m, p in zip(d2.month_t, d2.pm)]
    pair = d2[np.array(ok) & d2.prev.notna()]
    if len(pair) >= 30:
        ac = float(pair[["r_ms", "prev"]].corr(method="spearman").iloc[0, 1])
        log(f"   n consecutive pairs: {len(pair)}")
        log(f"   autocorr (Spearman):  {ac:+.4f}")
    else:
        ac = np.nan
        log(f"   only {len(pair)} consecutive pairs")

    # ---- C: shuffle controls ----
    log("\nC. SHUFFLE CONTROLS")
    rng = np.random.default_rng(0)
    sh = df.copy()
    sh["r_ms"] = sh.groupby("month_t")["r_ms"].transform(
        lambda x: rng.permutation(x.values))
    d3 = sh.sort_values(["key", "month_t"]).copy()
    d3["prev"] = d3.groupby("key")["r_ms"].shift(1)
    d3["pm"] = d3.groupby("key")["month_t"].shift(1)
    ok3 = [isinstance(p, str) and midx.get(m, -99) - midx.get(p, 99) == 1
            for m, p in zip(d3.month_t, d3.pm)]
    p3 = d3[np.array(ok3) & d3.prev.notna()]
    ac_sh = float(p3[["r_ms", "prev"]].corr(method="spearman").iloc[0, 1]) if len(p3) >= 30 else np.nan
    sh2 = df.copy()
    keys = sh2["key"].unique(); perm = dict(zip(keys, rng.permutation(keys)))
    sh2["key"] = [perm[k] for k in sh2["key"]]
    d4 = sh2.sort_values(["key", "month_t"]).copy()
    d4["prev"] = d4.groupby("key")["r_ms"].shift(1)
    d4["pm"] = d4.groupby("key")["month_t"].shift(1)
    ok4 = [isinstance(p, str) and midx.get(m, -99) - midx.get(p, 99) == 1
            for m, p in zip(d4.month_t, d4.pm)]
    p4 = d4[np.array(ok4) & d4.prev.notna()]
    ac_lp = float(p4[["r_ms", "prev"]].corr(method="spearman").iloc[0, 1]) if len(p4) >= 30 else np.nan
    log(f"   within-month shuffle:    {ac_sh:+.4f}  (expect ~0)")
    log(f"   label-permuted:          {ac_lp:+.4f}  (expect ~0)")

    # ---- D: out-of-sample ----
    log("\nD. OUT-OF-SAMPLE (first half -> second half)")
    parts = []
    for k, g in df.sort_values("month_t").groupby("key"):
        if len(g) < 4:
            continue
        h = len(g) // 2
        parts.append(dict(first=g.r_ms.iloc[:h].mean(),
                          second=g.r_ms.iloc[h:].mean()))
    if len(parts) >= 30:
        pp = pd.DataFrame(parts)
        oos = r2(pp.second, pp.first)
        log(f"   n constellations >= 4 months: {len(pp)}")
        log(f"   R2 first-half -> second-half: {oos:+.4f}")
    else:
        oos = np.nan

    # ---- READ ----
    log("\n" + "-" * 74)
    log("COMPARISON WITH GLOBAL RESULT (script 32)")
    log("-" * 74)
    log(f"                        GLOBAL          UK-ONLY")
    log(f"  R2 fitted additive    +0.4412         {r2(df.g_obs, df.g_add_fit):+.4f}")
    log(f"  R2 month+size mean    +0.2644         {r2(df.g_obs, so):+.4f}")
    log(f"  autocorr (demeaned)   +0.0801         {ac:+.4f}")
    log(f"  shuffle within-month  +0.0101         {ac_sh:+.4f}")
    log(f"  shuffle label-perm    +0.0801         {ac_lp:+.4f}")
    log(f"  OOS R2                -2.0773         {oos:+.4f}")
    log("")
    if not np.isnan(ac) and ac > 0.15 and (np.isnan(ac_sh) or abs(ac_sh) < 0.08):
        log("  Autocorrelation RISES in UK-only data and survives both shuffle")
        log("  controls. The global null was suppression by aggregation. A real")
        log("  constellation-level fitness effect exists, detectable only in a")
        log("  homogeneous population. This changes the paper significantly.")
    elif not np.isnan(ac) and abs(ac) < 0.08:
        log("  Autocorrelation stays near zero UK-only. The global null is real,")
        log("  not an aggregation artefact. Fitness is additive in a single")
        log("  well-sampled population as well as globally. This confirms the")
        log("  negative and strengthens it substantially -- epistasis is absent")
        log("  at the resolution surveillance sequences provide.")
    else:
        log("  Inconclusive. Check whether the autocorrelation exceeds the")
        log("  shuffle controls and whether it is consistent across years.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
