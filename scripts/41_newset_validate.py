#!/usr/bin/env python
"""
41_newset_validate.py

Two checks on the new_set_share result, on the CSV script 40 already wrote.
No reclustering. CPU, under a minute.

WHAT SCRIPT 40 FOUND
--------------------
  new_set_share      raw 0.611   partial 0.541   within-cluster 0.509
  by year            0.593 / 0.524 / 0.429
  lead margin        +0.173  (0.541 -> next month vs 0.367 -> this month)
  g_prev baseline    0.451

new_set_share is the share of a cluster's sequences sitting in constellations
first seen that month. It survived controls for current growth, current
frequency, sequencing depth AND the cluster's own sequence count -- so it is
not a detection artefact of larger clusters sampling more.

WHAT WAS NOT SETTLED
--------------------
1. The shuffle control returned 0.125, not ~0. One draw is not a null
   distribution, and 0.125 suggests the partial-correlation estimator carries
   some bias when features and controls are collinear. CHECK 1 repeats the
   shuffle many times and reports the distribution, so the effect can be stated
   against a quantified null rather than a single number.

2. Correlation is not prediction. CHECK 2 is a walk-forward test: fit on months
   strictly before t, predict next-month growth at t, and compare against
   predicting from g_prev alone. Script 39 found recent growth at AUC 0.839
   with eleven extra features adding 0.0009 -- but that was 20 rise events.
   This is 575 cluster-months predicting continuous growth, which is where the
   signal appeared.

CHECK 2 IS THE GATE. If new_set_share does not improve out-of-sample prediction
over the growth baseline, it is an association and not a forecasting signal, and
extending it with ESM or structural features is not worth building.

Usage
-----
  python scripts/41_newset_validate.py
  python scripts/41_newset_validate.py --n_shuffle 500 --min_train_months 12
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CTRL = ["g_prev", "freq", "depth", "n_seq"]


def log(m):
    print(m, flush=True)


def partial_corr(x, y, Z, rng=None):
    from scipy.stats import rankdata
    xr, yr = rankdata(x), rankdata(y)
    Zr = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])]
                         + [np.ones(len(x))])
    bx, *_ = np.linalg.lstsq(Zr, xr, rcond=None)
    by, *_ = np.linalg.lstsq(Zr, yr, rcond=None)
    rx, ry = xr - Zr @ bx, yr - Zr @ by
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float((rx * ry).mean() / (sx * sy))


def spearman(a, b):
    from scipy.stats import rankdata
    ar, br = rankdata(a), rankdata(b)
    if ar.std() < 1e-12 or br.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ar, br)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "40_internal.csv"))
    ap.add_argument("--feature", default="new_set_share")
    ap.add_argument("--n_shuffle", type=int, default=200)
    ap.add_argument("--min_train_months", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "41_validate.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = set(CTRL + [args.feature, "g_next", "month", "cluster"])
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns in {args.csv}: {sorted(missing)}")
    df = df.dropna(subset=list(need)).copy()
    df["month_i"] = df.month.rank(method="dense").astype(int) - 1
    log(f"{len(df)} cluster-months, {df.cluster.nunique()} clusters, "
        f"{df.month.nunique()} months\n")

    f = args.feature
    x = df[f].to_numpy(float)
    y = df.g_next.to_numpy(float)
    Z = df[CTRL].to_numpy(float)
    obs = partial_corr(x, y, Z)

    # ---------------- CHECK 1: null distribution ----------------
    log("=" * 76)
    log(f"CHECK 1  NULL DISTRIBUTION  ({args.n_shuffle} shuffles)")
    log("=" * 76)
    log("  Two nulls. Within-month permutation preserves each month's marginal")
    log("  distribution of growth but destroys the cluster pairing. Global")
    log("  permutation destroys the month structure too, and is the looser test.")

    rng = np.random.default_rng(0)
    nulls = {"within_month": [], "global": []}
    gi = df.groupby("month").indices
    for _ in range(args.n_shuffle):
        yp = y.copy()
        for _, ix in gi.items():
            yp[ix] = rng.permutation(y[ix])
        nulls["within_month"].append(partial_corr(x, yp, Z))
        nulls["global"].append(partial_corr(x, rng.permutation(y), Z))

    log(f"\n  observed partial correlation: {obs:+.4f}\n")
    log(f"  {'null':<16}{'mean':>9}{'sd':>8}{'p95':>9}{'p99':>9}{'max':>9}{'p-value':>10}")
    rows = []
    for nm, v in nulls.items():
        v = np.array([q for q in v if not np.isnan(q)])
        p = float((np.abs(v) >= abs(obs)).mean())
        log(f"  {nm:<16}{v.mean():>9.4f}{v.std():>8.4f}{np.percentile(v,95):>9.4f}"
            f"{np.percentile(v,99):>9.4f}{v.max():>9.4f}"
            f"{('<%.3f' % (1/len(v))) if p == 0 else '%.4f' % p:>10}")
        rows.append(dict(check="null", null=nm, observed=obs, null_mean=v.mean(),
                         null_sd=v.std(), null_p95=np.percentile(v, 95), pval=p))
    wm = np.array([q for q in nulls["within_month"] if not np.isnan(q)])
    log(f"\n  The single shuffle in script 40 gave 0.125. Against this")
    log(f"  distribution (mean {wm.mean():+.4f}, sd {wm.std():.4f}) that draw was")
    log(f"  {'unusual' if abs(0.125 - wm.mean()) > 2 * wm.std() else 'ordinary'};")
    log(f"  the honest null centre is {wm.mean():+.4f}, not 0.125.")

    # ---------------- CHECK 2: walk-forward forecasting ----------------
    log("\n" + "=" * 76)
    log("CHECK 2  WALK-FORWARD FORECASTING  (the gate)")
    log("=" * 76)
    log("  Fit on months strictly before t, predict next-month growth at t.")
    log("  Baseline is g_prev alone. Correlation is not prediction: the question")
    log("  is whether the feature improves OUT-OF-SAMPLE ranking of which")
    log("  clusters grow.")

    months = sorted(df.month.unique())
    res = []
    for pos, mo in enumerate(months):
        if pos < args.min_train_months:
            continue
        tr = df[df.month < mo]
        te = df[df.month == mo]
        if len(te) < 8 or len(tr) < 60:
            continue
        # rank-space linear models, fitted on training months only
        from scipy.stats import rankdata

        def fit_predict(cols):
            Xtr = np.column_stack([rankdata(tr[c]) for c in cols]
                                  + [np.ones(len(tr))])
            ytr = rankdata(tr.g_next)
            b, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            Xte = np.column_stack([rankdata(te[c]) for c in cols]
                                  + [np.ones(len(te))])
            return Xte @ b

        p_base = fit_predict(["g_prev"])
        p_full = fit_predict(["g_prev", f])
        p_feat = fit_predict([f])
        yt = te.g_next.to_numpy(float)
        res.append(dict(month=mo, n=len(te),
                        base=spearman(p_base, yt),
                        feat=spearman(p_feat, yt),
                        full=spearman(p_full, yt)))
        log(f"  {mo}  n={len(te):3d} | g_prev {res[-1]['base']:+.3f} | "
            f"{f} {res[-1]['feat']:+.3f} | both {res[-1]['full']:+.3f}")

    if not res:
        log("  no usable test months")
        return
    r = pd.DataFrame(res)
    log("\n  " + "-" * 62)
    log(f"  over {len(r)} test months, {int(r.n.sum())} cluster-months")
    log(f"    g_prev alone      {r.base.mean():+.4f}")
    log(f"    {f:<18}{r.feat.mean():+.4f}")
    log(f"    both              {r.full.mean():+.4f}")
    gain = r.full.mean() - r.base.mean()
    wins = int((r.full > r.base).sum())
    log(f"\n  gain over baseline: {gain:+.4f}   beats it in {wins}/{len(r)} months")
    try:
        from scipy.stats import binomtest
        log(f"  sign test p = "
            f"{binomtest(wins, len(r), 0.5, alternative='greater').pvalue:.4f}")
    except Exception:
        pass

    pd.concat([pd.DataFrame(rows), r.assign(check="forecast")],
              ignore_index=True).to_csv(args.out, index=False)

    log("\n" + "-" * 76)
    log("READ")
    log("-" * 76)
    p_wm = float((np.abs(wm) >= abs(obs)).mean())
    if p_wm < 0.01 and gain > 0.05 and wins > 0.65 * len(r):
        log("  Both checks pass. new_set_share is significant against a proper")
        log("  null AND improves out-of-sample forecasting over recent growth.")
        log("  That makes it a forecasting signal, not an association, and the")
        log("  ESM / structural extension is worth building: the next question")
        log("  is whether WHAT a cluster diversifies into matters, not just that")
        log("  it diversifies.")
    elif gain < 0.02:
        log("  The null is passed but forecasting is not improved. The feature")
        log("  correlates with growth in-sample and does not add out-of-sample --")
        log("  consistent with script 39, where growth alone scored 0.839 and")
        log("  eleven extra features added 0.0009. Report it as an association,")
        log("  and do not build on it.")
    else:
        log("  Mixed. Weigh the per-month win count above the mean gain -- with")
        log("  this many months a small consistent improvement is more credible")
        log("  than a large erratic one.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
