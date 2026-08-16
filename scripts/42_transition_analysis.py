#!/usr/bin/env python
"""
42_transition_analysis.py

Does the new_set_share advantage concentrate at variant transitions?
Reads the CSVs already written. CPU, seconds.

WHAT PROMPTED THIS
------------------
Script 41's walk-forward test gave, out of sample over 21 months:

  g_prev alone      +0.4333
  new_set_share     +0.5782
  both              +0.6054     gain +0.1721, better in 19/21, p=0.0001

One row stood out. At 2022-01 -- the month Omicron BA.1 took over -- g_prev
scored -0.373, i.e. recent growth was ACTIVELY WRONG about which clusters
would grow next, while new_set_share scored +0.545.

That suggests a mechanism rather than a general edge: momentum predicts growth
while the population is stable, and fails when it is being replaced, whereas
diversification keeps working. If that holds across all six transitions the
claim sharpens from "diversification helps" to "diversification predicts growth
precisely when momentum misleads" -- which is the regime that matters for
early warning, and where tfpscanner-style growth-rate scanning is weakest.

TRANSITION MONTHS
-----------------
From script 38, unsupervised clustering (edit distance, average linkage,
threshold 4-7, order-independent) put dominance switches at:

  2021-01  Alpha        2022-03  BA.2
  2021-06  Delta        2022-06  BA.5
  2022-01  Omicron BA.1 2023-02  XBB

These were recovered from mutation sets alone -- no tree, no Pango labels, no
designation dates -- and are stable across thresholds and input orderings.

WHAT IS COMPARED
----------------
Each test month is labelled TRANSITION if it is within --window months of a
switch, else STABLE. The two groups are then compared on the same walk-forward
scores script 41 produced. The specific quantities:

  does g_prev degrade at transitions?
  does new_set_share hold up?
  is the gain larger there?

A permutation test guards the obvious trap: with only 6 transitions among ~21
test months, any random labelling of 6 months will show some difference. The
test asks how often a random labelling produces a gain difference at least as
large.

Usage
-----
  python scripts/42_transition_analysis.py
  python scripts/42_transition_analysis.py --window 1 --n_perm 5000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SWITCHES = ["2021-01", "2021-06", "2022-01", "2022-03", "2022-06", "2023-02"]
LABEL = {"2021-01": "Alpha", "2021-06": "Delta", "2022-01": "Omicron BA.1",
         "2022-03": "BA.2", "2022-06": "BA.5", "2023-02": "XBB"}
CTRL = ["g_prev", "freq", "depth", "n_seq"]


def log(m):
    print(m, flush=True)


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
    ap.add_argument("--window", type=int, default=1,
                    help="a month counts as TRANSITION if within this many "
                         "months of a dominance switch")
    ap.add_argument("--min_train_months", type=int, default=12)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "42_transition.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(
        subset=CTRL + [args.feature, "g_next", "month"]).copy()
    f = args.feature
    months = sorted(df.month.unique())
    mi = {m: i for i, m in enumerate(months)}
    sw_i = [mi[s] for s in SWITCHES if s in mi]
    log(f"{len(df)} cluster-months, {df.month.nunique()} months")
    log(f"switches present in data: "
        f"{[s for s in SWITCHES if s in mi]}  (window +/-{args.window})\n")

    from scipy.stats import rankdata

    rows = []
    for pos, mo in enumerate(months):
        if pos < args.min_train_months:
            continue
        tr, te = df[df.month < mo], df[df.month == mo]
        if len(te) < 8 or len(tr) < 60:
            continue

        def fit_predict(cols):
            Xtr = np.column_stack([rankdata(tr[c]) for c in cols] + [np.ones(len(tr))])
            b, *_ = np.linalg.lstsq(Xtr, rankdata(tr.g_next), rcond=None)
            Xte = np.column_stack([rankdata(te[c]) for c in cols] + [np.ones(len(te))])
            return Xte @ b

        yt = te.g_next.to_numpy(float)
        base = spearman(fit_predict(["g_prev"]), yt)
        feat = spearman(fit_predict([f]), yt)
        full = spearman(fit_predict(["g_prev", f]), yt)
        near = min((abs(pos - s) for s in sw_i), default=99)
        rows.append(dict(month=mo, n=len(te), base=base, feat=feat, full=full,
                         gain=full - base, dist_to_switch=near,
                         regime="TRANSITION" if near <= args.window else "STABLE"))

    if not rows:
        raise SystemExit("no usable test months")
    r = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(args.out, index=False)

    log("=" * 74)
    log("PER MONTH")
    log("=" * 74)
    log(f"  {'month':<10}{'n':>4}{'regime':>12}{'g_prev':>9}{f[:9]:>10}"
        f"{'both':>8}{'gain':>8}   event")
    for _, x in r.iterrows():
        ev = LABEL.get(x.month, "")
        log(f"  {x.month:<10}{int(x.n):>4}{x.regime:>12}{x.base:>9.3f}"
            f"{x.feat:>10.3f}{x.full:>8.3f}{x.gain:>8.3f}   {ev}")

    t = r[r.regime == "TRANSITION"]
    s = r[r.regime == "STABLE"]
    log("\n" + "=" * 74)
    log("BY REGIME")
    log("=" * 74)
    log(f"  {'regime':<12}{'months':>8}{'g_prev':>9}{f[:9]:>10}{'both':>8}{'gain':>8}")
    for nm, g in [("TRANSITION", t), ("STABLE", s)]:
        if len(g):
            log(f"  {nm:<12}{len(g):>8}{g.base.mean():>9.3f}{g.feat.mean():>10.3f}"
                f"{g.full.mean():>8.3f}{g.gain.mean():>8.3f}")

    if len(t) and len(s):
        d_base = t.base.mean() - s.base.mean()
        d_feat = t.feat.mean() - s.feat.mean()
        d_gain = t.gain.mean() - s.gain.mean()
        log(f"\n  transition minus stable:")
        log(f"    g_prev  {d_base:+.3f}   <- negative means momentum degrades")
        log(f"    {f:<8}{d_feat:+.3f}   <- near zero means it holds up")
        log(f"    gain    {d_gain:+.3f}")

        # permutation test: with 6 transitions among ~21 months, a random
        # labelling will show SOME difference. How often at least this large?
        rng = np.random.default_rng(0)
        k = len(t)
        vals = r.gain.to_numpy(float)
        null = []
        for _ in range(args.n_perm):
            idx = rng.permutation(len(vals))
            null.append(vals[idx[:k]].mean() - vals[idx[k:]].mean())
        null = np.array(null)
        p = float((null >= d_gain).mean())
        log(f"\n  permutation test ({args.n_perm} random labellings of {k} months):")
        log(f"    null mean {null.mean():+.4f}  sd {null.std():.4f}  "
            f"p = {p:.4f}")

    log("\n" + "-" * 74)
    log("READ")
    log("-" * 74)
    if len(t) and len(s):
        if d_base < -0.10 and abs(d_feat) < 0.10:
            log("  Momentum degrades at transitions while diversification holds.")
            log("  That is the mechanism: g_prev extrapolates the incumbent, which")
            log("  is exactly wrong when the incumbent is being replaced, whereas")
            log("  new_set_share tracks the challenger's expansion.")
        elif d_gain > 0.05 and p < 0.10:
            log("  The advantage is larger at transitions, though the mechanism is")
            log("  less clean than momentum-failure alone.")
        else:
            log("  No clear concentration at transitions. The advantage found in")
            log("  script 41 is a general one, not specific to replacement events.")
            log("  That is a weaker but still valid claim -- report it as such")
            log("  rather than as early warning.")
        log("")
        log(f"  With only {len(t)} transition months the permutation p-value is the")
        log("  figure to quote, not the difference in means.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
