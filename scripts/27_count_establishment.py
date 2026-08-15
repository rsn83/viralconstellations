#!/usr/bin/env python
"""
27_count_establishment.py

The fork. One number, no model, no GPU.

QUESTION
--------
How many ESTABLISHMENT EVENTS are in the data?

An establishment event is a constellation that at month t is either absent or
below a low-frequency threshold, and by month t+k has risen above a meaningful
frequency threshold. That is "a variant emerged", stated so it can be counted.

WHY THIS DECIDES EVERYTHING
---------------------------
Three targets, increasingly hard:
  appearance     the set shows up at all           -- mostly sequencing noise
  establishment  it shows up AND grows             -- this script
  dominance      it outcompetes everything else    -- a handful of events ever

Appearance has been measured: source composition adds only ~0.02 AUC over
marginal frequency (37/38 months, p=1.4e-10 -- real but small). Dominance has
too few events to learn from. Establishment sits between, and nobody has
counted it.

  200+ events  -> a learning problem. Build the model.
  ~15 events   -> a case-study paper. No architecture fixes that.

FREQUENCY, NOT COUNT
--------------------
Thresholds are on FREQUENCY (share of that month's sequences), not raw count.
Sequencing volume in this dataset swings ~20x, so a fixed count threshold
would make "establishment" mean something different in 2021 than in 2024.

LEAD TIME
---------
For each event, how many months before it crossed the high threshold was it
already visible above the low one. That is the forecasting horizon actually
available -- if it is 0, the event was invisible until it had already
established and no model could have called it early.

Usage
-----
  python scripts/27_count_establishment.py
  python scripts/27_count_establishment.py --horizons 1 3 6
"""

import argparse
import pickle
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
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", type=str, default=None)
    ap.add_argument("--end_month", type=str, default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=30)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--low", type=float, nargs="+",
                    default=[0.0, 0.0001, 0.001],
                    help="frequency at t must be AT OR BELOW this (0.0 = absent)")
    ap.add_argument("--high", type=float, nargs="+",
                    default=[0.01, 0.05, 0.10],
                    help="frequency at t+k must EXCEED this")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "27_establishment.csv"))
    args = ap.parse_args()

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    if args.start_month:
        months = [m for m in months if m >= args.start_month]
    if args.end_month:
        months = [m for m in months if m <= args.end_month]

    cache = {}

    def H(mo):
        if mo not in cache:
            with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
                raw = constellations_of(pickle.load(fh))
            filt = {c: v for c, v in raw.items()
                    if v >= args.min_count and 1 <= len(c) <= args.max_set_size}
            tot = sum(filt.values())
            cache[mo] = (filt, {c: v / max(tot, 1) for c, v in filt.items()}, tot)
        return cache[mo]

    log(f"{len(months)} months: {months[0]} .. {months[-1]}  "
        f"min_count={args.min_count}\n")
    log(f"  {'month':<10}{'n_sets':>8}{'n_seqs':>10}")
    for mo in months[::6]:
        f, _, tot = H(mo)
        log(f"  {mo:<10}{len(f):>8}{tot:>10}")
    log("")

    rows = []
    for k in args.horizons:
        for lo in args.low:
            for hi in args.high:
                if hi <= lo:
                    continue
                events = []
                for i in range(len(months) - k):
                    mt, mtk = months[i], months[i + k]
                    _, ft, _ = H(mt)
                    _, ftk, _ = H(mtk)
                    for c, f_after in ftk.items():
                        if f_after <= hi:
                            continue
                        f_before = ft.get(c, 0.0)
                        if f_before > lo:
                            continue
                        # lead time: months before mtk that c was already
                        # visible at all (above the low bar) but not yet above hi
                        lead = 0
                        for j in range(i + k - 1, -1, -1):
                            fj = H(months[j])[1].get(c, 0.0)
                            if fj > hi:
                                break
                            if fj > 0:
                                lead += 1
                            else:
                                break
                        events.append(dict(
                            horizon=k, low=lo, high=hi,
                            month_t=mt, month_tk=mtk, set_size=len(c),
                            freq_before=f_before, freq_after=f_after,
                            lead_months=lead,
                            key="|".join(str(x) for x in sorted(c))))
                if not events:
                    rows.append(dict(horizon=k, low=lo, high=hi, n_events=0,
                                     n_unique=0, median_lead=np.nan,
                                     median_size=np.nan, from_absent=np.nan))
                    continue
                ev = pd.DataFrame(events)
                rows.append(dict(
                    horizon=k, low=lo, high=hi,
                    n_events=len(ev),
                    n_unique=ev["key"].nunique(),
                    median_lead=float(ev["lead_months"].median()),
                    frac_lead0=float((ev["lead_months"] == 0).mean()),
                    median_size=float(ev["set_size"].median()),
                    from_absent=float((ev["freq_before"] == 0).mean()),
                ))

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log("=" * 86)
    log("ESTABLISHMENT EVENT COUNTS")
    log("=" * 86)
    log(f"  {'h':>2}{'low':>9}{'high':>8}{'events':>9}{'unique':>8}"
        f"{'med_lead':>10}{'lead=0':>9}{'med_size':>10}{'from_0':>8}")
    for _, r in df.iterrows():
        log(f"  {int(r.horizon):>2}{r.low:>9.5f}{r.high:>8.3f}"
            f"{int(r.n_events):>9}{int(r.n_unique):>8}"
            f"{r.get('median_lead', np.nan):>10.1f}"
            f"{r.get('frac_lead0', np.nan):>9.2f}"
            f"{r.get('median_size', np.nan):>10.1f}"
            f"{r.get('from_absent', np.nan):>8.2f}")

    log("\n" + "-" * 86)
    log("READ")
    log("-" * 86)
    log("  events    total establishment events across all month pairs")
    log("  unique    distinct constellations (the same set establishing in")
    log("            consecutive windows is counted once here -- this is the")
    log("            number that bounds what a model can learn)")
    log("  med_lead  months the set was visible above 0 before crossing 'high'")
    log("  lead=0    fraction invisible until the month they established. These")
    log("            are UNFORECASTABLE by construction -- no model sees them")
    log("            coming. A high value means the task is mostly impossible.")
    log("  from_0    fraction that were completely absent at t")
    log("")
    mid = df[(df.horizon == 3)] if (df.horizon == 3).any() else df
    if len(mid):
        best = mid.loc[mid["n_unique"].idxmax()]
        log(f"  Most permissive setting at h=3: low={best.low}, high={best.high}")
        log(f"    -> {int(best.n_unique)} unique establishment events")
        n = int(best.n_unique)
        if n >= 200:
            log("    A LEARNING PROBLEM. Enough positives to train and evaluate.")
            log("    Build the naming/establishment head and compare against HELEN.")
        elif n >= 50:
            log("    MARGINAL. Enough to measure, not enough to train a large model.")
            log("    A simple model with few parameters, or a characterisation paper")
            log("    with these events as case studies.")
        else:
            log("    TOO FEW. This is a case-study paper, not a learning problem.")
            log("    No architecture fixes a sample size of %d." % n)
    log("")
    log("  Also check lead=0: if most events were invisible until they")
    log("  established, the forecastable subset is smaller than the count")
    log("  suggests, whatever the total.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
