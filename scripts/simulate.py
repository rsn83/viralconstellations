#!/usr/bin/env python3
"""
Simulate a constellation-evolution dataset with the SAME schema as
extract_spike.py output, so the forecasting harness can be validated
against known ground truth before touching real data.

Ground truth built in (so we know what a working model should find):
  * mutations have heterogeneous base propensities  -> an unconditional
    substitution table q(z) should beat uniform dispersion
  * some mutations are epistatically enabled by a background mutation
    -> a background-CONDITIONED model should beat q(z); the baselines
       here deliberately do NOT capture this, so it stays as headroom
  * constellations have fitness -> selection changes frequencies, so a
    growth model should beat persistence on incumbents
"""

import argparse
import csv
import datetime as dt
import numpy as np


def simulate(n_windows=40, n_mut=150, pop=4000, seed=0,
             lam=0.08, start="2021-01-01"):
    rng = np.random.default_rng(seed)

    # --- ground-truth mutation process -------------------------------
    # heterogeneous propensities (log-normal: a few common, many rare)
    prop = rng.lognormal(mean=0.0, sigma=1.4, size=n_mut)
    prop /= prop.sum()

    # epistasis: mutations 0..19 are "enablers"; mutations 100..149 get a
    # large propensity boost only when an enabler is already present
    enablers = set(range(20))
    epistatic = set(range(100, n_mut))
    boost = 25.0

    # fitness effects; a handful are strongly beneficial
    fit = rng.normal(0.0, 0.02, size=n_mut)
    fit[rng.choice(n_mut, 12, replace=False)] += rng.uniform(0.15, 0.45, 12)

    # --- founding population -----------------------------------------
    founder = frozenset(rng.choice(n_mut, 3, replace=False).tolist())
    counts = {founder: pop}

    d0 = dt.date.fromisoformat(start)
    rows = []

    for w in range(n_windows):
        cons = list(counts.keys())
        cnt = np.array([counts[c] for c in cons], dtype=float)

        # ---- selection: reproduce proportional to fitness ------------
        w_fit = np.array([np.exp(sum(fit[m] for m in c)) for c in cons])
        p = cnt * w_fit
        p /= p.sum()
        draw = rng.multinomial(pop, p)

        # ---- mutation: add mutations with background-dependent rates -
        new_counts = {}
        for c, k in zip(cons, draw):
            if k == 0:
                continue
            n_mutating = rng.binomial(k, min(lam, 0.95))
            if k - n_mutating > 0:
                new_counts[c] = new_counts.get(c, 0) + (k - n_mutating)
            if n_mutating == 0:
                continue

            # background-conditioned propensity
            pr = prop.copy()
            if c & enablers:
                for z in epistatic:
                    pr[z] *= boost
            for z in c:                       # no re-adding existing
                pr[z] = 0.0
            s = pr.sum()
            if s <= 0:
                new_counts[c] = new_counts.get(c, 0) + n_mutating
                continue
            pr /= s

            picks = rng.choice(n_mut, size=n_mutating, p=pr)
            for z in picks:
                child = frozenset(c | {int(z)})
                new_counts[child] = new_counts.get(child, 0) + 1

        counts = new_counts

        # ---- emit rows for this window ------------------------------
        # uneven sampling depth, like real surveillance
        depth = int(pop * rng.uniform(0.15, 0.6))
        keys = list(counts.keys())
        kc = np.array([counts[k] for k in keys], dtype=float)
        kc /= kc.sum()
        sampled = rng.multinomial(depth, kc)

        base = d0 + dt.timedelta(days=30 * w)
        for k, n in zip(keys, sampled):
            for _ in range(int(n)):
                day = base + dt.timedelta(days=int(rng.integers(0, 30)))
                rows.append((day.isoformat(), k))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="sim_spike.tsv")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = simulate(n_windows=args.windows, seed=args.seed)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["date", "month", "pango", "divergence",
                    "n_spike", "constellation"])
        for date, cons in rows:
            muts = sorted(cons)
            w.writerow([date, date[:7], "SIM", len(muts), len(muts),
                        ",".join(f"S:M{m}" for m in muts)])

    print(f"wrote {args.out}: {len(rows):,} rows")


if __name__ == "__main__":
    main()
