#!/usr/bin/env python
"""
58_edit_rate.py

The model
---------
The data says the process moves by single-mutation edits from circulating
states (min-edit-1). So the object to learn is not a distribution over 2^V --
it is the RATE at which a circulating constellation acquires a mutation:

    u_t(m | c)  =  rho_t(m)  *  g(m, c)

    rho_t(m) : how available the mutation is right now = its population
               frequency at t. Carries all the time dependence. Already
               measured, not learned.
    g(m, c)  : background compatibility. Permutation-invariant in c,
               time-independent. This is the co-occurrence term.

Why factorised this way
-----------------------
It makes the null explicit and testable. g == 1 is the MARGINAL-ONLY model:
mutations join backgrounds in proportion to how common they are, with no
compatibility term. The project's central claim is that this is insufficient --
that population-level co-occurrence structure carries information marginal
frequencies do not. This script measures how much g buys over g == 1 instead of
arguing about it.

g is held time-independent on purpose. If it turns out to need time dependence
(immune pressure changing which backgrounds accept which mutations) that is a
finding, and it cannot be detected if it is built in from the start.

Target
------
For every (c, m) with c circulating at t and m not in c:

    y = 1  if  c u {m}  is observed at t+1  AND was NOT observed at t

i.e. the APPEARANCE of a new constellation, not its persistence. Persistence is
a different problem with a trivial strong baseline (copy-forward), and mixing
them hides whether anything is being predicted at all.

What is observed : constellations and their counts per month
What is hidden   : whether c u {m} actually descended from c. The edge is a
                   reachability claim, not an ancestry claim.
What is predicted: P(c u {m} appears at t+1)
What is NOT claimed: that the mutation m occurred in month t, or in that genome
Why useful       : the candidate set is enumerable and built only from month-t
                   information, so it is evaluable without leakage

Models compared
---------------
  random          uniform score                    (base rate reference)
  marginal        score = rho_t(m)                 <- g == 1, THE null
  product         score = rho_t(m) * freq_t(c)     <- independence of set and label
  logistic        score = fitted P(y=1) on co-occurrence + trajectory features

Features of g (all permutation-invariant in c, all from months <= t)
--------------------------------------------------------------------
  mean / max / min pointwise mutual information between m and the labels of c
  fraction of c's labels that have ever co-occurred with m
  whether m has ever co-occurred with anything in c
  |c|, log frequency of c, log growth of c over the last month
  log rho_t(m), and the number of months m has been in support

Evaluation
----------
Rolling-origin (expanding training window, capped). Average precision and lift
over base rate, since the positive rate is well under 1%. Reported alongside the
reachability ceiling: the share of genuinely new constellations at t+1 that are
expressible as c u {m} for c circulating at t and m in the candidate pool. No
model can exceed it, and any AP quoted without it is misleading.

Outputs
-------
outputs/58_edit_rate_by_month.csv   per-origin metrics per model
outputs/58_edit_rate_summary.csv    pooled comparison
outputs/58_coefficients.csv         logistic coefficients at the final origin

Usage
-----
python scripts/58_edit_rate.py --min_count 3 --end_month 2024-12
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

FEATURES = [
    "log_rho_m", "m_months_in_support",
    "pmi_mean", "pmi_max", "pmi_min", "cooc_frac", "cooc_any",
    "c_size", "log_freq_c", "log_growth_c",
]


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_months(data_dir, min_count, start_month=None, end_month=None):
    files = []
    for fn in os.listdir(data_dir):
        m = MONTH_RE.search(fn)
        if m:
            files.append((m.group(1), os.path.join(data_dir, fn)))
    files.sort()
    out = []
    for month, path in files:
        if start_month and month < start_month:
            continue
        if end_month and month > end_month:
            continue
        with open(path, "rb") as f:
            occ = pickle.load(f)
        occ = {k: v for k, v in occ.items() if v >= min_count}
        if occ:
            out.append((month, occ))
    return out


def label_counts(occ):
    nc = defaultdict(float)
    for cs, w in occ.items():
        for lab in cs:
            nc[lab] += w
    return nc


# ----------------------------------------------------------------------------
# logistic regression (numpy, no sklearn)
# ----------------------------------------------------------------------------

def fit_logistic(X, y, l2=1.0, n_iter=60, tol=1e-7):
    n, p = X.shape
    w = np.zeros(p)
    R = l2 * np.eye(p)
    R[0, 0] = 0.0
    for _ in range(n_iter):
        z = np.clip(X @ w, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (mu - y) + R @ w
        s = np.clip(mu * (1 - mu), 1e-6, None)
        H = X.T @ (X * s[:, None]) + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        wn = w - step
        if np.max(np.abs(wn - w)) < tol:
            return wn
        w = wn
    return w


def predict_logistic(X, w):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -np.asarray(s, dtype=float)))
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, y.size + 1)
    return float((prec * yy).sum() / y.sum())


def recall_at_k(y, s, k):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -np.asarray(s, dtype=float)))
    return float(y[order][:k].sum() / y.sum())


# ----------------------------------------------------------------------------
# per-origin candidate construction
# ----------------------------------------------------------------------------

def build_candidates(occ_t, occ_next, labels_t, lab_index, PMI, COOC,
                     months_in_support, prev_occ, max_sets):
    """
    Returns X, y, plus the score components needed for the baselines.

    Candidates: (c, m) for c in the top `max_sets` constellations at t by count,
    and m in the month-t label pool with m not already in c.
    """
    sets = sorted(occ_t.items(), key=lambda kv: -kv[1])[:max_sets]
    csets = [c for c, _ in sets]
    cw = np.array([w for _, w in sets], dtype=float)
    tot = float(sum(occ_t.values()))
    freq_c = cw / tot

    growth = np.array([
        np.log((occ_t[c] + 1.0) / (prev_occ.get(c, 0.0) + 1.0)) for c in csets
    ])

    nlab = len(labels_t)
    lc = label_counts(occ_t)
    rho = np.array([lc[l] / tot for l in labels_t])
    mis = np.array([months_in_support.get(l, 0) for l in labels_t], dtype=float)

    # indicator matrix of which labels each set contains, over the month-t pool
    B = np.zeros((len(csets), nlab), dtype=bool)
    for i, c in enumerate(csets):
        for l in c:
            j = lab_index.get(l)
            if j is not None:
                B[i, j] = True

    Bf = B.astype(float)
    csize = np.array([len(c) for c in csets], dtype=float)
    npool = np.maximum(Bf.sum(axis=1), 1.0)  # labels of c inside the pool

    # aggregate PMI of m against the labels of c
    S = Bf @ PMI                       # sum of PMI(l, m) over l in c
    pmi_mean = S / npool[:, None]
    C_any = Bf @ (COOC > 0).astype(float)
    cooc_frac = C_any / npool[:, None]

    pmi_max = np.full((len(csets), nlab), -10.0)
    pmi_min = np.full((len(csets), nlab), 10.0)
    for i in range(len(csets)):
        idx = np.flatnonzero(B[i])
        if idx.size:
            sub = PMI[idx]
            pmi_max[i] = sub.max(axis=0)
            pmi_min[i] = sub.min(axis=0)

    valid = ~B                          # m must not already be in c
    ii, jj = np.nonzero(valid)

    H_t = set(occ_t.keys())
    y = np.zeros(ii.size, dtype=int)
    for k in range(ii.size):
        c = csets[ii[k]]
        m = labels_t[jj[k]]
        x = frozenset(c | {m})
        if x in occ_next and x not in H_t:
            y[k] = 1

    X = np.column_stack([
        np.log(np.clip(rho[jj], 1e-9, None)),
        mis[jj],
        pmi_mean[ii, jj],
        pmi_max[ii, jj],
        pmi_min[ii, jj],
        cooc_frac[ii, jj],
        (C_any[ii, jj] > 0).astype(float),
        csize[ii],
        np.log(np.clip(freq_c[ii], 1e-9, None)),
        growth[ii],
    ])

    comp = {
        "rho": rho[jj],
        "freq_c": freq_c[ii],
    }
    return X, y, comp, csets, H_t


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--max_sets", type=int, default=1500,
                    help="top-N constellations by count used as sources")
    ap.add_argument("--min_train", type=int, default=12)
    ap.add_argument("--train_window", type=int, default=6,
                    help="origins pooled for training, to bound memory")
    ap.add_argument("--neg_per_pos", type=int, default=50)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ = {m: o for m, o in months}
    T = len(names)
    print(f"loaded {T} months: {names[0]} .. {names[-1]}")

    # months-in-support counter, built causally as we walk forward
    months_in_support = defaultdict(int)

    cache = {}   # origin index -> (X, y, comp)
    rows = []
    last_w, last_mu, last_sd = None, None, None

    for t in range(T - 1):
        occ_t, occ_n = occ[names[t]], occ[names[t + 1]]
        prev_occ = occ[names[t - 1]] if t > 0 else {}

        lc = label_counts(occ_t)
        labels_t = sorted(lc.keys(), key=str)
        lab_index = {l: j for j, l in enumerate(labels_t)}
        for l in labels_t:
            months_in_support[l] += 1

        # co-occurrence and PMI from months <= t only
        nlab = len(labels_t)
        CO = np.zeros((nlab, nlab))
        marg = np.zeros(nlab)
        wtot = 0.0
        for j in range(t + 1):
            o = occ[names[j]]
            for cs, w in o.items():
                idx = [lab_index[l] for l in cs if l in lab_index]
                if not idx:
                    continue
                wtot += w
                a = np.array(idx)
                marg[a] += w
                CO[np.ix_(a, a)] += w
        if wtot <= 0:
            continue
        p = np.clip(marg / wtot, 1e-9, None)
        Pjoint = np.clip(CO / wtot, 1e-12, None)
        PMI = np.log(Pjoint / np.outer(p, p))
        np.fill_diagonal(PMI, 0.0)

        X, y, comp, csets, H_t = build_candidates(
            occ_t, occ_n, labels_t, lab_index, PMI, CO,
            months_in_support, prev_occ, args.max_sets)
        cache[t] = (X, y, comp)

        # reachability ceiling: new constellations at t+1 expressible as c u {m}
        new_sets = [c for c in occ_n if c not in H_t]
        pool = set(labels_t)
        src = set(csets)
        reach = 0
        for x in new_sets:
            for l in x:
                if l in pool and frozenset(x - {l}) in src:
                    reach += 1
                    break
        ceiling = reach / len(new_sets) if new_sets else np.nan

        if t < args.min_train:
            continue

        # ---- train on a pooled window of earlier origins, negatives subsampled
        Xs, ys = [], []
        for j in range(max(args.min_train - args.train_window, t - args.train_window), t):
            if j not in cache:
                continue
            Xj, yj, _ = cache[j]
            pos = np.flatnonzero(yj)
            neg = np.flatnonzero(yj == 0)
            if pos.size == 0:
                continue
            take = min(neg.size, pos.size * args.neg_per_pos)
            neg = rng.choice(neg, size=take, replace=False)
            sel = np.concatenate([pos, neg])
            Xs.append(Xj[sel])
            ys.append(yj[sel])
        if not Xs:
            continue
        Xtr = np.vstack(Xs)
        ytr = np.concatenate(ys).astype(float)

        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Xtr_s = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
        w = fit_logistic(Xtr_s, ytr, l2=args.l2)
        last_w, last_mu, last_sd = w, mu, sd

        Xte_s = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        scores = {
            "random": rng.random(len(y)),
            "marginal": comp["rho"],
            "product": comp["rho"] * comp["freq_c"],
            "logistic": predict_logistic(Xte_s, w),
        }

        base = float(y.mean())
        npos = int(y.sum())
        if npos == 0:
            continue
        for name, s in scores.items():
            apv = average_precision(y, s)
            rows.append({
                "origin": names[t], "target": names[t + 1], "model": name,
                "ap": apv, "base_rate": base,
                "lift": apv / base if base > 0 else np.nan,
                "recall_at_100": recall_at_k(y, s, 100),
                "recall_at_1000": recall_at_k(y, s, 1000),
                "n_candidates": int(y.size), "n_pos": npos,
                "n_new_sets": len(new_sets), "reach_ceiling": ceiling,
                "n_train": int(len(ytr)),
            })
        print(f"  {names[t]} -> {names[t+1]}: {y.size} candidates, "
              f"{npos} positives, base {base:.5f}, ceiling {ceiling:.3f}")

        # keep memory bounded
        for j in list(cache):
            if j < t - args.train_window:
                del cache[j]

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/58_edit_rate_by_month.csv", index=False)

    print("\n" + "=" * 74)
    print("CONSTELLATION APPEARANCE  (rolling-origin, one step ahead)")
    print("=" * 74)
    summ = df.groupby("model").agg(
        ap=("ap", "mean"), lift=("lift", "mean"),
        r_at_100=("recall_at_100", "mean"),
        r_at_1000=("recall_at_1000", "mean"),
        base_rate=("base_rate", "mean"),
        n_candidates=("n_candidates", "mean"),
        n_pos=("n_pos", "mean"),
        origins=("ap", "count"),
    ).reset_index().sort_values("ap", ascending=False)
    summ.to_csv(f"{args.out_dir}/58_edit_rate_summary.csv", index=False)
    print(summ.round(5).to_string(index=False))

    a = df[df["model"] == "logistic"].set_index("origin")["ap"]
    b = df[df["model"] == "marginal"].set_index("origin")["ap"]
    common = a.index.intersection(b.index)
    if len(common):
        print(f"\nlogistic (g learned) beats marginal (g == 1) on "
              f"{(a[common] > b[common]).sum()} / {len(common)} origins")
        print(f"mean AP ratio: {(a[common] / b[common]).mean():.2f}x")

    print(f"\nreachability ceiling: {df['reach_ceiling'].mean():.3f}")
    print("  share of genuinely new constellations expressible as c u {m} with c")
    print("  circulating at t and m in the month-t label pool. No model can")
    print("  exceed this; quote it beside any AP.")

    if last_w is not None:
        co = pd.DataFrame({
            "feature": ["intercept"] + FEATURES,
            "coef_standardised": last_w,
        }).sort_values("coef_standardised", key=np.abs, ascending=False)
        co.to_csv(f"{args.out_dir}/58_coefficients.csv", index=False)
        print("\n--- logistic coefficients at the final origin (standardised) ---")
        print(co.round(4).to_string(index=False))

    print("\nread:")
    print("  'marginal' is g == 1: availability only, no compatibility term.")
    print("  If logistic does not beat it, co-occurrence structure is not adding")
    print("  information at this resolution and the central claim fails as stated.")
    print("  If it does, the margin is the size of the co-occurrence effect, and")
    print("  the PMI coefficients say whether it comes from the background.")
    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
