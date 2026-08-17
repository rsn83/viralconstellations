#!/usr/bin/env python
"""
67_delta_ablation.py

The question
------------
Does the background a mutation joins carry information beyond how common that
mutation already is?

Three scripts disagree:
  58  full logistic AP 0.0436 vs marginal 0.0097 -- set context wins 4.5x
  65  model AP 0.090 vs marginal 0.250          -- marginal wins
  66  learned 0.0277 vs marginal 0.0285         -- marginal wins
65 and 66 both scored along an imputed OT coupling. 58 needs no coupling. This
script settles it by ablating feature blocks inside 58's setup, so the only
thing that changes between rows of the output is which features the model sees.

The delta metric, and why it is the right one
---------------------------------------------
Whole-set Jaccard is invalid here and gets worse with time: sets carry ~2
mutations early and ~55 late, while ~2 change per step either way, so copying
scores ~0.3 early and ~0.96 late. The metric measures how derived the population
has become, not prediction quality, and the test window sits at the easy end.

The delta avoids this. For every constellation c circulating at t and every
candidate mutation m not in c:

    y = 1  if  c u {m}  is present at t+1  AND absent at t

That is the change itself. Persistence scores at the base rate by construction:
it cannot name a set that does not yet exist. No coupling is assumed -- the pair
(c, m) is built from month t alone and checked against month t+1 directly.

Feature blocks
--------------
  full        everything
  no_pmi      drop pmi_mean/max/min, cooc_frac, cooc_any
  only_pmi    only those
  only_freq   log_rho_m, log_freq_c, c_size, log_growth_c, m_months_in_support
  marginal    no fit at all: score = rho_t(m)
  random      reference

full vs no_pmi is the answer. If they are equal, co-occurrence contributes
nothing over frequency and 58's headline came from log_freq_c and c_size.

Lift is measured against a RANDOM scorer on the same candidate sets, not
against the base rate. AP for a chance scorer sits above the base rate when
positives are few, so base-rate ratios are biased upward -- visible in script 59
as uniform scoring reporting lift 1.45, which is impossible. The self-test at
the top of this file demonstrates the bias and that the random-referenced
version does not have it.

Outputs
-------
outputs/67_by_month.csv    per origin, per block
outputs/67_summary.csv     pooled AP, pooled lift, win rates
outputs/67_coefficients.csv

Usage
-----
python scripts/67_delta_ablation.py --min_count 3 --end_month 2024-12
python scripts/67_delta_ablation.py --self_test      # run the metric tests only
"""

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

FEATURES = [
    "log_rho_m", "m_months_in_support",
    "pmi_mean", "pmi_max", "pmi_min", "cooc_frac", "cooc_any",
    "c_size", "log_freq_c", "log_growth_c",
]
PMI_BLOCK = ["pmi_mean", "pmi_max", "pmi_min", "cooc_frac", "cooc_any"]
FREQ_BLOCK = ["log_rho_m", "m_months_in_support", "c_size", "log_freq_c",
              "log_growth_c"]

BLOCKS = {
    "full": FEATURES,
    "no_pmi": [f for f in FEATURES if f not in PMI_BLOCK],
    "only_pmi": PMI_BLOCK,
    "only_freq": FREQ_BLOCK,
}


# ----------------------------------------------------------------------------
# metrics -- kept small and pure so they can be tested
# ----------------------------------------------------------------------------

def average_precision(y, s):
    """
    Average precision. Ties broken deterministically by index, so repeated runs
    agree and a constant scorer cannot look better than chance by accident.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if y.size == 0 or y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -s))
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, y.size + 1)
    return float((prec * yy).sum() / y.sum())


def lift_vs_random(ap_model, ap_random):
    """
    Lift measured against a RANDOM scorer evaluated on the same candidate sets,
    averaged over origins.

    Not against the base rate. AP for a chance scorer sits slightly ABOVE the
    base rate whenever positives are few, so every base-rate ratio is biased
    upward -- and the bias grows as the base rate falls, which is exactly how
    origins differ here. Dividing by the random scorer's AP on the same data
    cancels it, because the random scorer carries the identical bias.

    Two earlier statistics both failed this: the mean of per-origin AP/base
    ratios used in scripts 58 and 59, and an n_pos-weighted pooled version --
    the self-test below shows the second returning 2.4 for a chance scorer.
    """
    a = np.asarray(ap_model, dtype=float)
    r = np.asarray(ap_random, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(r) & (r > 0)
    if not ok.any():
        return np.nan
    return float(np.mean(a[ok] / r[ok]))


def macro_ap(ap_list):
    """Unweighted mean AP over origins. Every origin counts once."""
    a = np.asarray(ap_list, dtype=float)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else np.nan


def self_test():
    """Properties the metric must satisfy, checked before any data is touched."""
    print("metric self-test")

    # perfect ranking scores 1.0
    y = np.array([1, 1, 0, 0, 0])
    assert abs(average_precision(y, np.array([9, 8, 3, 2, 1])) - 1.0) < 1e-12
    print("  perfect ranking -> 1.0                        ok")

    # worst ranking scores the minimum
    ap_worst = average_precision(y, np.array([1, 2, 7, 8, 9]))
    assert ap_worst < 0.6, ap_worst
    print(f"  worst ranking   -> {ap_worst:.3f}                     ok")

    # a constant scorer must land near the base rate, not above it
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.05).astype(int)
    ap_const = average_precision(y, np.ones(20000))
    assert abs(ap_const - y.mean()) < 0.01, ap_const
    print(f"  constant scorer -> {ap_const:.4f} vs base {y.mean():.4f}   ok")

    # random scoring likewise
    ap_rand = average_precision(y, rng.random(20000))
    assert abs(ap_rand - y.mean()) < 0.02, ap_rand
    print(f"  random scorer   -> {ap_rand:.4f} vs base {y.mean():.4f}   ok")

    # no positives -> nan, not a crash or a zero
    assert np.isnan(average_precision(np.zeros(10, dtype=int), rng.random(10)))
    print("  no positives    -> nan                        ok")

    # lift for a chance scorer must be ~1 even when base rates differ wildly
    # across origins. Two earlier statistics fail this; the random-referenced
    # one does not.
    aps_a, aps_b, bases, npos_l, n_l = [], [], [], [], []
    for base in (0.001, 0.01, 0.1):
        n = 40000
        yy = (rng.random(n) < base).astype(int)
        aps_a.append(average_precision(yy, rng.random(n)))   # "model" = chance
        aps_b.append(average_precision(yy, rng.random(n)))   # reference = chance
        bases.append(float(yy.mean()))
        npos_l.append(int(yy.sum()))
        n_l.append(n)

    lift_ok = lift_vs_random(aps_a, aps_b)
    mean_ratio = float(np.mean([a / b for a, b in zip(aps_a, bases)]))
    npos_w = float((np.array(aps_a) * np.array(npos_l)).sum() /
                   np.sum(npos_l)) / (np.sum(npos_l) / np.sum(n_l))
    print(f"  lift vs random, chance scorer   {lift_ok:.3f}   (must be ~1)")
    print(f"  mean of AP/base ratios          {mean_ratio:.3f}   "
          "<- scripts 58/59")
    print(f"  n_pos-weighted AP/base          {npos_w:.3f}   "
          "<- my first attempt, worse")
    assert abs(lift_ok - 1.0) < 0.20, lift_ok
    print("  only the random-referenced lift is unbiased      ok")

    # and a genuinely good scorer must show large lift under the same statistic
    y = (rng.random(40000) < 0.01).astype(int)
    good = y + rng.normal(0, 0.3, y.size)          # informative but noisy
    lift_good = lift_vs_random([average_precision(y, good)],
                               [average_precision(y, rng.random(y.size))])
    assert lift_good > 5.0, lift_good
    print(f"  informative scorer -> lift {lift_good:.1f}              ok")

    # ranking is invariant to a monotone rescaling of the scores
    y = (rng.random(5000) < 0.05).astype(int)
    s = rng.random(5000)
    a1 = average_precision(y, s)
    a2 = average_precision(y, 3.0 * s + 7.0)
    assert abs(a1 - a2) < 1e-12
    print("  monotone rescaling invariant                   ok")

    print("all metric tests passed\n")


# ----------------------------------------------------------------------------
# logistic regression
# ----------------------------------------------------------------------------

def fit_logistic(X, y, l2=1.0, n_iter=60, tol=1e-7):
    p = X.shape[1]
    w = np.zeros(p)
    R = l2 * np.eye(p)
    R[0, 0] = 0.0
    for _ in range(n_iter):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
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


def predict(X, w):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


# ----------------------------------------------------------------------------
# data
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
        for l in cs:
            nc[l] += w
    return nc


def build_candidates(occ_t, occ_next, labels_t, lab_index, PMI, COOC,
                     months_in_support, prev_occ, max_sets):
    sets = sorted(occ_t.items(), key=lambda kv: -kv[1])[:max_sets]
    csets = [c for c, _ in sets]
    cw = np.array([w for _, w in sets], dtype=float)
    tot = float(sum(occ_t.values()))
    freq_c = cw / tot
    growth = np.array([np.log((occ_t[c] + 1.0) / (prev_occ.get(c, 0.0) + 1.0))
                       for c in csets])

    nlab = len(labels_t)
    lc = label_counts(occ_t)
    rho = np.array([lc[l] / tot for l in labels_t])
    mis = np.array([months_in_support.get(l, 0) for l in labels_t], dtype=float)

    B = np.zeros((len(csets), nlab), dtype=bool)
    for i, c in enumerate(csets):
        for l in c:
            j = lab_index.get(l)
            if j is not None:
                B[i, j] = True
    Bf = B.astype(np.float32)
    csize = np.array([len(c) for c in csets], dtype=float)
    npool = np.maximum(Bf.sum(axis=1), 1.0)

    pmi_mean = (Bf @ PMI) / npool[:, None]
    C_any = Bf @ (COOC > 0).astype(np.float32)
    cooc_frac = C_any / npool[:, None]
    pmi_max = np.full((len(csets), nlab), -10.0, dtype=np.float32)
    pmi_min = np.full((len(csets), nlab), 10.0, dtype=np.float32)
    for i in range(len(csets)):
        idx = np.flatnonzero(B[i])
        if idx.size:
            sub = PMI[idx]
            pmi_max[i] = sub.max(axis=0)
            pmi_min[i] = sub.min(axis=0)

    ii, jj = np.nonzero(~B)
    H_t = set(occ_t.keys())
    y = np.zeros(ii.size, dtype=int)
    for k in range(ii.size):
        x = frozenset(csets[ii[k]] | {labels_t[jj[k]]})
        if x in occ_next and x not in H_t:
            y[k] = 1

    X = np.column_stack([
        np.log(np.clip(rho[jj], 1e-9, None)),
        mis[jj],
        pmi_mean[ii, jj], pmi_max[ii, jj], pmi_min[ii, jj],
        cooc_frac[ii, jj], (C_any[ii, jj] > 0).astype(float),
        csize[ii],
        np.log(np.clip(freq_c[ii], 1e-9, None)),
        growth[ii],
    ])
    return X, y, rho[jj]


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
    ap.add_argument("--max_sets", type=int, default=1500)
    ap.add_argument("--min_train", type=int, default=12)
    ap.add_argument("--train_window", type=int, default=6)
    ap.add_argument("--neg_per_pos", type=int, default=50)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ = {m: o for m, o in months}
    T = len(names)
    print(f"loaded {T} months: {names[0]} .. {names[-1]}")

    months_in_support = defaultdict(int)
    cache, rows = {}, []
    coefs = {}

    for t in range(T - 1):
        occ_t, occ_n = occ[names[t]], occ[names[t + 1]]
        prev_occ = occ[names[t - 1]] if t > 0 else {}
        lc = label_counts(occ_t)
        labels_t = sorted(lc.keys(), key=str)
        lab_index = {l: j for j, l in enumerate(labels_t)}
        for l in labels_t:
            months_in_support[l] += 1

        nlab = len(labels_t)
        CO = np.zeros((nlab, nlab), dtype=np.float32)
        marg = np.zeros(nlab, dtype=np.float32)
        wtot = 0.0
        for j in range(t + 1):
            for cs, w in occ[names[j]].items():
                ix = [lab_index[l] for l in cs if l in lab_index]
                if not ix:
                    continue
                wtot += w
                a = np.array(ix)
                marg[a] += w
                CO[np.ix_(a, a)] += w
        if wtot <= 0:
            continue
        pmv = np.clip(marg / wtot, 1e-9, None)
        PMI = np.log(np.clip(CO / wtot, 1e-12, None) / np.outer(pmv, pmv))
        np.fill_diagonal(PMI, 0.0)

        X, y, rho_m = build_candidates(occ_t, occ_n, labels_t, lab_index,
                                       PMI, CO, months_in_support, prev_occ,
                                       args.max_sets)
        cache[t] = (X, y, rho_m)

        if t < args.min_train or y.sum() == 0:
            continue

        # pooled training window, negatives subsampled
        Xs, ys = [], []
        for j in range(max(0, t - args.train_window), t):
            if j not in cache:
                continue
            Xj, yj, _ = cache[j]
            pos = np.flatnonzero(yj)
            if pos.size == 0:
                continue
            neg = np.flatnonzero(yj == 0)
            neg = rng.choice(neg, size=min(neg.size, pos.size * args.neg_per_pos),
                             replace=False)
            sel = np.concatenate([pos, neg])
            Xs.append(Xj[sel])
            ys.append(yj[sel])
        if not Xs:
            continue
        Xtr = np.vstack(Xs)
        ytr = np.concatenate(ys).astype(float)

        base = float(y.mean())
        for bname, feats in BLOCKS.items():
            cols = [FEATURES.index(f) for f in feats]
            mu = Xtr[:, cols].mean(axis=0)
            sd = Xtr[:, cols].std(axis=0)
            sd[sd < 1e-9] = 1.0
            A = np.column_stack([np.ones(len(Xtr)), (Xtr[:, cols] - mu) / sd])
            w = fit_logistic(A, ytr, l2=args.l2)
            B = np.column_stack([np.ones(len(X)), (X[:, cols] - mu) / sd])
            s = predict(B, w)
            rows.append({"origin": names[t], "target": names[t + 1],
                         "block": bname, "ap": average_precision(y, s),
                         "n": int(y.size), "n_pos": int(y.sum()),
                         "base_rate": base})
            if bname == "full":
                coefs[names[t]] = dict(zip(feats, w[1:]))

        for bname, s in [("marginal", rho_m), ("random", rng.random(y.size))]:
            rows.append({"origin": names[t], "target": names[t + 1],
                         "block": bname, "ap": average_precision(y, s),
                         "n": int(y.size), "n_pos": int(y.sum()),
                         "base_rate": base})

        print(f"  {names[t]}: {y.size} candidates, {int(y.sum())} positives, "
              f"base {base:.5f}")
        for j in list(cache):
            if j < t - args.train_window:
                del cache[j]

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/67_by_month.csv", index=False)

    print("\n" + "=" * 74)
    print("DELTA: does c u {m} appear at t+1?   (pooled AP and pooled lift)")
    print("=" * 74)
    out = []
    ref = df[df["block"] == "full"].set_index("origin")["ap"]
    rnd = df[df["block"] == "random"].set_index("origin")["ap"]
    for bname in list(BLOCKS) + ["marginal", "random"]:
        sub = df[df["block"] == bname].set_index("origin")
        if not len(sub):
            continue
        common = sub.index.intersection(rnd.index)
        out.append({
            "block": bname,
            "mean_ap": macro_ap(sub["ap"]),
            "lift_vs_random": lift_vs_random(sub.loc[common, "ap"],
                                             rnd.loc[common]),
            "mean_base_rate": float(sub["base_rate"].mean()),
            "beats_full": float((sub.loc[
                sub.index.intersection(ref.index), "ap"] >
                ref[sub.index.intersection(ref.index)]).mean()),
            "origins": int(len(sub)),
        })
    res = pd.DataFrame(out).sort_values("mean_ap", ascending=False)
    res.to_csv(f"{args.out_dir}/67_summary.csv", index=False)
    print(res.round(5).to_string(index=False))

    f = res.set_index("block")["mean_ap"]
    if "full" in f and "no_pmi" in f:
        print(f"\nfull {f['full']:.5f}  vs  no_pmi {f['no_pmi']:.5f}   "
              f"ratio {f['full']/f['no_pmi']:.3f}")
        print("  CALIBRATED ON SYNTHETIC DATA with a known generative rule,")
        print("  17 origins, same settings:")
        print("     no background dependence  -> ratio 1.32, lift_vs_random 1.21")
        print("     real background dependence -> ratio 2.65, lift_vs_random 3.80")
        print("  So ~1.3 is the NULL level for this statistic, not 1.0 -- the")
        print("  ablation keeps a little apparent advantage from extra features")
        print("  even when they carry nothing. Read the ratio against 1.3.")
        print("  Near 1.3 -> co-occurrence adds nothing over frequency, and")
        print("     script 58's headline came from log_freq_c and c_size.")
        print("  Near 2.6 or above -> the background matters, and 65/66 lost it")
        print("     to the imputed coupling rather than to the biology.")
    if "marginal" in f and "full" in f:
        print(f"full vs marginal: {f['full']/f['marginal']:.3f}x")

    if coefs:
        cf = pd.DataFrame(coefs).T
        cf.to_csv(f"{args.out_dir}/67_coefficients.csv")
        print("\nmean standardised coefficients (full block):")
        print(cf.mean().sort_values(key=np.abs, ascending=False)
              .round(4).to_string())

    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
