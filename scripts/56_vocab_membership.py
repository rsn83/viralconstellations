#!/usr/bin/env python
"""
56_vocab_membership.py

Question
--------
Given the vocabulary at time t, can we predict the vocabulary at t+1?

Script 55 did NOT answer this. It fitted f(t+1) under a frequency loss and read
support off by thresholding, which is a by-product, not a test. This script
states the target directly.

Target
------
For every label in the causal universe at t, a binary label:
    y_i = 1 if label i is in the support at t+1, else 0

Support membership is judged at a fixed detection depth so that months of
different sequencing effort are comparable (script 51: raw support size is
largely a readout of sequencing effort, corr +0.815 with n_seqs).

What is observed : presence/absence of each label per month, at fixed depth
What is hidden   : whether an absence is true loss or a detection failure
What is predicted: P(label present at t+1)
What is NOT claimed: anything about which constellations form, or that a
                   label's first detection is its origination
Why useful       : this is the proposal channel at label level, and it sets the
                   candidate pool any set-level forecaster draws from

Features (all causal: computed from months <= t only)
-----------------------------------------------------
Per-label history, which script 55's models never had access to:
    present_now, months_since_last_seen, months_since_first_seen,
    n_months_present, n_runs, current_run_length, longest_gap,
    log freq now, log freq when last seen, freq slope over last 3 and 6 months,
    occupancy fraction over last 3/6/12 months
Plus two month-level covariates so the model can absorb surveillance effort:
    log n_seqs at t, change in log n_seqs

Models
------
  base_rate        predicts the training base rate for everyone (calibration ref)
  persistence      predicts present at t+1 iff present at t
  recency          score = 1 / (1 + months_since_last_seen)
  logistic         L2 logistic regression on the full feature set

Evaluation
----------
Rolling-origin (expanding window): fit on all month pairs up to t, predict t+1.
Reported as average precision and lift over base rate, and separately on the two
strata that matter, because pooled AP is dominated by the easy majority:
    ENTRY  : labels absent at t   -> did they appear?
    EXIT   : labels present at t  -> did they survive?
Persistence is by construction perfect on neither and is the baseline to beat.

Outputs
-------
outputs/56_membership_by_month.csv   per-origin metrics for every model
outputs/56_membership_summary.csv    pooled comparison, overall and by stratum
outputs/56_coefficients.csv          final-fit logistic coefficients

Usage
-----
python scripts/56_vocab_membership.py --min_count 3 --end_month 2024-12
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")


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


def node_freqs(occ):
    total = float(sum(occ.values()))
    nc = defaultdict(float)
    for cs, c in occ.items():
        for lab in cs:
            nc[lab] += c
    return {lab: v / total for lab, v in nc.items()}, total


def rarefy(occ, depth, min_count, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    if total < depth:
        return None
    draws = rng.multinomial(depth, counts / total)
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws >= min_count)[0]}


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if y.sum() == 0 or y.size == 0:
        return np.nan
    # deterministic tie-breaking: tiny jitter by index, stable across runs
    order = np.lexsort((np.arange(y.size), -s))
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, y.size + 1)
    return float((prec * yy).sum() / y.sum())


def auc_score(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------------
# L2 logistic regression (plain numpy, no sklearn dependency)
# ----------------------------------------------------------------------------

def fit_logistic(X, y, l2=1.0, n_iter=200, tol=1e-7):
    """Newton-IRLS with ridge. X should already include a constant column."""
    n, p = X.shape
    w = np.zeros(p)
    R = l2 * np.eye(p)
    R[0, 0] = 0.0  # do not penalise the intercept
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
        w_new = w - step
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def predict_logistic(X, w):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def standardise(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ----------------------------------------------------------------------------
# feature construction
# ----------------------------------------------------------------------------

FEATURES = [
    "present_now", "log_freq_now", "months_since_last_seen",
    "months_since_first_seen", "n_months_present", "n_runs",
    "current_run_length", "longest_gap", "log_freq_last_seen",
    "slope3", "slope6", "occ3", "occ6", "occ12",
    "log_n_seqs", "d_log_n_seqs",
]

LOGF_FLOOR = -9.0  # log frequency assigned when absent


def build_features(labels, hist_present, hist_freq, t, log_nseqs):
    """
    hist_present : dict label -> boolean array over months 0..t
    hist_freq    : dict label -> float array over months 0..t
    Returns X (len(labels) x len(FEATURES)) using only columns <= t.
    """
    X = np.zeros((len(labels), len(FEATURES)))
    d_log = (log_nseqs[t] - log_nseqs[t - 1]) if t >= 1 else 0.0

    for r, lab in enumerate(labels):
        p = hist_present[lab][:t + 1]
        f = hist_freq[lab][:t + 1]
        on = np.flatnonzero(p)
        present_now = bool(p[-1])

        if on.size:
            first, last = int(on[0]), int(on[-1])
            since_last = t - last
            since_first = t - first
            n_present = int(p.sum())
            n_runs = 1 + int((np.diff(on) > 1).sum())
            logf_last = np.log(max(f[last], 1e-9))
            gaps = np.diff(on) - 1
            longest_gap = int(gaps.max()) if gaps.size else 0
        else:
            since_last = t + 1
            since_first = 0
            n_present = 0
            n_runs = 0
            logf_last = LOGF_FLOOR
            longest_gap = 0

        # current run of consecutive presence ending at t
        run = 0
        for v in p[::-1]:
            if v:
                run += 1
            else:
                break

        def slope(w):
            if t + 1 < 2:
                return 0.0
            seg = f[max(0, t + 1 - w):t + 1]
            m = seg > 0
            if m.sum() < 2:
                return 0.0
            xs = np.arange(seg.size)[m].astype(float)
            ys = np.log(np.clip(seg[m], 1e-9, None))
            xs = xs - xs.mean()
            den = (xs ** 2).sum()
            return float((xs * (ys - ys.mean())).sum() / den) if den > 0 else 0.0

        def occ(w):
            seg = p[max(0, t + 1 - w):t + 1]
            return float(seg.mean()) if seg.size else 0.0

        X[r] = [
            1.0 if present_now else 0.0,
            np.log(max(f[-1], 1e-9)) if present_now else LOGF_FLOOR,
            float(since_last), float(since_first), float(n_present),
            float(n_runs), float(run), float(longest_gap), logf_last,
            slope(3), slope(6), occ(3), occ(6), occ(12),
            log_nseqs[t], d_log,
        ]
    return X


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
    ap.add_argument("--depth", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--min_train", type=int, default=18,
                    help="origins before the first evaluated forecast")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    print(f"loaded {len(months)} months: {names[0]} .. {names[-1]}")

    # ---- depth-controlled support and raw frequency per month ---------------
    support, freq, nseq = {}, {}, {}
    for month, occ in months:
        f, tot = node_freqs(occ)
        freq[month] = f
        nseq[month] = tot
        seen, nrep = defaultdict(int), 0
        for _ in range(args.reps):
            sub = rarefy(occ, args.depth, args.min_count, rng)
            if sub is None:
                continue
            nrep += 1
            sf, _ = node_freqs(sub)
            for lab in sf:
                seen[lab] += 1
        support[month] = ({lab for lab, c in seen.items() if c >= nrep / 2}
                          if nrep else set(f.keys()))

    keep = [m for m in names if support[m]]
    print(f"months with a depth-{args.depth} support: {len(keep)} / {len(names)}")
    names = keep
    T = len(names)

    all_labels = sorted({lab for m in names for lab in support[m]}, key=str)
    print(f"labels ever in support: {len(all_labels)}")

    hist_present, hist_freq = {}, {}
    for lab in all_labels:
        hist_present[lab] = np.array([lab in support[m] for m in names])
        hist_freq[lab] = np.array([freq[m].get(lab, 0.0) for m in names])
    log_nseqs = np.log(np.array([nseq[m] for m in names]))

    # causal universe: labels that have been in support at or before month t
    universe = []
    seen_so_far = set()
    for j, m in enumerate(names):
        seen_so_far |= support[m]
        universe.append(sorted(seen_so_far, key=str))
    print(f"causal universe grows {len(universe[0])} -> {len(universe[-1])}")

    # ---- cache design matrices per origin -----------------------------------
    print("building features ...")
    Xc, Yc = {}, {}
    for t in range(T - 1):
        labs = universe[t]
        Xc[t] = build_features(labs, hist_present, hist_freq, t, log_nseqs)
        Yc[t] = np.array([lab in support[names[t + 1]] for lab in labs], dtype=int)

    # ---- rolling-origin evaluation ------------------------------------------
    print("rolling-origin evaluation ...\n")
    model_names = ["base_rate", "persistence", "recency", "logistic"]
    rows = []
    last_w = None

    i_present = FEATURES.index("present_now")
    i_since = FEATURES.index("months_since_last_seen")

    for t in range(args.min_train, T - 1):
        labs = universe[t]
        Xte, yte = Xc[t], Yc[t]

        # training set: every origin strictly before t, restricted to the
        # universe available at that origin -- nothing from t or later
        Xtr = np.vstack([Xc[j] for j in range(t)])
        ytr = np.concatenate([Yc[j] for j in range(t)])

        Xtr_s, Xte_s = standardise(Xtr, Xte)
        Xtr_s = np.column_stack([np.ones(len(Xtr_s)), Xtr_s])
        Xte_s = np.column_stack([np.ones(len(Xte_s)), Xte_s])
        w = fit_logistic(Xtr_s, ytr, l2=args.l2)
        last_w = w

        scores = {
            "base_rate": np.full(len(yte), ytr.mean()),
            "persistence": Xte[:, i_present].astype(float),
            "recency": 1.0 / (1.0 + Xte[:, i_since]),
            "logistic": predict_logistic(Xte_s, w),
        }

        present_mask = Xte[:, i_present] > 0.5
        strata = {
            "all": np.ones(len(yte), dtype=bool),
            "entry": ~present_mask,   # absent at t: will it appear?
            "exit": present_mask,     # present at t: will it survive?
        }

        for mname, s in scores.items():
            for sname, msk in strata.items():
                if msk.sum() == 0 or yte[msk].sum() == 0:
                    continue
                base = float(yte[msk].mean())
                apv = average_precision(yte[msk], s[msk])
                rows.append({
                    "origin": names[t], "target": names[t + 1],
                    "model": mname, "stratum": sname,
                    "n": int(msk.sum()), "n_pos": int(yte[msk].sum()),
                    "base_rate": base,
                    "ap": apv,
                    "lift": apv / base if base > 0 else np.nan,
                    "auc": auc_score(yte[msk], s[msk]),
                    "n_train": len(ytr),
                })

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/56_membership_by_month.csv", index=False)

    # ---- summary ------------------------------------------------------------
    print("=" * 74)
    print("VOCABULARY MEMBERSHIP AT t+1  (rolling-origin, expanding window)")
    print("=" * 74)
    for sname in ["all", "entry", "exit"]:
        sub = df[df["stratum"] == sname]
        if not len(sub):
            continue
        g = sub.groupby("model").agg(
            ap=("ap", "mean"), lift=("lift", "mean"), auc=("auc", "mean"),
            base=("base_rate", "mean"), n=("n", "mean"),
            n_pos=("n_pos", "mean"), origins=("ap", "count"),
        ).reset_index().sort_values("ap", ascending=False)
        print(f"\n--- stratum: {sname} "
              f"(mean {g['n'].iloc[0]:.0f} labels, base rate "
              f"{g['base'].iloc[0]:.4f}) ---")
        print(g.round(4).to_string(index=False))

        best = g.iloc[0]
        pers = g[g["model"] == "persistence"]
        if len(pers):
            print(f"  best {best['model']} AP {best['ap']:.4f} vs persistence "
                  f"{pers['ap'].iloc[0]:.4f}  (lift over base "
                  f"{best['lift']:.2f}x)")

    summ = df.groupby(["stratum", "model"]).agg(
        ap=("ap", "mean"), lift=("lift", "mean"), auc=("auc", "mean"),
        base_rate=("base_rate", "mean"), origins=("ap", "count"),
    ).reset_index()
    summ.to_csv(f"{args.out_dir}/56_membership_summary.csv", index=False)

    # per-origin win rate against persistence, on the entry stratum
    for sname in ["entry", "exit"]:
        a = df[(df["model"] == "logistic") & (df["stratum"] == sname)] \
            .set_index("origin")["ap"]
        b = df[(df["model"] == "persistence") & (df["stratum"] == sname)] \
            .set_index("origin")["ap"]
        common = a.index.intersection(b.index)
        if len(common):
            print(f"\nlogistic beats persistence on {sname}: "
                  f"{(a[common] > b[common]).sum()} / {len(common)} origins")

    # ---- coefficients -------------------------------------------------------
    if last_w is not None:
        co = pd.DataFrame({
            "feature": ["intercept"] + FEATURES,
            "coef_standardised": last_w,
        }).sort_values("coef_standardised", key=np.abs, ascending=False)
        co.to_csv(f"{args.out_dir}/56_coefficients.csv", index=False)
        print("\n--- logistic coefficients at the final origin (standardised) ---")
        print(co.round(4).to_string(index=False))

    print("\nread:")
    print("  entry stratum is the real test. Persistence scores 0 there by")
    print("  construction: it can never predict a label that is currently absent.")
    print("  Lift well above 1 on entry means label history carries usable signal")
    print("  and vocabulary change IS predictable. Lift near 1 means it is not,")
    print("  and the earlier claim stands.")
    print("  exit stratum: persistence is strong by construction; the question is")
    print("  only whether history improves on it.")
    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
