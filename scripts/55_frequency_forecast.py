#!/usr/bin/env python
"""
55_frequency_forecast.py

The object being modelled
------------------------
f(t) in [0,1]^N, the within-month frequency of every mutation label.

Two identities make this the right object:

    mean labels per sequence(t) = sum_i f_i(t)          (exact)
    vocabulary(t)               = { i : f_i(t) > eps }  (support)

So "how does the vocabulary change" and "why does occupancy per sequence rise"
are not two questions. They are the support and the sum of one vector. Model
f(t), and both fall out.

What is observed : f(t), with sampling error ~ sqrt(f(1-f)/n)
What is hidden   : which labels are linked, i.e. travel together in genomes
What is predicted: f(t+1) from f(<=t)
What is NOT claimed: anything about which SETS appear. f is marginals only, and
                   marginals do not determine the joint. The shortfall between
                   what marginals predict and what constellations actually form
                   is the co-occurrence problem -- this script measures the
                   marginal half so that shortfall becomes quantifiable.

Leakage discipline (scripts 38-54 failed this in two ways)
----------------------------------------------------------
1. The label universe at month t is ONLY labels observed in months <= t.
   Script 54 built one universe from all 60 months, which contaminated every
   'absent' transition rate.
2. No switch months, no regime labels, no cluster partition anywhere. Those all
   descend from a partition fitted on pooled months. Here a sweep is simply a
   month where persistence forecasts badly -- observed, never labelled.

Models (all fitted on months <= t, predicting t+1)
--------------------------------------------------
  persistence     f(t+1) = f(t)                     -- the baseline to beat
  global_growth   one shared growth rate r          -- tests whether per-label
                                                       rates buy anything
  indep_logistic  logit f_i(t+1) = logit f_i(t) + r_i, r_i from the last W
                  months, shrunk toward the global rate
  lowrank_k       F[:, <=t] ~ U V^T at rank k; extrapolate the k component
                  trajectories and reconstruct. Rank k is the number of
                  co-moving blocks -- what clustering was reaching for, fitted
                  causally instead of on pooled months.

Noise floor
-----------
Sequences within a month are split in half and f computed on each. The MAE
between halves is the irreducible sampling error. Any model error near that
floor is as good as the data allows; comparing against it says whether a model
gap is real or just resampling noise.

Outputs
-------
outputs/55_forecast_by_month.csv  per-month error for every model
outputs/55_model_summary.csv      pooled comparison + noise floor
outputs/55_functionals.csv        predicted vs actual vocabulary size and
                                  occupancy (sum f), per month

Usage
-----
python scripts/55_frequency_forecast.py --min_count 3 --end_month 2024-12
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


def node_counts(occ):
    nc = defaultdict(float)
    for cs, c in occ.items():
        for lab in cs:
            nc[lab] += c
    return nc, float(sum(occ.values()))


def rarefy(occ, depth, min_count, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    if total < depth:
        return None
    draws = rng.multinomial(depth, counts / total)
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws >= min_count)[0]}


def split_half_mae(occ, rng, n_rep=5):
    """Irreducible sampling error: |f_A - f_B| over two halves of the month."""
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    if total < 200:
        return np.nan
    maes = []
    for _ in range(n_rep):
        a = rng.binomial(counts.astype(int), 0.5)
        b = counts - a
        if a.sum() == 0 or b.sum() == 0:
            continue
        fa, fb = defaultdict(float), defaultdict(float)
        for k, ca, cb in zip(keys, a, b):
            for lab in k:
                fa[lab] += ca
                fb[lab] += cb
        labs = set(fa) | set(fb)
        va = np.array([fa[l] / a.sum() for l in labs])
        vb = np.array([fb[l] / b.sum() for l in labs])
        maes.append(np.abs(va - vb).mean())
    return float(np.mean(maes)) if maes else np.nan


# ----------------------------------------------------------------------------
# models  (each takes F[:, :t+1] and returns a prediction for column t+1)
# ----------------------------------------------------------------------------

LO = 1e-4


def _logit(x):
    x = np.clip(x, LO, 1 - LO)
    return np.log(x / (1 - x))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def predict_persistence(F):
    return F[:, -1].copy()


def _per_label_rates(F, W, min_obs=3):
    """OLS slope of logit f_i over the last W months where the label is present."""
    H = F[:, -W:] if F.shape[1] >= W else F
    L = _logit(H)
    present = H > 0
    n = H.shape[1]
    x = np.arange(n, dtype=float)
    rates = np.full(F.shape[0], np.nan)
    nobs = present.sum(axis=1)
    for i in np.flatnonzero(nobs >= min_obs):
        m = present[i]
        xa, ya = x[m], L[i, m]
        xa = xa - xa.mean()
        denom = (xa ** 2).sum()
        if denom > 0:
            rates[i] = float((xa * (ya - ya.mean())).sum() / denom)
    return rates, nobs


def predict_global_growth(F, W):
    rates, nobs = _per_label_rates(F, W)
    g = np.nanmedian(rates) if np.isfinite(rates).any() else 0.0
    return _sigmoid(_logit(F[:, -1]) + g)


def predict_indep_logistic(F, W, prior_n=3.0, clip=2.0):
    """Per-label rate shrunk toward the global rate by number of observations."""
    rates, nobs = _per_label_rates(F, W)
    g = np.nanmedian(rates) if np.isfinite(rates).any() else 0.0
    r = np.where(np.isfinite(rates), rates, g)
    w = nobs / (nobs + prior_n)
    r = w * r + (1 - w) * g
    r = np.clip(r, -clip, clip)
    return _sigmoid(_logit(F[:, -1]) + r)


def predict_lowrank(F, k, W):
    """
    Truncated SVD of the labels x months matrix, then extrapolate each component
    trajectory one step by a linear fit on its last W values.
    """
    if F.shape[1] < max(k + 1, 3):
        return predict_persistence(F)
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    k = min(k, len(S))
    U, S, Vt = U[:, :k], S[:k], Vt[:k]
    comp = (S[:, None] * Vt)            # k x months
    w = min(W, comp.shape[1])
    x = np.arange(w, dtype=float)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    nxt = np.empty(k)
    for j in range(k):
        y = comp[j, -w:]
        slope = float((xc * (y - y.mean())).sum() / denom) if denom > 0 else 0.0
        nxt[j] = y[-1] + slope
    return np.clip(U @ nxt, 0.0, 1.0)


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
    ap.add_argument("--depth", type=int, default=5000,
                    help="detection depth; sets the support threshold only")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--window", type=int, default=6, help="months for rate fits")
    ap.add_argument("--min_train", type=int, default=12,
                    help="months of history before the first forecast")
    ap.add_argument("--ranks", default="1,2,3,4,6,8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ranks = [int(x) for x in args.ranks.split(",")]

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    print(f"loaded {len(months)} months: {names[0]} .. {names[-1]}")

    # ---- frequency estimates (raw, unbiased) and depth-controlled support ---
    freqs, supports, nseqs, noise = {}, {}, {}, {}
    for month, occ in months:
        nc, tot = node_counts(occ)
        freqs[month] = {lab: c / tot for lab, c in nc.items()}
        nseqs[month] = tot
        noise[month] = split_half_mae(occ, rng)
        seen = defaultdict(int)
        nrep = 0
        for _ in range(args.reps):
            sub = rarefy(occ, args.depth, args.min_count, rng)
            if sub is None:
                continue
            nrep += 1
            sc, _ = node_counts(sub)
            for lab in sc:
                seen[lab] += 1
        supports[month] = ({lab for lab, c in seen.items() if c >= nrep / 2}
                           if nrep else set(freqs[month].keys()))

    eps = args.min_count / args.depth  # detection floor at the reference depth
    print(f"support detection floor at depth {args.depth}: f > {eps:.5f}")

    # ---- causal universe: labels seen up to and including month t ------------
    cum_universe = []
    seen_so_far = set()
    for month in names:
        seen_so_far |= set(freqs[month].keys())
        cum_universe.append(frozenset(seen_so_far))
    print(f"universe grows {len(cum_universe[0])} -> {len(cum_universe[-1])} labels")
    print("(each forecast uses only the universe available at that time)")

    # ---- walk-forward -------------------------------------------------------
    model_names = (["persistence", "global_growth", "indep_logistic"]
                   + [f"lowrank_{k}" for k in ranks])
    rows, frows = [], []

    for t in range(args.min_train, len(names) - 1):
        m_t, m_n = names[t], names[t + 1]
        labs = sorted(cum_universe[t], key=str)          # causal universe
        idx = {l: i for i, l in enumerate(labs)}

        F = np.zeros((len(labs), t + 1))
        for j in range(t + 1):
            fj = freqs[names[j]]
            for lab, v in fj.items():
                if lab in idx:
                    F[idx[lab], j] = v

        y = np.zeros(len(labs))
        for lab, v in freqs[m_n].items():
            if lab in idx:
                y[idx[lab]] = v
        # labels first seen at t+1 are outside the causal universe by
        # construction; they are counted separately, not scored
        n_unseeable = len(set(freqs[m_n]) - cum_universe[t])
        mass_unseeable = sum(v for lab, v in freqs[m_n].items()
                             if lab not in cum_universe[t])

        in_supp_next = np.array([l in supports[m_n] for l in labs])

        preds = {
            "persistence": predict_persistence(F),
            "global_growth": predict_global_growth(F, args.window),
            "indep_logistic": predict_indep_logistic(F, args.window),
        }
        for k in ranks:
            preds[f"lowrank_{k}"] = predict_lowrank(F, k, args.window)

        for name, p in preds.items():
            p = np.clip(p, 0.0, 1.0)
            err = p - y
            active = (F[:, -1] > 0) | (y > 0)
            rows.append({
                "month_t": m_t, "month_t1": m_n, "model": name,
                "mae": float(np.abs(err).mean()),
                "mae_active": float(np.abs(err[active]).mean()) if active.any() else np.nan,
                "rmse": float(np.sqrt((err ** 2).mean())),
                "mae_logit": float(np.abs(_logit(p) - _logit(y)).mean()),
                "corr": (float(np.corrcoef(p, y)[0, 1])
                         if p.std() > 0 and y.std() > 0 else np.nan),
                "pred_occupancy": float(p.sum()),
                "true_occupancy": float(y.sum()),
                "pred_vocab": int((p > eps).sum()),
                "true_vocab": int(in_supp_next.sum()),
                "n_universe": len(labs),
                "n_unseeable": n_unseeable,
                "mass_unseeable": mass_unseeable,
                "noise_floor": noise[m_n],
                "n_seqs_t1": nseqs[m_n],
            })

        frows.append({
            "month_t1": m_n,
            "true_occupancy": float(y.sum()),
            "true_vocab": int(in_supp_next.sum()),
            "persistence_occupancy": float(preds["persistence"].sum()),
            "lowrank_occupancy": float(preds[f"lowrank_{ranks[len(ranks)//2]}"].sum()),
            "n_unseeable": n_unseeable,
            "mass_unseeable": mass_unseeable,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/55_forecast_by_month.csv", index=False)
    fx = pd.DataFrame(frows)
    fx.to_csv(f"{args.out_dir}/55_functionals.csv", index=False)

    # ---- pooled comparison --------------------------------------------------
    print("\n" + "=" * 74)
    print("MODEL COMPARISON (walk-forward, one-step-ahead)")
    print("=" * 74)
    summ = df.groupby("model").agg(
        mae=("mae", "mean"),
        mae_active=("mae_active", "mean"),
        rmse=("rmse", "mean"),
        mae_logit=("mae_logit", "mean"),
        corr=("corr", "mean"),
        n_months=("month_t", "count"),
    ).reset_index().sort_values("mae_active")
    base = df[df["model"] == "persistence"].set_index("month_t1")["mae_active"]
    wins = {}
    for name in model_names:
        s = df[df["model"] == name].set_index("month_t1")["mae_active"]
        common = s.index.intersection(base.index)
        wins[name] = float((s[common] < base[common]).mean())
    summ["beats_persistence"] = summ["model"].map(wins)
    print(summ.round(5).to_string(index=False))

    nf = df.groupby("month_t1")["noise_floor"].first().mean()
    print(f"\nsampling noise floor (half-split MAE): {nf:.6f}")
    best = summ.iloc[0]
    print(f"best model: {best['model']}  mae_active {best['mae_active']:.6f}")
    print(f"ratio to noise floor: {best['mae_active'] / nf:.2f}x"
          if np.isfinite(nf) and nf > 0 else "")
    print("\nread: a model near 1x the floor is as good as the data permits.")
    print("      a model far above it has real structure left to capture.")
    print("      if nothing beats persistence, month-to-month frequency change")
    print("      is not predictable at this resolution -- report that plainly.")

    # ---- rank curve ---------------------------------------------------------
    lr = summ[summ["model"].str.startswith("lowrank_")].copy()
    if len(lr):
        lr["k"] = lr["model"].str.split("_").str[1].astype(int)
        print("\n--- rank curve ---")
        print(lr.sort_values("k")[["k", "mae_active", "corr", "beats_persistence"]]
              .round(5).to_string(index=False))
        print("a clear minimum at some k is an estimate of the number of")
        print("co-moving blocks, obtained without any clustering.")

    # ---- the two functionals ------------------------------------------------
    print("\n" + "=" * 74)
    print("THE TWO FUNCTIONALS: occupancy (sum f) and vocabulary (support)")
    print("=" * 74)
    bestname = best["model"]
    b = df[df["model"] == bestname]
    print(f"(model: {bestname})")
    cols = ["month_t1", "true_occupancy", "pred_occupancy", "true_vocab",
            "pred_vocab", "n_unseeable", "mass_unseeable", "n_seqs_t1"]
    print(b[cols].round(4).tail(30).to_string(index=False))
    occ_err = (b["pred_occupancy"] - b["true_occupancy"]).abs().mean()
    voc_err = (b["pred_vocab"] - b["true_vocab"]).abs().mean()
    p_occ = df[df["model"] == "persistence"]
    print(f"\noccupancy MAE : {bestname} {occ_err:.3f}  vs persistence "
          f"{(p_occ['pred_occupancy'] - p_occ['true_occupancy']).abs().mean():.3f}")
    print(f"vocabulary MAE: {bestname} {voc_err:.1f}  vs persistence "
          f"{(p_occ['pred_vocab'] - p_occ['true_vocab']).abs().mean():.1f}")

    print("\n--- the hard ceiling ---")
    print(f"labels first appearing at t+1 (outside any causal universe): "
          f"mean {b['n_unseeable'].mean():.1f} per month")
    print(f"their share of frequency mass: {b['mass_unseeable'].mean():.4f}")
    print("no model of f can reach these; this bounds achievable recall on new")
    print("vocabulary and should be reported alongside any result.")

    print(f"\nwrote 3 files to {args.out_dir}/")
    print("\nNOTE: f(t) is marginals. It cannot say which constellations form.")
    print("The gap between what these forecasts imply and what sets actually")
    print("appear is the co-occurrence problem, and is now measurable.")


if __name__ == "__main__":
    main()
