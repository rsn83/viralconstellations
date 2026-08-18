#!/usr/bin/env python
"""
87_fit_model.py

Fit a two-parameter model to consecutive months and test whether it beats
copy-forward. This is the theoretical argument for why copy-forward cannot
be beaten.

The model
---------
Under the simplest parametric model of set evolution:

  p_{t+1}(c) proportional to  p_t(c) * exp(beta * |c|)
                             + mu * sum_{m in c} p_t(c \\ {m})

  beta  selection: sets with more mutations grow faster (beta > 0) or slower
  mu    mutation: mass flows from each set to its one-addition neighbours

Copy-forward is the special case beta = 0, mu = 0.

The MLE for beta and mu given two consecutive months is fitted by minimising
negative log-likelihood. The result is compared to copy-forward (KL divergence
of the true next month from the model vs from copy-forward).

The theoretical argument
------------------------
If the fitted beta clusters near zero across months, the data is consistent
with a neutral process and copy-forward is the MLE. That is a formal
justification: copy-forward is the maximum-likelihood predictor under the
simplest parametric model, and the data provides no evidence for departing
from it.

If beta is consistently positive and the model beats copy-forward, selection
is present and estimable -- and the result changes.

Usage
-----
python scripts/87_fit_model.py --min_count 3 --end_month 2024-12
python scripts/87_fit_model.py --self_test
"""

import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

def model_q(p, sets, beta, mu):
    set_idx = {c: i for i, c in enumerate(sets)}
    q = np.zeros(len(sets))
    for i, c in enumerate(sets):
        q[i] = p[i] * np.exp(beta * len(c))
        for m in c:
            par = frozenset(c - {m})
            if par in set_idx:
                q[i] += mu * p[set_idx[par]]
    s = q.sum()
    return q / s if s > 0 else np.ones(len(sets)) / len(sets)


def neg_ll(params, p, q_true, sets):
    beta, lmu = params
    mu = np.exp(np.clip(lmu, -10, 5))
    q = model_q(p, sets, beta, mu)
    return -float(np.sum(q_true * np.log(np.clip(q, 1e-300, None))))


def fit(p, q_true, sets):
    """Fit beta and mu. Try several starting points."""
    best_val, best_params = np.inf, (0.0, 0.0)
    for b0, lm0 in [(0.0, -4), (0.2, -3), (0.5, -3),
                    (-0.2, -4), (1.0, -2), (0.0, -6)]:
        try:
            r = minimize(neg_ll, [b0, lm0], args=(p, q_true, sets),
                         method="Nelder-Mead",
                         options={"xatol": 1e-9, "fatol": 1e-9,
                                  "maxiter": 10000})
            if r.fun < best_val:
                best_val = r.fun
                best_params = (float(r.x[0]),
                               float(np.exp(np.clip(r.x[1], -10, 5))))
        except Exception:
            pass
    return best_params


def kl(p, q):
    q = np.clip(q, 1e-300, None)
    ok = p > 0
    return float(np.sum(p[ok] * np.log(p[ok] / q[ok])))


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


def top_sets(occ, max_sets):
    items = sorted(occ.items(), key=lambda kv: -kv[1])[:max_sets]
    w = np.array([v for _, v in items], dtype=float)
    return [c for c, _ in items], w / w.sum()


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    print("self-test")

    # when beta=0 and mu=0, copy-forward is optimal and MLE should recover it
    sets = [frozenset({1}), frozenset({1, 2}), frozenset({1, 2, 3})]
    p = np.array([0.7, 0.2, 0.1])
    b, mu = fit(p, p, sets)
    assert abs(b) < 0.01 and mu < 0.01, (b, mu)
    print(f"  stationary data -> beta {b:.4f}, mu {mu:.4f} (both near 0) ok")

    # when mass shifts to larger sets, beta should be positive
    q = np.array([0.3, 0.4, 0.3])
    b2, mu2 = fit(p, q, sets)
    assert b2 > 0.1, b2
    print(f"  mass toward larger sets -> beta {b2:.4f} > 0          ok")

    # model_q at beta=0, mu=0 must return p unchanged
    q0 = model_q(p, sets, 0.0, 0.0)
    assert np.allclose(q0, p, atol=1e-9), q0
    print("  model at beta=0, mu=0 returns the current distribution  ok")

    # kl must be zero for identical distributions
    assert abs(kl(p, p)) < 1e-12
    assert kl(p, q) > 0
    print("  KL: zero for identical, positive for different          ok")

    print("all checks passed\n")


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
    ap.add_argument("--max_sets", type=int, default=200,
                    help="top sets per month by frequency")
    ap.add_argument("--train_months", type=int, default=30,
                    help="months used to estimate beta; rest are held out")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    print(f"loaded {len(names)} months: {names[0]} .. {names[-1]}")
    print(f"fitting on top {args.max_sets} sets per month\n")

    # ---- STEP 1: fit beta on the training months, in-sample ---------------
    # This shows the in-sample result (which always wins) and the distribution
    # of beta, so we can see whether it is consistently positive.
    print("IN-SAMPLE (fits beta to each transition individually):")
    print(f"{'month':>8} {'beta':>8} {'mu':>8} "
          f"{'KL_model':>10} {'KL_cf':>10} {'winner':>14}")
    rows = []
    for i in range(len(names) - 1):
        m_t, m_n = names[i], names[i + 1]
        sets_t, p_t = top_sets(occ_by[m_t], args.max_sets)
        sets_n, p_n = top_sets(occ_by[m_n], args.max_sets)

        # restrict to sets present in both months for a fair comparison
        common = [c for c in sets_t if c in set(sets_n)]
        if len(common) < 10:
            print(f"  {m_t}: too few common sets ({len(common)}), skipping")
            continue

        idx_t = {c: j for j, c in enumerate(sets_t)}
        idx_n = {c: j for j, c in enumerate(sets_n)}
        p_t_r = np.array([p_t[idx_t[c]] for c in common])
        p_n_r = np.array([p_n[idx_n[c]] for c in common])
        p_t_r /= p_t_r.sum()
        p_n_r /= p_n_r.sum()

        beta, mu = fit(p_t_r, p_n_r, common)
        q_model = model_q(p_t_r, common, beta, mu)
        kl_model = kl(p_n_r, q_model)
        kl_cf = kl(p_n_r, p_t_r)

        rows.append({
            "month": m_t, "next": m_n,
            "n_common_sets": len(common),
            "beta": beta, "mu": mu,
            "kl_model": kl_model, "kl_copy_forward": kl_cf,
            "model_wins": int(kl_model < kl_cf),
            "kl_improvement": kl_cf - kl_model,
        })
        winner = "model" if kl_model < kl_cf else "copy-forward"
        print(f"  {m_t}: beta={beta:+.4f} mu={mu:.4f} "
              f"KL_model={kl_model:.6f} KL_cf={kl_cf:.6f} -> {winner}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/87_fit_results.csv", index=False)

    print("\n" + "=" * 72)
    print("IN-SAMPLE SUMMARY (always wins -- not the real test)")
    print("=" * 72)
    print(f"  model wins: {df['model_wins'].sum()} / {len(df)} -- expected, "
          f"two free parameters fitted to the outcome")
    print(f"  mean beta : {df['beta'].mean():.4f}  "
          f"std {df['beta'].std():.4f}  "
          f"share > 0: {(df['beta'] > 0).mean():.3f}")
    print(f"  NOTE: large mu values ({df['mu'].max():.1f} max) signal the")
    print(f"  optimiser is fitting noise, not a mutation rate")

    # ---- STEP 2: out-of-sample using a fixed beta -------------------------
    # Estimate mean beta from the training months only, then use that single
    # fixed value (with mu=0) to predict the held-out months.
    train_df = df[df.index < args.train_months]
    mean_beta = float(train_df["beta"].mean()) if len(train_df) else 0.0
    print(f"\n" + "=" * 72)
    print("OUT-OF-SAMPLE  (the real test)")
    print("=" * 72)
    print(f"  estimated beta from first {args.train_months} months: "
          f"{mean_beta:.4f}")
    print(f"  using this FIXED beta (mu=0) to predict held-out months")
    print(f"  copy-forward uses no parameters at all")
    print()
    oos_rows = []
    for i in range(args.train_months, len(names) - 1):
        m_t, m_n = names[i], names[i + 1]
        sets_t, p_t = top_sets(occ_by[m_t], args.max_sets)
        sets_n, p_n = top_sets(occ_by[m_n], args.max_sets)
        common = [c for c in sets_t if c in set(sets_n)]
        if len(common) < 10:
            continue
        idx_t = {c: j for j, c in enumerate(sets_t)}
        idx_n = {c: j for j, c in enumerate(sets_n)}
        p_t_r = np.array([p_t[idx_t[c]] for c in common])
        p_n_r = np.array([p_n[idx_n[c]] for c in common])
        p_t_r /= p_t_r.sum()
        p_n_r /= p_n_r.sum()
        # predict with fixed beta, mu=0
        q_fixed = model_q(p_t_r, common, mean_beta, 0.0)
        kl_fixed = kl(p_n_r, q_fixed)
        kl_cf = kl(p_n_r, p_t_r)
        # also try with beta=0 (pure copy-forward via model)
        oos_rows.append({
            "month": m_t, "next": m_n,
            "kl_fixed_beta": kl_fixed, "kl_copy_forward": kl_cf,
            "model_wins": int(kl_fixed < kl_cf),
        })
        winner = "model" if kl_fixed < kl_cf else "copy-forward"
        print(f"  {m_t}: KL_fixed={kl_fixed:.6f}  KL_cf={kl_cf:.6f}  "
              f"-> {winner}")
    if oos_rows:
        oos = pd.DataFrame(oos_rows)
        oos.to_csv(f"{args.out_dir}/87_oos_results.csv", index=False)
        print(f"\n  out-of-sample: model wins "
              f"{oos['model_wins'].sum()} / {len(oos)}")
        print(f"  mean KL fixed beta : {oos['kl_fixed_beta'].mean():.6f}")
        print(f"  mean KL copy-forward: {oos['kl_copy_forward'].mean():.6f}")
        print(f"  mean improvement   : "
              f"{(oos['kl_copy_forward'] - oos['kl_fixed_beta']).mean():.6f}")
        print()
        if oos["model_wins"].mean() > 0.6:
            print("  -> model beats copy-forward out-of-sample: selection is")
            print("     present and estimable; a parametric model improves")
            print("     on copy-forward; the negative result does NOT hold.")
        else:
            print("  -> copy-forward wins out-of-sample: the in-sample beta")
            print("     does not generalise. The data is consistent with a")
            print("     neutral process where copy-forward is the MLE.")
            print("     FORMAL ARGUMENT: the two-parameter model cannot")
            print("     improve on copy-forward out-of-sample, so copy-forward")
            print("     is the best predictor achievable from this data under")
            print("     this model class.")

    print("\nreading:")
    print("  beta near 0 and copy-forward wins most months ->")
    print("    data consistent with neutral process; copy-forward is the MLE;")
    print("    formal justification that no simple model beats it.")
    print("  beta clearly positive and model wins most months ->")
    print("    selection is present and estimable; a parametric model")
    print("    improves on copy-forward; the negative result does not hold.")
    print("  beta positive but model still loses ->")
    print("    selection exists but the two-parameter model is too simple")
    print("    to capture it; need richer fitness term.")

    print(f"\nwrote outputs/87_fit_results.csv")


if __name__ == "__main__":
    main()
