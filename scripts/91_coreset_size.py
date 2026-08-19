#!/usr/bin/env python
"""
91_coreset_size.py

Does a coreset exist for the monthly mutation set distributions?

A coreset of size K exists if the monthly distribution can be approximated
by K weighted representative mutation sets without losing much information.

This script fits a mixture of K Bernoulli product distributions to each
month's sequences for K = 1, 2, 5, 10, 20, 30, 50 and measures:

  1. Reconstruction log-likelihood as a function of K -- the elbow plot.
     If there is a sharp elbow at small K, a coreset of that size exists.

  2. Coreset stability across consecutive months -- the fraction of next
     month's weight explained by this month's K representatives.
     If stability is high, the same coreset recurs across months.

  3. Whether instability spikes at known variant transitions.

The mixture model is fitted by EM. Each component is a Bernoulli product
over the 1180-mutation vocabulary -- the probability of each mutation
being present is fitted independently per component.

Usage
-----
python scripts/91_coreset_size.py --min_count 3 --end_month 2024-12
python scripts/91_coreset_size.py --self_test
"""

import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

TRANSITIONS = {
    "2021-01": "Alpha", "2021-06": "Delta", "2021-12": "Omicron",
    "2022-03": "BA2", "2022-06": "BA5", "2023-02": "XBB", "2023-12": "JN1",
}


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


def sample_matrix(occ, n_target, rng):
    """Draw n_target sequences, return binary matrix (n_target x V)."""
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    if counts.sum() < n_target:
        n_target = int(counts.sum())
    draws = rng.multinomial(n_target, counts / counts.sum())
    # collect all labels to build vocabulary
    all_labels = sorted({l for k in keys for l in k})
    lab_idx = {l: i for i, l in enumerate(all_labels)}
    V = len(all_labels)
    X = np.zeros((n_target, V), dtype=np.float32)
    r = 0
    for k, d in zip(keys, draws):
        if d == 0:
            continue
        cols = [lab_idx[l] for l in k]
        X[r:r+d, cols] = 1.0
        r += d
    return X, all_labels


# ----------------------------------------------------------------------------
# EM for mixture of Bernoulli products
# ----------------------------------------------------------------------------

def fit_mixture(X, K, n_iter=30, tol=1e-4, rng=None):
    """
    Fit a mixture of K Bernoulli product distributions to binary matrix X
    using EM. Returns (pi, theta, ll) where:
      pi: (K,) component weights
      theta: (K, V) component Bernoulli parameters
      ll: final log-likelihood per sequence
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, V = X.shape
    if K == 1:
        theta = X.mean(axis=0, keepdims=True) + 1e-6
        theta = np.clip(theta, 1e-6, 1-1e-6)
        pi = np.array([1.0])
        ll = (X * np.log(theta) + (1-X) * np.log(1-theta)).sum(axis=1).mean()
        return pi, theta, ll

    # initialise with random assignment
    pi = np.ones(K) / K
    # initialise theta by randomly selecting K sequences as centers
    idx = rng.choice(n, K, replace=False)
    theta = np.clip(X[idx] + rng.normal(0, 0.1, (K, V)), 1e-6, 1-1e-6)

    ll_prev = -np.inf
    for it in range(n_iter):
        # E step: compute responsibilities
        log_p = X @ np.log(theta.T) + (1-X) @ np.log(1-theta.T)  # (n, K)
        log_p += np.log(pi)[None, :]
        log_p -= log_p.max(axis=1, keepdims=True)
        r = np.exp(log_p)
        r /= r.sum(axis=1, keepdims=True)

        # M step
        rsum = r.sum(axis=0) + 1e-12  # (K,)
        pi = rsum / rsum.sum()
        theta = (r.T @ X) / rsum[:, None]
        theta = np.clip(theta, 1e-6, 1-1e-6)

        # log-likelihood
        log_p2 = X @ np.log(theta.T) + (1-X) @ np.log(1-theta.T)
        log_p2 += np.log(pi)[None, :]
        ll = float(np.log(np.exp(log_p2).sum(axis=1)).mean())
        if abs(ll - ll_prev) < tol:
            break
        ll_prev = ll

    return pi, theta, ll


def coreset_stability(occ_t, occ_n, theta_t, pi_t, all_labels_t):
    """
    What fraction of next month's sequence weight is explained by
    this month's K representative sets (theta_t)?

    For each sequence in next month, assign it to its most likely
    component from this month's mixture. The stability is the
    fraction of weight assigned to components with weight > 0.01.
    """
    # build binary matrix for next month using same vocabulary
    lab_idx = {l: i for i, l in enumerate(all_labels_t)}
    keys_n = list(occ_n.keys())
    counts_n = np.array([occ_n[k] for k in keys_n], dtype=float)
    tot_n = counts_n.sum()

    # for each set in next month, compute log likelihood under this month's mixture
    explained = 0.0
    for k, w in zip(keys_n, counts_n):
        # binary vector for this set
        x = np.zeros(len(all_labels_t), dtype=np.float32)
        for l in k:
            if l in lab_idx:
                x[lab_idx[l]] = 1.0
        # log likelihood under each component
        log_p = x @ np.log(theta_t.T) + (1-x) @ np.log(1-theta_t.T)
        log_p += np.log(pi_t)
        best = log_p.max()
        # if best component explains it reasonably, count as explained
        # threshold: within 10 nats of a perfect fit
        if best > -len(k) * 2:  # rough threshold
            explained += w
    return float(explained / tot_n) if tot_n > 0 else np.nan


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    print("self-test")
    rng = np.random.default_rng(0)

    # two clearly separated clusters: K=2 should fit much better than K=1
    X1 = np.zeros((200, 10), dtype=np.float32)
    X1[:100, :5] = 1.0   # cluster 1: mutations 0-4
    X1[100:, 5:] = 1.0   # cluster 2: mutations 5-9
    X1 += rng.normal(0, 0.05, X1.shape)
    X1 = np.clip(X1, 0, 1)

    _, _, ll1 = fit_mixture(X1, 1, rng=rng)
    _, _, ll2 = fit_mixture(X1, 2, rng=rng)
    assert ll2 > ll1 + 0.5, (ll1, ll2)
    print(f"  K=2 fits two-cluster data better than K=1 "
          f"({ll2:.2f} > {ll1:.2f})     ok")

    # one cluster: K=1 and K=2 should be similar
    X2 = rng.binomial(1, 0.3, (200, 10)).astype(np.float32)
    _, _, ll1b = fit_mixture(X2, 1, rng=rng)
    _, _, ll2b = fit_mixture(X2, 2, rng=rng)
    assert ll2b < ll1b + 2.0, (ll1b, ll2b)
    print(f"  K=1 and K=2 similar for one-cluster data "
          f"({ll1b:.2f} vs {ll2b:.2f})  ok")

    print("all checks passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--n_per_month", type=int, default=2000)
    ap.add_argument("--k_values", default="1,2,5,10,20,30,50")
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    K_vals = [int(k) for k in args.k_values.split(",")]

    months = load_months(args.data_dir, args.min_count,
                         end_month=args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    print(f"loaded {len(names)} months: {names[0]} .. {names[-1]}")
    print(f"K values: {K_vals}\n")

    # fit mixtures and compute elbow
    elbow_rows = []
    stability_rows = []
    prev_theta, prev_pi, prev_labels = None, None, None

    for i, m in enumerate(names):
        X, labels = sample_matrix(occ_by[m], args.n_per_month, rng)
        print(f"  {m}: {X.shape[0]} sequences, {X.shape[1]} mutations")

        lls = {}
        best_theta, best_pi = None, None
        for K in K_vals:
            if K > X.shape[0]:
                continue
            pi, theta, ll = fit_mixture(X, K, args.n_iter, rng=rng)
            lls[K] = ll
            if K == K_vals[min(3, len(K_vals)-1)]:  # save K=10 or similar
                best_theta, best_pi = theta, pi

        elbow_rows.append({
            "month": m,
            "variant": TRANSITIONS.get(m, ""),
            **{f"ll_K{k}": lls.get(k, np.nan) for k in K_vals}
        })

        # stability: how well does previous month's mixture explain this month
        if prev_theta is not None and prev_labels is not None:
            stab = coreset_stability(occ_by[names[i-1]], occ_by[m],
                                     prev_theta, prev_pi, prev_labels)
            stability_rows.append({
                "month": names[i-1], "next": m,
                "stability": stab,
                "is_transition": m in TRANSITIONS,
                "variant": TRANSITIONS.get(m, ""),
            })

        prev_theta = best_theta
        prev_pi = best_pi
        prev_labels = labels

    df_elbow = pd.DataFrame(elbow_rows)
    df_stab = pd.DataFrame(stability_rows)
    df_elbow.to_csv(f"{args.out_dir}/91_elbow.csv", index=False)
    df_stab.to_csv(f"{args.out_dir}/91_stability.csv", index=False)

    print("\n" + "=" * 80)
    print("ELBOW PLOT: LOG-LIKELIHOOD PER SEQUENCE BY K (every 6th month)")
    print("=" * 80)
    sel = df_elbow.iloc[::6]
    ll_cols = [f"ll_K{k}" for k in K_vals]
    print(sel[["month", "variant"] + ll_cols].round(3).to_string(index=False))

    print("\n" + "=" * 80)
    print("GAIN FROM ADDING COMPONENTS (averaged over all months)")
    print("=" * 80)
    for j in range(1, len(K_vals)):
        k_prev, k_curr = K_vals[j-1], K_vals[j]
        col_prev = f"ll_K{k_prev}"
        col_curr = f"ll_K{k_curr}"
        if col_prev in df_elbow and col_curr in df_elbow:
            gain = (df_elbow[col_curr] - df_elbow[col_prev]).mean()
            print(f"  K={k_prev} -> K={k_curr}: mean gain {gain:+.4f} bits/seq")

    print("\n" + "=" * 80)
    print("CORESET STABILITY ACROSS MONTHS")
    print("=" * 80)
    print(df_stab.round(3).to_string(index=False))
    print(f"\n  mean stability: {df_stab['stability'].mean():.3f}")
    trans = df_stab[df_stab["is_transition"]]
    non = df_stab[~df_stab["is_transition"]]
    if len(trans):
        print(f"  at transitions: {trans['stability'].mean():.3f}")
    if len(non):
        print(f"  non-transitions: {non['stability'].mean():.3f}")

    print("\nreading:")
    print("  if ll gains sharply from K=1 to K=10 then flattens: a coreset")
    print("  of size ~10-20 approximates the monthly distribution well.")
    print("  if stability is high (>0.85) and drops at transitions: the")
    print("  coreset is stable within variant eras and changes at switches.")
    print("  this would confirm that K stable cores exist and are trackable.")

    print(f"\nwrote 2 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
