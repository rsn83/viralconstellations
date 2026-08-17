#!/usr/bin/env python
"""
61_weights_vs_components.py

Question
--------
In a mixture model of the population -- K components, each a distribution over
mutation cells, with month-t weights pi_t -- vocabulary change and weight change
are the same parameters moving. A mutation appears because a component carrying
it gained weight, or because the component itself drifted toward it.

Those two are very different in cost. Fixed components with moving weights is
K*V + K parameters per month. Drifting components is K*V per month, which
60 months of data cannot support.

So: CAN WEIGHT MOVEMENT ALONE PRODUCE THE OBSERVED VOCABULARY ENTRY?

Design: deliberately optimistic
------------------------------
Components are fitted on months <= t and then FROZEN. Weights are refitted on
month t+1 itself -- which peeks at the answer. That is on purpose. This is an
upper bound, not a forecast: it asks whether weight movement could produce the
observed entry EVEN IF the right weights were known. If it fails here it fails
everywhere, and the components must drift.

Three weight settings are compared to separate the questions:
  pi_frozen   weights held at month t          -- a real forecast
  pi_refit    weights refitted on t+1          -- the upper bound
  theta_refit components refitted on t+1 too   -- the other bound, showing what
                                                 drifting components would buy

Two kinds of entry, scored separately
-------------------------------------
  TYPE A  cell was seen in some month <= t but is absent from the support at t.
          The model has an estimated theta for it, so it can rank these.
          This is the dominant case and the one the test is about.
  TYPE B  cell has never been seen at all. With free per-cell parameters the
          model has no information about these and cannot rank them -- they are
          counted and reported, never scored. Ranking them would need the
          embedded parameterisation (beta_k = softmax(rho^T alpha_k)), which is
          exactly the argument for it.

What is observed : constellations and counts per month
What is hidden   : component membership of each genome
What is predicted: which absent-but-known cells re-enter the support at t+1
What is NOT claimed: that components correspond to real lineages
Why useful       : it settles the parameterisation before anything is built on
                   it -- fixed components is a far smaller model, and this says
                   whether it is sufficient

Baselines
---------
  historical_freq  the cell's mean frequency over months <= t. Simple, strong,
                   and the thing a mixture must beat to justify its parameters.
  recency          1 / (1 + months since last seen). Best single feature from
                   script 56.
  uniform          reference.

Outputs
-------
outputs/61_entry_by_month.csv   per origin, per model, AP and counts
outputs/61_summary.csv          pooled comparison across K
outputs/61_counts.csv           predicted vs observed number of entries

Usage
-----
python scripts/61_weights_vs_components.py --min_count 3 --end_month 2024-12
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


def rarefy(occ, depth, min_count, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    total = counts.sum()
    if total < depth:
        return None
    draws = rng.multinomial(depth, counts / total)
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws >= min_count)[0]}


def support_of(occ, depth, min_count, reps, rng):
    """Depth-controlled support, so months of different effort are comparable."""
    seen, n = defaultdict(int), 0
    for _ in range(reps):
        sub = rarefy(occ, depth, min_count, rng)
        if sub is None:
            continue
        n += 1
        for cs in sub:
            for lab in cs:
                seen[lab] += 1
    if n == 0:
        out = set()
        for cs in occ:
            out |= set(cs)
        return out
    return {lab for lab, c in seen.items() if c >= n / 2}


# ----------------------------------------------------------------------------
# mixture of Bernoullis, EM on count-weighted set data
# ----------------------------------------------------------------------------

def e_step(rows, w, theta, pi):
    """
    rows : list of index arrays (the cells present in each constellation)
    theta: K x V
    Returns responsibilities R (N x K) and the weighted log-likelihood.
    """
    K, V = theta.shape
    th = np.clip(theta, 1e-6, 1 - 1e-6)
    base = np.log(1 - th).sum(axis=1)                 # K
    delta = np.log(th) - np.log(1 - th)               # K x V
    logp = np.empty((len(rows), K))
    for i, idx in enumerate(rows):
        logp[i] = base + (delta[:, idx].sum(axis=1) if idx.size else 0.0)
    logp += np.log(np.clip(pi, 1e-12, None))[None, :]
    mx = logp.max(axis=1, keepdims=True)
    ex = np.exp(logp - mx)
    s = ex.sum(axis=1, keepdims=True)
    R = ex / s
    ll = float((w * (mx[:, 0] + np.log(s[:, 0]))).sum())
    return R, ll


def m_step(rows, w, R, V, a=0.5, b=0.5, update_theta=True, theta=None):
    K = R.shape[1]
    Nk = (R * w[:, None]).sum(axis=0)                 # K
    pi = (Nk + 1e-6) / (Nk.sum() + K * 1e-6)
    if not update_theta:
        return theta, pi
    S = np.zeros((K, V))
    for i, idx in enumerate(rows):
        if idx.size:
            S[:, idx] += (R[i] * w[i])[:, None]
    theta_new = (S + a) / (Nk[:, None] + a + b)
    return np.clip(theta_new, 1e-6, 1 - 1e-6), pi


def fit_mixture(rows, w, V, K, rng, n_iter=60, tol=1e-4,
                theta_init=None, pi_init=None, update_theta=True):
    if theta_init is None:
        theta = rng.uniform(0.05, 0.95, size=(K, V))
    else:
        theta = theta_init.copy()
    pi = np.full(K, 1.0 / K) if pi_init is None else pi_init.copy()
    prev = -np.inf
    for _ in range(n_iter):
        R, ll = e_step(rows, w, theta, pi)
        theta, pi = m_step(rows, w, R, V, update_theta=update_theta, theta=theta)
        if abs(ll - prev) < tol * max(1.0, abs(prev)):
            break
        prev = ll
    return theta, pi, prev


def encode(occ, cell_index):
    rows, w = [], []
    for cs, c in occ.items():
        idx = np.fromiter((cell_index[l] for l in cs if l in cell_index),
                          dtype=int)
        rows.append(np.unique(idx))
        w.append(float(c))
    return rows, np.array(w)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -np.asarray(s, dtype=float)))
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, y.size + 1)
    return float((prec * yy).sum() / y.sum())


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
    ap.add_argument("--Ks", default="5,10,20,40")
    ap.add_argument("--min_train", type=int, default=18)
    ap.add_argument("--fit_window", type=int, default=12,
                    help="months pooled to fit the components")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    Ks = [int(k) for k in args.Ks.split(",")]

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    occ_by = {m: o for m, o in months}
    names = [m for m, _ in months]
    print(f"loaded {len(names)} months: {names[0]} .. {names[-1]}")

    print("computing depth-controlled supports ...")
    support = {m: support_of(occ_by[m], args.depth, args.min_count,
                             args.reps, rng) for m in names}
    T = len(names)
    eps = args.min_count / args.depth

    # causal cell history
    seen_by = []
    seen = set()
    last_seen = {}
    freq_hist = defaultdict(list)
    for j, m in enumerate(names):
        tot = float(sum(occ_by[m].values()))
        nc = defaultdict(float)
        for cs, c in occ_by[m].items():
            for lab in cs:
                nc[lab] += c
        for lab in support[m]:
            last_seen[lab] = j
        for lab, v in nc.items():
            freq_hist[lab].append(v / tot)
        seen |= support[m]
        seen_by.append(frozenset(seen))

    rows_out, count_out = [], []

    for t in range(args.min_train, T - 1):
        universe = sorted(seen_by[t], key=str)
        cell_index = {l: i for i, l in enumerate(universe)}
        V = len(universe)
        supp_t, supp_n = support[names[t]], support[names[t + 1]]

        # TYPE A candidates: known to the model, absent right now
        cand = [l for l in universe if l not in supp_t]
        if not cand:
            continue
        y = np.array([1 if l in supp_n else 0 for l in cand], dtype=int)
        # TYPE B: never seen at all -- counted, never scored
        type_b = len(supp_n - seen_by[t])
        if y.sum() == 0:
            continue
        cidx = np.array([cell_index[l] for l in cand])
        base = float(y.mean())

        # training data: months in the fit window, pooled
        rows, w = [], []
        for j in range(max(0, t - args.fit_window + 1), t + 1):
            r, ww = encode(occ_by[names[j]], cell_index)
            rows.extend(r)
            w.append(ww)
        w = np.concatenate(w)
        rows_next, w_next = encode(occ_by[names[t + 1]], cell_index)

        # simple baselines, no mixture involved
        hist = np.array([np.mean(freq_hist[l][-args.fit_window:])
                         if freq_hist[l] else 0.0 for l in cand])
        rec = np.array([1.0 / (1.0 + (t - last_seen.get(l, -99))) for l in cand])
        base_scores = {
            "uniform": rng.random(len(cand)),
            "historical_freq": hist,
            "recency": rec,
        }
        for nm, s in base_scores.items():
            rows_out.append({
                "origin": names[t], "target": names[t + 1], "K": 0,
                "model": nm, "ap": average_precision(y, s),
                "base_rate": base, "n_cand": len(cand), "n_pos": int(y.sum()),
                "type_b_unscorable": type_b,
            })

        for K in Ks:
            theta, pi, _ = fit_mixture(rows, w, V, K, rng)

            # pi frozen at t -- a real forecast
            marg_frozen = pi @ theta

            # pi refitted on t+1, components frozen -- the upper bound
            _, pi_refit, _ = fit_mixture(rows_next, w_next, V, K, rng,
                                         theta_init=theta, pi_init=pi,
                                         update_theta=False)
            marg_refit = pi_refit @ theta

            # components refitted too -- the other bound
            theta2, pi2, _ = fit_mixture(rows_next, w_next, V, K, rng,
                                         theta_init=theta, pi_init=pi,
                                         update_theta=True)
            marg_theta = pi2 @ theta2

            for nm, s in [("pi_frozen", marg_frozen),
                          ("pi_refit", marg_refit),
                          ("theta_refit", marg_theta)]:
                rows_out.append({
                    "origin": names[t], "target": names[t + 1], "K": K,
                    "model": nm, "ap": average_precision(y, s[cidx]),
                    "base_rate": base, "n_cand": len(cand),
                    "n_pos": int(y.sum()), "type_b_unscorable": type_b,
                })
                count_out.append({
                    "origin": names[t], "K": K, "model": nm,
                    "pred_entries": int((s[cidx] > eps).sum()),
                    "obs_entries": int(y.sum()),
                    "pred_support_size": int((s > eps).sum()),
                    "obs_support_size": len(supp_n),
                })

        print(f"  {names[t]} -> {names[t+1]}: V={V}, {len(cand)} candidates, "
              f"{int(y.sum())} entries (base {base:.4f}), "
              f"{type_b} never-seen (unscorable)")

    df = pd.DataFrame(rows_out)
    df.to_csv(f"{args.out_dir}/61_entry_by_month.csv", index=False)
    cdf = pd.DataFrame(count_out)
    cdf.to_csv(f"{args.out_dir}/61_counts.csv", index=False)

    print("\n" + "=" * 74)
    print("TYPE A ENTRY: cells known to the model, absent at t, present at t+1")
    print("=" * 74)
    summ = df.groupby(["model", "K"]).agg(
        ap=("ap", "mean"), base=("base_rate", "mean"),
        n_cand=("n_cand", "mean"), n_pos=("n_pos", "mean"),
        origins=("ap", "count"),
    ).reset_index()
    summ["lift_pooled"] = summ["ap"] / summ["base"]
    summ = summ.sort_values("ap", ascending=False)
    summ.to_csv(f"{args.out_dir}/61_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))

    print("\n--- predicted vs observed entry counts ---")
    if len(cdf):
        g = cdf.groupby(["model", "K"]).agg(
            pred_entries=("pred_entries", "mean"),
            obs_entries=("obs_entries", "mean"),
            pred_support=("pred_support_size", "mean"),
            obs_support=("obs_support_size", "mean"),
        ).reset_index()
        print(g.round(2).to_string(index=False))

    tb = df["type_b_unscorable"].mean()
    print(f"\nnever-seen cells per month (TYPE B, unscorable): {tb:.1f}")
    print("  a free per-cell mixture has no information about these. Ranking")
    print("  them requires the embedded parameterisation.")

    print("\nread:")
    print("  pi_refit near theta_refit -> weight movement is SUFFICIENT. Freeze")
    print("     the components; the model is K*V parameters and 60 months can")
    print("     support it.")
    print("  theta_refit much better than pi_refit -> the components themselves")
    print("     drift, and fixed-component mixtures cannot represent vocabulary")
    print("     entry however the weights move.")
    print("  pi_refit no better than historical_freq -> the mixture is not")
    print("     earning its parameters at all, whatever the answer above.")
    print("  NOTE pi_refit and theta_refit both use month t+1 to set weights.")
    print("  They are upper bounds, not forecasts. Only pi_frozen is a forecast.")
    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
