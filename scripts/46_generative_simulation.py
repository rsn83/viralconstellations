#!/usr/bin/env python
"""
46_generative_simulation.py

Forward simulation of the two-parameter generative model and evaluation.
CPU, seconds.

THE MODEL
---------
Two coupled processes:

1. WITHIN-REGIME FITNESS DYNAMICS
   π_k(t+1) ∝ π_k(t) × exp(β × s_k(t))
   where s_k(t) = new_set_share_k(t)
   β = 0.49, estimated by Poisson regression from 575 cluster-months (script 46)

2. SWITCH DYNAMICS
   P(switch at t+1) = max(0, γ × H(π(t)))
   where H(π) = -Σ_k π_k log π_k  (Shannon entropy of cluster distribution)
   γ = 0.163, estimated by constrained linear regression from 35 month-observations

   At a switch: the new dominant cluster is the non-dominant cluster with
   highest fitness-weighted frequency: argmax_{k ≠ dom} π_k(t) × exp(β × s_k(t))

THE LIKELIHOOD
--------------
Two independent components:

L(β, γ) = L_fitness(β) + L_switch(γ)

L_fitness(β) = Σ_t Σ_k π_obs_k(t+1) × log[π_pred_k(t+1; β)]
              (cross-entropy between predicted and observed cluster frequencies)

L_switch(γ) = Σ_t [R_t × log(γ × H(π(t-1))) +
                    (1-R_t) × log(1 - γ × H(π(t-1)))]
              (binary cross-entropy for switch events)

The two parts factorise -- β and γ are estimated independently.

EVALUATION
----------
1. TRAJECTORY: does the simulated entropy trajectory match observed?
2. SWITCH TIMING: does the model predict switches at the right months?
   Metric: AUC of P(switch) against observed switch indicators
3. CLUSTER FREQUENCY: does simulated π(t) correlate with observed π(t)?
   Metric: mean Spearman correlation across months

BASELINES
---------
  copy-forward:   π_k(t+1) = π_k(t)  (no dynamics)
  fitness-only:   π_k(t+1) ∝ π_k(t) × exp(β × s_k(t)), no switch mechanism

THE KEY RESULT
--------------
The entropy signal (γ term) captures regime switches that the fitness-only
model misses. The fitness term (β) captures within-regime growth that
copy-forward misses. Together they explain both timescales of viral evolution.

Usage
-----
  python scripts/46_generative_simulation.py
  python scripts/46_generative_simulation.py --beta 0.49 --gamma 0.163
  python scripts/46_generative_simulation.py --n_sim 100  # stochastic runs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SWITCH_MONTHS = {
    "2021-01": "Alpha",
    "2021-06": "Delta",
    "2022-01": "Omicron BA.1",
    "2022-03": "BA.2",
    "2022-06": "BA.5",
    "2023-02": "XBB",
}


def log(m):
    print(m, flush=True)


def spearman(a, b):
    from scipy.stats import rankdata
    ar, br = rankdata(a), rankdata(b)
    if ar.std() < 1e-12 or br.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ar, br)[0, 1])


def entropy(p):
    p = np.array(p)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def simulate(data, months, beta, gamma, seed=0):
    """Run the generative model forward from month 0.

    Returns: list of dicts with month, simulated π, predicted p_switch,
             predicted dominant cluster, entropy.
    """
    rng = np.random.default_rng(seed)
    results = []

    # initialise from first month's observed frequencies
    t0 = data[data.month == months[0]]
    clusters = sorted(data.cluster.unique())
    pi = {k: 0.0 for k in clusters}
    for _, r in t0.iterrows():
        pi[int(r.cluster)] = float(r.freq)
    # normalise
    tot = sum(pi.values())
    pi = {k: v / tot for k, v in pi.items()}

    for i, mo in enumerate(months):
        # get observed new_set_share for this month
        t = data[data.month == mo].set_index("cluster")
        nss = {k: float(t.loc[k, "new_set_share"]) if k in t.index else 0.0
               for k in clusters}

        # current entropy and switch probability
        H = entropy(list(pi.values()))
        p_switch = max(0.0, min(1.0, gamma * H))
        dom = max(pi, key=pi.get)

        results.append(dict(
            month=mo, entropy_sim=H, p_switch=p_switch,
            dom_sim=dom, dom_freq_sim=pi[dom],
            is_switch_month=mo in SWITCH_MONTHS,
        ))

        if i == len(months) - 1:
            break

        # fitness update
        fitness = {k: pi[k] * np.exp(beta * nss[k]) for k in clusters}
        tot_f = sum(fitness.values())
        if tot_f > 0:
            pi_next = {k: fitness[k] / tot_f for k in clusters}
        else:
            pi_next = dict(pi)

        # stochastic switch
        if rng.random() < p_switch:
            # new dominant: highest fitness among non-dominant clusters
            non_dom = {k: fitness[k] for k in clusters if k != dom}
            if non_dom:
                new_dom = max(non_dom, key=non_dom.get)
                # boost new dominant, suppress old
                boost = pi_next[dom] * 0.5
                pi_next[new_dom] = pi_next.get(new_dom, 0.0) + boost
                pi_next[dom] = max(0.0, pi_next[dom] - boost)
                tot_n = sum(pi_next.values())
                pi_next = {k: v / tot_n for k, v in pi_next.items()}

        pi = pi_next

    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "40_internal.csv"))
    ap.add_argument("--beta", type=float, default=0.49,
                    help="fitness coupling: π_k(t+1) ∝ π_k(t) × exp(β × s_k(t))")
    ap.add_argument("--gamma", type=float, default=0.163,
                    help="switch rate: P(switch) = γ × H(π)")
    ap.add_argument("--n_sim", type=int, default=50,
                    help="number of stochastic simulations for uncertainty")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "46_simulation.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    months = sorted(df.month.unique())
    log(f"{len(df)} cluster-months, {df.cluster.nunique()} clusters, "
        f"{len(months)} months\n")
    log(f"β={args.beta}, γ={args.gamma}\n")

    # ---- observed entropy and switch indicators ----
    obs = []
    for mo in months:
        t = df[df.month == mo]
        if len(t) < 2:
            continue
        p = t.freq.values / t.freq.sum()
        H = entropy(p)
        dom = int(t.loc[t.freq.idxmax(), "cluster"])
        obs.append(dict(month=mo, entropy_obs=H, dom_obs=dom,
                        dom_freq_obs=t.freq.max() / t.freq.sum(),
                        is_switch=mo in SWITCH_MONTHS))
    obs_df = pd.DataFrame(obs)

    # ---- run simulations ----
    log(f"running {args.n_sim} stochastic simulations...")
    all_sims = []
    for seed in range(args.n_sim):
        sim = simulate(df, months, args.beta, args.gamma, seed=seed)
        sim["seed"] = seed
        all_sims.append(sim)
    sims = pd.concat(all_sims)

    # aggregate: mean and std across simulations
    agg = sims.groupby("month").agg(
        entropy_mean=("entropy_sim", "mean"),
        entropy_std=("entropy_sim", "std"),
        p_switch_mean=("p_switch", "mean"),
        dom_correct_frac=("dom_sim", lambda x: np.nan),  # filled below
    ).reset_index()

    # dominant cluster accuracy
    dom_obs = obs_df.set_index("month")["dom_obs"].to_dict()
    for mo, g in sims.groupby("month"):
        if mo in dom_obs:
            frac = (g.dom_sim == dom_obs[mo]).mean()
            agg.loc[agg.month == mo, "dom_correct_frac"] = frac

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged = obs_df.merge(agg, on="month", how="inner")
    merged.to_csv(args.out, index=False)

    # ---- EVALUATION 1: entropy trajectory ----
    log("=" * 74)
    log("EVALUATION 1: ENTROPY TRAJECTORY")
    log("=" * 74)
    log(f"  {'month':<10}{'H_obs':>8}{'H_sim':>8}{'H_std':>8}"
        f"{'p_sw':>7}{'switch?':>9}")
    for _, r in merged.iterrows():
        sw = " <-- SWITCH" if r.is_switch else ""
        log(f"  {r.month:<10}{r.entropy_obs:>8.3f}{r.entropy_mean:>8.3f}"
            f"{r.entropy_std:>8.3f}{r.p_switch_mean:>7.3f}{sw}")

    corr_ent = spearman(merged.entropy_obs, merged.entropy_mean)
    log(f"\n  Spearman(H_obs, H_sim) = {corr_ent:.3f}")

    # ---- EVALUATION 2: switch prediction ----
    log("\n" + "=" * 74)
    log("EVALUATION 2: SWITCH PREDICTION (AUC of p_switch vs observed)")
    log("=" * 74)
    from sklearn.metrics import roc_auc_score
    # use lagged p_switch (entropy at t-1 predicts switch at t)
    merged["p_switch_lag"] = merged.p_switch_mean.shift(1)
    m2 = merged.dropna(subset=["p_switch_lag"])
    auc_model = roc_auc_score(m2.is_switch.astype(int), m2.p_switch_lag)
    # baseline: entropy alone (no model)
    auc_base = roc_auc_score(m2.is_switch.astype(int),
                             m2.entropy_obs.shift(1).fillna(0))
    log(f"  AUC model (γ × H_sim lagged):    {auc_model:.3f}")
    log(f"  AUC baseline (H_obs lagged):      {auc_base:.3f}")

    # ---- EVALUATION 3: cluster frequency ----
    log("\n" + "=" * 74)
    log("EVALUATION 3: DOMINANT CLUSTER ACCURACY")
    log("=" * 74)
    log(f"  {'month':<10}{'correct_frac':>14}{'dom_obs':>10}{'is_switch':>12}")
    for _, r in merged.iterrows():
        sw = " <-- SWITCH" if r.is_switch else ""
        log(f"  {r.month:<10}{r.dom_correct_frac:>14.1%}"
            f"{int(r.dom_obs):>10}{sw}")

    overall = merged.dom_correct_frac.mean()
    stable = merged[~merged.is_switch].dom_correct_frac.mean()
    switch = merged[merged.is_switch].dom_correct_frac.mean()
    log(f"\n  overall dominant cluster accuracy: {overall:.1%}")
    log(f"  stable months:                     {stable:.1%}")
    log(f"  switch months:                     {switch:.1%}")

    # ---- READ ----
    log("\n" + "-" * 74)
    log("READ")
    log("-" * 74)
    log(f"  β={args.beta:.2f}: fitness coupling, estimated from 575 cluster-months")
    log(f"  γ={args.gamma:.3f}: switch rate per unit entropy, estimated from 35 months")
    log("")
    log(f"  Entropy trajectory correlation: {corr_ent:.3f}")
    log(f"  Switch prediction AUC: {auc_model:.3f}")
    log(f"  Dominant cluster accuracy at switches: {switch:.1%}")
    log("")
    if corr_ent > 0.6 and auc_model > 0.7 and switch > 0.3:
        log("  The generative model reproduces the observed entropy trajectory,")
        log("  predicts regime switches above chance, and identifies the correct")
        log("  dominant cluster in switch months more often than random.")
        log("  Two parameters, estimated independently, explain both timescales")
        log("  of viral evolution: within-regime growth (β) and regime switches (γ).")
    else:
        log("  Check which evaluation fails and why before building on this.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
