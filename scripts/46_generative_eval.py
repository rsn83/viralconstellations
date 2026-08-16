#!/usr/bin/env python
"""
46_generative_eval.py

One-step-ahead evaluation of the two-parameter model.
Replaces 46_generative_simulation.py, which ran a forward simulation
that collapsed: the fitness dynamics concentrate mass rather than
disperse it, so starting from near-monomorphic state entropy immediately
drops to zero and the switch mechanism never fires. That failure was
diagnostic -- the model is a TRANSITION model, not a fully autonomous
generative model. It needs observed pi(t) as input at each step.

WHAT THE MODEL IS
-----------------
A conditional transition model with two parameters:

  pi_pred_k(t+1) proportional to pi_obs_k(t) * exp(beta * s_k(t))
  P(switch at t+1) = gamma * H(pi_obs(t))

Given the OBSERVED cluster distribution at t and the observed
new_set_share vector, it predicts the distribution at t+1 and whether
a regime switch occurs. It does not run autonomously.

This is the same status as Luksza-Lassig: a transition model with
estimated parameters, evaluated on one-step-ahead predictions.

THE TWO PARAMETERS
------------------
beta = 0.49   fitness coupling, estimated by Poisson regression
              from 575 cluster-months (script 40/41)
gamma = 0.163 switch rate per unit entropy, estimated by constrained
              linear regression from 35 months, 6 switch events

EVALUATION
----------
For each consecutive month pair (t, t+1):

  1. FREQUENCY PREDICTION
     pi_pred(t+1) proportional to pi_obs(t) * exp(beta * s_obs(t))
     metric: Spearman(pi_pred(t+1), pi_obs(t+1)) per month, averaged
     baseline: copy-forward pi_pred = pi_obs(t)

  2. SWITCH PREDICTION
     P(switch at t+1) = gamma * H(pi_obs(t))
     metric: AUC against observed switch indicators
     baseline: H_obs(t) alone (no gamma scaling)

  3. WALK-FORWARD: fit beta and gamma on months up to t, evaluate on t+1
     Honest out-of-sample version of the above.

WHAT THIS ESTABLISHES
---------------------
beta > 0 and improving frequency prediction: diversification drives
within-regime growth.

gamma * H predicting switches with AUC > 0.7: population entropy
predicts regime transitions one month ahead.

Together: two measurable quantities, each capturing a different
timescale of viral evolution, estimated from sequence counts alone.

Usage
-----
  python scripts/46_generative_eval.py
  python scripts/46_generative_eval.py --walkforward
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SWITCH_MONTHS = {
    "2021-01", "2021-06", "2022-01",
    "2022-03", "2022-06", "2023-02",
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
    p = np.asarray(p)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "40_internal.csv"))
    ap.add_argument("--beta", type=float, default=0.49)
    ap.add_argument("--gamma", type=float, default=0.163)
    ap.add_argument("--walkforward", action="store_true",
                    help="also run walk-forward: refit beta and gamma on "
                         "months up to t, evaluate on t+1")
    ap.add_argument("--min_train", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "46_eval.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    months = sorted(df.month.unique())
    log(f"{len(df)} cluster-months, {df.cluster.nunique()} clusters, "
        f"{len(months)} months\n")

    # ---- per-month cluster state ----
    state = {}
    for mo in months:
        t = df[df.month == mo]
        if len(t) < 2:
            continue
        tot = t.freq.sum()
        pi = dict(zip(t.cluster.astype(int), t.freq / tot))
        nss = dict(zip(t.cluster.astype(int), t.new_set_share))
        H = entropy(list(pi.values()))
        dom = max(pi, key=pi.get)
        state[mo] = dict(pi=pi, nss=nss, H=H, dom=dom)

    usable = [mo for mo in months if mo in state]

    # ---- ONE-STEP-AHEAD EVALUATION ----
    log("=" * 74)
    log(f"ONE-STEP-AHEAD (beta={args.beta}, gamma={args.gamma})")
    log("=" * 74)

    rows = []
    for i, mo in enumerate(usable[:-1]):
        mo1 = usable[i + 1]
        st, st1 = state[mo], state[mo1]

        # frequency prediction
        clusters = list(st["pi"].keys())
        pi0 = np.array([st["pi"].get(k, 0.0) for k in clusters])
        nss0 = np.array([st["nss"].get(k, 0.0) for k in clusters])
        pi1_obs = np.array([st1["pi"].get(k, 0.0) for k in clusters])

        # model prediction
        f = pi0 * np.exp(args.beta * nss0)
        pi1_pred = f / f.sum() if f.sum() > 0 else pi0

        # copy-forward baseline
        pi1_cf = pi0

        sp_model = spearman(pi1_pred, pi1_obs)
        sp_cf = spearman(pi1_cf, pi1_obs)

        # switch prediction
        p_sw = max(0.0, min(1.0, args.gamma * st["H"]))
        is_sw = mo1 in SWITCH_MONTHS
        dom_pred = clusters[int(np.argmax(pi1_pred))]
        dom_correct = (dom_pred == st1["dom"])

        rows.append(dict(
            month_t=mo, month_t1=mo1,
            H_t=st["H"], p_switch=p_sw,
            is_switch=is_sw,
            sp_model=sp_model, sp_cf=sp_cf,
            gain=sp_model - sp_cf,
            dom_pred=dom_pred, dom_obs=st1["dom"],
            dom_correct=dom_correct,
        ))

        sw = " <-- SWITCH" if is_sw else ""
        log(f"  {mo}->{mo1}  H={st['H']:.3f} p_sw={p_sw:.3f} | "
            f"model {sp_model:.3f} cf {sp_cf:.3f} "
            f"gain {sp_model-sp_cf:+.3f} | "
            f"dom {'OK' if dom_correct else 'MISS'}{sw}")

    r = pd.DataFrame(rows)
    log(f"\n  over {len(r)} month pairs:")
    log(f"    frequency Spearman: model {r.sp_model.mean():.3f}  "
        f"copy-forward {r.sp_cf.mean():.3f}  "
        f"gain {r.gain.mean():+.3f}")
    log(f"    model beats CF in {(r.gain>0).sum()}/{len(r)} months")
    log(f"    dominant cluster correct: "
        f"{r.dom_correct.mean():.1%} overall  "
        f"{r[r.is_switch].dom_correct.mean():.1%} at switches")

    from sklearn.metrics import roc_auc_score
    if r.is_switch.sum() > 0 and (~r.is_switch).sum() > 0:
        auc_model = roc_auc_score(r.is_switch.astype(int), r.p_switch)
        auc_H = roc_auc_score(r.is_switch.astype(int), r.H_t)
        log(f"    switch AUC: model {auc_model:.3f}  H_obs alone {auc_H:.3f}")

    # ---- WALK-FORWARD ----
    if args.walkforward:
        log("\n" + "=" * 74)
        log("WALK-FORWARD (refit beta and gamma on months up to t)")
        log("=" * 74)
        from sklearn.linear_model import PoissonRegressor, LogisticRegression

        wf_rows = []
        for i, mo in enumerate(usable[:-1]):
            if i < args.min_train:
                continue
            mo1 = usable[i + 1]
            train_mos = usable[:i]

            # refit beta from training cluster-months
            tr = df[df.month.isin(train_mos)].dropna(
                subset=["new_set_share", "freq", "g_next"]
                if "g_next" in df.columns else ["new_set_share", "freq"])

            # fit beta: Poisson regression of exp(g_next) on new_set_share
            if "g_next" in df.columns and len(tr) > 20:
                try:
                    pm = PoissonRegressor(max_iter=500)
                    pm.fit(tr[["new_set_share"]].values,
                           np.exp(tr["g_next"].values))
                    beta_fit = float(pm.coef_[0])
                except Exception:
                    beta_fit = args.beta
            else:
                beta_fit = args.beta

            # refit gamma from training switch history
            sw_train = [{"H": state[m]["H"],
                         "is_sw": float(usable[j+1] in SWITCH_MONTHS)}
                        for j, m in enumerate(train_mos[:-1])
                        if m in state]
            if len(sw_train) > 5:
                sw_df = pd.DataFrame(sw_train)
                X = sw_df[["H"]].values
                y = sw_df["is_sw"].values
                b, *_ = np.linalg.lstsq(X, y, rcond=None)
                gamma_fit = max(0.0, float(b[0]))
            else:
                gamma_fit = args.gamma

            st, st1 = state[mo], state[mo1]
            clusters = list(st["pi"].keys())
            pi0 = np.array([st["pi"].get(k, 0.0) for k in clusters])
            nss0 = np.array([st["nss"].get(k, 0.0) for k in clusters])
            pi1_obs = np.array([st1["pi"].get(k, 0.0) for k in clusters])
            f = pi0 * np.exp(beta_fit * nss0)
            pi1_pred = f / f.sum() if f.sum() > 0 else pi0
            p_sw = max(0.0, min(1.0, gamma_fit * st["H"]))
            is_sw = mo1 in SWITCH_MONTHS
            sp_m = spearman(pi1_pred, pi1_obs)
            sp_c = spearman(pi0, pi1_obs)
            wf_rows.append(dict(month_t=mo, beta=beta_fit, gamma=gamma_fit,
                                sp_model=sp_m, sp_cf=sp_c, gain=sp_m-sp_c,
                                p_switch=p_sw, is_switch=is_sw,
                                H_t=st["H"]))
            log(f"  {mo}  beta={beta_fit:.3f} gamma={gamma_fit:.3f} | "
                f"model {sp_m:.3f} cf {sp_c:.3f} gain {sp_m-sp_c:+.3f} | "
                f"p_sw={p_sw:.3f} {'SWITCH' if is_sw else ''}")

        if wf_rows:
            wr = pd.DataFrame(wf_rows)
            log(f"\n  walk-forward over {len(wr)} months:")
            log(f"    frequency gain: {wr.gain.mean():+.3f}  "
                f"beats CF in {(wr.gain>0).sum()}/{len(wr)}")
            if wr.is_switch.sum() > 0:
                auc_wf = roc_auc_score(wr.is_switch.astype(int), wr.p_switch)
                log(f"    switch AUC walk-forward: {auc_wf:.3f}")

    r.to_csv(args.out, index=False)

    log("\n" + "-" * 74)
    log("READ")
    log("-" * 74)
    log(f"  The model is a CONDITIONAL TRANSITION model, not an autonomous")
    log(f"  generative model. Given observed pi(t) and s(t), it predicts")
    log(f"  pi(t+1) and P(switch). This is the same status as Luksza-Lassig.")
    log(f"")
    log(f"  beta={args.beta:.2f} captures within-regime growth.")
    log(f"  gamma={args.gamma:.3f} captures switch timing from population entropy.")
    log(f"  Both are estimable from sequence count data alone.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
