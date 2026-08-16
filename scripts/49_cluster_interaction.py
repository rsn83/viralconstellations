#!/usr/bin/env python
"""
49_cluster_interaction.py

Does adding cluster-cluster interactions improve frequency prediction
beyond the simple fitness model?

THE SIMPLE MODEL (already validated)
--------------------------------------
pi_k(t+1) proportional to pi_k(t) * exp(beta * s_k(t))
Spearman 0.778 one-step-ahead, gain +0.012 over copy-forward

THE EXTENDED MODEL
------------------
pi_k(t+1) proportional to pi_k(t) * exp(beta * s_k(t) + A_k(t))

where A_k(t) = sum_j W[k,j] * pi_j(t)

W[k,j] is a learnable interaction weight:
  W[k,j] > 0: cluster j's presence helps cluster k grow
  W[k,j] < 0: cluster j's presence suppresses cluster k
  W[k,j] = 0: no interaction (the simple model)

W is regularised toward zero (L2). With 40 time steps and ~15 active
clusters per month, this is a small parameter count -- identifiable.

BIOLOGICAL INTERPRETATION
--------------------------
W[k,j] < 0 (suppression): immune pressure from lineage j reduces
  susceptible hosts available to lineage k. Classic competitive exclusion.

W[k,j] > 0 (facilitation): lineage j creates conditions that help k.
  Less common but possible if j depletes a competing lineage.

EVALUATION
-----------
Walk-forward: fit beta and W on months before t, predict t+1.
Compare Spearman against simple model (beta only).

If gain > 0.02 and holds across most months: interactions are real.
If gain < 0.01: clusters evolve independently, simple model sufficient.

Usage
-----
  python scripts/49_cluster_interaction.py
  python scripts/49_cluster_interaction.py --l2 0.1 --min_train 15
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


def fit_interaction_model(pi_mat, nss_mat, cluster_ids, l2=1.0):
    """
    Fit beta and W jointly by gradient descent.

    pi_mat:  (T, K) observed cluster frequencies
    nss_mat: (T, K) new_set_share per cluster per month
    l2:      regularisation on W

    Returns: beta (scalar), W (K x K matrix)
    """
    T, K = pi_mat.shape

    # initialise
    beta = np.float64(0.49)
    W = np.zeros((K, K), dtype=np.float64)

    lr_beta = 0.01
    lr_W = 0.001
    n_iter = 500

    for it in range(n_iter):
        total_loss = 0.0
        grad_beta = 0.0
        grad_W = np.zeros_like(W)

        for t in range(T - 1):
            pi_t = pi_mat[t]      # (K,)
            s_t = nss_mat[t]      # (K,)
            pi_t1 = pi_mat[t + 1] # (K,) observed

            # interaction term
            A_t = W @ pi_t        # (K,)

            # predicted log fitness
            log_f = beta * s_t + A_t

            # predicted frequencies (softmax)
            log_pi_pred = np.log(pi_t + 1e-12) + log_f
            log_pi_pred -= log_pi_pred.max()
            pi_pred = np.exp(log_pi_pred)
            pi_pred /= pi_pred.sum()

            # cross-entropy loss
            loss = -np.sum(pi_t1 * np.log(pi_pred + 1e-12))
            total_loss += loss

            # gradients (delta = pi_t1 - pi_pred weighted)
            delta = pi_t1 - pi_pred  # (K,)

            grad_beta -= np.sum(delta * s_t)
            grad_W -= np.outer(delta, pi_t)

        # L2 on W
        total_loss += l2 * np.sum(W ** 2)
        grad_W += 2 * l2 * W

        # update
        beta -= lr_beta * grad_beta / (T - 1)
        W -= lr_W * grad_W / (T - 1)

        # keep beta positive
        beta = max(0.0, beta)

        if (it + 1) % 100 == 0:
            log(f"    iter {it+1}/{n_iter}  loss={total_loss/(T-1):.4f}  "
                f"beta={beta:.3f}  |W|={np.abs(W).mean():.4f}")

    return beta, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "40_internal.csv"))
    ap.add_argument("--l2", type=float, default=1.0,
                    help="L2 regularisation on W (larger = more shrinkage toward 0)")
    ap.add_argument("--min_train", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "49_interaction.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    months = sorted(df.month.unique())
    log(f"{len(df)} cluster-months, {df.cluster.nunique()} clusters, "
        f"{len(months)} months\n")

    # build per-month state
    state = {}
    for mo in months:
        t = df[df.month == mo]
        if len(t) < 2:
            continue
        tot = t.freq.sum()
        pi = dict(zip(t.cluster.astype(int), t.freq / tot))
        nss = dict(zip(t.cluster.astype(int), t.new_set_share))
        state[mo] = dict(pi=pi, nss=nss)
    usable = [m for m in months if m in state]

    # get union of all clusters
    all_clusters = sorted(set(k for m in usable for k in state[m]["pi"]))
    K = len(all_clusters)
    cidx = {k: i for i, k in enumerate(all_clusters)}
    log(f"Using {K} clusters across {len(usable)} months\n")

    # build matrices
    pi_mat = np.zeros((len(usable), K))
    nss_mat = np.zeros((len(usable), K))
    for i, mo in enumerate(usable):
        for k, v in state[mo]["pi"].items():
            pi_mat[i, cidx[k]] = v
        for k, v in state[mo]["nss"].items():
            nss_mat[i, cidx[k]] = v

    # ---- WALK-FORWARD EVALUATION ----
    log("=" * 74)
    log("WALK-FORWARD EVALUATION")
    log("=" * 74)
    log(f"  {'month':<10} {'simple':>8} {'interact':>10} {'gain':>8} {'switch':>8}")

    rows = []
    for i, mo in enumerate(usable[:-1]):
        if i < args.min_train:
            continue
        mo1 = usable[i + 1]

        # training data: months up to i
        pi_tr = pi_mat[:i]
        nss_tr = nss_mat[:i]

        # fit simple model: just beta
        # use pre-estimated beta=0.49 from Poisson regression
        beta_simple = 0.49

        # fit interaction model on training data
        log(f"  fitting interaction model on {i} months...")
        beta_int, W = fit_interaction_model(
            pi_tr, nss_tr, all_clusters, l2=args.l2)

        # predict t+1
        pi_t = pi_mat[i]
        nss_t = nss_mat[i]
        pi_t1_obs = pi_mat[i + 1]

        # simple prediction
        f_simple = pi_t * np.exp(beta_simple * nss_t)
        pi_simple = f_simple / f_simple.sum() if f_simple.sum() > 0 else pi_t

        # interaction prediction
        A_t = W @ pi_t
        f_int = pi_t * np.exp(beta_int * nss_t + A_t)
        pi_int = f_int / f_int.sum() if f_int.sum() > 0 else pi_t

        sp_simple = spearman(pi_simple, pi_t1_obs)
        sp_int = spearman(pi_int, pi_t1_obs)
        gain = sp_int - sp_simple
        is_sw = mo1 in SWITCH_MONTHS

        rows.append(dict(
            month_t=mo, month_t1=mo1,
            beta_int=beta_int,
            W_mean_abs=float(np.abs(W).mean()),
            sp_simple=sp_simple, sp_int=sp_int,
            gain=gain, is_switch=is_sw,
        ))

        log(f"  {mo:<10} {sp_simple:>8.3f} {sp_int:>10.3f} "
            f"{gain:>+8.3f} {'SWITCH' if is_sw else '':>8}")

    if not rows:
        raise SystemExit("no test months")

    r = pd.DataFrame(rows)
    pd.DataFrame(r).to_csv(args.out, index=False)

    log("\n" + "=" * 74)
    log("SUMMARY")
    log("=" * 74)
    log(f"  over {len(r)} test months:")
    log(f"    simple model:      {r.sp_simple.mean():+.4f}")
    log(f"    interaction model: {r.sp_int.mean():+.4f}")
    log(f"    gain:              {r.gain.mean():+.4f}")
    log(f"    beats simple in {(r.gain>0).sum()}/{len(r)} months")
    log(f"    mean |W|:          {r.W_mean_abs.mean():.4f}")

    try:
        from scipy.stats import binomtest
        p = binomtest(int((r.gain > 0).sum()), len(r), 0.5,
                      alternative='greater').pvalue
        log(f"    sign test p={p:.4f}")
    except Exception:
        pass

    log()
    if r.gain.mean() > 0.02 and (r.gain > 0).sum() > 0.65 * len(r):
        log("  Interaction model beats simple model consistently.")
        log("  Clusters do not evolve independently -- competitive")
        log("  or facilitative interactions between co-circulating")
        log("  lineages contribute to frequency dynamics.")
        log("  |W| > 0 confirms interactions are not shrunk to zero.")
    elif r.W_mean_abs.mean() < 0.01:
        log("  W shrinks to near-zero. Clusters evolve independently.")
        log("  The simple model is sufficient and interactions add nothing.")
        log("  This is consistent with additive fitness (script 32) --")
        log("  if within-constellation fitness is additive, between-cluster")
        log("  competition may also be effectively independent.")
    else:
        log("  Mixed. Check whether gain concentrates at switch months.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
