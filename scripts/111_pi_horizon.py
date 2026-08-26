#!/usr/bin/env python
"""
111_pi_horizon.py -- is copying last month's mixture good enough at h > 1?

THE QUESTION IN PLAIN ENGLISH
-----------------------------
The model predicts next month's population as a mixture: a set of background
profiles (which mutations each background carries) and a set of weights (how
common each background is). At a one-month horizon the composition barely
moves, so just copying last month's weights is hard to beat. Over several
months real turnover accumulates -- Delta gives way to Omicron, BA.2 to BA.5 --
so a rule that extrapolates the weights might start to matter.

WHY RAW HELD-OUT LL CANNOT ANSWER IT
------------------------------------
At longer horizons the PROFILES degrade too: drift extrapolates badly and
genuinely new backgrounds appear that no training-window component represents.
So log-likelihood falls with h for every mixture rule. A weight rule can look
useless when the profiles are the bottleneck, or look helpful when it is really
riding a better or worse set of profiles. The horizon curve of LL is
uninterpretable on its own.

WHAT THIS SCRIPT COMPUTES INSTEAD
---------------------------------
Profiles are held FIXED at the model's h-step extrapolation. Only the weights
vary, across four settings:

  persist   copy the last training month's weights
  growth    extrapolate each weight by its last month-over-month ratio, h steps
  loglin    least-squares trend on log-weight over the last --trend-window months
  oracle    the best possible weights on the test month, fitted with the
            profiles frozen

The oracle uses the test month, so it is a DIAGNOSTIC CEILING, not a model. It
cannot be reported as a result. Its only job is to bound how much is available
to any weight rule at all.

READING THE OUTPUT
------------------
  gap = oracle - persist    how much a perfect weight rule could buy

  gap stays small as h grows        -> weights are not the bottleneck.
                                       Do not model them. Work on emissions.
  gap grows AND a rule closes it    -> model the weights. The rule that closes
                                       it is the candidate.
  gap grows and NO rule closes it   -> the missing mass is in backgrounds that
                                       do not exist in the training window.
                                       That is a proposal/birth problem, not a
                                       mixture-law problem.

Averaged over several anchors so the answer is not one month's accident.
"""
import argparse, importlib.util, os, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def oracle_pi(E, X, w, th, iters=300, tol=1e-9):
    """Best mixture weights on this month with the profiles frozen.

    Plain EM on the weights alone: responsibilities, then re-weight. Because
    the profiles never move, this is a concave problem in the weights and the
    fixed point is the true maximum, so the number really is a ceiling."""
    K = th.shape[0]
    L = E.loglik_matrix(X, th)
    pi = np.full(K, 1.0 / K)
    prev = -np.inf
    for _ in range(iters):
        lp = L + np.log(pi + EPS)[None, :]
        mx = lp.max(1, keepdims=True)
        R = np.exp(lp - mx)
        s = R.sum(1, keepdims=True)
        ll = float((w * (np.log(s.ravel() + EPS) + mx.ravel())).sum() / w.sum())
        R /= np.maximum(s, EPS)
        pi = (w[:, None] * R).sum(0)
        pi = pi / max(pi.sum(), EPS)
        if abs(ll - prev) < tol:
            break
        prev = ll
    return pi


def pi_growth(P, h):
    """Each weight continues its last month-over-month ratio for h more steps.
    Multiplicative in the weight, i.e. linear in log-weight, which is the shape
    a component under constant selective advantage actually follows early on."""
    if len(P) < 2:
        return P[-1].copy()
    a, b = np.maximum(P[-2], 1e-8), np.maximum(P[-1], 1e-8)
    r = np.clip(b / a, 0.2, 5.0)
    q = b * r ** h
    return q / max(q.sum(), EPS)


def pi_loglin(P, h, window):
    """Least-squares trend on log-weight over the last `window` months.
    Less jumpy than the two-point ratio when a single month is anomalous."""
    W = P[-window:] if len(P) >= window else P
    if len(W) < 2:
        return P[-1].copy()
    t = np.arange(len(W), dtype=float)
    Y = np.log(np.maximum(np.stack(W), 1e-8))
    tc = t - t.mean()
    slope = (tc[:, None] * (Y - Y.mean(0))).sum(0) / max((tc ** 2).sum(), EPS)
    slope = np.clip(slope, -1.6, 1.6)
    q = np.exp(Y[-1] + slope * h)
    return q / max(q.sum(), EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="scripts/110_hierarchical_birthdeath_v2.py",
                    help="path to the fitting script; its functions are reused")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--anchors", required=True,
                    help="comma-separated last-training-months, e.g. "
                         "2021-12,2022-03,2022-06")
    ap.add_argument("--train-len", type=int, default=12)
    ap.add_argument("--horizons", default="1,2,3,6")
    ap.add_argument("--trend-window", type=int, default=6)
    ap.add_argument("--max-K", type=int, default=120)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--half-life", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()

    E = load_engine(args.engine)
    names, V = E.load_names(args.vocab)
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    hs = [int(x) for x in args.horizons.split(",")]
    rules = ["persist", "growth", "loglin", "oracle"]

    acc = {h: {r: [] for r in rules} for h in hs}
    ncomp = []

    for anchor in anchors:
        tr = [E.ym_add(anchor, -(args.train_len - 1) + i)
              for i in range(args.train_len)]
        recs = [E.load_month(args.data_dir, ym) for ym in tr]
        if any(r is None for r in recs):
            print(f"  [{anchor}] skipped: missing training month", flush=True)
            continue
        Xs, ws = zip(*[E.build(r, V, args.min_count) for r in recs])
        Xs, ws = list(Xs), list(ws)

        for sd in range(args.seeds):
            # warm start exactly as the ladder does it
            _, _, _, _, _, w0 = E.fit_flat(Xs, ws, V, args.max_K, seed=sd,
                                           drift=False, split_merge=False)
            _, pi_sm, _, _, _, beta_sm = E.fit_flat(
                Xs, ws, V, args.max_K, seed=sd, drift=True, split_merge=True,
                half_life=args.half_life, init_beta=w0)
            model, Pi, tv, births, _, _ = E.fit(
                Xs, ws, V, args.max_K, seed=sd, sigma=args.sigma,
                half_life=args.half_life, drift=True, names=names,
                verbose=False, iters=args.iters, K_warm=args.max_K,
                birth_every=1, births_per_call=4, refit=0,
                penalty="prior", warm=(beta_sm, pi_sm), warm_mode="tree",
                hier_drift=True, rescan_every=25)
            ncomp.append(int(model.alive.sum()))
            dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
            P = [Pi[t] for t in range(len(Pi))]     # monthly weight history

            for h in hs:
                te_ym = E.ym_add(anchor, h)
                rec = E.load_month(args.data_dir, te_ym)
                if rec is None:
                    continue
                Xte, wte = E.build(rec, V, 1)
                # profiles held fixed at the h-step extrapolation
                th, ks = model.theta(tv[-1] + h * dt, True)
                cand = {
                    "persist": P[-1],
                    "growth": pi_growth(P, h),
                    "loglin": pi_loglin(P, h, args.trend_window),
                    "oracle": oracle_pi(E, Xte, wte, th),
                }
                for r in rules:
                    acc[h][r].append(E.score(Xte, wte, th, cand[r]))
        print(f"  [{anchor}] fitted, K={ncomp[-1]}", flush=True)

    print("\n" + "=" * 78)
    print(f"MIXTURE GAP BY HORIZON   anchors={len(anchors)}  "
          f"seeds={args.seeds}  mean K={np.mean(ncomp):.0f}")
    print("=" * 78)
    print(f"\n  {'h':>3}{'persist':>11}{'growth':>11}{'loglin':>11}"
          f"{'oracle':>11}{'gap':>10}{'best rule closes':>18}")
    for h in hs:
        if not acc[h]["persist"]:
            continue
        mu = {r: float(np.mean(acc[h][r])) for r in rules}
        gap = mu["oracle"] - mu["persist"]
        best = max(mu["growth"], mu["loglin"])
        frac = (best - mu["persist"]) / gap if gap > 1e-9 else 0.0
        print(f"  {h:>3}{mu['persist']:>11.4f}{mu['growth']:>11.4f}"
              f"{mu['loglin']:>11.4f}{mu['oracle']:>11.4f}{gap:>10.4f}"
              f"{frac:>17.0%}")
    print("""
  gap    = oracle - persist. The most any weight rule could ever buy.
           Oracle is fitted ON the test month: a ceiling, never a result.
  closes = how much of that gap the better of growth/loglin actually captures.

  gap flat in h                 -> weights are not the bottleneck; the loss at
                                   longer horizons is in the profiles.
  gap grows, closes is high     -> model the weights, using the winning rule.
  gap grows, closes near zero   -> the missing mass sits in backgrounds absent
                                   from the training window. That is a proposal
                                   problem: no mixture law over existing
                                   components can reach it.
""")


if __name__ == "__main__":
    sys.exit(main())
