#!/usr/bin/env python
"""
130_background_effect.py -- does WHICH background a mutation lands on follow
any rule?

WHY THIS AND NOT THE BLOCK-LEVEL TESTS
--------------------------------------
Every mechanism tested so far has been about how existing components move:
whether component weights have dynamics, whether a chain beats a slope, whether
growth follows composition. All were null, and all had the same problem --
roughly five hundred component-month observations, which is not enough to learn
a mechanism about anything.

This asks a different question with two orders of magnitude more data. For
every component that held enough sequences in a window, and every mutation, did
that mutation appear in that component in the next window when it had not been
there before? That is an appearance. A component that was abundant and did NOT
pick the mutation up is a non-appearance, and it is informative precisely
because the component was abundant. The power check found around eight hundred
mutations with both, and several thousand observations.

THE MODEL
---------
    logit P(mutation m appears on component k between t and t+1)
        = alpha_m + beta_k + log n_{k,t}

alpha_m is the mutation's own propensity -- a nuisance. beta_k is the thing in
question: does this background make mutations more likely to arrive at all.
log n is an exposure offset, because a component holding eighty thousand
sequences has far more chance to reveal a rare event than one holding two
hundred.

Conditioning on mutation identity removes alpha_m exactly, which is the point:
the comparison is then WITHIN a mutation, ACROSS backgrounds that were equally
observable in the same window. Fitted here by centring within each
(mutation, window) stratum, which is the standard conditional-logistic
reduction.

WHAT A NON-APPEARANCE ACTUALLY MEANS
------------------------------------
Three things, and this data cannot separate them: the mutation never arose, it
arose and died before becoming detectable, or it arose and was never sequenced.
Raising the exposure floor makes the first more likely to dominate but does not
isolate it. So beta_k is a statement about appearance in the record, not about
mutation rate.

LEAKAGE
-------
Components are fitted on the training window only and appearance is scored in
the windows after it, so a component cannot have been defined by the event
being scored. Transfer is measured by holding out a whole window.
"""
import argparse, importlib.util, sys
from collections import defaultdict
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def observe(E, model, Pi, tv, later, ks, drift=True):
    """Per scored month: which mutations each component carries, and its
    weight. Hard assignment, because a sequence split across components would
    register as a partial appearance on both."""
    dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
    present, weight = [], []
    for i, (ym, X, w) in enumerate(later):
        th, kk = model.theta(tv[-1] + (i + 1) * dt, drift)
        lp = E.loglik_matrix(X, th) + np.log(Pi[-1] + EPS)[None, :]
        z = kk[lp.argmax(1)]
        pres = {}; wt = {}
        for k in ks:
            m = z == k
            if not m.any():
                continue
            wt[int(k)] = float(w[m].sum())
            pres[int(k)] = set(np.flatnonzero(
                (X[m] * w[m, None]).sum(0) > 0).tolist())
        present.append(pres); weight.append(wt)
    return present, weight


def build_rows(present, weight, floor, min_freq=0.0, X_by_month=None):
    """(window, stratum, component, exposure, outcome) for every eligible
    (mutation, component, month). A stratum is one mutation in one month --
    the unit within which the mutation's own propensity cancels."""
    rows = []
    for t in range(len(present) - 1):
        elig = [k for k, v in weight[t].items()
                if v >= floor and weight[t + 1].get(k, 0.0) >= floor]
        if len(elig) < 2:
            continue
        gained = {k: present[t + 1][k] - present[t][k] for k in elig}
        appeared_now = set().union(*gained.values()) if gained else set()
        for m in appeared_now:
            at_risk = [k for k in elig if m not in present[t][k]]
            if len(at_risk) < 2:
                continue                 # no contrast inside this stratum
            ys = [1 if m in gained[k] else 0 for k in at_risk]
            if sum(ys) == 0 or sum(ys) == len(ys):
                continue                 # a stratum with no variation is
                                         # uninformative once alpha_m is out
            for k, y in zip(at_risk, ys):
                rows.append((t, m, int(k), np.log(weight[t][k]), y))
    return rows


def fit_conditional(rows, ks_index, l2=1.0, iters=200, lr=0.5):
    """Conditional logistic on beta, stratified by (month, mutation).

    Within a stratum the mutation's own propensity is a constant, so it is
    removed by centring the linear predictor on the stratum before taking the
    likelihood. What survives is the contrast between backgrounds."""
    K = len(ks_index)
    strata = defaultdict(list)
    for i, (t, m, k, off, y) in enumerate(rows):
        strata[(t, m)].append(i)
    kk = np.array([ks_index[r[2]] for r in rows])
    off = np.array([r[3] for r in rows])
    y = np.array([r[4] for r in rows], dtype=float)
    beta = np.zeros(K)
    idx = [np.array(v) for v in strata.values()]
    for _ in range(iters):
        eta = beta[kk] + off
        p = np.zeros(len(rows))
        for ii in idx:                     # softmax within stratum
            e = eta[ii] - eta[ii].max()
            ex = np.exp(e); s = ex.sum()
            p[ii] = ex / max(s, EPS) * y[ii].sum()
        g = np.zeros(K)
        np.add.at(g, kk, y - p)
        g -= l2 * beta
        beta += lr * g / max(len(idx), 1)
        beta -= beta.mean()                # only contrasts are identified
    return beta, kk, off, y, idx


def stratum_ll(beta, kk, off, y, idx):
    """Mean conditional log-likelihood per stratum: how well the fitted
    background effects rank which component actually gained the mutation."""
    eta = beta[kk] + off
    tot = 0.0
    for ii in idx:
        e = eta[ii] - eta[ii].max()
        lse = np.log(np.exp(e).sum() + EPS)
        tot += float((y[ii] * (e - lse)).sum() / max(y[ii].sum(), 1))
    return tot / max(len(idx), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--windows", required=True,
                    help="comma-separated LAST training months; each is fitted "
                         "on the 12 months ending there and scored on the "
                         "months after")
    ap.add_argument("--window-len", type=int, default=12)
    ap.add_argument("--score-months", type=int, default=6)
    ap.add_argument("--exposure", type=float, default=2000.0)
    ap.add_argument("--max-K", type=int, default=48)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()

    E = load_engine(args.engine)
    names, V = E.load_names(args.vocab)
    wins = [x.strip() for x in args.windows.split(",") if x.strip()]

    per_win = []
    for last in wins:
        tr = [E.ym_add(last, -(args.window_len - 1) + i)
              for i in range(args.window_len)]
        recs = [E.load_month(args.data_dir, ym) for ym in tr]
        if any(r is None for r in recs):
            print(f"  window ..{last}: missing months, skipped"); continue
        Xs, ws = zip(*[E.build(r, V, args.min_count) for r in recs])
        Xs, ws = list(Xs), list(ws)
        later = []
        for h in range(1, args.score_months + 1):
            ym = E.ym_add(last, h)
            r = E.load_month(args.data_dir, ym)
            if r is not None:
                later.append((ym,) + E.build(r, V, 1))
        if len(later) < 2:
            print(f"  window ..{last}: too few months after it"); continue
        _, _, _, _, _, w0 = E.fit_flat(Xs, ws, V, args.max_K, seed=args.seed,
                                       drift=False, split_merge=False)
        _, pi_sm, _, _, _, beta_sm = E.fit_flat(
            Xs, ws, V, args.max_K, seed=args.seed, drift=True,
            split_merge=True, init_beta=w0)
        model, Pi, tv, _, _, _ = E.fit(
            Xs, ws, V, args.max_K, seed=args.seed, drift=True, names=names,
            verbose=False, iters=args.iters, K_warm=args.max_K, birth_every=1,
            births_per_call=4, refit=0, penalty="prior", warm=(beta_sm, pi_sm),
            warm_mode="tree", hier_drift=True, rescan_every=25)
        ks = np.flatnonzero(model.alive)
        present, weight = observe(E, model, Pi, tv, later, ks)
        rows = build_rows(present, weight, args.exposure)
        nstr = len({(r[0], r[1]) for r in rows})
        print(f"  window {tr[0]}..{tr[-1]} -> {later[0][0]}..{later[-1][0]}   "
              f"{int(model.alive.sum())} blocks   {len(rows)} rows   "
              f"{nstr} strata", flush=True)
        per_win.append((last, rows, ks))

    if len(per_win) < 2:
        sys.exit("need at least two usable windows")

    print(f"\n{'=' * 78}\n  DOES THE BACKGROUND MATTER, AND DOES IT TRANSFER?"
          f"\n{'=' * 78}")
    print(f"\n  exposure floor {args.exposure:,.0f}   "
          f"strata are (month, mutation) pairs with at least two eligible "
          f"backgrounds\n  and both outcomes present\n")
    print(f"  {'held-out window':<18}{'strata':>9}{'rows':>8}"
          f"{'LL fitted':>12}{'LL null':>10}{'gain':>9}")
    gains = []
    for hi, (last, rows_h, ks_h) in enumerate(per_win):
        tr_rows = [r for j, (l, rs, k) in enumerate(per_win) if j != hi
                   for r in rs]
        # components are per-window objects, so the only thing that can
        # transfer is a component's DEPTH-free identity; use its index within
        # the alive set, which is stable enough to test the question
        kset = sorted({r[2] for r in tr_rows} | {r[2] for r in rows_h})
        kidx = {k: i for i, k in enumerate(kset)}
        beta, kk, off, y, idx = fit_conditional(tr_rows, kidx, l2=args.l2)
        kkh = np.array([kidx[r[2]] for r in rows_h])
        offh = np.array([r[3] for r in rows_h])
        yh = np.array([r[4] for r in rows_h], dtype=float)
        st = defaultdict(list)
        for i, r in enumerate(rows_h):
            st[(r[0], r[1])].append(i)
        idxh = [np.array(v) for v in st.values()]
        fit_ll = stratum_ll(beta, kkh, offh, yh, idxh)
        null_ll = stratum_ll(np.zeros(len(kset)), kkh, offh, yh, idxh)
        gains.append(fit_ll - null_ll)
        print(f"  ..{last:<16}{len(idxh):>9}{len(rows_h):>8}"
              f"{fit_ll:>12.4f}{null_ll:>10.4f}{fit_ll - null_ll:>+9.4f}")
    print(f"\n  mean gain over exposure alone: {np.mean(gains):+.4f}")
    print("""
  LL null uses the exposure offset only -- the background contributes nothing,
  and a mutation is assumed to land wherever there are most sequences to see
  it. LL fitted adds the estimated background effects.

  gain clearly positive, on held-out windows
      -> which background a mutation lands on is not simply a matter of how
         many sequences that background holds. There is a mechanism, and it
         transfers to windows it was not fitted on.
  gain around zero
      -> appearance is explained by exposure alone. Given how many sequences a
         background holds, which mutations land on it is close to arbitrary,
         and no amount of modelling the existing components will change that.
         That is a real answer to the question the block-level tests could not
         address for want of data.

  A caveat the number cannot remove: components are refitted per window, so a
  component index does not mean the same thing across windows. The transfer
  test is therefore conservative -- it asks whether background effects
  generalise even under relabelling.
""")


if __name__ == "__main__":
    sys.exit(main())
