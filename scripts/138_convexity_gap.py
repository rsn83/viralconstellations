#!/usr/bin/env python
"""
138_convexity_gap.py

    python scripts/138_convexity_gap.py \
      --data-dir data/processed/full_data_graphs_withdel \
      --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
      --train 2021-06:2022-05 --test 2022-11 \
      --max-K 48 --warm-from-sm --warm-mode tree --gamma-tau 0.5 --subsample 0.2

Tests whether the constellations a fitted mixture underpredicts at the test
month are the ones whose mutations are SPLIT ACROSS several components of the
training-window fit.

HYPOTHESIS: a mixture with factorized per-component emission cannot concentrate
mass on cross-component combinations, so predictive miss should be worse for
constellations needing >=2 components to cover -- AT FIXED CONSTELLATION SIZE.

The size stratification is the experiment. The raw correlation is confounded
(bigger sets need more components AND are rarer) and is printed only so it does
not get reported by accident.
"""

import argparse, importlib.util, os, sys
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

DEFAULT_ENGINE = "scripts/110_hierarchical_birthdeath_v2_fixed.py"


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["engine"] = m
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------- fit -----

def fit_window(E, Xs, ws, V, a):
    """Call the hierarchical engine, tolerating either return shape.
    fit_flat returns (th_next, Pi[-1], nK, ll, splits, beta); fit is assumed
    to lead with (theta, pi) likewise. Extra kwargs are dropped if unsupported."""
    kw = dict(seed=a.seed, sigma=a.sigma)
    for name, val in [("hier_drift", a.hier_drift),
                      ("warm_from_sm", a.warm_from_sm),
                      ("warm_mode", a.warm_mode),
                      ("gamma_tau", a.gamma_tau),
                      ("p_birth", a.p_birth)]:
        if val is not None and val is not False:
            kw[name] = val
    fn = getattr(E, "fit")
    while True:
        try:
            out = fn(Xs, ws, V, a.max_K, **kw)
            break
        except TypeError as err:
            msg = str(err)
            dropped = [k for k in list(kw) if k in msg]
            if not dropped:
                raise
            for k in dropped:
                kw.pop(k)
            print(f"[fit] engine.fit does not take {dropped}, dropping")

    model, Pi, tv = out[0], out[1], out[2]
    th, ks = model.theta(t=tv[-1], drift=True)
    th = np.asarray(th)
    pi = np.asarray(Pi[-1])
    if th.ndim != 2:
        raise RuntimeError(f"expected (K,V) theta, got shape {th.shape}. "
                           f"Check what engine.fit returns first.")
    if pi.ndim == 2:            # full Pi matrix -> take last training month
        pi = pi[-1]
    return th, pi


# ---------------------------------------------------------------- core -----

def owned_sets(theta, tau):
    return [set(np.flatnonzero(theta[k] > tau).tolist())
            for k in range(theta.shape[0])]


def cover(s_idx, owned, cap=12):
    """Greedy cover of a constellation's mutations by components.
         n == 1        inside one component        (reachable by the mixture)
         n >= 2        split across components     (the claim)
         uncovered > 0 no component owns it        (121's new-mutation channel)
    """
    remaining = set(s_idx.tolist())
    n = 0
    while remaining and n < cap:
        gains = [len(remaining & o) for o in owned]
        b = int(np.argmax(gains))
        if gains[b] == 0:
            break
        remaining -= owned[b]
        n += 1
    return n, len(remaining)


def logp_mixture(s_idx, lt, lf, base, log_pi):
    if len(s_idx) == 0:
        return logsumexp(log_pi + base)
    return logsumexp(log_pi + base + (lt[:, s_idx] - lf[:, s_idx]).sum(axis=1))


def to_idx(s_, name2idx):
    lst = list(s_)
    if not lst:
        return np.array([], dtype=int)
    if isinstance(lst[0], (int, np.integer)):
        return np.array(sorted(lst), dtype=int)
    return np.array(sorted(name2idx[x] for x in lst), dtype=int)


def month_measure(E, data_dir, ym, name2idx):
    recs = E.load_month(data_dir, ym)
    sets = [to_idx(s_, name2idx) for s_, c in recs]
    w = np.array([float(c) for s_, c in recs])
    return sets, w / w.sum()


def run(a):
    E = load_engine(a.engine)
    idx2name, V = E.load_names(a.vocab)
    name2idx = {n: i for i, n in idx2name.items()}

    train = E.months_in_range(a.train)
    print(f"[data] train {train[0]}..{train[-1]} ({len(train)} months), "
          f"test {a.test}, V={V}")

    recs = [E.load_month(a.data_dir, ym) for ym in train]
    if a.subsample and a.subsample < 1.0:
        rng = np.random.default_rng(a.seed)
        recs = [[(s_, c) for s_, c in r if rng.random() < a.subsample]
                for r in recs]
    Xs, ws = zip(*[E.build(r, V, a.min_count) for r in recs])
    theta, pi = fit_window(E, list(Xs), list(ws), V, a)

    theta = np.clip(theta, 1e-6, 1 - 1e-6)
    log_pi = np.log(np.asarray(pi) + 1e-300)
    lt, lf = np.log(theta), np.log1p(-theta)
    base = lf.sum(axis=1)
    owned = owned_sets(theta, a.tau)
    n_owned = float(np.mean([len(o) for o in owned]))
    print(f"[fit] K={theta.shape[0]}  mean positions owned per component "
          f"at tau={a.tau}: {n_owned:.1f}")
    if n_owned < 1:
        print("      -> tau above where your theta sit; everything will look "
              "cross-component. Lower --tau to 0.3 and rerun.")

    sets_t, w_t = month_measure(E, a.data_dir, train[-1], name2idx)
    sets_f, w_f = month_measure(E, a.data_dir, a.test, name2idx)
    present = {frozenset(s.tolist()) for s, w in zip(sets_t, w_t)
               if w >= a.thresh}

    rows = []
    for s, w in zip(sets_f, w_f):
        if w < a.thresh or frozenset(s.tolist()) in present:
            continue
        n_comp, n_unc = cover(s, owned)
        if n_comp == 0:
            continue
        rows.append((len(s), n_comp, n_unc, np.log(w),
                     np.log(w) - logp_mixture(s, lt, lf, base, log_pi)))

    if not rows:
        print(f"\nno novel constellations at thresh={a.thresh} "
              f"-- try {a.thresh/10:g}")
        return
    A = np.array(rows, dtype=float)
    size, ncomp, nunc, logw, miss = A.T

    print(f"\nnovel constellations above threshold: {len(A)}")
    print(f"  cross-component (n_comp >= 2): {(ncomp >= 2).mean():.1%}")
    print(f"  with uncovered mutations:      {(nunc > 0).mean():.1%}"
          f"   <- SANITY vs 121's 3.8%; if >30% the indexing is wrong")

    r, p = spearmanr(ncomp, miss)
    print(f"\nraw spearman(n_comp, miss) = {r:+.3f} p={p:.3g}"
          f"   <- CONFOUNDED by size, do not report")

    print("\nsize-stratified (this is the result):")
    print(f"{'size':>6} {'n':>5} {'within':>8} {'cross':>8} {'gap':>8} {'rho':>7}")
    strat = []
    for sz in np.unique(size):
        m = size == sz
        wi, cr = miss[m & (ncomp == 1)], miss[m & (ncomp >= 2)]
        if len(wi) < 3 or len(cr) < 3:
            continue
        rho, _ = spearmanr(ncomp[m], miss[m])
        gap = cr.mean() - wi.mean()
        print(f"{int(sz):>6} {int(m.sum()):>5} {wi.mean():>8.2f} "
              f"{cr.mean():>8.2f} {gap:>8.2f} {rho:>+7.3f}")
        strat.append((sz, m.sum(), gap))
    if strat:
        st = np.array(strat, dtype=float)
        print(f"\nsize-weighted mean gap: "
              f"{np.average(st[:,2], weights=st[:,1]):+.3f} nats")
        print("negative => cross-component novelty underpredicted, as claimed")
        print("~zero    => mechanism is wrong, stop here")
    else:
        print("\nno size bin had >=3 of each class -- lower --thresh")

    os.makedirs("results", exist_ok=True)
    np.savez("results/138_convexity.npz", size=size, ncomp=ncomp,
             nuncovered=nunc, logfreq=logw, miss=miss, tau=a.tau,
             train=a.train, test=a.test, K=theta.shape[0])
    print("\nsaved results/138_convexity.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--train", required=True, help="2021-06:2022-05")
    p.add_argument("--test", required=True, help="2022-11 (h months after)")
    p.add_argument("--engine", default=DEFAULT_ENGINE)
    p.add_argument("--max-K", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--subsample", type=float, default=None)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--p-birth", type=float, default=None, dest="p_birth")
    p.add_argument("--gamma-tau", type=float, default=None, dest="gamma_tau")
    p.add_argument("--warm-from-sm", action="store_true", dest="warm_from_sm")
    p.add_argument("--warm-mode", default=None, dest="warm_mode")
    p.add_argument("--hier-drift", action="store_true", dest="hier_drift")
    p.add_argument("--tau", type=float, default=0.5, help="component ownership")
    p.add_argument("--thresh", type=float, default=1e-3, help="prevalence")
    run(p.parse_args())
