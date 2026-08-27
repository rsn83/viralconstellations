#!/usr/bin/env python
"""
116_theta_time_model.py -- a slope, or a chain over months?

WHY THIS COMPARISON IS POSSIBLE AND THE OTHERS WERE NOT
-------------------------------------------------------
Every diagnostic so far refitted the mixture, and the mixture's own
initialisation noise (+/- 0.6 to 3.6 nats in 114) swamped the effect being
measured. Here the mixture is fitted ONCE. Its blocks and its monthly weights
are then frozen, and only the way theta moves over time is varied. Both time
models see the same blocks, the same memberships and the same mixture weights,
so the noise cancels and a small difference is readable.

WHAT IS BEING COMPARED
----------------------
For every (block, mutation) pair the frozen fit gives a 12-month series: how
many sequences in that block carried that mutation each month, out of how many
sequences were in the block. Three ways to turn that series into a forecast:

  constant   pool the twelve months. No time model at all.
  slope      sig(beta + gamma*t), fitted per series. What the model does now.
  chain      a hidden state per month -- ancestral or mutated (optionally a
             middle state) -- with a shared transition matrix. Forecast by
             carrying the last month's state posterior forward h steps.

THE ARGUMENT FOR THE CHAIN
--------------------------
A slope forces steady motion in logit space. It cannot represent a plateau and
it cannot stop. That is exactly the failure on the drift slide: fitted on
months where 486V was near zero, the line creeps to 1.8% for June while the
truth jumped to 52.5%, and by six months out the line has kept climbing past
sweeps that already finished, so drift scores WORSE than no drift at all.

A chain has a stationary distribution. It cannot run away. A position that has
flipped stays flipped instead of continuing upward forever.

The transition matrix is shared across all series rather than fitted per pair,
because twelve points cannot support a private transition matrix. What is
private to each series is its state posterior. So the shared part answers "how
often do positions flip at all" and the private part answers "has this one
flipped yet".

WHAT THIS DOES NOT TEST
-----------------------
The mixture weights are copied from the last training month for all three, so
this says nothing about how pi should move. It isolates theta.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def block_series(E, Xs, ws, model, Pi, tv, drift):
    """Freeze the fit and read off, per block and month, how many sequences
    carried each mutation and how many were in the block.

    Soft responsibilities are used rather than hard assignment: a sequence that
    is ambiguous between two blocks should contribute to both, and hardening it
    would put a spurious step into the series we are about to model."""
    T, K = len(Xs), int(model.alive.sum())
    num = np.zeros((K, T, model.V))
    den = np.zeros((K, T))
    for t, (X, w) in enumerate(zip(Xs, ws)):
        th, kk = model.theta(tv[t], drift)
        lp = E.loglik_matrix(X, th) + np.log(Pi[t] + EPS)[None, :]
        mx = lp.max(1, keepdims=True)
        R = np.exp(lp - mx); R /= np.maximum(R.sum(1, keepdims=True), EPS)
        wr = w[:, None] * R                       # N x K
        den[:, t] = wr.sum(0)
        num[:, t, :] = wr.T @ X
    return num, den


# ---------------------------------------------------------------- time models
def fit_constant(num, den):
    p = num.sum(1) / np.maximum(den.sum(1)[:, None], EPS)
    return np.clip(p, 1e-4, 1 - 1e-4)


def fit_slope(num, den, t, l2=1e-2, iters=25):
    """Binomial logistic regression of each series on month index, by Newton.

    Vectorised over all (block, mutation) series at once. The ridge term keeps
    a series that is all-zero-then-all-one from sending gamma to infinity --
    without it, separated series produce the runaway slopes that make the
    current model extrapolate absurdly."""
    K, T, V = num.shape
    y = num.reshape(K * V, T) if False else num.transpose(0, 2, 1).reshape(-1, T)
    n = np.repeat(den[:, None, :], V, axis=1).reshape(-1, T)
    b = np.zeros(len(y)); g = np.zeros(len(y))
    tc = t - t.mean()
    for _ in range(iters):
        z = b[:, None] + g[:, None] * tc[None, :]
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        r = y - n * p
        gb = r.sum(1) - l2 * b
        gg = (r * tc[None, :]).sum(1) - l2 * g
        wgt = n * p * (1 - p)
        h11 = wgt.sum(1) + l2
        h12 = (wgt * tc[None, :]).sum(1)
        h22 = (wgt * tc[None, :] ** 2).sum(1) + l2
        detr = h11 * h22 - h12 ** 2
        detr = np.where(np.abs(detr) < 1e-9, 1e-9, detr)
        b += (h22 * gb - h12 * gg) / detr
        g += (h11 * gg - h12 * gb) / detr
        b = np.clip(b, -30, 30); g = np.clip(g, -30, 30)
    return b, g, tc


def slope_predict(b, g, tc, dt, h, shape):
    z = b + g * (tc[-1] + h * dt)
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), 1e-4,
                   1 - 1e-4).reshape(shape)


def fit_chain(num, den, S=2, iters=40, seed=0):
    """Hidden state per month, shared transition matrix, per-series posterior.

    States are levels of mutation frequency. The transition matrix and the
    per-state emission levels are shared across every (block, mutation) series
    -- twelve points cannot support a private transition matrix, and the thing
    worth estimating from all series jointly is how readily a position flips at
    all. What stays private to a series is which state it is in."""
    K, T, V = num.shape
    y = num.transpose(0, 2, 1).reshape(-1, T)
    n = np.repeat(den[:, None, :], V, axis=1).reshape(-1, T)
    M = len(y)
    e = np.linspace(0.02, 0.95, S)
    A = np.full((S, S), 0.1 / max(S - 1, 1)); np.fill_diagonal(A, 0.9)
    p0 = np.full(S, 1.0 / S)

    for _ in range(iters):
        le = (y[:, :, None] * np.log(e)[None, None, :]
              + (n - y)[:, :, None] * np.log(1 - e)[None, None, :])   # M,T,S
        le = np.clip(le, -1e6, 1e6)
        # forward
        al = np.zeros((M, T, S)); sc = np.zeros((M, T))
        a = np.log(p0)[None, :] + le[:, 0, :]
        mx = a.max(1, keepdims=True); a = np.exp(a - mx)
        sc[:, 0] = mx.ravel() + np.log(a.sum(1) + EPS)
        al[:, 0, :] = a / np.maximum(a.sum(1, keepdims=True), EPS)
        for t in range(1, T):
            a = (al[:, t - 1, :] @ A) * np.exp(
                np.clip(le[:, t, :] - le[:, t, :].max(1, keepdims=True), -700, 0))
            s = a.sum(1, keepdims=True)
            al[:, t, :] = a / np.maximum(s, EPS)
            sc[:, t] = le[:, t, :].max(1) + np.log(s.ravel() + EPS)
        # backward
        be = np.ones((M, T, S))
        for t in range(T - 2, -1, -1):
            em = np.exp(np.clip(le[:, t + 1, :] - le[:, t + 1, :].max(1, keepdims=True),
                                -700, 0))
            b_ = (be[:, t + 1, :] * em) @ A.T
            be[:, t, :] = b_ / np.maximum(b_.sum(1, keepdims=True), EPS)
        gam = al * be
        gam /= np.maximum(gam.sum(2, keepdims=True), EPS)
        # transitions
        xi = np.zeros((S, S))
        for t in range(T - 1):
            em = np.exp(np.clip(le[:, t + 1, :] - le[:, t + 1, :].max(1, keepdims=True),
                                -700, 0))
            x = al[:, t, :, None] * A[None, :, :] * (be[:, t + 1, :] * em)[:, None, :]
            xi += x.sum(0) / max(x.sum(), EPS) * x.sum()
        A = (xi + 1e-3) / (xi.sum(1, keepdims=True) + 1e-3 * S)
        p0 = gam[:, 0, :].mean(0) + 1e-6; p0 /= p0.sum()
        e = ((gam * y[:, :, None]).sum((0, 1)) + .5) / \
            ((gam * n[:, :, None]).sum((0, 1)) + 1.0)
        e = np.clip(np.sort(e), 1e-4, 1 - 1e-4)
    return gam[:, -1, :], A, e


def chain_predict(post, A, e, h, shape):
    P = np.linalg.matrix_power(A, max(int(h), 1))
    return np.clip((post @ P) @ e, 1e-4, 1 - 1e-4).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--horizons", default="1,2,3,6")
    ap.add_argument("--max-K", type=int, default=48)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--states", default="2,3")
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()

    E = load_engine(args.engine)
    names, V = E.load_names(args.vocab)
    tr = E.months_in_range(args.train)
    recs = [E.load_month(args.data_dir, ym) for ym in tr]
    if any(r is None for r in recs):
        sys.exit("missing training months")
    Xs, ws = zip(*[E.build(r, V, args.min_count) for r in recs])
    Xs, ws = list(Xs), list(ws)
    hs = [int(x) for x in args.horizons.split(",")]

    te = {}
    for h in hs:
        ym = E.ym_add(tr[-1], h)
        r = E.load_month(args.data_dir, ym)
        if r is None:
            print(f"  test month {ym} (h={h}) missing, skipped"); continue
        te[h] = (ym,) + E.build(r, V, 1)
    print(f"train {tr[0]}..{tr[-1]}   " +
          "   ".join(f"h={h}:{te[h][0]}" for h in sorted(te)))

    # ---- ONE fit. Blocks and weights frozen from here on.
    _, _, _, _, _, w0 = E.fit_flat(Xs, ws, V, args.max_K, seed=args.seed,
                                   drift=False, split_merge=False)
    _, pi_sm, _, _, _, beta_sm = E.fit_flat(
        Xs, ws, V, args.max_K, seed=args.seed, drift=True, split_merge=True,
        init_beta=w0)
    model, Pi, tv, _, _, _ = E.fit(
        Xs, ws, V, args.max_K, seed=args.seed, drift=True, names=names,
        verbose=False, iters=args.iters, K_warm=args.max_K, birth_every=1,
        births_per_call=4, refit=0, penalty="prior", warm=(beta_sm, pi_sm),
        warm_mode="tree", hier_drift=True, rescan_every=25)
    K = int(model.alive.sum())
    dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
    print(f"one fit, {K} blocks, frozen. Only the time model varies below.\n")

    num, den = block_series(E, Xs, ws, model, Pi, tv, True)
    shape = (K, V)
    pi_last = Pi[-1]

    res = {}
    res["constant"] = {h: fit_constant(num, den) for h in te}
    b, g, tc = fit_slope(num, den, np.arange(len(tv), dtype=float))
    res["slope"] = {h: slope_predict(b, g, tc, 1.0, h, shape) for h in te}
    for S in (int(x) for x in args.states.split(",")):
        post, A, e = fit_chain(num, den, S=S)
        res[f"chain, {S} states"] = {h: chain_predict(post, A, e, h, shape)
                                     for h in te}
        stay = float(np.diag(A).mean())
        print(f"  chain with {S} states: levels "
              f"{np.array2string(e, precision=3)}   mean stay probability "
              f"{stay:.3f}")

    print(f"\n{'=' * 80}\n  TIME MODEL FOR THETA   {K} blocks, frozen\n{'=' * 80}")
    hdr = "".join(f"{'h=' + str(h):>12}" for h in sorted(te))
    print(f"\n  {'time model':<20}{hdr}")
    print(f"  {'':<20}" + "".join(f"{te[h][0]:>12}" for h in sorted(te)))
    out = {}
    for nm, d in res.items():
        row = []
        for h in sorted(te):
            _, Xte, wte = te[h]
            row.append(E.score(Xte, wte, d[h], pi_last))
        out[nm] = row
        print(f"  {nm:<20}" + "".join(f"{x:>12.3f}" for x in row))
    print(f"\n  gain over constant")
    for nm, row in out.items():
        if nm == "constant": continue
        print(f"  {nm:<20}" + "".join(
            f"{a - b_:>+12.3f}" for a, b_ in zip(row, out["constant"])))
    print("""
  Same blocks, same memberships, same mixture weights for every row. The only
  difference is how theta is carried from the last training month to the test
  month, so the mixture's own initialisation noise cancels.

  chain beats slope, and its advantage grows with h
      -> the slope was the wrong shape. It cannot plateau, so the further out
         it is carried the further it runs past sweeps that already finished.
  slope beats chain
      -> the motion really is steady in logit space and a two- or three-level
         state is too coarse to capture it.
  neither beats constant at large h
      -> theta has no usable time model at this horizon and the loss is
         elsewhere: in the mixture weights, or in blocks that do not exist yet.
""")


if __name__ == "__main__":
    sys.exit(main())
