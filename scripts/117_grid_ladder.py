#!/usr/bin/env python
"""
117_grid_ladder.py -- block structure and time model are separate questions,
so vary them separately.

The old ladder confounded them. Its rungs were "flat", "flat + drift",
"flat + drift + split-merge", "hierarchy" -- so the step from rung 1 to rung 2
changed the time model while every later step changed the block structure with
the time model held at the slope. Since the slope turns out to be the wrong
shape past one month, every later rung was measured under a handicap.

This runs the grid instead:

    block structure  x  time model
    ----------------    ----------
    flat                constant   theta does not move
    flat + split-merge  slope      sig(beta + gamma t), what the model does now
    hierarchy           chain      a hidden state per month, shared transitions

Three fits, not nine. Each block structure is fitted ONCE and then frozen; the
three time models are read off the frozen monthly series. So a difference down
a column is the block structure, a difference across a row is the time model,
and the mixture's initialisation noise cancels within each row.

The mixture weights are copied from the last training month everywhere, so
nothing here speaks to how pi should move.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def series_from(E, Xs, ws, theta_at, Pi, V):
    """Per block and month: weighted count carrying each mutation, and the
    weight of the block. theta_at(t) returns that month's emission table."""
    T = len(Xs)
    K = Pi.shape[1]
    num = np.zeros((K, T, V)); den = np.zeros((K, T))
    for t, (X, w) in enumerate(zip(Xs, ws)):
        th = theta_at(t)
        lp = E.loglik_matrix(X, th) + np.log(Pi[t] + EPS)[None, :]
        mx = lp.max(1, keepdims=True)
        R = np.exp(lp - mx); R /= np.maximum(R.sum(1, keepdims=True), EPS)
        wr = w[:, None] * R
        den[:, t] = wr.sum(0)
        num[:, t, :] = wr.T @ X
    return num, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--timemodels",
                    default="scripts/116_theta_time_model.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--horizons", default="1,2,3,6")
    ap.add_argument("--max-K", type=int, default=48)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()

    E = load(args.engine, "engine")
    TM = load(args.timemodels, "timemodels")
    names, V = E.load_names(args.vocab)
    tr = E.months_in_range(args.train)
    recs = [E.load_month(args.data_dir, ym) for ym in tr]
    if any(r is None for r in recs):
        sys.exit("missing training months")
    Xs, ws = zip(*[E.build(r, V, args.min_count) for r in recs])
    Xs, ws = list(Xs), list(ws)

    te = {}
    for h in (int(x) for x in args.horizons.split(",")):
        ym = E.ym_add(tr[-1], h)
        r = E.load_month(args.data_dir, ym)
        if r is None:
            print(f"  h={h} ({ym}) missing, skipped"); continue
        te[h] = (ym,) + E.build(r, V, 1)
    hs = sorted(te)
    print(f"train {tr[0]}..{tr[-1]}   " + "   ".join(
        f"h={h}:{te[h][0]}" for h in hs))
    print(f"max-K {args.max_K}   min-count {args.min_count}   seed {args.seed}\n")

    fits = []

    # ---- flat, and flat + split-merge. Both fitted with the slope, because
    #      that is how they are fitted today; the time model is swapped after.
    _, _, _, _, _, w0 = E.fit_flat(Xs, ws, V, args.max_K, seed=args.seed,
                                   drift=False, split_merge=False)
    for label, sm in (("flat", False), ("flat + split-merge", True)):
        r = E.fit_flat(Xs, ws, V, args.max_K, seed=args.seed, drift=True,
                       split_merge=sm, names=names, verbose=False,
                       init_beta=w0, return_pi=True)
        beta, gamma, tv, Pi = r[5], r[6], r[7], r[8]
        occ = np.flatnonzero(Pi.max(0) > 1e-3)
        b, g, P = beta[occ], gamma[occ], Pi[:, occ]
        P = P / np.maximum(P.sum(1, keepdims=True), EPS)
        th_at = lambda t, b=b, g=g: np.clip(E.sig(b + g * tv[t]), 1e-4, 1 - 1e-4)
        fits.append((label, len(occ), th_at, P, tv))
        if sm:
            beta_sm, pi_sm = beta, Pi[-1]
        print(f"  fitted {label:<22} {len(occ)} blocks", flush=True)

    # ---- hierarchy
    model, Pi, tv, _, _, _ = E.fit(
        Xs, ws, V, args.max_K, seed=args.seed, drift=True, names=names,
        verbose=False, iters=args.iters, K_warm=args.max_K, birth_every=1,
        births_per_call=4, refit=0, penalty="prior", warm=(beta_sm, pi_sm),
        warm_mode="tree", hier_drift=True, rescan_every=25)
    th_at = lambda t, m=model: m.theta(tv[t], True)[0]
    fits.append(("hierarchy + birth-death", int(model.alive.sum()),
                 th_at, Pi, tv))
    print(f"  fitted {'hierarchy + birth-death':<22} "
          f"{int(model.alive.sum())} blocks\n", flush=True)

    # ---- read the three time models off each frozen fit
    out = {}
    for label, K, th_at, P, tvv in fits:
        num, den = series_from(E, Xs, ws, th_at, P, V)
        shape = (K, V)
        dt = 1.0
        pred = {}
        c = TM.fit_constant(num, den)
        pred["constant"] = {h: c for h in hs}
        b_, g_, tc = TM.fit_slope(num, den, np.arange(len(tvv), dtype=float))
        pred["slope"] = {h: TM.slope_predict(b_, g_, tc, dt, h, shape)
                         for h in hs}
        post, A, e = TM.fit_chain(num, den, S=args.states)
        pred[f"chain ({args.states} states)"] = {
            h: TM.chain_predict(post, A, e, h, shape) for h in hs}
        for nm, d in pred.items():
            out[(label, nm)] = [E.score(te[h][1], te[h][2], d[h], P[-1])
                                for h in hs]
        print(f"  {label:<24} chain levels "
              f"{np.array2string(e, precision=3)}  stay {np.diag(A).mean():.3f}",
              flush=True)

    tms = ["constant", "slope", f"chain ({args.states} states)"]
    print(f"\n{'=' * 92}\n  BLOCK STRUCTURE x TIME MODEL   held-out per sequence"
          f"\n{'=' * 92}")
    hdr = "".join(f"{'h=' + str(h):>12}" for h in hs)
    print(f"\n  {'':<28}{'':<20}{hdr}")
    print(f"  {'':<28}{'':<20}" + "".join(f"{te[h][0]:>12}" for h in hs))
    for label, K, *_ in fits:
        for nm in tms:
            row = out[(label, nm)]
            tag = f"{label} ({K})" if nm == tms[0] else ""
            print(f"  {tag:<28}{nm:<20}" + "".join(f"{x:>12.3f}" for x in row))
        print()

    print(f"  gain over the same blocks with theta held constant")
    print(f"\n  {'':<28}{'':<20}{hdr}")
    for label, K, *_ in fits:
        base = out[(label, "constant")]
        for nm in tms[1:]:
            row = out[(label, nm)]
            tag = f"{label} ({K})" if nm == tms[1] else ""
            print(f"  {tag:<28}{nm:<20}" + "".join(
                f"{a - b:>+12.3f}" for a, b in zip(row, base)))
        print()
    print("""
  Down a column: what the block structure buys, at a fixed time model.
  Across a row: what the time model buys, on fixed blocks.

  The old ladder could not separate these. It changed the time model once, at
  the second rung, then held it at the slope while the block structure changed
  -- so every later rung was measured with a time model that turns out to be
  the wrong shape past one month.
""")


if __name__ == "__main__":
    sys.exit(main())
