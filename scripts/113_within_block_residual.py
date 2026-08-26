#!/usr/bin/env python
"""
113_within_block_residual.py -- does splitting clean up the dependence, or is
there something product-Bernoulli cannot represent?

THE QUESTION IN PLAIN ENGLISH
-----------------------------
The model says: pick a background, then each mutation is an independent coin.
Two mutations co-occur because they share a background, not because one makes
the other more likely.

That can only be wrong in one specific way. Inside a SINGLE background, two
mutations sit at middling frequency and still track each other. Independent
coins cannot produce that. The only repair available to the current model is
to split the background again -- so if splitting has stopped paying and the
correlation is still there, the emission family is the problem.

WHAT IS MEASURED
----------------
For every block, restricted to its middling-frequency mutations (0.05 < p <
0.95, since a mutation that is always on or always off carries no information
about anything):

    observed co-occurrence  minus  what independence predicts
    -------------------------------------------------------
                  sqrt(expected)

a Pearson residual. Under the model's own assumption these are noise, so |R|
of order 1. A pair at |R| = 40 is not noise.

Reported per K, so you can see whether the residuals die as blocks are added.

HOW TO READ IT
--------------
  residuals fall toward noise as K grows
      -> dependence is being absorbed into discrete backgrounds. The emission
         family is fine. Do NOT add pairwise terms.

  a large residual survives in a block that is well populated and stable,
  at a K where splitting has stopped buying held-out likelihood
      -> that pair is the thing product-Bernoulli cannot represent, and it is
         the evidence that would justify changing the emission.

The second outcome needs BOTH halves. A big residual in a block that is still
being split is not evidence of anything -- the model has not finished.

The null column matters. Finite samples produce nonzero residuals even when
the model is exactly true, so each block is also scored against data generated
FROM ITS OWN FITTED PROFILE at its own size. Compare against that, not
against 1.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def block_residuals(E, Xs, ws, model, Pi, tv, drift, k, lo=.05, hi=.95,
                    min_den=200.0):
    """Pearson residuals of the within-block co-occurrence matrix.

    Returns (residual matrix, the mutation indices it covers, block weight,
    per-mutation frequency). Sequences are assigned to their most likely block
    -- a hard assignment, which is the conservative choice: soft weights would
    blur two blocks together and manufacture correlation that is really just
    mixing."""
    num = np.zeros(model.V); den = 0.0; Co = None
    for t, (X, w) in enumerate(zip(Xs, ws)):
        th, kk = model.theta(tv[t], drift)
        lp = E.loglik_matrix(X, th) + np.log(Pi[t] + EPS)[None, :]
        m = kk[lp.argmax(1)] == k
        if not m.any():
            continue
        Xk, wk = X[m], w[m]
        num += (wk[:, None] * Xk).sum(0); den += wk.sum()
        C = (Xk * wk[:, None]).T @ Xk
        Co = C if Co is None else Co + C
    if den < min_den or Co is None:
        return None, None, den, None
    p = num / den
    var = np.flatnonzero((p > lo) & (p < hi))
    if len(var) < 2:
        return None, var, den, p
    pv = p[var]
    Exp = den * np.outer(pv, pv)
    R = (Co[np.ix_(var, var)] - Exp) / np.sqrt(Exp + 1.0)
    np.fill_diagonal(R, 0.0)
    return R, var, den, p


def null_residual(rng, pv, den, reps=3):
    """Largest |R| you get from data that really is independent coins, at this
    block's size and frequencies. Anything at or below this is finite-sample
    noise, not evidence of interaction."""
    n = max(int(round(den)), 2)
    n = min(n, 60000)                      # cap the simulation, not the claim
    scale = den / n
    best = 0.0
    for _ in range(reps):
        Y = (rng.random((n, len(pv))) < pv[None, :]).astype(np.float32)
        Co = (Y.T @ Y) * scale
        Exp = den * np.outer(pv, pv)
        R = (Co - Exp) / np.sqrt(Exp + 1.0)
        np.fill_diagonal(R, 0.0)
        best = max(best, float(np.abs(R).max()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True,
                    help="one month, used only to check whether splitting is "
                         "still buying anything at each K")
    ap.add_argument("--K-list", default="13,24,48,96",
                    help="fit at each of these and report residuals per K")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--half-life", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--top", type=int, default=8,
                    help="worst offending pairs to print per K")
    ap.add_argument("--min-block-weight", type=float, default=2000.0,
                    help="ignore blocks lighter than this; a residual in a "
                         "tiny block is not evidence about the emission")
    args = ap.parse_args()

    E = load_engine(args.engine)
    names, V = E.load_names(args.vocab)
    tr = E.months_in_range(args.train)
    recs = [E.load_month(args.data_dir, ym) for ym in tr]
    if any(r is None for r in recs):
        sys.exit("missing training months")
    Xs, ws = zip(*[E.build(r, V, args.min_count) for r in recs])
    Xs, ws = list(Xs), list(ws)
    rte = E.load_month(args.data_dir, args.test)
    if rte is None:
        sys.exit(f"missing test month {args.test}")
    Xte, wte = E.build(rte, V, 1)
    rng = np.random.default_rng(args.seed)

    print(f"train {tr[0]}..{tr[-1]}   test {args.test}   V={V:,}   "
          f"min-count {args.min_count}")
    Ks = [int(x) for x in args.K_list.split(",")]
    summary = []

    for K in Ks:
        _, _, _, _, _, w0 = E.fit_flat(Xs, ws, V, K, seed=args.seed,
                                       drift=False, split_merge=False)
        _, pi_sm, _, _, _, beta_sm = E.fit_flat(
            Xs, ws, V, K, seed=args.seed, drift=True, split_merge=True,
            half_life=args.half_life, init_beta=w0)
        model, Pi, tv, births, _, _ = E.fit(
            Xs, ws, V, K, seed=args.seed, sigma=args.sigma,
            half_life=args.half_life, drift=True, names=names, verbose=False,
            iters=args.iters, K_warm=K, birth_every=1, births_per_call=4,
            refit=0, penalty="prior", warm=(beta_sm, pi_sm), warm_mode="tree",
            hier_drift=True, rescan_every=25)
        dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
        th, ks = model.theta(tv[-1] + dt, True)
        held = E.score(Xte, wte, th, Pi[-1])
        Kused = int(model.alive.sum())

        rows = []
        for k in np.flatnonzero(model.alive):
            k = int(k)
            R, var, den, p = block_residuals(E, Xs, ws, model, Pi, tv, True, k)
            if R is None or den < args.min_block_weight:
                continue
            nl = null_residual(rng, p[var], den)
            i, j = np.unravel_index(np.abs(R).argmax(), R.shape)
            rows.append(dict(k=k, den=den, nvar=len(var),
                             mx=float(np.abs(R).max()), null=nl,
                             ratio=float(np.abs(R).max()) / max(nl, 1e-9),
                             a=int(var[i]), b=int(var[j])))
        rows.sort(key=lambda r: -r["ratio"])

        print(f"\n{'='*84}\nK requested {K}   occupied {Kused}   "
              f"held-out {held:.3f}   blocks examined {len(rows)}\n{'='*84}")
        print(f"  {'block':>6}{'weight':>12}{'mid-freq':>10}"
              f"{'max|R|':>10}{'null':>9}{'ratio':>8}   worst pair")
        for r in rows[:args.top]:
            pair = f"{names.get(r['a'], r['a'])} + {names.get(r['b'], r['b'])}"
            print(f"  {r['k']:>6}{r['den']:>12,.0f}{r['nvar']:>10}"
                  f"{r['mx']:>10.1f}{r['null']:>9.1f}{r['ratio']:>8.1f}   {pair}")
        if rows:
            summary.append(dict(K=K, Kused=Kused, held=held,
                                worst=rows[0]["ratio"],
                                med=float(np.median([r["ratio"] for r in rows])),
                                pair=(rows[0]["a"], rows[0]["b"])))

    print(f"\n{'='*84}\n  DOES SPLITTING CLEAN THE RESIDUALS?\n{'='*84}")
    print(f"\n  {'K':>5}{'occupied':>10}{'held-out':>11}"
          f"{'worst ratio':>14}{'median ratio':>14}   worst pair")
    for s in summary:
        pair = f"{names.get(s['pair'][0])} + {names.get(s['pair'][1])}"
        print(f"  {s['K']:>5}{s['Kused']:>10}{s['held']:>11.3f}"
              f"{s['worst']:>14.1f}{s['med']:>14.1f}   {pair}")
    print("""
  ratio = max|R| in the block, divided by the largest |R| that independent
  coins of the same size and frequencies produce. A ratio near 1 IS the model
  being right. A ratio of 20 is a pair the emission cannot represent.

  Falling ratios with K, while held-out still improves
      -> splitting is doing the job. Product-Bernoulli is adequate.
         Do not add pairwise terms.

  Ratio stuck high in a heavy block, at a K where held-out has stopped
  improving
      -> splitting has run out and the correlation is still there. That pair,
         in that block, is the evidence for changing the emission family.

  Both halves are required. A high ratio at a K where held-out is still rising
  only means the model has not finished splitting yet.
""")


if __name__ == "__main__":
    sys.exit(main())
