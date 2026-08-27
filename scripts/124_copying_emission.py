#!/usr/bin/env python
"""
124_copying_emission.py -- let a sequence be built from more than one block.

THE PROBLEM
-----------
The model assigns each sequence to ONE block, and all of its features are then
drawn from that block's profile. A set that carries part of one block's
signature and part of another's cannot be represented: whichever block wins the
responsibility charges full price for every feature belonging to the other.

That is exactly what an unseen combination of known features looks like, and
those are three quarters of the sequences six months out. The model scores them
about twenty nats below a table that assigns one flat constant to everything it
does not recognise, and the median such set sits only three edits from
something in the training data.

Every attempt so far to fix this by softening the emission has failed:
smoothing theta during fitting buys about three nats, a background component
buys almost nothing, and a per-position floor makes things worse at every
setting. All of those try to make ONE block tolerate a foreign feature. The
alternative is to let a DIFFERENT block explain it.

THE MODEL
---------
Walk along the positions. At each one, the sequence is copying from some block;
usually the same block as at the previous position, occasionally a new one:

    z_1 ~ pi
    z_v = z_{v-1} with probability 1-s,  otherwise drawn afresh from pi
    S_v ~ Bernoulli( theta[z_v, v] )

At s = 0 this is exactly the current model -- one block for the whole sequence.
As s rises, a sequence may be mostly block A with a stretch of block B, and it
pays a switch cost rather than a mismatch cost.

This is the copying model of Li and Stephens (2003), which is the standard way
recombination is handled in population genetics. P(S) stays exact: it is a sum
over block paths, computed by the forward algorithm, so held-out numbers remain
comparable with every other model in this project.

WHAT THIS RUN DOES
------------------
It fits nothing new. The blocks and their profiles come from the existing fit;
only the way a sequence is assembled from them changes. So a difference here is
attributable to the copying structure and not to a different model.

If it helps, the principled version is to fit with it, so the blocks form
knowing that sequences may be assembled from several of them.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tree_transition(model, ks, pi, s, lam, mode="tree"):
    """Where a switch is allowed to go.

    mode='flat'  a switch resamples the block from pi, so moving to a sibling
                 costs exactly what moving to an unrelated block costs. That
                 throws away the one thing the hierarchy knows.
    mode='tree'  a switch prefers nearby blocks:  A[k,l] ~ pi_l exp(-lam d(k,l))
                 with d the number of tree edges between them. Siblings differ
                 by a handful of features, so a short hop should be cheap and a
                 hop across the tree should not. lam = 0 recovers 'flat'.

    Returns a K x K row-stochastic matrix of WHERE a switch lands; whether a
    switch happens at all is still governed by s."""
    K = len(ks)
    if mode == "flat" or lam <= 0:
        return np.tile(pi[None, :], (K, 1))
    idx = {int(k): i for i, k in enumerate(ks)}

    def path(k):
        out = []; j = int(k)
        while j >= 0:
            out.append(j); j = int(model.parent[j])
        return out

    P = [path(k) for k in ks]
    D = np.zeros((K, K))
    for i in range(K):
        pi_set = {n: d for d, n in enumerate(P[i])}
        for j in range(K):
            best = None
            for d2, n in enumerate(P[j]):
                if n in pi_set:
                    best = pi_set[n] + d2
                    break
            D[i, j] = best if best is not None else 2 * max(len(P[i]), len(P[j]))
    A = pi[None, :] * np.exp(-lam * D)
    return A / np.maximum(A.sum(1, keepdims=True), EPS)


def copy_score(X, w, th, pi, s, A=None, chunk=4000):
    """Mean log P(set) under the copying model, by the forward algorithm.

    At s = 0 the recursion never leaves its starting block and this returns
    exactly the ordinary mixture likelihood, which is the check that the
    implementation is right."""
    N, V = X.shape
    K = th.shape[0]
    logth = np.log(np.clip(th, 1e-9, 1 - 1e-9))
    log1m = np.log(np.clip(1 - th, 1e-9, 1 - 1e-9))
    out = np.zeros(N)
    for a in range(0, N, chunk):
        Xc = X[a:a + chunk]
        n = len(Xc)
        # position 0
        e = np.where(Xc[:, 0:1] > 0, logth[:, 0][None, :], log1m[:, 0][None, :])
        al = np.log(pi + EPS)[None, :] + e
        mx = al.max(1, keepdims=True)
        al = np.exp(al - mx)
        ll = mx.ravel() + np.log(al.sum(1) + EPS)
        al /= np.maximum(al.sum(1, keepdims=True), EPS)
        for v in range(1, V):
            # stay in the same block, or switch to another one. Where a switch
            # lands is A: uniform over pi if flat, tree-weighted if not.
            al = (1.0 - s) * al + s * (al @ A if A is not None
                                       else pi[None, :])
            e = np.where(Xc[:, v:v + 1] > 0, th[:, v][None, :],
                         1.0 - th[:, v][None, :])
            al = al * e
            z = al.sum(1, keepdims=True)
            ll += np.log(z.ravel() + EPS)
            al /= np.maximum(z, EPS)
        out[a:a + chunk] = ll
    return float((w * out).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--horizons", default="1,6")
    ap.add_argument("--lam", default="",
                    help="tree distance penalty on where a switch lands. Empty "
                         "or 0 means a switch resamples from pi regardless of "
                         "how far it is. Comma-separated to sweep")
    ap.add_argument("--switch", default="0,0.001,0.01,0.05,0.2",
                    help="probability of switching block at a position. 0 is "
                         "the current model")
    ap.add_argument("--max-K", type=int, default=48)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
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
    filt = [[(s_, c) for s_, c in r if c >= args.min_count] for r in recs]
    train_sets = {s_ for r in filt for s_, _ in r}

    te = {}
    for h in (int(x) for x in args.horizons.split(",")):
        ym = E.ym_add(tr[-1], h)
        r = E.load_month(args.data_dir, ym)
        if r is None:
            print(f"  h={h} ({ym}) missing"); continue
        te[h] = (ym, r)
    print(f"train {tr[0]}..{tr[-1]}   " +
          "   ".join(f"h={h}:{te[h][0]}" for h in sorted(te)))

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
    dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
    print(f"one fit, {int(model.alive.sum())} blocks. Only the way a sequence "
          f"is assembled from them changes below.\n")

    ss = [float(x) for x in args.switch.split(",")]
    lams = [float(x) for x in args.lam.split(",")] if args.lam else [0.0]
    ks_alive = np.flatnonzero(model.alive)
    for h in sorted(te):
        ym, rec = te[h]
        th, ks = model.theta(tv[-1] + h * dt, True, ti=len(tv) - 1 + h)
        pi = Pi[-1]
        parts = {}
        for nm, part in (("seen", [(a, c) for a, c in rec if a in train_sets]),
                         ("unseen", [(a, c) for a, c in rec
                                     if a not in train_sets]),
                         ("all", list(rec))):
            if part:
                parts[nm] = E.build(part, V, 1)
        share = sum(c for a, c in rec if a not in train_sets) / \
            max(sum(c for _, c in rec), 1)
        print(f"{'=' * 66}\n  {ym}   h={h}   unseen share {100*share:.1f}%\n"
              f"{'=' * 66}")
        print(f"    {'lam':>6}{'switch':>8}" +
              "".join(f"{nm:>12}" for nm in parts))
        base = {}; best = {nm: -np.inf for nm in parts}
        for lam in lams:
            A = tree_transition(model, ks_alive, pi, 0.0, lam,
                                mode="flat" if lam <= 0 else "tree")
            for s in ss:
                row = []
                for nm, (Xp, wp) in parts.items():
                    sc = copy_score(Xp, wp, th, pi, s, A=A)
                    base.setdefault(nm, sc)
                    best[nm] = max(best[nm], sc)
                    row.append(sc)
                print(f"    {lam:>6.1f}{s:>8.3f}"
                      + "".join(f"{x:>12.3f}" for x in row), flush=True)
        print(f"    {'':>6}{'best':>8}" + "".join(
            f"{best[nm] - base[nm]:>+12.3f}" for nm in parts))
        print()

    print("""
  switch = 0 must reproduce the ordinary mixture exactly; if it does not, the
  forward recursion is wrong and nothing below it means anything.

  unseen improves as the switch rate rises
      -> novel sets really are assemblies of existing blocks, and the single-
         block assumption was the constraint. Fit with copying.
  unseen does not improve
      -> the novelty is not recombination of known blocks, and the deficit
         lies somewhere else again.

  lam > 0 beats lam = 0
      -> switches go to NEARBY blocks. The novelty is a short hop within a
         clade rather than a splice of unrelated lineages, and the tree
         already carried the information needed to score it.
  lam = 0 wins
      -> where a switch lands is not constrained by the tree, and the
         proximity the hierarchy encodes is not the proximity that matters
         here.
""")


if __name__ == "__main__":
    sys.exit(main())
