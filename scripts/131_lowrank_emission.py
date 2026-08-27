#!/usr/bin/env python
"""
131_lowrank_emission.py -- put the blocks on a continuous surface.

THE PROBLEM
-----------
Each block currently owns a free number for every mutation: forty-eight
isolated points in a space of over a thousand dimensions, with nothing in
between them. A set that differs from its block at three positions is charged
about nine nats for each, because the block asserts near-certainty there and
no other block is any closer. Six months out, three quarters of sequences are
combinations never seen before, and the model scores them worse than a table
that assigns one flat constant to everything it does not recognise.

Softening the emission does not fix this. Smoothing every entry during fitting
buys a few nats and costs accuracy on familiar sequences; a background
component buys almost nothing; capping the per-position cost makes things
worse at every setting. All of those blur the points. None of them creates
anything between the points.

THE CHANGE
----------
    now:        theta[k,v] = sigmoid( free number per (block, mutation) )
    low rank:   theta[k,v] = sigmoid( u_k . phi_v )

Every mutation gets a vector phi_v, every block a vector u_k, both learned. The
profiles a block can express are then a D-dimensional surface rather than a set
of isolated corners, and two blocks that use similar mutations end up with
similar u. Move u a little and you get a coherent profile no block currently
holds -- which is what "the unseen combination is nearby" has to mean for a
model that scores sets.

Nothing else changes. The emission is still a product of Bernoullis, so P(S)
is still exact and the numbers stay comparable with every other model here.
This is the emission of an embedded topic model, and the reason to expect it
to help is not parameter count -- it is that mutations stop being unrelated
coordinates.

WHAT IS COMPARED
----------------
The same K, the same data, the same warm start, fitted two ways, scored on
seen and unseen sets separately. Rank is swept, because the whole question is
how much sharing across mutations is worth.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_lowrank(Sd, N, U, P, b, iters, lr, l2, T):
    """Gradient ascent on the expected complete-data log-likelihood with
    theta[k,v] = sigmoid(u_k . phi_v + b_v).

    Sd[t] is (K,V): the weight of sequences in block k carrying mutation v in
    month t. N[t] is (K,): the weight in block k. The residual Sd - N*theta is
    the same object the full-rank M-step uses; the only difference is that it
    is projected onto U and P instead of being written straight into a free
    parameter.

    b is a per-mutation intercept, kept out of the low-rank part so that a
    mutation's overall prevalence does not have to be spent on a rank
    dimension."""
    def obj(U, P, b):
        """Expected complete-data log-likelihood. Monitored rather than
        assumed: with logits reaching plus or minus nine a fixed step size
        overshoots and the parameters run to infinity, so the step is halved
        whenever it fails to improve."""
        v = 0.0
        z = U @ P.T + b[None, :]
        th = np.clip(sig(z), 1e-6, 1 - 1e-6)
        lt = np.log(th); lm = np.log(1 - th)
        for t in range(T):
            v += float((Sd[t] * lt).sum()
                       + ((N[t][:, None] - Sd[t]) * lm).sum())
        return v - 0.5 * l2 * (float((U * U).sum()) + float((P * P).sum()))

    tot = max(sum(float(N[t].sum()) for t in range(T)), 1.0)
    cur0 = obj(U, P, b) / tot
    cur = obj(U, P, b)
    nacc = 0
    for _ in range(iters):
        gU = np.zeros_like(U); gP = np.zeros_like(P); gb = np.zeros_like(b)
        th = sig(U @ P.T + b[None, :])
        for t in range(T):
            R = Sd[t] - N[t][:, None] * th
            gU += R @ P
            gP += R.T @ U
            gb += R.sum(0)
        gU = gU / tot - l2 * U
        gP = gP / tot - l2 * P
        gb = gb / tot
        step = lr
        for _ in range(12):
            Un = U + step * gU; Pn = P + step * gP; bn = b + step * gb
            if np.all(np.isfinite(Un)) and np.all(np.isfinite(Pn)):
                new = obj(Un, Pn, bn)
                if np.isfinite(new) and new > cur:
                    U, P, b, cur = Un, Pn, bn, new
                    nacc += 1
                    break
            step *= 0.5
        else:
            break                     # no step improves; converged or stuck
    return U, P, b, cur0, cur / tot, nacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--horizons", default="1,6")
    ap.add_argument("--ranks", default="8,16,32,64")
    ap.add_argument("--max-K", type=int, default=48)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr-iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--l2", type=float, default=1e-4)
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
        if r is not None:
            te[h] = (ym, r)
    print(f"train {tr[0]}..{tr[-1]}   " +
          "   ".join(f"h={h}:{te[h][0]}" for h in sorted(te)))

    # ---- the fitted model, unchanged. Its blocks and weights are the
    #      starting point, so the only thing that varies is the emission.
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
    ks = np.flatnonzero(model.alive); K = len(ks)
    dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
    print(f"{K} blocks, {V:,} mutations, "
          f"{K * V:,} free emission parameters as it stands\n")

    # responsibilities from the fitted model, held fixed while the emission
    # is refitted -- so a difference is the emission and not a different
    # partition of the data
    Sd, N = [], []
    for t, (X, w) in enumerate(zip(Xs, ws)):
        th, _ = model.theta(tv[t], True, ti=t)
        lp = E.loglik_matrix(X, th) + np.log(Pi[t] + EPS)[None, :]
        mx = lp.max(1, keepdims=True)
        R = np.exp(lp - mx); R /= np.maximum(R.sum(1, keepdims=True), EPS)
        Rw = R * w[:, None]
        Sd.append(Rw.T @ X); N.append(Rw.sum(0))
    T = len(Xs)

    def report(tag, th_fn, nparam):
        out = []
        for h in sorted(te):
            ym, rec = te[h]
            th = th_fn(h)
            for nm, part in (("seen", [(a, c) for a, c in rec
                                       if a in train_sets]),
                             ("unseen", [(a, c) for a, c in rec
                                         if a not in train_sets]),
                             ("all", list(rec))):
                if not part: continue
                Xp, wp = E.build(part, V, 1)
                out.append((h, nm, E.score(Xp, wp, th, Pi[-1])))
        return out

    base = report("full rank", lambda h: model.theta(
        tv[-1] + h * dt, True, ti=len(tv) - 1 + h)[0], K * V)
    rows = {("full rank", h, nm): s for h, nm, s in base}
    order = [(h, nm) for h, nm, _ in base]

    def train_obj(th):
        """Expected complete-data LL per observation under a given emission,
        with the responsibilities held fixed. This is the quantity the
        low-rank fit is maximising, so it is the only fair reference for
        judging whether the fit worked."""
        thc = np.clip(th, 1e-6, 1 - 1e-6)
        lt = np.log(thc); lm = np.log(1 - thc); v = 0.0; tot = 0.0
        for t in range(T):
            v += float((Sd[t] * lt).sum()
                       + ((N[t][:, None] - Sd[t]) * lm).sum())
            tot += float(N[t].sum())
        return v / max(tot, 1.0)

    th_fullrank, _ = model.theta(tv[-1], True, ti=T - 1)
    print(f"  reference: full-rank emission scores "
          f"{train_obj(th_fullrank):.4f} on the same objective\n")

    rng = np.random.default_rng(args.seed)
    th_last, _ = model.theta(tv[-1], True, ti=T - 1)
    lg = np.log(np.clip(th_last, 1e-4, 1 - 1e-4)
                / (1 - np.clip(th_last, 1e-4, 1 - 1e-4)))
    for D in (int(x) for x in args.ranks.split(",")):
        # start from the truncated SVD of the fitted logits, so the low-rank
        # model begins as the best rank-D approximation of what it replaces
        b = lg.mean(0)
        Um, S, Vt = np.linalg.svd(lg - b[None, :], full_matrices=False)
        U = Um[:, :D] * np.sqrt(S[:D])[None, :]
        P = (Vt[:D].T * np.sqrt(S[:D])[None, :])
        U, P, b, ll0, ll1, nacc = fit_lowrank(Sd, N, U, P, b, args.lr_iters,
                                              args.lr, args.l2, T)
        thL = np.clip(sig(U @ P.T + b[None, :]), 1e-4, 1 - 1e-4)
        r = report(f"rank {D}", lambda h, thL=thL: thL, K * D + V * D + V)
        for h, nm, s in r:
            rows[(f"rank {D}", h, nm)] = s
        print(f"  rank {D:>3}: {K*D + V*D + V:>8,} params   "
              f"train LL/obs at SVD init {ll0:9.4f}  after fitting {ll1:9.4f}"
              f"   steps accepted {nacc}/{args.lr_iters}", flush=True)

    variants = ["full rank"] + [f"rank {x}" for x in args.ranks.split(",")]
    print(f"\n{'=' * 78}\n  LOW-RANK EMISSION   {K} blocks, same "
          f"responsibilities\n{'=' * 78}")
    print(f"\n  {'variant':<12}" + "".join(
        f"{'h' + str(h) + ' ' + nm:>14}" for h, nm in order))
    for v in variants:
        print(f"  {v:<12}" + "".join(
            f"{rows[(v, h, nm)]:>14.3f}" for h, nm in order))
    print(f"\n  change from full rank")
    for v in variants[1:]:
        print(f"  {v:<12}" + "".join(
            f"{rows[(v, h, nm)] - rows[('full rank', h, nm)]:>+14.3f}"
            for h, nm in order))
    print("""
  The unseen columns are the point. A free number per (block, mutation) makes
  the blocks isolated corners in a space of a thousand dimensions, so a set
  three edits away has nothing near it. Sharing a low-dimensional structure
  across mutations puts the blocks on a surface, and a combination that is not
  any block can still be close to the surface.

  Read the SVD-init and after-fitting columns first. The truncated SVD of the
  fitted logits is where each rank STARTS, so if a high rank starts close to
  the full-rank reference and ends up far from it, the optimiser is at fault
  and the ranks say nothing about the model. If instead even a high rank
  starts far away, the fitted emission genuinely is not low rank and the idea
  is dead on this data.

  unseen improves while seen barely moves
      -> the surface exists and the isolated-corners geometry was the
         constraint. Worth building into the fitting rather than refitting the
         emission afterwards.
  both get worse as rank falls
      -> the profiles genuinely need their full dimensionality, and there is
         no low-dimensional structure to exploit.
  seen degrades and unseen improves
      -> the same trade the smoothing prior gives, and no better; the surface
         is not buying anything that blurring did not.
""")


if __name__ == "__main__":
    sys.exit(main())
