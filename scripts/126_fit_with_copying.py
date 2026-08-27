#!/usr/bin/env python
"""
126_fit_with_copying.py -- refit the blocks under the copying model.

WHY REFIT AT ALL
----------------
Scoring an existing fit with copying is worth about one and a half nats. But
those blocks were formed under the assumption that one block produces a whole
sequence, which forces the model to spend a separate block on every observed
combination of features. Under copying it does not have to: a combination can
be assembled from two blocks and a switch. So the blocks themselves should
come out differently -- fewer of them, each covering a stretch of positions
rather than a whole observed pattern.

This runs the ordinary fit first, then continues from it with copying turned
on, and reports both. Same data, same K, same starting point, so the
difference is the fitting assumption.

WHAT CHANGES IN THE E-STEP
--------------------------
Without copying, a sequence has one responsibility vector over blocks, and the
statistic the M-step needs is (responsibility) x (features present), summed
over sequences.

With copying, EVERY POSITION has its own distribution over blocks, obtained by
forward-backward along the sequence. So the denominator in the M-step becomes
position-dependent: block k is credited with position v in proportion to how
much of that position it was responsible for, not how much of the sequence.
That is the whole change, and it is why the fit can move.

P(S) stays exact, so the numbers remain comparable with everything else.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def copy_fb(X, w, th, pi, s, chunk=256):
    """Forward-backward along positions.

    Returns the per-position sufficient statistics the M-step needs --
    Sd[k,v] = expected weight of sequences copying position v from block k AND
    carrying it, Nd[k,v] = expected weight copying position v from block k at
    all -- plus the mean log-likelihood and the marginal block occupancy.

    The transition is 'stay, or resample from pi', so the backward recursion
    costs O(K) per position rather than O(K^2)."""
    N, V = X.shape
    K = th.shape[0]
    Sd = np.zeros((K, V)); Nd = np.zeros((K, V)); occ = np.zeros(K)
    ll = 0.0; tw = 0.0
    for a in range(0, N, chunk):
        Xc = X[a:a + chunk].astype(np.float32)
        wc = w[a:a + chunk]
        n = len(Xc)
        # emissions, (n, V, K); built once per chunk
        e = np.where(Xc[:, :, None] > 0, th.T[None, :, :],
                     (1.0 - th).T[None, :, :]).astype(np.float32)
        al = np.empty((n, V, K), dtype=np.float32)
        cur = pi[None, :].astype(np.float32) * e[:, 0, :]
        z = cur.sum(1, keepdims=True); lz = np.log(z.ravel() + EPS)
        cur /= np.maximum(z, EPS); al[:, 0, :] = cur
        for v in range(1, V):
            cur = ((1.0 - s) * cur + s * pi[None, :].astype(np.float32)) \
                * e[:, v, :]
            z = cur.sum(1, keepdims=True); lz = lz + np.log(z.ravel() + EPS)
            cur /= np.maximum(z, EPS); al[:, v, :] = cur
        ll += float((wc * lz).sum()); tw += float(wc.sum())
        be = np.ones((n, K), dtype=np.float32)
        for v in range(V - 1, -1, -1):
            g = al[:, v, :] * be
            g /= np.maximum(g.sum(1, keepdims=True), EPS)
            gw = g * wc[:, None]
            Nd[:, v] += gw.sum(0)
            Sd[:, v] += (gw * Xc[:, v:v + 1]).sum(0)
            occ += gw.sum(0)
            if v == 0:
                break
            eb = e[:, v, :] * be
            be = (1.0 - s) * eb + s * (eb @ pi.astype(np.float32))[:, None]
            be /= np.maximum(be.max(1, keepdims=True), EPS)
    return Sd, Nd, ll / max(tw, EPS), occ / max(occ.sum(), EPS)


def copy_score(X, w, th, pi, s, chunk=4000):
    N, V = X.shape
    out = np.zeros(N)
    for a in range(0, N, chunk):
        Xc = X[a:a + chunk]
        cur = pi[None, :] * np.where(Xc[:, 0:1] > 0, th[:, 0][None, :],
                                     1 - th[:, 0][None, :])
        z = cur.sum(1, keepdims=True); ll = np.log(z.ravel() + EPS)
        cur /= np.maximum(z, EPS)
        for v in range(1, V):
            cur = ((1.0 - s) * cur + s * pi[None, :]) * np.where(
                Xc[:, v:v + 1] > 0, th[:, v][None, :], 1 - th[:, v][None, :])
            z = cur.sum(1, keepdims=True); ll += np.log(z.ravel() + EPS)
            cur /= np.maximum(z, EPS)
        out[a:a + chunk] = ll
    return float((w * out).sum() / w.sum())


def refine(E, model, Pi, tv, Xs, ws, s, rounds, lr=1.0, verbose=True):
    """Continue fitting with copying. Blocks and tree are inherited; only the
    responsibilities, and therefore where each block's evidence comes from,
    change."""
    ks = np.flatnonzero(model.alive)
    T = len(Xs)
    for it in range(rounds):
        tot_ll = 0.0; tot_w = 0.0
        newPi = np.zeros_like(Pi)
        SdT = []; NdT = []
        for t, (X, w) in enumerate(zip(Xs, ws)):
            th, _ = model.theta(tv[t], True, ti=t)
            Sd, Nd, ll, occ = copy_fb(X, w, th, Pi[t], s)
            SdT.append(Sd); NdT.append(Nd)
            newPi[t] = occ
            tot_ll += ll * w.sum(); tot_w += w.sum()
        Pi = newPi / np.maximum(newPi.sum(1, keepdims=True), EPS)
        # gradient step on each node's deviation, accumulated up the tree as
        # in the ordinary M-step, but with a position-dependent denominator
        kidx = {int(k): i for i, k in enumerate(ks)}
        gb = np.zeros((len(ks), model.V))
        for t in range(T):
            th, _ = model.theta(tv[t], True, ti=t)
            gb += SdT[t] - NdT[t] * th
        order = sorted(range(len(ks)), key=lambda i: -model.depth(int(ks[i])))
        acc = gb.copy()
        for i in order:
            p = int(model.parent[int(ks[i])])
            if p in kidx:
                acc[kidx[p]] += acc[i]
        scale = float(sum(Nd.sum() for Nd in NdT)) / model.V
        for i, k in enumerate(ks):
            k = int(k)
            model.delta[k] += lr * (acc[i]
                                    - model.delta[k] / model.sigma ** 2) / max(scale, 1.0)
        if verbose:
            print(f"      copy-refine {it+1}/{rounds}  LL/obs "
                  f"{tot_ll/max(tot_w,EPS):.5f}", flush=True)
    return model, Pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--horizons", default="1,6")
    ap.add_argument("--switch", type=float, default=0.005)
    ap.add_argument("--rounds", type=int, default=6,
                    help="EM rounds with copying, continuing from the ordinary "
                         "fit. Each is a forward-backward pass over every "
                         "position of every distinct set, so this is the "
                         "expensive part")
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
        if r is not None:
            te[h] = (ym, r)
    print(f"train {tr[0]}..{tr[-1]}   " +
          "   ".join(f"h={h}:{te[h][0]}" for h in sorted(te)))
    print(f"switch rate {args.switch}   copy rounds {args.rounds}\n")

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

    def report(tag, mdl, P):
        rows = []
        for h in sorted(te):
            ym, rec = te[h]
            th, _ = mdl.theta(tv[-1] + h * dt, True, ti=len(tv) - 1 + h)
            for nm, part in (("seen", [(a, c) for a, c in rec
                                       if a in train_sets]),
                             ("unseen", [(a, c) for a, c in rec
                                         if a not in train_sets]),
                             ("all", list(rec))):
                if not part: continue
                Xp, wp = E.build(part, V, 1)
                rows.append((h, nm,
                             E.score(Xp, wp, th, P[-1]),
                             copy_score(Xp, wp, th, P[-1], args.switch)))
        print(f"\n  {tag}")
        print(f"    {'h':>3}{'part':>8}{'single block':>15}{'copying':>12}")
        for h, nm, a, b in rows:
            print(f"    {h:>3}{nm:>8}{a:>15.3f}{b:>12.3f}")
        return {(h, nm): (a, b) for h, nm, a, b in rows}

    before = report("blocks fitted WITHOUT copying", model, Pi)
    print(f"\n  refitting with copying...")
    model, Pi = refine(E, model, Pi, tv, Xs, ws, args.switch, args.rounds)
    after = report("blocks fitted WITH copying", model, Pi)

    print(f"\n  change from refitting (copying score in both)")
    print(f"    {'h':>3}{'part':>8}{'before':>12}{'after':>12}{'delta':>10}")
    for key in before:
        h, nm = key
        b0 = before[key][1]; b1 = after[key][1]
        print(f"    {h:>3}{nm:>8}{b0:>12.3f}{b1:>12.3f}{b1 - b0:>+10.3f}")
    print("""
  The 'single block' column is the ordinary mixture likelihood and the
  'copying' column allows switches. Refitting should help the copying column:
  blocks formed under copying no longer need to cover a whole observed
  pattern, so they can specialise on stretches of positions.

  If refitting makes the copying column worse, the blocks were already as good
  as they get for this purpose and the gain from copying is purely a scoring
  effect.
""")


if __name__ == "__main__":
    sys.exit(main())
