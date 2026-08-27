#!/usr/bin/env python
"""
128_fitness_regression.py -- does a block grow because of what it carries?

THE HYPOTHESIS
--------------
The model currently has no connection between the two halves of the mixture. A
block's weight next month is fitted freely; nothing says a block is common
BECAUSE of the mutations it carries. If a per-mutation effect existed and were
shared across blocks, then

    log pi_k(t+1) - log pi_k(t)  ~  sum_v theta[k,v] * f_v

with a single vector f estimated from every block at once. That is the
replicator equation with fitness decomposed over mutations, and it is the
standard object in population genetics -- absent from mixture models, which
treat component weights as free parameters because there is no theory of why a
topic becomes popular.

WHY IT MATTERS MORE THAN THE FIT
--------------------------------
An oracle that refits the weights on the test month bounds what better weights
could buy, and that bound came out small. But it can only reweight blocks that
already exist, using the observed future. A shared f does something the oracle
cannot express: it assigns a growth rate to a block from its CONTENTS, so a
block first seen last month at a fraction of a percent gets a rate immediately,
estimated from every other block carrying those mutations. That is the only
route in this model to anticipating a lineage before it is established.

WHAT IS ESTIMATED
-----------------
Ridge regression of monthly log-weight changes on block composition, with
blocks weighted by how much population they hold -- a change in a block with
four sequences is noise, one in a block with eighty thousand is not. Fitted on
early months and evaluated on held-out later ones, so the number reported is
predictive rather than in-sample.

WHAT WOULD MAKE IT CREDIBLE
---------------------------
Not the R-squared on its own. The test is whether the largest positive weights
in f land on mutations independently known to matter -- and the script prints
the top and bottom of f by name so that can be judged rather than asserted. If
f is signal, the joint model is worth building. If f is noise, it is not, and
that is worth knowing in an afternoon rather than a fortnight.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_design(model, Pi, tv, drift, min_weight):
    """Rows are (block, month-transition). Features are the block's emission
    profile that month; the target is its log-weight change into the next."""
    ks = np.flatnonzero(model.alive)
    T = len(Pi)
    Xr, yr, wr, meta = [], [], [], []
    for t in range(T - 1):
        th, _ = model.theta(tv[t], drift, ti=t)
        a = Pi[t]; b = Pi[t + 1]
        for i, k in enumerate(ks):
            if a[i] < min_weight or b[i] < min_weight:
                continue          # a ratio between two tiny weights is noise
            Xr.append(th[i])
            yr.append(np.log(b[i] + EPS) - np.log(a[i] + EPS))
            wr.append(a[i])       # trust changes in blocks that hold people
            meta.append((t, int(k)))
    if not Xr:
        return None
    return (np.stack(Xr), np.array(yr), np.array(wr), meta)


def ridge(X, y, w, lam):
    """Weighted ridge with an intercept, centred so lam does not shrink it."""
    sw = np.sqrt(w)[:, None]
    mx = (w[:, None] * X).sum(0) / w.sum()
    my = float((w * y).sum() / w.sum())
    Xc = (X - mx[None, :]) * sw
    yc = (y - my) * sw.ravel()
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    f = np.linalg.solve(A, Xc.T @ yc)
    return f, my, mx


def predict(X, f, my, mx):
    return my + (X - mx[None, :]) @ f


def r2(y, p, w):
    my = float((w * y).sum() / w.sum())
    ss = float((w * (y - p) ** 2).sum())
    st = float((w * (y - my) ** 2).sum())
    return 1.0 - ss / max(st, EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--windows", required=True,
                    help="comma-separated LAST training months, each fitted on "
                         "the 12 months ending there, e.g. "
                         "2021-04,2021-10,2022-05,2022-11. Transitions from "
                         "all of them are pooled into one regression")
    ap.add_argument("--window-len", type=int, default=12)
    ap.add_argument("--max-K", type=int, default=48)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--min-weight", type=float, default=1e-3,
                    help="ignore blocks below this share of a month")
    ap.add_argument("--holdout", type=int, default=3,
                    help="last N month-transitions held out from the fit")
    ap.add_argument("--lams", default="1,10,100,1000,10000")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()

    E = load_engine(args.engine)
    names, V = E.load_names(args.vocab)
    # One fit per window. Transitions are pooled, and the window index is
    # carried so that whole windows can be held out -- which is the point.
    # Within a single window a mutation that tracks the lineage currently
    # sweeping is indistinguishable from one that confers an advantage,
    # because there is only one replacement event to learn from. Holding out
    # an ENTIRE window asks whether f transfers to a different sweep, which is
    # the only version of the question worth answering.
    wins = [x.strip() for x in args.windows.split(",") if x.strip()]
    Xall, yall, wall, meta_all = [], [], [], []
    for wi, last in enumerate(wins):
        tr = [E.ym_add(last, -(args.window_len - 1) + i)
              for i in range(args.window_len)]
        recs = [E.load_month(args.data_dir, ym) for ym in tr]
        if any(r is None for r in recs):
            print(f"  window ..{last}: missing months, skipped"); continue
        Xs, ws = zip(*[E.build(r, V, args.min_count) for r in recs])
        Xs, ws = list(Xs), list(ws)
        _, _, _, _, _, w0 = E.fit_flat(Xs, ws, V, args.max_K, seed=args.seed,
                                       drift=False, split_merge=False)
        _, pi_sm, _, _, _, beta_sm = E.fit_flat(
            Xs, ws, V, args.max_K, seed=args.seed, drift=True,
            split_merge=True, init_beta=w0)
        model, Pi, tv, _, _, _ = E.fit(
            Xs, ws, V, args.max_K, seed=args.seed, drift=True, names=names,
            verbose=False, iters=args.iters, K_warm=args.max_K, birth_every=1,
            births_per_call=4, refit=0, penalty="prior",
            warm=(beta_sm, pi_sm), warm_mode="tree", hier_drift=True,
            rescan_every=25)
        built = build_design(model, Pi, tv, True, args.min_weight)
        if built is None:
            print(f"  window ..{last}: no usable transitions"); continue
        Xw, yw, ww, mw = built
        Xall.append(Xw); yall.append(yw); wall.append(ww)
        meta_all += [(wi, t, k) for (t, k) in mw]
        print(f"  window {tr[0]}..{tr[-1]}   "
              f"{int(model.alive.sum())} blocks   {len(Xw)} transitions",
              flush=True)
    if not Xall:
        sys.exit("no windows produced usable transitions")
    X = np.vstack(Xall); y = np.concatenate(yall); w = np.concatenate(wall)
    meta = meta_all
    nwin = len({m[0] for m in meta})
    print(f"\npooled: {len(X)} block-month transitions from {nwin} windows\n")
    # leave-one-window-out: fit on every window but one, predict the held-out
    # one. A mutation that only marks whichever lineage happened to be sweeping
    # in its own window cannot transfer; one that carries an advantage across
    # different sweeps can.
    wid = np.array([m[0] for m in meta])
    uw = sorted(set(wid.tolist()))
    if len(uw) < 2:
        sys.exit("need at least two windows to hold one out")

    print(f"  leave-one-window-out\n")
    print(f"  {'lambda':>10}" + "".join(f"{'win ' + str(u):>10}" for u in uw)
          + f"{'mean':>10}{'nonzero f':>12}")
    best = None
    for lam in (float(x) for x in args.lams.split(",")):
        outs = []
        for u in uw:
            tri = np.flatnonzero(wid != u); tei = np.flatnonzero(wid == u)
            f, my, mx = ridge(X[tri], y[tri], w[tri], lam)
            outs.append(r2(y[tei], predict(X[tei], f, my, mx), w[tei]))
        f, my, mx = ridge(X, y, w, lam)
        nz = int((np.abs(f) > 1e-6).sum())
        m = float(np.mean(outs))
        print(f"  {lam:>10.0f}" + "".join(f"{o:>10.3f}" for o in outs)
              + f"{m:>10.3f}{nz:>12,}")
        if best is None or m > best[0]:
            best = (m, lam, f, my, mx)
    tr_i = np.arange(len(X)); te_i = np.arange(len(X))

    rout, lam, f, my, mx = best
    print(f"\n  best mean leave-one-window-out R2 {rout:.3f} at lambda {lam:.0f}")
    print(f"\n  A shuffled control: the same regression with the targets "
          f"permuted,\n  which is what a fit to noise scores.")
    rng = np.random.default_rng(args.seed)
    ctrl = []
    for _ in range(20):
        yp = rng.permutation(y)
        outs = []
        for u in uw:
            tri = np.flatnonzero(wid != u); tei = np.flatnonzero(wid == u)
            fp, myp, mxp = ridge(X[tri], yp[tri], w[tri], lam)
            outs.append(r2(yp[tei], predict(X[tei], fp, myp, mxp), w[tei]))
        ctrl.append(float(np.mean(outs)))
    print(f"    shuffled held-out R2: mean {np.mean(ctrl):+.3f}  "
          f"max {np.max(ctrl):+.3f}   (real {rout:+.3f})")

    order = np.argsort(-f)
    print(f"\n  mutations with the largest FITTED effect on how fast a block "
          f"grows")
    print(f"    {'rank':>5}{'mutation':>14}{'f':>10}"
          f"      {'rank':>5}{'mutation':>14}{'f':>10}")
    for r in range(args.top):
        a = int(order[r]); b = int(order[-(r + 1)])
        print(f"    {r+1:>5}{names.get(a, str(a)):>14}{f[a]:>10.3f}"
              f"      {r+1:>5}{names.get(b, str(b)):>14}{f[b]:>10.3f}")
    print("""
  Left column is the largest positive effects, right the largest negative.

  R2 is measured by holding out a WHOLE window, so a positive number means f
  transferred from one set of sweeps to a different one.

  positive and above the shuffled control
      -> an advantage carried by mutations, not merely a label for whichever
         lineage was winning. A shared f can then give a rate to a block never
         seen growing, which is the one thing reweighting existing blocks
         cannot do.
  around zero, with the top of f reading as a marker list for one lineage
      -> f has learned WHICH lineage was replacing which, not why. That does
         not transfer, and coupling pi to theta this way will not help.

  Caveat that no R2 removes: composition and growth are both read off the same
  fitted blocks, so f is a statement about this partition, not about mutations
  in the abstract. A block that grew is partly WHY the model made that block.
""")


if __name__ == "__main__":
    sys.exit(main())
