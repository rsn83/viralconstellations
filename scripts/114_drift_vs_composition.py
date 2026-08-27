#!/usr/bin/env python
"""
114_drift_vs_composition.py -- is the drift real, or is it composition leaking
into the signatures?

THE IDEA IN PLAIN ENGLISH
-------------------------
A block's frequency for a mutation can rise for two different reasons.

  1. The block itself is changing. Its sequences really are picking up that
     mutation over time.
  2. The block is a bag holding two lineages, and one is replacing the other.
     Nothing inside either lineage changed. The block's average moved because
     its composition moved.

With few blocks, reason 2 is unavoidable: there are not enough blocks to give
each lineage its own, so gamma absorbs the composition shift and reports it as
drift. With enough blocks, each lineage gets its own and there is nothing left
for gamma to absorb.

If that is what is happening, three things follow, and all three are checked
here:

  A. |gamma| shrinks as K grows. The drift was never intrinsic.
  B. What drift buys shrinks as K grows. Adding blocks and letting entries
     drift become substitutes, not complements.
  C. Blocks get purer as K grows -- their entries move toward 0 or 1 -- so
     there is less middling frequency for a slope to act on.

If instead |gamma| holds up at large K, the drift is intrinsic: blocks really
are changing from within, and the two clocks are separate processes.

WHY IT MATTERS FOR PI
---------------------
If drift is composition in disguise, then pi is the only real time process in
the model and theta is downstream of it. That reframes the pi question. It is
not "we have not found the right trend law for pi" -- it is that pi moves in
rare jumps when one lineage replaces another, and sits still in between, which
is why copying beats a trend and why a stationary transition matrix loses.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def weighted_stats(model, Pi, tv, mid_lo=.05, mid_hi=.95):
    """Drift magnitude and block purity, weighted by how much of the
    population each block actually holds.

    Only middling-frequency entries count toward the drift statistic. An entry
    pinned at 0 or 1 has nothing for a slope to do, and including those would
    let a model look 'less drifty' merely by having more dead entries."""
    ks = np.flatnonzero(model.alive)
    w = Pi.mean(0)
    w = w / max(w.sum(), EPS)
    gsum = 0.0; gmax = 0.0; pur = 0.0; nmid = 0.0
    for i, k in enumerate(ks):
        k = int(k)
        th = 1.0 / (1.0 + np.exp(-np.clip(model.beta(k), -30, 30)))
        mid = (th > mid_lo) & (th < mid_hi)
        g = np.abs(model.slope(k))
        if mid.any():
            gsum += w[i] * float(g[mid].mean())
            gmax = max(gmax, float(g[mid].max()))
        nmid += w[i] * float(mid.sum())
        # purity: how far entries sit from 0.5, 1 = every entry pinned
        pur += w[i] * float((np.abs(th - 0.5) * 2).mean())
    return dict(gmean=gsum, gmax=gmax, purity=pur, nmid=nmid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--K-list", default="13,24,48,96")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3,
                    help="EM from a warm start still lands in different optima; "
                         "one seed cannot tell a real effect from a bad fit")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--half-life", type=float, default=1.0)
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
    rte = E.load_month(args.data_dir, args.test)
    if rte is None:
        sys.exit(f"missing test month {args.test}")
    Xte, wte = E.build(rte, V, 1)

    print(f"train {tr[0]}..{tr[-1]}   test {args.test}   V={V:,}   "
          f"min-count {args.min_count}")
    rows = []
    for K in (int(x) for x in args.K_list.split(",")):
        acc = {k: [] for k in ("ll_d", "ll_n", "worth", "gmean", "gmax",
                               "purity", "nmid", "Kused")}
        for sd in range(args.seeds):
            _, _, _, _, _, w0 = E.fit_flat(Xs, ws, V, K, seed=sd,
                                           drift=False, split_merge=False)
            _, pi_sm, _, _, _, beta_sm = E.fit_flat(
                Xs, ws, V, K, seed=sd, drift=True, split_merge=True,
                half_life=args.half_life, init_beta=w0)
            out = {}
            for tag, drift in (("drift", True), ("nodrift", False)):
                model, Pi, tv, _, _, _ = E.fit(
                    Xs, ws, V, K, seed=sd, sigma=args.sigma,
                    half_life=args.half_life, drift=drift, names=names,
                    verbose=False, iters=args.iters, K_warm=K, birth_every=1,
                    births_per_call=4, refit=0, penalty="prior",
                    warm=(beta_sm, pi_sm), warm_mode="tree", hier_drift=drift,
                    rescan_every=25)
                dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
                th, ks = model.theta(tv[-1] + dt, drift)
                out[tag] = dict(ll=E.score(Xte, wte, th, Pi[-1]),
                                K=int(model.alive.sum()))
                if drift:
                    out[tag].update(weighted_stats(model, Pi, tv))
            d = out["drift"]
            acc["ll_d"].append(d["ll"]); acc["ll_n"].append(out["nodrift"]["ll"])
            acc["worth"].append(d["ll"] - out["nodrift"]["ll"])
            acc["Kused"].append(d["K"])
            for k in ("gmean", "gmax", "purity", "nmid"):
                acc[k].append(d[k])
        rows.append(dict(K=K, Kused=np.mean(acc["Kused"]),
                         **{k: float(np.mean(acc[k])) for k in
                            ("ll_d", "ll_n", "worth", "gmean", "gmax",
                             "purity", "nmid")},
                         worth_sd=float(np.std(acc["worth"])),
                         g_sd=float(np.std(acc["gmean"])),
                         ll_n_sd=float(np.std(acc["ll_n"]))))
        r = rows[-1]
        print(f"  K={K:<4} occupied {r['Kused']:<5.0f} "
              f"drift {r['ll_d']:.3f}  no-drift {r['ll_n']:.3f}"
              f" +/-{r['ll_n_sd']:.3f}  worth {r['worth']:+.3f}"
              f" +/-{r['worth_sd']:.3f}   mean|gamma| {r['gmean']:.4f}"
              f" +/-{r['g_sd']:.4f}", flush=True)

    print(f"\n{'=' * 86}\n  IS THE DRIFT REAL, OR IS IT COMPOSITION?\n{'=' * 86}")
    print(f"\n  {'K':>5}{'occ':>6}{'drift':>10}{'no drift':>10}{'+/-':>8}"
          f"{'worth':>9}{'+/-':>8}{'mean|g|':>10}{'+/-':>8}"
          f"{'max|g|':>8}{'purity':>8}")
    for r in rows:
        print(f"  {r['K']:>5}{r['Kused']:>6.0f}{r['ll_d']:>10.3f}"
              f"{r['ll_n']:>10.3f}{r['ll_n_sd']:>8.3f}{r['worth']:>+9.3f}"
              f"{r['worth_sd']:>8.3f}{r['gmean']:>10.4f}{r['g_sd']:>8.4f}"
              f"{r['gmax']:>8.2f}{r['purity']:>8.3f}")
    print("\n  A 'worth' smaller than its own spread is not a difference, and a "
          "no-drift\n  column that is not monotone in K is a bad fit, not a "
          "property of the model.")

    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        fg = b["gmean"] / max(a["gmean"], 1e-12)
        print(f"\n  from K={a['K']} to K={b['K']}:")
        print(f"    mean |gamma| on middling entries   x{fg:.2f}"
              f"   ({a['gmean']:.4f} -> {b['gmean']:.4f})")
        if abs(a["worth"]) > 0.05:
            print(f"    what drift is worth                "
                  f"x{b['worth'] / a['worth']:.2f}"
                  f"   ({a['worth']:+.3f} -> {b['worth']:+.3f})")
        else:
            print(f"    what drift is worth                "
                  f"{a['worth']:+.3f} -> {b['worth']:+.3f}"
                  f"   (too small at K={a['K']} for a ratio to mean anything)")
        print(f"    block purity                       "
              f"{a['purity']:.3f} -> {b['purity']:.3f}")
    print("""
  worth   = held-out with drift minus held-out without it, same K, same warm
            start. What letting entries move actually buys at that K.
  mean|g| = average slope magnitude, over entries at middling frequency only,
            weighted by how much population each block holds.
  purity  = mean distance of entries from 0.5, doubled. 1 means every entry is
            pinned at 0 or 1, i.e. the block is one clean signature.

  mean|g| is the statistic that discriminates; 'worth' is a supporting check
  and its ratio is only meaningful when it is sizeable at the smallest K.

  Both shrink toward zero as K grows
      -> the drift was composition. Few blocks means each is a bag of
         lineages, and gamma reports the replacement inside the bag as if the
         bag itself were changing. Add blocks and there is nothing left to
         absorb. pi is then the only real time process, and theta is
         downstream of it.

  mean|g| holds up at large K, and drift still pays there
      -> the drift is intrinsic. Blocks really do change from within, and
         theta and pi are two separate processes on two separate clocks.
""")


if __name__ == "__main__":
    sys.exit(main())
