#!/usr/bin/env python
"""
122_responsibility_temperature.py -- is the model too sure which block a novel
sequence came from?

THE PUZZLE THIS IS CHASING
--------------------------
At six months out, three quarters of sequences carry a set never seen in
training, but 95% of that novelty is built entirely from mutations that WERE
seen, and the median set sits about three edits from something in the training
data. The model can reach these sequences in principle. It scores them roughly
twelve nats below a table that assigns one flat constant to everything it does
not recognise.

Smoothing theta helps a little and does not close it. So something else is
going on, and the candidate is the mixture step rather than the emission.

THE MECHANISM BEING TESTED
--------------------------
P(S) is a weighted sum over blocks. A sequence three edits from a training set
should be partly explained by several blocks, each of which accounts for a
different part of it. But the blocks in this model are extremely pure, so the
log-likelihoods under different blocks differ by hundreds of nats and the
softmax over them saturates: essentially all the weight lands on one block.
That block then charges the sequence for every one of its three mismatches,
with nothing else contributing.

If that is what happens, the responsibility distribution for a novel sequence
will have near-zero entropy -- the model is certain, and wrong.

THE INTERVENTION
----------------
A temperature on the responsibilities. Instead of

    P(S) = sum_k pi_k P(S|k)

score with the block log-likelihoods divided by T before combining, which
flattens the mixture without touching theta. T = 1 is the model as it stands.
T > 1 spreads weight across blocks.

This is a SCORING change, not a refit: the same fitted model is scored several
ways, so nothing here is confounded by a different fit.

WHAT THE OUTCOME MEANS
----------------------
  entropy near zero on novel sets, and T > 1 improves them
      -> responsibility collapse is real and is part of the deficit. The fix
         belongs in the mixture step, not in theta.
  entropy already high, or T > 1 does not help
      -> the model is not overconfident about block membership, and the
         deficit is in the emission after all.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def resp_stats(E, X, w, th, pi):
    """Entropy of the block posterior per sequence, and how much weight the
    single best block takes. Both in natural units, weighted by count."""
    lp = E.loglik_matrix(X, th) + np.log(pi + EPS)[None, :]
    mx = lp.max(1, keepdims=True)
    R = np.exp(lp - mx); R /= np.maximum(R.sum(1, keepdims=True), EPS)
    ent = -(R * np.log(R + EPS)).sum(1)
    top = R.max(1)
    tw = w.sum()
    return float((w * ent).sum() / tw), float((w * top).sum() / tw), R.shape[1]


def score_bg(E, X, w, th, pi, eps, th_bg):
    """Held-out score with a background component mixed in.

        P(S) = (1-eps) * sum_k pi_k P(S|k)  +  eps * P_bg(S)

    P_bg is one Bernoulli at the pooled training frequencies -- deliberately
    blunt, no block structure. This stays a proper distribution over sets, so
    the number is comparable to every other held-out score in the project.

    Tempering the block log-likelihoods was the obvious thing to try and it is
    wrong: dividing a log-likelihood by T leaves something that does not
    normalise, so the score rises with T mechanically rather than because the
    model improved. Mixing in a background component achieves the same intent
    -- stop one sharp block from being charged for every mismatch -- while
    remaining a distribution."""
    if eps <= 0:
        lp = E.loglik_matrix(X, th) + np.log(pi + EPS)[None, :]
        mx = lp.max(1, keepdims=True)
        v = (np.log(np.exp(lp - mx).sum(1, keepdims=True)) + mx).ravel()
        return float((w * v).sum() / w.sum())
    lp = E.loglik_matrix(X, th) + np.log(pi + EPS)[None, :] + np.log(1 - eps)
    lb = E.loglik_matrix(X, th_bg[None, :]) + np.log(eps)
    both = np.concatenate([lp, lb], axis=1)
    mx = both.max(1, keepdims=True)
    v = (np.log(np.exp(both - mx).sum(1, keepdims=True)) + mx).ravel()
    return float((w * v).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--horizons", default="1,6")
    ap.add_argument("--background", default="0,0.01,0.05,0.2,0.5",
                    help="weight given to a single blunt background component "
                         "mixed alongside the blocks. 0 is the model as it "
                         "stands")
    ap.add_argument("--max-K", type=int, default=48)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--beta-prior", type=float, default=1.0)
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

    # ---- one fit, scored several ways
    _, _, _, _, _, w0 = E.fit_flat(Xs, ws, V, args.max_K, seed=args.seed,
                                   drift=False, split_merge=False)
    _, pi_sm, _, _, _, beta_sm = E.fit_flat(
        Xs, ws, V, args.max_K, seed=args.seed, drift=True, split_merge=True,
        init_beta=w0)
    model, Pi, tv, _, _, _ = E.fit(
        Xs, ws, V, args.max_K, seed=args.seed, drift=True, names=names,
        verbose=False, iters=args.iters, K_warm=args.max_K, birth_every=1,
        births_per_call=4, refit=0, penalty="prior", warm=(beta_sm, pi_sm),
        warm_mode="tree", hier_drift=True, rescan_every=25,
        beta_prior=args.beta_prior)
    dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
    K = int(model.alive.sum())
    print(f"one fit, {K} blocks, Beta prior {args.beta_prior:g}. "
          f"Everything below scores THAT fit, so nothing is confounded by a "
          f"different model.\n")

    Es = [float(x) for x in args.background.split(",")]
    Xall = np.vstack(Xs); wall = np.concatenate(ws)
    th_bg = np.clip((wall[:, None] * Xall).sum(0) / wall.sum(), 1e-4, 1 - 1e-4)
    for h in sorted(te):
        ym, rec = te[h]
        th, ks = model.theta(tv[-1] + h * dt, True, ti=len(tv) - 1 + h)
        seen = [(s_, c) for s_, c in rec if s_ in train_sets]
        unseen = [(s_, c) for s_, c in rec if s_ not in train_sets]
        print(f"{'=' * 74}\n  {ym}   h={h}   "
              f"unseen share {100*sum(c for _, c in unseen)/max(sum(c for _, c in rec),1):.1f}%"
              f"\n{'=' * 74}")
        print(f"\n  how sure is the model which block a sequence came from?")
        print(f"    {'':<10}{'entropy':>10}{'max entropy':>13}{'top block':>12}")
        parts = {}
        for nm, part in (("seen", seen), ("unseen", unseen)):
            if not part: continue
            Xp, wp = E.build(part, V, 1)
            parts[nm] = (Xp, wp)
            e, t, kk = resp_stats(E, Xp, wp, th, Pi[-1])
            print(f"    {nm:<10}{e:>10.4f}{np.log(kk):>13.3f}{t:>11.1%}")
        print(f"\n  mixing in a blunt background component")
        print(f"    {'eps':>6}" + "".join(f"{nm:>12}" for nm in parts)
              + f"{'all':>12}")
        Xa, wa = E.build(rec, V, 1)
        base = {}
        for ep in Es:
            row = []
            for nm, (Xp, wp) in parts.items():
                sc = score_bg(E, Xp, wp, th, Pi[-1], ep, th_bg)
                base.setdefault(nm, sc)
                row.append(sc)
            alls = score_bg(E, Xa, wa, th, Pi[-1], ep, th_bg)
            base.setdefault("all", alls)
            print(f"    {ep:>6.2f}" + "".join(f"{x:>12.3f}" for x in row)
                  + f"{alls:>12.3f}")
        best = {nm: max(score_bg(E, parts[nm][0], parts[nm][1], th, Pi[-1],
                                 ep, th_bg) for ep in Es) for nm in parts}
        best["all"] = max(score_bg(E, Xa, wa, th, Pi[-1], ep, th_bg)
                          for ep in Es)
        print(f"    {'gain':>6}" + "".join(
            f"{best[nm] - base[nm]:>+12.3f}" for nm in parts)
            + f"{best['all'] - base['all']:>+12.3f}")
        print()

    print("""
  entropy is of the block posterior, in nats. Zero means the model is certain
  which block a sequence came from. max entropy is log K, what total
  uncertainty would look like.

  entropy near zero on unseen sets, and a background component helps them
      -> the model is confidently assigning a novel sequence to one sharp
         block, which then charges it for every mismatch while no other block
         contributes. That is a mixture problem, and the fix belongs there
         rather than in theta.
  entropy already sizeable, or the background component does not help
      -> block membership is not the issue and the deficit sits in the
         emission.

  The background component is applied at scoring time here, not fitted. If it
  helps, the principled version is to include it in the mixture during fitting
  so the blocks form knowing it is there.
""")


if __name__ == "__main__":
    sys.exit(main())
