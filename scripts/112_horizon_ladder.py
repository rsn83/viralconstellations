#!/usr/bin/env python
"""
112_horizon_ladder.py -- the same four models, scored 1, 2, 3 and 6 months out.

WHAT THIS ANSWERS
-----------------
Letting each block's signature move over time -- one number per (block,
mutation) replaced by a level and a slope -- buys something at one month
ahead. This asks how far that carries.

Nothing is refitted per horizon. One training window, one fit, then the
signatures are evaluated at t+1, t+2, t+3, t+6 and scored against months the
model never saw. "Scored" means the mean per-sequence log-likelihood of every
sequence observed in that calendar month -- not of a variant, not of a subset.
The months are printed before fitting so there is no ambiguity about which
ones are being predicted. The mixture is held at the last training month's weights
throughout, so any difference between horizons is the signatures, not the mix.

TWO BASELINES FROM OUTSIDE THE MIXTURE FAMILY
  independence     one block, no structure at all. Mutation frequencies from
                   the last training month, every mutation independent. This
                   is the formal version of the null that "combinations are
                   arbitrary" -- the thing block structure is supposed to beat.
  exact-set lookup non-parametric. P(set) = how often that exact set was seen
                   in training, smoothed. Very strong on sets that recur,
                   at its smoothing floor on anything new. If the mixture
                   wins, it should win on the sets this cannot reach.

FOUR RUNGS, each fitted by the same code on the same data
  flat                          one fixed signature per block
  flat + drift                  each entry gets a level and a slope
  flat + drift + split-merge    plus scheduled splitting
  hierarchical + birth-death    blocks form a tree; children hold deviations

Read the GAIN row, not the raw likelihoods. Later test months are harder for
every model, so raw likelihood falls with h no matter what. The gain over the
fixed-signature rung cancels that and isolates what the slopes are worth.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def oracle_pi(E, X, w, th, iters=300, tol=1e-9):
    """Best mixture weights for THIS month with the signatures frozen.

    EM on the weights alone. The signatures never move, so this is concave in
    the weights and the fixed point is the true maximum -- a genuine ceiling.
    It uses the test month, so it is a diagnostic, never a reportable model."""
    L = E.loglik_matrix(X, th)
    pi = np.full(th.shape[0], 1.0 / th.shape[0])
    prev = -np.inf
    for _ in range(iters):
        lp = L + np.log(pi + EPS)[None, :]
        mx = lp.max(1, keepdims=True)
        R = np.exp(lp - mx); s = R.sum(1, keepdims=True)
        ll = float((w * (np.log(s.ravel() + EPS) + mx.ravel())).sum() / w.sum())
        R /= np.maximum(s, EPS)
        pi = (w[:, None] * R).sum(0); pi /= max(pi.sum(), EPS)
        if abs(ll - prev) < tol: break
        prev = ll
    return pi


def independence_theta(E, records, V, min_count):
    """One block. Every mutation independent, at last-month frequencies.

    This is the null that slide 3 argues against: combinations carry no
    information beyond how common each mutation is on its own."""
    X, w = E.build(records, V, min_count)
    p = (w[:, None] * X).sum(0) / w.sum()
    return np.clip(p, 1e-4, 1 - 1e-4)[None, :], np.array([1.0])


def lookup_table(train_records_list, alpha=0.5):
    """P(set) from how often that exact set was seen in training, smoothed.

    Non-parametric and hard to beat on sets that recur. By construction it
    cannot do better than its smoothing floor on a set it has never seen,
    which is exactly the case the mixture is supposed to handle."""
    cnt = {}
    for recs in train_records_list:
        for s_, c in recs:
            cnt[s_] = cnt.get(s_, 0.0) + float(c)
    N = sum(cnt.values()); M = len(cnt)
    denom = N + alpha * (M + 1)
    return cnt, alpha, denom


def lookup_score(cnt, alpha, denom, records):
    lp = []; ws = []
    for s_, c in records:
        lp.append(np.log((cnt.get(s_, 0.0) + alpha) / denom))
        ws.append(float(c))
    lp = np.array(lp); ws = np.array(ws)
    return float((ws * lp).sum() / ws.sum())


def split_seen(records, train_sets):
    seen = [(s_, c) for s_, c in records if s_ in train_sets]
    unseen = [(s_, c) for s_, c in records if s_ not in train_sets]
    return seen, unseen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True,
                    help="YYYY-MM:YYYY-MM inclusive")
    ap.add_argument("--horizons", default="1,2,3,6",
                    help="months ahead of the last training month. Test months "
                         "are derived from this and printed before fitting")
    ap.add_argument("--test", default="",
                    help="name the test months explicitly instead, e.g. "
                         "2022-06,2022-07,2022-08,2022-11. Overrides --horizons")
    ap.add_argument("--max-K", type=int, default=48)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--half-life", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--warm-mode", choices=["star", "tree"], default="tree")
    ap.add_argument("--no-hier-drift", action="store_true")
    ap.add_argument("--time-model", choices=["slope", "chain"], default="slope")
    ap.add_argument("--eps-scales", default="",
                    help="comma-separated per-position tolerance scales to "
                         "sweep, e.g. 0,0.05,0.2,0.5. Overrides the others")
    ap.add_argument("--eps-cap", type=float, default=0.25)
    ap.add_argument("--emission-floors", default="",
                    help="comma-separated per-position floors to sweep, e.g. "
                         "0,0.001,0.01,0.05. Overrides --beta-priors when set")
    ap.add_argument("--beta-priors", default="1.0",
                    help="comma-separated Beta(a,a) strengths to sweep. Each "
                         "one refits the hierarchical model and is reported as "
                         "its own row, seen and unseen split out")
    ap.add_argument("--chain-states", type=int, default=3)
    args = ap.parse_args()

    E = load_engine(args.engine)
    names, V = E.load_names(args.vocab)
    tr = E.months_in_range(args.train)
    if args.test:
        want = [(E.months_in_range(f"{tr[-1]}:{x}").__len__() - 1, x)
                for x in (m.strip() for m in args.test.split(",")) if x]
    else:
        want = [(h, E.ym_add(tr[-1], h)) for h in
                (int(x) for x in args.horizons.split(","))]
    print(f"train  {tr[0]}..{tr[-1]}  ({len(tr)} months, "
          f"min-count {args.min_count})")
    for h, ym in want:
        print(f"test   {ym}   h={h}   scored with every sequence in that month")

    recs = [E.load_month(args.data_dir, ym) for ym in tr]
    missing = [ym for ym, r in zip(tr, recs) if r is None]
    if missing:
        sys.exit(f"missing training months: {missing}")
    Xs, ws = zip(*[E.build(r, V, args.min_count) for r in recs])
    Xs, ws = list(Xs), list(ws)

    # test months, loaded once
    te = {}
    for h, ym in want:
        r = E.load_month(args.data_dir, ym)
        if r is None:
            print(f"  test month {ym} (h={h}) missing, skipped", flush=True)
            continue
        te[h] = (ym,) + E.build(r, V, 1) + (r,)

    # what the model could actually have learned from: the training records
    # AFTER the min-count filter, since anything filtered out was not seen.
    filt = [[(s_, c) for s_, c in r if c >= args.min_count] for r in recs]
    train_sets = {s_ for r in filt for s_, _ in r}
    cnt, alpha, denom = lookup_table(filt)
    print(f"\ndistinct sets in training (after min-count): {len(train_sets):,}")
    print(f"lookup floor for an unseen set: "
          f"log({alpha}/{denom:,.0f}) = {np.log(alpha/denom):.3f}")

    if args.eps_scales:
        esc = [float(x) for x in args.eps_scales.split(",")]
        bps = [1.0] * len(esc); fls = [0.0] * len(esc)
        hier_rows = [("hierarchical + birth-death" if e == 0 else
                      f"hierarchical, eps-scale {e:g}") for e in esc]
    elif args.emission_floors:
        fls = [float(x) for x in args.emission_floors.split(",")]
        bps = [1.0] * len(fls); esc = [0.0] * len(fls)
        hier_rows = [("hierarchical + birth-death" if f == 0 else
                      f"hierarchical, floor {f:g}") for f in fls]
    else:
        bps = [float(x) for x in args.beta_priors.split(",")]
        fls = [0.0] * len(bps); esc = [0.0] * len(bps)
        hier_rows = [("hierarchical + birth-death" if b == 1.0 else
                      f"hierarchical, Beta({b:g},{b:g})") for b in bps]
    rungs = ["flat", "flat + drift", "flat + drift + split-merge"] + hier_rows
    outside = ["independence (1 block)", "exact-set lookup"]
    res = {r: {h: [] for h in te} for r in rungs + outside}
    used = {r: [] for r in rungs + outside}
    orac = {r: {h: [] for h in te} for r in rungs}
    seen_ll = {r: {h: [] for h in te} for r in rungs + outside}
    unseen_ll = {r: {h: [] for h in te} for r in rungs + outside}
    seen_share = {}

    th_ind, pi_ind = independence_theta(E, filt[-1], V, 1)
    used["independence (1 block)"].append(1)
    used["exact-set lookup"].append(len(train_sets))
    for h, (ym, Xte, wte, rte) in te.items():
        sn, un = split_seen(rte, train_sets)
        seen_share[h] = sum(c for _, c in sn) / max(
            sum(c for _, c in rte), 1e-9)
        res["independence (1 block)"][h].append(E.score(Xte, wte, th_ind, pi_ind))
        res["exact-set lookup"][h].append(
            lookup_score(cnt, alpha, denom, rte))
        for nm, part in [("seen", sn), ("unseen", un)]:
            if not part: continue
            Xp, wp = E.build(part, V, 1)
            tgt = seen_ll if nm == "seen" else unseen_ll
            tgt["independence (1 block)"][h].append(E.score(Xp, wp, th_ind, pi_ind))
            tgt["exact-set lookup"][h].append(
                lookup_score(cnt, alpha, denom, part))

    for sd in range(args.seeds):
        print(f"\n--- seed {sd} ---", flush=True)

        # rung 1: fixed signatures. Same theta at every horizon.
        th0, pi0, u0, _, _, warm = E.fit_flat(Xs, ws, V, args.max_K, seed=sd,
                                              drift=False, split_merge=False)
        used["flat"].append(u0)
        for h, (ym, Xte, wte, rte) in te.items():
            res["flat"][h].append(E.score(Xte, wte, th0, pi0))
            orac["flat"][h].append(
                E.score(Xte, wte, th0, oracle_pi(E, Xte, wte, th0)))
            for nm, part in zip(("seen", "unseen"),
                                split_seen(rte, train_sets)):
                if not part: continue
                Xp, wp = E.build(part, V, 1)
                (seen_ll if nm == "seen" else unseen_ll)["flat"][h].append(
                    E.score(Xp, wp, th0, pi0))

        # rungs 2 and 3: signatures drift, so evaluate at t + h*dt
        for rung, sm in [("flat + drift", False),
                         ("flat + drift + split-merge", True)]:
            thn, pin, un, _, _, beta, gamma, tv = E.fit_flat(
                Xs, ws, V, args.max_K, seed=sd, drift=True, split_merge=sm,
                half_life=args.half_life, names=names,
                verbose=(sd == 0 and sm), init_beta=warm, return_gamma=True)
            used[rung].append(un)
            dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
            for h, (ym, Xte, wte, rte) in te.items():
                th = np.clip(E.sig(beta + gamma * (tv[-1] + h * dt)),
                             1e-4, 1 - 1e-4)
                res[rung][h].append(E.score(Xte, wte, th, pin))
                orac[rung][h].append(
                    E.score(Xte, wte, th, oracle_pi(E, Xte, wte, th)))
                for nm, part in zip(("seen", "unseen"),
                                    split_seen(rte, train_sets)):
                    if not part: continue
                    Xp, wp = E.build(part, V, 1)
                    (seen_ll if nm == "seen" else unseen_ll)[rung][h].append(
                        E.score(Xp, wp, th, pin))
            if sm:
                beta_sm, pi_sm = beta, pin

        # rung 4: hierarchical
        for bp, fl, es, hr in zip(bps, fls, esc, hier_rows):
          if sd == 0:
              print(f"\n  --- fitting {hr} "
                    f"({bps.index(bp)+1} of {len(bps)}) ---", flush=True)
          model, Pi, tv, births, _, remerges = E.fit(
              Xs, ws, V, args.max_K, seed=sd, sigma=args.sigma, beta_prior=bp, floor_eps=fl, eps_scale=es, eps_cap=args.eps_cap,
              half_life=args.half_life, drift=True, names=names,
              verbose=(sd == 0), iters=args.iters, K_warm=args.max_K,
              birth_every=1, births_per_call=4, refit=0, penalty="prior",
              warm=(beta_sm, pi_sm), warm_mode=args.warm_mode,
              hier_drift=not args.no_hier_drift, rescan_every=25,
              time_model=args.time_model, chain_states=args.chain_states)
          used[hr].append(int(model.alive.sum()))
          dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
          for h, (ym, Xte, wte, rte) in te.items():
              # ti tells a chain time model how far past the window to carry
              # the state posterior; ignored by the slope model
              th, ks = model.theta(tv[-1] + h * dt, True, ti=len(tv) - 1 + h)
              res[hr][h].append(E.score(Xte, wte, th, Pi[-1]))
              orac[hr][h].append(
                  E.score(Xte, wte, th, oracle_pi(E, Xte, wte, th)))
              for nm, part in zip(("seen", "unseen"),
                                  split_seen(rte, train_sets)):
                  if not part: continue
                  Xp, wp = E.build(part, V, 1)
                  (seen_ll if nm == "seen" else unseen_ll)[hr][h].append(
                      E.score(Xp, wp, th, Pi[-1]))

    hs_sorted = sorted(te)
    hdr = "".join(f"{'h='+str(h):>12}" for h in hs_sorted)
    allr = outside + rungs

    print("\n" + "=" * 90)
    print(f"HORIZON LADDER   train {tr[0]}..{tr[-1]}   "
          f"max-K={args.max_K}   {args.seeds} seed(s)")
    print("=" * 90)
    print(f"\n  {'model':<30}{hdr}{'blocks':>12}")
    print(f"  {'':<30}" + "".join(f"{te[h][0]:>12}" for h in hs_sorted))
    for r in allr:
        if r == "flat": print()
        row = "".join(f"{np.mean(res[r][h]):>12.3f}" for h in hs_sorted)
        print(f"  {r:<30}{row}{np.mean(used[r]):>12.0f}")

    if len(hier_rows) > 1:
        base = hier_rows[0]
        print(f"\n  SWEEP: what softening the emission costs and buys")
        for h in hs_sorted:
            print(f"\n    {te[h][0]}  h={h}   seen share "
                  f"{100*seen_share[h]:.1f}%")
            print(f"      {'prior':<26}{'seen':>10}{'unseen':>10}"
                  f"{'d seen':>10}{'d unseen':>10}")
            for r in hier_rows:
                if not seen_ll[r][h] or not unseen_ll[r][h]: continue
                a, b = np.mean(seen_ll[r][h]), np.mean(unseen_ll[r][h])
                a0 = np.mean(seen_ll[base][h]); b0 = np.mean(unseen_ll[base][h])
                print(f"      {r:<26}{a:>10.3f}{b:>10.3f}"
                      f"{a - a0:>+10.3f}{b - b0:>+10.3f}")
        print("""
      d seen / d unseen are against the unsmoothed model. The prediction being
      tested is that they move in OPPOSITE directions: sharp blocks describe
      seen sets well and assign almost nothing to a set one mutation off. If
      both columns get worse, sharpness is not the mechanism.
""")

    print(f"\n  gain over flat")
    for r in rungs[1:]:
        row = "".join(f"{np.mean(res[r][h]) - np.mean(res['flat'][h]):>+12.3f}"
                      for h in hs_sorted)
        print(f"  {r:<30}{row}")

    print(f"\n{'=' * 90}\n  PI HEADROOM   signatures frozen, weights refit on "
          f"the test month\n{'=' * 90}")
    print(f"\n  {'model':<30}{hdr}")
    for r in rungs:
        c = [np.mean(res[r][h]) for h in hs_sorted]
        o = [np.mean(orac[r][h]) for h in hs_sorted]
        print(f"  {r:<30}" + "".join(f"{x:>12.3f}" for x in c) + "   copied")
        print(f"  {'':<30}" + "".join(f"{x:>12.3f}" for x in o) + "   best possible")
        print(f"  {'':<30}" + "".join(f"{b-a:>+12.3f}" for a, b in zip(c, o))
              + "   headroom\n")
    print("  Headroom is what copying last month's weights costs. Small means"
          "\n  the comparison is about the signatures, as intended. Large means"
          "\n  a rung is being held back by copy-pi and its gain understates it.")

    print(f"\n{'=' * 90}\n  SEEN vs UNSEEN SETS   split by whether the exact set "
          f"appeared in training\n{'=' * 90}")
    for h in hs_sorted:
        print(f"\n  {te[h][0]}  h={h}   sequences whose set was seen in "
              f"training: {100*seen_share[h]:.1f}%")
        print(f"    {'model':<30}{'seen':>12}{'unseen':>12}{'gap':>12}")
        for r in allr:
            if not seen_ll[r][h] or not unseen_ll[r][h]: continue
            a, b = np.mean(seen_ll[r][h]), np.mean(unseen_ll[r][h])
            print(f"    {r:<30}{a:>12.3f}{b:>12.3f}{b-a:>+12.3f}")
    print("""
  Seen sets measure frequency tracking: the lookup table is near-optimal there
  by construction. Unseen sets measure putting probability on combinations
  never observed, where the lookup table sits at its smoothing floor. A gain
  concentrated in the unseen column is the result this project claims. A gain
  concentrated in the seen column is a good density model of the population
  that already exists -- worth saying plainly rather than leaving to inference.

  Read the GAIN row, not the raw likelihoods: later months are harder for every
  model, so raw likelihood falls with h regardless of what the model does.
""")


if __name__ == "__main__":
    sys.exit(main())
