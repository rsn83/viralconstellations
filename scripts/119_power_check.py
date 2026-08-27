#!/usr/bin/env python
"""
119_power_check.py -- is there anything to contrast?

THE DESIGN THIS IS CHECKING
---------------------------
To ask whether a mutation is more likely to appear on one background than
another, the contrast has to exist in the data: the same mutation must appear
on one sufficiently abundant background and NOT on another that was equally
abundant in the same month. Then the mutation's own rate cancels and any
asymmetry is a background effect.

That design has no power if every mutation lives inside a single clade. The
blocks in this model are 0.997 pure -- essentially lineages -- so a mutation
may simply never get the chance to land anywhere else. This script counts,
rather than assumes.

WHAT IS COUNTED
---------------
For each month t and each block k that held at least --min-exposure sequences:

    did any sequence in block k during month t+1 carry mutation m,
    when no sequence in block k during month t did?

That is an APPEARANCE of m on background k. A block that was abundant and did
not pick m up is a NON-appearance -- an informative negative, which is the
whole point of using a census rather than an event log.

The usable pairs are mutations with at least one appearance and at least one
non-appearance, both on blocks above the exposure floor, in the same month.

WHAT THIS DOES NOT ESTABLISH
----------------------------
A non-appearance has three causes and this data cannot separate them: the
mutation never arose, it arose and died before being detectable, or it arose
and was never sequenced. The exposure floor makes the first more likely to
dominate but does not isolate it. So a background effect estimated this way is
a statement about appearance in the record, not about mutation rate.

Blocks are fitted on months up to t and appearance is scored in t+1, so a
block cannot have been defined by the event being scored.
"""
import argparse, importlib.util, sys
from collections import defaultdict
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def assign(E, X, w, model, Pi_row, tv_t, drift):
    """Hard block assignment. Hard rather than soft on purpose: a sequence
    split across two blocks would register as a partial appearance on both,
    which is not what the contrast is about."""
    th, kk = model.theta(tv_t, drift)
    lp = E.loglik_matrix(X, th) + np.log(Pi_row + EPS)[None, :]
    return kk[lp.argmax(1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True,
                    help="window the blocks are fitted on; appearance is then "
                         "scored in each month AFTER it")
    ap.add_argument("--score-months", type=int, default=6,
                    help="how many months after the training window to score")
    ap.add_argument("--max-K", type=int, default=48)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--exposures", default="100,500,2000,8000",
                    help="abundance floors to try. A non-appearance from a "
                         "block holding 4 sequences carries no information; "
                         "from one holding 8,000 it carries a lot")
    args = ap.parse_args()

    E = load_engine(args.engine)
    names, V = E.load_names(args.vocab)
    tr = E.months_in_range(args.train)
    recs = [E.load_month(args.data_dir, ym) for ym in tr]
    if any(r is None for r in recs):
        sys.exit("missing training months")
    Xs, ws = zip(*[E.build(r, V, args.min_count) for r in recs])
    Xs, ws = list(Xs), list(ws)

    later = []
    for h in range(1, args.score_months + 1):
        ym = E.ym_add(tr[-1], h)
        r = E.load_month(args.data_dir, ym)
        if r is None:
            continue
        later.append((ym,) + E.build(r, V, 1))
    if len(later) < 2:
        sys.exit("need at least two months after the window to see appearances")
    print(f"blocks fitted on {tr[0]}..{tr[-1]}; appearance scored in "
          f"{later[0][0]}..{later[-1][0]}")

    # ---- fit once, on the training window only
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
    ks = np.flatnonzero(model.alive)
    print(f"{len(ks)} blocks\n")

    # ---- per scored month: which mutations each block carries, and its weight
    dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.0
    present = []          # present[t][k] = set of mutations seen in block k
    weight = []           # weight[t][k]  = sequences assigned to block k
    for i, (ym, X, w) in enumerate(later):
        z = assign(E, X, w, model, Pi[-1], tv[-1] + (i + 1) * dt, True)
        pres = defaultdict(set); wt = defaultdict(float)
        for k in ks:
            m = z == k
            if not m.any(): continue
            wt[int(k)] = float(w[m].sum())
            cols = np.flatnonzero((X[m] * w[m, None]).sum(0) > 0)
            pres[int(k)] = set(cols.tolist())
        present.append(pres); weight.append(wt)
        print(f"  {ym}: {len(wt)} blocks occupied, "
              f"largest {max(wt.values()) if wt else 0:,.0f} sequences")

    print(f"\n{'=' * 78}\n  IS THERE A CONTRAST TO ESTIMATE?\n{'=' * 78}")
    print(f"\n  {'exposure':>10}{'blocks/mo':>11}{'appear':>9}{'non-appear':>12}"
          f"{'usable muts':>13}{'usable obs':>12}")
    for floor in (float(x) for x in args.exposures.split(",")):
        napp = nnon = 0
        per_mut = defaultdict(lambda: [0, 0])
        nblk = []
        for t in range(len(present) - 1):
            elig = [k for k, v in weight[t].items() if v >= floor
                    and weight[t + 1].get(k, 0.0) >= floor]
            nblk.append(len(elig))
            for k in elig:
                before, after = present[t][k], present[t + 1][k]
                for m in after - before:                 # appeared on k
                    per_mut[m][0] += 1; napp += 1
                # a mutation that appeared on SOME eligible block this month
                # but not on this one is an informative negative
            appeared_now = set()
            for k in elig:
                appeared_now |= (present[t + 1][k] - present[t][k])
            for k in elig:
                gained = present[t + 1][k] - present[t][k]
                for m in appeared_now - gained:
                    if m in present[t][k]:               # already had it
                        continue
                    per_mut[m][1] += 1; nnon += 1
        usable = {m: v for m, v in per_mut.items() if v[0] >= 1 and v[1] >= 1}
        obs = sum(v[0] + v[1] for v in usable.values())
        print(f"  {floor:>10,.0f}{np.mean(nblk) if nblk else 0:>11.1f}"
              f"{napp:>9,}{nnon:>12,}{len(usable):>13,}{obs:>12,}")

    # what the biggest contrasts look like, at the middle exposure floor
    floors = [float(x) for x in args.exposures.split(",")]
    mid = floors[len(floors) // 2]
    per_mut = defaultdict(lambda: [0, 0])
    for t in range(len(present) - 1):
        elig = [k for k, v in weight[t].items() if v >= mid
                and weight[t + 1].get(k, 0.0) >= mid]
        appeared_now = set()
        for k in elig:
            appeared_now |= (present[t + 1][k] - present[t][k])
        for k in elig:
            gained = present[t + 1][k] - present[t][k]
            for m in gained:
                per_mut[m][0] += 1
            for m in appeared_now - gained:
                if m not in present[t][k]:
                    per_mut[m][1] += 1
    top = sorted(((v[0], v[1], m) for m, v in per_mut.items()
                  if v[0] >= 1 and v[1] >= 1), reverse=True)[:15]
    if top:
        print(f"\n  mutations with the most contrast at exposure {mid:,.0f}")
        print(f"    {'mutation':<14}{'landed on':>11}{'skipped':>10}")
        for a, b, m in top:
            print(f"    {names.get(m, str(m)):<14}{a:>11}{b:>10}")
    print("""
  usable muts = mutations that both appeared on at least one abundant block and
  failed to appear on another abundant block in the same month. Only these
  contribute to a design that conditions on mutation identity, because only
  these carry a within-mutation contrast.

  usable muts in the low hundreds or more
      -> the design has power. Fit the conditional model.
  usable muts in the low tens
      -> nearly every mutation lives inside one clade. Blocks are so pure that
         a mutation never gets the chance to land elsewhere, and there is no
         background contrast to estimate. Worth reporting as a finding rather
         than working around.

  Raising the exposure floor makes each negative more informative and cuts the
  number of them. If usable muts collapses as the floor rises, the contrast
  exists only among blocks too small to trust.
""")


if __name__ == "__main__":
    sys.exit(main())
