#!/usr/bin/env python3
"""
174 -- DOES CONDITIONING WITHIN THE SET HELP?

The first DIRECT test of the argument this whole line of work rests on.

THE TOY, RESTATED
-----------------
Population: {A,B} at 0.5, {A,C} at 0.5.
Marginals: P(A)=1, P(B)=0.5, P(C)=0.5.
Sample each independently -> {A,B,C} and {A} each get 0.25, and both have
ZERO mass in the population. Marginals do not determine the joint.

Autoregression fixes it with one conditional: p(C | A,B) = 0.
Whether that matters ON THIS DATA has never been measured. 171/172/173 all
predict a SINGLE addition p(D|S) -- one Bernoulli, no joint anywhere.

WHAT THIS MEASURES
------------------
Constellations that appear at T+1 having gained TWO mutations {D1,D2}
relative to a background S present at T. For each, compare:

  INDEPENDENT   p(D1|S) x p(D2|S)          both drawn from the same marginal
  CHAIN         p(D1|S) x p(D2|S+D1)       second conditioned on the first

Same training data, same evaluation months, same candidate construction.
The ONLY difference is whether the second addition sees the first.

  chain > independent  =>  within-set conditioning carries signal.
                           the autoregressive model is warranted.

  chain ~ independent  =>  at this data scale, additions are conditionally
                           independent given the background. the marginal-vs-
                           joint argument is clean on the toy but does not
                           cash out here -- a defensible negative result.

This also reaches RADIUS-2 constellations, which are roughly half of what
the radius-1 pipeline currently misses.

NOTE ON ANCESTRY: a two-mutation extension of several backgrounds generates
one example per background. No claim is made about which was the parent, or
about the order in which D1 and D2 arose -- the ORDER IS IMPOSED by us
(frequency order), and is a modeling choice, not an observation.

USAGE
    python scripts/174_joint.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --test-end 2025-02 --out results/joint.json

GIT
    git add scripts/174_joint.py
    git commit -m "174: chain vs independent -- does within-set conditioning help"
    git push
"""

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict

import numpy as np


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ALPHA = 0.5


class Joint:
    """p(D | S) by similarity-weighted counting, same estimator as 173's P1.

    The point of this script is NOT a better estimator. It is to hold the
    estimator fixed and vary ONLY whether the second addition is conditioned
    on the first.
    """

    def __init__(self, pops, train_months, tau=0.5):
        self.tau = tau
        self.marg = Counter()
        self.attach = defaultdict(Counter)
        self.bg_seen = Counter()
        self._by_mut = defaultdict(set)
        self._cache = {}
        self._fit(pops, train_months)

    def _fit(self, pops, train_months):
        n = max(len(train_months), 1)
        for m in train_months:
            for S, w in pops.get(m, {}).items():
                for i in S:
                    self.marg[i] += w
        for k in self.marg:
            self.marg[k] /= n

        # radius-1 AND radius-2 attachments, so the estimator has seen
        # two-step extensions during training too
        for a in range(len(train_months) - 1):
            pT = pops.get(train_months[a], {})
            pN = pops.get(train_months[a + 1], {})
            if not pT or not pN:
                continue
            by_size = defaultdict(list)
            for S in pT:
                by_size[len(S)].append(S)
            for Sn in pN:
                for S in by_size.get(len(Sn) - 1, ()):
                    if S < Sn:
                        self.attach[S][next(iter(Sn - S))] += 1
                        self.bg_seen[S] += 1
                # two-step: credit BOTH additions to the background, and
                # additionally credit the second to the intermediate S+D1.
                # The intermediate may never have been observed -- that is
                # exactly the point, and why similarity backoff is needed.
                for S in by_size.get(len(Sn) - 2, ()):
                    if S < Sn:
                        d1, d2 = sorted(Sn - S, key=lambda x: -self.marg.get(x, 0))
                        self.attach[S][d1] += 1
                        self.bg_seen[S] += 1
                        mid = S | {d1}
                        self.attach[mid][d2] += 1
                        self.bg_seen[mid] += 1
        for Sbg in self.attach:
            for i in Sbg:
                self._by_mut[i].add(Sbg)

    def _profile(self, S):
        if S in self._cache:
            return self._cache[S]
        num, den = Counter(), 0.0
        cand = set()
        for i in S:
            cand |= self._by_mut.get(i, set())
        for Sbg in cand:
            j = len(S & Sbg) / len(S | Sbg)
            if j < self.tau:
                continue
            for D, c in self.attach[Sbg].items():
                num[D] += j * c
            den += j * self.bg_seen.get(Sbg, 0)
        self._cache[S] = (num, den)
        return num, den

    def logp(self, S, D):
        """log p(D | S). Backs off to the marginal when S has no neighbours."""
        num, den = self._profile(S)
        if den < 1.0:
            return math.log(self.marg.get(D, 0.0) + 1e-9)
        V = max(len(self.marg), 1)
        return math.log((num.get(D, 0.0) + ALPHA) / (den + ALPHA * V))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-bg", type=int, default=300)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--train-window", type=int, default=0)
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    L = load_ladder(args.ladder)
    print("loading ...")
    monthly = L.load_events(args.events)
    pops = {m: L.population(monthly[m]) for m in sorted(monthly)}
    pops = {m: p for m, p in pops.items() if p}
    months = sorted(pops)

    tr_end = L.TRAIN_END[:7]
    te_end = args.test_end or L.TEST_END[:7]
    train_months = [m for m in months if m <= tr_end]
    if args.train_window > 0:
        train_months = train_months[-args.train_window:]
    test_months = [m for m in months if tr_end < m <= te_end]
    print(f"  train {len(train_months)}m | test {len(test_months)}m")

    J = Joint(pops, train_months, args.tau)
    print(f"  {len(J.attach):,} backgrounds with attachment history")

    seen_ever = set()
    for m in [x for x in months if x <= tr_end]:
        seen_ever |= set(pops[m])

    rows = []
    for m in test_months:
        t_ix = months.index(m)
        nxt = t_ix + args.horizon
        if nxt >= len(months):
            break
        pT, pN = pops[m], pops[months[nxt]]
        new = L.new_constellations(pT, pN, seen_ever)
        seen_ever |= set(pT)
        if not new:
            continue

        bgs = sorted(pT.items(), key=lambda kv: -kv[1])[:args.max_bg]
        by_size = defaultdict(list)
        for S, w in bgs:
            by_size[len(S)].append((S, w))

        ind, chn, n2 = [], [], 0
        for Sn in new:
            for S, w in by_size.get(len(Sn) - 2, ()):
                if not S < Sn:
                    continue
                # ORDER IS IMPOSED: more frequent mutation first. This is a
                # modeling choice. The data does not say which arose first.
                d1, d2 = sorted(Sn - S, key=lambda x: -J.marg.get(x, 0))
                lp1 = J.logp(S, d1)
                ind.append(lp1 + J.logp(S, d2))          # both from S
                chn.append(lp1 + J.logp(S | {d1}, d2))   # second sees first
                n2 += 1

        if n2 == 0:
            continue
        rows.append({"month": m, "n_new": len(new), "n_two_step": n2,
                     "independent": -float(np.mean(ind)),
                     "chain": -float(np.mean(chn))})

    if not rows:
        print("\n  NO TWO-STEP EXTENSIONS FOUND.")
        print("  Either radius-2 arrivals are rare at this horizon, or")
        print("  MIN_COUNT is filtering them out. Try lowering it in 171.")
        return

    ind_m = float(np.mean([r["independent"] for r in rows]))
    chn_m = float(np.mean([r["chain"] for r in rows]))
    per = [r["independent"] - r["chain"] for r in rows]
    tot2 = sum(r["n_two_step"] for r in rows)

    print(f"\n  months evaluated       {len(rows)}")
    print(f"  two-step extensions    {tot2:,}")
    print(f"\n  INDEPENDENT  p(D1|S) p(D2|S)        {ind_m:.4f} nats")
    print(f"  CHAIN        p(D1|S) p(D2|S+D1)     {chn_m:.4f} nats"
          f"   gain {ind_m - chn_m:+.4f}")
    print(f"  per-month gain  median {np.median(per):+.3f}  "
          f"min {min(per):+.3f}  max {max(per):+.3f}  "
          f"n_pos {sum(x > 0 for x in per)}/{len(per)}")
    print("\n  gain > 0  =>  within-set conditioning carries signal;")
    print("                the autoregressive model is warranted.")
    print("  gain ~ 0  =>  additions are conditionally independent given the")
    print("                background at this data scale. A negative result,")
    print("                and a defensible one.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"independent": ind_m, "chain": chn_m,
                       "gain": ind_m - chn_m, "n_months": len(rows),
                       "n_two_step": tot2, "test_end": te_end,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
