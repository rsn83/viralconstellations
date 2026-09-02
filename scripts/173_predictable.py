#!/usr/bin/env python3
"""
173 -- IS THE NEXT MONTH PREDICTABLE, AND DOES HISTORY HELP?

Two questions, in order. Nothing else.

  Q1  Given the constellations present in month T, can we predict which NEW
      constellations appear in month T+1?

  Q2  Does knowing months T-1, T-2, ... T-k as well improve that?

WHY recall@K AND NOT log-loss
-----------------------------
171 reported log-loss ratios. A 15x improvement over a null sounds good but
says nothing about whether the prediction is USABLE -- probability spread over
36,000 candidates is 15x better than spread over 500,000 and still useless.

recall@K answers the question a person actually asks:

    of the new constellations that appeared next month,
    how many were in our top K guesses?

That is a number you can say out loud.

THE PREDICTORS
--------------
  P0  frequency only        mass(S) x how common D is overall
                            knows nothing about WHICH constellation D joins
                            -> this is the null. beating it is the minimum.

  P1  background            what got added, historically, to constellations
                            overlapping S by >= tau
                            -> answers Q1

  P2  + recency             same, but recent months weighted more
                            -> does WHEN an attachment happened matter?

  P3  + momentum            P2 scaled by whether S is growing or shrinking
                            -> do growing backgrounds spawn more children?

P2 and P3 are the history increments. Each adds ONE thing so any gain is
attributable. P3 - P1 is the total value of history.

Everything is counting. No model is trained. Fit on months <= TRAIN_END,
evaluated on later months only.

USAGE
    python scripts/173_predictable.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --test-end 2025-02 --out results/predictable.json

GIT
    git add scripts/173_predictable.py
    git commit -m "173: recall@K -- is T+1 predictable from T, and does history add"
    git push
"""

import argparse
import importlib.util
import json
from collections import Counter, defaultdict

import numpy as np


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# THE PREDICTORS
# ----------------------------------------------------------------------------

class Predictors:
    """All four scoring rules, fit by counting on the training months."""

    def __init__(self, pops, train_months, tau=0.5, half_life=6.0):
        self.tau = tau
        self.marg = Counter()                 # how common each mutation is
        self.attach = defaultdict(Counter)    # background -> {D: count}
        self.attach_rec = defaultdict(Counter)  # same, recency-weighted
        self.bg_seen = Counter()
        self.bg_seen_rec = Counter()
        self._by_mut = defaultdict(set)
        self._cache = {}
        self._fit(pops, train_months, half_life)

    def _fit(self, pops, train_months, half_life):
        n = max(len(train_months), 1)
        for m in train_months:
            for S, w in pops.get(m, {}).items():
                for i in S:
                    self.marg[i] += w
        for k in self.marg:
            self.marg[k] /= n

        last = len(train_months) - 1
        for a in range(len(train_months) - 1):
            pT = pops.get(train_months[a], {})
            pN = pops.get(train_months[a + 1], {})
            if not pT or not pN:
                continue
            # recency weight: months further back count for less
            age = last - a
            rw = 0.5 ** (age / half_life)
            by_size = defaultdict(list)
            for S in pT:
                by_size[len(S)].append(S)
            for Sn in pN:
                for S in by_size.get(len(Sn) - 1, ()):
                    if S < Sn:
                        D = next(iter(Sn - S))
                        self.attach[S][D] += 1
                        self.attach_rec[S][D] += rw
                        self.bg_seen[S] += 1
                        self.bg_seen_rec[S] += rw
        for Sbg in self.attach:
            for i in Sbg:
                self._by_mut[i].add(Sbg)

    def _profile(self, S):
        """Aggregate attachment counts over training backgrounds similar to S.

        Similarity is Jaccard >= tau. Computed once per background, not once
        per candidate.
        """
        if S in self._cache:
            return self._cache[S]
        num, num_r = Counter(), Counter()
        den = den_r = 0.0
        cand = set()
        for i in S:
            cand |= self._by_mut.get(i, set())
        for Sbg in cand:
            j = len(S & Sbg) / len(S | Sbg)
            if j < self.tau:
                continue
            for D, c in self.attach[Sbg].items():
                num[D] += j * c
            for D, c in self.attach_rec[Sbg].items():
                num_r[D] += j * c
            den += j * self.bg_seen.get(Sbg, 0)
            den_r += j * self.bg_seen_rec.get(Sbg, 0)
        out = (num, den, num_r, den_r)
        self._cache[S] = out
        return out

    # -- P0 : frequency only (the null) --------------------------------------
    def p0(self, S, D, w, mom):
        return w * self.marg.get(D, 0.0)

    # -- P1 : background -----------------------------------------------------
    def p1(self, S, D, w, mom):
        num, den, _, _ = self._profile(S)
        if den < 1.0:
            return self.p0(S, D, w, mom)
        return w * (num.get(D, 0.0) / den)

    # -- P2 : + recency ------------------------------------------------------
    def p2(self, S, D, w, mom):
        _, _, num_r, den_r = self._profile(S)
        if den_r < 1e-6:
            return self.p0(S, D, w, mom)
        return w * (num_r.get(D, 0.0) / den_r)

    # -- P3 : + momentum -----------------------------------------------------
    def p3(self, S, D, w, mom):
        return self.p2(S, D, w, mom) * mom


# ----------------------------------------------------------------------------
# MOMENTUM  (needs history, so it is computed per evaluation month)
# ----------------------------------------------------------------------------

def momentum(pops, months, t_ix, S, k=3):
    """Is background S growing or shrinking over the last k months?

    Returns a multiplier > 1 for growing, < 1 for shrinking. Growing
    backgrounds host more replication, so should spawn more children --
    that is the hypothesis this tests.
    """
    now = pops[months[t_ix]].get(S, 0.0)
    prev_ix = t_ix - k
    if prev_ix < 0:
        return 1.0
    prev = pops[months[prev_ix]].get(S, 0.0)
    if prev <= 0:
        return 2.0 if now > 0 else 1.0        # newly arrived background
    return float(np.clip(now / prev, 0.25, 4.0))


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-bg", type=int, default=300)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--half-life", type=float, default=6.0)
    ap.add_argument("--mom-k", type=int, default=3)
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--train-window", type=int, default=0,
                    help="use only the last N training months. 0 = all. "
                         "The co-occurrence heatmap shows 2020-21 structure "
                         "is near-orthogonal to the 2024+ test regime, so "
                         "old months may be actively harmful, not just stale.")
    ap.add_argument("--vocab-from-window", action="store_true",
                    help="build candidates from the training-window "
                         "vocabulary instead of all training months. This "
                         "reproduces the CONFOUNDED sweep; off by default.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    KS = [10, 100, 1000]
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

    # 177 showed the 12-month window contains only 697 mutations against
    # 3,335 across all training months. Building candidates from the window
    # vocabulary therefore makes ranking easier AND makes ~15% of new
    # constellations unreachable, confounding the sweep. By default the
    # candidate vocabulary is held fixed at all training months so the
    # sweep varies the ESTIMATOR only.
    vocab_months = ([m for m in months if m <= tr_end]
                    if not args.vocab_from_window else train_months)
    vocab = sorted({i for m in vocab_months for S in pops[m] for i in S})
    print(f"  train {len(train_months)}m | test {len(test_months)}m "
          f"| vocab {len(vocab):,}")

    P = Predictors(pops, train_months, args.tau, args.half_life)
    print(f"  {len(P.attach):,} backgrounds with attachment history")

    names = ["P0 frequency", "P1 background", "P2 +recency", "P3 +momentum"]
    fns = [P.p0, P.p1, P.p2, P.p3]

    seen_ever = set()
    for m in train_months:
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

        cands = L.radius1_candidates(pT, vocab, args.max_bg)
        if not cands:
            continue

        # which candidates are the truth
        truth = set()
        for Sn in new:
            for k, (S, D, _) in enumerate(cands):
                if len(Sn) == len(S) + 1 and S < Sn and D in Sn:
                    truth.add(k)
        if not truth:
            continue

        mom = {S: momentum(pops, months, t_ix, S, args.mom_k)
               for S, _, _ in cands}

        row = {"month": m, "n_new": len(new), "n_truth": len(truth),
               "n_cand": len(cands),
               "coverage": len(truth) / max(len(new), 1)}
        for nm, fn in zip(names, fns):
            sc = np.array([fn(S, D, w, mom[S]) for S, D, w in cands])
            order = np.argsort(-sc)
            rank_of = np.empty(len(sc), dtype=np.int64)
            rank_of[order] = np.arange(len(sc))
            tr = np.array(sorted(truth))
            ranks = rank_of[tr]
            for K in KS:
                row[f"{nm}@{K}"] = float(np.mean(ranks < K))
            row[f"{nm}_medrank"] = float(np.median(ranks))
        rows.append(row)

    if not rows:
        print("  NO EVALUABLE MONTHS")
        return

    print(f"\n  months evaluated  {len(rows)}")
    print(f"  radius-1 coverage {np.mean([r['coverage'] for r in rows]):.3f}"
          "   <- ceiling on any recall below")
    print(f"  candidates/month  {int(np.mean([r['n_cand'] for r in rows])):,}")

    print("\n  Q1: is T+1 predictable from T?")
    print(f"  {'predictor':16s} {'@10':>7s} {'@100':>7s} {'@1000':>7s} "
          f"{'med rank':>10s}")
    print("  " + "-" * 52)
    for nm in names:
        cells = " ".join(f"{np.mean([r[f'{nm}@{K}'] for r in rows]):7.3f}"
                         for K in KS)
        mr = np.mean([r[f"{nm}_medrank"] for r in rows])
        print(f"  {nm:16s} {cells} {mr:10.0f}")

    g1 = np.mean([r["P1 background@100"] - r["P0 frequency@100"] for r in rows])
    g3 = np.mean([r["P3 +momentum@100"] - r["P1 background@100"] for r in rows])
    npos1 = sum(r["P1 background@100"] > r["P0 frequency@100"] for r in rows)
    npos3 = sum(r["P3 +momentum@100"] > r["P1 background@100"] for r in rows)

    print(f"\n  Q1  background over frequency, recall@100  {g1:+.3f}"
          f"   ({npos1}/{len(rows)} months)")
    print(f"  Q2  history over background,   recall@100  {g3:+.3f}"
          f"   ({npos3}/{len(rows)} months)")
    print("\n  Q2 <= 0  =>  the previous month is sufficient; earlier months")
    print("  add nothing, and a trajectory model is not warranted.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"test_end": te_end, "n_months": len(rows),
                       "gain_Q1_at100": float(g1), "gain_Q2_at100": float(g3),
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
