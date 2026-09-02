#!/usr/bin/env python3
"""
176 -- IS COVERAGE OR RANKING THE BINDING CONSTRAINT?

THE QUESTION
------------
175 measured where new constellations sit relative to the population:

    <=1 mutation   0.653
    <=2            0.208   (cumulative 0.861)

The current predictor (173) enumerates radius-1 candidates only, so it is
capped at recall 0.653 however good its scoring is. It achieves recall@100
= 0.239 at the 12-month operating point.

This script adds radius-2 candidates. The ceiling rises 0.653 -> 0.861.

    if recall@100 rises      -> coverage was binding; a model with an
                                explicit size distribution p(k|S) is worth
                                building.
    if recall@100 falls/flat -> ranking is binding; a larger candidate pool
                                makes it worse, and effort belongs in the
                                scorer, not the generator.

BASELINES (state before running)
    frequency, radius-1                recall@100 = 0.049
    counting,  radius-1, 12mo window   recall@100 = 0.239   <- the bar
    ceiling,   radius-1                0.653
    ceiling,   radius-2                0.861

Surprising only if radius-2 beats 0.239 AND the gain comes from k=2 truths
rather than re-ranking k=1 truths. Those are reported separately below; if
the k=1 recall drops while total rises, the pool is diluting the ranking.

HOW RADIUS-2 CANDIDATES ARE RESTRICTED
--------------------------------------
All pairs would be 300 backgrounds x 3335^2 / 2 ~ 1.7e9. Instead, for each
background take the top-M mutations by p(D|S) and form pairs among those.

This restriction is NOT arbitrary. 174 measured additions to be conditionally
independent given the background (-0.003 nats), so

    p({D1,D2} | S) = p(D1|S) p(D2|S)

and the highest-scoring PAIRS are therefore pairs drawn from the highest-
scoring SINGLES. Under the independence result the top-M restriction is
lossless for the top of the ranking. It would not be if additions were
dependent -- so this script's validity rests on 174, and 174 is measured at
radius 2 only.

SCORING
    k=1 candidate (S, D)        w_S * p(k=1) * p(D|S)
    k=2 candidate (S, {D1,D2})  w_S * p(k=2) * p(D1|S) * p(D2|S)

p(k) is the empirical extension-size distribution from the training months.
It is global, not per-background -- a deliberate first cut. A per-background
p(k|S) is the next increment and is NOT tested here.

USAGE
    python scripts/176_radius2.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --train-window 12 --test-end 2025-02 \
        --out results/radius2.json

    Compare against radius-1 only:
    python scripts/176_radius2.py ... --max-k 1 --out results/radius1.json

GIT
    git add scripts/176_radius2.py
    git commit -m "176: radius-2 candidates -- is coverage or ranking binding"
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


ALPHA = 0.5


class Counting:
    """p(D|S) by similarity-weighted counting. Same estimator as 173's P1.

    Also records the empirical extension-size distribution p(k).
    """

    def __init__(self, pops, train_months, tau=0.5, mass_weighted=False):
        self.tau = tau
        self.mw = mass_weighted
        self.marg = Counter()
        self.attach = defaultdict(Counter)
        self.bg_seen = Counter()
        self.ksize = Counter()
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

        for a in range(len(train_months) - 1):
            pT = pops.get(train_months[a], {})
            pN = pops.get(train_months[a + 1], {})
            if not pT or not pN:
                continue
            by_size = defaultdict(list)
            for S, w in pT.items():
                by_size[len(S)].append((S, w))
            for Sn in pN:
                for k in (1, 2):
                    for S, w in by_size.get(len(Sn) - k, ()):
                        if S < Sn:
                            self.ksize[k] += 1
                            inc = w if self.mw else 1.0
                            for D in Sn - S:
                                self.attach[S][D] += inc
                            self.bg_seen[S] += inc * k
        for Sbg in self.attach:
            for i in Sbg:
                self._by_mut[i].add(Sbg)

        tot = sum(self.ksize.values()) or 1
        self.pk = {k: c / tot for k, c in self.ksize.items()}

    def profile(self, S):
        """Returns p(D|S) as (Counter numerator, denominator)."""
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

    def pD(self, S, D):
        num, den = self.profile(S)
        if den < 1.0:
            return self.marg.get(D, 0.0) + 1e-12
        V = max(len(self.marg), 1)
        return (num.get(D, 0.0) + ALPHA) / (den + ALPHA * V)

    def top_additions(self, S, M):
        """Top-M mutations by p(D|S), excluding members of S."""
        num, den = self.profile(S)
        if den < 1.0:
            pool = self.marg.most_common(M * 3)
        else:
            pool = num.most_common(M * 3)
            if len(pool) < M:
                pool = pool + self.marg.most_common(M * 3)
        out = []
        seen = set()
        for D, _ in pool:
            if D in S or D in seen:
                continue
            seen.add(D)
            out.append(D)
            if len(out) >= M:
                break
        return out


# ----------------------------------------------------------------------------
# METRIC UNIT TESTS  (required by the project's standing checks)
# ----------------------------------------------------------------------------

def recall_at_k(scores, truth_idx, Ks, seed=0):
    """Fraction of truths ranked in the top K.

    Ties are broken by tiny random jitter, NOT by argsort's positional
    ordering. That positional ordering is the bug that previously gave a
    baseline which structurally cannot predict appearance a nonzero
    appearance recall. The seed is fixed by default so runs are
    reproducible, and varied in the self-test so tie handling is checked
    in expectation rather than on one draw.
    """
    s = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    s = s + rng.uniform(-1e-12, 1e-12, size=s.shape)
    order = np.argsort(-s)
    rank = np.empty(len(s), dtype=np.int64)
    rank[order] = np.arange(len(s))
    r = rank[np.asarray(sorted(truth_idx))]
    return {K: float(np.mean(r < K)) for K in Ks}, r


def _self_test():
    n, Ks = 1000, [10, 100]
    truth = [3, 17, 900]

    # 1. a perfect oracle must score 1.0
    sc = np.zeros(n); sc[truth] = 1.0
    r, _ = recall_at_k(sc, truth, Ks)
    assert r[10] == 1.0, f"oracle failed: {r}"

    # 2. a random scorer must sit at chance (100/1000 = 0.1)
    rng = np.random.default_rng(1)
    hits = [recall_at_k(rng.normal(size=n), truth, Ks)[0][100]
            for _ in range(300)]
    assert abs(np.mean(hits) - 0.1) < 0.03, \
        f"random not at chance: {np.mean(hits)}"

    # 3. a CONSTANT scorer must also sit at chance. If this fails, ties are
    #    being resolved by array position and any structurally-incapable
    #    baseline will appear to work.
    flat = np.ones(n)
    hits = [recall_at_k(flat, truth, Ks, seed=s)[0][100] for s in range(300)]
    assert abs(np.mean(hits) - 0.1) < 0.03, \
        f"tie handling broken: {np.mean(hits)}"

    print("  metric self-test PASSED (oracle=1.0, random=chance, ties=chance)")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-bg", type=int, default=300)
    ap.add_argument("--max-k", type=int, default=2, help="1 = radius-1 only")
    ap.add_argument("--top-m", type=int, default=60,
                    help="per background, pairs are formed among the top-M "
                         "single additions")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--train-window", type=int, default=12)
    ap.add_argument("--mass-weighted", action="store_true",
                    help="weight attachment counts by background mass")
    ap.add_argument("--test-end", default="2025-02")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    KS = [10, 100, 1000]
    print("metric checks ...")
    _self_test()

    L = load_ladder(args.ladder)
    print("loading ...")
    monthly = L.load_events(args.events)
    pops = {m: L.population(monthly[m]) for m in sorted(monthly)}
    pops = {m: p for m, p in pops.items() if p}
    months = sorted(pops)

    tr_end = L.TRAIN_END[:7]
    train_months = [m for m in months if m <= tr_end]
    if args.train_window > 0:
        train_months = train_months[-args.train_window:]
    test_months = [m for m in months if tr_end < m <= args.test_end]
    print(f"  train {len(train_months)}m | test {len(test_months)}m")

    C = Counting(pops, train_months, args.tau, args.mass_weighted)
    print(f"  {len(C.attach):,} backgrounds | p(k): "
          + ", ".join(f"k={k} {v:.3f}" for k, v in sorted(C.pk.items())))

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

        cands, scores, key = [], [], {}
        for S, w in bgs:
            tops = C.top_additions(S, args.top_m)
            pk1 = C.pk.get(1, 0.5)
            # k = 1 : every mutation, not only the top-M
            for D in C.marg:
                if D in S:
                    continue
                key[(S, frozenset([D]))] = len(cands)
                cands.append((S, frozenset([D])))
                scores.append(w * pk1 * C.pD(S, D))
            # k = 2 : pairs among the top-M (see docstring for why this is
            # lossless for the top of the ranking under 174)
            if args.max_k >= 2:
                pk2 = C.pk.get(2, 0.1)
                for a in range(len(tops)):
                    for b in range(a + 1, len(tops)):
                        add = frozenset([tops[a], tops[b]])
                        key[(S, add)] = len(cands)
                        cands.append((S, add))
                        scores.append(w * pk2 * C.pD(S, tops[a])
                                      * C.pD(S, tops[b]))

        if not cands:
            continue

        truth, truth_k, covered = set(), {}, set()
        for Sn in new:
            for S, _ in bgs:
                if S < Sn and len(Sn - S) <= args.max_k:
                    ix = key.get((S, frozenset(Sn - S)))
                    if ix is not None:
                        truth.add(ix)
                        truth_k[ix] = len(Sn - S)
                        covered.add(Sn)
        if not truth:
            continue

        rk, ranks = recall_at_k(scores, truth, KS)
        tl = sorted(truth)
        k1 = [i for i, ix in enumerate(tl) if truth_k[ix] == 1]
        k2 = [i for i, ix in enumerate(tl) if truth_k[ix] == 2]

        row = {"month": m, "n_new": len(new), "n_cand": len(cands),
               "n_truth": len(truth),
               "coverage": len(covered) / max(len(new), 1),
               "n_truth_k1": len(k1), "n_truth_k2": len(k2)}
        for K in KS:
            row[f"r@{K}"] = rk[K]
            row[f"r@{K}_k1"] = (float(np.mean(ranks[k1] < K)) if k1 else None)
            row[f"r@{K}_k2"] = (float(np.mean(ranks[k2] < K)) if k2 else None)
        rows.append(row)

    if not rows:
        print("  NO EVALUABLE MONTHS")
        return

    def avg(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n  months            {len(rows)}")
    print(f"  candidates/month  {int(np.mean([r['n_cand'] for r in rows])):,}")
    print(f"  coverage          {avg('coverage'):.3f}"
          f"   <- ceiling on recall below")
    print(f"  truths: k=1 {int(np.mean([r['n_truth_k1'] for r in rows]))}"
          f"  k=2 {int(np.mean([r['n_truth_k2'] for r in rows]))} per month")

    print(f"\n  {'':10s} {'@10':>8s} {'@100':>8s} {'@1000':>8s}")
    for lbl, suf in [("all", ""), ("k=1 only", "_k1"), ("k=2 only", "_k2")]:
        cells = " ".join(f"{avg(f'r@{K}{suf}'):8.3f}" for K in KS)
        print(f"  {lbl:10s} {cells}")

    print(f"\n  BASELINE (173, radius-1, 12mo): recall@100 = 0.239")
    print(f"  radius-1 ceiling 0.653 | radius-2 ceiling 0.861")
    print("\n  If total r@100 > 0.239 driven by k=2, coverage was binding.")
    print("  If k=1 r@100 dropped below 0.239, the larger pool is diluting")
    print("  the ranking and the scorer is the constraint.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"max_k": args.max_k, "top_m": args.top_m,
                       "train_window": args.train_window,
                       "mass_weighted": args.mass_weighted,
                       "test_end": args.test_end, "n_months": len(rows),
                       "coverage": avg("coverage"),
                       "recall": {f"@{K}": avg(f"r@{K}") for K in KS},
                       "recall_k1": {f"@{K}": avg(f"r@{K}_k1") for K in KS},
                       "recall_k2": {f"@{K}": avg(f"r@{K}_k2") for K in KS},
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
