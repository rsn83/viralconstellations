#!/usr/bin/env python3
"""
179 -- WHAT IS ACTUALLY PRODUCING THE GAIN IN 171/173?

THE PROBLEM WITH THE EXISTING COMPARISON
----------------------------------------
171/173 compare

    P0  mass(S) x  how common D is in the population
    P1  mass(S) x  similarity-weighted attachment counts for backgrounds
                   resembling S

and report P1 > P0, 8/8 months. That gap has been read as evidence for
background-dependent accessibility -- the project's central hypothesis.

But P0 and P1 differ in TWO ways at once, not one:

    (i)  P0 uses POPULATION frequency, P1 uses ATTACHMENT frequency.
         A mutation can be present in most circulating constellations yet
         almost never be the mutation newly added. These are different
         quantities and P1 was never compared against attachment frequency.

    (ii) P1 conditions on WHICH background, P0 does not.

Only (ii) is the hypothesis. If the whole gain comes from (i), the result
says "additions look like past additions", which is a frequency statement
and not background-dependent accessibility.

178 makes this a live worry: attachment distributions of backgrounds at
Jaccard distance < 0.05 already differ by TV 0.78, so similarity carries
much less information than the backoff design assumes.

THE DECOMPOSITION
-----------------
    A0  population frequency          mass(S) x p_pop(D)
    A1  GLOBAL attachment frequency   mass(S) x p_att(D)
            how often D was added to ANY background. No background
            specificity whatsoever.
    A2  similarity-weighted           mass(S) x p_att(D | backgrounds ~ S)
            this is P1 from 171/173.
    A3  SHUFFLED control              A2, but each test background is scored
            using the attachment profile of a DIFFERENT, randomly chosen
            background of the same size. Destroys background specificity
            while preserving the profile's shape, sparsity and count scale.

READING THE RESULT
    A2 >> A1   background specificity is real. The hypothesis survives.
    A2 ~= A1   the gain is attachment frequency, not background. The
               hypothesis is NOT supported by the 171/173 comparison.
    A2 ~= A3   the similarity weighting is doing nothing; any background
               profile works as well as the right one.

A3 is the rank-matched control. An earlier convexity result in this project
died to exactly this kind of null, which is why it is included here rather
than after the fact.

USAGE
    python scripts/179_ablate.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --train-window 12 --test-end 2025-02 --out results/ablate.json

GIT
    git add scripts/179_ablate.py
    git commit -m "179: decompose the gain -- attachment frequency vs background specificity"
    git push
"""

import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict

import numpy as np


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ALPHA = 0.5


def recall_at_k(scores, truth_idx, Ks, seed=0):
    s = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    s = s + rng.uniform(-1e-12, 1e-12, size=s.shape)
    order = np.argsort(-s)
    rank = np.empty(len(s), dtype=np.int64)
    rank[order] = np.arange(len(s))
    return {K: float(np.mean(rank[np.asarray(sorted(truth_idx))] < K))
            for K in Ks}


def _self_test():
    n, Ks, truth = 1000, [10, 100], [3, 17, 900]
    sc = np.zeros(n); sc[truth] = 1.0
    assert recall_at_k(sc, truth, Ks)[10] == 1.0, "oracle failed"
    rng = np.random.default_rng(1)
    h = [recall_at_k(rng.normal(size=n), truth, Ks)[100] for _ in range(300)]
    assert abs(np.mean(h) - 0.1) < 0.03, f"random not chance: {np.mean(h)}"
    flat = np.ones(n)
    h = [recall_at_k(flat, truth, Ks, seed=s)[100] for s in range(300)]
    assert abs(np.mean(h) - 0.1) < 0.03, f"ties broken: {np.mean(h)}"
    print("  metric self-test PASSED")


class Model:
    def __init__(self, pops, train_months, tau=0.5, seed=0):
        self.tau = tau
        self.rng = random.Random(seed)
        self.p_pop = Counter()      # population frequency
        self.p_att = Counter()      # GLOBAL attachment frequency
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
                    self.p_pop[i] += w
        for k in self.p_pop:
            self.p_pop[k] /= n

        for a in range(len(train_months) - 1):
            pT, pN = pops.get(train_months[a], {}), pops.get(train_months[a + 1], {})
            if not pT or not pN:
                continue
            by_size = defaultdict(list)
            for S in pT:
                by_size[len(S)].append(S)
            for Sn in pN:
                for S in by_size.get(len(Sn) - 1, ()):
                    if S < Sn:
                        D = next(iter(Sn - S))
                        self.attach[S][D] += 1
                        self.bg_seen[S] += 1
                        self.p_att[D] += 1
        tot = sum(self.p_att.values()) or 1
        for k in self.p_att:
            self.p_att[k] /= tot

        for Sbg in self.attach:
            for i in Sbg:
                self._by_mut[i].add(Sbg)

        # size-bucketed background list, for the shuffled control
        self._by_size = defaultdict(list)
        for S in self.attach:
            self._by_size[len(S)].append(S)
        self._sizes = sorted(self._by_size)

    def profile(self, S):
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

    def decoy(self, S):
        """A background of similar size, chosen at random. Preserves profile
        shape and count scale while destroying the match to S."""
        if not self._sizes:
            return None
        near = min(self._sizes, key=lambda k: abs(k - len(S)))
        pool = self._by_size[near]
        return self.rng.choice(pool) if pool else None

    # -- the four scorers ---------------------------------------------------
    def a0(self, S, D, w):
        return w * (self.p_pop.get(D, 0.0) + 1e-12)

    def a1(self, S, D, w):
        return w * (self.p_att.get(D, 0.0) + 1e-12)

    def _from_profile(self, num, den, D, w):
        if den < 1.0:
            return self.a1(S=None, D=D, w=w) if False else \
                w * (self.p_att.get(D, 0.0) + 1e-12)
        V = max(len(self.p_pop), 1)
        return w * (num.get(D, 0.0) + ALPHA) / (den + ALPHA * V)

    def a2(self, S, D, w):
        num, den = self.profile(S)
        return self._from_profile(num, den, D, w)

    def a3(self, S, D, w, decoy_cache):
        Sd = decoy_cache.get(S)
        if Sd is None:
            return self.a1(S, D, w)
        num, den = self.profile(Sd)
        return self._from_profile(num, den, D, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-bg", type=int, default=300)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--train-window", type=int, default=12)
    ap.add_argument("--test-end", default="2025-02")
    ap.add_argument("--seed", type=int, default=0)
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
    all_train = [m for m in months if m <= tr_end]
    train_months = all_train[-args.train_window:] if args.train_window > 0 \
        else all_train
    test_months = [m for m in months if tr_end < m <= args.test_end]
    # vocabulary from ALL training months, never the window (177)
    vocab = sorted({i for m in all_train for S in pops[m] for i in S})
    print(f"  train {len(train_months)}m | test {len(test_months)}m "
          f"| vocab {len(vocab):,}")

    M = Model(pops, train_months, args.tau, args.seed)
    print(f"  {len(M.attach):,} backgrounds with attachments")

    # how different are population and attachment frequency?
    common = set(M.p_pop) & set(M.p_att)
    if common:
        xs = np.array([M.p_pop[k] for k in common])
        ys = np.array([M.p_att[k] for k in common])
        r = float(np.corrcoef(np.log(xs + 1e-9), np.log(ys + 1e-9))[0, 1])
        print(f"  corr(log p_pop, log p_att) = {r:.3f} over "
              f"{len(common):,} shared mutations")

    seen_ever = set()
    for m in all_train:
        seen_ever |= set(pops[m])

    names = ["A0 pop-freq", "A1 attach-freq", "A2 background",
             "A3 shuffled"]
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
        index = {(S, D): k for k, (S, D, _) in enumerate(cands)}
        truth = set()
        for Sn in new:
            for (S, D, _) in cands:
                if len(Sn) == len(S) + 1 and S < Sn and D in Sn:
                    truth.add(index[(S, D)])
        if not truth:
            continue

        decoys = {S: M.decoy(S) for S, _, _ in cands}
        fns = [M.a0, M.a1, M.a2,
               lambda S, D, w: M.a3(S, D, w, decoys)]

        row = {"month": m, "n_new": len(new), "n_truth": len(truth),
               "n_cand": len(cands)}
        for nm, fn in zip(names, fns):
            rk = recall_at_k([fn(S, D, w) for S, D, w in cands], truth, KS)
            for K in KS:
                row[f"{nm}@{K}"] = rk[K]
        rows.append(row)

    if not rows:
        print("  NO EVALUABLE MONTHS")
        return

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"\n  months {len(rows)} | candidates/month "
          f"{int(np.mean([r['n_cand'] for r in rows])):,}")
    print(f"\n  {'':16s} {'@10':>8s} {'@100':>8s} {'@1000':>8s}")
    print("  " + "-" * 44)
    for nm in names:
        print(f"  {nm:16s} " + " ".join(f"{avg(f'{nm}@{K}'):8.3f}"
                                        for K in KS))

    g_att = avg("A1 attach-freq@100") - avg("A0 pop-freq@100")
    g_bg = avg("A2 background@100") - avg("A1 attach-freq@100")
    g_shuf = avg("A2 background@100") - avg("A3 shuffled@100")
    npos_bg = sum(r["A2 background@100"] > r["A1 attach-freq@100"]
                  for r in rows)
    npos_shuf = sum(r["A2 background@100"] > r["A3 shuffled@100"]
                    for r in rows)

    print(f"\n  attachment freq over population freq  {g_att:+.3f}")
    print(f"  background over attachment freq       {g_bg:+.3f}"
          f"   ({npos_bg}/{len(rows)} months)   <- THE HYPOTHESIS")
    print(f"  background over shuffled control      {g_shuf:+.3f}"
          f"   ({npos_shuf}/{len(rows)} months)   <- rank-matched null")
    print("\n  If the second and third lines are ~0, the 171/173 gain is")
    print("  attachment frequency, not background-dependent accessibility.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"train_window": args.train_window,
                       "test_end": args.test_end, "n_months": len(rows),
                       "recall": {nm: {f"@{K}": avg(f"{nm}@{K}") for K in KS}
                                  for nm in names},
                       "gain_attach_over_pop": g_att,
                       "gain_bg_over_attach": g_bg,
                       "gain_bg_over_shuffled": g_shuf,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
