#!/usr/bin/env python3
"""
181 -- DOES ATTACHMENT FREQUENCY HAVE TEMPORAL STRUCTURE?

WHERE THIS SITS
---------------
Three things have been ruled out by measurement:

  174  conditioning the second addition on the first     -0.003 nats
  179  conditioning on WHICH background                  untestable
  180  ...because from 2021-09 onward the population is a single lineage
       group (n_components = 1, frac_dissim = 0.000, mean Jaccard 0.90).
       The entire test window sits in that clonal regime.

One thing survived: p_att(D), how often D is the mutation newly ADDED,
predicts better than p_pop(D), how often D is present. +0.059 recall@100
(179, A1 vs A0), with the two correlated at r = 0.72 in log space.

So the object that carries signal is a distribution over mutations, indexed
by time, with no background conditioning. That is a NODE-LEVEL quantity --
the level at which DHyperNodeTPP (AAAI 2025) models events. The difference
is that they adopt node level for scalability; here it is what the data
supports, and 179/180 are the evidence.

THE QUESTION THIS RUN ANSWERS
-----------------------------
Is p_att(D) worth modelling as a time series at all?

  B0  uniform over the vocabulary                  floor
  B1  static: all training months pooled           = A1 in 179, current best
  B2  recent: last W months only
  B3  trend: recent, times a growth factor estimated from the last two
      windows. Tests whether a mutation whose attachment rate is RISING
      keeps rising.

  B2 or B3 >> B1  -> attachment frequency moves in a predictable way. A
                     temporal model (intensity / TPP over mutations) is
                     warranted, and this is the object to build.

  B2 ~= B1        -> attachment frequency is effectively stationary. A
                     static table is already optimal, no temporal model
                     helps, and that is a hard negative worth reporting --
                     it would mean the temporal machinery in HGDHE and
                     DHyperNodeTPP has nothing to grip on in this regime.

Evaluated with the same protocol, months and candidate pool as 179, so the
numbers are directly comparable to A0/A1/A2/A3 there.

USAGE
    python scripts/181_temporal.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --test-end 2025-02 --out results/temporal.json

GIT
    git add scripts/181_temporal.py
    git commit -m "181: is attachment frequency worth modelling as a time series"
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
    h = [recall_at_k(np.ones(n), truth, Ks, seed=s)[100] for s in range(300)]
    assert abs(np.mean(h) - 0.1) < 0.03, f"ties broken: {np.mean(h)}"
    print("  metric self-test PASSED")


def attach_counts(pops, months):
    """Per-month Counter of which mutations were newly added.

    An addition is credited when a constellation at month a+1 is a strict
    superset, by exactly one element, of a constellation at month a. Several
    backgrounds may qualify; all are credited. This is reachability, not
    ancestry.
    """
    out = {}
    for a in range(len(months) - 1):
        pT, pN = pops.get(months[a], {}), pops.get(months[a + 1], {})
        c = Counter()
        if pT and pN:
            by_size = defaultdict(list)
            for S in pT:
                by_size[len(S)].append(S)
            for Sn in pN:
                for S in by_size.get(len(Sn) - 1, ()):
                    if S < Sn:
                        c[next(iter(Sn - S))] += 1
        out[months[a]] = c
    return out


def norm(c, vocab, alpha=0.5):
    tot = sum(c.values()) + alpha * len(vocab)
    return {D: (c.get(D, 0) + alpha) / tot for D in vocab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-bg", type=int, default=300)
    ap.add_argument("--window", type=int, default=3,
                    help="months in the recent window for B2/B3")
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
    all_train = [m for m in months if m <= tr_end]
    test_months = [m for m in months if tr_end < m <= args.test_end]
    vocab = sorted({i for m in all_train for S in pops[m] for i in S})
    print(f"  train {len(all_train)}m | test {len(test_months)}m "
          f"| vocab {len(vocab):,}")

    # per-month attachment counts over the WHOLE series; at each test month
    # only months strictly before it are used, so there is no leakage
    ac = attach_counts(pops, months)

    p_static = norm(sum((ac[m] for m in all_train if m in ac), Counter()),
                    vocab)
    uni = 1.0 / len(vocab)

    seen_ever = set()
    for m in all_train:
        seen_ever |= set(pops[m])

    names = ["B0 uniform", "B1 static", "B2 recent", "B3 trend"]
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

        # windows use only months strictly before m
        prior = [x for x in months if x < m and x in ac]
        recent = prior[-args.window:]
        older = prior[-2 * args.window:-args.window]
        p_recent = norm(sum((ac[x] for x in recent), Counter()), vocab)
        p_older = norm(sum((ac[x] for x in older), Counter()), vocab) \
            if older else p_recent

        # growth factor, clipped so a single count cannot dominate
        growth = {D: float(np.clip(p_recent[D] / (p_older[D] + 1e-12),
                                   0.25, 4.0)) for D in vocab}

        fns = {
            "B0 uniform": lambda S, D, w: w * uni,
            "B1 static": lambda S, D, w: w * p_static[D],
            "B2 recent": lambda S, D, w: w * p_recent[D],
            "B3 trend": lambda S, D, w: w * p_recent[D] * growth[D],
        }

        row = {"month": m, "n_new": len(new), "n_truth": len(truth),
               "n_cand": len(cands)}
        for nm in names:
            rk = recall_at_k([fns[nm](S, D, w) for S, D, w in cands],
                             truth, KS)
            for K in KS:
                row[f"{nm}@{K}"] = rk[K]
        rows.append(row)
        print(f"    {m}  truth {len(truth):4d}  "
              + "  ".join(f"{nm.split()[0]} {row[f'{nm}@100']:.3f}"
                          for nm in names))

    if not rows:
        print("  NO EVALUABLE MONTHS")
        return

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"\n  months {len(rows)} | window {args.window} months")
    print(f"\n  {'':14s} {'@10':>8s} {'@100':>8s} {'@1000':>8s}")
    print("  " + "-" * 42)
    for nm in names:
        print(f"  {nm:14s} " + " ".join(f"{avg(f'{nm}@{K}'):8.3f}"
                                        for K in KS))

    g_rec = avg("B2 recent@100") - avg("B1 static@100")
    g_tr = avg("B3 trend@100") - avg("B2 recent@100")
    n_rec = sum(r["B2 recent@100"] > r["B1 static@100"] for r in rows)
    n_tr = sum(r["B3 trend@100"] > r["B2 recent@100"] for r in rows)

    print(f"\n  recent over static   {g_rec:+.3f}  ({n_rec}/{len(rows)})")
    print(f"  trend over recent    {g_tr:+.3f}  ({n_tr}/{len(rows)})")
    print(f"\n  for reference, 179 on the same months:")
    print(f"    A0 population frequency  0.038 @100")
    print(f"    A1 attachment frequency  0.097 @100")
    print("\n  Both lines ~0 -> attachment frequency is stationary; a static")
    print("  table is optimal and no temporal model helps in this regime.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"window": args.window, "test_end": args.test_end,
                       "n_months": len(rows),
                       "recall": {nm: {f"@{K}": avg(f"{nm}@{K}") for K in KS}
                                  for nm in names},
                       "gain_recent_over_static": g_rec,
                       "gain_trend_over_recent": g_tr,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
