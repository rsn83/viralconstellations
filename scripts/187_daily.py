#!/usr/bin/env python3
"""
187 -- DOES WITHIN-MONTH TIMING CARRY SIGNAL?

THE QUESTION
------------
All analysis so far uses monthly aggregations. events_v3.tsv has daily
timestamps. Is there signal in WHEN within a month a mutation first appears
that predicts whether it will be in new constellations next month?

Two predictors compared, both evaluated on the same held-out months:

    FREQ   end-of-month attachment frequency          <- current best (179 A1)
    EARLY  fraction of days in the month where the
           mutation appeared, weighted toward early
           appearances (day 1 = weight 1.0, last day
           = weight 0.0, linear decay)

If EARLY beats FREQ, within-month dynamics carry signal that monthly
aggregation discards, and a continuous-time model is warranted.
If EARLY ~ FREQ, the monthly aggregation is lossless for this task.

This is the one direction not yet ruled out by the monthly analyses
(180, 182, 185, 186).

USAGE
    python scripts/187_daily.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --test-end 2025-02 --out results/daily.json

GIT
    git add scripts/187_daily.py
    git commit -m "187: does within-month timing predict new constellations"
    git push
"""

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import datetime

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
    assert recall_at_k(sc, truth, Ks)[10] == 1.0
    rng = np.random.default_rng(1)
    h = [recall_at_k(rng.normal(size=n), truth, Ks)[100] for _ in range(300)]
    assert abs(np.mean(h) - 0.1) < 0.03
    h = [recall_at_k(np.ones(n), truth, Ks, seed=s)[100] for s in range(300)]
    assert abs(np.mean(h) - 0.1) < 0.03
    print("  metric self-test PASSED")


def load_daily(path):
    """Return {date_str: Counter(mutation_id: count)}.
    date_str is the raw date field from the file (e.g. '2024-07-15').
    """
    from collections import Counter
    daily = defaultdict(Counter)
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            date, var, cnt = parts[0].strip(), parts[1].strip(), parts[2].strip()
            try:
                c = int(float(cnt))
            except ValueError:
                continue
            muts = frozenset(var.split(",")) if "," in var else frozenset([var])
            for m in muts:
                daily[date][m] += c
    return daily


def month_of(date_str):
    return date_str[:7]


def early_score(daily, month_str, alpha=0.5):
    """For each mutation, compute a weighted early-appearance score.

    Days are sorted within the month. Weight = exp(-alpha * day_rank/n_days).
    A mutation that appears only on day 1 gets weight ~1.0; one that appears
    only on the last day gets weight ~exp(-alpha) ~ 0.6 at alpha=0.5.

    The score is the sum of weighted counts, normalised so scores sum to 1.
    """
    days = sorted(d for d in daily if month_of(d) == month_str)
    if not days:
        return {}
    n = len(days)
    scores = defaultdict(float)
    for rank, d in enumerate(days):
        w = np.exp(-alpha * rank / max(n - 1, 1))
        for mut, c in daily[d].items():
            scores[mut] += w * c
    tot = sum(scores.values()) or 1.0
    return {m: v / tot for m, v in scores.items()}


def month_attach_freq(pops, months, target_m):
    """Global attachment frequency using months strictly before target_m."""
    from collections import Counter
    counts = Counter()
    prior = [m for m in months if m < target_m]
    for a in range(len(prior) - 1):
        pT = pops.get(prior[a], {})
        pN = pops.get(prior[a + 1], {})
        if not pT or not pN:
            continue
        by_size = defaultdict(list)
        for S in pT:
            by_size[len(S)].append(S)
        for Sn in pN:
            for S in by_size.get(len(Sn) - 1, ()):
                if S < Sn:
                    counts[next(iter(Sn - S))] += 1
    tot = sum(counts.values()) or 1
    return {m: c / tot for m, c in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-bg", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="decay rate for early-appearance weighting")
    ap.add_argument("--test-end", default="2025-02")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    KS = [10, 100, 1000]
    print("metric checks ...")
    _self_test()

    L = load_ladder(args.ladder)
    print("loading monthly ...")
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

    print("loading daily ...")
    daily = load_daily(args.events)
    print(f"  {len(daily):,} distinct dates")

    seen_ever = set()
    for m in all_train:
        seen_ever |= set(pops[m])

    names = ["FREQ", "EARLY"]
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

        p_freq = month_attach_freq(pops, months, m)
        p_early = early_score(daily, m, args.alpha)

        fns = {
            "FREQ":  lambda S, D, w: w * p_freq.get(D, 1e-9),
            "EARLY": lambda S, D, w: w * p_early.get(D, 1e-9),
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
              + "  ".join(f"{nm} {row[f'{nm}@100']:.3f}" for nm in names))

    if not rows:
        print("  NO EVALUABLE MONTHS")
        return

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"\n  months {len(rows)}")
    print(f"\n  {'':10s} {'@10':>8s} {'@100':>8s} {'@1000':>8s}")
    print("  " + "-" * 38)
    for nm in names:
        print(f"  {nm:10s} " + " ".join(f"{avg(f'{nm}@{K}'):8.3f}"
                                        for K in KS))

    g = avg("EARLY@100") - avg("FREQ@100")
    npos = sum(r["EARLY@100"] > r["FREQ@100"] for r in rows)
    print(f"\n  EARLY over FREQ at @100: {g:+.3f}  ({npos}/{len(rows)} months)")
    print("\n  > 0  -> within-month timing carries signal; continuous-time")
    print("         model warranted.")
    print("  ~ 0  -> monthly aggregation is lossless; no benefit from")
    print("         finer time resolution.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"alpha": args.alpha, "test_end": args.test_end,
                       "n_months": len(rows),
                       "recall_freq": {f"@{K}": avg(f"FREQ@{K}") for K in KS},
                       "recall_early": {f"@{K}": avg(f"EARLY@{K}") for K in KS},
                       "gain": g, "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
