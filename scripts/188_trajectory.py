#!/usr/bin/env python3
"""
188 -- DOES MUTATION TRAJECTORY SHAPE PREDICT ATTACHMENT?

THE QUESTION
------------
181 tested whether RECENT attachment frequency beats STATIC (pooled history).
It found a small gain (+0.057) that vanished under fixed vocabulary (173
rerun). So frequency LEVEL doesn't carry temporal signal.

This script tests whether the SHAPE of a mutation's frequency trajectory
predicts attachment, BEYOND its current frequency level.

Analogy to time series decomposition:
    full signal  = core(t) + drift(t) + noise(t)
    core(t)      ~ rank-1, deterministic, captured by frequency
    drift(t)     ~ trajectory of residual mutations over time
    noise(t)     ~ singletons, sequencing artefacts

186 showed the residual has no low-rank structure at monthly resolution.
But that measured the residual as a static object, not as a trajectory.
A mutation rising from 1% to 5% to 15% carries directional information
that a snapshot doesn't capture.

WHAT IS COMPARED
----------------
For each candidate (background S, addition D):

    FREQ    w_S * p_attach(D)              current attachment frequency
                                           (= 179's A1, the current best)

    LEVEL   w_S * freq_residual(D, t)      D's residual frequency this month
                                           (how present D is right now,
                                           excluding core mutations)

    TREND   w_S * freq_residual(D, t)
                * growth(D, t, W)          freq times a growth multiplier:
                                           growth = freq(t) / freq(t-W)
                                           clipped to [0.25, 4.0]

    ACCEL   w_S * freq_residual(D, t)
                * growth1 / growth2        freq times trend over trend:
                                           is D accelerating?

READING
    TREND >> FREQ   -> trajectory shape carries signal; a model tracking
                       mutation-level dynamics is warranted.
    TREND ~ FREQ    -> shape adds nothing; level is sufficient; no temporal
                       model beyond counting helps on this data.

If TREND loses, that is the final closure: no temporal signal at any
resolution tested (monthly level: 181; within-month: 187; trajectory: 188).

USAGE
    python scripts/188_trajectory.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --test-end 2025-02 --out results/trajectory.json

GIT
    git add scripts/188_trajectory.py
    git commit -m "188: does residual mutation trajectory shape predict attachment"
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
    assert recall_at_k(sc, truth, Ks)[10] == 1.0
    rng = np.random.default_rng(1)
    h = [recall_at_k(rng.normal(size=n), truth, Ks)[100] for _ in range(300)]
    assert abs(np.mean(h) - 0.1) < 0.03
    h = [recall_at_k(np.ones(n), truth, Ks, seed=s)[100] for s in range(300)]
    assert abs(np.mean(h) - 0.1) < 0.03
    print("  metric self-test PASSED")


def residual_freq(pops, m, core_thresh=0.8):
    """Per-mutation residual frequency at month m.

    Core mutations (present in >= core_thresh of sets) are excluded.
    Returns {mutation: frequency} where frequency sums to 1 over residual.
    """
    sets = list(pops.get(m, {}).keys())
    n = max(len(sets), 1)
    cnt = Counter()
    for S in sets:
        for mut in S:
            cnt[mut] += 1
    core = {mut for mut, c in cnt.items() if c / n >= core_thresh}
    resid = {mut: c / n for mut, c in cnt.items() if mut not in core}
    tot = sum(resid.values()) or 1.0
    return {mut: v / tot for mut, v in resid.items()}


def attach_freq_static(pops, months, target_m):
    """Global attachment frequency from all months before target_m."""
    counts = Counter()
    prior = [m for m in months if m < target_m]
    for a in range(len(prior) - 1):
        pT, pN = pops.get(prior[a], {}), pops.get(prior[a + 1], {})
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
    ap.add_argument("--core-thresh", type=float, default=0.8,
                    dest="core_thresh")
    ap.add_argument("--window", type=int, default=3,
                    help="months back for growth computation")
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

    seen_ever = set()
    for m in all_train:
        seen_ever |= set(pops[m])

    names = ["FREQ", "LEVEL", "TREND", "ACCEL"]
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

        # precompute per-month residual frequencies
        rf_now = residual_freq(pops, m, args.core_thresh)
        prior = [x for x in months if x < m]
        rf_w = residual_freq(pops, prior[-args.window],
                             args.core_thresh) if len(prior) >= args.window \
            else rf_now
        rf_2w = residual_freq(pops, prior[-2 * args.window],
                              args.core_thresh) \
            if len(prior) >= 2 * args.window else rf_w
        p_freq = attach_freq_static(pops, months, m)

        def growth(D, past):
            return float(np.clip(
                (rf_now.get(D, 1e-9)) / (past.get(D, 1e-9)), 0.25, 4.0))

        fns = {
            "FREQ":  lambda S, D, w: w * p_freq.get(D, 1e-9),
            "LEVEL": lambda S, D, w: w * rf_now.get(D, 1e-9),
            "TREND": lambda S, D, w: w * rf_now.get(D, 1e-9)
                                     * growth(D, rf_w),
            "ACCEL": lambda S, D, w: w * rf_now.get(D, 1e-9)
                                     * growth(D, rf_w)
                                     / growth(D, rf_2w) if growth(D, rf_2w) > 0
                                     else w * rf_now.get(D, 1e-9),
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

    print(f"\n  months {len(rows)} | window {args.window}m")
    print(f"\n  {'':8s} {'@10':>8s} {'@100':>8s} {'@1000':>8s}")
    print("  " + "-" * 36)
    for nm in names:
        print(f"  {nm:8s} " + " ".join(f"{avg(f'{nm}@{K}'):8.3f}"
                                       for K in KS))

    g_trend = avg("TREND@100") - avg("FREQ@100")
    g_accel = avg("ACCEL@100") - avg("TREND@100")
    n_trend = sum(r["TREND@100"] > r["FREQ@100"] for r in rows)
    n_accel = sum(r["ACCEL@100"] > r["TREND@100"] for r in rows)

    print(f"\n  TREND over FREQ   {g_trend:+.3f}  ({n_trend}/{len(rows)})")
    print(f"  ACCEL over TREND  {g_accel:+.3f}  ({n_accel}/{len(rows)})")
    print(f"\n  For reference, 179 A1 (attachment freq) = 0.097 @100")
    print("\n  TREND >> FREQ -> trajectory shape predicts attachment;")
    print("  temporal model warranted.")
    print("  TREND ~ FREQ -> shape adds nothing; no temporal signal at")
    print("  any resolution tested (181: level, 187: within-month,")
    print("  188: trajectory shape).")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"window": args.window, "core_thresh": args.core_thresh,
                       "test_end": args.test_end, "n_months": len(rows),
                       "recall": {nm: {f"@{K}": avg(f"{nm}@{K}") for K in KS}
                                  for nm in names},
                       "gain_trend": g_trend, "gain_accel": g_accel,
                       "per_month": rows}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
