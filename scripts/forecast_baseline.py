#!/usr/bin/env python3
"""
Baseline forecasting harness for constellation populations.

Reads the schema produced by extract_spike.py (or simulate.py):
    date, month, pango, divergence, n_spike, constellation

Builds time windows, then walks forward: train on everything up to T,
predict the population in the next window, score against what actually
happened.  Every model predicts a FULL population (incumbents + novel),
so persistence is a fair competitor rather than a strawman.

The ladder, each rung adding exactly one ingredient:

  M1 persistence                p(T+1) = p(T)
  M2 growth                     + per-constellation trend extrapolation
  M3 growth + flat novelty      + novel mass spread uniformly over
                                  all Hamming-1 children of incumbents
  M4 growth + parent-weighted   + novel mass ∝ parent frequency
  M5 growth + q(z) weighted     + novel mass ∝ parent freq × how often
                                  mutation z has historically been added

Every model shares the same floor for unpredicted constellations, so
log-score differences reflect structure, not floor tuning.
"""

import argparse
import csv
import datetime as dt
import math
import sys
from collections import Counter, defaultdict

import numpy as np


# ----------------------------------------------------------------- data

def load_windows(path, window_days, stride_days, min_seqs):
    """Stream the tsv into non-overlapping-by-stride time windows."""
    dates, cons = [], []
    with open(path) as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        i_date = header.index("date")
        i_cons = header.index("constellation")
        for row in r:
            if len(row) <= i_cons:
                continue
            dates.append(dt.date.fromisoformat(row[i_date]))
            # canonical form: lexicographically sorted token list.  Without
            # this, the same constellation written in two orders counts as
            # two distinct states and every novel-recall number is garbage.
            cons.append(",".join(sorted(row[i_cons].split(","))))

    if not dates:
        sys.exit("no rows loaded")

    order = np.argsort(dates)
    dates = [dates[i] for i in order]
    cons = [cons[i] for i in order]

    t0, t1 = dates[0], dates[-1]
    windows = []
    start = t0
    j = 0
    while start <= t1:
        end = start + dt.timedelta(days=window_days)
        c = Counter()
        k = j
        while k < len(dates) and dates[k] < end:
            if dates[k] >= start:
                c[cons[k]] += 1
            k += 1
        if sum(c.values()) >= min_seqs:
            windows.append((start, end, c))
        start = start + dt.timedelta(days=stride_days)
        while j < len(dates) and dates[j] < start:
            j += 1
    return windows


def as_set(cons_str):
    return frozenset(cons_str.split(",")) if cons_str else frozenset()


# ----------------------------------------------------- model components

def growth_predict(prev, cur, shrink=0.5, clip=1.5):
    """Extrapolate each constellation's frequency one step forward."""
    n_cur = sum(cur.values())
    n_prev = sum(prev.values()) if prev else 0
    a = 1.0 / max(n_cur, 1)

    pred = {}
    for x, k in cur.items():
        p_now = k / n_cur
        if n_prev:
            p_before = prev.get(x, 0) / n_prev
            r = math.log((p_now + a) / (p_before + a))
            r = max(-clip, min(clip, r)) * shrink
        else:
            r = 0.0
        pred[x] = p_now * math.exp(r)
    s = sum(pred.values())
    return {x: v / s for x, v in pred.items()}


def build_candidates(cur_freq, vocab, top_parents):
    """Hamming-1 children of the most frequent incumbents."""
    parents = sorted(cur_freq.items(), key=lambda kv: -kv[1])[:top_parents]
    cands = {}
    for x_str, pf in parents:
        xs = as_set(x_str)
        for z in vocab:
            if z in xs:
                continue
            child = ",".join(sorted(xs | {z}))
            cands.setdefault(child, []).append((pf, z))
    return cands


# ------------------------------------------------------------- scoring

def score_window(pred_inc, pred_nov, floor, obs, seen_before):
    """Mean log score, split by incumbent vs novel observed sequences."""
    tot = sum(obs.values())
    ll_all = ll_inc = ll_nov = 0.0
    n_inc = n_nov = 0
    hit_nov = 0

    for x, k in obs.items():
        p = pred_inc.get(x, 0.0) + pred_nov.get(x, 0.0)
        p = max(p, floor)
        lp = math.log(p)
        ll_all += k * lp
        if x in seen_before:
            ll_inc += k * lp
            n_inc += k
        else:
            ll_nov += k * lp
            n_nov += k
            if x in pred_nov:
                hit_nov += k

    return {
        "ll_all": ll_all / tot,
        "ll_inc": ll_inc / n_inc if n_inc else float("nan"),
        "ll_nov": ll_nov / n_nov if n_nov else float("nan"),
        "novel_frac": n_nov / tot,
        "novel_recall": hit_nov / n_nov if n_nov else float("nan"),
    }


# ---------------------------------------------------------------- main

def run(windows, top_parents, vocab_size, floor_space, verbose):
    # historical novel-mass fraction, used as epsilon
    seen = set()
    novel_fracs = []
    for _, _, c in windows:
        tot = sum(c.values())
        nv = sum(k for x, k in c.items() if x not in seen)
        novel_fracs.append(nv / tot)
        seen.update(c.keys())

    models = ["M1_persist", "M2_growth", "M3_flat", "M4_parent", "M5_q"]
    acc = {m: defaultdict(list) for m in models}

    seen = set()
    add_counts = Counter()          # how often each mutation has been added
    for t in range(len(windows) - 1):
        _, _, cur = windows[t]
        prev = windows[t - 1][2] if t > 0 else None
        _, _, nxt = windows[t + 1]

        # update the "mutation addition" table from what just happened
        for x in cur:
            if x not in seen:
                for z in as_set(x):
                    add_counts[z] += 1
        seen_before = set(seen)
        seen.update(cur.keys())

        if t < 2:                    # need history before predicting
            continue

        n_cur = sum(cur.values())
        cur_freq = {x: k / n_cur for x, k in cur.items()}
        eps = float(np.mean(novel_fracs[max(0, t - 5):t + 1]))
        eps = min(max(eps, 1e-4), 0.5)

        # vocabulary = most frequently added mutations so far
        vocab = [z for z, _ in add_counts.most_common(vocab_size)]
        if not vocab:
            vocab = sorted({z for x in cur for z in as_set(x)})[:vocab_size]

        cands = build_candidates(cur_freq, vocab, top_parents)
        floor = eps / floor_space

        g = growth_predict(prev, cur)

        for name in models:
            if name == "M1_persist":
                inc = {x: (1 - eps) * p for x, p in cur_freq.items()}
                nov = {}
            elif name == "M2_growth":
                inc = {x: (1 - eps) * p for x, p in g.items()}
                nov = {}
            else:
                inc = {x: (1 - eps) * p for x, p in g.items()}
                w = {}
                for child, srcs in cands.items():
                    if name == "M3_flat":
                        w[child] = float(len(srcs))
                    elif name == "M4_parent":
                        w[child] = sum(pf for pf, _ in srcs)
                    else:  # M5_q
                        w[child] = sum(
                            pf * (add_counts.get(z, 0) + 1.0) for pf, z in srcs
                        )
                s = sum(w.values())
                nov = {c: eps * v / s for c, v in w.items()} if s > 0 else {}

            r = score_window(inc, nov, floor, nxt, seen_before)
            for k, v in r.items():
                if not math.isnan(v):
                    acc[name][k].append(v)

        if verbose:
            print(f"  window {t:3d} -> {windows[t+1][0]}  "
                  f"eps={eps:.4f}  cands={len(cands):,}", file=sys.stderr)

    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="tsv from extract_spike.py or simulate.py")
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--stride-days", type=int, default=30)
    ap.add_argument("--min-seqs", type=int, default=50)
    ap.add_argument("--top-parents", type=int, default=200)
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--floor-space", type=float, default=1e7,
                    help="assumed size of unreachable space, for the shared floor")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    windows = load_windows(args.data, args.window_days,
                           args.stride_days, args.min_seqs)
    print(f"{len(windows)} windows, "
          f"{windows[0][0]} .. {windows[-1][1]}", file=sys.stderr)

    acc = run(windows, args.top_parents, args.vocab,
              args.floor_space, args.verbose)

    print()
    hdr = f"{'model':<12} {'logscore':>10} {'incumbent':>10} " \
          f"{'novel':>10} {'nov_recall':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name, d in acc.items():
        print(f"{name:<12} "
              f"{np.mean(d['ll_all']):>10.4f} "
              f"{np.mean(d['ll_inc']):>10.4f} "
              f"{np.mean(d['ll_nov']):>10.4f} "
              f"{np.mean(d['novel_recall']):>11.3f}")
    print(f"\nmean novel mass per window: "
          f"{np.mean(acc['M1_persist']['novel_frac']):.4f}")
    print("(log scores: higher is better)")


if __name__ == "__main__":
    main()
