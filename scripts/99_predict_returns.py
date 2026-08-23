#!/usr/bin/env python3
"""
99_predict_returns.py

Tests the 15% fraction that the 18 August breakdown identified as testable but
never tested: constellations that appeared in an earlier month, disappeared,
and came back.

  one step from this month        55%   tested -- features exhausted
  SEEN BEFORE AND RETURNED        15%   <- THIS
  two steps from this month        7%   untested
  one step from an earlier month   3%   untested
  more than two steps, never seen 20%   unreachable

WHY THIS ONE
  Recency was one of only two features found to carry signal, and returning
  constellations are exactly what recency should predict. This is the direct
  test of whether the infinite-to-finite sites transition is PREDICTIVE rather
  than descriptive. The lookback data says 80% of returns come from within two
  months and 95% within five, so the candidate pool is small.

TASK
  At month t, the candidate pool is every constellation observed in months
  t-1..t-L but absent at t. Label = did it reappear at t+1?
  Rank the pool. Report precision@k and recall@k against the base rate.

BASELINE TO BEAT
  frequency when last seen. If time-since-disappearance adds nothing over that,
  recency is not predictive and the finite-sites finding stays descriptive.

FEATURES  (all from months <= t)
  last_freq         its share of sequences in the month it was last seen
  peak_freq         the highest share it ever reached
  months_gone       how many months since it was last seen
  n_appearances     how many separate months it has been seen in
  total_span        months between its first and last appearance
  set_size          number of mutations in it
  mean_node_freq    average current frequency of its mutations
  min_node_freq     the rarest of its mutations, now
  frac_nodes_live   share of its mutations present anywhere this month

Usage:
  python 99_predict_returns.py \
      --data-dir data/processed/full_data_graphs_posres \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --months   2020-06:2024-12 --lookback 5
"""
import argparse, pickle, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np


def ym_add(ym, k):
    y, m = map(int, ym.split("-")); m += k
    y += (m - 1) // 12; m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def months_in_range(spec):
    a, b = spec.split(":") if ":" in spec else (spec, spec)
    out = [a]
    while out[-1] != b:
        out.append(ym_add(out[-1], 1))
        if len(out) > 300: sys.exit("bad range")
    return out


def load_month(data_dir, ym):
    p = Path(data_dir) / f"{ym}_occupied.pkl"
    if not p.exists(): return None
    obj = pickle.load(open(p, "rb"))
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and isinstance(vals[0], (int, np.integer)):
            return [(frozenset(k), int(v)) for k, v in obj.items()]
        return [(frozenset(v), 1) for v in vals]
    items = list(obj)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:
        return [(frozenset(s), int(c)) for s, c in items]
    return [(frozenset(s), 1) for s in items]


def load_V(path):
    n = 0
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n = max(n, int(row["node_idx"]) + 1)
    return n


def fit_logistic(X, y, l2=1.0, iters=500, lr=.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.zeros(X.shape[1]); b = 0.0
    pw = (len(y) - y.sum()) / max(y.sum(), 1)
    sw = np.where(y > 0, pw, 1.0)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        g = sw * (y - p)
        w += lr * (Z.T @ g / sw.sum() - l2 * w / len(y))
        b += lr * g.sum() / sw.sum()
    return w, b, mu, sd


def predict(X, w, b, mu, sd):
    return (X - mu) / sd @ w + b


def pr_at_k(s, y, ks=(10, 50, 100, 500)):
    o = np.argsort(-s); out = []
    for k in ks:
        if k > len(y): continue
        h = y[o[:k]].sum()
        out.append((k, h, h / k, h / max(y.sum(), 1)))
    return out


FEATS = ["last_freq", "peak_freq", "months_gone", "n_appearances", "total_span",
         "set_size", "mean_node_freq", "min_node_freq", "frac_nodes_live"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--months", required=True)
    ap.add_argument("--lookback", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=.34)
    args = ap.parse_args()

    V = load_V(args.vocab)
    months = months_in_range(args.months)
    print(f"V = {V:,}   months {months[0]}..{months[-1]} ({len(months)})   "
          f"lookback = {args.lookback}")

    print("\nloading months ...", flush=True)
    present, freq, nodefreq = [], [], []
    for ym in months:
        r = load_month(args.data_dir, ym)
        if r is None:
            present.append({}); nodefreq.append(np.zeros(V)); continue
        tot = sum(c for _, c in r)
        d = {s: c / tot for s, c in r}
        present.append(d)
        nf = np.zeros(V)
        for s, c in r:
            for n in s:
                if n < V: nf[n] += c / tot
        nodefreq.append(nf)
    print(f"  {sum(len(p) for p in present):,} (month, constellation) observations")

    # running history per constellation, built forward so nothing uses the future
    last_seen, last_f, peak_f, n_app, first_seen = {}, {}, {}, {}, {}

    rows_X, rows_y, rows_m = [], [], []
    for t in range(len(months) - 1):
        cur, nxt = present[t], present[t + 1]
        # candidate pool: seen in t-1..t-L, absent at t
        pool = set()
        for b in range(1, args.lookback + 1):
            if t - b < 0: break
            pool |= set(present[t - b])
        pool -= set(cur)
        if pool:
            nf = nodefreq[t]
            live = nf > 0
            F = []
            for s in pool:
                ls = last_seen.get(s, t - 1)
                idx = [n for n in s if n < V]
                sub = nf[idx] if idx else np.array([0.0])
                F.append([
                    last_f.get(s, 0.0),
                    peak_f.get(s, 0.0),
                    (t - ls) / 6.0,
                    n_app.get(s, 1) / 6.0,
                    (ls - first_seen.get(s, ls)) / 12.0,
                    len(s) / 50.0,
                    sub.mean(),
                    sub.min(),
                    float(live[idx].mean()) if idx else 0.0,
                ])
            pool = list(pool)
            y = np.array([1.0 if s in nxt else 0.0 for s in pool])
            rows_X.append(np.array(F)); rows_y.append(y)
            rows_m.append(np.full(len(pool), t))

        # update history with month t (after building features for t)
        for s, f in cur.items():
            last_seen[s] = t
            last_f[s] = f
            peak_f[s] = max(peak_f.get(s, 0.0), f)
            n_app[s] = n_app.get(s, 0) + 1
            first_seen.setdefault(s, t)

    X = np.vstack(rows_X); y = np.concatenate(rows_y); mth = np.concatenate(rows_m)
    print(f"\ncandidate rows {len(y):,}   returns {int(y.sum()):,} ({y.mean():.2%})")

    cut = int(len(months) * (1 - args.test_frac))
    tr, te = mth < cut, mth >= cut
    print(f"train months 0..{cut-1}  {tr.sum():,} rows, {int(y[tr].sum()):,} returns")
    print(f"test  months {cut}..     {te.sum():,} rows, {int(y[te].sum()):,} returns")
    if y[te].sum() < 20: sys.exit("too few test positives")

    w, b, mu, sd = fit_logistic(X[tr], y[tr])
    s_full = predict(X[te], w, b, mu, sd)
    base = y[te].mean()

    print("\n" + "=" * 62)
    print("FITTED WEIGHTS")
    print("=" * 62)
    for f, wi in sorted(zip(FEATS, w), key=lambda kv: -abs(kv[1])):
        print(f"  {f:<18}{wi:>+8.3f}  {'#'*int(abs(wi)*18)}")

    print("\n" + "=" * 62)
    print(f"HELD-OUT RANKING   base rate {base:.2%}")
    print("=" * 62)

    def show(name, s):
        print(f"\n  {name}")
        print(f"  {'k':>6}{'hits':>7}{'precision':>12}{'recall':>10}{'lift':>8}")
        for k, h, p, r in pr_at_k(s, y[te]):
            print(f"  {k:>6}{int(h):>7}{p:>12.1%}{r:>10.1%}{p/base:>7.1f}x")

    show("FULL MODEL (9 features)", s_full)

    i_lf = FEATS.index("last_freq")
    show("BASELINE -- frequency when last seen", X[te][:, i_lf])

    i_mg = FEATS.index("months_gone")
    show("RECENCY ONLY -- months_gone (negated)", -X[te][:, i_mg])

    hi = [FEATS.index(f) for f in ("last_freq", "peak_freq", "months_gone",
                                    "n_appearances", "total_span")]
    wh, bh, muh, sdh = fit_logistic(X[tr][:, hi], y[tr])
    show("HISTORY ONLY -- no current-month information", predict(X[te][:, hi], wh, bh, muh, sdh))

    ci = [FEATS.index(f) for f in ("mean_node_freq", "min_node_freq",
                                    "frac_nodes_live", "set_size")]
    wc, bc, muc, sdc = fit_logistic(X[tr][:, ci], y[tr])
    show("CURRENT ONLY -- are its mutations circulating now?", predict(X[te][:, ci], wc, bc, muc, sdc))

    print("""

HOW TO READ
  The base rate here is far higher than the 0.1% of the birth task, because the
  candidate pool is small -- only constellations seen in the last few months.
  That makes this an easier problem, which is the point: it is the fraction the
  18 August breakdown left untested.

  The comparison that matters is FULL vs BASELINE (frequency when last seen).
  If the full model does not clearly beat it, then how long a constellation has
  been gone adds nothing over how common it was, and the finite-sites finding
  stays descriptive.

  RECENCY ONLY is the sharpest version of the same question: does time-since-
  disappearance alone rank returns above chance?

  HISTORY vs CURRENT separates two stories. History winning means a
  constellation's own past predicts its return. Current winning means what
  matters is whether its mutations are still circulating in other backgrounds --
  a recombination story rather than a revival one.
""")


if __name__ == "__main__":
    main()
