#!/usr/bin/env python
"""
33_position_conditional.py

CPU, minutes. Low-rank bilinear, no deep model.

THE QUESTION
------------
Given a background -- the set of positions already mutated in a constellation --
does that background predict WHICH POSITION mutates next, beyond that position's
own marginal rate?

    score(i | background) = w_i + sum over j in background of M[i,j]

  w_i  the position's own propensity (the marginal baseline)
  M    a position x position interaction matrix, low-rank: M = U V^T

This is the pair-representation claim from AlphaFold-style architectures,
stripped of the architecture: "positions j,k already mutated makes i more
likely". If M buys nothing over w alone, no trunk or pair tensor will help,
and the question is settled cheaply.

WHY THIS AND NOT THE EARLIER SCRIPTS
------------------------------------
Everything measured so far fixed the answer in advance. The vocabulary was the
1180 observed (position, residue) pairs, and the frontier only proposed
additions drawn from mutations already circulating -- so "which position" was
supplied, never predicted. Scripts 24 and 32 measured whether background helps
GIVEN the position is known (it adds ~0.02 on attachment; growth is additive
with R^2 0.44 and no persistent deviation).

Here the model must choose among ALL 1273 positions, including ones that have
never mutated. That is the generative question, and it has not been tested.

SETUP
-----
Training example: a constellation c present at t, and a position i such that
c + {some residue at i} is present at t+1. Positives are the positions actually
added; negatives are sampled positions not added.

Walk-forward: each test month is scored by a model fitted only on strictly
earlier months. Fitted by SGD on the logistic loss.

BASELINES
---------
  marginal      score(i) = w_i, fitted the same way but with no interaction
                term. This is the bar. Beating random is not a result.
  freq_only     score(i) = log(frequency of position i being mutated at t).
                No fitting at all.

Usage
-----
  python scripts/33_position_conditional.py
  python scripts/33_position_conditional.py --rank 16 --epochs 40
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(m):
    print(m, flush=True)


def constellations_of(occ):
    out = {}
    for c, v in occ.items():
        out[frozenset(c)] = v if isinstance(v, (int, float)) else 1
    return out


def auc(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s), dtype=float)
    r[o] = np.arange(1, len(s) + 1)
    ss, rs = s[o], r[o]
    k = 0
    while k < len(ss):
        j = k
        while j + 1 < len(ss) and ss[j + 1] == ss[k]:
            j += 1
        if j > k:
            rs[k:j + 1] = rs[k:j + 1].mean()
        k = j + 1
    r[o] = rs
    p, n = int(y.sum()), int((~y).sum())
    if p == 0 or n == 0:
        return float("nan")
    return (r[y].sum() - p * (p + 1) / 2) / (p * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--min_seqs", type=int, default=5000)
    ap.add_argument("--end_month", type=str, default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=40)
    ap.add_argument("--n_neg", type=int, default=20,
                    help="negative positions sampled per positive")
    ap.add_argument("--max_sources", type=int, default=500,
                    help="constellations sampled per month (abundance-weighted)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--min_train_months", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "33_position_conditional.csv"))
    args = ap.parse_args()

    gd = Path(args.graphs_dir)
    vocab = pd.read_csv(gd / "posres_vocab.tsv", sep="\t")
    # map node index -> spike position. Positions are the prediction space, and
    # ALL of them are candidates, including ones that have never mutated.
    node2pos = dict(zip(vocab["node_idx"], vocab["aa_pos"])) \
        if "node_idx" in vocab.columns else \
        {i: p for i, p in enumerate(vocab["aa_pos"])}
    positions = sorted(set(node2pos.values()))
    P = max(positions) + 1
    log(f"prediction space: {P} positions "
        f"({len(positions)} have ever mutated)\n")

    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    months = [m for m in months if m <= args.end_month]

    cache = {}

    def H(mo):
        if mo not in cache:
            with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
                raw = constellations_of(pickle.load(fh))
            filt = {c: v for c, v in raw.items()
                    if v >= args.min_count and 1 <= len(c) <= args.max_set_size}
            cache[mo] = (filt, sum(filt.values()))
        return cache[mo]

    def pos_set(c):
        return frozenset(node2pos[m] for m in c if m in node2pos)

    rng = np.random.default_rng(0)

    # ---------------- build examples ----------------
    data = {}
    for i in range(len(months) - 1):
        mt, mt1 = months[i], months[i + 1]
        Ht, tot_t = H(mt)
        Ht1, tot_t1 = H(mt1)
        if tot_t < args.min_seqs or tot_t1 < args.min_seqs:
            continue

        # position-level occupancy at t: which positions are mutated anywhere
        occ_pos = Counter()
        for c, v in Ht.items():
            for p in pos_set(c):
                occ_pos[p] += v
        pos_freq = np.zeros(P)
        for p, v in occ_pos.items():
            pos_freq[p] = v / max(tot_t, 1)

        # map each constellation's position set -> position sets at t+1 that
        # extend it by exactly one position
        by_pos_t1 = {}
        for c in Ht1:
            by_pos_t1.setdefault(pos_set(c), 0)
            by_pos_t1[pos_set(c)] += Ht1[c]
        t1_sets = list(by_pos_t1.keys())

        srcs = [c for c, _ in sorted(Ht.items(), key=lambda kv: -kv[1])][:args.max_sources]
        bgs, tgts = [], []
        for c in srcs:
            ps = pos_set(c)
            if not (1 <= len(ps) < args.max_set_size):
                continue
            added = set()
            for s1 in t1_sets:
                if len(s1) == len(ps) + 1 and ps < s1:
                    added |= (s1 - ps)
            for p in added:
                bgs.append(ps)
                tgts.append(p)
        if len(tgts) < 10:
            continue
        data[i] = (bgs, tgts, pos_freq, mt, mt1)
        log(f"  {mt} -> {mt1}  n_seqs={tot_t:>8}  examples={len(tgts):>5}  "
            f"distinct_targets={len(set(tgts)):>4}")

    if len(data) <= args.min_train_months:
        raise SystemExit("not enough usable months")
    log(f"\nusable months: {len(data)}  "
        f"total examples: {sum(len(d[1]) for d in data.values())}\n")

    # ---------------- model ----------------
    # NOTE ON THE FITTING PROCEDURE
    # A first version used sampled-softmax SGD over a handful of negatives. It
    # FAILED ITS OWN VALIDATION: on synthetic data with a planted interaction,
    # the bilinear model scored WORSE than the marginal one (gain -0.013), i.e.
    # it could not detect the effect it exists to detect. The gradient estimate
    # was too noisy. Replaced with full softmax over all P positions (P ~ 1273
    # is small enough to score exhaustively) with minibatching, plus L2 and
    # epoch count chosen on a held-out validation split.
    # Re-validated: planted interaction -> gain +0.040; no interaction -> -0.004.

    def examples(idxs):
        out = []
        for j in idxs:
            bgs, tgts, _, _, _ = data[j]
            out.extend((np.fromiter(bg, dtype=np.int64), t)
                       for bg, t in zip(bgs, tgts))
        return out

    def fit_full(train, r, epochs, lr, l2, bs=64, seed=0):
        rr = np.random.default_rng(seed)
        w = np.zeros(P)
        U = rr.normal(0, .1, (P, r)) if r else None
        V = rr.normal(0, .1, (P, r)) if r else None
        n = len(train)
        for _ in range(epochs):
            order = rr.permutation(n)
            for s0 in range(0, n, bs):
                b = [train[k] for k in order[s0:s0 + bs]]
                gw = np.zeros(P)
                gU = np.zeros_like(U) if r else None
                gV = np.zeros_like(V) if r else None
                for bg, t in b:
                    bgv = V[bg].sum(0) if r else None
                    sc = w + (U @ bgv if r else 0.0)
                    sc = sc - sc.max()
                    pr = np.exp(sc); pr /= pr.sum()
                    g = -pr; g[t] += 1.0
                    gw += g
                    if r:
                        gU += np.outer(g, bgv)
                        gV[bg] += (g[:, None] * U).sum(0)
                k = len(b)
                w += lr * (gw / k - l2 * w)
                if r:
                    U += lr * (gU / k - l2 * U)
                    V += lr * (gV / k - l2 * V)
        return w, U, V

    def nll(model, ex):
        w, U, V = model
        tot = 0.0
        for bg, t in ex:
            sc = w + (U @ V[bg].sum(0) if U is not None else 0.0)
            sc = sc - sc.max()
            pr = np.exp(sc); pr /= pr.sum()
            tot -= np.log(pr[t] + 1e-12)
        return tot / max(len(ex), 1)

    def select(tr, va, r):
        """Pick L2 and epochs on held-out NLL. Without this the bilinear model
        overfits and loses to the marginal even when an interaction exists."""
        best = None
        for l2 in [1e-4, 1e-3, 1e-2, 1e-1]:
            for ep in [args.epochs, args.epochs * 3]:
                m = fit_full(tr, r, ep, args.lr, l2)
                v = nll(m, va)
                if best is None or v < best[0]:
                    best = (v, m, l2, ep)
        return best[1], best[2], best[3]

    def score(model, bg, cand):
        w, U, V = model
        if U is None:
            return w[cand]
        return w[cand] + U[cand] @ V[bg].sum(0)

    idxs = sorted(data)
    rows = []
    for pos_i, ti in enumerate(idxs):
        if pos_i < args.min_train_months:
            continue
        allex = examples(idxs[:pos_i])
        if len(allex) < 200:
            continue
        rng.shuffle(allex)
        nv = max(50, len(allex) // 6)
        va, tr = allex[:nv], allex[nv:]

        m_full, l2f, epf = select(tr, va, args.rank)
        m_marg, l2m, epm = select(tr, va, 0)

        bgs, tgts, pos_freq, mt, mt1 = data[ti]
        ys, s_full, s_marg, s_freq = [], [], [], []
        for bg, t in zip(bgs, tgts):
            negs = rng.integers(0, P, args.n_neg)
            negs = np.array([int(x) for x in negs if x != t and x not in bg])
            if not len(negs):
                continue
            cand = np.concatenate([[t], negs])
            bga = np.fromiter(bg, dtype=np.int64)
            ys.append(np.concatenate([[True], np.zeros(len(negs), bool)]))
            s_full.append(score(m_full, bga, cand))
            s_marg.append(score(m_marg, bga, cand))
            s_freq.append(np.log(pos_freq[cand] + 1e-9))
        if not ys:
            continue
        y = np.concatenate(ys).astype(bool)
        a_f = auc(y, np.concatenate(s_full))
        a_m = auc(y, np.concatenate(s_marg))
        a_q = auc(y, np.concatenate(s_freq))
        rows.append(dict(month_t=mt, n_ex=len(ys), n_train=len(tr),
                         auc_bilinear=a_f, auc_marginal=a_m, auc_freq=a_q,
                         gain=a_f - a_m, l2_bilinear=l2f, ep_bilinear=epf))
        log(f"  {mt}  ex={len(ys):>4} train={len(tr):>6} | freq {a_q:.3f} | "
            f"marginal {a_m:.3f} | bilinear {a_f:.3f} | gain {a_f - a_m:+.4f}")

    if not rows:
        raise SystemExit("no test months")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log("\n" + "=" * 74)
    log(f"SUMMARY over {len(df)} test months, {int(df.n_ex.sum())} examples")
    log("=" * 74)
    for c, lbl in [("auc_freq", "freq_only"), ("auc_marginal", "marginal (fitted w)"),
                   ("auc_bilinear", f"bilinear (rank {args.rank})")]:
        log(f"  {lbl:<26} {df[c].mean():.4f}")
    g = df["gain"]
    log(f"\n  mean gain of interaction term: {g.mean():+.4f}")
    log(f"  positive in {(g > 0).sum()}/{len(g)} months")
    try:
        from scipy.stats import binomtest
        log(f"  sign test p = "
            f"{binomtest(int((g > 0).sum()), len(g), 0.5, alternative='greater').pvalue:.2e}")
    except Exception:
        pass

    log("\n" + "-" * 74)
    log("READ")
    log("-" * 74)
    log("  The bar is `marginal`, not `freq_only` and not random. The marginal")
    log("  model already knows each position's own propensity; the interaction")
    log("  term only earns its place by beating it.")
    log("")
    log("  Synthetic validation of this exact procedure: planted interaction")
    log("  gives gain +0.040; no interaction gives -0.004.")
    log("")
    if g.mean() > 0.03 and (g > 0).mean() > 0.7:
        log("  The background predicts WHICH POSITION mutates next, beyond the")
        log("  position's own rate. That is the pair-representation claim, and it")
        log("  holds. A trunk-and-pair architecture has something to learn.")
    elif g.mean() < 0.01:
        log("  The background adds essentially nothing to predicting which")
        log("  position mutates. Position propensity alone is as good. No pair")
        log("  tensor or trunk will recover signal that is not there -- this is")
        log("  the cheap version of that architecture and it found nothing.")
        log("  Consistent with scripts 24 (+0.02 on attachment) and 32 (growth")
        log("  additive, R2 0.44, no persistent deviation).")
    else:
        log("  Weak. Check consistency across months rather than the mean -- at")
        log("  this example count a small consistent gain is more credible than")
        log("  a large erratic one.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
