#!/usr/bin/env python3
"""
102_what_did_drift_learn.py

Script 101 showed drifting emissions gain about +1.0 in held-out log-likelihood
and raise the ceiling by the same amount. This asks WHAT they learned, using
the same fit -- no new model.

Three questions:

  1  DOES SET SIZE GROW?
     sum_n theta_k,n(t) is the expected number of mutations a sequence from
     background k carries at month t. In the data, sequences go from 1.1
     mutations at the start to 55.5 at the end. If drift is capturing that
     accumulation, this sum should rise. If it is flat, drift gained its +1.0
     from something else and accumulation is still unmodelled.

  2  WHICH MUTATIONS HAVE THE LARGEST SLOPES?
     If 486V, 452R or the residues the literature reports recurring across
     lineages are near the top, drift is tracking real emergence -- which would
     contradict script 96's finding that BA.5 is unreachable by extrapolation,
     and is the most interesting outcome available here. If the large slopes
     are scattered and unrecognisable, drift is fitting gradual background
     change or noise.

  3  WHERE DID THE +1.0 COME FROM?
     Per-mutation decomposition of the held-out gain: for each mutation, how
     much better the test month is explained under drift than under fixed
     emissions. A few mutations carrying most of it means drift found something
     specific; a flat spread means it is a general sharpening.

Usage:
  python 102_what_did_drift_learn.py \
      --data-dir data/processed/full_data_graphs_posres \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --train 2021-06:2022-05 --test 2022-06 --K 8 --seed 0
"""
import argparse, pickle, csv, sys, importlib.util
from pathlib import Path
import numpy as np

EPS = 1e-12
CONVERGENT = {346, 417, 444, 450, 452, 460, 484, 486, 490, 493, 494, 501, 681}


def load_101(path="scripts/101_drift_and_shrinkage.py"):
    for p in (path, "101_drift_and_shrinkage.py",
              str(Path(__file__).parent / "101_drift_and_shrinkage.py")):
        if Path(p).exists():
            spec = importlib.util.spec_from_file_location("s101", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    sys.exit("cannot find 101_drift_and_shrinkage.py -- pass --script")


def load_names(path):
    names, pos, V = {}, {}, 0
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = int(row["node_idx"]); V = max(V, i + 1)
            names[i] = f"{row['aa_pos']}{row['residue'].strip()}"
            pos[i] = int(row["aa_pos"])
    return names, pos, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--script", default="scripts/101_drift_and_shrinkage.py")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    S = load_101(args.script)
    names, pos, V = load_names(args.vocab)
    tr, te = S.months_in_range(args.train), S.months_in_range(args.test)

    Xs, ws = [], []
    for ym in tr:
        X, w, _ = S.build(S.load_month(args.data_dir, ym), V)
        Xs.append(X); ws.append(w)
    T = len(Xs)
    rec = []
    for ym in te: rec += S.load_month(args.data_dir, ym)
    Xte, wte, _ = S.build(rec, V)
    print(f"K = {args.K}   V = {V:,}   train {tr[0]}..{tr[-1]}   test {te[0]}")

    print("\nfitting fixed emissions ...", flush=True)
    base = S.em(Xs, ws, args.K, drift=False, seed=args.seed)
    print("fitting drifting emissions (warm start) ...", flush=True)
    dr = S.em(Xs, ws, args.K, drift=True, seed=args.seed, init=base)

    beta, gamma, Pi, tv = dr["beta"], dr["gamma"], dr["Pi"], dr["tv"]
    dt = tv[-1] - tv[-2] if T > 1 else 0.0
    th_first = S.theta_at(beta, gamma, tv[0], True)
    th_last = S.theta_at(beta, gamma, tv[-1], True)
    th_next = S.theta_at(beta, gamma, tv[-1] + dt, True)
    th_base = S.theta_at(base["beta"], base["gamma"], 0.0, False)

    # ---------------------------------------------------------------- Q1
    print("\n" + "=" * 78)
    print("Q1  DOES THE EXPECTED SET SIZE GROW?")
    print("=" * 78)
    obs = [float((w[:, None] * X).sum() / w.sum()) for X, w in zip(Xs, ws)]
    print(f"\n  observed mutations per sequence: {obs[0]:.1f} at {tr[0]}"
          f"  ->  {obs[-1]:.1f} at {tr[-1]}")
    print(f"\n  {'row':<6}{'share (last)':>14}{'size at ' + tr[0]:>18}"
          f"{'size at ' + tr[-1]:>18}{'change':>10}")
    order = np.argsort(-Pi[-1])
    grew = 0
    for k in order:
        s0, s1 = th_first[k].sum(), th_last[k].sum()
        if s1 > s0: grew += 1
        print(f"  blk{k:<3}{Pi[-1, k]:>14.3f}{s0:>18.1f}{s1:>18.1f}"
              f"{s1 - s0:>+10.1f}")
    # weighted by how much of the population each row holds
    w0 = float((Pi[0] * th_first.sum(1)).sum())
    w1 = float((Pi[-1] * th_last.sum(1)).sum())
    print(f"\n  population-weighted expected size: {w0:.1f} -> {w1:.1f}"
          f"   ({w1 - w0:+.1f})")
    print(f"  rows whose fingerprint grew: {grew} of {args.K}")
    if w1 > w0 + 1:
        print("""
  -> drift IS carrying the accumulation. The rise in mutations per sequence is
     represented inside the backgrounds, not only by mass moving to larger ones.""")
    else:
        print("""
  -> drift is NOT carrying the accumulation. Its gain came from something else,
     and the rise in set size is still unmodelled.""")

    # ---------------------------------------------------------------- Q2
    print("\n" + "=" * 78)
    print("Q2  WHICH MUTATIONS HAVE THE LARGEST SLOPES?")
    print("=" * 78)
    # rank by how much theta actually moved, in probability, for prevalent rows
    move = (th_last - th_first)                      # (K,V)
    wmove = (Pi[-1][:, None] * move)                 # weight by row prevalence
    flat = [(abs(wmove[k, n]), k, n, move[k, n]) for k in range(args.K)
            for n in range(V)]
    flat.sort(reverse=True)
    print(f"\n  {'row':<7}{'mutation':<11}{'theta first':>13}{'theta last':>12}"
          f"{'change':>9}{'convergent':>12}")
    nconv = 0
    for a, k, n, d in flat[:args.top]:
        c = "yes" if pos.get(n, -1) in CONVERGENT else ""
        if c: nconv += 1
        print(f"  blk{k:<4}{names.get(n, n):<11}{th_first[k, n]:>13.3f}"
              f"{th_last[k, n]:>12.3f}{d:>+9.3f}{c:>12}")
    print(f"\n  {nconv} of the top {args.top} are at residues the literature "
          f"reports recurring")
    for target in ("486V", "452R", "704L", "452Q"):
        hits = [(k, th_first[k, n], th_last[k, n], th_next[k, n])
                for n in range(V) if names.get(n) == target
                for k in range(args.K) if Pi[-1, k] > .01]
        for k, a, b, c in hits:
            print(f"    {target:<6} in blk{k:<3} {a:.3f} -> {b:.3f} "
                  f"(forecast next month {c:.3f})")

    # ---------------------------------------------------------------- Q3
    print("\n" + "=" * 78)
    print("Q3  WHERE DID THE HELD-OUT GAIN COME FROM?")
    print("=" * 78)
    lp_b = S.loglik_matrix(Xte, th_base) + np.log(base["Pi"][-1] + EPS)[None, :]
    lp_d = S.loglik_matrix(Xte, th_next) + np.log(Pi[-1] + EPS)[None, :]
    def tot(lp):
        mx = lp.max(1, keepdims=True)
        return (np.log(np.exp(lp - mx).sum(1, keepdims=True)) + mx).ravel()
    gain = float((wte * (tot(lp_d) - tot(lp_b))).sum() / wte.sum())
    print(f"\n  total held-out gain, drift over fixed: {gain:+.3f}")

    # per-mutation: how much each node's emission change contributes
    zb = lp_b.argmax(1); zd = lp_d.argmax(1)
    contrib = np.zeros(V)
    for i in range(Xte.shape[0]):
        x = Xte[i]
        tb = th_base[zb[i]]; td = th_next[zd[i]]
        c = np.where(x > 0, np.log(td + EPS) - np.log(tb + EPS),
                     np.log(1 - td + EPS) - np.log(1 - tb + EPS))
        contrib += wte[i] * c
    contrib /= wte.sum()
    o = np.argsort(-np.abs(contrib))[:args.top]
    print(f"\n  {'mutation':<11}{'contribution':>14}{'share of gain':>16}"
          f"{'convergent':>12}")
    for n in o:
        c = "yes" if pos.get(n, -1) in CONVERGENT else ""
        sh = contrib[n] / gain if abs(gain) > 1e-9 else float("nan")
        print(f"  {names.get(n, n):<11}{contrib[n]:>+14.3f}{sh:>16.1%}{c:>12}")
    top5 = np.abs(contrib)[np.argsort(-np.abs(contrib))[:5]].sum()
    print(f"\n  top 5 mutations account for {top5 / (np.abs(contrib).sum() + EPS):.1%}"
          f" of the total absolute movement")
    print("""
  Concentrated in a few mutations means drift found something specific. Spread
  thinly means it is a general sharpening of the emissions rather than tracking
  a particular lineage.
""")


if __name__ == "__main__":
    main()
