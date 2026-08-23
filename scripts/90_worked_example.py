#!/usr/bin/env python3
"""
90_worked_example.py

Prints a human-readable walkthrough of the FITTED model, using the real
parameters saved by script 86 -- no invented numbers.

Produces, for a chosen K:
  1. theta   : each block's mutation fingerprint, with real mutation names
  2. Pi      : the fitted monthly composition, as a text trajectory
  3. A       : the fitted transition matrix, block-to-block
  4. a worked scoring of one REAL observed constellation, end to end
  5. the same set scored under independence, to show where they differ

Usage:
  python 90_worked_example.py \
      --npz    results/86_K.npz \
      --vocab  data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --data-dir data/processed/full_data_graphs_posres \
      --train  2021-06:2022-05 --test 2022-06 --K 8
"""
import argparse, csv, pickle
from pathlib import Path
import numpy as np

EPS = 1e-9


def months_in_range(spec):
    if ":" not in spec: return [spec]
    a, b = spec.split(":")
    ya, ma = map(int, a.split("-")); yb, mb = map(int, b.split("-"))
    out, y, m = [], ya, ma
    while (y, m) <= (yb, mb):
        out.append(f"{y:04d}-{m:02d}"); m += 1
        if m == 13: m, y = 1, y + 1
    return out


def load_month(data_dir, ym):
    obj = pickle.load(open(Path(data_dir) / f"{ym}_occupied.pkl", "rb"))
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and isinstance(vals[0], (int, np.integer)):
            return [(frozenset(k), int(v)) for k, v in obj.items()]
        return [(frozenset(v), 1) for v in vals]
    items = list(obj)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:
        return [(frozenset(s), int(c)) for s, c in items]
    return [(frozenset(s), 1) for s in items]


def load_names(path):
    """node_idx -> 'A570D'-style label, using the reference residue if present."""
    names, V = {}, 0
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = int(row["node_idx"]); V = max(V, i + 1)
            names[i] = f"{row['aa_pos']}{row['residue'].strip()}"
    return names, V


def bar(x, width=28, ch="#"):
    return ch * int(round(x * width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--top", type=int, default=8, help="mutations shown per block")
    args = ap.parse_args()

    d = np.load(args.npz)
    pre = f"K{args.K}_"
    if pre + "theta" not in d:
        avail = sorted({k.split("_")[0] for k in d.files if k.startswith("K")})
        raise SystemExit(f"K={args.K} not in {args.npz}. Available: {avail}")
    theta = d[pre + "theta"]; Pi = d[pre + "Pi"]; A = d[pre + "A"]
    theta_F = d["theta_F"][0]
    names, V = load_names(args.vocab)
    months = months_in_range(args.train)
    K = theta.shape[0]

    print("=" * 84)
    print(f"THE FITTED MODEL   K = {K} blocks,  V = {theta.shape[1]:,} mutations,"
          f"  T = {Pi.shape[0]} months")
    print("=" * 84)
    print("""
Every observed genome is one binary vector over the mutation vocabulary.
The model says each genome comes from one of K latent BLOCKS. Given the block,
each mutation is present independently with probability theta[block, mutation].
""")

    # ---------------------------------------------------------------- theta
    print("=" * 84)
    print("1. theta  -- each block's fingerprint   (rows = blocks, cols = mutations)")
    print("=" * 84)
    order = np.argsort(-Pi[-1])                     # most prevalent block first
    for k in order:
        top = np.argsort(-theta[k])[:args.top]
        print(f"\n  block {k:<2}   (last month share {Pi[-1, k]:6.1%})")
        for n in top:
            if theta[k, n] < 0.05: continue
            print(f"      {names.get(n, n):<10} {theta[k, n]:6.3f}  {bar(theta[k, n])}")
        nz = (theta[k] > 0.5).sum()
        print(f"      ... {nz} mutations above 0.5;  expected set size "
              f"= {theta[k].sum():.1f}")

    # ---------------------------------------------------------------- Pi
    print("\n" + "=" * 84)
    print("2. Pi  -- fitted monthly composition   (rows = months, cols = blocks)")
    print("=" * 84)
    print(f"\n  {'month':<10}" + "".join(f"{('blk'+str(k)):>8}" for k in order))
    for t, ym in enumerate(months[:Pi.shape[0]]):
        print(f"  {ym:<10}" + "".join(f"{Pi[t, k]:>8.3f}" for k in order))
    print("\n  same thing as a picture (dominant block per month):")
    for t, ym in enumerate(months[:Pi.shape[0]]):
        k = int(np.argmax(Pi[t]))
        print(f"    {ym}  block {k:<2} {Pi[t,k]:5.1%}  {bar(Pi[t,k], 40)}")

    # ---------------------------------------------------------------- A
    print("\n" + "=" * 84)
    print("3. A  -- fitted transition matrix   (row = block at t, col = block at t+1)")
    print("=" * 84)
    hdr = "from/to"
    print(f"\n  {hdr:<10}" + "".join(f"{('blk'+str(k)):>8}" for k in order))
    for j in order:
        print(f"  {('blk'+str(j)):<10}" + "".join(f"{A[j, k]:>8.3f}" for k in order))
    pin = Pi[-1] @ A; pin = pin / pin.sum()
    print(f"\n  forecast for the month after {months[Pi.shape[0]-1]}:")
    print(f"    persistence (copy last month) : "
          + " ".join(f"blk{k}={Pi[-1,k]:.3f}" for k in order[:4]))
    print(f"    via A                         : "
          + " ".join(f"blk{k}={pin[k]:.3f}" for k in order[:4]))

    # ---------------------------------------------------------------- worked score
    print("\n" + "=" * 84)
    print("4. SCORING ONE REAL CONSTELLATION, END TO END")
    print("=" * 84)
    te = months_in_range(args.test)
    rec = []
    for ym in te: rec += load_month(args.data_dir, ym)
    rec.sort(key=lambda sc: -sc[1])
    S, cnt = rec[0]
    S = frozenset(n for n in S if n < theta.shape[1])
    tot = sum(c for _, c in rec)
    print(f"\n  the most common constellation in {te[0]}: {cnt:,} of {tot:,} genomes"
          f"  ({cnt/tot:.1%})")
    print(f"  |S| = {len(S)} mutations: "
          + ", ".join(names.get(n, str(n)) for n in sorted(S))[:300])

    x = np.zeros(theta.shape[1]); x[sorted(S)] = 1
    print(f"\n  p(S | block) = product over ALL {theta.shape[1]:,} mutations of")
    print( "                 theta[k,n]      if n IS  in S")
    print( "                 1 - theta[k,n]  if n NOT in S")
    lp = np.where(x == 1, np.log(theta + EPS), np.log(1 - theta + EPS)).sum(1)
    print(f"\n  {'block':<8}{'log p(S|blk)':>15}{'pi (last month)':>18}"
          f"{'contribution':>15}{'share':>9}")
    contrib = np.exp(lp - lp.max()) * Pi[-1]
    share = contrib / contrib.sum()
    for k in order:
        print(f"  blk{k:<5}{lp[k]:>15.2f}{Pi[-1, k]:>18.4f}"
              f"{contrib[k]:>15.3e}{share[k]:>9.1%}")
    mx = lp.max()
    ll_mix = np.log((np.exp(lp - mx) * Pi[-1]).sum()) + mx
    print(f"\n  p(S) = sum over blocks           log p(S) = {ll_mix:.2f}")
    print(f"  the model assigns this genome to block {int(np.argmax(share))} "
          f"with {share.max():.1%} responsibility")

    # ---------------------------------------------------------------- vs independence
    print("\n" + "=" * 84)
    print("5. THE SAME SET UNDER INDEPENDENCE  (the baseline being beaten)")
    print("=" * 84)
    ll_ind = float(np.where(x == 1, np.log(theta_F + EPS),
                            np.log(1 - theta_F + EPS)).sum())
    print(f"\n  independence uses ONE profile -- the marginal rate of each mutation,")
    print( "  multiplied as if mutations were unrelated.\n")
    print(f"    log p(S)  mixture      {ll_mix:>10.2f}")
    print(f"    log p(S)  independence {ll_ind:>10.2f}")
    print(f"    difference             {ll_mix - ll_ind:>+10.2f} nats  "
          f"= {np.exp(min(ll_mix - ll_ind, 700)):.3g}x more probable")
    print("""
  Independence has to spread its probability over every COMBINATION of mutations
  that its marginal rates allow, including the vast majority that never occur.
  The mixture concentrates mass on the handful of real backgrounds, so a genuine
  constellation gets far more probability.
""")


if __name__ == "__main__":
    main()
