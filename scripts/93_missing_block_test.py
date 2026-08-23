#!/usr/bin/env python3
"""
93_missing_block_test.py

Tests one hypothesis, and only one:

    Is there an unmodelled background hiding inside the dominant block?

If a new lineage (e.g. BA.5) emerges in the test month, the model has no row for
it, so its genomes are forced into whichever existing block is nearest. If that
is happening, genomes assigned to that block in the TEST month should fit its
fingerprint measurably worse than genomes assigned to it in TRAINING months.

If the fit is flat, there is no hidden background and A's failure is purely an
estimation problem -- reparameterise A and move on. If the test month fits
clearly worse, the model is missing a block, and no reparameterisation of A can
fix that.

Reads theta / Pi / A saved by script 86 or 91.

Usage:
  python 93_missing_block_test.py \
      --npz results/91_exact.npz \
      --vocab data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --data-dir data/processed/full_data_graphs_posres \
      --train 2021-06:2022-05 --test 2022-06 [--K 8]
"""
import argparse, pickle, csv
from pathlib import Path
import numpy as np

EPS = 1e-12


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
    names, V = {}, 0
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = int(row["node_idx"]); V = max(V, i + 1)
            names[i] = f"{row['aa_pos']}{row['residue'].strip()}"
    return names, V


def build(records, V):
    sets = [s for s, _ in records]
    w = np.array([c for _, c in records], float)
    X = np.zeros((len(sets), V), dtype=np.float32)
    for i, s in enumerate(sets):
        X[i, [n for n in s if 0 <= n < V]] = 1.0
    return X, w, sets


def loglik_matrix(X, theta):
    lt, lc = np.log(theta + EPS), np.log(1 - theta + EPS)
    return X @ (lt - lc).T + lc.sum(1)[None, :]


def assign(X, theta, pi):
    lp = loglik_matrix(X, theta) + np.log(pi + EPS)[None, :]
    return lp.argmax(1), lp


def wq(v, w, q):
    """weighted quantile"""
    o = np.argsort(v); v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--K", type=int, default=0, help="only if npz uses K-prefixed keys")
    args = ap.parse_args()

    d = np.load(args.npz)
    if "theta" in d:
        theta, Pi = d["theta"], d["Pi"]
    else:
        p = f"K{args.K}_"
        if p + "theta" not in d:
            avail = sorted({k.split('_')[0] for k in d.files if k.startswith('K')})
            raise SystemExit(f"pass --K; available: {avail}")
        theta, Pi = d[p + "theta"], d[p + "Pi"]
    names, V = load_names(args.vocab)
    K = theta.shape[0]
    tr, te = months_in_range(args.train), months_in_range(args.test)

    # ---- dominant block in the last training month ----
    kstar = int(np.argmax(Pi[-1]))
    print(f"K = {K},  dominant block in {tr[-1]}: block {kstar} "
          f"({Pi[-1, kstar]:.1%} of that month)")
    print(f"  its top mutations: "
          + ", ".join(names.get(n, str(n)) for n in np.argsort(-theta[kstar])[:12]))

    # ---- per-genome fit to that block, by month ----
    print(f"\n{'month':<10}{'assigned to blk'+str(kstar):>18}{'share':>8}"
          f"{'median log p(S|blk)':>22}{'10th pct':>11}")
    rows = []
    for ym in tr + te:
        X, w, sets = build(load_month(args.data_dir, ym), V)
        pi = Pi[tr.index(ym)] if ym in tr else Pi[-1]
        z, lp = assign(X, theta, pi)
        m = z == kstar
        if m.sum() == 0:
            print(f"{ym:<10}{'0':>18}"); continue
        v, ww = lp[m, kstar], w[m]
        med = wq(v, ww, .5); p10 = wq(v, ww, .1)
        rows.append((ym, ww.sum(), ww.sum() / w.sum(), med, p10, ym in te))
        tag = "  <-- TEST" if ym in te else ""
        print(f"{ym:<10}{ww.sum():>18,.0f}{ww.sum()/w.sum():>8.1%}"
              f"{med:>22.2f}{p10:>11.2f}{tag}")

    tr_rows = [r for r in rows if not r[5]]
    te_rows = [r for r in rows if r[5]]
    if not te_rows or len(tr_rows) < 3:
        raise SystemExit("\nnot enough data to compare")

    # compare test against the training months where this block was substantial
    ref = [r for r in tr_rows if r[2] > 0.2][-3:] or tr_rows[-3:]
    ref_med = np.mean([r[3] for r in ref]); ref_sd = np.std([r[3] for r in ref])
    te_med = te_rows[0][3]
    drop = ref_med - te_med

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"\n  reference (last {len(ref)} training months where block {kstar} "
          f"was >20%): median log p = {ref_med:.2f} +/- {ref_sd:.2f}")
    print(f"  test month {te[0]}:                                 "
          f"median log p = {te_med:.2f}")
    print(f"  drop = {drop:+.2f} nats"
          + (f"   ({drop/ref_sd:.1f} sd)" if ref_sd > 1e-6 else ""))

    if drop > max(1.0, 3 * ref_sd):
        print(f"""
  -> HIDDEN BACKGROUND. Genomes assigned to block {kstar} in the test month fit
     it clearly worse than in training. Something the model has no row for is
     being absorbed into it. Reparameterising A cannot fix this; the model
     needs a way to CREATE a block. GATE = Y.""")
    elif drop < 0.5:
        print(f"""
  -> NO HIDDEN BACKGROUND. Block {kstar} fits the test month about as well as
     training. A's failure is an estimation problem (rank-deficient design,
     backflow into extinct blocks), not a missing block. Reparameterise A --
     growth rates, 2K parameters. GATE = N.""")
    else:
        print(f"""
  -> AMBIGUOUS ({drop:+.2f} nats). Check what is driving it before committing:
     look at the mutations below that are enriched in the test month.""")

    # ---- which mutations are over-represented in the test month? ----
    Xte, wte, _ = build(load_month(args.data_dir, te[0]), V)
    zte, _ = assign(Xte, theta, Pi[-1])
    mte = zte == kstar
    if mte.sum():
        obs = (wte[mte, None] * Xte[mte]).sum(0) / wte[mte].sum()
        exc = obs - theta[kstar]
        top = np.argsort(-exc)[:12]
        print(f"\n  mutations MOST over-represented in block {kstar}'s test-month"
              f" genomes\n  relative to block {kstar}'s own fingerprint:\n")
        print(f"    {'mutation':<12}{'theta[k,n]':>12}{'observed':>11}{'excess':>10}")
        for n in top:
            if exc[n] < 0.02: break
            print(f"    {names.get(n, n):<12}{theta[kstar, n]:>12.3f}"
                  f"{obs[n]:>11.3f}{exc[n]:>+10.3f}")
        print("""
    A coherent group of mutations here, all with large excess, is the
    signature of an unmodelled lineage sitting inside this block. Scattered
    single mutations with small excess are just drift.""")


if __name__ == "__main__":
    main()
