#!/usr/bin/env python
"""
73_inspect_pairing.py

Runs optimal transport between two months and SHOWS the pairings, in readable
mutation names, so the coupling can be judged by eye instead of trusted.

Prints, for the top-k pairings by transported mass and the bottom-k by cost:
  the source constellation's size and population share
  the matched target's size and share
  the edit distance between them
  exactly which mutations were ADDED and which were REMOVED

Also prints the distribution of pairing distances and how much mass sits on
near-identity pairings, since that is what determines whether a model trained on
these chains has anything to learn.

Node ids are resolved to position+residue through posres_vocab.tsv, so a pairing
reads as "gained 501Y, lost 484K" rather than as integers.

Optionally writes a figure: the transport plan as a heatmap plus a histogram of
pairing distances.

Usage
-----
python scripts/73_inspect_pairing.py --month_a 2021-11 --month_b 2021-12
python scripts/73_inspect_pairing.py --month_a 2022-05 --month_b 2022-06 --k 15
python scripts/73_inspect_pairing.py --self_test
"""

import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")


# ----------------------------------------------------------------------------
# label names
# ----------------------------------------------------------------------------

def load_vocab(path):
    """node_idx -> 'posRES', e.g. 501 + Y -> '501Y'. Falls back to the id."""
    if not path or not os.path.exists(path):
        print(f"(no vocab file at {path}; showing raw node ids)")
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = {c.lower(): c for c in df.columns}
    idc = next((cols[c] for c in ("node_idx", "node", "id", "idx")
                if c in cols), None)
    pc = next((cols[c] for c in ("aa_pos", "pos", "position") if c in cols), None)
    rc = next((cols[c] for c in ("residue", "res", "aa") if c in cols), None)
    if pc is None or rc is None:
        print(f"(could not find position/residue columns in {path})")
        return {}
    out = {}
    for i, row in enumerate(df.itertuples(index=False)):
        d = dict(zip(df.columns, row))
        key = int(d[idc]) if idc else i
        out[key] = f"{str(d[pc]).strip()}{str(d[rc]).strip()}"
    print(f"resolved {len(out)} node names from {os.path.basename(path)}")
    return out


def fmt(labels, names, limit=12):
    if not labels:
        return "-"
    s = sorted(labels, key=lambda l: (int(names.get(l, "0")[:-1])
                                      if l in names else l))
    txt = [names.get(l, str(l)) for l in s]
    if len(txt) > limit:
        return ", ".join(txt[:limit]) + f", +{len(txt) - limit} more"
    return ", ".join(txt)


# ----------------------------------------------------------------------------
# data and OT
# ----------------------------------------------------------------------------

def load_month(data_dir, month, min_count):
    path = os.path.join(data_dir, f"{month}_occupied.pkl")
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")
    with open(path, "rb") as f:
        occ = pickle.load(f)
    return {k: v for k, v in occ.items() if v >= min_count}


def top_sets(occ, max_sets):
    items = sorted(occ.items(), key=lambda kv: -kv[1])[:max_sets]
    w = np.array([v for _, v in items], dtype=float)
    return [c for c, _ in items], w / w.sum()


def edit_cost(a, b):
    labs = sorted({l for s in a for l in s} | {l for s in b for l in s},
                  key=str)
    idx = {l: i for i, l in enumerate(labs)}
    A = np.zeros((len(a), len(labs)), dtype=np.float32)
    B = np.zeros((len(b), len(labs)), dtype=np.float32)
    for i, s in enumerate(a):
        for l in s:
            A[i, idx[l]] = 1.0
    for i, s in enumerate(b):
        for l in s:
            B[i, idx[l]] = 1.0
    return A.sum(1)[:, None] + B.sum(1)[None, :] - 2.0 * (A @ B.T)


def sinkhorn(C, a, b, reg, n_iter=500, tol=1e-9):
    K = np.clip(np.exp(-C / max(reg, 1e-9)), 1e-300, None)
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(n_iter):
        up = u
        u = a / np.clip(K @ v, 1e-300, None)
        v = b / np.clip(K.T @ u, 1e-300, None)
        if np.max(np.abs(u - up)) < tol:
            break
    return u[:, None] * K * v[None, :]


def self_test():
    print("self-test")
    a = [frozenset({1, 2, 3}), frozenset({7, 8, 9})]
    b = [frozenset({1, 2, 3, 4}), frozenset({7, 8})]
    C = edit_cost(a, b)
    assert C[0, 0] == 1 and C[1, 1] == 1 and C[0, 1] == 5 and C[1, 0] == 7, C
    print("  cost matrix is the symmetric difference        ok")

    P = sinkhorn(C / C.max(), np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0.02)
    assert abs(P.sum() - 1.0) < 1e-6
    # each source should send most of its mass to the matching target
    assert P[0, 0] > P[0, 1] and P[1, 1] > P[1, 0], P
    print("  transport favours the low-cost pairing         ok")
    assert np.allclose(P.sum(1), [0.5, 0.5], atol=1e-3)
    assert np.allclose(P.sum(0), [0.5, 0.5], atol=1e-3)
    print("  marginals are respected                       ok")

    names = {1: "10A", 2: "20B", 3: "30C"}
    assert fmt({1, 3}, names) == "10A, 30C"
    assert fmt(set(), names) == "-"
    print("  label formatting                              ok")
    print("all tests passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--vocab", default=None,
                    help="defaults to <data_dir>/posres_vocab.tsv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--month_a", required=False, default="2021-11")
    ap.add_argument("--month_b", required=False, default="2021-12")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--max_sets", type=int, default=300)
    ap.add_argument("--reg", type=float, default=0.02)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--figure", action="store_true",
                    help="also save a heatmap and distance histogram")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    names = load_vocab(args.vocab or
                       os.path.join(args.data_dir, "posres_vocab.tsv"))

    occ_a = load_month(args.data_dir, args.month_a, args.min_count)
    occ_b = load_month(args.data_dir, args.month_b, args.min_count)
    sa, wa = top_sets(occ_a, args.max_sets)
    sb, wb = top_sets(occ_b, args.max_sets)
    print(f"\n{args.month_a}: {len(occ_a)} distinct sets, showing top {len(sa)}")
    print(f"{args.month_b}: {len(occ_b)} distinct sets, showing top {len(sb)}")

    C = edit_cost(sa, sb)
    P = sinkhorn(C / max(C.max(), 1.0), wa, wb, args.reg)
    best = P.argmax(axis=1)

    rows = []
    for i, c in enumerate(sa):
        j = int(best[i])
        t = sb[j]
        rows.append({
            "src_idx": i, "dst_idx": j,
            "src_size": len(c), "dst_size": len(t),
            "src_share": wa[i], "dst_share": wb[j],
            "dist": int(C[i, j]),
            "mass_moved": float(P[i, j]),
            "added": fmt(t - c, names),
            "removed": fmt(c - t, names),
            "n_added": len(t - c), "n_removed": len(c - t),
            "identical": bool(c == t),
        })
    df = pd.DataFrame(rows)
    out = f"{args.out_dir}/73_pairing_{args.month_a}_{args.month_b}.csv"
    df.to_csv(out, index=False)

    def show(sub, title):
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        for _, r in sub.iterrows():
            print(f"  src #{r['src_idx']:<4d} size {r['src_size']:<3d} "
                  f"share {r['src_share']:.4f}   ->   "
                  f"dst #{r['dst_idx']:<4d} size {r['dst_size']:<3d} "
                  f"share {r['dst_share']:.4f}   "
                  f"dist {r['dist']:<3d} mass {r['mass_moved']:.5f}")
            print(f"        added   : {r['added']}")
            print(f"        removed : {r['removed']}")

    show(df.nlargest(args.k, "mass_moved"),
         f"TOP {args.k} PAIRINGS BY TRANSPORTED MASS  "
         f"(where the population actually goes)")
    show(df.nlargest(args.k, "dist"),
         f"WORST {args.k} PAIRINGS BY EDIT DISTANCE  "
         f"(sets with no close partner -- probable arrivals)")

    print("\n" + "=" * 100)
    print("SUMMARY OF THE COUPLING")
    print("=" * 100)
    print(f"pairings: {len(df)}")
    print(f"identical source and target : {df['identical'].mean():.3f} "
          f"of pairings, {df.loc[df['identical'], 'src_share'].sum():.3f} "
          f"of mass")
    print(f"edit distance  : median {df['dist'].median():.0f}   "
          f"mean {df['dist'].mean():.2f}   max {df['dist'].max()}")
    print(f"mass-weighted mean distance : "
          f"{float((df['dist'] * df['src_share']).sum()):.2f}")
    print(f"pairings within 1 edit      : {(df['dist'] <= 1).mean():.3f} "
          f"of pairings, {df.loc[df['dist'] <= 1, 'src_share'].sum():.3f} "
          f"of mass")
    print(f"mutations added per pairing  : mean {df['n_added'].mean():.2f}")
    print(f"mutations removed per pairing: mean {df['n_removed'].mean():.2f}")
    print(f"distinct targets used        : {df['dst_idx'].nunique()} "
          f"of {len(sb)} available")
    print("\n  a high identical share and a low mass-weighted distance mean the")
    print("  coupling is near-identity: it says almost nothing changed, and a")
    print("  model trained on these pairings learns to copy. A long tail of")
    print("  large distances is where the arrivals are -- sets with no")
    print("  plausible local ancestor, which no pairing should really assign.")
    print(f"\nwrote {out}")

    if args.figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
            n = min(120, len(sa), len(sb))
            im = ax[0].imshow(np.log10(P[:n, :n] + 1e-12), aspect="auto",
                              cmap="magma")
            ax[0].set_title(f"transport plan (log10), top {n}")
            ax[0].set_xlabel(args.month_b)
            ax[0].set_ylabel(args.month_a)
            fig.colorbar(im, ax=ax[0])

            ax[1].hist(df["dist"], bins=range(0, int(df["dist"].max()) + 2),
                       color="steelblue")
            ax[1].set_title("pairing edit distance")
            ax[1].set_xlabel("edits between matched sets")

            ax[2].scatter(df["dist"], df["mass_moved"], s=14, alpha=0.6,
                          color="darkred")
            ax[2].set_yscale("log")
            ax[2].set_title("mass moved vs distance")
            ax[2].set_xlabel("edits")
            ax[2].set_ylabel("mass")
            fig.suptitle(f"OT coupling {args.month_a} -> {args.month_b}")
            fig.tight_layout()
            fp = f"{args.out_dir}/73_pairing_{args.month_a}_{args.month_b}.png"
            fig.savefig(fp, dpi=130)
            print(f"wrote {fp}")
        except ImportError:
            print("(matplotlib not available; skipped the figure)")


if __name__ == "__main__":
    main()
