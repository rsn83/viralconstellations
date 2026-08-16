#!/usr/bin/env python
"""
35_sinkhorn_two_months.py

Smallest useful OT experiment. CPU, seconds. numpy + pandas only.

WHAT
----
Take constellations at month t and month t+1 as two distributions -- weights
are counts normalised to sum to 1 -- and compute an entropic optimal transport
plan between them, with Jaccard distance between sets as the ground cost.
Then look at where the mass goes.

WHY THIS FIRST
--------------
HM-OT (Halmos et al., RECOMB 2025) couples CLUSTERS rather than individuals,
which is what makes trajectory inference tractable when individuals cannot be
tracked across timepoints. Your constellations already ARE the clusters, so the
supervised case reduces to a plain pairwise OT problem -- no low-rank
factorisation, nothing to learn. This script is that reduced case on one month
pair, to find out whether OT behaves sensibly here before any of the machinery
is worth building.

THE BUILT-IN CHECK
------------------
A constellation present at both t and t+1 has Jaccard distance 0 to itself, so
transport SHOULD route it mostly to itself. If self-transport dominates and the
remaining mass lands on near-neighbours (low Jaccard), OT is doing something
reasonable. If the plan looks arbitrary -- mass scattered onto distant sets --
then either the cost is wrong for this data or OT is not the right tool, and
that is worth knowing before reading further into the method.

MASS IS NORMALISED, NOT CONSERVED IN COUNTS
-------------------------------------------
Balanced OT forces all mass to transfer. Sequencing depth in this dataset
varies ~4000x across months, so transporting raw counts would manufacture
transitions purely to balance the books -- the same failure Halmos et al. note
for the yolk-syncytial layer. Normalising each month to a probability
distribution sidesteps this: the object being modelled is the shift in
FREQUENCIES, which is what the depth confound leaves interpretable anyway.

Usage
-----
  python scripts/35_sinkhorn_two_months.py
  python scripts/35_sinkhorn_two_months.py --month_t 2021-11 --eps 0.01
  python scripts/35_sinkhorn_two_months.py --top_k 800 --cost jaccard
"""

import argparse
import pickle
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


def build_cost(A, B, kind):
    """Ground cost between every pair of constellations. Vectorised via a
    membership matrix, so a 1000x1000 cost is instant."""
    nodes = sorted({m for s in A for m in s} | {m for s in B for m in s})
    idx = {m: i for i, m in enumerate(nodes)}
    MA = np.zeros((len(A), len(nodes)), dtype=np.float32)
    MB = np.zeros((len(B), len(nodes)), dtype=np.float32)
    for i, s in enumerate(A):
        for m in s:
            MA[i, idx[m]] = 1
    for j, s in enumerate(B):
        for m in s:
            MB[j, idx[m]] = 1
    inter = MA @ MB.T
    sa = MA.sum(1)[:, None]
    sb = MB.sum(1)[None, :]
    union = sa + sb - inter
    if kind == "jaccard":
        C = 1.0 - inter / np.maximum(union, 1e-9)
    elif kind == "edit":                      # symmetric difference, normalised
        C = (union - inter) / np.maximum(union, 1e-9) * 0 + (sa + sb - 2 * inter)
        C = C / max(C.max(), 1e-9)
    else:
        raise ValueError(kind)
    return C.astype(np.float64), inter, sa, sb


def sinkhorn(a, b, C, eps, n_iter=2000, tol=1e-9):
    """Entropic OT (Cuturi 2013). Log-domain for numerical stability."""
    logK = -C / eps
    f = np.zeros(len(a))
    g = np.zeros(len(b))
    la, lb = np.log(a + 1e-300), np.log(b + 1e-300)
    for it in range(n_iter):
        f_prev = f
        f = eps * (la - _lse(logK + g[None, :] / eps, axis=1))
        g = eps * (lb - _lse(logK + f[:, None] / eps, axis=0))
        if it % 25 == 0 and np.max(np.abs(f - f_prev)) < tol:
            break
    P = np.exp(logK + f[:, None] / eps + g[None, :] / eps)
    return P, it


def _lse(M, axis):
    mx = M.max(axis=axis, keepdims=True)
    return (mx + np.log(np.exp(M - mx).sum(axis=axis, keepdims=True))).squeeze(axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--month_t", default="2021-11")
    ap.add_argument("--month_t1", default=None, help="defaults to the next month in index")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--max_set_size", type=int, default=40)
    ap.add_argument("--top_k", type=int, default=800,
                    help="keep the K most abundant constellations per month")
    ap.add_argument("--cost", default="jaccard", choices=["jaccard", "edit"])
    ap.add_argument("--eps", type=float, default=0.01,
                    help="entropic regularisation. Smaller = sharper plan but "
                         "slower and less stable. Try 0.05 / 0.01 / 0.005.")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "35_sinkhorn.csv"))
    args = ap.parse_args()

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    mt = args.month_t
    if mt not in months:
        raise SystemExit(f"{mt} not in index; available e.g. {months[10:16]}")
    mt1 = args.month_t1 or months[months.index(mt) + 1]

    def H(mo):
        with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
            raw = constellations_of(pickle.load(fh))
        f = {c: v for c, v in raw.items()
             if v >= args.min_count and 1 <= len(c) <= args.max_set_size}
        top = sorted(f.items(), key=lambda kv: -kv[1])[:args.top_k]
        return [c for c, _ in top], np.array([v for _, v in top], dtype=np.float64)

    A, wa = H(mt)
    B, wb = H(mt1)
    log(f"{mt}: {len(A)} constellations, {wa.sum():.0f} sequences")
    log(f"{mt1}: {len(B)} constellations, {wb.sum():.0f} sequences")
    log(f"cost={args.cost}  eps={args.eps}\n")

    a = wa / wa.sum()
    b = wb / wb.sum()

    C, inter, sa, sb = build_cost(A, B, args.cost)
    log(f"cost matrix {C.shape}, range [{C.min():.3f}, {C.max():.3f}], "
        f"mean {C.mean():.3f}")

    P, iters = sinkhorn(a, b, C, args.eps)
    log(f"sinkhorn converged in ~{iters} iterations, "
        f"total mass {P.sum():.6f} (should be 1.0)")
    log(f"marginal error: rows {np.abs(P.sum(1) - a).max():.2e}  "
        f"cols {np.abs(P.sum(0) - b).max():.2e}")
    log(f"transport cost <P,C> = {(P * C).sum():.4f}\n")

    # ---------- THE CHECK: does a persisting set transport to itself? ----------
    keyA = {c: i for i, c in enumerate(A)}
    pairs = [(keyA[c], j) for j, c in enumerate(B) if c in keyA]
    log("=" * 72)
    log("CHECK 1: SELF-TRANSPORT")
    log("=" * 72)
    log(f"  {len(pairs)} constellations present in BOTH months "
        f"({len(pairs)/len(A):.1%} of month t)")
    if pairs:
        ii = np.array([p[0] for p in pairs])
        jj = np.array([p[1] for p in pairs])
        self_mass = P[ii, jj].sum()
        row_mass = a[ii].sum()
        log(f"  mass on the diagonal (same set -> same set): {self_mass:.4f}")
        log(f"  mass available on those rows:                {row_mass:.4f}")
        log(f"  fraction of persisting mass routed to itself: "
            f"{self_mass/max(row_mass,1e-12):.1%}")
        log("  -> should be HIGH. Jaccard distance to itself is 0, so any")
        log("     sensible plan sends most of this mass straight across.")

    # ---------- CHECK 2: where does the rest go? ----------
    log("\n" + "=" * 72)
    log("CHECK 2: COST OF THE MASS THAT MOVES")
    log("=" * 72)
    Pn = P / P.sum()
    off = P.copy()
    if pairs:
        off[ii, jj] = 0.0
    off_mass = off.sum()
    mean_cost_moved = (off * C).sum() / max(off_mass, 1e-12)
    rnd_cost = float((a[:, None] * b[None, :] * C).sum())
    log(f"  mass NOT on the diagonal: {off_mass:.4f}")
    log(f"  mean Jaccard of that moved mass:      {mean_cost_moved:.4f}")
    log(f"  mean Jaccard under independent coupling: {rnd_cost:.4f}")
    log("  -> moved mass should land on NEARBY sets, so the first number")
    log("     should be clearly BELOW the second. If they are equal, the plan")
    log("     is scattering mass arbitrarily and OT is adding nothing.")

    # ---------- CHECK 3: biggest flows ----------
    log("\n" + "=" * 72)
    log("CHECK 3: LARGEST OFF-DIAGONAL FLOWS")
    log("=" * 72)
    flat = np.argsort(off.ravel())[::-1][:15]
    log(f"  {'mass':>9}{'jacc':>7}{'|t|':>5}{'|t+1|':>7}{'added':>7}{'lost':>6}")
    rows = []
    for f_ in flat:
        i, j = np.unravel_index(f_, off.shape)
        s1, s2 = A[i], B[j]
        added, lost = len(s2 - s1), len(s1 - s2)
        log(f"  {off[i, j]:9.5f}{C[i, j]:7.3f}{len(s1):5d}{len(s2):7d}"
            f"{added:7d}{lost:6d}")
        rows.append(dict(mass=off[i, j], jaccard=C[i, j], size_t=len(s1),
                         size_t1=len(s2), n_added=added, n_lost=lost))
    log("  -> if `added` is mostly 1 and `lost` mostly 0, transport is")
    log("     recovering the one-mutation expansion measured in script 22")
    log("     (62% of new constellations, median one source) WITHOUT being")
    log("     told to look for it. That would be the interesting outcome.")

    if rows:
        d = pd.DataFrame(rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(args.out, index=False)

    # ---------- READ ----------
    log("\n" + "-" * 72)
    log("READ")
    log("-" * 72)
    if pairs:
        frac = self_mass / max(row_mass, 1e-12)
        if frac > 0.7:
            log(f"  Self-transport {frac:.0%}: persisting sets stay put, as they")
            log("  should. The plan is not being distorted by the balanced")
            log("  constraint at this eps.")
        else:
            log(f"  Self-transport only {frac:.0%}. Mass that should stay put is")
            log("  being moved -- either eps is too large (plan too diffuse; try")
            log("  --eps 0.005) or the balanced constraint is forcing transfers.")
    if mean_cost_moved < rnd_cost - 0.05:
        log(f"  Moved mass lands on near-neighbours ({mean_cost_moved:.3f} vs")
        log(f"  {rnd_cost:.3f} independent). OT is finding structure, not noise.")
    else:
        log(f"  Moved mass is no closer than independent coupling")
        log(f"  ({mean_cost_moved:.3f} vs {rnd_cost:.3f}). The plan is not")
        log("  informative -- stop here rather than building on it.")
    log("\n  Next only if both checks pass: repeat across all month pairs and")
    log("  see whether the flows are consistent, then consider the semi-relaxed")
    log("  variant (DeST-OT) which learns growth rates instead of forcing mass")
    log("  conservation.")


if __name__ == "__main__":
    main()
