#!/usr/bin/env python
"""
36_ot_sweep.py

Corrects the diagnostic in script 35 and runs it across every month pair.
CPU, a few minutes. numpy + pandas only.

WHAT SCRIPT 35 GOT WRONG
------------------------
Check 2 compared the mean cost of MOVED mass against the mean cost under an
independent coupling. But the independent baseline was computed over ALL pairs
including the diagonal -- every zero-cost self-pair -- which drags it down
artificially. So the comparison was rigged against the transport plan, and the
verdict printed "not informative" for 2022-08, where the fifteen largest flows
were `added=1, lost=0` at Jaccard 0.033. Clean one-mutation steps, called a
failure by a broken metric.

Fixed here: the independent baseline is computed over OFF-DIAGONAL pairs only,
matching what the moved mass is actually being compared against.

WHAT THIS ADDS
--------------
Script 35 showed fifteen rows and left the pattern to the eye. This quantifies
it: of all transported mass that moves, what fraction lands at

  ADD-1     exactly one mutation added, none lost   (s -> s + {m})
  ADD-k     k added, none lost
  SWAP      some added AND some lost
  LOSS      only mutations lost
  FAR       Jaccard above a threshold; unrelated

ADD-1 is the quantity of interest. Script 22 measured, model-free, that ~62%
of newly-appearing constellations are one addition from a circulating set with
a median of exactly one source. If transport independently routes most moving
mass along ADD-1 steps, it is recovering that structure from the geometry
alone rather than being told to look for it. On synthetic data with planted
one-mutation expansions, Sinkhorn recovered 18-20 of 20; on unplanted data it
did not.

INTERPRETING SELF-TRANSPORT
---------------------------
Balanced OT must move all mass somewhere, so when many sets die the survivors
give up mass to cover the displacement. Self-transport therefore tracks
turnover and is not by itself a failure: 84% in quiet Alpha (2021-03), 67%
mid-Omicron, 54% across the Omicron sweep (2021-11), where Delta's mass had
nowhere near to go and was forced onto Omicron at Jaccard 0.9. That is the
motivation for semi-relaxed OT (DeST-OT), which lets mass be created and
destroyed instead of forcing transfer.

Usage
-----
  python scripts/36_ot_sweep.py
  python scripts/36_ot_sweep.py --top_k 600 --eps 0.01
  python scripts/36_ot_sweep.py --month_t 2022-08     # single pair, verbose
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


def _lse(M, axis):
    mx = M.max(axis=axis, keepdims=True)
    return (mx + np.log(np.exp(M - mx).sum(axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn(a, b, C, eps, n_iter=1500, tol=1e-9):
    logK = -C / eps
    f = np.zeros(len(a)); g = np.zeros(len(b))
    la, lb = np.log(a + 1e-300), np.log(b + 1e-300)
    for it in range(n_iter):
        fp = f
        f = eps * (la - _lse(logK + g[None, :] / eps, axis=1))
        g = eps * (lb - _lse(logK + f[:, None] / eps, axis=0))
        if it % 25 == 0 and np.max(np.abs(f - fp)) < tol:
            break
    return np.exp(logK + f[:, None] / eps + g[None, :] / eps), it


def membership(A, B):
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
    return MA, MB


def analyse(A, wa, B, wb, eps, far=0.5, verbose=False):
    a = wa / wa.sum(); b = wb / wb.sum()
    MA, MB = membership(A, B)
    inter = MA @ MB.T
    sa = MA.sum(1)[:, None]; sb = MB.sum(1)[None, :]
    union = sa + sb - inter
    C = (1.0 - inter / np.maximum(union, 1e-9)).astype(np.float64)
    added = (sb - inter)          # mutations in B[j] not in A[i]
    lost = (sa - inter)           # mutations in A[i] not in B[j]

    P, iters = sinkhorn(a, b, C, eps)

    keyA = {c: i for i, c in enumerate(A)}
    pairs = [(keyA[c], j) for j, c in enumerate(B) if c in keyA]
    diag = np.zeros(P.shape, dtype=bool)
    if pairs:
        ii = np.array([p[0] for p in pairs]); jj = np.array([p[1] for p in pairs])
        diag[ii, jj] = True

    self_mass = P[diag].sum()
    row_mass = a[[p[0] for p in pairs]].sum() if pairs else np.nan
    off = np.where(diag, 0.0, P)
    off_mass = off.sum()

    # CORRECTED BASELINE: independent coupling restricted to OFF-DIAGONAL pairs.
    # Script 35 included the diagonal, whose cost is 0, which depressed the
    # baseline and made any real plan look bad by comparison.
    ind = a[:, None] * b[None, :]
    ind_off = np.where(diag, 0.0, ind)
    ind_off_mass = ind_off.sum()
    base_off = (ind_off * C).sum() / max(ind_off_mass, 1e-12)
    moved_cost = (off * C).sum() / max(off_mass, 1e-12)

    # edit-type decomposition of the moving mass
    is_add1 = (added == 1) & (lost == 0)
    is_addk = (added >= 2) & (lost == 0)
    is_swap = (added >= 1) & (lost >= 1) & (C <= far)
    is_loss = (added == 0) & (lost >= 1)
    is_far = C > far
    frac = {}
    for nm, msk in [("add1", is_add1), ("addk", is_addk), ("swap", is_swap),
                    ("loss", is_loss), ("far", is_far)]:
        frac[nm] = float((off * msk).sum() / max(off_mass, 1e-12))
        # same decomposition under the independent coupling, as the null
        frac[nm + "_null"] = float((ind_off * msk).sum() / max(ind_off_mass, 1e-12))

    out = dict(n_A=len(A), n_B=len(B), iters=iters,
               n_persist=len(pairs), frac_persist=len(pairs) / max(len(A), 1),
               self_mass=self_mass,
               self_frac=self_mass / max(row_mass, 1e-12) if pairs else np.nan,
               off_mass=off_mass, moved_cost=moved_cost, base_off=base_off,
               cost_ratio=moved_cost / max(base_off, 1e-12),
               transport_cost=float((P * C).sum()), mean_C=float(C.mean()))
    out.update(frac)

    if verbose:
        log(f"\n  largest off-diagonal flows:")
        log(f"  {'mass':>9}{'jacc':>7}{'|t|':>5}{'|t+1|':>7}{'added':>7}{'lost':>6}")
        for f_ in np.argsort(off.ravel())[::-1][:12]:
            i, j = np.unravel_index(f_, off.shape)
            log(f"  {off[i, j]:9.5f}{C[i, j]:7.3f}{len(A[i]):5d}{len(B[j]):7d}"
                f"{int(added[i, j]):7d}{int(lost[i, j]):6d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--month_t", default=None,
                    help="run a single pair verbosely; default runs all pairs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--min_seqs", type=int, default=5000)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=40)
    ap.add_argument("--top_k", type=int, default=800)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "36_ot_sweep.csv"))
    args = ap.parse_args()

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    months = [m for m in months if m <= args.end_month]

    def H(mo):
        with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
            raw = constellations_of(pickle.load(fh))
        f = {c: v for c, v in raw.items()
             if v >= args.min_count and 1 <= len(c) <= args.max_set_size}
        tot = sum(f.values())
        top = sorted(f.items(), key=lambda kv: -kv[1])[:args.top_k]
        return [c for c, _ in top], np.array([v for _, v in top], float), tot

    if args.month_t:
        i = months.index(args.month_t)
        A, wa, ta = H(months[i]); B, wb, tb = H(months[i + 1])
        log(f"{months[i]} ({ta} seqs) -> {months[i+1]} ({tb} seqs)\n")
        r = analyse(A, wa, B, wb, args.eps, args.far, verbose=True)
        for k, v in r.items():
            log(f"  {k:<18}{v:.4f}" if isinstance(v, float) else f"  {k:<18}{v}")
        return

    rows = []
    for i in range(len(months) - 1):
        mt, mt1 = months[i], months[i + 1]
        A, wa, ta = H(mt); B, wb, tb = H(mt1)
        if ta < args.min_seqs or tb < args.min_seqs or len(A) < 50 or len(B) < 50:
            continue
        r = analyse(A, wa, B, wb, args.eps, args.far)
        r.update(month_t=mt, month_t1=mt1, year=mt[:4], n_seqs_t=ta)
        rows.append(r)
        log(f"  {mt}  self={r['self_frac']:.1%}  moved/indep={r['cost_ratio']:.3f}  "
            f"add1={r['add1']:.1%} (null {r['add1_null']:.1%})  far={r['far']:.1%}")

    if not rows:
        raise SystemExit("no usable month pairs")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log("\n" + "=" * 76)
    log(f"SUMMARY over {len(df)} month pairs")
    log("=" * 76)
    log(f"  self-transport            mean {df.self_frac.mean():.1%}  "
        f"[{df.self_frac.min():.1%}, {df.self_frac.max():.1%}]")
    log(f"  moved cost / independent  mean {df.cost_ratio.mean():.3f}  "
        f"below 1.0 in {(df.cost_ratio < 1).sum()}/{len(df)} months")
    log("")
    log(f"  {'edit type':<10}{'transported':>13}{'independent':>13}{'lift':>8}")
    for nm in ["add1", "addk", "swap", "loss", "far"]:
        t, n = df[nm].mean(), df[nm + "_null"].mean()
        log(f"  {nm:<10}{t:>12.1%}{n:>13.1%}{t/max(n,1e-9):>8.1f}x")

    log("\n  by year:")
    log(df.groupby("year").agg(
        pairs=("add1", "size"), self=("self_frac", "mean"),
        add1=("add1", "mean"), far=("far", "mean"),
        ratio=("cost_ratio", "mean")).round(3).to_string())

    log("\n" + "-" * 76)
    log("READ")
    log("-" * 76)
    a1, a1n = df.add1.mean(), df.add1_null.mean()
    log(f"  ADD-1 share of moving mass: {a1:.1%}, against {a1n:.1%} under an")
    log(f"  independent coupling -- a lift of {a1/max(a1n,1e-9):.1f}x.")
    if a1 / max(a1n, 1e-9) > 3 and a1 > 0.2:
        log("  Transport routes moving mass along one-mutation additions far more")
        log("  than chance. It recovers the expansion structure script 22 measured")
        log("  (62% of new constellations, median one source) from the geometry")
        log("  alone, without being told to look for it.")
    elif a1 / max(a1n, 1e-9) < 1.5:
        log("  ADD-1 is no more common than chance. Transport is not finding the")
        log("  expansion structure, and the fifteen clean rows in script 35 were")
        log("  not representative of where the mass goes.")
    else:
        log("  Moderate enrichment. Check whether it concentrates in the quiet")
        log("  months rather than holding throughout.")
    log("")
    log(f"  FAR share (Jaccard > {args.far}): {df.far.mean():.1%}. High values mark")
    log("  months where mass was forced onto unrelated sets because balanced OT")
    log("  cannot let it die. That is what semi-relaxed OT (DeST-OT) fixes, and")
    log("  the per-year table shows which periods need it.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
