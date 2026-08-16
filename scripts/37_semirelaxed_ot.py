#!/usr/bin/env python
"""
37_semirelaxed_ot.py

Semi-relaxed optimal transport, compared head-to-head against balanced.
CPU, a few minutes for all month pairs. numpy + pandas only.

WHY
---
Script 36 found that balanced OT is regime-dependent:

  year  self_frac  add1  add1_null  far    ratio
  2020    0.819    0.396   0.347   0.558   1.056
  2021    0.726    0.215   0.111   0.592   1.092
  2022    0.540    0.232   0.107   0.086   0.754
  2023    0.526    0.274   0.064   0.011   0.574

FAR -- mass landing on sets with Jaccard > 0.5 -- is 56-59% in 2020-2021 and
1-9% in 2022-2023, and the cost ratio crosses 1.0 at the same boundary. That
is the signature of population REPLACEMENT: balanced OT must move every unit
of mass somewhere, so when Delta is displaced by Omicron its mass gets dumped
onto sets sharing almost nothing (2021-11: size-9 -> size-27, 25 mutations
added, Jaccard 0.9). Halmos et al. report the same failure for the
yolk-syncytial layer in zebrafish.

Semi-relaxed OT replaces the hard row-marginal constraint with a soft KL
penalty, so mass may be DESTROYED at a cost rather than forced to transfer.
That is the variant DeST-OT (Halmos et al., Cell Systems 2025) uses to learn
growth rates, and it is what a population with ~48% monthly set death needs.

WHAT IS RELAXED
---------------
BOTH marginals are softened with a KL penalty of weight rho. Rows soft means a
constellation may lose mass without sending it anywhere -- extinction. Columns
soft means a constellation at t+1 may receive mass from nowhere -- birth.

Relaxing only the rows does not work, and the synthetic check caught it: with
columns hard, P^T 1 = b pins total mass at 1, so nothing can die and the
relaxation only shuffles mass between rows.

  rho large   recovers balanced OT
  rho ~ 0.2   validated: kills 45% of mass in a replacement month, 10% in a
              drift month
  rho < 0.05  unstable; total mass exceeds 1 (mass created from nothing)

THE COMPARISON
--------------
Balanced and semi-relaxed are run on the same cost matrix, and the same
diagnostics computed on both. The question is whether relaxation rescues the
replacement months: does FAR drop, does the cost ratio fall below 1, and does
the mass that DOES move land on near neighbours.

The killed-mass column is itself informative -- it should be high in exactly
the replacement months and low in the drift months, without being told which
is which.

Usage
-----
  python scripts/37_semirelaxed_ot.py
  python scripts/37_semirelaxed_ot.py --rho 0.05 0.1 0.5
  python scripts/37_semirelaxed_ot.py --month_t 2021-11    # single pair
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


def sinkhorn_balanced(a, b, C, eps, n_iter=1500, tol=1e-9):
    logK = -C / eps
    f = np.zeros(len(a)); g = np.zeros(len(b))
    la, lb = np.log(a + 1e-300), np.log(b + 1e-300)
    for it in range(n_iter):
        fp = f
        f = eps * (la - _lse(logK + g[None, :] / eps, axis=1))
        g = eps * (lb - _lse(logK + f[:, None] / eps, axis=0))
        if it % 25 == 0 and np.max(np.abs(f - fp)) < tol:
            break
    return np.exp(logK + f[:, None] / eps + g[None, :] / eps)


def sinkhorn_unbalanced(a, b, C, eps, rho_row, rho_col, n_iter=1500, tol=1e-9):
    """BOTH marginals soft, so mass may DIE (rows) and be CREATED (cols).

    A first version relaxed only the rows and kept columns hard. That was
    wrong and the synthetic check caught it: with columns hard, P^T 1 = b, so
    total mass is pinned at 1 and NOTHING CAN DIE -- relaxation merely shifts
    mass between rows. Extinction requires both marginals soft.

    Update rule: the balanced step  f = eps*(log a - LSE_row)  is damped by
    rho/(rho+eps). As rho -> infinity the damping tends to 1 and the balanced
    solver is recovered; as rho -> 0 the marginal constraint is ignored.

    Validated on synthetic data: with rho = 0.2, a REPLACEMENT month (half the
    sets displaced by an unrelated lineage) keeps 0.55 of its mass, while a
    DRIFT month (everything persists or gains one mutation) keeps 0.90 with
    ADD-1 at 100% and FAR at 0%.

    NOTE: below rho ~ 0.05 the iteration becomes unstable and total mass can
    exceed 1 (mass created from nothing). The script flags this.
    """
    logK = -C / eps
    f = np.zeros(len(a)); g = np.zeros(len(b))
    la, lb = np.log(a + 1e-300), np.log(b + 1e-300)
    dr, dc = rho_row / (rho_row + eps), rho_col / (rho_col + eps)
    for it in range(n_iter):
        fp = f
        f = dr * eps * (la - _lse(logK + g[None, :] / eps, axis=1))
        g = dc * eps * (lb - _lse(logK + f[:, None] / eps, axis=0))
        if it % 25 == 0 and np.max(np.abs(f - fp)) < tol:
            break
    return np.exp(logK + f[:, None] / eps + g[None, :] / eps)


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


def diagnose(P, a, b, C, added, lost, diag, far):
    """Same diagnostics as script 36, on any transport plan."""
    tot = P.sum()
    self_mass = P[diag].sum()
    row_avail = a[diag.any(axis=1)].sum() if diag.any() else np.nan
    off = np.where(diag, 0.0, P)
    off_mass = off.sum()

    ind = a[:, None] * b[None, :]
    ind_off = np.where(diag, 0.0, ind)
    base = (ind_off * C).sum() / max(ind_off.sum(), 1e-12)
    moved = (off * C).sum() / max(off_mass, 1e-12)

    out = dict(total_mass=tot, killed=1.0 - tot,
               self_mass=self_mass,
               self_frac=self_mass / max(row_avail, 1e-12),
               off_mass=off_mass, moved_cost=moved, base_off=base,
               cost_ratio=moved / max(base, 1e-12))
    masks = {"add1": (added == 1) & (lost == 0),
             "addk": (added >= 2) & (lost == 0),
             "swap": (added >= 1) & (lost >= 1) & (C <= far),
             "loss": (added == 0) & (lost >= 1),
             "far": C > far}
    for nm, msk in masks.items():
        out[nm] = float((off * msk).sum() / max(off_mass, 1e-12))
        out[nm + "_null"] = float((ind_off * msk).sum() / max(ind_off.sum(), 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--month_t", default=None)
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--min_seqs", type=int, default=5000)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=40)
    ap.add_argument("--top_k", type=int, default=800)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--rho", type=float, nargs="+", default=[1.0, 0.2, 0.05],
                    help="KL weight on BOTH marginals. Large = near-balanced; "
                         "small = mass dies cheaply. Below ~0.05 the iteration "
                         "destabilises and total mass can exceed 1 -- flagged "
                         "in the output if it happens.")
    ap.add_argument("--far", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "37_semirelaxed.csv"))
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

    def one_pair(mt, mt1, verbose=False):
        A, wa, ta = H(mt); B, wb, tb = H(mt1)
        if ta < args.min_seqs or tb < args.min_seqs or len(A) < 50 or len(B) < 50:
            return None
        a = wa / wa.sum(); b = wb / wb.sum()
        MA, MB = membership(A, B)
        inter = MA @ MB.T
        sa = MA.sum(1)[:, None]; sb = MB.sum(1)[None, :]
        C = (1.0 - inter / np.maximum(sa + sb - inter, 1e-9)).astype(np.float64)
        added = sb - inter
        lost = sa - inter

        keyA = {c: i for i, c in enumerate(A)}
        diag = np.zeros(C.shape, dtype=bool)
        for j, c in enumerate(B):
            if c in keyA:
                diag[keyA[c], j] = True

        res = []
        P = sinkhorn_balanced(a, b, C, args.eps)
        r = diagnose(P, a, b, C, added, lost, diag, args.far)
        r.update(method="balanced", rho=np.nan, unstable=False)
        res.append(r)
        for rho in args.rho:
            P = sinkhorn_unbalanced(a, b, C, args.eps, rho, rho)
            r = diagnose(P, a, b, C, added, lost, diag, args.far)
            r.update(method="unbalanced", rho=rho,
                     unstable=bool(P.sum() > 1.05))
            res.append(r)
        for r in res:
            r.update(month_t=mt, month_t1=mt1, year=mt[:4],
                     n_seqs_t=ta, n_A=len(A), n_B=len(B))
        if verbose:
            log(f"\n{mt} -> {mt1}")
            log(f"  {'method':<14}{'rho':>6}{'kept':>7}{'self':>8}{'add1':>8}"
                f"{'far':>8}{'ratio':>8}")
            for r in res:
                rr = "  --" if np.isnan(r["rho"]) else f"{r['rho']:.2f}"
                log(f"  {r['method']:<14}{rr:>6}{r['total_mass']:>7.3f}"
                    f"{r['self_frac']:>8.1%}{r['add1']:>8.1%}"
                    f"{r['far']:>8.1%}{r['cost_ratio']:>8.3f}")
        return res

    if args.month_t:
        i = months.index(args.month_t)
        one_pair(months[i], months[i + 1], verbose=True)
        return

    rows = []
    for i in range(len(months) - 1):
        res = one_pair(months[i], months[i + 1])
        if res:
            rows.extend(res)
            bal = [r for r in res if r["method"] == "balanced"][0]
            sr = [r for r in res if r["rho"] == min(args.rho)][0]
            log(f"  {months[i]}  bal: far={bal['far']:.1%} ratio={bal['cost_ratio']:.2f}"
                f"  |  sr(rho={min(args.rho)}): kept={sr['total_mass']:.2f} "
                f"far={sr['far']:.1%} ratio={sr['cost_ratio']:.2f} add1={sr['add1']:.1%}")

    if not rows:
        raise SystemExit("no usable pairs")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log("\n" + "=" * 78)
    log(f"SUMMARY  ({df.month_t.nunique()} month pairs)")
    log("=" * 78)
    log(f"  {'method':<14}{'rho':>6}{'kept':>7}{'self':>8}{'add1':>8}{'null':>8}"
        f"{'lift':>7}{'far':>8}{'ratio':>8}")
    for (meth, rho), g in df.groupby(["method", "rho"], dropna=False):
        rr = "  --" if pd.isna(rho) else f"{rho:.2f}"
        lift = g.add1.mean() / max(g.add1_null.mean(), 1e-9)
        log(f"  {meth:<14}{rr:>6}{g.total_mass.mean():>7.3f}{g.self_frac.mean():>8.1%}"
            f"{g.add1.mean():>8.1%}{g.add1_null.mean():>8.1%}{lift:>6.1f}x"
            f"{g['far'].mean():>8.1%}{g.cost_ratio.mean():>8.3f}")

    log("\n  BY YEAR  (balanced vs most-relaxed)")
    lo = min(args.rho)
    sub = df[(df.method == "balanced") | (df.rho == lo)].copy()
    sub["m"] = np.where(sub.method == "balanced", "bal", f"sr{lo}")
    log(sub.pivot_table(index="year", columns="m",
                        values=["total_mass", "far", "cost_ratio", "add1"],
                        aggfunc="mean").round(3).to_string())

    if "unstable" in df.columns and df.unstable.any():
        bad = df[df.unstable].groupby("rho").size().to_dict()
        log(f"\n  WARNING: unstable solves (total mass > 1.05) at rho: {bad}")
        log("  Those rows are numerically meaningless -- raise rho.")
    log("\n  KILLED MASS BY MONTH  (unbalanced, rho=%.2f)" % lo)
    k = df[df.rho == lo].sort_values("killed", ascending=False)
    log(f"  {'month':<10}{'killed':>8}{'far_bal':>9}")
    bal = df[df.method == "balanced"].set_index("month_t")
    for _, r in k.head(10).iterrows():
        log(f"  {r.month_t:<10}{r.killed:>8.3f}{bal.loc[r.month_t,'far']:>9.1%}")
    log("  -> killed mass should peak in the REPLACEMENT months, the same ones")
    log("     where balanced OT had high FAR. If it does, the relaxation is")
    log("     letting displaced mass die instead of dumping it on unrelated sets.")

    log("\n" + "-" * 78)
    log("READ")
    log("-" * 78)
    b = df[df.method == "balanced"]
    s = df[df.rho == lo]
    log(f"  FAR:   balanced {b['far'].mean():.1%}  ->  semi-relaxed {s['far'].mean():.1%}")
    log(f"  ratio: balanced {b.cost_ratio.mean():.3f}  ->  semi-relaxed "
        f"{s.cost_ratio.mean():.3f}")
    log(f"  ADD-1 lift: balanced {b.add1.mean()/max(b.add1_null.mean(),1e-9):.1f}x"
        f"  ->  semi-relaxed {s.add1.mean()/max(s.add1_null.mean(),1e-9):.1f}x")
    log("")
    if s["far"].mean() < b["far"].mean() - 0.1 and s.cost_ratio.mean() < 0.9:
        log("  Relaxation removes the forced long-range transfers and the plan")
        log("  beats independence. Mass that had nowhere to go now dies instead,")
        log("  which is what a population with ~48% monthly set death requires.")
    elif s["far"].mean() > b["far"].mean() - 0.05:
        log("  Relaxation does not reduce FAR much. Either rho is still too large")
        log("  (try smaller) or the long-range transfers were not driven by the")
        log("  marginal constraint after all.")
    else:
        log("  Partial improvement. Compare per-year: the 2020-2021 replacement")
        log("  months are where the relaxation should matter and 2022-2023 where")
        log("  balanced was already fine.")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
