#!/usr/bin/env python3
"""
185 -- IS THERE LOW-RANK MUTATION-GROUP STRUCTURE, AND IS IT STABLE?

TWO PROBLEMS THIS IS THE PRECONDITION FOR
-----------------------------------------
  (1) transitions between discrete sets are not learnable directly
  (2) it is unknown whether hidden mutation groups explain the observed
      set structure

A proposed answer is to encode constellations in a k-dimensional latent
space, evolve that space over time, and decode back. Before implementing
anything, two properties must hold, and both are measurable without a model.

  TEST A -- does low-rank structure EXIST?
      Build the incidence matrix H(t): mutations x constellations, binary.
      Take its singular values. If the top k capture most of the spectral
      mass for small k, constellations are combinations of a few mutation
      groups and a k-dimensional encoding has something to represent.
      If the spectrum is flat, no k-dimensional encoding will find one.

  TEST B -- is that structure STABLE across months?
      Take the top-k left singular subspace at month t and at t+1 and
      measure the principal angles between them. Small angles mean the
      groups persist and a recurrent model over them has something to
      carry forward. Large angles mean the subspace rotates every month
      and there is nothing to track.

BOTH MUST PASS. A passing but rotating subspace is as fatal as a flat
spectrum: the first says the groups are not real, the second says they are
not persistent.

WHY THE NULL MATTERS
    A binary matrix with heterogeneous row and column sums has a large
    leading singular value for trivial reasons -- common mutations and
    large constellations. So the spectrum is compared against a degree-
    preserving null: the same matrix with entries reshuffled to keep row
    and column sums (Chung-Lu style). Structure counts only where the real
    spectrum exceeds the null.

    Without this, "the top 5 components explain 80%" would be a statement
    about mutation frequency, not about groups -- the same error that made
    the population-frequency baseline look strong in 179.

INTERPRETATION
    A passes, B passes  -> a k-dimensional latent state with a temporal
                           transition is justified; k and the timescale are
                           measured rather than chosen.
    A passes, B fails   -> groups exist but do not persist; a latent state
                           carried across months has nothing to carry.
    A fails             -> constellations are not low-rank combinations of
                           groups in this representation. The hidden-group
                           idea has no target.

USAGE
    python scripts/185_lowrank.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --out results/lowrank.json

GIT
    git add scripts/185_lowrank.py
    git commit -m "185: low-rank group structure and subspace stability"
    git push
"""

import argparse
import importlib.util
import json

import numpy as np


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def incidence(sets, muts):
    ix = {m: k for k, m in enumerate(muts)}
    H = np.zeros((len(muts), len(sets)), dtype=np.float64)
    for j, S in enumerate(sets):
        for m in S:
            if m in ix:
                H[ix[m], j] = 1.0
    return H


def degree_preserving_null(H, rng):
    """Chung-Lu style: independent Bernoulli with p_ij = r_i c_j / total,
    preserving expected row and column sums. Isolates structure beyond
    mutation frequency and constellation size."""
    r = H.sum(1, keepdims=True)
    c = H.sum(0, keepdims=True)
    tot = H.sum()
    if tot <= 0:
        return H.copy()
    P = np.clip(r @ c / tot, 0.0, 1.0)
    return (rng.random(H.shape) < P).astype(np.float64)


def spectrum_stats(H, rng, n_null=3):
    s = np.linalg.svd(H, compute_uv=False)
    s2 = s ** 2
    tot = s2.sum()
    if tot <= 0:
        return None
    frac = np.cumsum(s2) / tot
    # participation ratio: effective number of components
    p = s2 / tot
    eff = float(np.exp(-(p * np.log(p + 1e-15)).sum()))

    null_eff = []
    for _ in range(n_null):
        sn = np.linalg.svd(degree_preserving_null(H, rng), compute_uv=False)
        sn2 = sn ** 2
        tn = sn2.sum()
        if tn <= 0:
            continue
        pn = sn2 / tn
        null_eff.append(float(np.exp(-(pn * np.log(pn + 1e-15)).sum())))
    return {
        "eff_rank": eff,
        "eff_rank_null": float(np.mean(null_eff)) if null_eff else float("nan"),
        "frac_top5": float(frac[min(4, len(frac) - 1)]),
        "frac_top10": float(frac[min(9, len(frac) - 1)]),
        "frac_top30": float(frac[min(29, len(frac) - 1)]),
    }


def principal_angles(U, W):
    """Mean cosine of principal angles between two subspaces. 1.0 means
    identical subspace, 0.0 means orthogonal."""
    k = min(U.shape[1], W.shape[1])
    if k == 0:
        return float("nan")
    s = np.linalg.svd(U[:, :k].T @ W[:, :k], compute_uv=False)
    return float(np.clip(s, 0, 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--max-bg", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--core-thresh", type=float, default=0.8,
                    dest="core_thresh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    L = load_ladder(a.ladder)
    print("loading ...")
    monthly = L.load_events(a.events)
    pops = {m: L.population(monthly[m]) for m in sorted(monthly)}
    pops = {m: p for m, p in pops.items() if p}
    months = sorted(pops)
    print(f"  {len(months)} months  |  core threshold {a.core_thresh}")

    rows, subs = [], {}
    for m in months:
        sets = [S for S, _ in
                sorted(pops[m].items(), key=lambda kv: -kv[1])[:a.max_bg]]
        if len(sets) < 5:
            continue
        n = len(sets)
        freq = {}
        for S in sets:
            for mut in S:
                freq[mut] = freq.get(mut, 0) + 1
        core = {mut for mut, c in freq.items() if c / n >= a.core_thresh}
        resid_muts = sorted({mut for S in sets for mut in S if mut not in core})
        if len(resid_muts) < 5:
            continue
        Hr = incidence(sets, resid_muts)
        st = spectrum_stats(Hr, rng)
        if st is None:
            continue
        U, _, _ = np.linalg.svd(Hr, full_matrices=False)
        subs[m] = (resid_muts, U[:, :a.k])
        st.update({"month": m, "n_sets": len(sets),
                   "n_core": len(core), "n_resid": len(resid_muts)})
        rows.append(st)

    if not rows:
        print("  NO MONTHS -- try lowering --core-thresh")
        return

    print(f"\n  TEST A (RESIDUAL) -- structure beyond degree heterogeneity?")
    print(f"  {'month':9s} {'sets':>5s} {'core':>5s} {'resid':>6s} "
          f"{'top5':>7s} {'effrank':>8s} {'null':>7s} {'gap':>6s}")
    print("  " + "-" * 62)
    for r in rows:
        gap = r["eff_rank"] - r["eff_rank_null"]
        print(f"  {r['month']:9s} {r['n_sets']:5d} {r['n_core']:5d} "
              f"{r['n_resid']:6d} {r['frac_top5']:7.3f} "
              f"{r['eff_rank']:8.1f} {r['eff_rank_null']:7.1f} "
              f"{gap:+6.1f}")

    # TEST B: subspace stability between consecutive months, on the
    # mutations they share
    ms = [r["month"] for r in rows]
    stab = []
    for a_, b_ in zip(ms[:-1], ms[1:]):
        (mu, U), (mw, W) = subs[a_], subs[b_]
        common = sorted(set(mu) & set(mw))
        if len(common) < a.k:
            continue
        iu = {m: k for k, m in enumerate(mu)}
        iw = {m: k for k, m in enumerate(mw)}
        Us = U[[iu[c] for c in common], :]
        Ws = W[[iw[c] for c in common], :]
        # re-orthonormalise after row subsetting
        Us, _ = np.linalg.qr(Us)
        Ws, _ = np.linalg.qr(Ws)
        stab.append({"pair": f"{a_}->{b_}", "n_common": len(common),
                     "cos": principal_angles(Us, Ws)})

    print(f"\n  TEST B -- is the top-{a.k} subspace stable month to month?")
    if stab:
        cs = np.array([s["cos"] for s in stab])
        for s in stab[-12:]:
            print(f"  {s['pair']:20s} common {s['n_common']:5d}  "
                  f"cos {s['cos']:.3f}")
        print(f"\n  mean cos over {len(stab)} pairs: {cs.mean():.3f}"
              f"  (min {cs.min():.3f}  max {cs.max():.3f})")
    else:
        print("  not enough shared mutations between consecutive months")

    er = np.array([r["eff_rank"] for r in rows])
    nr = np.array([r["eff_rank_null"] for r in rows])
    print(f"\n  effective rank: real {er.mean():.1f}  null {nr.mean():.1f}")
    print("\n  READING")
    print("  real effective rank << null  -> genuine low-rank structure,")
    print("     beyond what mutation frequency and set size alone produce.")
    print("  real ~ null                  -> the spectrum is explained by")
    print("     degree heterogeneity; there are no groups to find.")
    print("  cos near 1.0                 -> groups persist; a recurrent")
    print("     latent state has something to carry.")
    print("  cos low                      -> the subspace rotates monthly;")
    print("     nothing to carry forward.")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"k": a.k, "max_bg": a.max_bg,
                       "per_month": rows, "stability": stab}, fh, indent=2)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
