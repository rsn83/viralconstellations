#!/usr/bin/env python3
"""
96_is_new_lineage_reachable.py

Before spending a day on a continuous-latent model, ask the question that
decides whether it can possibly help:

    Is the new lineage's fingerprint REACHABLE from the existing ones?

A continuous latent works only if new lineages sit in a smooth region spanned
by what has been seen. If BA.5's fingerprint is a genuine outward step -- a
mutation at near-zero in every existing group -- then no interpolation in any
latent space reaches it, and the continuous route inherits the same blind spot
as the discrete one, in a harder-to-diagnose form.

Three tests, all cheap, all on parameters you already have:

  1  BEST CONVEX MIX. Find the mixture of existing fingerprints closest to the
     new lineage's observed profile. If the residual is small, the new lineage
     is interpolation. If large, it is extrapolation.

  2  PER-MUTATION REACH. For each mutation the new lineage carries, what is the
     highest value ANY existing group assigns it? Mutations where the maximum
     is near zero cannot be reached by any convex or linear combination.

  3  LINEAR SPAN. Allow unconstrained (not just convex) combinations. This is
     the most generous possible notion of "in the span of what we have seen".

Usage:
  python 96_is_new_lineage_reachable.py \
      --npz    results/91_exact.npz \
      --vocab  data/processed/full_data_graphs_posres/posres_vocab.tsv \
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
    w = np.array([c for _, c in records], float)
    X = np.zeros((len(records), V), dtype=np.float32)
    for i, (s, _) in enumerate(records):
        X[i, [n for n in s if 0 <= n < V]] = 1.0
    return X, w


def loglik_matrix(X, theta):
    lt, lc = np.log(theta + EPS), np.log(1 - theta + EPS)
    return X @ (lt - lc).T + lc.sum(1)[None, :]


def simplex_lstsq(B, y, iters=8000, lr=.5):
    """min ||w B - y||^2  subject to w >= 0, sum w = 1  (projected gradient)."""
    K = B.shape[0]
    w = np.full(K, 1.0 / K)
    for _ in range(iters):
        g = 2 * (w @ B - y) @ B.T
        w = w - lr * g / (np.abs(g).max() + 1e-12)
        w = np.maximum(w, 0)
        s = w.sum()
        w = w / s if s > 0 else np.full(K, 1.0 / K)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--top", type=int, default=12)
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
    te = months_in_range(args.test)

    # ---- the new lineage's observed profile ----
    rec = []
    for ym in te: rec += load_month(args.data_dir, ym)
    Xte, wte = build(rec, V)
    lp = loglik_matrix(Xte, theta) + np.log(Pi[-1] + EPS)[None, :]
    z = lp.argmax(1)
    kstar = int(np.bincount(z, weights=wte, minlength=K).argmax())
    m = z == kstar
    obs = (wte[m, None] * Xte[m]).sum(0) / wte[m].sum()
    exc = obs - theta[kstar]
    new_muts = [int(i) for i in np.argsort(-exc)[:6] if exc[i] > .15]

    print(f"dominant group in {te[0]}: blk{kstar}  "
          f"({wte[m].sum():,.0f} of {wte.sum():,.0f} sequences)")
    print(f"mutations it carries but blk{kstar} does not expect:")
    for i in new_muts:
        print(f"    {names.get(i, i):<10} blk{kstar} says {theta[kstar,i]:.3f}, "
              f"observed {obs[i]:.3f}")
    if not new_muts:
        raise SystemExit("\nno large excess found -- nothing to test")

    live = [k for k in range(K) if Pi.max(0)[k] > 1e-3]
    print(f"\nlive groups (ever above 0.1% in any month): {live}")

    # ---------------- TEST 1 ----------------
    print("\n" + "=" * 78)
    print("TEST 1   best convex mixture of existing fingerprints")
    print("=" * 78)
    B = theta[live]
    w = simplex_lstsq(B, obs)
    fit = w @ B
    print(f"\n  weights: " + "  ".join(f"blk{k}={wi:.3f}" for k, wi in zip(live, w)))
    print(f"  overall residual  ||obs - mix||_2 = {np.linalg.norm(obs-fit):.4f}")
    print(f"\n  {'mutation':<10}{'observed':>10}{'best mix':>10}{'residual':>10}")
    for i in new_muts:
        print(f"  {names.get(i,i):<10}{obs[i]:>10.3f}{fit[i]:>10.3f}{obs[i]-fit[i]:>+10.3f}")

    # ---------------- TEST 2 ----------------
    print("\n" + "=" * 78)
    print("TEST 2   per-mutation reach -- can ANY existing group supply it?")
    print("=" * 78)
    print(f"\n  {'mutation':<10}{'observed':>10}{'max over groups':>18}"
          f"{'which':>8}   verdict")
    unreachable = []
    for i in new_muts:
        col = theta[live, i]
        j = int(np.argmax(col))
        mx = col[j]
        if mx < .1:
            verd = "UNREACHABLE -- no group has it"
            unreachable.append(i)
        elif mx < obs[i] * .5:
            verd = "partially reachable"
        else:
            verd = "reachable -- borrowable from another group"
        print(f"  {names.get(i,i):<10}{obs[i]:>10.3f}{mx:>18.3f}"
              f"{'blk'+str(live[j]):>8}   {verd}")

    # ---------------- TEST 3 ----------------
    print("\n" + "=" * 78)
    print("TEST 3   linear span -- the most generous case (weights may be any sign)")
    print("=" * 78)
    coef, *_ = np.linalg.lstsq(B.T, obs, rcond=None)
    lin = coef @ B
    print(f"\n  residual ||obs - span||_2 = {np.linalg.norm(obs-lin):.4f}")
    print(f"\n  {'mutation':<10}{'observed':>10}{'best span':>11}{'residual':>10}")
    for i in new_muts:
        print(f"  {names.get(i,i):<10}{obs[i]:>10.3f}{lin[i]:>11.3f}"
              f"{obs[i]-lin[i]:>+10.3f}")

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    n_un = len(unreachable)
    print(f"\n  {n_un} of {len(new_muts)} defining mutations are at <0.10 in EVERY"
          f" existing group.")
    if n_un:
        print("    " + ", ".join(names.get(i, str(i)) for i in unreachable))
    if n_un == 0:
        print("""
  -> INTERPOLATION. Every mutation the new lineage carries is already high in
     some existing group; the new lineage recombines what has been seen. A
     continuous latent should reach it, because the decoder has learned to
     produce each of these mutations somewhere in the space. Worth building.""")
    elif n_un < len(new_muts):
        print("""
  -> MIXED. Some mutations are borrowable from other groups, some are not. A
     continuous latent gets you part of the way. The mutations no group has
     still require raising a value the model has never produced.""")
    else:
        print("""
  -> EXTRAPOLATION. No existing group carries these mutations at any appreciable
     level, so no combination -- convex OR linear, in ANY latent space fitted on
     this data -- reaches the new fingerprint. A continuous latent inherits the
     same blind spot as the discrete one, and hides it: with discrete groups the
     gap is visible as a missing row, in a latent space it is an unvisited
     region that looks no different from any other. Do NOT build the continuous
     model expecting it to solve this.""")
    print("""
  Caveat: this tests reachability of the FINGERPRINT, not whether a decoder
  trained with a smooth prior might generalise to produce it anyway. It is a
  necessary condition for the continuous route, not a sufficient one.
""")


if __name__ == "__main__":
    main()
