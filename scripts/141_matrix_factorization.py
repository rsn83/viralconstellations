#!/usr/bin/env python3
"""
141_matrix_factorization.py

The matrix-factorisation analogue of the mixture, scored on the SAME held-out
likelihood as the 112 ladder so the numbers are directly comparable.

WHAT IS BEING COMPARED
    Mixture     each sequence belongs to ONE of K components, and a component
                is a free V-vector of Bernoulli probabilities.
                    P(x) = sum_k pi_k prod_v theta_kv^x_v (1-theta_kv)^(1-x_v)

    Factorised  each sequence has a continuous R-vector u, and a mutation has
                a loading phi_v. No components at all.
                    P(x|u) = prod_v sig(u.phi_v + b_v)^x_v (...)^(1-x_v)
                    P(x)   = integral N(u;0,I) P(x|u) du

    The mixture is a latent-variable model with K atoms; the factorised model
    is the same thing with a continuous latent. So the comparison asks whether
    discrete components were the right representation, or whether a smooth
    low-dimensional space describes the population better.

    This is NOT script 131. There the components survived and only the
    emission was made low-rank. Here the components are gone.

HOW HELD-OUT IS MADE COMPARABLE
    The integral has no closed form, so it is evaluated by drawing S vectors
    from the fitted prior and averaging. That turns the factorised model into
    a mixture with S equally weighted components, which is scored by exactly
    the same routine as the ladder. No approximation is hidden in the metric:
    both sides compute log of a weighted sum of Bernoulli products.

    S prior samples is a crude integrator in R dimensions. It is a LOWER
    bound on the true likelihood, so a factorised model that wins is really
    winning, while one that loses may be losing to the integrator. --R 2..8
    keeps the bound tight enough to be informative; do not read a narrow loss
    at R=16 as a verdict on the model.

WHAT IT REPORTS
    Held-out nats per sequence, split into sets SEEN in training and sets NOT
    seen, because the second column is the one the project is about and the
    two behave very differently.

USAGE
    python 141_matrix_factorization.py \
        --engine scripts/110_hierarchical_birthdeath_v2_fixed.py \
        --data-dir data/processed/full_data_graphs_withdel \
        --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
        --train 2021-06:2022-05 --test 2022-06 --ranks 2,4,8,16
"""
import argparse
import importlib.util
import sys

import numpy as np

EPS = 1e-12


def load_engine(path):
    """Reuse the ladder's own loaders so the data is byte-identical."""
    spec = importlib.util.spec_from_file_location("engine", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["engine"] = mod
    spec.loader.exec_module(mod)
    return mod


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def dedupe(X, w):
    """Collapse to distinct constellations with summed weights.

    The likelihood is otherwise dominated by whichever lineage was sequenced
    most, which is an artefact of surveillance effort rather than biology.
    Fitting on distinct sets and weighting by count keeps the arithmetic
    identical while making the structure visible.
    """
    if len(X) == 0:
        return X, w
    order = np.lexsort(X.T[::-1])
    Xs = X[order]
    ws = w[order]
    keep = np.ones(len(Xs), dtype=bool)
    keep[1:] = (Xs[1:] != Xs[:-1]).any(axis=1)
    idx = np.flatnonzero(keep)
    out_w = np.add.reduceat(ws, idx)
    return Xs[idx], out_w


def fit_factorised(X, w, R, iters=60, lam_u=1.0, lam_p=1.0, seed=0,
                   verbose=True):
    """MAP fit of  sig(U @ Phi.T + b)  by alternating per-coordinate Newton.

    Same optimiser reasoning as the fixed m_step: the logistic curvature is
    p(1-p), which spans orders of magnitude across a sparse mutation matrix,
    so a fixed step size under-converges rare positions by thousands of fold.
    Each block below divides the gradient by its own curvature.

    lam_u is the Gaussian prior on the latent vectors -- it must match the
    prior used to integrate at scoring time, or the held-out number is not a
    likelihood.
    """
    rng = np.random.default_rng(seed)
    n, V = X.shape
    p0 = np.clip((w[:, None] * X).sum(0) / max(w.sum(), EPS), 1e-4, 1 - 1e-4)
    b = np.log(p0 / (1 - p0))
    U = rng.normal(0, 0.1, (n, R))
    Phi = rng.normal(0, 0.1, (V, R))
    wc = w[:, None]

    for it in range(iters):
        Z = U @ Phi.T + b[None, :]
        P = sig(Z)
        Gr = wc * (X - P)                      # residual
        H = wc * P * (1 - P)                   # curvature

        # loadings, one latent dimension at a time
        for r in range(R):
            u = U[:, r]
            g = Gr.T @ u - lam_p * Phi[:, r]
            h = H.T @ (u * u) + lam_p
            Phi[:, r] += g / np.maximum(h, EPS)
            Z = U @ Phi.T + b[None, :]
            P = sig(Z); Gr = wc * (X - P); H = wc * P * (1 - P)

        # intercepts
        b += Gr.sum(0) / np.maximum(H.sum(0) + 1e-6, EPS)
        Z = U @ Phi.T + b[None, :]
        P = sig(Z); Gr = wc * (X - P); H = wc * P * (1 - P)

        # latent vectors
        for r in range(R):
            ph = Phi[:, r]
            g = Gr @ ph - lam_u * U[:, r]
            h = H @ (ph * ph) + lam_u
            U[:, r] += g / np.maximum(h, EPS)
            Z = U @ Phi.T + b[None, :]
            P = sig(Z); Gr = wc * (X - P); H = wc * P * (1 - P)

        if verbose and (it + 1) % 20 == 0:
            th = np.clip(P, 1e-6, 1 - 1e-6)
            ll = float((wc * (X * np.log(th)
                              + (1 - X) * np.log(1 - th))).sum()) / w.sum()
            print(f"    R={R} iter {it+1}  train LL/seq {ll:8.3f}", flush=True)

    return U, Phi, b


def score_factorised(Xte, wte, Phi, b, R, S=4000, seed=0, batch=256):
    """Held-out nats per sequence.

    S vectors are drawn from the SAME N(0, 1/lam_u) prior the fit assumed, and
    the model becomes an S-component equal-weight mixture. Scored by the same
    log-sum-exp over Bernoulli products the ladder uses.
    """
    rng = np.random.default_rng(seed)
    Us = rng.normal(0, 1.0, (S, R))
    tot = 0.0
    for i in range(0, S, batch):
        Ub = Us[i:i + batch]
        Th = np.clip(sig(Ub @ Phi.T + b[None, :]), 1e-6, 1 - 1e-6)
        lp = (Xte @ np.log(Th).T
              + (1 - Xte) @ np.log(1 - Th).T)          # n_te x batch
        if i == 0:
            acc = lp
        else:
            acc = np.concatenate([acc, lp], axis=1)
    mx = acc.max(1, keepdims=True)
    ll = np.log(np.exp(acc - mx).sum(1)) + mx.ravel() - np.log(S)
    return float((wte * ll).sum() / wte.sum()), ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True,
                    help="path to the fitted-mixture script; its loaders are "
                         "reused so both models see identical data")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--ranks", default="2,4,8,16")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--no-dedupe", action="store_true",
                    help="weight by sequence count rather than by distinct "
                         "constellation. Reproduces the ladder's weighting, "
                         "which is dominated by surveillance effort.")
    args = ap.parse_args()

    E = load_engine(args.engine)
    names, V = E.load_names(args.vocab)
    months = E.months_in_range(args.train)
    Xs, ws = [], []
    for ym in months:
        rec = E.load_month(args.data_dir, ym)
        X, w = E.build(rec, V, min_count=args.min_count)
        if len(X):
            Xs.append(X); ws.append(w)
    Xtr = np.vstack(Xs).astype(np.float64)
    wtr = np.concatenate(ws).astype(np.float64)

    rec = E.load_month(args.data_dir, args.test)
    Xte, wte = E.build(rec, V, min_count=args.min_count)
    Xte = Xte.astype(np.float64); wte = wte.astype(np.float64)

    print(f"train {Xtr.shape[0]:,} sequences   test {Xte.shape[0]:,}   V={V}")

    if not args.no_dedupe:
        Xtr, wtr = dedupe(Xtr, wtr)
        print(f"deduped to {Xtr.shape[0]:,} distinct training constellations")

    # which test sets were seen in training -- the split that matters
    tr_keys = set(map(tuple, Xtr.astype(np.int8)))
    seen = np.array([tuple(r) in tr_keys for r in Xte.astype(np.int8)])
    print(f"test: {int(seen.sum()):,} seen sets, {int((~seen).sum()):,} unseen "
          f"({100*(~seen).mean():.1f}% unseen)\n")

    print(f"{'rank':>6}  {'all':>9}  {'seen':>9}  {'unseen':>9}")
    print("-" * 40)
    for R in [int(x) for x in args.ranks.split(",")]:
        U, Phi, b = fit_factorised(Xtr, wtr, R, iters=args.iters,
                                   verbose=False)
        allv, ll = score_factorised(Xte, wte, Phi, b, R, S=args.samples)
        sv = (float((wte[seen] * ll[seen]).sum() / max(wte[seen].sum(), EPS))
              if seen.any() else float("nan"))
        uv = (float((wte[~seen] * ll[~seen]).sum() / max(wte[~seen].sum(), EPS))
              if (~seen).any() else float("nan"))
        print(f"{R:>6}  {allv:9.3f}  {sv:9.3f}  {uv:9.3f}", flush=True)

    print("\nCompare against the 112 ladder's rows on the same train/test "
          "split.\nThe unseen column is the one to read: the mixture's "
          "impossibility results\nare about novel sets, and a continuous "
          "latent is the cheapest way to test\nwhether discrete components "
          "were the wrong representation for them.")


if __name__ == "__main__":
    main()
