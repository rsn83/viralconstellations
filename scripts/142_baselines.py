#!/usr/bin/env python3
"""
142_baselines.py

External baselines for the 112 ladder.

WHY
    Every rung of the ladder is an ablation of the same model: flat is the
    model without drift, drift is the model without splits. So a number like
    -6.880 says only "better than our other configuration". It cannot say
    whether the model is worth having.

    These baselines are outside the model family. Three of them produce nats
    directly comparable to the ladder; two produce rankings that test what the
    ladder's unseen column is actually measuring.

WHAT IS REPORTED
    all / seen / unseen, on the same split 112 uses. The decomposition is the
    point: the seen column is dominated by whichever lineage was sequenced
    most and a lookup table should win it, while the unseen column is the only
    place a generative model can earn anything.

  1. lookup        P(x) = count(x)/denom for sets in training, floor for
                   others. Zero parameters.
  2. persistence   the same, using only the LAST training month.
  3. bernoulli     one global theta_v per mutation, no components at all.
                   Tests whether component structure does anything.

  4. distance      minimum Hamming distance from each unseen test set to the
                   training population, correlated against the mixture's own
                   log-likelihood on those sets. This is the impossibility
                   result as a measurement rather than a proof: if the
                   correlation is near 1, the mixture's unseen column IS
                   distance ranking.
  5. diffusion     mass reaching each test set under the mutation kernel of
                   the RECOMB 2026 constrained-subspace diffusion method,
                   (1-m)^(L-h) m^h summed over the training population
                   weighted by frequency. An external, published, citable
                   scorer for the same quantity.

A NOTE ON THE LOOKUP FLOOR
    log(alpha/denom) for an unseen set is NOT a normalised probability over
    the 2^V possible sets -- the lookup table does not pay for the mass it
    assigns outside its support. So the lookup number is an UPPER bound on
    what a lookup table could legitimately score. It is reported because it
    is the convention 112 already prints, and because a model that cannot
    beat an over-generous lookup table has a problem. Do not quote it as a
    like-for-like likelihood.

USAGE
    python 142_baselines.py \
        --engine scripts/110_hierarchical_birthdeath_v2_fixed.py \
        --data-dir data/processed/full_data_graphs_withdel \
        --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
        --train 2021-06:2022-05 --test 2022-06
"""
import argparse
import importlib.util
import sys

import numpy as np

EPS = 1e-12


def load_engine(path):
    """Reuse the ladder's loaders so the data is byte-identical."""
    spec = importlib.util.spec_from_file_location("engine", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["engine"] = mod
    spec.loader.exec_module(mod)
    return mod


def keys_of(X):
    """Hashable key per row. Sets are sparse, so the tuple of set positions is
    far smaller than the full binary row."""
    return [tuple(np.flatnonzero(r).tolist()) for r in X.astype(np.int8)]


def split_report(name, ll, wte, seen):
    w_all = wte.sum()
    a = float((wte * ll).sum() / max(w_all, EPS))
    s = (float((wte[seen] * ll[seen]).sum() / max(wte[seen].sum(), EPS))
         if seen.any() else float("nan"))
    u = (float((wte[~seen] * ll[~seen]).sum() / max(wte[~seen].sum(), EPS))
         if (~seen).any() else float("nan"))
    print(f"  {name:<22}{a:10.3f}{s:10.3f}{u:10.3f}", flush=True)
    return a, s, u


def lookup_ll(tr_counts, denom, alpha, te_keys):
    """Empirical table with an explicit floor for unseen sets."""
    fl = np.log(alpha / denom)
    return np.array([np.log(tr_counts[k] / denom) if k in tr_counts else fl
                     for k in te_keys])


def bernoulli_ll(Xte, theta):
    th = np.clip(theta, 1e-6, 1 - 1e-6)
    return (Xte @ np.log(th) + (1 - Xte) @ np.log(1 - th))


def min_distance(Xte, Xtr, wtr, cap, chunk=512, verbose=True):
    """Minimum Hamming distance from each test row to the training population.

    d(x,y) = |x| + |y| - 2 x.y, so a matmul gives every distance at once. The
    training set is capped at the `cap` heaviest constellations: the nearest
    neighbour of anything is overwhelmingly likely to be a common set, and the
    full 166k x 20k product is not worth its runtime for a baseline.
    """
    order = np.argsort(-wtr)[:cap]
    A = Xtr[order].astype(np.float32)
    na = A.sum(1)
    B = Xte.astype(np.float32)
    nb = B.sum(1)
    out = np.empty(len(B), dtype=np.float32)
    for i in range(0, len(B), chunk):
        b = B[i:i + chunk]
        d = nb[i:i + chunk, None] + na[None, :] - 2.0 * (b @ A.T)
        out[i:i + chunk] = d.min(1)
        if verbose and (i // chunk) % 20 == 0:
            print(f"      distance {i}/{len(B)}", flush=True)
    return out


def diffusion_score(Xte, Xtr, wtr, L, m, cap, chunk=512, verbose=True):
    """log of the mass reaching each test set under the DiffEvol kernel.

    M_xy = (1-m)^(L-h) m^h with h the Hamming distance, so
        log score(x) = logsumexp_y [ log pi_y + h_xy log(m/(1-m)) ]
                       + (L) log(1-m)
    The constant is dropped: it is identical for every candidate and cannot
    affect a ranking.

    Note what this makes explicit -- the kernel is a monotone function of
    Hamming distance alone, so the only thing separating two candidates at the
    same distance is how much training mass sits at that distance. That is
    the whole content of the method's proposal step.
    """
    order = np.argsort(-wtr)[:cap]
    A = Xtr[order].astype(np.float32)
    pi = wtr[order] / wtr[order].sum()
    logpi = np.log(pi + EPS).astype(np.float32)
    na = A.sum(1)
    B = Xte.astype(np.float32)
    nb = B.sum(1)
    lam = np.float32(np.log(m / (1.0 - m)))
    out = np.empty(len(B), dtype=np.float64)
    for i in range(0, len(B), chunk):
        b = B[i:i + chunk]
        h = nb[i:i + chunk, None] + na[None, :] - 2.0 * (b @ A.T)
        z = logpi[None, :] + lam * h
        mx = z.max(1, keepdims=True)
        out[i:i + chunk] = (np.log(np.exp(z - mx).sum(1)) + mx.ravel())
        if verbose and (i // chunk) % 20 == 0:
            print(f"      diffusion {i}/{len(B)}", flush=True)
    return out


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="pseudocount for the lookup floor; match 112")
    ap.add_argument("--mut-rate", type=float, default=1e-3,
                    help="per-site per-window mutation rate for the diffusion "
                         "kernel. Only the ranking matters, and the ranking "
                         "is monotone in this, so the exact value is not "
                         "critical -- but report whichever you use")
    ap.add_argument("--genome-length", type=int, default=0,
                    help="L for the diffusion kernel; defaults to the "
                         "vocabulary size, which is the right L for a model "
                         "defined over spike mutation slots")
    ap.add_argument("--nn-cap", type=int, default=20000,
                    help="how many of the heaviest training constellations to "
                         "search for nearest neighbours and diffusion mass")
    ap.add_argument("--mixture-ll", default="",
                    help="optional .npy of the mixture's per-test-row "
                         "log-likelihood, in the same row order. Enables the "
                         "distance and diffusion correlations, which are the "
                         "point of baselines 4 and 5.")
    args = ap.parse_args()

    E = load_engine(args.engine)
    names, V = E.load_names(args.vocab)
    L = args.genome_length or V

    Xs, ws = [], []
    for ym in E.months_in_range(args.train):
        X, w = E.build(E.load_month(args.data_dir, ym), V, args.min_count)
        if len(X):
            Xs.append(X); ws.append(w)
    Xtr = np.vstack(Xs).astype(np.float32)
    wtr = np.concatenate(ws).astype(np.float64)
    Xlast, wlast = Xs[-1].astype(np.float32), ws[-1].astype(np.float64)

    Xte, wte = E.build(E.load_month(args.data_dir, args.test), V,
                       args.min_count)
    Xte = Xte.astype(np.float32); wte = wte.astype(np.float64)

    print(f"V={V:,}  L={L:,}   train {len(Xtr):,} rows / {wtr.sum():,.0f} seqs"
          f"   test {len(Xte):,} rows / {wte.sum():,.0f} seqs")

    tr_keys = keys_of(Xtr)
    te_keys = keys_of(Xte)
    counts = {}
    for k, c in zip(tr_keys, wtr):
        counts[k] = counts.get(k, 0.0) + c
    last = {}
    for k, c in zip(keys_of(Xlast), wlast):
        last[k] = last.get(k, 0.0) + c

    seen = np.array([k in counts for k in te_keys])
    print(f"test: {int(seen.sum()):,} seen / {int((~seen).sum()):,} unseen "
          f"({100*(~seen).mean():.1f}% of rows unseen, "
          f"{100*wte[~seen].sum()/wte.sum():.1f}% of sequences)\n")

    print(f"  {'baseline':<22}{'all':>10}{'seen':>10}{'unseen':>10}")
    print("  " + "-" * 52)

    denom = wtr.sum() + args.alpha
    split_report("lookup", lookup_ll(counts, denom, args.alpha, te_keys),
                 wte, seen)
    denom_l = wlast.sum() + args.alpha
    split_report("persistence", lookup_ll(last, denom_l, args.alpha, te_keys),
                 wte, seen)
    theta = (wtr[:, None] * Xtr).sum(0) / wtr.sum()
    split_report("bernoulli", bernoulli_ll(Xte, theta), wte, seen)

    print("\n  ranking checks on the UNSEEN sets")
    print("  " + "-" * 52)
    unseen_idx = np.flatnonzero(~seen)
    if len(unseen_idx) < 3:
        print("  too few unseen sets to correlate")
        return
    Xu = Xte[unseen_idx]
    dist = min_distance(Xu, Xtr, wtr, args.nn_cap)
    diff = diffusion_score(Xu, Xtr, wtr, L, args.mut_rate, args.nn_cap)
    print(f"  min distance to training: median {np.median(dist):.1f}, "
          f"range {dist.min():.0f}-{dist.max():.0f}")
    print(f"  spearman(diffusion, -distance) = {spearman(diff, -dist):+.3f}")
    print("    near +1 means the diffusion kernel adds nothing to distance "
          "beyond\n    weighting by how much mass sits at each radius")

    if args.mixture_ll:
        mll = np.load(args.mixture_ll)
        if len(mll) != len(Xte):
            print(f"  mixture-ll has {len(mll)} rows, test has {len(Xte)} "
                  f"-- skipping")
        else:
            mu = mll[unseen_idx]
            print(f"  spearman(mixture, -distance)   = "
                  f"{spearman(mu, -dist):+.3f}")
            print(f"  spearman(mixture, diffusion)   = "
                  f"{spearman(mu, diff):+.3f}")
            print("    the first is the impossibility result as a "
                  "measurement: near +1\n    means the mixture's unseen "
                  "column is nearest-neighbour distance")
    else:
        print("\n  pass --mixture-ll to correlate against the fitted model; "
              "without it\n  baselines 4 and 5 only describe each other")


if __name__ == "__main__":
    main()
