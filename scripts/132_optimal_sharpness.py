#!/usr/bin/env python
"""
132_optimal_sharpness.py -- how sharp should the components be, given the
horizon?

THE BOUND
---------
Let a component hold every emission probability within eps of zero or one. For
a set at Hamming distance d from that component's mode,

    log P(S | k)  =  log P(mode | k)  -  d * log((1-eps)/eps)   + O(d*eps)

Each mismatched position costs log((1-eps)/eps) and each matching position
costs log(1-eps). So sharpness acts twice, in opposite directions: it makes the
mode almost certain, and it makes anything a few edits away almost impossible.
The whole of the seen-versus-unseen behaviour is this one parameter.

THE DERIVATION
--------------
A month at horizon h contains a fraction p(h) of sets the model has seen and
1-p(h) it has not, the latter at distances drawn from an observed distribution.
Expected log-likelihood per sequence, for a component carrying M features out
of V:

    L(eps) = M*log(1-eps) + (V-M)*log(1-eps)
             - (1-p) * E[d] * log((1-eps)/eps)

The first part is what sharpness buys on positions that match; the second is
what it costs on positions that do not. Differentiating and solving gives an
interior optimum -- sharper is better only while the fraction of novel material
is small.

Both ingredients are measured rather than assumed: p(h) and the distance
distribution come from the novelty anatomy, and V and M from the vocabulary and
the observed set sizes.

WHY THIS IS WORTH DOING
-----------------------
A sweep tells you which eps worked. A derivation tells you which eps to use
before running anything, and can be checked: it predicts a number, and the
sweep either lands there or does not. If it lands, the bound is usable as a
selection rule -- given a forecast horizon, how much sharpness, and hence how
many components, to allow. If it does not land, the bound is too loose to
prescribe anything, which is also worth knowing.
"""
import argparse, sys
import numpy as np


def L(eps, V, M, p_seen, d_mean):
    """Expected log-likelihood per sequence under the bound.

    Matching positions: V of them, each costing log(1-eps) in the sense that a
    probability of 1-eps is assigned where the component is confident.
    Mismatching positions: only novel sequences have them, E[d] on average,
    each paying the odds ratio."""
    eps = np.clip(eps, 1e-9, 0.5 - 1e-9)
    match = V * np.log(1 - eps)
    miss = (1 - p_seen) * d_mean * np.log((1 - eps) / eps)
    return match - miss


def optimal_eps(V, p_seen, d_mean):
    """dL/deps = -V/(1-eps) + (1-p)*d*[1/eps + 1/(1-eps)] = 0

    Writing q = (1-p)*d, this is  q/eps = (V - q)/(1-eps),  so

        eps* = q / V

    The optimum is simply the expected number of mismatching positions per
    sequence divided by the number of positions. Sharpness should match the
    rate at which the data actually departs from the components -- which is
    an interpretable statement, not a tuned constant."""
    q = (1 - p_seen) * d_mean
    return float(np.clip(q / max(V, 1), 1e-6, 0.49))


def beta_to_eps(a, n_eff):
    """A Beta(a,a) prior pulls a proportion estimated from n_eff observations
    toward one half. An entry that would have been 0 or 1 lands at about
    a/(n_eff + 2a), which is the eps it induces."""
    return a / (n_eff + 2.0 * a)


def eps_to_beta(eps, n_eff):
    return eps * n_eff / max(1.0 - 2.0 * eps, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=1359,
                    help="vocabulary size")
    ap.add_argument("--seen", default="0.917,0.859,0.817,0.205",
                    help="fraction of sequences whose exact set was seen in "
                         "training, one per horizon (from the novelty anatomy)")
    ap.add_argument("--dmean", default="1.5,2.0,2.5,4.0",
                    help="mean edit distance of the NOVEL sets to the nearest "
                         "training set, one per horizon")
    ap.add_argument("--horizons", default="1,2,3,6")
    ap.add_argument("--n-eff", type=float, default=2000.0,
                    help="effective observations behind a typical emission "
                         "entry: used only to translate eps into the Beta "
                         "strength that would produce it")
    ap.add_argument("--observed", default="",
                    help="optional: 'a:gain,a:gain,...' from the Beta sweep at "
                         "the LAST horizon, to check the prediction")
    args = ap.parse_args()

    hs = [int(x) for x in args.horizons.split(",")]
    ps = [float(x) for x in args.seen.split(",")]
    ds = [float(x) for x in args.dmean.split(",")]
    if not (len(hs) == len(ps) == len(ds)):
        sys.exit("horizons, seen and dmean must have the same length")
    V = args.V

    print(f"vocabulary {V:,}   effective observations per entry "
          f"{args.n_eff:,.0f}\n")
    print(f"  {'h':>3}{'seen':>9}{'E[d|novel]':>12}{'mismatches':>12}"
          f"{'eps*':>10}{'Beta a*':>10}")
    star = []
    for h, p, d in zip(hs, ps, ds):
        e = optimal_eps(V, p, d)
        a = eps_to_beta(e, args.n_eff)
        star.append((h, e, a))
        print(f"  {h:>3}{p:>9.3f}{d:>12.2f}{(1-p)*d:>12.3f}"
              f"{e:>10.5f}{a:>10.2f}")

    print(f"\n  eps* = (fraction novel) x (mean edits) / V")
    print(f"  -- the rate at which sequences depart from their component.")
    print(f"  Sharper than that and the mismatches dominate; blunter and the "
          f"matches are\n  given away for nothing.\n")

    # shape of the objective at the longest horizon
    h, p, d = hs[-1], ps[-1], ds[-1]
    print(f"  objective at h={h} (seen {p:.3f}, E[d] {d:.1f}):")
    print(f"    {'eps':>10}{'Beta a':>10}{'L(eps)':>14}{'vs eps*':>12}")
    e0 = optimal_eps(V, p, d)
    Lb = L(e0, V, V, p, d)
    for e in sorted({1e-4, 3e-4, 1e-3, e0, 3e-3, 1e-2, 3e-2, 1e-1}):
        print(f"    {e:>10.5f}{eps_to_beta(e, args.n_eff):>10.2f}"
              f"{L(e, V, V, p, d):>14.3f}{L(e, V, V, p, d) - Lb:>+12.3f}"
              + ("   <- predicted" if abs(e - e0) < 1e-12 else ""))

    if args.observed:
        print(f"\n  against the measured Beta sweep at h={hs[-1]}:")
        print(f"    {'Beta a':>10}{'measured gain':>16}"
              f"{'predicted gain':>16}")
        ref = None
        pairs = []
        for tok in args.observed.split(","):
            a, g = tok.split(":")
            pairs.append((float(a), float(g)))
        for a, g in pairs:
            e = beta_to_eps(a, args.n_eff)
            pl = L(e, V, V, p, d) - L(beta_to_eps(pairs[0][0], args.n_eff),
                                      V, V, p, d)
            print(f"    {a:>10.2f}{g:>16.3f}{pl:>16.3f}")
        best_obs = max(pairs, key=lambda t: t[1])[0]
        print(f"\n    best measured a = {best_obs:g}   "
              f"predicted a* = {eps_to_beta(e0, args.n_eff):.2f}")
        print(f"    The prediction is useful if these are the same order of "
              f"magnitude.\n    If they are not, the bound is too loose to "
              f"prescribe anything.")

    print("""
  How to use this rather than sweep:

  1. Measure the seen fraction and the novel-set distance at the horizon you
     care about. Neither needs a model -- both come from the data.
  2. eps* follows. It is the departure rate, so it rises with horizon exactly
     as novelty does.
  3. Components should be no sharper than eps*. Since sharpness grows with the
     number of components -- more components, fewer sequences each, more
     extreme estimates -- this caps K for a given horizon.

  The failure this addresses: choosing K by aggregate held-out likelihood picks
  whatever is sharpest, because most sequences in most months are familiar. On
  a month when a new variant arrives, that choice is the worst available.
""")


if __name__ == "__main__":
    sys.exit(main())
