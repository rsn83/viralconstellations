#!/usr/bin/env python
"""
115_factored_emission.py -- does splitting the emission into "which position
changed" and "what it changed to" beat one coin per node?

THE PROBLEM WITH THE CURRENT EMISSION
-------------------------------------
Right now every (position, residue) pair is its own independent coin. So 501Y
and 501N are unrelated events that merely happen never to co-occur, and the
model puts probability mass on sequences carrying both. The same goes for a
deletion covering position 212 and a substitution at 212 -- impossible in the
data, but the model does not know that. Those impossible pairs are exactly the
ones that dominate the leftover-dependence measurement in 113, so part of what
looks like epistasis is really the encoding.

THE FACTORED EMISSION
---------------------
Two questions per position instead of one coin per node:

    does position p differ from the reference?      Bernoulli(rho[k,p])
    given that it differs, which residue is it?     Categorical(psi[k,p,.])

    P(node (p,r) | block k) = rho[k,p] * psi[k,p,r]

The categorical sums to one, so the residues at a position are mutually
exclusive by construction. A deletion is just another category alongside the
substitutions, so deletion/substitution conflicts vanish without a mask.

TIED OR UNTIED
--------------
untied  psi is per block: K*P + K*P*R parameters. At K=48 that is roughly 15x
        the current model.
tied    psi is shared across blocks: K*P + P*R. Fewer parameters than the
        current model, because psi no longer scales with K and there are more
        nodes than positions.

Tying encodes a claim worth testing rather than assuming: WHICH positions a
lineage mutates is what distinguishes lineages; WHAT residue appears there is
mostly chemistry and codon accessibility, shared across lineages. Running both
tests it.

WHAT IS HELD FIXED
------------------
Same months, same K, same initialisation, same number of EM steps, no drift on
either side. The only thing that differs is the emission family, so the
held-out difference is attributable to it.
"""
import argparse, csv, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_positions(path):
    """node index -> position group index, plus a readable name per node."""
    pos_of, names, V = {}, {}, 0
    seen = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = int(row["node_idx"]); V = max(V, i + 1)
            aa = str(row["aa_pos"]).strip()
            if aa not in seen:
                seen[aa] = len(seen)
            pos_of[i] = seen[aa]
            names[i] = f"{aa}{row['residue'].strip()}"
    g = np.zeros(V, dtype=int)
    for i in range(V):
        g[i] = pos_of.get(i, len(seen) + i)      # unlisted nodes get their own
    P = int(g.max()) + 1
    return g, P, names, V


# ---------------------------------------------------------------- emissions
class Bernoulli:
    """One independent coin per node. What the current model does."""
    name = "bernoulli per node"

    def __init__(self, K, V, g, P, rng, tie=False):
        self.K, self.V = K, V

    def init(self, X, w, R):
        self.m_step(X, w, R)

    def scores(self):
        """(per-node score, constant) such that log P(S|k) = const + sum score."""
        th = np.clip(self.th, 1e-4, 1 - 1e-4)
        return np.log(th) - np.log(1 - th), np.log(1 - th).sum(1)

    def m_step(self, X, w, R):
        wr = w[:, None] * R                       # N x K
        tot = wr.sum(0) + 1e-9
        self.th = (wr.T @ X + .5) / (tot[:, None] + 1.0)

    def n_params(self):
        return self.K * self.V


class Factored:
    """Position occupancy times residue identity."""

    def __init__(self, K, V, g, P, rng, tie=True):
        self.K, self.V, self.g, self.P, self.tie = K, V, g, P, tie
        self.name = f"factored, psi {'tied' if tie else 'per block'}"
        # group -> node index list, as a sparse sum matrix (P x V)
        self.G = np.zeros((P, V))
        self.G[g, np.arange(V)] = 1.0

    def init(self, X, w, R):
        self.m_step(X, w, R)

    def scores(self):
        rho = np.clip(self.rho, 1e-6, 1 - 1e-6)            # K x P
        psi = np.clip(self.psi, 1e-6, 1.0)                 # K x V
        # log rho - log(1-rho) expanded from positions back onto nodes
        lr = (np.log(rho) - np.log(1 - rho))[:, self.g]
        return lr + np.log(psi), np.log(1 - rho).sum(1)

    def m_step(self, X, w, R):
        wr = w[:, None] * R
        tot = wr.sum(0) + 1e-9                             # K
        node = wr.T @ X                                    # K x V, weighted counts
        posc = node @ self.G.T                             # K x P
        self.rho = (posc + .5) / (tot[:, None] + 1.0)
        if self.tie:
            n = node.sum(0); p = posc.sum(0)               # pooled over blocks
            self.psi = ((n + .5) / (p[self.g] + .5 * np.bincount(
                self.g, minlength=self.P)[self.g]))[None, :].repeat(self.K, 0)
        else:
            cnt = np.bincount(self.g, minlength=self.P)[self.g][None, :]
            self.psi = (node + .5) / (posc[:, self.g] + .5 * cnt)

    def n_params(self):
        R = self.V                                          # nodes across all groups
        return self.K * self.P + (R if self.tie else self.K * R)


# ---------------------------------------------------------------- EM
def em(emis, Xs, ws, K, iters, rng, verbose=False):
    Xall = np.vstack(Xs); wall = np.concatenate(ws)
    N = len(Xall)
    R = rng.dirichlet(np.ones(K) * 5, size=N)
    emis.init(Xall, wall, R)
    T = len(Xs)
    Pi = np.full((T, K), 1.0 / K)
    prev = -np.inf
    for it in range(iters):
        Rs, lls, offs = [], 0.0, 0
        A, b = emis.scores()
        tot_w = 0.0
        for t, (X, w) in enumerate(zip(Xs, ws)):
            lp = X @ A.T + b[None, :] + np.log(Pi[t] + EPS)[None, :]
            mx = lp.max(1, keepdims=True)
            e = np.exp(lp - mx); s = e.sum(1, keepdims=True)
            lls += float((w * (np.log(s.ravel() + EPS) + mx.ravel())).sum())
            tot_w += w.sum()
            r = e / np.maximum(s, EPS)
            Rs.append(r)
            Pi[t] = (w[:, None] * r).sum(0)
            Pi[t] /= max(Pi[t].sum(), EPS)
        ll = lls / tot_w
        emis.m_step(Xall, wall, np.vstack(Rs))
        if verbose and (it + 1) % 20 == 0:
            print(f"      {emis.name:26s} iter {it+1:3d}  LL/obs {ll:.5f}",
                  flush=True)
        if abs(ll - prev) < 1e-7:
            break
        prev = ll
    return Pi, ll, it + 1


def score(emis, X, w, pi):
    A, b = emis.scores()
    lp = X @ A.T + b[None, :] + np.log(pi + EPS)[None, :]
    mx = lp.max(1, keepdims=True)
    v = (np.log(np.exp(lp - mx).sum(1, keepdims=True)) + mx).ravel()
    return float((w * v).sum() / w.sum())


def impossible_mass(emis, g, P):
    """Probability the model puts on sequences that cannot exist: two different
    residues at the same position at once. Zero for the factored family by
    construction; a real quantity for the current one. Reported as the expected
    number of such conflicting positions per sequence."""
    A, b = emis.scores()
    th = 1.0 / (1.0 + np.exp(-(A)))            # only meaningful for Bernoulli
    if isinstance(emis, Factored):
        return 0.0
    p = np.clip(emis.th, 1e-9, 1 - 1e-9)
    out = 0.0
    for k in range(p.shape[0]):
        for start in range(P):
            idx = np.flatnonzero(g == start)
            if len(idx) < 2: continue
            q = p[k, idx]
            none = np.prod(1 - q)
            one = sum(q[i] * np.prod(np.delete(1 - q, i)) for i in range(len(q)))
            out += max(0.0, 1.0 - none - one)
    return out / p.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--K-list", default="13,24,48")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3,
                    help="plain EM from a random start is init-sensitive, so "
                         "one run tells you very little; the spread is the point")
    args = ap.parse_args()

    E = load_engine(args.engine)
    g, P, names, V = load_positions(args.vocab)
    tr = E.months_in_range(args.train)
    recs = [E.load_month(args.data_dir, ym) for ym in tr]
    if any(r is None for r in recs):
        sys.exit("missing training months")
    Xs, ws = zip(*[E.build(r, V, args.min_count) for r in recs])
    Xs, ws = list(Xs), list(ws)
    rte = E.load_month(args.data_dir, args.test)
    if rte is None:
        sys.exit(f"missing test month {args.test}")
    Xte, wte = E.build(rte, V, 1)

    multi = int((np.bincount(g, minlength=P) > 1).sum())
    print(f"train {tr[0]}..{tr[-1]}   test {args.test}")
    print(f"nodes {V:,}   positions {P:,}   "
          f"positions carrying more than one node {multi:,}")
    print(f"(only those positions can produce an impossible pair under the "
          f"current emission)\n")

    rows = []
    for K in (int(x) for x in args.K_list.split(",")):
        for cls, tie in ((Bernoulli, False), (Factored, True), (Factored, False)):
            ins, outs, npar, nm = [], [], None, None
            for sd in range(args.seeds):
                # same seed across families at a given K, so all three start
                # from the same responsibilities and only the emission differs
                rng = np.random.default_rng(1000 * sd + args.__dict__.get("base", 0))
                emis = cls(K, V, g, P, rng, tie=tie)
                Pi, ll_in, nit = em(emis, Xs, ws, K, args.iters, rng)
                ins.append(ll_in); outs.append(score(emis, Xte, wte, Pi[-1]))
                npar, nm = emis.n_params(), emis.name
            rows.append(dict(K=K, name=nm, npar=npar,
                             ll_in=float(np.mean(ins)),
                             ll_out=float(np.mean(outs)),
                             sd=float(np.std(outs)),
                             lo=float(np.min(outs)), hi=float(np.max(outs))))
            r = rows[-1]
            print(f"  K={K:<4}{nm:26s} params {npar:>10,}"
                  f"   train {r['ll_in']:8.3f}   held-out {r['ll_out']:8.3f}"
                  f" +/-{r['sd']:.3f}   [{r['lo']:.3f}, {r['hi']:.3f}]",
                  flush=True)

    print(f"\n{'='*88}\n  EMISSION FAMILY COMPARISON   test {args.test}\n{'='*88}")
    print(f"\n  {'K':>5}  {'emission':<26}{'params':>12}{'train':>10}"
          f"{'held-out':>10}{'+/- sd':>9}{'range':>20}{'vs bern':>10}")
    base = {r["K"]: r["ll_out"] for r in rows if r["name"].startswith("bernoulli")}
    for r in rows:
        d = r["ll_out"] - base[r["K"]]
        mark = "" if r["name"].startswith("bernoulli") else f"{d:+.3f}"
        rng_s = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
        print(f"  {r['K']:>5}  {r['name']:<26}{r['npar']:>12,}"
              f"{r['ll_in']:>10.3f}{r['ll_out']:>10.3f}{r['sd']:>9.3f}"
              f"{rng_s:>20}{mark:>10}")
    print("""
  Same K, same data, same initialisation, no drift on either side. The only
  difference is the emission family.

  What to look for:
    tied psi beats bernoulli with FEWER parameters
        -> the current emission was spending capacity on residue identity that
           is really shared chemistry, and putting mass on impossible pairs.
    untied beats tied
        -> which residue appears is lineage-specific after all, and the tying
           assumption is wrong.
  A difference smaller than the seed spread is not a difference. Plain EM from
  a random start lands in different optima at large K, which is the same
  init-sensitivity the main model needed split-merge and a warm start to fix.

    neither beats bernoulli
        -> mutual exclusion was not costing anything; keep the simpler model
           and fix the deletion encoding directly instead.
""")


if __name__ == "__main__":
    sys.exit(main())
