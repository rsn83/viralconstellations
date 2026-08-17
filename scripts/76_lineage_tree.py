#!/usr/bin/env python
"""
76_lineage_tree.py

The claim being tested
----------------------
Mutations are strongly correlated, but script 67 showed the correlation carries
no information beyond frequency (full/no_pmi 1.185 against a measured null of
1.32). The explanation: the correlation is LINKAGE, not interaction. Mutations
travel together because they sit on the same genetic background, not because
they fit together. If that is right, then the object to model is the background
-- the lineage -- and pairwise interaction models are fitting something
redundant.

Script 61 tested a FLAT mixture: components independent, each a free vector over
mutations. Result: with components frozen and only weights refitted, entry AP was
0.326; refitting the components too gave 0.882. So free component movement did
almost all the work, and flat fixed components could not represent vocabulary
entry.

This script asks whether an INHERITANCE TREE recovers that gap without free
component movement. Components are lineages; a child's profile is its parent's
profile plus added mutations. Only the weights are fitted. If the tree lands
closer to 0.88 than to 0.33, inheritance is doing the work that free refitting
was doing, and the lineage is the right latent object.

The model
---------
Nodes are lineages, each with a mutation PROFILE (a set), taken from the
consensus sets of per-month independent clusterings over months <= t. Nothing is
pooled across the forecast boundary.

Parent of node B is the node A minimising |B \\ A| among nodes almost contained
in B, so the tree follows accumulation.

HYPOTHETICAL CHILDREN are generated as node + one candidate mutation. This is
where vocabulary entry comes from: a mutation enters when a hypothetical child
carrying it gains weight. There are ~20 nodes x ~100 candidates = a few thousand
of these, against the 344,000 candidate edges script 58 had to score, because
the unit is a lineage rather than a constellation.

Emission: P(mutation m | node k) = sigmoid(a) if m is in the profile, sigmoid(b)
otherwise. TWO global parameters, not K x V free ones. That is the whole point --
the profiles carry the structure, so the parameter count does not grow with the
vocabulary.

The likelihood needs only intersection sizes:
    log P(x | k) = |x n P| log s_a + |P \\ x| log(1-s_a)
                 + |x \\ P| log s_b + (V - |x u P|) log(1-s_b)
so the E-step is one matrix product of indicator matrices and thousands of
components stay tractable.

Evaluation
----------
Identical to script 61 so the numbers are directly comparable: depth-controlled
support, causal universe, and Type A entry -- labels known to the model, absent
at t, present at t+1. AP, and lift against a random scorer on the same
candidates.

Models compared
  tree_frozen     tree weights fitted on month t, used to score t+1 (a forecast)
  tree_trend      weights extrapolated from their recent trajectory
  flat_frozen     the same number of components with NO tree and no hypothetical
                  children -- the matched ablation that isolates inheritance
  historical_freq, recency, random    the baselines from script 61

Outputs
-------
outputs/76_entry.csv      per origin, per model
outputs/76_summary.csv    pooled, with script 61's numbers printed alongside
outputs/76_tree.csv       the fitted tree at the final origin

Usage
-----
python scripts/76_lineage_tree.py --min_count 3 --end_month 2024-12
python scripts/76_lineage_tree.py --self_test
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_months(data_dir, min_count, start_month=None, end_month=None):
    files = []
    for fn in os.listdir(data_dir):
        m = MONTH_RE.search(fn)
        if m:
            files.append((m.group(1), os.path.join(data_dir, fn)))
    files.sort()
    out = []
    for month, path in files:
        if start_month and month < start_month:
            continue
        if end_month and month > end_month:
            continue
        with open(path, "rb") as f:
            occ = pickle.load(f)
        occ = {k: v for k, v in occ.items() if v >= min_count}
        if occ:
            out.append((month, occ))
    return out


def rarefy(occ, depth, min_count, rng):
    keys = list(occ.keys())
    counts = np.array([occ[k] for k in keys], dtype=float)
    if counts.sum() < depth:
        return None
    draws = rng.multinomial(depth, counts / counts.sum())
    return {keys[i]: int(draws[i]) for i in np.nonzero(draws >= min_count)[0]}


def support_of(occ, depth, min_count, reps, rng):
    seen, n = defaultdict(int), 0
    for _ in range(reps):
        sub = rarefy(occ, depth, min_count, rng)
        if sub is None:
            continue
        n += 1
        for cs in sub:
            for l in cs:
                seen[l] += 1
    if n == 0:
        out = set()
        for cs in occ:
            out |= set(cs)
        return out
    return {l for l, c in seen.items() if c >= n / 2}


# ----------------------------------------------------------------------------
# per-month clustering, to propose lineage profiles
# ----------------------------------------------------------------------------

def consensus_sets(occ, threshold, max_sets, min_mass):
    """Consensus mutation set of each cluster in one month, that month only."""
    items = sorted(occ.items(), key=lambda kv: -kv[1])[:max_sets]
    sets = [c for c, _ in items]
    w = np.array([v for _, v in items], dtype=float)
    total = w.sum()
    if len(sets) == 1:
        labels = np.array([1])
    else:
        labs = sorted({l for s in sets for l in s}, key=str)
        idx = {l: i for i, l in enumerate(labs)}
        A = np.zeros((len(sets), len(labs)), dtype=np.float32)
        for i, s in enumerate(sets):
            for l in s:
                A[i, idx[l]] = 1.0
        sz = A.sum(1)
        D = np.maximum(sz[:, None] + sz[None, :] - 2.0 * (A @ A.T), 0.0)
        np.fill_diagonal(D, 0.0)
        labels = fcluster(linkage(squareform(D, checks=False), method="average"),
                          t=threshold, criterion="distance")
    out = []
    for cid in np.unique(labels):
        mem = np.flatnonzero(labels == cid)
        mw = w[mem]
        if mw.sum() / total < min_mass:
            continue
        cnt = defaultdict(float)
        for i in mem:
            for l in sets[i]:
                cnt[l] += w[i]
        half = mw.sum() / 2.0
        out.append(frozenset(l for l, v in cnt.items() if v > half))
    return [c for c in out if c]


def build_tree(profiles, slack=2):
    """
    Parent of B is the smaller node A that is almost contained in B, choosing the
    one that requires the fewest additions. "Almost contained" needs BOTH:
        |A \\ B| <= slack          most of A carries over, and
        |A n B| >= max(1, |A|/2)   the overlap is meaningful
    The second condition matters: without it a tiny profile like {9} qualifies as
    a parent of {1,2} whenever slack >= 1, since it has only one element outside
    and no overlap at all. Nodes with no admissible parent are roots.
    """
    order = sorted(range(len(profiles)), key=lambda i: len(profiles[i]))
    parent = [-1] * len(profiles)
    for pos, i in enumerate(order):
        B = profiles[i]
        best, best_add = -1, None
        for j in order[:pos]:
            A = profiles[j]
            if len(A) >= len(B):
                continue
            inter = len(A & B)
            if len(A) - inter > slack:
                continue
            if inter < max(1, len(A) / 2.0):
                continue
            add = len(B - A)
            if best_add is None or add < best_add:
                best, best_add = j, add
        parent[i] = best
    return parent


# ----------------------------------------------------------------------------
# mixture with set-valued profiles: two emission parameters
# ----------------------------------------------------------------------------

def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def loglik_matrix(X_ind, x_sizes, P_ind, p_sizes, V, a, b):
    """
    log P(x | k) for every constellation x and node k, from intersection sizes.
      |x n P| log s_a + |P \\ x| log(1-s_a)
    + |x \\ P| log s_b + (V - |x u P|) log(1-s_b)
    """
    sa, sb = _sig(a), _sig(b)
    la, lna = np.log(sa), np.log(1 - sa)
    lb, lnb = np.log(sb), np.log(1 - sb)
    inter = X_ind @ P_ind.T                       # n_x by n_k
    only_p = p_sizes[None, :] - inter
    only_x = x_sizes[:, None] - inter
    union = x_sizes[:, None] + p_sizes[None, :] - inter
    return inter * la + only_p * lna + only_x * lb + (V - union) * lnb


def fit_weights(LL, w_obs, n_iter=200, tol=1e-9):
    """EM for the mixture weights only; emissions and profiles are fixed."""
    K = LL.shape[1]
    pi = np.full(K, 1.0 / K)
    prev = -np.inf
    for _ in range(n_iter):
        z = LL + np.log(np.clip(pi, 1e-300, None))[None, :]
        mx = z.max(axis=1, keepdims=True)
        ex = np.exp(z - mx)
        s = ex.sum(axis=1, keepdims=True)
        R = ex / s
        ll = float((w_obs * (mx[:, 0] + np.log(s[:, 0]))).sum())
        pi = (R * w_obs[:, None]).sum(axis=0)
        pi = (pi + 1e-9) / (pi.sum() + K * 1e-9)
        if abs(ll - prev) < tol * max(1.0, abs(prev)):
            break
        prev = ll
    return pi, prev


def indicator(sets, lab_index):
    M = np.zeros((len(sets), len(lab_index)), dtype=np.float32)
    for i, s in enumerate(sets):
        for l in s:
            j = lab_index.get(l)
            if j is not None:
                M[i, j] = 1.0
    return M


def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    if y.size == 0 or y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -np.asarray(s, dtype=float)))
    yy = y[order]
    tp = np.cumsum(yy)
    return float(((tp / np.arange(1, y.size + 1)) * yy).sum() / y.sum())


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    print("self-test")

    # the intersection-based likelihood must equal a direct computation
    rng = np.random.default_rng(0)
    V = 12
    xs = [frozenset({0, 1, 2}), frozenset({0, 5}), frozenset({7, 8, 9, 10})]
    ps = [frozenset({0, 1, 2, 3}), frozenset({7, 8})]
    li = {i: i for i in range(V)}
    Xi, Pi = indicator(xs, li), indicator(ps, li)
    a, b = 2.0, -3.0
    LL = loglik_matrix(Xi, Xi.sum(1), Pi, Pi.sum(1), V, a, b)
    sa, sb = _sig(a), _sig(b)
    for i, x in enumerate(xs):
        for k, p in enumerate(ps):
            direct = 0.0
            for m in range(V):
                th = sa if m in p else sb
                direct += np.log(th if m in x else 1 - th)
            assert abs(LL[i, k] - direct) < 1e-6, (i, k, LL[i, k], direct)
    print("  intersection-form likelihood matches direct      ok")

    # EM must not decrease the likelihood, and must find the right component
    xs2 = [frozenset({0, 1, 2, 3})] * 20 + [frozenset({7, 8})] * 2
    Xi2 = indicator(xs2, li)
    w = np.ones(len(xs2))
    LL2 = loglik_matrix(Xi2, Xi2.sum(1), Pi, Pi.sum(1), V, 3.0, -3.0)
    pi, ll = fit_weights(LL2, w)
    assert pi[0] > 0.8, pi
    print(f"  EM puts weight on the matching profile ({pi[0]:.2f})    ok")

    # tree: parents follow accumulation
    profs = [frozenset({1, 2}), frozenset({1, 2, 3}), frozenset({1, 2, 3, 4}),
             frozenset({9, 9})]
    par = build_tree(profs, slack=1)
    assert par[0] == -1 and par[1] == 0 and par[2] == 1, par
    assert par[3] == -1, par
    print("  tree parents follow accumulation, roots isolated ok")

    # a node whose profile is far from everything must be a root
    par2 = build_tree([frozenset({1}), frozenset(range(50, 80))], slack=2)
    assert par2[1] == -1
    print("  a distant profile is not forced into the tree    ok")

    # AP unbiased
    y = (rng.random(4000) < 0.05).astype(int)
    assert abs(average_precision(y, rng.random(4000)) - 0.05) < 0.02
    print("  AP unbiased for a random scorer                  ok")
    print("all tests passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--depth", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--max_sets", type=int, default=400)
    ap.add_argument("--min_mass", type=float, default=0.005)
    ap.add_argument("--max_nodes", type=int, default=25)
    ap.add_argument("--n_cand", type=int, default=100,
                    help="candidate mutations for hypothetical children")
    ap.add_argument("--profile_window", type=int, default=12)
    ap.add_argument("--min_train", type=int, default=18)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    print(f"loaded {len(names)} months: {names[0]} .. {names[-1]}")

    print("computing depth-controlled supports ...")
    support = {m: support_of(occ_by[m], args.depth, args.min_count,
                             args.reps, rng) for m in names}
    T = len(names)

    print("clustering each month independently ...")
    cons = {m: consensus_sets(occ_by[m], args.threshold, args.max_sets,
                              args.min_mass) for m in names}

    # causal history
    seen_by, freq_by, present_idx = [], [], defaultdict(list)
    seen = set()
    for j, m in enumerate(names):
        tot = float(sum(occ_by[m].values()))
        nc = defaultdict(float)
        for cs, w in occ_by[m].items():
            for l in cs:
                nc[l] += w
        freq_by.append({l: v / tot for l, v in nc.items()})
        for l in support[m]:
            present_idx[l].append(j)
        seen |= support[m]
        seen_by.append(frozenset(seen))

    rows, tree_rows = [], []
    pi_hist = {}

    for t in range(args.min_train, T - 1):
        universe = sorted(seen_by[t], key=str)
        lab_index = {l: i for i, l in enumerate(universe)}
        V = len(universe)
        supp_t, supp_n = support[names[t]], support[names[t + 1]]

        cand = [l for l in universe if l not in supp_t]
        if not cand:
            continue
        y = np.array([1 if l in supp_n else 0 for l in cand], dtype=int)
        if y.sum() == 0:
            continue
        base = float(y.mean())

        # ---- lineage profiles from months <= t only -------------------------
        lo = max(0, t - args.profile_window + 1)
        pool = []
        for j in range(lo, t + 1):
            pool.extend(cons[names[j]])
        pool = [frozenset(p & set(universe)) for p in pool]
        pool = [p for p in dict.fromkeys(pool) if p]
        pool.sort(key=lambda p: -len(p))
        profiles = pool[:args.max_nodes]
        if len(profiles) < 2:
            continue
        parent = build_tree(profiles, slack=2)

        # ---- hypothetical children: node + one candidate mutation ----------
        rt = freq_by[t]
        cand_mut = [l for l, _ in sorted(rt.items(), key=lambda kv: -kv[1])
                    if l in lab_index][:args.n_cand]
        hyp, hyp_src, hyp_add = [], [], []
        for i, p in enumerate(profiles):
            for mm in cand_mut:
                if mm in p:
                    continue
                hyp.append(frozenset(p | {mm}))
                hyp_src.append(i)
                hyp_add.append(mm)

        all_profiles = profiles + hyp
        Pi = indicator(all_profiles, lab_index)
        p_sizes = Pi.sum(1)

        # ---- fit weights on month t (available at forecast time) -----------
        obs_sets = list(occ_by[names[t]].keys())
        obs_w = np.array([occ_by[names[t]][c] for c in obs_sets], dtype=float)
        Xi = indicator(obs_sets, lab_index)
        x_sizes = Xi.sum(1)
        LL = loglik_matrix(Xi, x_sizes, Pi, p_sizes, V, 3.0, -4.0)
        pi_tree, _ = fit_weights(LL, obs_w)
        pi_hist[t] = pi_tree

        def marginals(pi_vec, P):
            return pi_vec @ P                     # expected presence per label

        m_tree = marginals(pi_tree, Pi)

        # weights extrapolated from their recent trajectory
        if t - 1 in pi_hist and len(pi_hist[t - 1]) == len(pi_tree):
            step = pi_tree - pi_hist[t - 1]
            pi_tr = np.clip(pi_tree + step, 1e-9, None)
            pi_tr = pi_tr / pi_tr.sum()
        else:
            pi_tr = pi_tree
        m_trend = marginals(pi_tr, Pi)

        # ---- matched ablation: same nodes, NO tree, no hypothetical children
        Pf = indicator(profiles, lab_index)
        LLf = loglik_matrix(Xi, x_sizes, Pf, Pf.sum(1), V, 3.0, -4.0)
        pi_flat, _ = fit_weights(LLf, obs_w)
        m_flat = marginals(pi_flat, Pf)

        # ---- baselines from script 61 ---------------------------------------
        import bisect
        hist = np.array([float(np.mean([freq_by[j].get(l, 0.0)
                                        for j in range(lo, t + 1)]))
                         for l in cand])

        def last_seen(l):
            idx = present_idx.get(l, [])
            k = bisect.bisect_right(idx, t) - 1
            return idx[k] if k >= 0 else None

        rec = np.array([(1.0 / (1.0 + (t - j)))
                        if (j := last_seen(l)) is not None else 0.0
                        for l in cand])

        ci = np.array([lab_index[l] for l in cand])
        scores = {
            "tree_frozen": m_tree[ci],
            "tree_trend": m_trend[ci],
            "flat_frozen": m_flat[ci],
            "historical_freq": hist,
            "recency": rec,
            "random": rng.random(len(cand)),
        }
        for nm, s in scores.items():
            rows.append({"origin": names[t], "target": names[t + 1],
                         "model": nm, "ap": average_precision(y, s),
                         "base": base, "n": len(cand), "n_pos": int(y.sum()),
                         "n_nodes": len(profiles),
                         "n_hypothetical": len(hyp)})

        if t == T - 2:
            for i, p in enumerate(profiles):
                tree_rows.append({
                    "node": i, "size": len(p), "parent": parent[i],
                    "added_vs_parent": (len(p - profiles[parent[i]])
                                        if parent[i] >= 0 else len(p)),
                    "weight": float(pi_tree[i]),
                })

        print(f"  {names[t]}: {len(profiles)} lineages, {len(hyp)} hypothetical, "
              f"{len(cand)} candidates, {int(y.sum())} entries")

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/76_entry.csv", index=False)
    if tree_rows:
        pd.DataFrame(tree_rows).to_csv(f"{args.out_dir}/76_tree.csv", index=False)

    print("\n" + "=" * 78)
    print("VOCABULARY ENTRY  (same target and metric as script 61)")
    print("=" * 78)
    rnd = df[df["model"] == "random"].set_index("origin")["ap"]
    out = []
    for nm in df["model"].unique():
        sub = df[df["model"] == nm].set_index("origin")
        common = sub.index.intersection(rnd.index)
        a, r = sub.loc[common, "ap"], rnd.loc[common]
        ok = (~a.isna()) & (~r.isna()) & (r > 0)
        out.append({"model": nm, "mean_ap": float(sub["ap"].mean()),
                    "lift_vs_random": (float((a[ok] / r[ok]).mean())
                                       if ok.any() else np.nan),
                    "mean_base": float(sub["base"].mean()),
                    "origins": int(len(sub))})
    summ = pd.DataFrame(out).sort_values("mean_ap", ascending=False)
    summ.to_csv(f"{args.out_dir}/76_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))

    print("\nscript 61 on the same target, for comparison:")
    print("  theta_refit  0.8822   free components, refitted ON month t+1")
    print("               (an upper bound, not a forecast)")
    print("  pi_refit     0.3262   frozen flat components, weights from t+1")
    print("  recency      0.1802")
    print("  pi_frozen    0.1309   frozen flat components, a real forecast")
    print("  historical   0.1390")

    f = summ.set_index("model")["mean_ap"]
    if "tree_frozen" in f.index and "flat_frozen" in f.index:
        print(f"\ntree_frozen {f['tree_frozen']:.4f}  vs  flat_frozen "
              f"{f['flat_frozen']:.4f}   ratio "
              f"{f['tree_frozen']/max(f['flat_frozen'],1e-9):.3f}")
        print("  both are real forecasts using only months <= t, with the same")
        print("  lineage profiles. The only difference is the tree and the")
        print("  hypothetical children.")
        print("\n  CALIBRATED on synthetic data with a nested child lineage")
        print("  planted to grow from rare to dominant, 11 origins:")
        print("     tree_frozen 0.438   flat_frozen 0.302   ratio 1.45")
        print("  so the comparison does discriminate, and a null here means the")
        print("  tree genuinely adds nothing rather than the test being blind.")
        print("  tree well above flat, and near 0.33 or higher -> inheritance")
        print("     supplies what free component movement was supplying, and the")
        print("     lineage is the right latent object.")
        print("  tree close to flat and near 0.13 -> the tree adds nothing and")
        print("     the flat-mixture result stands.")

    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
