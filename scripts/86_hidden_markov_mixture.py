#!/usr/bin/env python3
"""
86_hidden_markov_mixture.py

Bernoulli mixture over mutation sets, with and without cross-month coupling.

THREE MODELS
  M-A  per-month mixtures, fitted INDEPENDENTLY.
       Each month gets its own theta. Block labels are arbitrary per month:
       month t's "block 2" has no relation to month t+1's "block 2".
       This is the correspondence problem, instantiated.

  M-B  ONE theta shared across all months, pi_t free per month.
       Block identity is shared by construction. Fitted by EM over pooled months.

  M-C  M-B plus a transition matrix A fitted to the pi_t trajectory,
       giving pi_{t+h} = pi_t A^h. This is what forecasts.

BASELINE
  M-F  frequency / independence: single Bernoulli profile, K = 1.
       Exactly the null your project tests against.

EVALUATION
  (1) CORRESPONDENCE. Pool two adjacent months, compute ARI/NMI between inferred
      block and Pango lineage. M-A must first match labels across months
      (Hungarian on the confusion matrix) -- if similarity alone cannot recover
      identity, M-A scores badly and M-B does not. That gap is the result.
  (2) HELD-OUT LIKELIHOOD on a future month, M-C (pi via A) vs M-F.
  (3) APPEARANCE. Rank genuinely new sets against hard negatives (size-matched,
      frequency-sampled). Report AUC and lift over the frequency baseline.

Usage:
  python 86_hidden_markov_mixture.py \
      --data-dir data/processed/full_data_graphs_posres \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --train    2020-03:2020-12 --test 2021-01 \
      --K 8 --pango data/processed/pango_by_set.tsv
"""
import argparse, pickle, sys, csv
from pathlib import Path
import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------- data
def months_in_range(spec):
    if ":" not in spec:
        return [spec]
    a, b = spec.split(":")
    ya, ma = map(int, a.split("-")); yb, mb = map(int, b.split("-"))
    out, y, m = [], ya, ma
    while (y, m) <= (yb, mb):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13: m, y = 1, y + 1
    return out


def load_month(data_dir, ym):
    """-> list of (frozenset(node_ids), count)"""
    p = Path(data_dir) / f"{ym}_occupied.pkl"
    obj = pickle.load(open(p, "rb"))
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and isinstance(vals[0], (int, np.integer)):
            return [(frozenset(k), int(v)) for k, v in obj.items()]
        for key in ("sets", "occupied", "constellations"):
            if key in obj:
                return [(frozenset(s), 1) for s in obj[key]]
        return [(frozenset(v), 1) for v in vals]
    items = list(obj)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:
        return [(frozenset(s), int(c)) for s, c in items]
    return [(frozenset(s), 1) for s in items]


def load_vocab_size(path):
    n = 0
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            n = max(n, int(row["node_idx"]) + 1)
    return n


def load_pango(path):
    """Optional TSV: sorted node ids joined by ',' -> lineage. Used ONLY for scoring."""
    if not path or not Path(path).exists():
        return None
    m = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            key = frozenset(int(x) for x in parts[0].split(",") if x != "")
            m[key] = parts[1]
    return m


def build_month_matrix(records, V):
    """-> X (n_unique, V) float32 binary, w (n_unique,) counts, sets list"""
    sets = [s for s, _ in records]
    w = np.array([c for _, c in records], dtype=np.float64)
    X = np.zeros((len(sets), V), dtype=np.float32)
    for i, s in enumerate(sets):
        idx = [n for n in s if 0 <= n < V]
        X[i, idx] = 1.0
    return X, w, sets


# ---------------------------------------------------------------- EM core
def loglik_matrix(X, theta):
    """log p(S_i | z=k) for all i,k.  (n, K)

    log p = sum_{n in S} log th + sum_{n not in S} log(1-th)
          = X @ [log th - log(1-th)]^T + sum_n log(1-th)
    """
    lt = np.log(theta + EPS)
    lc = np.log(1.0 - theta + EPS)
    return X @ (lt - lc).T + lc.sum(axis=1)[None, :]


def responsibilities(X, theta, log_pi):
    lp = loglik_matrix(X, theta) + log_pi[None, :]
    mx = lp.max(axis=1, keepdims=True)
    P = np.exp(lp - mx)
    Z = P.sum(axis=1, keepdims=True)
    return P / Z, (np.log(Z) + mx).ravel()          # resp, per-set log p(S)


def em_single(X, w, K, iters=200, tol=1e-6, seed=0, verbose=False, prior=0.5):
    """Plain Bernoulli mixture on one block of data (used for M-A and M-F)."""
    rng = np.random.default_rng(seed)
    n, V = X.shape
    theta = np.clip(rng.random((K, V)) * 0.4 + X.mean(axis=0)[None, :] * 0.6, 0.02, 0.98)
    pi = np.full(K, 1.0 / K)
    prev = -np.inf
    for it in range(iters):
        R, lps = responsibilities(X, theta, np.log(pi + EPS))
        ll = float((w * lps).sum() / w.sum())
        Rw = R * w[:, None]
        Nk = Rw.sum(axis=0) + EPS
        theta = (Rw.T @ X + prior) / (Nk[:, None] + 2 * prior)
        theta = np.clip(theta, 1e-4, 1 - 1e-4)
        pi = Nk / Nk.sum()
        if verbose and (it + 1) % 50 == 0:
            print(f"      iter {it+1}  LL/seq = {ll:.4f}", flush=True)
        if abs(ll - prev) < tol: break
        prev = ll
    return theta, pi, ll


def em_shared(Xs, ws, K, iters=300, tol=1e-6, seed=0, verbose=True, prior=0.5):
    """M-B: ONE theta shared across months, pi_t free per month.

    E-step is per month (uses that month's pi_t); M-step pools all months to
    update the single theta. That pooling is what makes block identity
    consistent across months.
    """
    rng = np.random.default_rng(seed)
    V = Xs[0].shape[1]
    Xall = np.vstack(Xs)
    theta = np.clip(rng.random((K, V)) * 0.4 + Xall.mean(axis=0)[None, :] * 0.6, 0.02, 0.98)
    Pi = np.full((len(Xs), K), 1.0 / K)
    prev = -np.inf
    for it in range(iters):
        num = np.zeros((K, V)); den = np.zeros(K); tot_ll = 0.0; tot_w = 0.0
        newPi = np.zeros_like(Pi)
        for t, (X, w) in enumerate(zip(Xs, ws)):
            R, lps = responsibilities(X, theta, np.log(Pi[t] + EPS))
            Rw = R * w[:, None]
            num += Rw.T @ X
            den += Rw.sum(axis=0)
            newPi[t] = Rw.sum(axis=0) / (w.sum() + EPS)
            tot_ll += float((w * lps).sum()); tot_w += w.sum()
        theta = np.clip((num + prior) / (den[:, None] + 2 * prior), 1e-4, 1 - 1e-4)
        Pi = newPi / newPi.sum(axis=1, keepdims=True)
        ll = tot_ll / tot_w
        if verbose and (it + 1) % 50 == 0:
            print(f"      iter {it+1}  LL/seq = {ll:.4f}", flush=True)
        if abs(ll - prev) < tol:
            if verbose: print(f"      converged at iter {it+1}", flush=True)
            break
        prev = ll
    return theta, Pi, ll


def fit_A(Pi, weights=None, ridge=1e-3):
    """M-C: least-squares A with pi_t A ~ pi_{t+1}, rows on the simplex.
    Weighted by sqrt(monthly sequence count) so thin months count less."""
    Xp, Yp = Pi[:-1], Pi[1:]
    if weights is not None:
        s = np.sqrt(weights[:-1])[:, None]
        Xp, Yp = Xp * s, Yp * s
    K = Pi.shape[1]
    A = np.linalg.solve(Xp.T @ Xp + ridge * np.eye(K), Xp.T @ Yp)
    A = np.clip(A, 1e-6, None)
    return A / A.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- evaluation
def hungarian_match(labels_a, labels_b, K):
    """Best permutation aligning two labelings (used to give M-A its best shot)."""
    from scipy.optimize import linear_sum_assignment
    C = np.zeros((K, K))
    for a, b in zip(labels_a, labels_b):
        C[a, b] += 1
    r, c = linear_sum_assignment(-C)
    return dict(zip(c, r))


def eval_correspondence(theta_A_list, Pi_A_list, theta_B, Pi_B,
                        Xs, ws, sets_list, months, pango, K):
    """The correspondence test.

    Pool two adjacent months. Compute ARI/NMI between inferred block and Pango.
    M-A gets label matching via Hungarian (its best case). M-B needs none.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rows = []
    for t in range(len(months) - 1):
        idx = [t, t + 1]
        truth, lab_A, lab_B = [], [], []
        # match month t+1's M-A labels onto month t's, using Pango-free overlap
        R0, _ = responsibilities(Xs[t],     theta_A_list[t],     np.log(Pi_A_list[t] + EPS))
        R1s, _ = responsibilities(Xs[t + 1], theta_A_list[t],     np.log(Pi_A_list[t] + EPS))
        R1o, _ = responsibilities(Xs[t + 1], theta_A_list[t + 1], np.log(Pi_A_list[t + 1] + EPS))
        perm = hungarian_match(R1o.argmax(1), R1s.argmax(1), K)
        for j, tt in enumerate(idx):
            R, _ = responsibilities(Xs[tt], theta_B, np.log(Pi_B[tt] + EPS))
            zB = R.argmax(1)
            if j == 0:
                zA = R0.argmax(1)
            else:
                zA = np.array([perm.get(z, z) for z in R1o.argmax(1)])
            for i, s in enumerate(sets_list[tt]):
                lin = pango.get(s)
                if lin is None: continue
                rep = int(min(ws[tt][i], 50))          # cap so huge sets don't dominate
                truth += [lin] * rep
                lab_A += [int(zA[i])] * rep
                lab_B += [int(zB[i])] * rep
        if len(set(truth)) < 2:
            continue
        rows.append((f"{months[t]}+{months[t+1]}", len(truth),
                     adjusted_rand_score(truth, lab_A), normalized_mutual_info_score(truth, lab_A),
                     adjusted_rand_score(truth, lab_B), normalized_mutual_info_score(truth, lab_B)))
    return rows


def score_sets(X, theta, pi):
    """log p(S) under a mixture."""
    lp = loglik_matrix(X, theta) + np.log(pi + EPS)[None, :]
    mx = lp.max(axis=1, keepdims=True)
    return (np.log(np.exp(lp - mx).sum(axis=1, keepdims=True)) + mx).ravel()


def make_negatives(pos_sets, node_freq, V, rng, mult=5):
    """Hard negatives: size-matched, nodes drawn proportional to frequency.
    A model that only knows marginal frequency cannot separate these from
    real sets -- which is exactly the discrimination we are testing."""
    p = node_freq / node_freq.sum()
    live = np.flatnonzero(node_freq > 0)
    negs = []
    for s in pos_sets:
        k = max(1, len(s))
        for _ in range(mult):
            pick = rng.choice(live, size=min(k, len(live)), replace=False,
                              p=p[live] / p[live].sum())
            negs.append(frozenset(int(x) for x in pick))
    return negs


def sets_to_X(sets, V):
    X = np.zeros((len(sets), V), dtype=np.float32)
    for i, s in enumerate(sets):
        idx = [n for n in s if 0 <= n < V]
        X[i, idx] = 1.0
    return X


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--K-sweep", type=str, default="", help="e.g. 2,4,8,16,32")
    ap.add_argument("--pango", default="", help="optional TSV: 'n1,n2,n3<TAB>lineage'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/86_mixture.npz")
    args = ap.parse_args()

    V = load_vocab_size(args.vocab)
    tr_months = months_in_range(args.train)
    te_months = months_in_range(args.test)
    pango = load_pango(args.pango)
    print(f"vocab size V = {V:,}   K = {args.K}   pango labels: "
          f"{'yes (' + str(len(pango)) + ' sets)' if pango else 'NO -- correspondence test skipped'}")

    Xs, ws, sets_list = [], [], []
    print(f"\nloading train {tr_months[0]} .. {tr_months[-1]}")
    for ym in tr_months:
        X, w, s = build_month_matrix(load_month(args.data_dir, ym), V)
        Xs.append(X); ws.append(w); sets_list.append(s)
        print(f"  {ym}: {len(s):,} unique sets, {w.sum():,.0f} sequences", flush=True)
    vol = np.array([w.sum() for w in ws])

    te_recs = []
    for ym in te_months: te_recs += load_month(args.data_dir, ym)
    Xte, wte, sets_te = build_month_matrix(te_recs, V)
    print(f"test {te_months}: {len(sets_te):,} unique sets, {wte.sum():,.0f} sequences")

    Ks = [int(x) for x in args.K_sweep.split(",")] if args.K_sweep else [args.K]

    # ---------------- M-F frequency baseline (K=1) ----------------
    print("\n" + "=" * 74)
    print("M-F  frequency baseline (K=1, independence)")
    print("=" * 74)
    Xall = np.vstack(Xs); wall = np.concatenate(ws)
    theta_F = np.clip(((wall[:, None] * Xall).sum(0) + 0.5) / (wall.sum() + 1.0),
                      1e-4, 1 - 1e-4)[None, :]
    pi_F = np.array([1.0])
    ll_F = float((wte * score_sets(Xte, theta_F, pi_F)).sum() / wte.sum())
    print(f"  held-out LL/seq = {ll_F:.3f}")

    results = {}
    for K in Ks:
        print("\n" + "=" * 74)
        print(f"K = {K}")
        print("=" * 74)

        print("  M-A  per-month independent mixtures ...")
        thA, piA = [], []
        for t, (X, w) in enumerate(zip(Xs, ws)):
            th, pi, _ = em_single(X, w, K, seed=args.seed + t)
            thA.append(th); piA.append(pi)

        print("  M-B  shared theta, free pi_t ...")
        theta_B, Pi_B, ll_B = em_shared(Xs, ws, K, seed=args.seed)

        print("  M-C  fitting A on the pi trajectory ...")
        A = fit_A(Pi_B, weights=vol)
        pi_next = Pi_B[-1] @ A

        # ---- (2) held-out likelihood ----
        ll_C = float((wte * score_sets(Xte, theta_B, pi_next)).sum() / wte.sum())
        ll_B_last = float((wte * score_sets(Xte, theta_B, Pi_B[-1])).sum() / wte.sum())
        print(f"\n  (2) HELD-OUT LOG-LIKELIHOOD per sequence, {te_months}")
        print(f"      M-F  frequency (K=1)              {ll_F:>10.3f}")
        print(f"      M-B  mixture, pi = last month     {ll_B_last:>10.3f}")
        print(f"      M-C  mixture, pi = last month @ A {ll_C:>10.3f}")
        print(f"      gain of M-C over M-F              {ll_C - ll_F:>10.3f} nats/seq")

        # ---- (3) appearance ----
        train_sets = set().union(*[set(s) for s in sets_list])
        new_sets = [s for s in sets_te if s not in train_sets]
        rng = np.random.default_rng(args.seed)
        node_freq = (wall[:, None] * Xall).sum(0)
        if len(new_sets) >= 5:
            negs = make_negatives(new_sets, node_freq, V, rng, mult=5)
            Xp, Xn = sets_to_X(new_sets, V), sets_to_X(negs, V)
            from sklearn.metrics import roc_auc_score
            y = np.r_[np.ones(len(new_sets)), np.zeros(len(negs))]
            auc_F = roc_auc_score(y, np.r_[score_sets(Xp, theta_F, pi_F),
                                           score_sets(Xn, theta_F, pi_F)])
            auc_C = roc_auc_score(y, np.r_[score_sets(Xp, theta_B, pi_next),
                                           score_sets(Xn, theta_B, pi_next)])
            print(f"\n  (3) APPEARANCE: {len(new_sets):,} genuinely new sets vs "
                  f"{len(negs):,} hard negatives")
            print(f"      AUC  M-F frequency  {auc_F:.4f}")
            print(f"      AUC  M-C mixture    {auc_C:.4f}")
            print(f"      lift                {auc_C - auc_F:+.4f}")
        else:
            auc_F = auc_C = float("nan")
            print(f"\n  (3) APPEARANCE: only {len(new_sets)} new sets -- skipped")

        # ---- (1) correspondence ----
        if pango:
            print(f"\n  (1) CORRESPONDENCE: inferred block vs Pango, pooled adjacent months")
            rows = eval_correspondence(thA, piA, theta_B, Pi_B, Xs, ws,
                                        sets_list, tr_months, pango, K)
            if rows:
                print(f"      {'months':<18}{'n':>9}{'ARI M-A':>10}{'NMI M-A':>10}"
                      f"{'ARI M-B':>10}{'NMI M-B':>10}")
                for r in rows:
                    print(f"      {r[0]:<18}{r[1]:>9,}{r[2]:>10.4f}{r[3]:>10.4f}"
                          f"{r[4]:>10.4f}{r[5]:>10.4f}")
                a = np.array([[r[2], r[3], r[4], r[5]] for r in rows]).mean(0)
                print(f"      {'MEAN':<18}{'':>9}{a[0]:>10.4f}{a[1]:>10.4f}"
                      f"{a[2]:>10.4f}{a[3]:>10.4f}")
                print(f"\n      -> M-B minus M-A, ARI: {a[2]-a[0]:+.4f}   "
                      f"this gap IS the correspondence result")
        else:
            print("\n  (1) CORRESPONDENCE: skipped (no --pango file)")

        results[f"K{K}"] = dict(theta=theta_B, Pi=Pi_B, A=A, ll_C=ll_C,
                                auc_C=auc_C, auc_F=auc_F)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, theta_F=theta_F, ll_F=ll_F,
             **{f"{k}_{n}": v for k, d in results.items() for n, v in d.items()
                if isinstance(v, np.ndarray)})
    print(f"\nsaved -> {args.out}")

    print("""
HOW TO READ THIS
  (1) is the headline. M-A fits each month separately, so its block labels are
      arbitrary per month; it is given the best possible cross-month label
      matching (Hungarian) and still has to recover identity from set similarity
      alone. M-B shares one theta, so identity holds by construction. If
      ARI(M-B) >> ARI(M-A), temporal coupling recovered lineage identity that
      similarity alone could not. If the gap is ~0, coupling bought nothing and
      the correspondence problem is not solved by this model -- report that.

  (2) tests whether blocks help at all: does K>1 beat K=1 on a future month.
      M-C vs M-B(last month) isolates what A contributes over freezing pi.

  (3) is the target that matters. Negatives are size-matched and drawn
      proportional to node frequency, so a model that knows only marginal
      frequencies scores ~0.5. Any AUC above that is joint structure. The
      relevant number is M-C minus M-F, not M-C alone.

  Run (1) BEFORE trusting (2) or (3). If the blocks are noise, a good AUC is
  not evidence of lineage structure and cannot be interpreted as such.
""")


if __name__ == "__main__":
    main()
