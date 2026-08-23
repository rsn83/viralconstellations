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
    """TSV from script 87: 'n1,n2,n3<TAB>lineage<TAB>count<TAB>purity'.

    Returns (label_map, meta_map) where meta_map[set] = (count, purity).
    The purity column is the TRUE ceiling -- it was computed on the raw
    metadata, before collapsing each set to its majority lineage. Recomputing
    purity from the collapsed labels gives 1.0 and is meaningless.
    """
    if not path or not Path(path).exists():
        return None, None
    lab, meta = {}, {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            key = frozenset(int(x) for x in parts[0].split(",") if x != "")
            lab[key] = parts[1]
            if len(parts) >= 4:
                try: meta[key] = (float(parts[2]), float(parts[3]))
                except ValueError: pass
    return lab, meta


def load_label_sets(specs):
    """--labels name=path (repeatable) -> [(name, dict)]"""
    out = []
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).stem, spec
        d, meta = load_pango(path)
        if d is None:
            print(f"  [warn] label file not found, skipping: {path}", file=sys.stderr)
            continue
        if not meta:
            print(f"  [warn] {path} has no purity column -- ceiling unavailable. "
                  f"Regenerate with script 87.", file=sys.stderr)
        out.append((name, d, meta))
    return out


def set_purity(sets_list, ws, labels, meta):
    """CEILING: accuracy of the best classifier that sees only the mutation set.

    Uses the per-set purity computed by script 87 on the RAW metadata, weighted
    by this window's sequence counts. Do NOT recompute from `labels` -- those
    are already majority-collapsed and would give a trivial 1.0.
    """
    if not meta:
        return float("nan"), 0.0
    num = den = 0.0
    for t in range(len(sets_list)):
        for i, s in enumerate(sets_list[t]):
            if s not in labels or s not in meta: continue
            _, pur = meta[s]
            num += pur * ws[t][i]; den += ws[t][i]
    return (num / den if den else float("nan")), den


def block_purity(z, w, truth):
    """ACHIEVED: assign each inferred block its majority label, weighted accuracy.
    Directly comparable to set_purity -- same scale, fewer clusters (K << #sets)."""
    from collections import Counter, defaultdict
    agg = defaultdict(Counter)
    for zi, wi, ti in zip(z, w, truth):
        agg[zi][ti] += wi
    num = den = 0.0
    for c in agg.values():
        num += c.most_common(1)[0][1]; den += sum(c.values())
    return num / den if den else float("nan")


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


def fit_A(Pi, weights=None, ridge=1.0, shrink=0.0):
    """M-C: least-squares A with pi_t A ~ pi_{t+1}, rows on the simplex.
    Weighted by sqrt(monthly sequence count) so thin months count less."""
    Xp, Yp = Pi[:-1], Pi[1:]
    if weights is not None:
        s = np.sqrt(weights[:-1])[:, None]
        Xp, Yp = Xp * s, Yp * s
    K = Pi.shape[1]
    # ridge shrinks A toward 0; adding I to the target shrinks it toward
    # PERSISTENCE (pi_{t+1} = pi_t), which is the honest default when there are
    # only T-1 transitions to fit K^2 parameters.
    A = np.linalg.solve(Xp.T @ Xp + ridge * np.eye(K),
                        Xp.T @ Yp + ridge * np.eye(K))
    A = np.clip(A, 1e-6, None)
    A = A / A.sum(axis=1, keepdims=True)
    if shrink > 0:
        A = (1 - shrink) * A + shrink * np.eye(K)
        A = A / A.sum(axis=1, keepdims=True)
    return A


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


def make_negatives(pos_sets, node_freq, V, rng, mult=5, scheme="perturb",
                   pool=None, observed=None):
    """Negatives for the appearance task.

    'freq'    size-matched, nodes drawn proportional to marginal frequency.
              BROKEN as a control: real new sets are a long backbone plus one
              or two RARE additions, so frequency-drawn sets of the same size
              score HIGHER under independence and the baseline AUC lands below
              0.5. Kept only for reference.

    'perturb' (default) take a real set observed in TRAINING and swap one node
              for another. Same size, same backbone, one mutation different.
              This is the discrimination that matters: given a plausible
              background, which single addition actually happens?

    'swap'    take a real training set and move it to a different backbone by
              exchanging a block of nodes. Harder than 'perturb'.
    """
    live = np.flatnonzero(node_freq > 0)
    p = node_freq[live] / node_freq[live].sum()
    observed = observed or set()
    negs = []

    if scheme == "freq":
        for s in pos_sets:
            k = max(1, len(s))
            for _ in range(mult):
                pick = rng.choice(live, size=min(k, len(live)), replace=False, p=p)
                negs.append(frozenset(int(x) for x in pick))
        return negs

    pool = pool if pool is not None else list(pos_sets)
    for _ in range(mult * len(pos_sets)):
        for _try in range(20):
            base = list(pool[rng.integers(len(pool))])
            if not base: continue
            n_swap = 1 if scheme == "perturb" else max(1, len(base) // 4)
            keep = list(base)
            for _ in range(min(n_swap, len(keep))):
                keep.pop(rng.integers(len(keep)))
            add = []
            while len(add) < n_swap:
                c = int(rng.choice(live, p=p))
                if c not in keep and c not in add: add.append(c)
            cand = frozenset(keep + add)
            if cand not in observed and len(cand) > 0:
                negs.append(cand); break
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
    ap.add_argument("--pango", default="", help="single label TSV (back-compat)")
    ap.add_argument("--labels", action="append", default=[],
                    help="repeatable: name=path, e.g. --labels who=... --labels pango2=...")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1.0,
                    help="ridge on A, shrinking toward persistence (identity)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="restarts for M-B; reports the spread across seeds")
    ap.add_argument("--out", default="results/86_mixture.npz")
    args = ap.parse_args()

    V = load_vocab_size(args.vocab)
    tr_months = months_in_range(args.train)
    te_months = months_in_range(args.test)
    specs = list(args.labels)
    if args.pango: specs.append(f"pango={args.pango}")
    label_sets = load_label_sets(specs)
    print(f"vocab size V = {V:,}   K = {args.K}")
    if label_sets:
        for nm, d, meta in label_sets:
            print(f"  labels '{nm}': {len(d):,} sets"
                  + ("" if meta else "   [no purity column -- ceiling unavailable]"))
    else:
        print("  NO label files -- correspondence test skipped")

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

        print(f"  M-C  fitting A on the pi trajectory "
              f"({Pi_B.shape[0]-1} transitions for {K*K} parameters) ...")
        A = fit_A(Pi_B, weights=vol, ridge=args.ridge)
        pi_next = Pi_B[-1] @ A
        if K * K > 3 * (Pi_B.shape[0] - 1):
            print(f"      [warn] A is underdetermined: {K*K} parameters from "
                  f"{Pi_B.shape[0]-1} transitions. Expect it to lose to persistence.")

        # ---- (2) held-out likelihood ----
        ll_C = float((wte * score_sets(Xte, theta_B, pi_next)).sum() / wte.sum())
        ll_B_last = float((wte * score_sets(Xte, theta_B, Pi_B[-1])).sum() / wte.sum())
        print(f"\n  (2) HELD-OUT LOG-LIKELIHOOD per sequence, {te_months}")
        print(f"      M-F  frequency baseline (K=1)         {ll_F:>10.3f}")
        print(f"      M-B  mixture, pi = last month "
              f"(persistence)                              {ll_B_last:>10.3f}")
        print(f"      M-C  mixture, pi = last month @ A     {ll_C:>10.3f}")
        print(f"      mixture over independence  (M-B - M-F) {ll_B_last - ll_F:>+9.3f}")
        print(f"      chain over persistence     (M-C - M-B) {ll_C - ll_B_last:>+9.3f}"
              f"   <- negative means A loses to doing nothing")

        # ---- (3) appearance ----
        train_sets = set().union(*[set(s) for s in sets_list])
        new_sets = [s for s in sets_te if s not in train_sets]
        rng = np.random.default_rng(args.seed)
        node_freq = (wall[:, None] * Xall).sum(0)
        if len(new_sets) >= 5:
            from sklearn.metrics import roc_auc_score
            all_obs = set(train_sets) | set(sets_te)
            pool = [s for s in train_sets if len(s) > 0]
            print(f"\n  (3) APPEARANCE: {len(new_sets):,} genuinely new sets in "
                  f"{te_months}, ranked against negatives")
            print(f"      {'negatives':<30}{'#neg':>8}{'M-F':>9}{'M-C':>9}{'lift':>9}")
            aucs = {}
            for scheme, desc in [("perturb", "1-node swap on a real set"),
                                  ("swap",    "block swap on a real set"),
                                  ("freq",    "frequency-drawn (reference only)")]:
                negs = make_negatives(new_sets, node_freq, V,
                                      np.random.default_rng(args.seed),
                                      mult=5, scheme=scheme, pool=pool,
                                      observed=all_obs)
                if len(negs) < 10: continue
                Xp, Xn = sets_to_X(new_sets, V), sets_to_X(negs, V)
                y = np.r_[np.ones(len(new_sets)), np.zeros(len(negs))]
                aF = roc_auc_score(y, np.r_[score_sets(Xp, theta_F, pi_F),
                                            score_sets(Xn, theta_F, pi_F)])
                aC = roc_auc_score(y, np.r_[score_sets(Xp, theta_B, pi_next),
                                            score_sets(Xn, theta_B, pi_next)])
                aucs[scheme] = (aF, aC)
                print(f"      {desc:<30}{len(negs):>8,}{aF:>9.4f}{aC:>9.4f}"
                      f"{aC-aF:>+9.4f}")
            auc_F, auc_C = aucs.get("perturb", (float("nan"), float("nan")))
            print("      report the 1-node-swap row: same backbone, one mutation"
                  " different.")
        else:
            auc_F = auc_C = float("nan")
            print(f"\n  (3) APPEARANCE: only {len(new_sets)} new sets -- skipped")

        # ---- (1) correspondence, per label granularity ----
        if label_sets:
            print(f"\n  (1) CORRESPONDENCE, by label granularity")
            print(f"      {'labels':<12}{'n':>10}{'ceiling':>10}{'M-A blk':>10}"
                  f"{'M-B blk':>10}{'M-A ARI':>10}{'M-B ARI':>10}")
            from sklearn.metrics import adjusted_rand_score
            for nm, labels, meta in label_sets:
                ceil, nlab = set_purity(sets_list, ws, labels, meta)
                truth, wt, zA, zB = [], [], [], []
                for t in range(len(tr_months)):
                    RA, _ = responsibilities(Xs[t], thA[t], np.log(piA[t] + EPS))
                    RB, _ = responsibilities(Xs[t], theta_B, np.log(Pi_B[t] + EPS))
                    a, b = RA.argmax(1), RB.argmax(1)
                    for i, s in enumerate(sets_list[t]):
                        lin = labels.get(s)
                        if lin is None: continue
                        # M-A labels are per-month, so make them month-specific:
                        # this is the correspondence problem, not a penalty we impose
                        truth.append(lin); wt.append(ws[t][i])
                        zA.append(f"{t}:{a[i]}"); zB.append(int(b[i]))
                if not truth:
                    print(f"      {nm:<12}{'0':>10}   (no sets matched -- check the join)")
                    continue
                wt = np.array(wt)
                bpA = block_purity(zA, wt, truth)
                bpB = block_purity(zB, wt, truth)
                rep = np.minimum(wt, 50).astype(int)
                tr_ = np.repeat(truth, rep); zA_ = np.repeat(zA, rep); zB_ = np.repeat(zB, rep)
                ariA = adjusted_rand_score(tr_, zA_); ariB = adjusted_rand_score(tr_, zB_)
                print(f"      {nm:<12}{int(wt.sum()):>10,}{ceil:>10.4f}{bpA:>10.4f}"
                      f"{bpB:>10.4f}{ariA:>10.4f}{ariB:>10.4f}")
            print("      ceiling = best possible with one cluster per distinct set")
            print("      blk     = achieved with K blocks (majority label per block)")
            print("      M-A blocks are per-month, so they cannot be reused across months")
        else:
            print("\n  (1) CORRESPONDENCE: skipped (no label files)")

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
