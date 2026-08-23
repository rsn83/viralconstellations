#!/usr/bin/env python3
"""
98_birth_rate_model.py

Fit a birth rate: given the groups circulating at month t, which
(parent group, mutation) pair produces a new group at t+1?

  log lambda(j, n)  =  w . features(j, n, t)

CANDIDATE SPACE
  K groups x V mutations. With K=8, V=1180 that is 9,440 candidates per month
  and roughly 9 real births per month -- a 0.1% positive rate. Report
  precision@k and recall@k, NOT AUC: at this imbalance AUC looks excellent
  while the ranking is useless.

FEATURES  (all computable from the monthly pickles; none use the future)
  parent side
    pi_parent          how prevalent the parent group is this month
    parent_size        expected number of mutations in the parent fingerprint
  mutation side
    log_freq           the mutation's frequency this month, logged
    months_since_seen  how long since it was last observed at all
    hist_peak          the highest monthly frequency it ever reached
    n_groups           how many groups currently carry it above 0.5
    pos_mutability     how many distinct residues have appeared at this position
    ever_seen          whether it has ever been observed
  interaction
    already_in_parent  1 if the parent fingerprint already has it (these are
                       excluded as candidates -- you cannot acquire what you have)
    max_other_group    the highest value any OTHER group assigns it (borrowable)

  months_since_seen, hist_peak and ever_seen test the finite-sites finding
  directly: does a mutation's history predict its return better than its
  current frequency alone?

LABELS
  From script 97's birth_events.tsv. A (parent_group, node) pair at month t is
  positive if a birth event in month t+1 has that node among its added
  mutations and its parent lineage maps to that group.

  Parent lineage -> group is resolved by assigning each lineage to the group
  that best explains its consensus fingerprint under the fitted theta.

Usage:
  python 98_birth_rate_model.py \
      --npz    results/91_exact.npz \
      --vocab  data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --data-dir data/processed/full_data_graphs_posres \
      --events data/processed/birth_events.tsv \
      --months 2021-06:2022-11 [--K 8]
"""
import argparse, pickle, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- io
def ym_add(ym, k):
    y, m = map(int, ym.split("-")); m += k
    y += (m - 1) // 12; m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def months_in_range(spec):
    a, b = spec.split(":") if ":" in spec else (spec, spec)
    out = [a]
    while out[-1] != b:
        out.append(ym_add(out[-1], 1))
        if len(out) > 300: sys.exit("bad range")
    return out


def load_month(data_dir, ym):
    p = Path(data_dir) / f"{ym}_occupied.pkl"
    if not p.exists(): return None
    obj = pickle.load(open(p, "rb"))
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and isinstance(vals[0], (int, np.integer)):
            return [(frozenset(k), int(v)) for k, v in obj.items()]
        return [(frozenset(v), 1) for v in vals]
    items = list(obj)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:
        return [(frozenset(s), int(c)) for s, c in items]
    return [(frozenset(s), 1) for s in items]


def load_vocab(path):
    names, pos, V = {}, {}, 0
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = int(row["node_idx"]); V = max(V, i + 1)
            names[i] = f"{row['aa_pos']}{row['residue'].strip()}"
            pos[i] = int(row["aa_pos"])
    return names, pos, V


def build(records, V):
    w = np.array([c for _, c in records], float)
    X = np.zeros((len(records), V), dtype=np.float32)
    for i, (s, _) in enumerate(records):
        X[i, [n for n in s if 0 <= n < V]] = 1.0
    return X, w


def loglik_matrix(X, theta):
    lt, lc = np.log(theta + EPS), np.log(1 - theta + EPS)
    return X @ (lt - lc).T + lc.sum(1)[None, :]


# ---------------------------------------------------------------- features
def monthly_freq(data_dir, months, V):
    """freq[t, n] = fraction of month t's sequences carrying node n."""
    F = np.zeros((len(months), V))
    for t, ym in enumerate(months):
        r = load_month(data_dir, ym)
        if r is None: continue
        X, w = build(r, V)
        F[t] = (w[:, None] * X).sum(0) / w.sum()
    return F


def make_features(t, F, theta, Pi_t, pos, V, K):
    """Feature matrix for all K x V candidates at month t. Uses only months <= t."""
    hist = F[:t + 1]                                    # (t+1, V)
    cur = hist[-1]
    ever = (hist > 0).any(0)
    peak = hist.max(0)
    # months since last seen (large if never)
    seen_t = np.where(hist > 0, np.arange(t + 1)[:, None], -1).max(0)
    since = np.where(seen_t >= 0, t - seen_t, t + 12).astype(float)
    # distinct residues seen at this position, ever
    by_pos = defaultdict(float)
    for n in range(V):
        if ever[n]: by_pos[pos.get(n, -1)] += 1
    mut_by_pos = np.array([by_pos.get(pos.get(n, -1), 0) for n in range(V)])

    n_groups = (theta > .5).sum(0).astype(float)        # (V,)
    parent_size = theta.sum(1)                          # (K,)
    conv = np.array([1.0 if pos.get(n, -1) in CONVERGENT_ALL else 0.0
                     for n in range(V)])                # (V,) the published list

    rows, meta = [], []
    for j in range(K):
        other = np.delete(theta, j, axis=0).max(0)      # (V,)
        in_par = (theta[j] > .5).astype(float)
        f = np.column_stack([
            np.full(V, Pi_t[j]),                        # pi_parent
            np.full(V, parent_size[j] / 50.0),          # parent_size (scaled)
            np.log(cur + 1e-6),                         # log_freq
            since / 12.0,                               # months_since_seen (years)
            peak,                                       # hist_peak
            n_groups / max(K, 1),                       # n_groups
            mut_by_pos / 20.0,                          # pos_mutability
            ever.astype(float),                         # ever_seen
            other,                                      # max_other_group
            conv,                                       # convergent (published)
        ])
        rows.append(f)
        meta += [(j, n, in_par[n]) for n in range(V)]
    return np.vstack(rows), meta


# Residues repeatedly acquired by unrelated lineages under immune selection.
# Pre-Omicron recurrent set: K417, L452, E484, N501, P681 (Alpha/Beta/Gamma/Delta).
# Omicron-era convergent set: R346, K444, N450, N460, F486, F490, Q493, S494.
# Sources: Focosi & Casadevall, IJMS 2023 ("variant soup"); Focosi et al. 2022.
CONVERGENT_EARLY = {417, 452, 484, 501, 681}
CONVERGENT_OMICRON = {346, 444, 450, 460, 486, 490, 493, 494}
CONVERGENT_ALL = CONVERGENT_EARLY | CONVERGENT_OMICRON

FEATS = ["pi_parent", "parent_size", "log_freq", "months_since_seen",
         "hist_peak", "n_groups", "pos_mutability", "ever_seen",
         "max_other_group", "convergent"]


# ---------------------------------------------------------------- group refit
def _resp(X, theta, log_pi):
    lp = loglik_matrix(X, theta) + log_pi[None, :]
    mx = lp.max(1, keepdims=True)
    P = np.exp(lp - mx)
    return P / P.sum(1, keepdims=True)


def em_pool(Xs, ws, K, iters=200, tol=1e-6, seed=0, prior=.5):
    rng = np.random.default_rng(seed)
    T, V = len(Xs), Xs[0].shape[1]
    mean = np.vstack(Xs).mean(0)
    theta = np.clip(rng.random((K, V)) * .4 + mean[None, :] * .6, .02, .98)
    Pi = np.full((T, K), 1.0 / K); prev = -np.inf
    for _ in range(iters):
        num = np.zeros((K, V)); den = np.zeros(K); N = np.zeros((T, K))
        ll = tot = 0.
        for t, (X, w) in enumerate(zip(Xs, ws)):
            lp = loglik_matrix(X, theta) + np.log(Pi[t] + EPS)[None, :]
            mx = lp.max(1, keepdims=True)
            P = np.exp(lp - mx); Z = P.sum(1, keepdims=True)
            R = P / Z; Rw = R * w[:, None]
            num += Rw.T @ X; den += Rw.sum(0); N[t] = Rw.sum(0)
            ll += float((w * (np.log(Z).ravel() + mx.ravel())).sum()); tot += w.sum()
        theta = np.clip((num + prior) / (den[:, None] + 2 * prior), 1e-4, 1 - 1e-4)
        Pi = N / N.sum(1, keepdims=True)
        ll /= tot
        if abs(ll - prev) < tol: break
        prev = ll
    return theta, Pi


def pi_one(X, w, theta, iters=60, tol=1e-7):
    K = theta.shape[0]
    pi = np.full(K, 1.0 / K); prev = -np.inf
    for _ in range(iters):
        lp = loglik_matrix(X, theta) + np.log(pi + EPS)[None, :]
        mx = lp.max(1, keepdims=True)
        P = np.exp(lp - mx); Z = P.sum(1, keepdims=True)
        R = P / Z
        pi = (R * w[:, None]).sum(0); pi /= pi.sum()
        ll = float((w * (np.log(Z).ravel() + mx.ravel())).sum()) / w.sum()
        if abs(ll - prev) < tol: break
        prev = ll
    return pi


# ---------------------------------------------------------------- model
def fit_logistic(X, y, l2=1.0, iters=400, lr=.5):
    """Plain logistic regression, class-balanced, gradient ascent."""
    n, d = X.shape
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.zeros(d); b = 0.0
    pos_w = (len(y) - y.sum()) / max(y.sum(), 1)        # balance the 0.1% rate
    sw = np.where(y > 0, pos_w, 1.0)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (sw * (y - p))
        w += lr * (Z.T @ g / sw.sum() - l2 * w / len(y))
        b += lr * g.sum() / sw.sum()
    return w, b, mu, sd


def predict(X, w, b, mu, sd):
    return (X - mu) / sd @ w + b


def prec_recall_at_k(scores, y, ks=(10, 50, 100, 500)):
    o = np.argsort(-scores)
    out = []
    for k in ks:
        hit = y[o[:k]].sum()
        out.append((k, hit, hit / k, hit / max(y.sum(), 1)))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--months", required=True, help="e.g. 2021-06:2022-11")
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=.34,
                    help="last fraction of months held out, in time order")
    ap.add_argument("--refit-groups", action="store_true",
                    help="refit theta/pi on the TRAINING months of this window "
                         "instead of reusing the fingerprints in --npz. Use this "
                         "when the npz was fitted on a different period.")
    args = ap.parse_args()

    d = np.load(args.npz)
    if "theta" in d: theta, Pi = d["theta"], d["Pi"]
    else:
        p = f"K{args.K}_"
        if p + "theta" not in d:
            avail = sorted({k.split('_')[0] for k in d.files if k.startswith('K')})
            sys.exit(f"pass --K; available: {avail}")
        theta, Pi = d[p + "theta"], d[p + "Pi"]
    names, pos, V = load_vocab(args.vocab)
    K = theta.shape[0]
    months = months_in_range(args.months)
    print(f"K = {K}   V = {V:,}   months {months[0]}..{months[-1]} ({len(months)})")

    print("\nbuilding monthly frequency table ...", flush=True)
    F = monthly_freq(args.data_dir, months, V)

    cut_pre = int(len(months) * (1 - args.test_frac))
    if args.refit_groups:
        print(f"\nrefitting theta on the TRAINING months only "
              f"({months[0]}..{months[cut_pre-1]}) ...", flush=True)
        Xs, ws = [], []
        for ym in months[:cut_pre]:
            r = load_month(args.data_dir, ym)
            if r is None: continue
            X_, w_ = build(r, V); Xs.append(X_); ws.append(w_)
        theta, Pi_tr = em_pool(Xs, ws, K, seed=0)
        # pi for every month in the window, theta frozen
        Pi = np.zeros((len(months), K))
        for t, ym in enumerate(months):
            r = load_month(args.data_dir, ym)
            if r is None:
                Pi[t] = Pi[t-1] if t else np.full(K, 1.0/K); continue
            X_, w_ = build(r, V)
            Pi[t] = pi_one(X_, w_, theta)
        print(f"  refit done. theta now reflects {months[0]}..{months[cut_pre-1]}"
              f" only; pi re-estimated per month with theta frozen.")

    # ---- load events with their actual added nodes ----
    ev = []
    with open(args.events) as f:
        r = csv.DictReader(f, delimiter="\t")
        if "added_nodes" not in (r.fieldnames or []):
            sys.exit("events file has no 'added_nodes' column -- re-run script 97")
        for row in r:
            if int(row["n_added"]) == 0: continue
            add = [int(x) for x in row["added_nodes"].split(",") if x != ""]
            par = [int(x) for x in row["parent_nodes"].split(",") if x != ""]
            ev.append(dict(month=row["month"], added=add, parent_nodes=par,
                           n_seq=int(row["n_seq"]), child=row["child"]))
    print(f"loaded {len(ev):,} spike-changing birth events")

    # ---- attribute each event's PARENT to a group, by fingerprint match ----
    def parent_group(par_nodes):
        v = np.zeros(V); v[[n for n in par_nodes if n < V]] = 1.0
        return int(loglik_matrix(v[None, :], theta)[0].argmax())

    idx_of = {ym: t for t, ym in enumerate(months)}
    pos_by_t = defaultdict(set)
    used = 0
    for e in ev:
        t = idx_of.get(e["month"])
        if t is None or t == 0: continue
        g = parent_group(e["parent_nodes"])
        for n in e["added"]:
            if n < V: pos_by_t[t - 1].add((g, n)); used += 1
    print(f"  {used:,} (group, mutation) positives placed inside the window")
    if used < 20:
        sys.exit("too few labelled positives inside --months; widen the range")

    # ---- candidate rows ----
    rows_X, rows_y, rows_m = [], [], []
    for t in range(len(months) - 1):
        Xf, meta = make_features(t, F, theta, Pi[min(t, len(Pi) - 1)], pos, V, K)
        P = pos_by_t.get(t, set())
        y = np.array([1.0 if ((j, n) in P) else 0.0 for (j, n, _) in meta])
        keep = np.array([ip == 0 for (_, _, ip) in meta])
        rows_X.append(Xf[keep]); rows_y.append(y[keep])
        rows_m.append(np.full(int(keep.sum()), t))
    X = np.vstack(rows_X); y = np.concatenate(rows_y); mth = np.concatenate(rows_m)
    print(f"candidate rows {len(y):,}   positives {int(y.sum()):,} "
          f"({y.mean():.4%})")

    cut = int(len(months) * (1 - args.test_frac))
    tr, te = mth < cut, mth >= cut
    print(f"train months 0..{cut-1}  ({tr.sum():,} rows, {int(y[tr].sum())} pos)")
    print(f"test  months {cut}..     ({te.sum():,} rows, {int(y[te].sum())} pos)")
    if y[te].sum() < 5: sys.exit("too few test positives")

    w, b, mu, sd = fit_logistic(X[tr], y[tr])
    s_te = predict(X[te], w, b, mu, sd)

    print("\n" + "=" * 66)
    print("FITTED WEIGHTS  (standardised; sign and magnitude are comparable)")
    print("=" * 66)
    for f, wi in sorted(zip(FEATS, w), key=lambda kv: -abs(kv[1])):
        bar = "#" * int(abs(wi) * 20)
        print(f"  {f:<20}{wi:>+8.3f}  {bar}")

    print("\n" + "=" * 66)
    print("HELD-OUT RANKING   (report these, not AUC)")
    print("=" * 66)
    base = y[te].mean()
    print(f"\n  random baseline precision = {base:.4%}")
    print(f"\n  {'k':>6}{'hits':>7}{'precision':>12}{'recall':>10}{'lift':>9}")
    for k, hit, pr, rc in prec_recall_at_k(s_te, y[te]):
        print(f"  {k:>6}{int(hit):>7}{pr:>12.2%}{rc:>10.1%}{pr/base:>8.0f}x")

    # frequency-only baseline
    fi = FEATS.index("log_freq")
    s_freq = X[te][:, fi]
    print(f"\n  frequency-only baseline:")
    for k, hit, pr, rc in prec_recall_at_k(s_freq, y[te]):
        print(f"  {k:>6}{int(hit):>7}{pr:>12.2%}{rc:>10.1%}{pr/base:>8.0f}x")

    # ---- CONTROL: the published convergent-residue list, ALONE ----
    ci_conv = FEATS.index("convergent")
    s_conv = X[te][:, ci_conv] + 1e-6 * np.random.default_rng(0).random(te.sum())
    n_conv = int((X[te][:, ci_conv] > 0).sum())
    print(f"\n  CONTROL -- PUBLISHED CONVERGENT-RESIDUE LIST, ALONE:")
    print(f"  binary: is this mutation at 346/417/444/450/452/460/484/486/490/")
    print(f"  493/494/501/681. No fitting, no data -- just the literature.")
    print(f"  {n_conv:,} of {te.sum():,} test candidates ({n_conv/te.sum():.1%}) "
          f"are at a convergent residue.")
    for k, hit, pr, rc in prec_recall_at_k(s_conv, y[te]):
        print(f"  {k:>6}{int(hit):>7}{pr:>12.2%}{rc:>10.1%}{pr/base:>8.0f}x")

    # ---- and the model WITHOUT it, to see what the model adds ----
    ni = [i for i in range(len(FEATS)) if i != ci_conv]
    wn, bn, mun, sdn = fit_logistic(X[tr][:, ni], y[tr])
    s_noconv = predict(X[te][:, ni], wn, bn, mun, sdn)
    print(f"\n  CONTROL -- full model WITHOUT the convergent feature:")
    for k, hit, pr, rc in prec_recall_at_k(s_noconv, y[te]):
        print(f"  {k:>6}{int(hit):>7}{pr:>12.2%}{rc:>10.1%}{pr/base:>8.0f}x")

    # ---- CONTROL: group-only. No mutation-level information at all. ----
    gi = [FEATS.index(f) for f in ("pi_parent", "parent_size")]
    wg, bg, mug, sdg = fit_logistic(X[tr][:, gi], y[tr])
    s_grp = predict(X[te][:, gi], wg, bg, mug, sdg)
    print(f"\n  CONTROL -- group-only (pi_parent, parent_size):")
    print(f"  these two features are CONSTANT within a group, so this ranks")
    print(f"  groups, not mutations. If it matches the full model, the full")
    print(f"  model is picking groups too.")
    for k, hit, pr, rc in prec_recall_at_k(s_grp, y[te]):
        print(f"  {k:>6}{int(hit):>7}{pr:>12.2%}{rc:>10.1%}{pr/base:>8.0f}x")

    # ---- CONTROL: mutation-only. Everything except the two group features. ----
    mi = [i for i in range(len(FEATS)) if i not in gi]
    wm, bm, mum, sdm = fit_logistic(X[tr][:, mi], y[tr])
    s_mut = predict(X[te][:, mi], wm, bm, mum, sdm)
    print(f"\n  CONTROL -- mutation-only (everything except the group features):")
    for k, hit, pr, rc in prec_recall_at_k(s_mut, y[te]):
        print(f"  {k:>6}{int(hit):>7}{pr:>12.2%}{rc:>10.1%}{pr/base:>8.0f}x")

    # ---- WITHIN-GROUP ranking: the only test that isolates mutation choice ----
    grp_te = np.array([g for g, _, _ in
                       [m for t in range(len(months)-1)
                        for m in make_features(t, F, theta,
                                               Pi[min(t, len(Pi)-1)], pos, V, K)[1]
                        if m[2] == 0]])[te]
    print(f"\n  WITHIN-GROUP ranking -- for each group, rank only its own")
    print(f"  candidates. Group identity carries no information here.")
    print(f"  {'k/group':>9}{'hits':>7}{'precision':>12}{'recall':>10}{'lift':>9}")
    for kk in (5, 10, 25):
        hit = tot = 0
        for g in np.unique(grp_te):
            m_ = grp_te == g
            if m_.sum() < kk: continue
            o = np.argsort(-s_te[m_])[:kk]
            hit += y[te][m_][o].sum(); tot += kk
        if tot:
            pr = hit / tot
            print(f"  {kk:>9}{int(hit):>7}{pr:>12.2%}"
                  f"{hit/max(y[te].sum(),1):>10.1%}{pr/base:>8.0f}x")

    # history features only
    hi = [FEATS.index(f) for f in ("months_since_seen", "hist_peak", "ever_seen")]
    wh, bh, muh, sdh = fit_logistic(X[tr][:, hi], y[tr])
    s_hist = predict(X[te][:, hi], wh, bh, muh, sdh)
    print(f"\n  history-only (months_since_seen, hist_peak, ever_seen):")
    for k, hit, pr, rc in prec_recall_at_k(s_hist, y[te]):
        print(f"  {k:>6}{int(hit):>7}{pr:>12.2%}{rc:>10.1%}{pr/base:>8.0f}x")

    print("""
HOW TO READ
  Precision@k against the random baseline is the number that matters. AUC at a
  0.1% positive rate is misleading -- it can exceed 0.95 while the top 100
  candidates contain nothing.

  THE DECISIVE COMPARISON is the full model against the PUBLISHED CONVERGENT-
  RESIDUE LIST used alone. That list is a fixed set of positions taken from the
  literature -- no fitting, no data. If it matches the full model, this work
  rediscovers something already catalogued and the claim has to change. If the
  full model clearly beats it, the model knows something the list does not:
  which BACKGROUND acquires the mutation, and when.

  Compare also the full model with and without the convergent feature. If
  removing it costs nothing, the model was never using the published knowledge
  and found the signal independently.

  READ THE CONTROLS FIRST. pi_parent and parent_size are constant within a
  group, so a model using only those ranks GROUPS, not mutations. If the
  group-only control matches the full model, the headline lift means "we can
  say which group is active", not "we can say which mutation is acquired" --
  a much weaker claim, and not the project's target. The within-group ranking
  removes group identity entirely and is the honest test of mutation choice.

  The comparison to beat is frequency-only. If the full model does not clearly
  exceed it, the history features are not carrying information and the
  finite-sites finding is descriptive rather than predictive.

  If history-only is close to the full model, then a mutation's past -- when it
  was last seen, how high it ever got -- predicts its return better than its
  current frequency, which is the predictive form of the infinite-to-finite
  sites transition.

  Labels are exact: each positive is a (parent group, mutation) pair taken from
  a real Pango birth event, with the parent lineage assigned to the group whose
  fingerprint best explains it. Only mutations the parent does not already carry
  are scored as candidates.
""")


if __name__ == "__main__":
    main()
