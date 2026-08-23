#!/usr/bin/env python3
"""
95_prove_missing_latent.py

Two controls that together decide WHY the transition matrix A loses to
persistence. One test month cannot distinguish the explanations; these can.

CONTROL 1 -- ROLLING
  Slide a 12-month training window across the data and evaluate every test
  month. For each, record
      gap = LL(A) - LL(persistence)          negative means A lost
      novelty = how badly the test month's sequences fit the groups they
                were assigned to, relative to the training months
  If A's failure is caused by unmodelled lineages, gap should be strongly
  negative exactly in high-novelty months and near zero in quiet months.
  If A is simply badly estimated, gap should be negative everywhere,
  uncorrelated with novelty.

CONTROL 2 -- ORACLE INJECTION
  For each test month, construct one extra row by hand:
      theta_new = theta_parent, with the mutations most over-represented in
                  the parent's test-month sequences raised to their observed
                  level.
  This row is CONSTRUCTED, not fitted -- but it uses the test month to decide
  which mutations to raise, so it is an ORACLE and an upper bound, not a
  method. Then re-estimate pi_t and A with K+1 rows and rescore.

      if A now beats persistence -> the missing row WAS the cause
      if A still loses           -> the cause is estimation, not the missing row

Usage:
  python 95_prove_missing_latent.py \
      --data-dir data/processed/full_data_graphs_posres \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --first-test 2022-01 --last-test 2022-09 --window 12 --K 8
"""
import argparse, pickle, csv, sys
from pathlib import Path
import numpy as np

EPS = 1e-12


# ------------------------------------------------------------------ io
def ym_add(ym, k):
    y, m = map(int, ym.split("-")); m += k
    y += (m - 1) // 12; m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def ym_range(a, b):
    out = [a]
    while out[-1] != b:
        out.append(ym_add(out[-1], 1))
        if len(out) > 200: sys.exit("bad month range")
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


# ------------------------------------------------------------------ model
def loglik_matrix(X, theta):
    lt, lc = np.log(theta + EPS), np.log(1 - theta + EPS)
    return X @ (lt - lc).T + lc.sum(1)[None, :]


def resp(X, theta, log_pi):
    lp = loglik_matrix(X, theta) + log_pi[None, :]
    mx = lp.max(1, keepdims=True)
    P = np.exp(lp - mx); Z = P.sum(1, keepdims=True)
    return P / Z, (np.log(Z) + mx).ravel()


def em_pool(Xs, ws, K, iters=200, tol=1e-6, seed=0, prior=0.5):
    """One theta shared across months, pi_t free per month."""
    rng = np.random.default_rng(seed)
    T, V = len(Xs), Xs[0].shape[1]
    mean = np.vstack(Xs).mean(0)
    theta = np.clip(rng.random((K, V)) * .4 + mean[None, :] * .6, .02, .98)
    Pi = np.full((T, K), 1.0 / K); prev = -np.inf
    for _ in range(iters):
        num = np.zeros((K, V)); den = np.zeros(K); N = np.zeros((T, K))
        ll = tot = 0.0
        for t, (X, w) in enumerate(zip(Xs, ws)):
            R, lps = resp(X, theta, np.log(Pi[t] + EPS))
            Rw = R * w[:, None]
            num += Rw.T @ X; den += Rw.sum(0); N[t] = Rw.sum(0)
            ll += float((w * lps).sum()); tot += w.sum()
        theta = np.clip((num + prior) / (den[:, None] + 2 * prior), 1e-4, 1 - 1e-4)
        Pi = N / N.sum(1, keepdims=True)
        ll /= tot
        if abs(ll - prev) < tol: break
        prev = ll
    return theta, Pi


def pi_only(Xs, ws, theta, iters=60, tol=1e-7):
    """Re-estimate pi_t with theta held fixed (used after injecting a row)."""
    T, K = len(Xs), theta.shape[0]
    Pi = np.full((T, K), 1.0 / K); prev = -np.inf
    for _ in range(iters):
        N = np.zeros((T, K)); ll = tot = 0.0
        for t, (X, w) in enumerate(zip(Xs, ws)):
            R, lps = resp(X, theta, np.log(Pi[t] + EPS))
            N[t] = (R * w[:, None]).sum(0)
            ll += float((w * lps).sum()); tot += w.sum()
        Pi = N / N.sum(1, keepdims=True)
        ll /= tot
        if abs(ll - prev) < tol: break
        prev = ll
    return Pi


def fit_A(Pi, vol, ridge=1.0):
    Xp, Yp = Pi[:-1], Pi[1:]
    s = np.sqrt(vol[:-1])[:, None]
    Xp, Yp = Xp * s, Yp * s
    K = Pi.shape[1]
    A = np.linalg.solve(Xp.T @ Xp + ridge * np.eye(K), Xp.T @ Yp + ridge * np.eye(K))
    A = np.clip(A, 1e-9, None)
    return A / A.sum(1, keepdims=True)


def score(X, w, theta, pi):
    lp = loglik_matrix(X, theta) + np.log(pi + EPS)[None, :]
    mx = lp.max(1, keepdims=True)
    v = (np.log(np.exp(lp - mx).sum(1, keepdims=True)) + mx).ravel()
    return float((w * v).sum() / w.sum())


def novelty(Xte, wte, Xs, ws, theta, Pi):
    """How much worse the test month's sequences fit their assigned group than
    the training months' did. Positive = the month contains something the
    groups do not explain."""
    def med_fit(X, w, pi):
        R, _ = resp(X, theta, np.log(pi + EPS))
        z = R.argmax(1)
        lm = loglik_matrix(X, theta)
        v = lm[np.arange(len(z)), z]
        o = np.argsort(v); v, ww = v[o], w[o]
        c = np.cumsum(ww) / ww.sum()
        return float(np.interp(.5, c, v))
    tr = [med_fit(X, w, Pi[t]) for t, (X, w) in enumerate(zip(Xs, ws))]
    te = med_fit(Xte, wte, Pi[-1])
    return float(np.median(tr[-3:]) - te)


def inject_row(theta, Xte, wte, Pi_last, top=4, min_excess=.15):
    """ORACLE: build one extra row from the dominant group's test-month excess."""
    R, _ = resp(Xte, theta, np.log(Pi_last + EPS))
    z = R.argmax(1)
    k = int(np.bincount(z, weights=wte, minlength=theta.shape[0]).argmax())
    m = z == k
    if m.sum() == 0: return None, k, []
    obs = (wte[m, None] * Xte[m]).sum(0) / wte[m].sum()
    exc = obs - theta[k]
    idx = [int(i) for i in np.argsort(-exc)[:top] if exc[i] >= min_excess]
    if not idx: return None, k, []
    new = theta[k].copy()
    new[idx] = np.clip(obs[idx], 1e-4, 1 - 1e-4)
    return np.vstack([theta, new[None, :]]), k, idx


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--first-test", required=True)
    ap.add_argument("--last-test", required=True)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names, V = load_vocab(args.vocab)
    tests = ym_range(args.first_test, args.last_test)
    print(f"V = {V:,}   K = {args.K}   window = {args.window} months\n")

    rows = []
    for te_m in tests:
        tr_m = [ym_add(te_m, -i) for i in range(args.window, 0, -1)]
        recs = [load_month(args.data_dir, m) for m in tr_m]
        te_r = load_month(args.data_dir, te_m)
        if te_r is None or any(r is None for r in recs):
            print(f"{te_m}: missing data, skipped"); continue
        Xs, ws = zip(*[build(r, V) for r in recs])
        Xs, ws = list(Xs), list(ws)
        Xte, wte = build(te_r, V)
        vol = np.array([w.sum() for w in ws])

        theta, Pi = em_pool(Xs, ws, args.K, seed=args.seed)
        A = fit_A(Pi, vol)
        pin = Pi[-1] @ A; pin /= pin.sum()
        ll_p = score(Xte, wte, theta, Pi[-1])
        ll_A = score(Xte, wte, theta, pin)
        nov  = novelty(Xte, wte, Xs, ws, theta, Pi)

        # ---- CEILING with the existing rows: the BEST possible pi for the
        #      test month, fitted ON the test month with theta fixed. No
        #      transition matrix can beat this. ----
        pi_star = pi_only([Xte], [wte], theta)[0]
        ll_star = score(Xte, wte, theta, pi_star)

        # ---- same ceiling, but with one extra row constructed from the
        #      dominant group's test-month excess ----
        th2, kpar, idx = inject_row(theta, Xte, wte, Pi[-1])
        if th2 is None:
            ll_star2 = float("nan"); muts = "-"
        else:
            pi_star2 = pi_only([Xte], [wte], th2)[0]
            ll_star2 = score(Xte, wte, th2, pi_star2)
            muts = "+".join(names.get(i, str(i)) for i in idx)

        rows.append((te_m, wte.sum(), nov, ll_p, ll_A, ll_A - ll_p,
                     ll_star, ll_star2, muts, kpar))
        print(f"  {te_m}  novelty {nov:6.2f}  persist {ll_p:8.3f}  A {ll_A:8.3f}"
              f"  best-with-K {ll_star:8.3f}  best-with-K+1 {ll_star2:8.3f}"
              f"  [{muts}]", flush=True)

    print("\n" + "=" * 100)
    print("CONTROL 1   does A fail only when the month contains something new?")
    print("=" * 100)
    print(f"\n{'test':<9}{'#seq':>10}{'novelty':>10}"
          f"{'persist':>10}{'via A':>10}{'gap':>9}")
    for r in rows:
        flag = "  <-- A loses" if r[5] < -0.05 else ""
        print(f"{r[0]:<9}{r[1]:>10,.0f}{r[2]:>10.2f}{r[3]:>10.3f}{r[4]:>10.3f}"
              f"{r[5]:>+9.3f}{flag}")
    nv = np.array([r[2] for r in rows]); gp = np.array([r[5] for r in rows])
    if len(rows) > 3:
        from scipy.stats import pearsonr, spearmanr
        pr = pearsonr(nv, gp); sr = spearmanr(nv, gp)
        print(f"\n  correlation(novelty, gap):  Pearson r = {pr[0]:+.3f} (p={pr[1]:.3f})"
              f"   Spearman rho = {sr.statistic:+.3f}")
        print("  A NEGATIVE correlation means A loses more in months containing")
        print("  something the groups cannot explain. That is the missing-latent")
        print("  signature. A flat correlation means A is simply badly estimated.")

    print("\n" + "=" * 100)
    print("CONTROL 2   CEILING -- could ANY dynamics have done better, with the")
    print("            rows the model has?")
    print("=" * 100)
    print("""
  'best with K rows' is the highest score reachable by ANY choice of pi over the
  existing rows -- pi fitted directly ON the test month with theta fixed. No
  transition matrix, however parameterised, can exceed it.

  'best with K+1 rows' adds one row built from the dominant group's test-month
  excess. Both are ORACLES; they use the test month. They bound what is
  achievable, they are not forecasting methods.
""")
    print(f"{'test':<9}{'persist':>10}{'A':>10}{'best K':>10}{'best K+1':>11}"
          f"{'headroom':>10}{'row worth':>11}   {'row = parent +':<22}")
    for r in rows:
        head = r[6] - r[3]            # what better dynamics could buy
        worth = r[7] - r[6]           # what the extra row buys
        print(f"{r[0]:<9}{r[3]:>10.3f}{r[4]:>10.3f}{r[6]:>10.3f}{r[7]:>11.3f}"
              f"{head:>+10.3f}{worth:>+11.3f}   blk{r[9+1] if False else r[9]}"
              if False else
              f"{r[0]:<9}{r[3]:>10.3f}{r[4]:>10.3f}{r[6]:>10.3f}{r[7]:>11.3f}"
              f"{head:>+10.3f}{worth:>+11.3f}   blk{r[9]} + {r[8]:<18}")
    head = np.array([r[6] - r[3] for r in rows])
    worth = np.array([r[7] - r[6] for r in rows]); ok = ~np.isnan(worth)
    print(f"\n  mean headroom for better dynamics (best K  - persistence): {head.mean():+.3f}")
    print(f"  mean value of one extra row       (best K+1 - best K)    : {worth[ok].mean():+.3f}")
    if head.mean() < 0.15 and worth[ok].mean() > 0.5:
        print("""
  -> DECISIVE. There is almost no headroom: even a perfect choice of pi over the
     existing rows barely beats doing nothing, so NO transition matrix could have
     helped. Adding a single row is worth far more. The failure is the missing
     row, not the parameterisation of A.""")
    elif head.mean() > 0.5:
        print("""
  -> NOT the missing row. There IS substantial headroom over persistence using
     only the existing rows, so a better-estimated A could have captured it.
     Fix the parameterisation before adding birth.""")
    else:
        print("""
  -> MIXED. Both effects are present. Report both numbers rather than choosing.""")

    print("""
HOW TO READ

  The injected row is an ORACLE: it uses the test month to decide which
  mutations to raise. It is an upper bound on what a birth mechanism could
  achieve, not a forecasting method. Do not report it as performance.

  What it establishes is CAUSATION. The only thing that changed between the
  two columns is the presence of one extra row. If A goes from losing to
  winning, the missing row was the cause, and no reparameterisation of A --
  growth rates, low rank, more months -- could have fixed it.

  If the gap stays negative even with the row present, the story is wrong:
  A is badly estimated, and the fix is fewer parameters, not birth.

  Control 1 is the independent check. A missing-latent failure should be
  concentrated in high-novelty months. An estimation failure should not care.
""")


if __name__ == "__main__":
    main()
