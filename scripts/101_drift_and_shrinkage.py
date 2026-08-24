#!/usr/bin/env python3
"""
101_drift_and_shrinkage.py

Two changes to the block mixture, tested separately and together so the result
can be attributed.

  DRIFT       theta_k,n(t) = sigmoid(beta_k,n + gamma_k,n * t)
              Backgrounds change over time instead of being fixed. This is
              Weinstein & Marks' move (a time covariate on the emission) applied
              at background level rather than position level. Because theta is a
              FUNCTION of t, a month never fitted can still be evaluated: plug
              in t+1.

  SHRINKAGE   A = (1 - lam) * I + lam * M,  lam fitted
              The transition starts at persistence and has to earn every
              departure from it. At lam = 0 the forecast is exactly "copy last
              month", so this variant can never do worse than persistence by
              construction. M carries the direction of movement.

FOUR FITS
  base          theta fixed,  A free            what we have now
  shrink        theta fixed,  A shrunk
  drift         theta(t),     A free
  both          theta(t),     A shrunk

PREDICTIONS, recorded before running
  1. Shrinkage recovers part of the quiet-month headroom (0.27 available; the
     free A currently loses 0.78) and reduces to persistence elsewhere. It
     should never lose to persistence.
  2. Drift helps where backgrounds accumulate mutations gradually and misses
     BA.5 entirely: 486V sits at 0.012 in every fitted background and reached
     0.538 in one month, which a linear slope in logit space cannot produce.
  3. Neither moves the turnover-month ceiling of 0.02, because that bound is
     over all mixes of whatever backgrounds exist.

  If drift DOES help at BA.5 it contradicts script 96 and is the most
  interesting outcome available here.

Usage:
  python 101_drift_and_shrinkage.py \
      --data-dir data/processed/full_data_graphs_posres \
      --vocab    data/processed/full_data_graphs_posres/posres_vocab.tsv \
      --train 2021-06:2022-05 --test 2022-06 --K 8 --seeds 3
"""
import argparse, pickle, csv, sys
from pathlib import Path
import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- io
def ym_add(ym, k):
    y, m = map(int, ym.split("-")); m += k
    y += (m - 1) // 12; m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def months_in_range(spec):
    if ":" not in spec: return [spec]
    a, b = spec.split(":")
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


def load_V(path):
    n = 0
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n = max(n, int(row["node_idx"]) + 1)
    return n


def load_labels(specs):
    out = []
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path: name, path = Path(spec).stem, spec
        if not Path(path).exists(): continue
        d = {}
        for line in open(path):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                d[frozenset(int(x) for x in p[0].split(",") if x)] = p[1]
        out.append((name, d))
    return out


def build(records, V):
    sets = [s for s, _ in records]
    w = np.array([c for _, c in records], float)
    X = np.zeros((len(sets), V), dtype=np.float32)
    for i, s in enumerate(sets):
        X[i, [n for n in s if 0 <= n < V]] = 1.0
    return X, w, sets


# ---------------------------------------------------------------- core
def loglik_matrix(X, theta):
    lt, lc = np.log(theta + EPS), np.log(1 - theta + EPS)
    return X @ (lt - lc).T + lc.sum(1)[None, :]


def resp(X, theta, log_pi):
    lp = loglik_matrix(X, theta) + log_pi[None, :]
    mx = lp.max(1, keepdims=True)
    P = np.exp(lp - mx); Z = P.sum(1, keepdims=True)
    return P / Z, (np.log(Z) + mx).ravel()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def theta_at(beta, gamma, t, drift):
    """Emission at time t. Without drift, gamma is zero and t is ignored."""
    if not drift:
        return np.clip(sigmoid(beta), 1e-4, 1 - 1e-4)
    return np.clip(sigmoid(beta + gamma * t), 1e-4, 1 - 1e-4)


def em(Xs, ws, K, drift=False, iters=250, tol=1e-6, seed=0, prior=.5,
       inner=12, lr=2.0, verbose=False):
    """EM for the block mixture, with or without drifting emissions.

    Without drift the M-step for theta is closed form. With drift it has no
    closed form (theta is nonlinear in beta, gamma) so it is a few gradient
    steps on the expected complete-data log-likelihood -- generalised EM.
    """
    rng = np.random.default_rng(seed)
    T, V = len(Xs), Xs[0].shape[1]
    mean = np.vstack(Xs).mean(0)
    th0 = np.clip(rng.random((K, V)) * .4 + mean[None, :] * .6, .02, .98)
    beta = np.log(th0 / (1 - th0)); gamma = np.zeros((K, V))
    Pi = np.full((T, K), 1.0 / K)
    tv = (np.arange(T) - (T - 1) / 2.0) / max(T - 1, 1)     # centred, scaled
    prev = -np.inf

    for it in range(iters):
        # ---------- E-step ----------
        Rw_all = []; N = np.zeros((T, K)); ll = tot = 0.0
        for t, (X, w) in enumerate(zip(Xs, ws)):
            th = theta_at(beta, gamma, tv[t], drift)
            R, lps = resp(X, th, np.log(Pi[t] + EPS))
            Rw = R * w[:, None]
            Rw_all.append(Rw); N[t] = Rw.sum(0)
            ll += float((w * lps).sum()); tot += w.sum()
        ll /= tot

        # ---------- M-step: emissions ----------
        if not drift:
            num = np.zeros((K, V)); den = np.zeros(K)
            for t, X in enumerate(Xs):
                num += Rw_all[t].T @ X; den += Rw_all[t].sum(0)
            th = np.clip((num + prior) / (den[:, None] + 2 * prior),
                         1e-4, 1 - 1e-4)
            beta = np.log(th / (1 - th)); gamma[:] = 0.0
        else:
            # S_t and n_t do not depend on beta or gamma, so compute them once
            # per M-step rather than inside the gradient loop.
            S_t = [Rw_all[t].T @ Xs[t] for t in range(T)]        # (K,V) each
            n_t = [Rw_all[t].sum(0)[:, None] for t in range(T)]  # (K,1) each
            tot_n = float(sum(float(n.sum()) for n in n_t))
            for _ in range(inner):
                gb = np.zeros((K, V)); gg = np.zeros((K, V))
                for t in range(T):
                    th = theta_at(beta, gamma, tv[t], True)
                    g = S_t[t] - n_t[t] * th
                    gb += g; gg += g * tv[t]
                beta += lr * gb / (tot_n + EPS)
                gamma += lr * gg / (tot_n + EPS)

        # ---------- M-step: weights ----------
        Pi = N / N.sum(1, keepdims=True)

        if verbose and (it + 1) % 50 == 0:
            print(f"      iter {it+1}  LL/obs {ll:.5f}", flush=True)
        if abs(ll - prev) < tol: break
        prev = ll

    return dict(beta=beta, gamma=gamma, Pi=Pi, tv=tv, ll=ll, drift=drift)


# ---------------------------------------------------------------- transitions
def fit_A_free(Pi, vol, ridge=1e-2):
    Xp, Yp = Pi[:-1], Pi[1:]
    s = np.sqrt(vol[:-1])[:, None]
    Xp, Yp = Xp * s, Yp * s
    K = Pi.shape[1]
    A = np.linalg.solve(Xp.T @ Xp + ridge * np.eye(K), Xp.T @ Yp)
    A = np.clip(A, 1e-9, None)
    return A / A.sum(1, keepdims=True)


def fit_A_shrunk(Pi, vol, grid=None):
    """A = (1-lam) I + lam M.  lam chosen on the training transitions.

    At lam = 0 the forecast is exactly persistence, so this cannot do worse
    than persistence on the quantity it is fitted to.
    """
    K = Pi.shape[1]
    M = fit_A_free(Pi, vol)
    I = np.eye(K)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    best, best_lam = np.inf, 0.0
    w = vol[:-1] / vol[:-1].sum()
    for lam in grid:
        A = (1 - lam) * I + lam * M
        A = A / A.sum(1, keepdims=True)
        err = (w * np.abs(Pi[:-1] @ A - Pi[1:]).sum(1)).sum()
        if err < best: best, best_lam = err, lam
    A = (1 - best_lam) * I + best_lam * M
    return A / A.sum(1, keepdims=True), best_lam


# ---------------------------------------------------------------- scoring
def score(X, w, theta, pi):
    lp = loglik_matrix(X, theta) + np.log(pi + EPS)[None, :]
    mx = lp.max(1, keepdims=True)
    v = (np.log(np.exp(lp - mx).sum(1, keepdims=True)) + mx).ravel()
    return float((w * v).sum() / w.sum())


def best_pi(X, w, theta, iters=80, tol=1e-8):
    """The mix fitted ON the test month -- the highest score any mix can reach
    with these emissions. No forecasting rule can exceed it."""
    K = theta.shape[0]
    pi = np.full(K, 1.0 / K); prev = -np.inf
    for _ in range(iters):
        R, lps = resp(X, theta, np.log(pi + EPS))
        pi = (R * w[:, None]).sum(0); pi /= pi.sum()
        ll = float((w * lps).sum() / w.sum())
        if abs(ll - prev) < tol: break
        prev = ll
    return pi, ll


def ari_of(theta_fn, Pi, Xs, ws, sets_list, labels, tv):
    from sklearn.metrics import adjusted_rand_score
    truth, z = [], []
    for t in range(len(Xs)):
        R, _ = resp(Xs[t], theta_fn(t), np.log(Pi[t] + EPS))
        zz = R.argmax(1)
        for i, s in enumerate(sets_list[t]):
            lin = labels.get(s)
            if lin is None: continue
            n = int(min(ws[t][i], 50))
            truth += [lin] * n; z += [int(zz[i])] * n
    return adjusted_rand_score(truth, z) if truth else float("nan")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--labels", action="append", default=[])
    ap.add_argument("--out", default="results/101_drift.npz")
    args = ap.parse_args()

    V = load_V(args.vocab)
    tr, te = months_in_range(args.train), months_in_range(args.test)
    label_sets = load_labels(args.labels)
    print(f"V = {V:,}   K = {args.K}   train {tr[0]}..{tr[-1]}   test {te}")

    Xs, ws, sets_list = [], [], []
    for ym in tr:
        X, w, s = build(load_month(args.data_dir, ym), V)
        Xs.append(X); ws.append(w); sets_list.append(s)
    T = len(Xs)
    vol = np.array([w.sum() for w in ws])
    print(f"  {vol.sum():,.0f} training sequences over {T} months")

    rec = []
    for ym in te: rec += load_month(args.data_dir, ym)
    Xte, wte, _ = build(rec, V)
    print(f"  {wte.sum():,.0f} test sequences")

    # independence baseline
    Xall = np.vstack(Xs); wall = np.concatenate(ws)
    th_F = np.clip(((wall[:, None] * Xall).sum(0) + .5) / (wall.sum() + 1.),
                   1e-4, 1 - 1e-4)[None, :]
    ll_F = score(Xte, wte, th_F, np.array([1.]))

    variants = [("base",   False, False),
                ("shrink", False, True),
                ("drift",  True,  False),
                ("both",   True,  True)]
    res = {nm: dict(pers=[], fA=[], ceil=[], ari=[], lam=[]) for nm, _, _ in variants}

    for sd in range(args.seeds):
        print(f"\n--- seed {sd} ---", flush=True)
        for nm, drift, shrink in variants:
            f = em(Xs, ws, args.K, drift=drift, seed=sd)
            beta, gamma, Pi, tv = f["beta"], f["gamma"], f["Pi"], f["tv"]
            # emission at the TEST month: one step past the last training month
            dt = (tv[-1] - tv[-2]) if T > 1 else 0.0
            th_te = theta_at(beta, gamma, tv[-1] + dt, drift)

            if shrink:
                A, lam = fit_A_shrunk(Pi, vol)
            else:
                A, lam = fit_A_free(Pi, vol), float("nan")
            pin = Pi[-1] @ A; pin = pin / pin.sum()

            ll_pers = score(Xte, wte, th_te, Pi[-1])
            ll_A    = score(Xte, wte, th_te, pin)
            _, ll_c = best_pi(Xte, wte, th_te)

            res[nm]["pers"].append(ll_pers)
            res[nm]["fA"].append(ll_A)
            res[nm]["ceil"].append(ll_c)
            res[nm]["lam"].append(lam)
            if label_sets:
                res[nm]["ari"].append(
                    ari_of(lambda t: theta_at(beta, gamma, tv[t], drift),
                           Pi, Xs, ws, sets_list, label_sets[0][1], tv))
            print(f"  {nm:<7} persistence {ll_pers:8.3f}   via A {ll_A:8.3f}"
                  f"   ceiling {ll_c:8.3f}"
                  + (f"   lam {lam:.2f}" if shrink else ""), flush=True)

    def ms(v):
        v = [x for x in v if not (isinstance(x, float) and np.isnan(x))]
        return f"{np.mean(v):.3f}+/-{np.std(v):.3f}" if v else "--"

    print("\n" + "=" * 92)
    print(f"RESULTS   K = {args.K}, {args.seeds} seeds, test {te[0]}")
    print("=" * 92)
    print(f"\n  independence baseline (K=1): {ll_F:.3f}\n")
    print(f"  {'variant':<10}{'persistence':>16}{'via A':>16}{'ceiling':>16}"
          f"{'A - pers':>11}{'ceil - pers':>13}")
    for nm, _, shrink in variants:
        r = res[nm]
        d1 = np.mean(r["fA"]) - np.mean(r["pers"])
        d2 = np.mean(r["ceil"]) - np.mean(r["pers"])
        print(f"  {nm:<10}{ms(r['pers']):>16}{ms(r['fA']):>16}{ms(r['ceil']):>16}"
              f"{d1:>+11.3f}{d2:>+13.3f}")

    if label_sets:
        print(f"\n  {'variant':<10}{'ARI vs ' + label_sets[0][0]:>20}")
        for nm, _, _ in variants:
            print(f"  {nm:<10}{ms(res[nm]['ari']):>20}")

    lams = [x for x in res["shrink"]["lam"] if not np.isnan(x)]
    if lams:
        print(f"\n  fitted shrinkage lambda: {np.mean(lams):.3f}"
              f" (0 = pure persistence, 1 = the free transition)")

    b_p, d_p = np.mean(res["base"]["pers"]), np.mean(res["drift"]["pers"])
    print(f"\n  drift over fixed emissions (persistence forecast): "
          f"{d_p - b_p:+.3f}")
    s_a = np.mean(res["shrink"]["fA"]) - np.mean(res["shrink"]["pers"])
    print(f"  shrunk transition over persistence:                  {s_a:+.3f}")

    print("""
HOW TO READ
  'persistence' copies the last training month's mix. 'via A' pushes it through
  the fitted transition. 'ceiling' fits the mix directly on the test month and
  is the highest score any forecast could reach with those emissions -- no
  transition model can exceed it.

  ceiling - persistence is the total amount available to any forecast of the
  mix. Compare it across variants: DRIFT CHANGES THE EMISSIONS, so it moves the
  ceiling. Shrinkage does not -- it only changes how close A gets to it.

  A fitted lambda near 0 means the data does not support movement and the
  transition has collapsed to persistence. That is a cleaner statement of the
  same finding than a transition that simply loses.

  Predictions recorded before running: shrinkage never loses to persistence;
  drift helps where backgrounds accumulate gradually and misses an abrupt
  arrival; neither moves the turnover-month bound.
""")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{f"{nm}_{k}": np.array(v, dtype=float)
                          for nm in res for k, v in res[nm].items()})
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
