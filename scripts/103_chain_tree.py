#!/usr/bin/env python3
"""
103_chain_tree.py

Three changes asked for, each switchable so the result is attributable.

  --emission {fixed, drift, chain, chain-trend}

      fixed        theta_k                                   (script 101 base)
      drift        theta_k(t) = sigmoid(beta_k + gamma_k t)  (script 101 drift)
      chain        beta_{k,t} = beta_{k,t-1} + noise         random walk
      chain-trend  beta_{k,t} = beta_{k,t-1} + gamma_k + noise

      The chain lets a row's fingerprint bend rather than follow one straight
      line. Note what each forecasts for an unseen month:
          drift        beta_T + gamma          extrapolates the slope
          chain        beta_T                  a random walk's best forecast of
                                               the next value is the last one
          chain-trend  beta_T + gamma          bends AND extrapolates
      So the plain chain is expected to fit the past better and forecast no
      better. chain-trend is the version with both properties.

  --tree
      Rows are seeded as a nested hierarchy: row 0 is the smallest fingerprint,
      each subsequent row is its parent's fingerprint plus further mutations.
      A soft penalty then discourages a child from losing mutations its parent
      has. Without this, rows are seeded randomly and nothing relates them.

  --no-A
      Forecast the mix by persistence instead of a fitted transition. In the
      previous run the fitted shrinkage collapsed to persistence on three of
      four seeds, so this tests whether the transition is needed at all.

REPORTED
  held-out log-likelihood under persistence and under the transition; the
  ceiling (mix fitted on the test month, which no forecast can exceed); and a
  FILL-IN table showing, for each row, when it first held mass and how its
  fingerprint size changed -- which is the question of which rows fill in over
  time.

Usage:
  python 103_chain_tree.py --data-dir ... --vocab ... \
      --train 2021-06:2022-05 --test 2022-06 --K 8 \
      --emission chain-trend --tree --no-A --seeds 3
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
    a, b = spec.split(":"); out = [a]
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


def load_names(path):
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


def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def loglik_matrix(X, th):
    lt, lc = np.log(th + EPS), np.log(1 - th + EPS)
    return X @ (lt - lc).T + lc.sum(1)[None, :]


def resp(X, th, lpi):
    lp = loglik_matrix(X, th) + lpi[None, :]
    mx = lp.max(1, keepdims=True)
    P = np.exp(lp - mx); Z = P.sum(1, keepdims=True)
    return P / Z, (np.log(Z) + mx).ravel()


# ---------------------------------------------------------------- init
def tree_init(Xs, ws, K, rng, V):
    """Nested seeding: row 0 is the smallest fingerprint, each later row is its
    parent's plus further mutations, ordered by observed frequency."""
    Xall = np.vstack(Xs); wall = np.concatenate(ws)
    f = (wall[:, None] * Xall).sum(0) / wall.sum()          # marginal rates
    order = np.argsort(-f)                                   # commonest first
    sizes = np.linspace(max(3, int(0.05 * (f > .05).sum())),
                        max(6, int((f > .05).sum())), K).astype(int)
    th = np.full((K, V), .02)
    parent = np.full(K, -1)
    for k in range(K):
        take = order[:sizes[k]]
        th[k, take] = np.clip(f[take] * .9 + .1, .05, .98)
        # jitter so rows are not identical
        th[k] = np.clip(th[k] + rng.normal(0, .02, V), .02, .98)
        parent[k] = k - 1
    return th, parent


def random_init(Xs, ws, K, rng, V):
    mean = np.vstack(Xs).mean(0)
    th = np.clip(rng.random((K, V)) * .4 + mean[None, :] * .6, .02, .98)
    return th, np.full(K, -1)


# ---------------------------------------------------------------- EM
def em(Xs, ws, K, emission="fixed", tree=False, seed=0, iters=200, inner=12,
       lr=2.0, sigma=1.0, tree_w=0.0, tol=1e-6, prior=.5, init_beta=None,
       verbose=False, horizon=1, half_life=0.0):
    rng = np.random.default_rng(seed)
    T, V = len(Xs), Xs[0].shape[1]
    th0, parent = (tree_init if tree else random_init)(Xs, ws, K, rng, V)
    per_month = emission in ("chain", "chain-trend")

    if init_beta is not None:
        b0 = init_beta.copy()
    else:
        b0 = np.log(th0 / (1 - th0))
    B = np.repeat(b0[None, :, :], T, axis=0) if per_month else b0.copy()
    gamma = np.zeros((K, V))
    tv = (np.arange(T) - (T - 1) / 2.0) / max(T - 1, 1)
    Pi = np.full((T, K), 1.0 / K)
    prev = -np.inf
    # recency weights for the SLOPE only. A mutation can sit at zero for ten
    # months and rise in the last two; a slope fitted across the whole window
    # is dominated by the zeros and comes out flat. Weighting recent months
    # lets the recent rise set the rate. half_life = 0 disables it.
    if half_life > 0:
        age = np.arange(T)[::-1].astype(float)          # 0 = most recent
        rw = 0.5 ** (age / half_life)
    else:
        rw = np.ones(T)
    rw = rw / rw.mean()

    def theta_t(t):
        if per_month: return np.clip(sig(B[t]), 1e-4, 1 - 1e-4)
        if emission == "drift":
            return np.clip(sig(B + gamma * tv[t]), 1e-4, 1 - 1e-4)
        return np.clip(sig(B), 1e-4, 1 - 1e-4)

    for it in range(iters):
        S_t, n_t = [], []; N = np.zeros((T, K)); ll = tot = 0.0
        for t, (X, w) in enumerate(zip(Xs, ws)):
            R, lps = resp(X, theta_t(t), np.log(Pi[t] + EPS))
            Rw = R * w[:, None]
            S_t.append(Rw.T @ X); n_t.append(Rw.sum(0)[:, None])
            N[t] = Rw.sum(0); ll += float((w * lps).sum()); tot += w.sum()
        ll /= tot

        if emission == "fixed":
            num = sum(S_t); den = sum(n.ravel() for n in n_t)
            th = np.clip((num + prior) / (den[:, None] + 2 * prior),
                         1e-4, 1 - 1e-4)
            B = np.log(th / (1 - th))
        elif emission == "drift":
            tn = float(sum(float(n.sum()) for n in n_t))
            for _ in range(inner):
                gb = np.zeros((K, V)); gg = np.zeros((K, V))
                for t in range(T):
                    g = S_t[t] - n_t[t] * np.clip(sig(B + gamma * tv[t]),
                                                  1e-4, 1 - 1e-4)
                    gb += g                    # level: all months equally
                    gg += rw[t] * g * tv[t]    # slope: recent months weighted
                B += lr * gb / (tn + EPS); gamma += lr * gg / (tn + EPS)
        else:                                    # chain / chain-trend
            tn = float(sum(float(n.sum()) for n in n_t))
            for _ in range(inner):
                G = np.zeros_like(B)
                for t in range(T):
                    G[t] = S_t[t] - n_t[t] * np.clip(sig(B[t]), 1e-4, 1 - 1e-4)
                # smoothness: penalise deviation of the step from gamma
                step = np.zeros_like(B)
                for t in range(1, T):
                    d = B[t] - B[t - 1] - (gamma if emission == "chain-trend"
                                           else 0.0)
                    step[t] -= d / sigma ** 2
                    step[t - 1] += d / sigma ** 2
                if tree_w > 0:
                    for k in range(1, K):
                        viol = np.maximum(B[:, k - 1] - B[:, k], 0.0)
                        G[:, k] += tree_w * viol
                        G[:, k - 1] -= tree_w * viol
                B += lr * (G / (tn + EPS) + step / max(T, 1))
                if emission == "chain-trend":
                    d = np.diff(B, axis=0).mean(0) if T > 1 else np.zeros((K, V))
                    gamma += .5 * (d - gamma)

        Pi = N / N.sum(1, keepdims=True)
        if verbose and (it + 1) % 50 == 0:
            print(f"      iter {it+1}  LL/obs {ll:.5f}", flush=True)
        if abs(ll - prev) < tol: break
        prev = ll

    # emission for one month past the last training month
    if per_month:
        th_next = np.clip(sig(B[-1] + horizon * (gamma if emission == "chain-trend"
                                                 else 0.0)), 1e-4, 1 - 1e-4)
        th_by_t = [np.clip(sig(B[t]), 1e-4, 1 - 1e-4) for t in range(T)]
    elif emission == "drift":
        dt = tv[-1] - tv[-2] if T > 1 else 0.0
        th_next = np.clip(sig(B + gamma * (tv[-1] + horizon * dt)),
                          1e-4, 1 - 1e-4)
        th_by_t = [np.clip(sig(B + gamma * tv[t]), 1e-4, 1 - 1e-4)
                   for t in range(T)]
    else:
        th_next = np.clip(sig(B), 1e-4, 1 - 1e-4)
        th_by_t = [th_next] * T
    return dict(B=B, gamma=gamma, Pi=Pi, ll=ll, th_next=th_next,
                th_by_t=th_by_t, parent=parent)


# ---------------------------------------------------------------- transitions
def fit_A(Pi, vol, shrink=True, ridge=1e-2):
    Xp, Yp = Pi[:-1], Pi[1:]
    s = np.sqrt(vol[:-1])[:, None]
    K = Pi.shape[1]
    M = np.linalg.solve((Xp * s).T @ (Xp * s) + ridge * np.eye(K),
                        (Xp * s).T @ (Yp * s))
    M = np.clip(M, 1e-9, None); M /= M.sum(1, keepdims=True)
    if not shrink: return M, 1.0
    I = np.eye(K); w = vol[:-1] / vol[:-1].sum()
    best, lam_b = np.inf, 0.0
    for lam in np.linspace(0, 1, 21):
        A = (1 - lam) * I + lam * M; A /= A.sum(1, keepdims=True)
        e = (w * np.abs(Pi[:-1] @ A - Pi[1:]).sum(1)).sum()
        if e < best: best, lam_b = e, lam
    A = (1 - lam_b) * I + lam_b * M
    return A / A.sum(1, keepdims=True), lam_b


def score(X, w, th, pi):
    lp = loglik_matrix(X, th) + np.log(pi + EPS)[None, :]
    mx = lp.max(1, keepdims=True)
    v = (np.log(np.exp(lp - mx).sum(1, keepdims=True)) + mx).ravel()
    return float((w * v).sum() / w.sum())


def best_pi(X, w, th, iters=80, tol=1e-8):
    K = th.shape[0]; pi = np.full(K, 1.0 / K); prev = -np.inf
    for _ in range(iters):
        R, lps = resp(X, th, np.log(pi + EPS))
        pi = (R * w[:, None]).sum(0); pi /= pi.sum()
        ll = float((w * lps).sum() / w.sum())
        if abs(ll - prev) < tol: break
        prev = ll
    return ll


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--emission", default="drift",
                    choices=["fixed", "drift", "chain", "chain-trend", "all"])
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--no-A", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="chain smoothness; small = stiff, large = free")
    ap.add_argument("--tree-w", type=float, default=0.0,
                    help="penalty for a child losing its parent's mutations")
    ap.add_argument("--half-life", type=float, default=0.0,
                    help="months; the slope is fitted with weights decaying by "
                         "half every this many months. 0 = no recency "
                         "weighting. 2 makes the last 2-3 months dominate.")
    ap.add_argument("--horizon", type=int, default=1,
                    help="forecast h months ahead: training ends h months "
                         "before the test month, emissions are extrapolated h "
                         "steps, and persistence copies the last TRAINING month")
    args = ap.parse_args()

    names, V = load_names(args.vocab)
    te = months_in_range(args.test)
    h = args.horizon
    tr = months_in_range(args.train)
    if h > 1:
        tr = tr[:-(h - 1)]        # drop the h-1 months closest to the test
    Xs, ws = [], []
    for ym in tr:
        X, w = build(load_month(args.data_dir, ym), V)
        Xs.append(X); ws.append(w)
    T = len(Xs); vol = np.array([w.sum() for w in ws])
    rec = []
    for ym in te: rec += load_month(args.data_dir, ym)
    Xte, wte = build(rec, V)
    print(f"V={V:,} K={args.K} train {tr[0]}..{tr[-1]} test {te[0]}  "
          f"h={h}  tree={args.tree}  no-A={args.no_A}"
          + (f"  half-life={args.half_life}mo" if args.half_life > 0 else ""))
    if h > 1:
        print(f"  training ends {h} months before the test month; persistence "
              f"copies {tr[-1]}")

    modes = (["fixed", "drift", "chain", "chain-trend"]
             if args.emission == "all" else [args.emission])
    res = {m: dict(p=[], a=[], c=[], lam=[]) for m in modes}
    keep = None

    for sd in range(args.seeds):
        print(f"\n--- seed {sd} ---", flush=True)
        base = em(Xs, ws, args.K, "fixed", tree=args.tree, seed=sd,
                  horizon=h)
        for m in modes:
            f = (base if m == "fixed"
                 else em(Xs, ws, args.K, m, tree=args.tree, seed=sd,
                         init_beta=base["B"], sigma=args.sigma,
                         tree_w=args.tree_w, horizon=h,
                         half_life=args.half_life))
            if args.no_A:
                A, lam = np.eye(args.K), 0.0
            else:
                A, lam = fit_A(f["Pi"], vol)
            pin = f["Pi"][-1].copy()
            for _ in range(h): pin = pin @ A
            pin /= pin.sum()
            p = score(Xte, wte, f["th_next"], f["Pi"][-1])
            a = score(Xte, wte, f["th_next"], pin)
            c = best_pi(Xte, wte, f["th_next"])
            res[m]["p"].append(p); res[m]["a"].append(a)
            res[m]["c"].append(c); res[m]["lam"].append(lam)
            print(f"  {m:<12} persistence {p:8.3f}  via A {a:8.3f}  "
                  f"ceiling {c:8.3f}" + ("" if args.no_A else f"  lam {lam:.2f}"),
                  flush=True)
            if sd == 0 and m == modes[-1]: keep = f

    def ms(v): return f"{np.mean(v):.3f}+/-{np.std(v):.3f}"
    print("\n" + "=" * 84)
    print(f"RESULTS  K={args.K}  {args.seeds} seeds  test {te[0]}")
    print("=" * 84)
    print(f"\n  {'emission':<13}{'persistence':>16}{'via A':>16}{'ceiling':>16}"
          f"{'ceil-pers':>12}")
    for m in modes:
        r = res[m]
        print(f"  {m:<13}{ms(r['p']):>16}{ms(r['a']):>16}{ms(r['c']):>16}"
              f"{np.mean(r['c'])-np.mean(r['p']):>+12.3f}")
    if "fixed" in res and len(modes) > 1:
        b = np.mean(res["fixed"]["p"])
        print()
        for m in modes:
            if m == "fixed": continue
            print(f"  {m} over fixed: {np.mean(res[m]['p']) - b:+.3f}")

    # ---- which rows fill in over time ----
    if keep is not None:
        Pi = keep["Pi"]; tb = keep["th_by_t"]
        print("\n" + "=" * 84)
        print("WHICH ROWS FILL IN")
        print("=" * 84)
        print(f"\n  {'row':<6}{'first month above 1%':>22}{'share (last)':>14}"
              f"{'size first':>12}{'size last':>11}{'change':>9}")
        for k in np.argsort(-Pi[-1]):
            above = np.flatnonzero(Pi[:, k] > .01)
            first = tr[above[0]] if len(above) else "never"
            s0, s1 = tb[0][k].sum(), tb[-1][k].sum()
            print(f"  blk{k:<3}{first:>22}{Pi[-1, k]:>14.3f}{s0:>12.1f}"
                  f"{s1:>11.1f}{s1-s0:>+9.1f}")
        obs0 = float((ws[0][:, None] * Xs[0]).sum() / ws[0].sum())
        obs1 = float((ws[-1][:, None] * Xs[-1]).sum() / ws[-1].sum())
        w0 = float((Pi[0] * np.array([t.sum() for t in [tb[0][k] for k in range(args.K)]])).sum())
        w1 = float((Pi[-1] * np.array([t.sum() for t in [tb[-1][k] for k in range(args.K)]])).sum())
        print(f"\n  observed mutations per sequence: {obs0:.1f} -> {obs1:.1f}")
        print(f"  model expected size (weighted):  {w0:.1f} -> {w1:.1f}")

    print("""
HOW TO READ
  'ceiling' is the mix fitted on the test month -- the highest score any
  forecast could reach with those emissions. Changing the EMISSION moves it;
  changing the transition cannot.

  A plain chain forecasts the next fingerprint as the last one, so it should
  fit the past better and forecast no better than fixed emissions. chain-trend
  keeps a slope to extrapolate. If the plain chain wins anyway, that prediction
  was wrong and worth knowing.

  With --no-A the forecast is persistence, so the 'via A' column equals
  'persistence' by construction.
""")


if __name__ == "__main__":
    main()
