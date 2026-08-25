#!/usr/bin/env python3
"""
109_split_merge.py

Two additions to the block mixture, switchable independently.

  --split-merge
      At K=48, 29 of 48 rows never left their initialisation value of 0.5 and
      one block held 92% of sequences. That is not a missing mechanism -- the
      rows exist. A row at 0.5 explains nothing, so it receives no
      responsibility, so its gradient is zero, so EM never touches it. It is a
      fixed point the algorithm cannot leave.

      Periodically: take the block with the most internal dependence, split it
      in two, and give one half to a dead row. The split direction is the
      mutation with the largest within-block residual -- for the block holding
      BA.2 and BA.5 together that is 486V, whose residual was +1026.

      This creates nothing. It uses rows already allocated.

  --pi-mode growth
      pi_{t,k} proportional to exp(alpha_k + r_k t), fitted jointly, 2K
      parameters. The mixture analogue of drift on theta.

      The ceiling says this cannot win much: at K=48 the best possible mixture
      is only 0.019 above copying last month. Running it makes that bound
      concrete rather than theoretical, and r_k is the growth advantage per
      block per month -- checkable against published Delta and Omicron
      estimates, which is external validation available nowhere else here.

Usage:
  python 109_split_merge.py \
      --data-dir data/processed/full_data_graphs_withdel \
      --vocab    data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
      --train 2021-06:2022-05 --test 2022-06 --K 48 \
      --split-merge --pi-mode growth --seeds 3
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


def build(records, V, min_count=1):
    if min_count > 1:
        records = [(s_, c) for s_, c in records if c >= min_count]
    w = np.array([c for _, c in records], float)
    X = np.zeros((len(records), V), dtype=np.float32)
    for i, (s, _) in enumerate(records):
        X[i, [n for n in s if 0 <= n < V]] = 1.0
    return X, w


def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def loglik_matrix(X, th):
    lt, lc = np.log(th + EPS), np.log(1 - th + EPS)
    return X @ (lt - lc).T + lc.sum(1)[None, :]


def softmax_rows(M):
    M = M - M.max(-1, keepdims=True)
    E = np.exp(M)
    return E / E.sum(-1, keepdims=True)


# ---------------------------------------------------------------- split
def worst_block(Xs, ws, theta, Pi, dead_thresh=1e-4):
    """The occupied block with the largest within-block dependence, and the
    mutation to split it on.

    Dependence is measured as in script 108: observed co-occurrence against
    what independence-given-block predicts, using the block's own rates. The
    split mutation is the one contributing the most.
    """
    K, V = theta.shape
    share = Pi.mean(0)
    best = (-1.0, -1, -1)
    # only the largest occupied blocks are worth scanning: a block holding 1%
    # of sequences cannot be hiding a lineage worth splitting out
    cand = [k for k in np.argsort(-share)[:6] if share[k] >= .01]
    for k in cand:
        num = np.zeros(V); den = 0.0
        Co = None
        for t, (X, w) in enumerate(zip(Xs, ws)):
            lp = loglik_matrix(X, theta) + np.log(Pi[t] + EPS)[None, :]
            m = lp.argmax(1) == k
            if not m.any(): continue
            Xk, wk = X[m], w[m]
            num += (wk[:, None] * Xk).sum(0); den += wk.sum()
            C = (Xk * wk[:, None]).T @ Xk
            Co = C if Co is None else Co + C
        if den < 100 or Co is None: continue
        p = num / den
        var = np.flatnonzero((p > .05) & (p < .95))
        if len(var) < 2: continue
        pv = p[var]
        Exp = den * np.outer(pv, pv)
        R = (Co[np.ix_(var, var)] - Exp) / np.sqrt(Exp + 1.0)
        np.fill_diagonal(R, 0.0)
        # the mutation whose residuals are largest overall
        score = np.abs(R).sum(1)
        j = int(np.argmax(score))
        if score[j] > best[0]:
            best = (float(score[j]), k, int(var[j]))
    return best


def do_split(theta, Pi, Xs, ws, k, n, dead):
    """Split block k on mutation n: sequences with n go to a dead row."""
    K, V = theta.shape
    num_a = np.zeros(V); den_a = 0.0
    num_b = np.zeros(V); den_b = 0.0
    mass_a = np.zeros(len(Xs)); mass_b = np.zeros(len(Xs))
    for t, (X, w) in enumerate(zip(Xs, ws)):
        lp = loglik_matrix(X, theta) + np.log(Pi[t] + EPS)[None, :]
        m = lp.argmax(1) == k
        if not m.any(): continue
        has = X[:, n] > 0
        A = m & has; B = m & ~has
        if A.any():
            num_a += (w[A, None] * X[A]).sum(0); den_a += w[A].sum()
            mass_a[t] = w[A].sum() / w.sum()
        if B.any():
            num_b += (w[B, None] * X[B]).sum(0); den_b += w[B].sum()
            mass_b[t] = w[B].sum() / w.sum()
    if den_a < 50 or den_b < 50: return theta, Pi, False
    theta = theta.copy(); Pi = Pi.copy()
    theta[dead] = np.clip((num_a + .5) / (den_a + 1.0), 1e-4, 1 - 1e-4)
    theta[k] = np.clip((num_b + .5) / (den_b + 1.0), 1e-4, 1 - 1e-4)
    Pi[:, dead] = mass_a
    Pi[:, k] = mass_b
    Pi = Pi / Pi.sum(1, keepdims=True)
    return theta, Pi, True


# ---------------------------------------------------------------- EM
def em(Xs, ws, K, drift=True, pi_mode="free", split_merge=False, seed=0,
       iters=250, inner=12, lr=2.0, half_life=1.0, tol=1e-6, prior=.5,
       names=None, verbose=True, init=None, track_tree=False):
    rng = np.random.default_rng(seed)
    T, V = len(Xs), Xs[0].shape[1]
    if init is not None:
        # Fitting beta and gamma together from a random start collapses every
        # row onto the pooled mean -- the baseline came out with 1 of 48 rows
        # used. Warm-starting from a converged fixed-emission fit means beta
        # already separates the rows and only the slopes have to be learned.
        beta = init["beta"].copy(); Pi = init["Pi"].copy()
    else:
        mean = np.vstack(Xs).mean(0)
        th0 = np.clip(rng.random((K, V)) * .4 + mean[None, :] * .6, .02, .98)
        beta = np.log(th0 / (1 - th0))
        Pi = np.full((T, K), 1.0 / K)
    gamma = np.zeros((K, V))
    parent = np.full(K, -1)          # who each row was split from, -1 = root
    alpha = np.zeros(K); r = np.zeros(K)
    tv = (np.arange(T) - (T - 1) / 2.) / max(T - 1, 1)
    rw = (0.5 ** (np.arange(T)[::-1] / half_life)) if half_life > 0 else np.ones(T)
    rw /= rw.mean()
    prev = -np.inf
    splits = []

    def theta_t(t):
        return np.clip(sig(beta + gamma * tv[t]) if drift else sig(beta),
                       1e-4, 1 - 1e-4)

    for it in range(iters):
        S_t, n_t = [], []; N = np.zeros((T, K)); ll = tot = 0.
        for t, (X, w) in enumerate(zip(Xs, ws)):
            lp = loglik_matrix(X, theta_t(t)) + np.log(Pi[t] + EPS)[None, :]
            mx = lp.max(1, keepdims=True)
            P = np.exp(lp - mx); Z = P.sum(1, keepdims=True)
            R = P / Z; Rw = R * w[:, None]
            S_t.append(Rw.T @ X); n_t.append(Rw.sum(0)[:, None]); N[t] = Rw.sum(0)
            ll += float((w * (np.log(Z).ravel() + mx.ravel())).sum()); tot += w.sum()
        ll /= tot

        # ---- emissions ----
        if not drift:
            num = sum(S_t); den = sum(x.ravel() for x in n_t)
            th = np.clip((num + prior) / (den[:, None] + 2 * prior), 1e-4, 1 - 1e-4)
            beta = np.log(th / (1 - th))
        else:
            tn = float(sum(float(x.sum()) for x in n_t))
            for _ in range(inner):
                gb = np.zeros((K, V)); gg = np.zeros((K, V))
                for t in range(T):
                    g = S_t[t] - n_t[t] * np.clip(sig(beta + gamma * tv[t]),
                                                  1e-4, 1 - 1e-4)
                    gb += g; gg += rw[t] * g * tv[t]
                beta += lr * gb / (tn + EPS); gamma += lr * gg / (tn + EPS)

        # ---- mixture weights ----
        if pi_mode == "free":
            Pi = N / N.sum(1, keepdims=True)
        else:                                   # growth
            Nt = N.sum(1, keepdims=True)
            for _ in range(40):
                P = softmax_rows(alpha[None, :] + r[None, :] * tv[:, None])
                G = N - Nt * P
                alpha += .5 * G.sum(0) / N.sum()
                r += .5 * (G * tv[:, None]).sum(0) / N.sum()
                alpha -= alpha.mean(); r -= r.mean()
            Pi = softmax_rows(alpha[None, :] + r[None, :] * tv[:, None])

        # ---- split-merge ----
        if split_merge and it in (30, 60, 90, 120, 150, 180):
            th_now = theta_t(T - 1)
            dead = [k for k in range(K) if Pi[:, k].max() < 1e-4]
            if dead:
                score, k, n = worst_block(Xs, ws, th_now, Pi)
                if k >= 0:
                    th2, Pi2, ok = do_split(th_now, Pi, Xs, ws, k, n, dead[0])
                    if ok:
                        beta = np.log(th2 / (1 - th2))
                        gamma[dead[0]] = 0.0
                        Pi = Pi2
                        nm = names.get(n, str(n)) if names else str(n)
                        splits.append((it, k, dead[0], nm, score))
                        parent[dead[0]] = k
                        if verbose:
                            print(f"      split blk{k} on {nm} -> blk{dead[0]}"
                                  f"   (residual score {score:.0f})", flush=True)

        if verbose and (it + 1) % 50 == 0:
            print(f"      iter {it+1}  LL/obs {ll:.5f}", flush=True)
        if abs(ll - prev) < tol and it > 170: break
        prev = ll

    dt = tv[-1] - tv[-2] if T > 1 else 0.
    th_next = np.clip(sig(beta + gamma * (tv[-1] + dt)) if drift else sig(beta),
                      1e-4, 1 - 1e-4)
    if pi_mode == "growth":
        pi_next = softmax_rows(alpha[None, :] + r[None, :] *
                               np.array([[tv[-1] + dt]]))[0]
    else:
        pi_next = Pi[-1]
    return dict(beta=beta, gamma=gamma, Pi=Pi, th_next=th_next,
                pi_next=pi_next, r=r, ll=ll, splits=splits, parent=parent)


def score(X, w, th, pi):
    lp = loglik_matrix(X, th) + np.log(pi + EPS)[None, :]
    mx = lp.max(1, keepdims=True)
    v = (np.log(np.exp(lp - mx).sum(1, keepdims=True)) + mx).ravel()
    return float((w * v).sum() / w.sum())


def best_pi(X, w, th, iters=80, tol=1e-8):
    K = th.shape[0]; pi = np.full(K, 1. / K); prev = -np.inf
    for _ in range(iters):
        lp = loglik_matrix(X, th) + np.log(pi + EPS)[None, :]
        mx = lp.max(1, keepdims=True)
        P = np.exp(lp - mx); Z = P.sum(1, keepdims=True)
        pi = ((P / Z) * w[:, None]).sum(0); pi /= pi.sum()
        ll = float((w * (np.log(Z).ravel() + mx.ravel())).sum() / w.sum())
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
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--half-life", type=float, default=1.0)
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop sets seen fewer than this many times. The row "
                         "count drives runtime, not the sequence count, and "
                         "the set-size distribution is long-tailed, so this "
                         "cuts rows sharply for little lost mass. Reports what "
                         "is lost.")
    ap.add_argument("--best-of", type=int, default=1,
                    help="fit this many restarts per variant and keep the one "
                         "with the best TRAINING likelihood. Held-out scores "
                         "are never used to choose. Seed-to-seed spread is 5.3 "
                         "nats here, larger than any mechanism tested, so a "
                         "single fit is not a measurement.")
    args = ap.parse_args()

    names, V = load_names(args.vocab)
    tr, te = months_in_range(args.train), months_in_range(args.test)
    Xs, ws = [], []
    kept = dropped = kept_seq = all_seq = 0
    for ym in tr:
        r_ = load_month(args.data_dir, ym)
        all_seq += sum(c for _, c in r_); dropped += len(r_)
        X, w = build(r_, V, args.min_count)
        kept += len(X); kept_seq += w.sum()
        Xs.append(X); ws.append(w)
    rec = []
    for ym in te: rec += load_month(args.data_dir, ym)
    Xte, wte = build(rec, V)          # test month never filtered
    if args.min_count > 1:
        print(f"  min-count {args.min_count}: kept {kept:,} of {dropped:,} rows "
              f"({kept/dropped:.1%}), {kept_seq/all_seq:.2%} of sequences")
    print(f"V={V:,} K={args.K} train {tr[0]}..{tr[-1]} test {te[0]}  "
          f"drift={not args.no_drift}  half-life={args.half_life}")

    variants = [("baseline",      False, "free"),
                ("split-merge",   True,  "free"),
                ("growth pi",     False, "growth"),
                ("both",          True,  "growth")]
    res = {n: dict(p=[], c=[], used=[], r=[]) for n, _, _ in variants}
    last_tree = None

    for sd in range(args.seeds):
        print(f"\n--- seed {sd} ---", flush=True)
        warm = em(Xs, ws, args.K, drift=False, pi_mode="free",
                  split_merge=False, seed=sd, iters=120, verbose=False)
        print(f"  warm start (fixed emissions): rows used "
              f"{int((warm['Pi'].max(0) > 1e-3).sum())}/{args.K}", flush=True)
        for nm, sm, pm in variants:
            best_f, best_ll = None, -np.inf
            for rs in range(args.best_of):
                w2 = (warm if rs == 0 else
                      em(Xs, ws, args.K, drift=False, pi_mode="free",
                         split_merge=False, seed=sd * 100 + rs, iters=120,
                         verbose=False))
                cand = em(Xs, ws, args.K, drift=not args.no_drift, pi_mode=pm,
                          split_merge=sm, seed=sd * 100 + rs,
                          half_life=args.half_life, names=names,
                          verbose=(sd == 0 and sm and rs == 0), init=w2)
                if cand["ll"] > best_ll:
                    best_ll, best_f = cand["ll"], cand
            f = best_f
            p = score(Xte, wte, f["th_next"], f["pi_next"])
            c = best_pi(Xte, wte, f["th_next"])
            used = int((f["Pi"].max(0) > 1e-3).sum())
            res[nm]["p"].append(p); res[nm]["c"].append(c); res[nm]["used"].append(used)
            if pm == "growth": res[nm]["r"].append(f["r"].copy())
            if sm and pm == "free" and sd == 0:
                last_tree = (f["parent"], f["splits"])
            print(f"  {nm:<13} held-out {p:8.3f}   best-mixture {c:8.3f}"
                  f"   rows used {used:>3}/{args.K}", flush=True)

    def ms(v): return f"{np.mean(v):.3f}+/-{np.std(v):.3f}"
    print("\n" + "=" * 76)
    print(f"RESULTS  K={args.K}  {args.seeds} seeds  test {te[0]}")
    print("=" * 76)
    print(f"\n  {'variant':<14}{'held-out':>16}{'best mixture':>16}"
          f"{'rows used':>12}{'mixture gap':>14}")
    for nm, _, _ in variants:
        r_ = res[nm]
        print(f"  {nm:<14}{ms(r_['p']):>16}{ms(r_['c']):>16}"
              f"{np.mean(r_['used']):>12.1f}"
              f"{np.mean(r_['c'])-np.mean(r_['p']):>+14.3f}")
    b = np.mean(res["baseline"]["p"])
    print()
    for nm, _, _ in variants[1:]:
        print(f"  {nm} over baseline: {np.mean(res[nm]['p']) - b:+.3f}")

    if last_tree is not None:
        par, sp = last_tree
        print("\n" + "=" * 76)
        print("THE TREE SPLIT-MERGE LEARNED  (unsupervised)")
        print("=" * 76)
        print("""
  Each split records which block it came from, so the splits form a tree. No
  lineage labels were used -- the split mutation is chosen as the one carrying
  the most within-block dependence.
""")
        if not sp:
            print("  no splits fired")
        else:
            kids = {}
            for _, k, d, nm, sc in sp: kids.setdefault(k, []).append((d, nm, sc))
            roots = sorted({k for _, k, _, _, _ in sp} - {d for _, _, d, _, _ in sp})
            def show(k, depth):
                for d, nm, sc in kids.get(k, []):
                    print(f"  {'    '*depth}blk{k} --{nm}--> blk{d}"
                          f"   (residual {sc:,.0f})")
                    show(d, depth + 1)
            for rt in roots: show(rt, 0)
            print(f"\n  depth reached: "
                  f"{max(len([1 for _ in range(48) ]) for _ in [0]) if False else ''}"
                  f"{len(sp)} splits")

    if res["growth pi"]["r"]:
        rr = res["growth pi"]["r"][0]
        o = np.argsort(-rr)[:5]
        print(f"\n  largest fitted growth rates r_k (log advantage per month):")
        for k in o:
            print(f"    blk{k:<4}{rr[k]:>+8.3f}   x{np.exp(rr[k]):.2f} per month")

    print("""
HOW TO READ
  'rows used' is how many of the K blocks ever held above 0.1% of a month. At
  K=48 the baseline used 19 and left 29 at their initialisation value. If
  split-merge raises this, EM was abandoning capacity it had been given.

  'mixture gap' is the best possible mixture minus what the model actually
  produced. It bounds every rule for forecasting the mixture. Growth pi should
  close it and cannot exceed it -- which is the point of running it.

  r_k is each block's growth advantage per month. Check the largest against
  published estimates for the variant that block corresponds to; that is
  external validation available nowhere else in this project.
""")


if __name__ == "__main__":
    main()
