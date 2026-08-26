#!/usr/bin/env python3
"""
110_hierarchical_birthdeath.py

Two changes, both fitted rather than scheduled.

  HIERARCHY
      A component is stored as a deviation from its parent, not as V free
      numbers:

          beta_child = beta_parent + delta_child,     delta ~ Normal(0, sigma^2)

      The prior shrinks a child toward its parent unless its own sequences
      insist otherwise. Two consequences, both aimed at failures we measured:

        - no component ever sits at 0.5 with zero gradient. An unused child is
          parent + 0, which already explains something, so it competes. The
          flat model left 13-23 of 48 rows at their initialisation value.
        - a component born two months before it sweeps needs a deviation, not a
          full profile. 72% of real births add one or two mutations.

  BIRTH-DEATH
      How many components exist is not fixed. A geometric prior on the count
      penalises each extra component by log(1 - p_birth); a birth is accepted
      only if the gain in expected complete-data log-likelihood exceeds that
      penalty. Deaths are automatic: a component whose responsibility falls
      below a floor is returned to the pool.

      So the schedule in script 109 -- split at iterations 30, 60, 90 -- is
      replaced by a criterion the data evaluates. p_birth is itself fitted from
      the accepted births, so nothing is hand-set except its prior.

  WHY THIS ORDER
      Hierarchy first because it makes birth cheap. Proposing a birth in a flat
      model means proposing V free numbers; here it means proposing a deviation
      concentrated on one mutation, which is what the birth events look like.

INFERENCE
      Variational EM. The E-step is exact (responsibilities over components).
      The M-step maximises the expected complete-data log-likelihood plus the
      log prior -- closed form for delta given the Gaussian prior, gradient
      steps for the drift slopes. The birth-death step is a Metropolis-style
      accept on the same objective, so every move is judged by one quantity.

Usage:
  python 110_hierarchical_birthdeath.py \
      --data-dir data/processed/full_data_graphs_withdel \
      --vocab    data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
      --train 2021-06:2022-05 --test 2022-06 --max-K 48 --seeds 3
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
    for i, (s_, _) in enumerate(records):
        X[i, [n for n in s_ if 0 <= n < V]] = 1.0
    return X, w


def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def loglik_matrix(X, th):
    lt, lc = np.log(th + EPS), np.log(1 - th + EPS)
    return X @ (lt - lc).T + lc.sum(1)[None, :]


# ---------------------------------------------------------------- model
class HierMixture:
    """Components in a tree. Each stores a deviation from its parent, so its
    profile is the sum of deviations along the path to the root."""

    def __init__(self, V, max_K, sigma=1.5, rng=None):
        self.V, self.max_K, self.sigma = V, max_K, sigma
        self.rng = rng or np.random.default_rng(0)
        self.parent = np.full(max_K, -1, dtype=int)
        self.delta = np.zeros((max_K, V))
        self.gamma = np.zeros((max_K, V))       # drift slope, per component
        self.alive = np.zeros(max_K, dtype=bool)
        self.split_on = np.full(max_K, -1, dtype=int)

    def beta(self, k):
        """Profile logits: deviations summed along the path to the root."""
        b = np.zeros(self.V); j = k
        while j >= 0:
            b += self.delta[j]; j = self.parent[j]
        return b

    def theta(self, t=0.0, drift=True):
        """(K_alive, V) profiles at time t, and the index of each."""
        ks = np.flatnonzero(self.alive)
        B = np.stack([self.beta(k) for k in ks])
        if drift:
            B = B + np.stack([self.gamma[k] for k in ks]) * t
        return np.clip(sig(B), 1e-4, 1 - 1e-4), ks

    def depth(self, k):
        d = 0; j = self.parent[k]
        while j >= 0: d += 1; j = self.parent[j]
        return d

    def free(self):
        f = np.flatnonzero(~self.alive)
        return int(f[0]) if len(f) else -1


def log_prior(model, p_birth):
    """Geometric prior on the number of components, plus the Gaussian prior on
    each deviation. This is what a birth has to overcome."""
    K = int(model.alive.sum())
    lp = K * np.log(max(p_birth, 1e-6)) + np.log(max(1 - p_birth, 1e-6))
    for k in np.flatnonzero(model.alive):
        d = model.delta[k]
        lp -= 0.5 * float((d * d).sum()) / model.sigma ** 2
    return lp


# ---------------------------------------------------------------- fitting
def e_step(Xs, ws, model, Pi, tv, drift):
    """Exact responsibilities, plus the sufficient statistics the M-step needs."""
    ks = np.flatnonzero(model.alive); K = len(ks)
    V = model.V; T = len(Xs)
    S = np.zeros((K, V)); N = np.zeros((T, K)); n = np.zeros((K, 1))
    Sd = [np.zeros((K, V)) for _ in range(T)]
    ll = tot = 0.0
    for t, (X, w) in enumerate(zip(Xs, ws)):
        th, _ = model.theta(tv[t], drift)
        lp = loglik_matrix(X, th) + np.log(Pi[t] + EPS)[None, :]
        mx = lp.max(1, keepdims=True)
        P = np.exp(lp - mx); Z = P.sum(1, keepdims=True)
        R = P / Z; Rw = R * w[:, None]
        Sd[t] = Rw.T @ X
        S += Sd[t]; N[t] = Rw.sum(0); n += Rw.sum(0)[:, None]
        ll += float((w * (np.log(Z).ravel() + mx.ravel())).sum()); tot += w.sum()
    return S, Sd, n, N, ll / tot, ks


def m_step(model, ks, Sd, N, tv, drift, inner, lr, rw):
    """Update each deviation and slope.

    For component i the gradient of the expected complete-data log-likelihood
    with respect to its profile logits is  S_t[i] - N[t,i] * theta_t[i],
    summed over months. Because a child's profile is the sum of deviations
    along its path to the root, changing a parent moves every descendant -- so
    a node's gradient with respect to its OWN deviation is its own residual
    plus all of its descendants'. That is the accumulation below, done in order
    of decreasing depth so children are added before their parents are used.

    The Gaussian prior contributes -delta / sigma^2, which is the shrinkage
    toward the parent.
    """
    V = model.V
    kidx = {int(k): i for i, k in enumerate(ks)}
    tot = float(N.sum())
    for _ in range(inner):
        gb = np.zeros((len(ks), V)); gg = np.zeros((len(ks), V))
        for t in range(len(Sd)):
            th, _ = model.theta(tv[t], drift)
            g = Sd[t] - N[t][:, None] * th
            gb += g
            gg += rw[t] * g * tv[t]
        # push each node's residual up to its ancestors
        acc = gb.copy()
        for i in sorted(range(len(ks)), key=lambda i: -model.depth(int(ks[i]))):
            p = int(model.parent[int(ks[i])])
            if p in kidx: acc[kidx[p]] += acc[i]
        for i, k in enumerate(ks):
            k = int(k)
            model.delta[k] += lr * (acc[i] - model.delta[k] / model.sigma ** 2) / tot
            if drift:
                model.gamma[k] += lr * gg[i] / tot
    return model


def try_birth(Xs, ws, model, Pi, tv, drift, p_birth, names, verbose):
    """Propose splitting the component with the most within-component
    dependence, and accept only if the gain beats the prior penalty.

    The proposal is a deviation concentrated on the offending mutation, which
    is what the hierarchy makes cheap.
    """
    slot = model.free()
    if slot < 0: return model, None
    ks = np.flatnonzero(model.alive)
    best = (-np.inf, -1, -1)
    for k in ks:
        num = np.zeros(model.V); den = 0.0; Co = None
        for t, (X, w) in enumerate(zip(Xs, ws)):
            th, kk = model.theta(tv[t], drift)
            lp = loglik_matrix(X, th) + np.log(Pi[t] + EPS)[None, :]
            m = kk[lp.argmax(1)] == k
            if not m.any(): continue
            Xk, wk = X[m], w[m]
            num += (wk[:, None] * Xk).sum(0); den += wk.sum()
            C = (Xk * wk[:, None]).T @ Xk
            Co = C if Co is None else Co + C
        if den < 200 or Co is None: continue
        p = num / den
        var = np.flatnonzero((p > .05) & (p < .95))
        if len(var) < 2: continue
        pv = p[var]
        Exp = den * np.outer(pv, pv)
        R = (Co[np.ix_(var, var)] - Exp) / np.sqrt(Exp + 1.0)
        np.fill_diagonal(R, 0.0)
        sc = np.abs(R).sum(1)
        order = np.argsort(-sc)
        for jj in order[:5]:
            n_try = int(var[jj])
            # a (parent, mutation) pair already used is not a new candidate
            if any(model.parent[c] == k and model.split_on[c] == n_try
                   for c in np.flatnonzero(model.alive)):
                continue
            if sc[jj] > best[0]: best = (float(sc[jj]), int(k), n_try)
            break

    score, k, n_mut = best
    if k < 0: return model, None

    # the proposed child: parent plus a deviation on the offending mutation
    prev_ll = None
    S, Sd, nmass, N, ll0, _ = e_step(Xs, ws, model, Pi, tv, drift)
    lp0 = log_prior(model, p_birth)

    # Initialise the child from the sequences that actually carry the mutation,
    # and the parent from those that do not, then store the child as the
    # DIFFERENCE. Setting delta[n_mut] to a constant instead only makes the
    # child carry the mutation more strongly than its parent, so the two are
    # not separated and the same candidate wins again next time.
    na = np.zeros(model.V); da = 0.0
    nb = np.zeros(model.V); db = 0.0
    ma = np.zeros(len(Xs)); mb = np.zeros(len(Xs))
    for t, (X, w) in enumerate(zip(Xs, ws)):
        th, kk = model.theta(tv[t], drift)
        lp = loglik_matrix(X, th) + np.log(Pi[t] + EPS)[None, :]
        m = kk[lp.argmax(1)] == k
        if not m.any(): continue
        has = X[:, n_mut] > 0
        A_, B_ = m & has, m & ~has
        if A_.any():
            na += (w[A_, None] * X[A_]).sum(0); da += w[A_].sum()
            ma[t] = w[A_].sum() / w.sum()
        if B_.any():
            nb += (w[B_, None] * X[B_]).sum(0); db += w[B_].sum()
            mb[t] = w[B_].sum() / w.sum()
    if da < 50 or db < 50:
        return model, None

    beta_parent_old = model.beta(k)
    p_child = np.clip((na + .5) / (da + 1.0), 1e-4, 1 - 1e-4)
    p_par   = np.clip((nb + .5) / (db + 1.0), 1e-4, 1 - 1e-4)
    b_child = np.log(p_child / (1 - p_child))
    b_par   = np.log(p_par / (1 - p_par))

    # parent keeps the sequences without the mutation; its own deviation moves
    # by the same amount so its ancestors are undisturbed
    delta_k_old = model.delta[k].copy()
    model.delta[k] = delta_k_old + (b_par - beta_parent_old)

    model.alive[slot] = True
    model.parent[slot] = k
    model.split_on[slot] = n_mut
    model.delta[slot] = b_child - b_par          # child as a deviation

    ki = list(np.flatnonzero(model.alive[:slot]))
    Pi2 = np.zeros((len(Pi), Pi.shape[1] + 1))
    Pi2[:, :Pi.shape[1]] = Pi
    kpos = [i for i, kk_ in enumerate(np.flatnonzero(model.alive)) if kk_ == k]
    if kpos:
        Pi2[:, kpos[0]] = mb
    Pi2[:, -1] = ma
    Pi2 = Pi2 / np.maximum(Pi2.sum(1, keepdims=True), EPS)

    S, Sd, nmass, N, ll1, _ = e_step(Xs, ws, model, Pi2, tv, drift)
    lp1 = log_prior(model, p_birth)

    gain = (ll1 - ll0) * sum(w.sum() for w in ws) + (lp1 - lp0)
    if gain > 0:
        if verbose:
            nm = names.get(n_mut, str(n_mut))
            print(f"      birth: blk{k} --{nm}--> blk{slot}"
                  f"   dependence {score:,.0f}   gain {gain:,.0f}", flush=True)
        return model, (Pi2, k, slot, n_mut, score, gain)
    model.alive[slot] = False
    model.parent[slot] = -1
    model.delta[slot] = 0.0
    model.delta[k] = delta_k_old              # undo the parent's move too
    return model, None


def fit(Xs, ws, V, max_K, seed=0, iters=200, inner=8, lr=2.0, sigma=1.5,
        half_life=1.0, drift=True, p_birth=0.5, birth_every=20,
        death_floor=1e-4, names=None, verbose=True):
    rng = np.random.default_rng(seed)
    T = len(Xs)
    model = HierMixture(V, max_K, sigma=sigma, rng=rng)
    # root: the pooled profile
    Xall = np.vstack(Xs); wall = np.concatenate(ws)
    p0 = np.clip((wall[:, None] * Xall).sum(0) / wall.sum(), .02, .98)
    model.alive[0] = True
    model.delta[0] = np.log(p0 / (1 - p0))
    Pi = np.ones((T, 1))
    tv = (np.arange(T) - (T - 1) / 2.) / max(T - 1, 1)
    rw = (0.5 ** (np.arange(T)[::-1] / half_life)) if half_life > 0 else np.ones(T)
    rw /= rw.mean()
    births = []
    prev = -np.inf

    for it in range(iters):
        S, Sd, nmass, N, ll, ks = e_step(Xs, ws, model, Pi, tv, drift)
        model = m_step(model, ks, Sd, N, tv, drift, inner, lr, rw)
        Pi = N / np.maximum(N.sum(1, keepdims=True), EPS)

        # death: a component nobody uses returns to the pool
        if it > 10:
            dead = [int(ks[i]) for i in range(len(ks))
                    if Pi[:, i].max() < death_floor and model.parent[ks[i]] >= 0]
            for k in dead:
                model.alive[k] = False
                for j in range(max_K):          # reattach orphans to grandparent
                    if model.parent[j] == k: model.parent[j] = model.parent[k]
                model.delta[k] = 0.0
            if dead:
                keep = [i for i in range(len(ks)) if int(ks[i]) not in dead]
                Pi = Pi[:, keep]; Pi /= np.maximum(Pi.sum(1, keepdims=True), EPS)
                if verbose:
                    print(f"      death: {len(dead)} component(s) returned",
                          flush=True)

        # birth, judged against the prior rather than a schedule
        if it >= 20 and it % birth_every == 0:
            model, res = try_birth(Xs, ws, model, Pi, tv, drift, p_birth,
                                   names or {}, verbose)
            if res is not None:
                Pi = res[0]; births.append(res[1:])
                # p_birth is itself fitted from how often births are accepted
                K_now = int(model.alive.sum())
                p_birth = min(.95, max(.05, K_now / (K_now + 2.0)))

        if verbose and (it + 1) % 50 == 0:
            print(f"      iter {it+1}  LL/obs {ll:.5f}  "
                  f"components {int(model.alive.sum())}", flush=True)
        if abs(ll - prev) < 1e-6 and it > 120: break
        prev = ll

    return model, Pi, tv, births, ll


def fit_flat(Xs, ws, V, K, seed=0, drift=False, split_merge=False,
             iters=200, inner=8, lr=2.0, half_life=1.0, prior=.5,
             names=None, verbose=False, init_beta=None):
    """Flat mixture: K independent components, optionally with drifting
    emissions and scheduled split-merge. These are the ladder's lower rungs, so
    every rung is fitted by the same code on the same data."""
    rng = np.random.default_rng(seed)
    T = len(Xs)
    if init_beta is not None:
        # Fitting beta and gamma together from a random start collapses every
        # component onto the pooled mean. Warm-starting from a converged
        # fixed-emission fit means beta already separates them and only the
        # slopes have to be learned.
        beta = init_beta.copy()
    else:
        mean = np.vstack(Xs).mean(0)
        th0 = np.clip(rng.random((K, V)) * .4 + mean[None, :] * .6, .02, .98)
        beta = np.log(th0 / (1 - th0))
    gamma = np.zeros((K, V))
    Pi = np.full((T, K), 1.0 / K)
    tv = (np.arange(T) - (T - 1) / 2.) / max(T - 1, 1)
    rw = (0.5 ** (np.arange(T)[::-1] / half_life)) if half_life > 0 else np.ones(T)
    rw /= rw.mean()
    splits = []; prev = -np.inf

    def th_at(t):
        return np.clip(sig(beta + gamma * tv[t]) if drift else sig(beta),
                       1e-4, 1 - 1e-4)

    for it in range(iters):
        Sd = []; N = np.zeros((T, K)); ll = tot = 0.
        for t, (X, w) in enumerate(zip(Xs, ws)):
            lp = loglik_matrix(X, th_at(t)) + np.log(Pi[t] + EPS)[None, :]
            mx = lp.max(1, keepdims=True)
            P = np.exp(lp - mx); Z = P.sum(1, keepdims=True)
            Rw = (P / Z) * w[:, None]
            Sd.append(Rw.T @ X); N[t] = Rw.sum(0)
            ll += float((w * (np.log(Z).ravel() + mx.ravel())).sum()); tot += w.sum()
        ll /= tot

        if not drift:
            num = sum(Sd); den = N.sum(0)
            th = np.clip((num + prior) / (den[:, None] + 2 * prior), 1e-4, 1 - 1e-4)
            beta = np.log(th / (1 - th))
        else:
            for _ in range(inner):
                gb = np.zeros((K, V)); gg = np.zeros((K, V))
                for t in range(T):
                    g = Sd[t] - N[t][:, None] * th_at(t)
                    gb += g; gg += rw[t] * g * tv[t]
                beta += lr * gb / max(N.sum(), 1.); gamma += lr * gg / max(N.sum(), 1.)
        Pi = N / np.maximum(N.sum(1, keepdims=True), EPS)

        if split_merge and it in (30, 60, 90, 120, 150):
            dead = [k for k in range(K) if Pi[:, k].max() < 1e-4]
            if dead:
                th_now = th_at(T - 1); best = (-np.inf, -1, -1)
                for k in np.argsort(-Pi.mean(0))[:6]:
                    if Pi[:, k].mean() < .01: continue
                    num = np.zeros(V); den = 0.; Co = None
                    for t, (X, w) in enumerate(zip(Xs, ws)):
                        lp = loglik_matrix(X, th_now) + np.log(Pi[t] + EPS)[None, :]
                        m = lp.argmax(1) == k
                        if not m.any(): continue
                        Xk, wk = X[m], w[m]
                        num += (wk[:, None] * Xk).sum(0); den += wk.sum()
                        C = (Xk * wk[:, None]).T @ Xk
                        Co = C if Co is None else Co + C
                    if den < 200 or Co is None: continue
                    p = num / den
                    var = np.flatnonzero((p > .05) & (p < .95))
                    if len(var) < 2: continue
                    pv = p[var]; Exp = den * np.outer(pv, pv)
                    R = (Co[np.ix_(var, var)] - Exp) / np.sqrt(Exp + 1.)
                    np.fill_diagonal(R, 0.)
                    sc = np.abs(R).sum(1); j = int(np.argmax(sc))
                    if sc[j] > best[0]: best = (float(sc[j]), int(k), int(var[j]))
                sc, k, n_mut = best
                if k >= 0:
                    d = dead[0]
                    na = np.zeros(V); da = 0.; nb = np.zeros(V); db = 0.
                    ma = np.zeros(T); mb = np.zeros(T)
                    for t, (X, w) in enumerate(zip(Xs, ws)):
                        lp = loglik_matrix(X, th_now) + np.log(Pi[t] + EPS)[None, :]
                        m = lp.argmax(1) == k; has = X[:, n_mut] > 0
                        A_, B_ = m & has, m & ~has
                        if A_.any():
                            na += (w[A_, None] * X[A_]).sum(0); da += w[A_].sum()
                            ma[t] = w[A_].sum() / w.sum()
                        if B_.any():
                            nb += (w[B_, None] * X[B_]).sum(0); db += w[B_].sum()
                            mb[t] = w[B_].sum() / w.sum()
                    if da > 50 and db > 50:
                        th_new = th_now.copy()
                        th_new[d] = np.clip((na + .5) / (da + 1.), 1e-4, 1 - 1e-4)
                        th_new[k] = np.clip((nb + .5) / (db + 1.), 1e-4, 1 - 1e-4)
                        beta = np.log(th_new / (1 - th_new)); gamma[d] = 0.
                        Pi[:, d] = ma; Pi[:, k] = mb
                        Pi = Pi / np.maximum(Pi.sum(1, keepdims=True), EPS)
                        splits.append((k, d, n_mut, sc))
                        if verbose:
                            nm = (names or {}).get(n_mut, str(n_mut))
                            print(f"      split blk{k} on {nm} -> blk{d}", flush=True)
        if abs(ll - prev) < 1e-6 and it > 120: break
        prev = ll

    dt = tv[-1] - tv[-2] if T > 1 else 0.
    th_next = np.clip(sig(beta + gamma * (tv[-1] + dt)) if drift else sig(beta),
                      1e-4, 1 - 1e-4)
    return th_next, Pi[-1], int((Pi.max(0) > 1e-3).sum()), ll, splits, beta



def score(X, w, th, pi):
    lp = loglik_matrix(X, th) + np.log(pi + EPS)[None, :]
    mx = lp.max(1, keepdims=True)
    v = (np.log(np.exp(lp - mx).sum(1, keepdims=True)) + mx).ravel()
    return float((w * v).sum() / w.sum())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--max-K", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=1.5,
                    help="prior width on a child's deviation from its parent. "
                         "Small = children stay close, few effective components")
    ap.add_argument("--half-life", type=float, default=1.0)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--no-drift", action="store_true")
    args = ap.parse_args()

    names, V = load_names(args.vocab)
    tr, te = months_in_range(args.train), months_in_range(args.test)
    Xs, ws = [], []
    for ym in tr:
        X, w = build(load_month(args.data_dir, ym), V, args.min_count)
        Xs.append(X); ws.append(w)
    rec = []
    for ym in te: rec += load_month(args.data_dir, ym)
    Xte, wte = build(rec, V)
    print(f"V={V:,} max-K={args.max_K} train {tr[0]}..{tr[-1]} test {te[0]}  "
          f"sigma={args.sigma}  drift={not args.no_drift}")

    rungs = ["flat", "flat + drift", "flat + drift + split-merge",
             "hierarchical + birth-death"]
    res = {r: dict(ll=[], used=[]) for r in rungs}
    trees = None

    for sd in range(args.seeds):
        print(f"\n--- seed {sd} ---", flush=True)

        th, pi, used, _, _, warm = fit_flat(Xs, ws, V, args.max_K, seed=sd,
                                            drift=False, split_merge=False)
        p = score(Xte, wte, th, pi)
        res["flat"]["ll"].append(p); res["flat"]["used"].append(used)
        print(f"  {'flat':<28}{p:8.3f}   components {used}", flush=True)

        th, pi, used, _, _, _ = fit_flat(Xs, ws, V, args.max_K, seed=sd,
                                         drift=not args.no_drift,
                                         split_merge=False,
                                         half_life=args.half_life,
                                         init_beta=warm)
        p = score(Xte, wte, th, pi)
        res["flat + drift"]["ll"].append(p); res["flat + drift"]["used"].append(used)
        print(f"  {'flat + drift':<28}{p:8.3f}   components {used}", flush=True)

        th, pi, used, _, sp, _ = fit_flat(Xs, ws, V, args.max_K, seed=sd,
                                          drift=not args.no_drift,
                                          split_merge=True,
                                          half_life=args.half_life,
                                          names=names, verbose=(sd == 0),
                                          init_beta=warm)
        p = score(Xte, wte, th, pi)
        res[rungs[2]]["ll"].append(p); res[rungs[2]]["used"].append(used)
        print(f"  {'flat + drift + split-merge':<28}{p:8.3f}   components {used}",
              flush=True)

        model, Pi, tv, births, _ = fit(
            Xs, ws, V, args.max_K, seed=sd, sigma=args.sigma,
            half_life=args.half_life, drift=not args.no_drift,
            names=names, verbose=(sd == 0))
        dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.
        th, ks = model.theta(tv[-1] + dt, not args.no_drift)
        p = score(Xte, wte, th, Pi[-1])
        nc = int(model.alive.sum())
        res[rungs[3]]["ll"].append(p); res[rungs[3]]["used"].append(nc)
        print(f"  {'hierarchical + birth-death':<28}{p:8.3f}   components {nc}"
              f"   births {len(births)}", flush=True)
        if sd == 0: trees = (model, births)

    print("\n" + "=" * 76)
    print(f"LADDER  max-K={args.max_K}  {args.seeds} seeds  test {te[0]}")
    print("=" * 76)
    print(f"\n  {'model':<30}{'held-out':>18}{'components':>14}{'gain':>9}")
    base = np.mean(res["flat"]["ll"])
    for r in rungs:
        m = np.mean(res[r]["ll"])
        print(f"  {r:<30}{m:8.3f}+/-{np.std(res[r]['ll']):<7.3f}"
              f"{np.mean(res[r]['used']):>14.1f}{m - base:>+9.3f}")

    if trees is not None:
        model, births = trees
        print(f"\n  THE TREE (births accepted against the prior):")
        if not births:
            print("    none accepted")
        else:
            kids = {}
            for k, slot, n_mut, sc, gain in births:
                kids.setdefault(k, []).append((slot, n_mut, gain))
            def show(k, d):
                for slot, n_mut, gain in kids.get(k, []):
                    print(f"    {'    '*d}blk{k} --{names.get(n_mut, n_mut)}-->"
                          f" blk{slot}   gain {gain:,.0f}")
                    show(slot, d + 1)
            show(0, 0)

    print("""
NOTE
  Components are not fixed: max-K is a ceiling on the pool, not the number
  fitted. A birth is accepted only when the gain in expected complete-data
  log-likelihood exceeds the prior penalty for one more component, and a
  component whose responsibility falls below the floor is returned to the pool.
  So the number of components reported above was chosen by the data.

  sigma controls how far a child may sit from its parent. Smaller sigma means
  tighter children and, indirectly, fewer accepted births -- worth sweeping,
  since it is the one quantity here that is set rather than fitted.
""")


if __name__ == "__main__":
    main()
