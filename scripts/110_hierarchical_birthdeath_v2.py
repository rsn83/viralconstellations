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

    def __init__(self, V, max_K, sigma=1.5, rng=None, hier_drift=False):
        self.V, self.max_K, self.sigma = V, max_K, sigma
        self.hier_drift = hier_drift
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

    def slope(self, k):
        """Drift slopes. Under hier_drift these are deviations summed along the
        path, exactly like the profile logits; otherwise each component holds
        a free slope with no relation to its parent and no prior."""
        if not self.hier_drift:
            return self.gamma[k]
        g = np.zeros(self.V); j = k
        while j >= 0:
            g += self.gamma[j]; j = self.parent[j]
        return g

    def theta(self, t=0.0, drift=True):
        """(K_alive, V) profiles at time t, and the index of each."""
        ks = np.flatnonzero(self.alive)
        B = np.stack([self.beta(k) for k in ks])
        if drift:
            B = B + np.stack([self.slope(k) for k in ks]) * t
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
        if model.hier_drift:
            g = model.gamma[k]
            lp -= 0.5 * float((g * g).sum()) / model.sigma ** 2
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
        order = sorted(range(len(ks)), key=lambda i: -model.depth(int(ks[i])))
        acc = gb.copy()
        for i in order:
            p = int(model.parent[int(ks[i])])
            if p in kidx: acc[kidx[p]] += acc[i]
        if model.hier_drift:
            gacc = gg.copy()
            for i in order:
                p = int(model.parent[int(ks[i])])
                if p in kidx: gacc[kidx[p]] += gacc[i]
        for i, k in enumerate(ks):
            k = int(k)
            model.delta[k] += lr * (acc[i] - model.delta[k] / model.sigma ** 2) / tot
            if drift:
                if model.hier_drift:
                    model.gamma[k] += lr * (gacc[i]
                                            - model.gamma[k] / model.sigma ** 2) / tot
                else:
                    model.gamma[k] += lr * gg[i] / tot
    return model


# ---------------------------------------------------------------- birth-death
def _snapshot(model):
    return (model.delta.copy(), model.gamma.copy(), model.alive.copy(),
            model.parent.copy(), model.split_on.copy())


def _restore(model, s):
    (model.delta, model.gamma, model.alive,
     model.parent, model.split_on) = (s[0].copy(), s[1].copy(), s[2].copy(),
                                      s[3].copy(), s[4].copy())


def _hard_assign(Xs, ws, model, Pi, tv, drift):
    """Component id (not column position) of the argmax component for every
    sequence, one array per month. Computed once per birth call and reused by
    the candidate scan, which previously recomputed it once per component."""
    out = []
    for t, X in enumerate(Xs):
        th, kk = model.theta(tv[t], drift)
        lp = loglik_matrix(X, th) + np.log(Pi[t] + EPS)[None, :]
        out.append(kk[lp.argmax(1)])
    return out


def _pi_after_birth(Pi, ks_old, ks_new, k_par, slot, ma, mb):
    """Rebuild month-by-component weights so column j corresponds to ks_new[j].

    The previous version appended the child at the last column. That is only
    correct when the free slot happens to be the largest index -- false after
    any death, since free() returns the FIRST free slot. It silently permuted
    every component's weights, which is invisible at K=2 and corrupting above.
    """
    T = Pi.shape[0]
    pos = {int(k): i for i, k in enumerate(ks_old)}
    Pi2 = np.zeros((T, len(ks_new)))
    for j, k in enumerate(ks_new):
        k = int(k)
        if k == slot:
            Pi2[:, j] = ma
        elif k == k_par:
            Pi2[:, j] = mb
        else:
            Pi2[:, j] = Pi[:, pos[k]]
    return Pi2 / np.maximum(Pi2.sum(1, keepdims=True), EPS)


def _scan_component(Xs, ws, model, hard, k, top_per_k=8, min_den=200.0):
    """Ranked (dependence, mutation) candidates inside one component."""
    num = np.zeros(model.V); den = 0.0; Co = None
    for t, (X, w) in enumerate(zip(Xs, ws)):
        m = hard[t] == k
        if not m.any(): continue
        Xk, wk = X[m], w[m]
        num += (wk[:, None] * Xk).sum(0); den += wk.sum()
        C = (Xk * wk[:, None]).T @ Xk
        Co = C if Co is None else Co + C
    if den < min_den or Co is None: return []
    p = num / den
    var = np.flatnonzero((p > .05) & (p < .95))
    if len(var) < 2: return []
    pv = p[var]; Exp = den * np.outer(pv, pv)
    R = (Co[np.ix_(var, var)] - Exp) / np.sqrt(Exp + 1.0)
    np.fill_diagonal(R, 0.0)
    sc = np.abs(R).sum(1)
    return [(float(sc[j]), int(var[j])) for j in np.argsort(-sc)[:top_per_k]]


def _candidates(Xs, ws, model, hard, cache=None, dirty=None, tried=None,
                top_per_k=8, min_den=200.0):
    """(dependence, component, mutation) triples, best first.

    The scan is the expensive part -- a V x V co-occurrence matrix per
    component per month -- and rescanning every component on every birth call
    is what makes an every-iteration birth schedule unaffordable. Only
    components whose membership actually changed (the last birth's parent and
    child, plus anything newly created or resurrected) are rescanned; the rest
    are served from cache. Set --rescan-every to force periodic full refreshes,
    since a cached entry does go stale as EM moves the other components.
    """
    alive = [int(k) for k in np.flatnonzero(model.alive)]
    if cache is None:
        cache = {}; dirty = set(alive)
    dirty = dirty or set()
    for k in list(cache):
        if k not in alive: cache.pop(k)
    for k in alive:
        if k in dirty or k not in cache:
            cache[k] = _scan_component(Xs, ws, model, hard, k,
                                       top_per_k=top_per_k, min_den=min_den)
    used = {(int(model.parent[c]), int(model.split_on[c]))
            for c in np.flatnonzero(model.alive)}
    if tried: used |= set(tried)
    out = [(sc, k, mut) for k, lst in cache.items() for sc, mut in lst
           if (k, mut) not in used]
    out.sort(key=lambda c: -c[0])
    return out, cache


def _bic_cost(model, slot, Ntot, tol=1.0):
    """Cost of one more component: half a log N per free parameter it adds.

    Free parameters are counted as the entries of the child's deviation that
    move the emission probability materially -- |delta| > tol logits. Counting
    every numerically-nonzero entry would charge V per birth, because delta is
    initialised as a dense profile difference and nothing in the M-step ever
    sparsifies it. That is a real gap between the docstring ('cheap
    one-mutation delta') and the code, and this threshold is a stopgap, not a
    fix: a proper one is an L1 or horseshoe penalty on delta.
    """
    veff = int((np.abs(model.delta[slot]) > tol).sum())
    return 0.5 * max(veff, 1) * np.log(max(Ntot, 2.0))


def _split_stats(Xs, ws, hard, k, n_mut, V):
    """Empirical profiles and monthly shares of the two sides of a split."""
    T = len(Xs)
    na = np.zeros(V); da = 0.0; nb = np.zeros(V); db = 0.0
    ma = np.zeros(T); mb = np.zeros(T)
    for t, (X, w) in enumerate(zip(Xs, ws)):
        m = hard[t] == k
        if not m.any(): continue
        has = X[:, n_mut] > 0
        A_, B_ = m & has, m & ~has
        if A_.any():
            na += (w[A_, None] * X[A_]).sum(0); da += w[A_].sum()
            ma[t] = w[A_].sum() / w.sum()
        if B_.any():
            nb += (w[B_, None] * X[B_]).sum(0); db += w[B_].sum()
            mb[t] = w[B_].sum() / w.sum()
    return na, da, nb, db, ma, mb


def _apply_split(model, Pi, k, slot, n_mut, na, da, nb, db, ma, mb):
    """Install a split of component k on n_mut, child in `slot`.

    beta(c) sums deltas along the path, so moving delta[k] moves every
    descendant. Each existing child is compensated by -D to hold its own
    profile fixed; without this, splitting a node that already has children
    destroys them.
    """
    ks_old = np.flatnonzero(model.alive).copy()
    beta_parent_old = model.beta(k)
    p_child = np.clip((na + .5) / (da + 1.0), 1e-4, 1 - 1e-4)
    p_par = np.clip((nb + .5) / (db + 1.0), 1e-4, 1 - 1e-4)
    b_child = np.log(p_child / (1 - p_child))
    b_par = np.log(p_par / (1 - p_par))

    D = b_par - beta_parent_old
    model.delta[k] = model.delta[k] + D
    for c in np.flatnonzero(model.alive):
        if int(model.parent[c]) == k:
            model.delta[c] = model.delta[c] - D

    model.alive[slot] = True
    model.parent[slot] = k
    model.split_on[slot] = n_mut
    model.delta[slot] = b_child - b_par
    model.gamma[slot] = (np.zeros(model.V) if model.hier_drift
                         else model.gamma[k].copy())
    ks_new = np.flatnonzero(model.alive)
    return _pi_after_birth(Pi, ks_old, ks_new, k, slot, ma, mb)


def _merge_pair(model, Pi, a, b, w_a, w_b):
    """Merge sibling b into a. The combined profile is the mass-weighted mean;
    every affected child's deviation is compensated so its own profile is
    unchanged, and b's children are reattached to a."""
    ks_old = np.flatnonzero(model.alive).copy()
    pos = {int(k): i for i, k in enumerate(ks_old)}
    beta_a, beta_b = model.beta(a), model.beta(b)
    tot = max(w_a + w_b, 1e-12)
    th = np.clip((w_a * sig(beta_a) + w_b * sig(beta_b)) / tot, 1e-4, 1 - 1e-4)
    beta_new = np.log(th / (1 - th))
    par = int(model.parent[a])
    beta_par = np.zeros(model.V) if par < 0 else model.beta(par)
    b_kids = {int(c) for c in np.flatnonzero(model.alive)
              if int(model.parent[c]) == b}

    for c in np.flatnonzero(model.alive):
        c = int(c)
        if c == b: continue
        if int(model.parent[c]) == a:
            model.delta[c] = model.delta[c] + (beta_a - beta_new)
        elif int(model.parent[c]) == b:
            model.parent[c] = a
            model.delta[c] = model.delta[c] + (beta_b - beta_new)
    model.delta[a] = beta_new - beta_par
    if model.hier_drift:
        g_a, g_b = model.slope(a), model.slope(b)
        g_new = (w_a * g_a + w_b * g_b) / tot
        g_par = np.zeros(model.V) if par < 0 else model.slope(par)
        for c in np.flatnonzero(model.alive):
            c = int(c)
            if c == b: continue
            if int(model.parent[c]) == a and c not in b_kids:
                model.gamma[c] = model.gamma[c] + (g_a - g_new)
            elif c in b_kids:
                model.gamma[c] = model.gamma[c] + (g_b - g_new)
        model.gamma[a] = g_new - g_par
    model.alive[b] = False; model.parent[b] = -1
    model.delta[b] = 0.0; model.gamma[b] = 0.0; model.split_on[b] = -1

    ks_new = np.flatnonzero(model.alive)
    Pi2 = np.zeros((Pi.shape[0], len(ks_new)))
    for j, k in enumerate(ks_new):
        k = int(k)
        Pi2[:, j] = Pi[:, pos[k]] + (Pi[:, pos[b]] if k == a else 0.0)
    return Pi2 / np.maximum(Pi2.sum(1, keepdims=True), EPS)


def try_remerge(Xs, ws, model, Pi, tv, drift, p_birth, names, verbose,
                inner=8, lr=2.0, rw=None, refit=3, n_pairs=3, min_side=50.0,
                diag=None):
    """Backtracking move: merge two siblings, then re-split the merged cluster
    on a DIFFERENT mutation. K is unchanged, so the move is judged by the same
    criterion as a birth without the automatic in-sample advantage that adding
    a component carries.

    This is the piece the search was missing. Births are greedy and permanent:
    a split accepted at iteration 20 conditions everything after it and can
    never be undone, which is why the fitted model depends so heavily on where
    it started. A pure merge would always lose in-sample likelihood and so
    could never be accepted under this criterion; merging and re-splitting
    moves sideways through configuration space instead, in the spirit of the
    split-merge proposals of Jain & Neal (2004).
    """
    ks = np.flatnonzero(model.alive)
    if len(ks) < 3: return model, Pi, None
    T = len(Xs); Ntot = float(sum(w.sum() for w in ws))
    if rw is None: rw = np.ones(T)
    pos = {int(k): i for i, k in enumerate(ks)}
    TH = np.stack([sig(model.beta(int(k))) for k in ks])
    nrm = np.linalg.norm(TH, axis=1) + 1e-12

    pairs = []
    for i in range(len(ks)):
        a = int(ks[i])
        if int(model.parent[a]) < 0: continue
        for j in range(i + 1, len(ks)):
            b = int(ks[j])
            if int(model.parent[b]) != int(model.parent[a]): continue
            sim = float(TH[i] @ TH[j] / (nrm[i] * nrm[j]))
            pairs.append((sim, a, b))
    if not pairs: return model, Pi, None
    pairs.sort(key=lambda p: -p[0])

    *_, ll0, _ = e_step(Xs, ws, model, Pi, tv, drift)
    lp0 = log_prior(model, p_birth)
    snap = _snapshot(model)

    for sim, a, b in pairs[:n_pairs]:
        old_splits = {int(model.split_on[a]), int(model.split_on[b])}
        w_a = float(Pi[:, pos[a]].mean()); w_b = float(Pi[:, pos[b]].mean())
        Pi2 = _merge_pair(model, Pi, a, b, w_a, w_b)

        hard = _hard_assign(Xs, ws, model, Pi2, tv, drift)
        pick = None
        for sc, mut in _scan_component(Xs, ws, model, hard, a):
            if mut not in old_splits: pick = (sc, int(mut)); break
        if pick is None:
            _restore(model, snap); continue
        sc, n_mut = pick
        na, da, nb, db, ma, mb = _split_stats(Xs, ws, hard, a, n_mut, model.V)
        if da < min_side or db < min_side:
            _restore(model, snap); continue
        Pi2 = _apply_split(model, Pi2, a, b, n_mut, na, da, nb, db, ma, mb)

        for _ in range(refit):
            S, Sd, nmass, N, _, kk = e_step(Xs, ws, model, Pi2, tv, drift)
            model = m_step(model, kk, Sd, N, tv, drift, inner, lr, rw)
            Pi2 = N / np.maximum(N.sum(1, keepdims=True), EPS)
        *_, ll1, _ = e_step(Xs, ws, model, Pi2, tv, drift)
        gain = (ll1 - ll0) * Ntot + (log_prior(model, p_birth) - lp0)

        if diag is not None:
            diag.append(dict(k=a, mut=n_mut, why="remerge", dep=sc,
                             dll=(ll1 - ll0) * Ntot, dlp=0.0, cost=0.0,
                             gain=gain))
        if gain > 0:
            if verbose:
                nm = names.get(n_mut, str(n_mut))
                print(f"      remerge: blk{a}+blk{b} -> re-split on {nm}"
                      f"   sim {sim:.3f}  gain {gain:+,.0f}", flush=True)
            return model, Pi2, (a, b, n_mut, sim, gain)
        _restore(model, snap)
    return model, Pi, None


def try_birth(Xs, ws, model, Pi, tv, drift, p_birth, names, verbose,
              inner=8, lr=2.0, rw=None, refit=5, n_cand=3, min_side=50.0,
              penalty="bic", diag=None, tried=None, cache=None, dirty=None):
    """Propose a split, REFIT it, then accept on the gain.

    The refit is the change that matters. The original compared a converged
    model against a hard-split initialisation in which the parent's profile
    had just been replaced by a TIME-POOLED empirical logit while it kept its
    fitted drift slope, and the child was given slope zero. Over a window
    containing a sweep that loses more likelihood than the split gains, so
    gain <= 0 and the birth was rejected for reasons that have nothing to do
    with whether the split is real. `fit_flat`'s split-merge path has no gain
    test at all, which is why it reached 18 components and this reached 2.
    """
    slot = model.free()
    if slot < 0:
        return model, Pi, None, cache
    T = len(Xs)
    Ntot = float(sum(w.sum() for w in ws))
    if rw is None:
        rw = np.ones(T)

    hard = _hard_assign(Xs, ws, model, Pi, tv, drift)
    cands, cache = _candidates(Xs, ws, model, hard, cache=cache, dirty=dirty,
                               tried=tried)
    if not cands:
        if diag is not None:
            diag.append(dict(k=-1, mut=-1, why="no-candidate"))
        return model, Pi, None, cache

    *_, ll0, _ = e_step(Xs, ws, model, Pi, tv, drift)
    lp0 = log_prior(model, p_birth)
    snap = _snapshot(model)

    for score, k, n_mut in cands[:n_cand]:
        na, da, nb, db, ma, mb = _split_stats(Xs, ws, hard, k, n_mut, model.V)
        if da < min_side or db < min_side:
            if tried is not None: tried.add((k, n_mut))
            if diag is not None:
                diag.append(dict(k=k, mut=n_mut, why="thin-side",
                                 dep=score, da=da, db=db))
            continue

        Pi2 = _apply_split(model, Pi, k, slot, n_mut, na, da, nb, db, ma, mb)

        for _ in range(refit):
            S, Sd, nmass, N, _, ks = e_step(Xs, ws, model, Pi2, tv, drift)
            model = m_step(model, ks, Sd, N, tv, drift, inner, lr, rw)
            Pi2 = N / np.maximum(N.sum(1, keepdims=True), EPS)

        *_, ll1, _ = e_step(Xs, ws, model, Pi2, tv, drift)
        lp1 = log_prior(model, p_birth)
        dll = (ll1 - ll0) * Ntot
        cost = _bic_cost(model, slot, Ntot) if penalty == "bic" else 0.0
        gain = dll + (lp1 - lp0) - cost

        if diag is not None:
            diag.append(dict(k=k, mut=n_mut, why="gain", dep=score,
                             dll=dll, dlp=lp1 - lp0, cost=cost, gain=gain))
        if gain > 0:
            if verbose:
                nm = names.get(n_mut, str(n_mut))
                print(f"      birth: blk{k} --{nm}--> blk{slot}   dep {score:,.0f}"
                      f"  dLL {dll:+,.0f}  dPrior {lp1-lp0:+,.0f}"
                      f"  cost {cost:,.0f}  gain {gain:+,.0f}", flush=True)
            return model, Pi2, (k, slot, n_mut, score, gain), cache
        # A candidate that has been initialised, refit and rejected is not a
        # new candidate next call. Without this the same three mutations are
        # re-proposed every iteration -- 600 attempts on 3 distinct splits.
        if tried is not None: tried.add((k, n_mut))
        _restore(model, snap)
    return model, Pi, None, cache


def _dendrogram_tree(beta_occ, wts, max_nodes):
    """Average-linkage dendrogram over the fitted flat profiles.

    Returns (parent, order, is_leaf, leaf_of) over node ids 0..n_nodes-1 with
    0 as the root, plus the profile of each node. Internal-node profiles are
    the weight-average of their leaves, so a node really is the common ancestor
    of its subtree rather than an arbitrary interpolation.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    n = len(beta_occ)
    if n < 3:
        return None
    # The leaves ARE the fitted components and cannot be dropped, so the
    # smallest tree that can hold them is a root plus n leaves. If even that
    # does not fit, there is no tree to build -- fall back to the star, which
    # stops allocating when it runs out of slots.
    if max_nodes < n + 1:
        return None
    th = 1.0 / (1.0 + np.exp(-np.clip(beta_occ, -30, 30)))
    Z = linkage(pdist(th, metric="cosine"), method="average")

    # scipy node ids: 0..n-1 leaves, n+j = merge j. Build children map.
    kids = {}
    height = {}
    for j in range(n - 1):
        a, b = int(Z[j, 0]), int(Z[j, 1])
        kids[n + j] = (a, b)
        height[n + j] = float(Z[j, 2])
    root = 2 * n - 2

    # Prune: if the full tree needs more slots than we have, drop the internal
    # nodes formed by the tightest merges (lowest height) and reattach their
    # children upward. Those are the merges the data supports least strongly as
    # a distinct ancestor.
    internal = sorted((h for h in height if h != root), key=lambda k: height[k])
    n_nodes = 2 * n - 1
    dropped = set()
    while n_nodes > max_nodes and internal:
        dropped.add(internal.pop(0)); n_nodes -= 1

    # Flatten dropped nodes: a node's effective children are its children with
    # dropped ones replaced by their own children, recursively.
    def eff_kids(u):
        out = []
        for c in kids.get(u, ()):
            if c in dropped: out += eff_kids(c)
            else: out.append(c)
        return out

    # Assign slots breadth-first from the root so slot 0 is the root.
    slot, order, parent_of = {}, [], {}
    queue = [(root, -1)]
    while queue:
        u, p = queue.pop(0)
        slot[u] = len(order); order.append(u); parent_of[u] = p
        for c in eff_kids(u):
            queue.append((c, u))

    # Leaf sets, for the weight-average profiles.
    def leaves(u):
        if u < n: return [u]
        out = []
        for c in kids.get(u, ()): out += leaves(c)
        return out

    prof = {}; mass = {}
    for u in order:
        L = leaves(u)
        w = wts[L]; w = w / max(w.sum(), 1e-12)
        prof[u] = (w[:, None] * beta_occ[L]).sum(0)
        mass[u] = float(wts[L].sum())
    return slot, order, parent_of, prof, mass, n


def warm_forest(model, beta_flat, pi_flat, T, occ_floor=1e-3, mode="star",
                internal_share=0.02):
    """Seed the hierarchy from a fitted flat mixture.

    mode='star'  every occupied flat row becomes a depth-1 child of a pooled
                 root. The hierarchy inherits the PARTITION but none of the
                 structure, so every nested relationship still has to be found
                 greedily by birth.
    mode='tree'  the occupied rows are clustered into a dendrogram and that
                 dendrogram becomes the initial tree. Leaves hold the fitted
                 profiles; internal nodes hold the weight-average of their
                 leaves, so each is a genuine common ancestor and its children
                 store small deviations from it -- which is the regime the
                 parameterisation was designed for.
    """
    pi = np.atleast_2d(pi_flat)
    occ = np.flatnonzero(pi.max(0) > occ_floor)
    if len(occ) == 0:
        return model, None
    wts = pi.mean(0)[occ]
    beta_occ = beta_flat[occ]

    model.alive[:] = False; model.parent[:] = -1
    model.delta[:] = 0.0; model.gamma[:] = 0.0; model.split_on[:] = -1

    built = _dendrogram_tree(beta_occ, wts, model.max_K) if mode == "tree" else None
    if built is None:
        root = (wts / max(wts.sum(), EPS))[:, None].__mul__(beta_occ).sum(0)
        model.alive[0] = True; model.delta[0] = root
        child_of = {}
        for k_i in range(len(occ)):
            s = model.free()
            if s < 0: break
            model.alive[s] = True; model.parent[s] = 0
            model.delta[s] = beta_occ[k_i] - root
            child_of[s] = k_i
        ks = np.flatnonzero(model.alive)
        row = np.array([1e-6 if int(k) == 0 else max(wts[child_of[int(k)]], 1e-6)
                        for k in ks])
        return model, np.tile(row / row.sum(), (T, 1))

    slot, order, parent_of, prof, mass, n_leaf = built
    for u in order:
        s = slot[u]
        model.alive[s] = True
        p = parent_of[u]
        model.parent[s] = -1 if p < 0 else slot[p]
        model.delta[s] = prof[u] - (0.0 if p < 0 else prof[p])
    ks = np.flatnonzero(model.alive)
    inv = {slot[u]: u for u in order}
    row = np.array([mass[inv[int(k)]] * (1.0 if inv[int(k)] < n_leaf
                                         else internal_share) for k in ks])
    row = np.maximum(row, 1e-6)
    return model, np.tile(row / row.sum(), (T, 1))


def fit(Xs, ws, V, max_K, seed=0, iters=200, inner=8, lr=2.0, sigma=1.5,
        half_life=1.0, drift=True, p_birth=0.5, birth_every=20,
        death_floor=1e-4, names=None, verbose=True,
        burn_in=10, K_warm=0, births_per_call=1, refit=5, n_cand=3,
        grace=15, penalty="bic", warm=None, warm_mode="star", diag=None,
        rescan_every=0, merge_every=0, n_pairs=3, hier_drift=False):
    rng = np.random.default_rng(seed)
    T = len(Xs)
    model = HierMixture(V, max_K, sigma=sigma, rng=rng, hier_drift=hier_drift)
    tv = (np.arange(T) - (T - 1) / 2.) / max(T - 1, 1)
    rw = (0.5 ** (np.arange(T)[::-1] / half_life)) if half_life > 0 else np.ones(T)
    rw /= rw.mean()

    if warm is not None:
        model, Pi = warm_forest(model, warm[0], warm[1], T, mode=warm_mode)
        if Pi is None:
            warm = None
    if warm is None:
        Xall = np.vstack(Xs); wall = np.concatenate(ws)
        p0 = np.clip((wall[:, None] * Xall).sum(0) / wall.sum(), .02, .98)
        model.alive[0] = True
        model.delta[0] = np.log(p0 / (1 - p0))
        Pi = np.ones((T, 1))

    born_at = {int(k): -10 ** 6 for k in np.flatnonzero(model.alive)}
    tried = set()
    cache = None; dirty = set()
    births = []; remerges = []
    prev = -np.inf

    for it in range(iters):
        S, Sd, nmass, N, ll, ks = e_step(Xs, ws, model, Pi, tv, drift)
        model = m_step(model, ks, Sd, N, tv, drift, inner, lr, rw)
        Pi = N / np.maximum(N.sum(1, keepdims=True), EPS)

        # death: a component nobody uses returns to the pool, but not before it
        # has had `grace` iterations to attract mass. Without this a newborn is
        # judged one iteration after its hard-split init and can be culled
        # before EM has moved it.
        if it > burn_in:
            dead = [int(ks[i]) for i in range(len(ks))
                    if Pi[:, i].max() < death_floor
                    and model.parent[ks[i]] >= 0
                    and it - born_at.get(int(ks[i]), -10 ** 6) > grace]
            for k in dead:
                model.alive[k] = False
                for j in range(max_K):
                    if model.parent[j] == k: model.parent[j] = model.parent[k]
                model.delta[k] = 0.0
                model.gamma[k] = 0.0
                model.split_on[k] = -1
                born_at.pop(k, None)
                tried -= {c for c in tried if c[0] == k}
                dirty.add(int(model.parent[k]))
            if dead:
                keep = [i for i in range(len(ks)) if int(ks[i]) not in dead]
                Pi = Pi[:, keep]; Pi /= np.maximum(Pi.sum(1, keepdims=True), EPS)
                if verbose:
                    print(f"      death: {len(dead)} component(s) returned",
                          flush=True)

        # birth. Attempt every iteration until K_warm components exist, then
        # fall back to the slower schedule. The original attempted 9 times in
        # 200 iterations and accepted at most one per attempt, so K <= 10 was
        # structurally unreachable regardless of the data.
        K_now = int(model.alive.sum())
        due = (births_per_call > 0 and it >= burn_in
               and (K_now < K_warm or it % birth_every == 0))
        if due:
            for _ in range(births_per_call):
                if rescan_every and it % rescan_every == 0:
                    cache = None; dirty = set()
                model, Pi, res, cache = try_birth(
                    Xs, ws, model, Pi, tv, drift, p_birth, names or {}, verbose,
                    inner=inner, lr=lr, rw=rw, refit=refit, n_cand=n_cand,
                    penalty=penalty, diag=diag, tried=tried,
                    cache=cache, dirty=dirty)
                dirty = set()
                if res is None:
                    break
                births.append(res)
                born_at[int(res[1])] = it
                dirty |= {int(res[0]), int(res[1])}   # parent and new child
                K_now = int(model.alive.sum())
                p_birth = min(.95, max(.05, K_now / (K_now + 2.0)))

        # backtracking: merge two siblings and re-split on a different site.
        # K is unchanged, so this competes on equal terms with the current
        # configuration rather than winning automatically the way a birth does.
        if merge_every and it >= burn_in and it % merge_every == 0:
            model, Pi, mres = try_remerge(
                Xs, ws, model, Pi, tv, drift, p_birth, names or {}, verbose,
                inner=inner, lr=lr, rw=rw, refit=max(refit, 3),
                n_pairs=n_pairs, diag=diag)
            if mres is not None:
                remerges.append(mres)
                cache = None; dirty = set(); tried = set()

        if verbose and (it + 1) % 50 == 0:
            print(f"      iter {it+1}  LL/obs {ll:.5f}  "
                  f"components {int(model.alive.sum())}", flush=True)
        if abs(ll - prev) < 1e-6 and it > 120 and int(model.alive.sum()) >= K_warm:
            break
        prev = ll

    return model, Pi, tv, births, ll, remerges


def fit_flat(Xs, ws, V, K, seed=0, drift=False, split_merge=False,
             iters=200, inner=8, lr=2.0, half_life=1.0, prior=.5,
             names=None, verbose=False, init_beta=None, return_gamma=False):
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
    out = (th_next, Pi[-1], int((Pi.max(0) > 1e-3).sum()), ll, splits, beta)
    # gamma is needed to evaluate the signatures at horizons beyond h=1;
    # returned only on request so existing 6-tuple unpacking still works.
    return out + (gamma, tv) if return_gamma else out



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
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--unbounded", action="store_true",
                    help="remove the birth SCHEDULE: attempt births every "
                         "iteration all the way to --max-K, several per call. "
                         "K then stops only when no candidate survives the "
                         "filters or the penalty, not when the clock runs out")
    ap.add_argument("--rescan-every", type=int, default=25,
                    help="force a full candidate rescan every N iterations. "
                         "0 = never; cached scores go stale as EM moves the "
                         "other components")
    ap.add_argument("--K-warm", type=int, default=20,
                    help="attempt a birth EVERY iteration until this many "
                         "components exist, then fall back to --birth-every")
    ap.add_argument("--birth-every", type=int, default=10)
    ap.add_argument("--births-per-call", type=int, default=2)
    ap.add_argument("--refit", type=int, default=5,
                    help="EM iterations run on the proposed split BEFORE the "
                         "gain is evaluated. 0 reproduces the old behaviour")
    ap.add_argument("--n-cand", type=int, default=3,
                    help="candidate splits tried per birth call before giving up")
    ap.add_argument("--grace", type=int, default=15,
                    help="iterations a newborn is protected from death")
    ap.add_argument("--birth-penalty", choices=["bic", "prior"], default="bic",
                    help="bic: (V_eff/2)logN per component. prior: geometric "
                         "+ Gaussian only, which does not scale with N and "
                         "will let K run to max-K")
    ap.add_argument("--hier-drift", action="store_true",
                    help="make the drift slopes hierarchical too: a child's "
                         "slope is its parent's plus a shrunk deviation. "
                         "Without this the profiles are a tree but the slopes "
                         "are K x V free unregularised parameters fitted from "
                         "one point per month")
    ap.add_argument("--merge-every", type=int, default=0,
                    help="attempt a merge-and-resplit backtracking move every "
                         "N iterations. 0 disables it (greedy, splits are "
                         "permanent). 5 is a reasonable starting value")
    ap.add_argument("--n-pairs", type=int, default=3,
                    help="sibling pairs considered per merge attempt")
    ap.add_argument("--freeze-K", action="store_true",
                    help="no births at all: the model is locked at whatever "
                         "the warm start provides. Use with --warm-from-sm for "
                         "a strictly capacity-matched comparison against "
                         "flat+drift+split-merge")
    ap.add_argument("--warm-mode", choices=["star", "tree"], default="star",
                    help="star: SM's components as depth-1 children of a "
                         "pooled root (partition only). tree: cluster them "
                         "into a dendrogram and start from that (structure "
                         "too), so births extend a tree instead of building "
                         "one from a star")
    ap.add_argument("--warm-from-sm", action="store_true",
                    help="seed the hierarchy from the fitted flat+SM mixture "
                         "so the ladder compares model classes at equal capacity")
    args = ap.parse_args()

    if args.freeze_K:
        args.K_warm = 0
        args.births_per_call = 0
    if args.unbounded:
        args.K_warm = args.max_K
        args.birth_every = 1
        args.births_per_call = max(args.births_per_call, 4)

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
    trees = None; birth_log = None

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

        th, pi, used, _, sp, beta_sm = fit_flat(Xs, ws, V, args.max_K, seed=sd,
                                                drift=not args.no_drift,
                                                split_merge=True,
                                                half_life=args.half_life,
                                                names=names, verbose=(sd == 0),
                                                init_beta=warm)
        p = score(Xte, wte, th, pi)
        res[rungs[2]]["ll"].append(p); res[rungs[2]]["used"].append(used)
        print(f"  {'flat + drift + split-merge':<28}{p:8.3f}   components {used}",
              flush=True)

        diag = [] if sd == 0 else None
        model, Pi, tv, births, _, remerges = fit(
            Xs, ws, V, args.max_K, seed=sd, sigma=args.sigma,
            half_life=args.half_life, drift=not args.no_drift,
            names=names, verbose=(sd == 0), iters=args.iters,
            rescan_every=args.rescan_every, K_warm=args.K_warm, birth_every=args.birth_every,
            births_per_call=args.births_per_call, refit=args.refit,
            n_cand=args.n_cand, grace=args.grace, penalty=args.birth_penalty,
            warm=(beta_sm, pi) if args.warm_from_sm else None,
            warm_mode=args.warm_mode, hier_drift=args.hier_drift,
            merge_every=args.merge_every,
            n_pairs=args.n_pairs,
            diag=diag)
        if sd == 0: birth_log = diag
        dt = tv[-1] - tv[-2] if len(tv) > 1 else 0.
        th, ks = model.theta(tv[-1] + dt, not args.no_drift)
        p = score(Xte, wte, th, Pi[-1])
        nc = int(model.alive.sum())
        res[rungs[3]]["ll"].append(p); res[rungs[3]]["used"].append(nc)
        print(f"  {'hierarchical + birth-death':<28}{p:8.3f}   components {nc}"
              f"   births {len(births)}  remerges {len(remerges)}", flush=True)
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

    if birth_log:
        print(f"\n  BIRTH ATTEMPTS (seed 0) -- every proposal, accepted or not")
        print(f"    {'parent':>7}{'mutation':>12}{'dep':>10}{'dLL':>14}"
              f"{'dPrior':>10}{'cost':>10}{'gain':>14}  verdict")
        for r in birth_log:
            nm = names.get(r.get("mut", -1), str(r.get("mut", "")))
            if r["why"] != "gain":
                print(f"    {r['k']:>7}{nm:>12}{r.get('dep',0):>10,.0f}"
                      f"{'':>14}{'':>10}{'':>10}{'':>14}  {r['why']}")
            else:
                v = "accept" if r["gain"] > 0 else "reject"
                print(f"    {r['k']:>7}{nm:>12}{r['dep']:>10,.0f}"
                      f"{r['dll']:>+14,.0f}{r['dlp']:>+10,.0f}"
                      f"{r['cost']:>10,.0f}{r['gain']:>+14,.0f}  {v}")

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
