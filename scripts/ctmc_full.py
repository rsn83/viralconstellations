#!/usr/bin/env python3
"""
CTMC generator over edit operations, fit by EM with an exact predecessor
marginalization.

This is the model the propositions describe, in its simplest honest form.
The earlier `ctmc` class in residue_compare.py was a placeholder: it took the
FIRST valid predecessor and stopped, used a uniform one-month gap, and made
the rate independent of the predecessor -- which makes the marginalization
pointless, since summing over predecessors adds nothing when the rate does
not depend on which one it was.

WHAT IS IMPLEMENTED
-------------------
Predecessor sum.  For each arrival y at month t, all observed x within
Hamming r in the window are candidates.  Which one it actually came from is
latent and summed over, never chosen (Prop 1: no rate between tips; Prop 4:
the observed neighbourhood carries the mass, with error alpha).

Real elapsed time.  Each in-edge carries its own dt = t - t(x).  The rate is
per unit time and enters as q*dt for the arrival and exp(-lambda*dt) for
survival, both inside the likelihood rather than only at prediction.

Mixture for unexplained arrivals.  pi_0 absorbs arrivals with no close
observed predecessor -- recombinants, importations, multi-step jumps.
Measured at ~0.21 at r=2.  Without it the model inflates rates to explain
jumps that never happened as independent single substitutions.

Conditioning, as three nested variants.  This is the corollary made
empirical:
    marginal   q(s)          rate depends on the substitution only
    spike      q(s | x)      ... and on the predecessor's spike sequence
    genome     q(s | x, b)   ... and on its non-spike background
The gap between `spike` and `genome` is what genome context is worth, in
nats.  The gap between `marginal` and `spike` is what the marginalization
itself buys.

Integration.  h months is n applications of the one-step operator at step
h/n, each using the UPDATED distribution, so arrivals at step i become
predecessors at step i+1.  A single jump of size h would make h=6 differ from
h=1 only in an exponent; iterating makes it differ in kind, and reaches
Hamming-n variants naturally.  Support is pruned to the top M by mass after
each step and the discarded mass is reported.

FITTING
-------
EM.  E-step: posterior over which predecessor produced each arrival.  M-step:
rates as weighted counts over exposure -- closed form for `marginal`, a few
gradient steps on the weighted likelihood otherwise.

Usage
-----
  from ctmc_full import CTMCFull
  m = CTMCFull(conditioning='genome', radius=1).fit(counts, bg, train)
  dist = m.predict(counts, T, h=3)
"""

import math
import sys
from collections import defaultdict

import numpy as np


class CTMCFull:

    def __init__(self, conditioning='genome', radius=1, window=2,
                 n_clusters=8, em_iters=6, mstep_iters=25, lr=0.15,
                 l2=1e-3, prior=0.5, prune=50000, steps_per_month=2,
                 pi0=0.21,
                 top_preds=600, top_subs=600, growth=True, seed=0,
                 verbose=True):
        assert conditioning in ('marginal', 'spike', 'genome')
        self.conditioning = conditioning
        self._lam_cache = {}
        self.radius = radius
        self.window = window
        self.n_clusters = n_clusters
        self.em_iters = em_iters
        self.mstep_iters = mstep_iters
        self.lr = lr
        self.l2 = l2
        self.prior = prior
        self.pi0_fixed = pi0
        self.prune = prune
        self.steps_per_month = steps_per_month
        self.top_preds = top_preds
        self.top_subs = top_subs
        self.growth = growth
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------ helpers

    def _log(self, msg):
        if self.verbose:
            print(msg, file=sys.stderr)

    def _cluster_backgrounds(self, bg_profiles):
        """Coarsen non-spike backgrounds into clusters.

        The raw non-spike set is far too large to index a rate table, and the
        corollary only requires enough of the background to restore
        exchangeability -- not all of it.  Whatever the clustering fails to
        capture shows up as residual insufficiency, which is measurable."""
        if not bg_profiles:
            return {}, 0
        keys = list(bg_profiles)
        vocab = defaultdict(int)
        for k in keys:
            for s in bg_profiles[k]:
                vocab[s] += 1
        top = [s for s, _ in sorted(vocab.items(), key=lambda z: -z[1])[:200]]
        idx = {s: i for i, s in enumerate(top)}
        if not top:
            return {k: 0 for k in keys}, 1

        X = np.zeros((len(keys), len(top)), dtype=np.float32)
        for i, k in enumerate(keys):
            for s in bg_profiles[k]:
                if s in idx:
                    X[i, idx[s]] = 1.0

        k = min(self.n_clusters, len(keys))
        rng = np.random.default_rng(self.seed)
        cen = X[rng.choice(len(X), k, replace=False)]
        assign = np.zeros(len(X), dtype=int)
        for _ in range(25):
            d = ((X[:, None, :] - cen[None]) ** 2).sum(-1)
            new = d.argmin(1)
            if (new == assign).all():
                break
            assign = new
            for j in range(k):
                if (assign == j).any():
                    cen[j] = X[assign == j].mean(0)
        return {keys[i]: int(assign[i]) for i in range(len(keys))}, k

    def _features(self, x, sub, cluster):
        """Feature vector for the rate of adding `sub` to predecessor `x`.

        `marginal` uses the substitution alone.  `spike` adds the predecessor
        -- without this the marginalization is pointless, since a rate that
        ignores which predecessor it came from makes the sum over
        predecessors equivalent to frequency weighting.  `genome` adds the
        background cluster, which is what the sufficiency test says is
        missing from spike alone."""
        f = {('sub', sub): 1.0}
        if self.conditioning == 'marginal':
            return f
        # spike context: the predecessor's own substitutions, paired with the
        # one being added.  Restricted to the marker set so the parameter
        # count stays proportional to the event count.
        for m in x:
            if m in self.markers:
                f[('pair', sub, m)] = 1.0
        f[('size', sub)] = len(x) / 50.0
        if self.conditioning == 'genome':
            f[('bg', sub, cluster)] = 1.0
        return f

    def _score(self, feats):
        return sum(self.w.get(k, 0.0) * v for k, v in feats.items())

    def _rate(self, feats):
        return math.exp(min(self._score(feats), 20.0))

    # ---------------------------------------------------------------- fit

    def fit(self, counts, bg_of, train):
        """counts[month][variant] -> int.  bg_of[variant] -> non-spike tuple."""
        self.train = train
        self.months = train

        first = {}
        for m in train:
            for v in counts[m]:
                if v not in first or m < first[v]:
                    first[v] = m
        self.first = first

        # marker positions: the substitutions that most often appear IN a
        # predecessor when an arrival happens.  Chosen on training data only.
        marker_count = defaultdict(int)
        for m in train:
            for v, c in counts[m].items():
                for s in v:
                    marker_count[s] += c
        self.markers = {s for s, _ in
                        sorted(marker_count.items(), key=lambda z: -z[1])[:60]}

        clusters, n_cl = self._cluster_backgrounds(
            {v: bg_of.get(v, ()) for m in train for v in counts[m]}
            if self.conditioning == 'genome' else {})
        self.clusters = clusters
        self.n_clusters_fit = max(n_cl, 1)

        # ---- build the arrival table: one row per arrival, with ALL its
        # candidate in-edges.  This is the DAG in-neighbourhood.
        rows = []
        exposure = defaultdict(float)   # (feature key) -> mass * time
        total_exposure = 0.0

        for t in train:
            win = [m for m in train if t - self.window <= m < t]
            if not win:
                continue
            pf = defaultdict(float)
            latest = {}
            for m in win:
                for v, c in counts[m].items():
                    pf[v] += c
                    latest[v] = max(latest.get(v, m), m)
            tot = sum(pf.values())
            if tot <= 0:
                continue
            pnorm = {v: c / tot for v, c in pf.items()}
            preds = dict(sorted(pnorm.items(), key=lambda z: -z[1])
                         [:self.top_preds])
            wset = set(preds)

            for y in counts[t]:
                if first.get(y) != t:
                    continue
                cands = []
                # radius 1: drop one substitution
                for s in y:
                    x = tuple(v for v in y if v != s)
                    if x in wset:
                        cands.append((x, (s,), t - latest[x]))
                if self.radius >= 2:
                    ys = list(y)
                    for i in range(len(ys)):
                        for j in range(i + 1, len(ys)):
                            x = tuple(v for v in ys
                                      if v != ys[i] and v != ys[j])
                            if x in wset:
                                cands.append((x, (ys[i], ys[j]),
                                              t - latest[x]))
                if cands:
                    rows.append((y, cands, preds))

            # exposure: predecessor mass integrated over one month
            total_exposure += 1.0

        self._log(f'  arrivals with >=1 candidate predecessor: {len(rows):,}')
        if not rows:
            raise RuntimeError('no arrivals with candidate predecessors')

        # ---- initialise weights from marginal counts
        # Every substitution in the vocabulary must be initialised, not just
        # the ones with observed arrivals.  A missing key falls through
        # defaultdict to 0.0, and exp(0) = 1.0 -- so hundreds of unseen
        # substitutions each contributed rate 1.0 and lambda ran to ~171 per
        # month instead of ~0.3.  That killed survival and made every
        # conditioning identical.
        self.w = defaultdict(float)
        base = defaultdict(float)
        for s_ in self.sub_vocab:
            base[s_] = 0.0
        for y, cands, preds in rows:
            for x, subs, dt in cands:
                for s in subs:
                    base[s] += 1.0 / len(cands)
        for s, c in base.items():
            self.w[('sub', s)] = math.log(
                max((c + self.prior) / max(total_exposure, 1.0), 1e-9))

        # Rescale so the total escape rate matches what the data actually
        # shows.  Without this lambda = sum_s q(s) runs to hundreds per month,
        # exp(-lambda*dt) underflows, every predecessor term vanishes and pi0
        # saturates -- which makes all conditionings identical because all of
        # them are effectively zero.
        # lambda is a PER-LINEAGE escape rate, so the target is arrivals per
        # circulating lineage per month -- not arrivals per unit population
        # mass, which is larger by the number of distinct lineages and drives
        # exp(-lambda*dt) to underflow.
        n_lineages = max(np.mean([len(counts[m]) for m in train]), 1.0)
        target = len(rows) / (max(total_exposure, 1.0) * n_lineages)
        cur = sum(math.exp(self.w[('sub', s)]) for s in self.sub_vocab
                  if ('sub', s) in self.w)
        if cur > 0:
            shift = math.log(max(target, 1e-6) / cur)
            for k in list(self.w):
                if k[0] == 'sub':
                    self.w[k] += shift
        self.target_lambda = target
        self._log(f'  total escape rate {target:.4f} / lineage / month '
                  f'({n_lineages:.0f} lineages)')
        # pi0 is FIXED at the measured unexplained-arrival rate rather than
        # fitted.  Fitting it alongside the background density is degenerate:
        # the two trade off against each other and pi0 runs to its cap,
        # zeroing every predecessor term and making all conditionings
        # identical.  The coverage diagnostic already measured this (~0.21 of
        # arrivals have no observed predecessor within r=2), so use it.
        self.pi0 = self.pi0_fixed
        # background density on the scale of a typical predecessor term, so
        # the mixture components are comparable
        probe = []
        for y, cands, preds in rows[:400]:
            for x, subs, dt in cands:
                probe.append(preds.get(x, 0.0)
                             * math.exp(sum(self.w.get(('sub', s), -20.0)
                                            for s in subs)))
        self.bg_density = (float(np.median(probe)) if probe else 1e-9) or 1e-9

        # ---- EM
        for it in range(self.em_iters):
            # E-step: posterior over which predecessor produced each arrival
            resp = []
            ll = 0.0
            unexplained = 0.0
            for y, cands, preds in rows:
                terms = []
                for x, subs, dt in cands:
                    dt = max(dt, 0.5)
                    cl = self.clusters.get(x, 0)
                    logq = 0.0
                    for s in subs:
                        logq += self._score(self._features(x, s, cl))
                    lam = self._lambda(x, cl)
                    # arrival x -> y in dt, times survival of x until then
                    val = (preds.get(x, 0.0) * math.exp(min(logq, 10.0))
                           * dt * math.exp(-min(lam * dt, 30.0)))
                    terms.append(val)
                z = sum(terms) * (1 - self.pi0) + self.pi0 * self.bg_density
                if z <= 0:
                    continue
                ll += math.log(z)
                unexplained += self.pi0 * self.bg_density / z
                scale = (1 - self.pi0) / z
                resp.append([(cands[i], terms[i] * scale)
                             for i in range(len(cands)) if terms[i] > 0])

            self._log(f'  EM {it + 1}/{self.em_iters}  '
                      f'loglik/arrival {ll / max(len(rows), 1):+.4f}  '
                      f'unexplained {unexplained / max(len(rows), 1):.3f}')

            # M-step: weighted maximum likelihood for the rate parameters
            self._mstep(resp, rows, total_exposure)

        # ---- growth, fit separately: proposal and selection stay apart
        self.f = {}
        if self.growth:
            recent = train[-6:]
            tot = {m: sum(counts[m].values()) for m in recent}
            pts = defaultdict(list)
            for i, m in enumerate(recent):
                for v, c in counts[m].items():
                    if c >= 3:
                        pts[v].append((i, math.log(c / max(tot[m], 1))))
            for v, ps in pts.items():
                if len(ps) < 3:
                    continue
                sl = np.polyfit([p[0] for p in ps], [p[1] for p in ps], 1)[0]
                # shrink: six noisy points over-fit badly
                self.f[v] = float(sl) * len(ps) / (len(ps) + 4.0)

        return self

    def _lambda(self, x, cl):
        """Total escape rate.  Persistence is the complement of this, not a
        separate prediction -- which is why a rate model cannot collapse to
        copying the way sequence-to-sequence MLE does."""
        key = ('lam', x)
        if key in self._lam_cache:
            return self._lam_cache[key]
        xs = set(x)
        tot = 0.0
        for s in self.sub_vocab:
            if s in xs:
                continue
            tot += self._rate(self._features(x, s, cl))
        self._lam_cache[key] = tot
        return tot

    def _mstep(self, resp, rows, total_exposure):
        """Weighted gradient ascent on the responsibilities.  Closed form is
        available only for `marginal`; the conditioned variants need a few
        steps, which is cheap since the feature vectors are sparse."""
        for _ in range(self.mstep_iters):
            grad = defaultdict(float)
            for arrival in resp:
                for (x, subs, dt), r in arrival:
                    if r <= 0:
                        continue
                    dt = max(dt, 0.5)
                    cl = self.clusters.get(x, 0)
                    for s in subs:
                        f = self._features(x, s, cl)
                        for k, v in f.items():
                            grad[k] += r * v
                        rate = self._rate(f)
                        for k, v in f.items():
                            grad[k] -= r * rate * dt * v
            for k, g in grad.items():
                self.w[k] += self.lr * (g / max(len(rows), 1)
                                        - self.l2 * self.w[k])
            self._lam_cache = {}

    # ------------------------------------------------------------ predict

    def predict(self, counts, T, h):
        """Integrate the forward equation to T+h.

        h is applied as `steps_per_month * h` applications of the one-step
        operator, each on the UPDATED distribution.  A single jump would make
        h=6 differ from h=1 only in an exponent; iterating makes arrivals at
        one step act as predecessors at the next, which is what multi-step
        forecasting actually involves, and reaches multi-substitution
        variants without ever enumerating them."""
        win = [m for m in self.train if T - self.window < m <= T]
        p = defaultdict(float)
        for m in win:
            for v, c in counts[m].items():
                p[v] += c
        tot = sum(p.values())
        if tot <= 0:
            return {}, 0.0
        p = {v: c / tot for v, c in p.items()}

        n_steps = max(1, int(round(self.steps_per_month * h)))
        dt = h / n_steps
        discarded = 0.0
        self._lam_cache = {}

        for step in range(n_steps):
            cur = dict(sorted(p.items(), key=lambda z: -z[1])[:self.top_preds])
            zc = sum(cur.values())
            if zc <= 0:
                break
            out = defaultdict(float)
            # mass on variants outside the active set simply persists
            for v, m in p.items():
                if v not in cur:
                    out[v] += m
            for x, px in cur.items():
                cl = self.clusters.get(x, 0)
                g = math.exp(self.f.get(x, 0.0) * dt) if self.growth else 1.0
                lam = self._lambda(x, cl)
                surv = math.exp(-lam * dt)
                out[x] += px * g * surv
                xs = set(x)
                for s in self.sub_vocab:
                    if s in xs:
                        continue
                    r = self._rate(self._features(x, s, cl))
                    pr = 1.0 - math.exp(-r * dt)
                    if pr <= 1e-12:
                        continue
                    out[tuple(sorted(xs | {s}))] += px * g * pr
            # prune and record what was thrown away, so the truncation is a
            # measured approximation rather than a silent one
            if len(out) > self.prune:
                allm = sum(out.values())
                keep = sorted(out.items(), key=lambda z: -z[1])[:self.prune]
                kept = sum(v for _, v in keep)
                if allm > 0:
                    discarded += max(0.0, (allm - kept) / allm)
                out = dict(keep)
            z = sum(out.values())
            p = {v: c / z for v, c in out.items()} if z > 0 else {}

        return p, discarded

    def set_vocab(self, subs):
        self.sub_vocab = list(subs)[:self.top_subs]
        return self
