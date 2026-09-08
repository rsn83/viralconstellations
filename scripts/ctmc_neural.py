#!/usr/bin/env python3
"""
Neural generator over edit operations, trained by direct gradient through the
predecessor-marginalized likelihood.

This is the version the propositions describe with a learned rate function
instead of a table.  The tabular EM model conditions on a background cluster
index and a handful of marker pairs; here the rate is a function of the
ACTUAL predecessor residue sequence, so nothing is coarsened away.

    q_theta(s | x, b) = exp( head_s( encode(x_residues, b) ) )

WHY GRADIENT AND NOT EM
-----------------------
EM was the right choice for a table because the M-step is closed form there.
With a network there is no closed form, so the E-step buys nothing -- and the
latent predecessor sum is differentiable, so you can backpropagate straight
through the log-sum-exp.  That is both simpler and tighter than alternating:
no responsibilities to hold fixed, no risk of the E-step and M-step
disagreeing about scale, which is what produced the pi0 degeneracy in the
tabular fit.

THE LIKELIHOOD
--------------
For each arrival y_j at month t_j, over its DAG in-neighbourhood N_r(y_j):

  L = -sum_j log[ (1-pi0) * sum_k p_t(x_k) q(s_k|x_k) dt_k exp(-lambda_k dt_k)
                  + pi0 * rho ]
      - sum_persist lambda_x dt

Every term is per-edge and uses the real elapsed time dt_k = t_j - t(x_k),
not a uniform month.  lambda_k is the row sum over the whole substitution
vocabulary, so persistence is the complement of the escape rate rather than a
separate prediction -- which is why this cannot collapse to copying.

Prediction is the same iterated forward integration as the tabular model:
`steps_per_month * h` applications of the one-step operator, each on the
updated distribution.

Requires torch.
"""

import math
import sys
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn


AA = 'ACDEFGHIKLMNPQRSTVWY*-X'
AA_IDX = {c: i for i, c in enumerate(AA)}


class RateNet(nn.Module):
    """Encodes a predecessor's residue string plus its non-spike background,
    and emits a log-rate for every substitution in the vocabulary."""

    def __init__(self, n_pos, n_sub, n_bg, emb=24, hidden=256):
        super().__init__()
        self.res_emb = nn.Embedding(len(AA), emb)
        self.pos_emb = nn.Parameter(torch.randn(n_pos, emb) * 0.02)
        self.bg = nn.Linear(n_bg, 64) if n_bg > 0 else None
        d = emb + (64 if n_bg > 0 else 0)
        self.trunk = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, n_sub)
        # start every rate small; a large initial lambda underflows the
        # survival term and stalls training before it begins
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -8.0)

    def forward(self, res_idx, bg_vec):
        # res_idx: (B, n_pos) residue indices; mean-pool position-aware
        # embeddings so the encoder sees the whole sequence, not a summary
        e = self.res_emb(res_idx) + self.pos_emb.unsqueeze(0)
        h = e.mean(1)
        if self.bg is not None:
            h = torch.cat([h, torch.relu(self.bg(bg_vec))], -1)
        return self.head(self.trunk(h))          # log-rates, (B, n_sub)


class CTMCNeural:

    def __init__(self, radius=1, window=2, epochs=25, lr=3e-3, batch=64,
                 pi0=0.21, prune=50000, steps_per_month=2, top_preds=600,
                 top_subs=400, growth=True, use_bg=True, n_bg_feats=200,
                 seed=0, device='cpu', verbose=True):
        self.radius, self.window = radius, window
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.pi0, self.prune = pi0, prune
        self.steps_per_month = steps_per_month
        self.top_preds, self.top_subs = top_preds, top_subs
        self.growth, self.use_bg = growth, use_bg
        self.n_bg_feats = n_bg_feats
        self.seed, self.device, self.verbose = seed, device, verbose

    def _log(self, m):
        if self.verbose:
            print(m, file=sys.stderr)

    # ------------------------------------------------------------ encoding

    def _res_idx(self, variant):
        d = dict(variant)
        return [AA_IDX.get(d.get(p, self.reference[p - 1]
                                 if p <= len(self.reference) else 'X'),
                           AA_IDX['X'])
                for p in self.positions]

    def _bg_vec(self, variant):
        v = np.zeros(len(self.bg_feats), dtype=np.float32)
        if not self.use_bg:
            return v
        for s in self.bg_of.get(variant, ()):
            i = self.bg_idx.get(s)
            if i is not None:
                v[i] = 1.0
        return v

    # ----------------------------------------------------------------- fit

    def fit(self, counts, bg_of, train, positions, reference, sub_vocab):
        torch.manual_seed(self.seed)
        self.positions, self.reference = positions, reference
        self.bg_of, self.train = bg_of, train
        self.sub_vocab = list(sub_vocab)[:self.top_subs]
        self.sub_pos = {s: i for i, s in enumerate(self.sub_vocab)}

        bgc = defaultdict(int)
        if self.use_bg:
            for m in train:
                for v, c in counts[m].items():
                    for s in bg_of.get(v, ()):
                        bgc[s] += c
        self.bg_feats = [s for s, _ in
                         sorted(bgc.items(), key=lambda z: -z[1])
                         [:self.n_bg_feats]]
        self.bg_idx = {s: i for i, s in enumerate(self.bg_feats)}

        first = {}
        for m in train:
            for v in counts[m]:
                if v not in first or m < first[v]:
                    first[v] = m

        # ---- assemble the DAG in-edges, one group per arrival
        groups = []
        for t in train:
            win = [m for m in train if t - self.window <= m < t]
            if not win:
                continue
            pf, latest = defaultdict(float), {}
            for m in win:
                for v, c in counts[m].items():
                    pf[v] += c
                    latest[v] = max(latest.get(v, m), m)
            tot = sum(pf.values())
            if tot <= 0:
                continue
            preds = dict(sorted(((v, c / tot) for v, c in pf.items()),
                                key=lambda z: -z[1])[:self.top_preds])
            wset = set(preds)
            for y in counts[t]:
                if first.get(y) != t:
                    continue
                edges = []
                for s in y:
                    x = tuple(v for v in y if v != s)
                    if x in wset and s in self.sub_pos:
                        edges.append((x, [s], max(t - latest[x], 0.5), None))
                if self.radius >= 2:
                    ys = list(y)
                    for i in range(len(ys)):
                        for j in range(i + 1, len(ys)):
                            x = tuple(v for v in ys
                                      if v != ys[i] and v != ys[j])
                            if (x in wset and ys[i] in self.sub_pos
                                    and ys[j] in self.sub_pos):
                                # store the two intermediates as well: each
                                # ordering routes through a different one, and
                                # the second rate must condition on the state
                                # it acts on, not on x
                                xs2 = set(x)
                                edges.append((x, [ys[i], ys[j]],
                                              max(t - latest[x], 0.5),
                                              (tuple(sorted(xs2 | {ys[i]})),
                                               tuple(sorted(xs2 | {ys[j]})))))
                if edges:
                    groups.append((edges, preds))
        n1 = sum(1 for g, _ in groups for e in g if e[3] is None)
        n2 = sum(1 for g, _ in groups for e in g if e[3] is not None)
        deg = np.mean([len(g) for g, _ in groups]) if groups else 0.0
        self._log(f'  arrivals with in-edges: {len(groups):,}  '
                  f'(edges: {n1:,} one-hop, {n2:,} two-hop; '
                  f'mean in-degree {deg:.2f})')
        if self.radius >= 2 and n2 == 0:
            self._log('  WARNING: radius=2 requested but no two-hop edges '
                      'found; results will equal radius=1')
        if not groups:
            raise RuntimeError('no arrivals with candidate predecessors')

        # cache encodings for every distinct predecessor
        uniq = set()
        for g, _ in groups:
            for e in g:
                uniq.add(e[0])
                if e[3] is not None:
                    uniq.update(e[3])
        uniq = sorted(uniq)
        self.pred_cache = {x: i for i, x in enumerate(uniq)}
        R = torch.tensor([self._res_idx(x) for x in uniq], dtype=torch.long)
        B = torch.tensor(np.stack([self._bg_vec(x) for x in uniq]))

        self.net = RateNet(len(positions), len(self.sub_vocab),
                           len(self.bg_feats)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        log_pi0 = math.log(max(self.pi0, 1e-6))
        log_rho = math.log(1.0 / max(len(groups), 1))

        for ep in range(self.epochs):
            perm = np.random.default_rng(ep).permutation(len(groups))
            tot_loss, nb = 0.0, 0
            for i in range(0, len(groups), self.batch):
                chunk = [groups[j] for j in perm[i:i + self.batch]]
                need = set()
                for g, _ in chunk:
                    for e in g:
                        need.add(self.pred_cache[e[0]])
                        if e[3] is not None:
                            for mstate in e[3]:
                                need.add(self.pred_cache[mstate])
                need = sorted(need)
                remap = {v: k for k, v in enumerate(need)}
                lr_all = self.net(R[need].to(self.device),
                                  B[need].to(self.device))
                # row sum over the vocabulary = escape rate.  Restricted to
                # top_subs, so this is a lower bound on the true escape rate;
                # the omitted substitutions are individually rare but their
                # absence biases survival slightly upward.
                lam = torch.exp(lr_all).sum(1)

                terms = []
                for edges, preds in chunk:
                    vals = []
                    for x, subs, dt, mids in edges:
                        r = remap[self.pred_cache[x]]
                        base = math.log(max(preds.get(x, 1e-12), 1e-12))
                        if mids is None:
                            # one hop: q * dt * exp(-lambda*dt)
                            vals.append(base
                                        + lr_all[r, self.sub_pos[subs[0]]]
                                        + math.log(dt) - lam[r] * dt)
                        else:
                            # Two hops are not a product of two rates.  The
                            # order is unobserved, the second substitution
                            # acts on the intermediate (different rate,
                            # different escape rate), and the time of the
                            # first arrival is latent.  Sum the two orderings
                            # and integrate that time out in closed form.
                            parts = []
                            for oi, (a, b) in enumerate(
                                    ((subs[0], subs[1]), (subs[1], subs[0]))):
                                mid = mids[oi]
                                rm = remap[self.pred_cache[mid]]
                                lq1 = lr_all[r, self.sub_pos[a]]
                                lq2 = lr_all[rm, self.sub_pos[b]]
                                la, lb = lam[r], lam[rm]
                                d = la - lb
                                safe = torch.where(
                                    d.abs() < 1e-6,
                                    torch.log(torch.tensor(dt)) - lb * dt,
                                    torch.log(torch.clamp(
                                        1 - torch.exp(-torch.clamp(
                                            d * dt, -30, 30)), min=1e-30))
                                    - torch.log(torch.clamp(d.abs(), min=1e-9))
                                    - lb * dt)
                                parts.append(lq1 + lq2 + safe)
                            vals.append(base + torch.logsumexp(
                                torch.stack(parts), 0))
                    v = torch.stack(vals)
                    mix = torch.logaddexp(
                        torch.log(torch.tensor(1.0 - self.pi0))
                        + torch.logsumexp(v, 0),
                        torch.tensor(log_pi0 + log_rho))
                    terms.append(mix)
                loss = -torch.stack(terms).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot_loss += loss.item()
                nb += 1
            if self.verbose and (ep % 5 == 0 or ep == self.epochs - 1):
                with torch.no_grad():
                    ml = torch.exp(self.net(R[:256].to(self.device),
                                            B[:256].to(self.device))).sum(1)
                self._log(f'    epoch {ep + 1}/{self.epochs} '
                          f'loss {tot_loss / max(nb, 1):.4f}  '
                          f'mean lambda {ml.mean().item():.4f}/month')

        # growth stays a separate stage: proposal and selection do not mix
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
                self.f[v] = float(sl) * len(ps) / (len(ps) + 4.0)
        return self

    # ------------------------------------------------------------- predict

    @torch.no_grad()
    def predict(self, counts, T, h):
        win = [m for m in self.train if T - self.window < m <= T]
        p = defaultdict(float)
        for m in win:
            for v, c in counts[m].items():
                p[v] += c
        tot = sum(p.values())
        if tot <= 0:
            return {}, 0.0
        p = {v: c / tot for v, c in p.items()}

        # the one-step operator applies a single substitution, so reaching
        # radius-r variants within a month needs at least r substeps per
        # month; otherwise --full-radius 2 would change training only and
        # leave generation inconsistent with it
        spm = max(self.steps_per_month, self.radius)
        n_steps = max(1, int(round(spm * h)))
        dt = h / n_steps
        discarded = 0.0

        for _ in range(n_steps):
            cur = dict(sorted(p.items(), key=lambda z: -z[1])[:self.top_preds])
            if not cur:
                break
            xs = list(cur)
            R = torch.tensor([self._res_idx(x) for x in xs], dtype=torch.long)
            Bv = torch.tensor(np.stack([self._bg_vec(x) for x in xs]))
            rates = torch.exp(self.net(R.to(self.device),
                                       Bv.to(self.device))).cpu().numpy()
            lam = rates.sum(1)

            out = defaultdict(float)
            for v, m in p.items():
                if v not in cur:
                    out[v] += m
            for i, x in enumerate(xs):
                px = cur[x]
                g = math.exp(self.f.get(x, 0.0) * dt) if self.growth else 1.0
                out[x] += px * g * math.exp(-min(lam[i] * dt, 30.0))
                sx = set(x)
                pr = 1.0 - np.exp(-rates[i] * dt)
                for j, s in enumerate(self.sub_vocab):
                    if pr[j] <= 1e-12 or s in sx:
                        continue
                    out[tuple(sorted(sx | {s}))] += px * g * float(pr[j])

            # first-order arrival mass can exceed the source mass if rates
            # are large for the step size; renormalising afterwards would
            # hide that, so check it explicitly
            gross = sum(out.values())
            if gross > 1.5:
                self._log(f'  WARNING: step mass {gross:.2f} >> 1; '
                          f'reduce dt (raise --steps-per-month)')
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
