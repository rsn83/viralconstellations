#!/usr/bin/env python3
"""
163_dotprod.py -- Population transition model with exhaustive radius-1 scoring.

DESIGN
------
For each circulating variant B at time T, we want to score every possible
one-hop neighbour (add one mutation, or remove one). With V=5172 mutations
this is exhaustive coverage of radius-1.

The key insight: if the scorer factors as a dot product,
    score(B + {m}) = z_B · v_m
then scoring ALL V additions for ALL circulating variants is one matmul:
    Z @ table.T   shape (n_circ, V)

No enumeration, no sampling, no radius parameter. One operation covers
every possible addition. Same for removals.

This is fully backpropagatable: gradient flows from the loss through the
softmax through the matmul into z_B and v_m, which come from node memory
and neighbourhood attention. The whole chain trains end to end.

WHAT IS PREDICTED
-----------------
One distribution over variants at T+h, scored by weighted Jaccard / overlap
against the observed population. Persistence is the zero-init default.

THREE AXES OF REPRESENTATION
-----------------------------
1. Temporal (vertical):   GRU memory per mutation, updated every day
2. Cross-set (horizontal): sparse attention over top-K mutations by mass,
                           seeing which variants they currently co-occur in
3. Within-set:            the dot product itself -- z_B encodes the background
                          as a whole, v_m is the mutation being added

WHAT IS MISSING (accepted tradeoffs)
-------------------------------------
- Radius-2: add-two, swap. 159 says radius-1 covers 84-91% at 1 month.
- Full epistasis: z_B is a pooled background, not member-by-member.
- Recombination: not reachable by local perturbation.
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch

_DEVICE = torch.device('cpu')  # set in run()

def _t(*args, **kw):
    return torch.tensor(*args, **kw).to(_DEVICE)
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# DATA
# ======================================================================

def load_events(path, vocab_path=None, verbose=True):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if ln == 0 and not parts[0][:4].isdigit():
                continue
            date, muts = parts[0].strip(), parts[1].strip()
            cnt = float(parts[2]) if len(parts) > 2 else 1.0
            s = frozenset(int(x) for x in muts.split(",") if x)
            if s:
                rows.append((date, s, cnt))
    rows.sort(key=lambda r: r[0])
    days = sorted({r[0] for r in rows})
    day_ix = {d: i for i, d in enumerate(days)}
    events = [(s, day_ix[d], w) for d, s, w in rows]
    V = 1 + max(max(s) for s, _, _ in events if s)
    if verbose:
        sizes = [len(s) for s, _, _ in events]
        print(f"events {len(events):,}  days {len(days)}"
              f"  ({days[0]}..{days[-1]})  V {V}")
        print(f"variant size: median {int(np.median(sizes))} "
              f"[{min(sizes)}-{max(sizes)}]  "
              f"unique {len({s for s,_,_ in events}):,}")
    return {"events": events, "days": days, "V": V}


def parse_posres(vocab_path, V):
    import re
    names = {}
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path) as f:
            for i, line in enumerate(f):
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[0].isdigit():
                    names[int(p[0])] = p[1]
                else:
                    names[i] = p[0]
    pat = re.compile(r"(\d+)")
    pos, res, n_ok = [], [], 0
    for i in range(V):
        nm = str(names.get(i, i)).split(":")[-1].strip()
        m = pat.search(nm)
        if m:
            pos.append(int(m.group(1)))
            tail = nm[m.end():].strip()
            res.append(tail[:3].upper() if tail else
                       ("DEL" if "del" in nm.lower() else "?"))
            n_ok += 1
        else:
            pos.append(-(i + 1)); res.append("?")
    up = {p: k for k, p in enumerate(sorted(set(pos)))}
    ur = {c: k for k, c in enumerate(sorted(set(res)))}
    return (torch.tensor([up[p] for p in pos]),
            torch.tensor([ur[c] for c in res]),
            len(up), len(ur), n_ok)


def load_first_seen(vocab_path, V, cutoff):
    seen = torch.ones(V, dtype=torch.bool)
    if not (vocab_path and os.path.exists(vocab_path) and cutoff):
        return seen
    n_late = 0
    with open(vocab_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[0].isdigit():
                i = int(p[0])
                if i < V and p[2] > cutoff:
                    seen[i] = False; n_late += 1
    print(f"  {n_late:,} mutations first appear after {cutoff}; "
          "memory suppressed, posres only")
    return seen



def to(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [to(i, device) for i in x]
    return x

def group_by_day(events):
    by_day = defaultdict(list)
    for s, t, w in events:
        by_day[t].append((s, w))
    return by_day


def mass_by_day(by_day):
    out = {}
    for t, batch in by_day.items():
        agg = defaultdict(float)
        for s, w in batch:
            agg[s] += w
        tot = sum(agg.values()) or 1.0
        out[t] = {s: v / tot for s, v in agg.items()}
    return out


# ======================================================================
# MODEL
# ======================================================================

class FourierTime(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(
            1.0 / 10 ** np.linspace(0, 3, d)).float())
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, dt):
        return torch.cos(dt.unsqueeze(-1) * self.w + self.b)


class Memory(nn.Module):
    def __init__(self, V, d, msg_dim, decay=True):
        super().__init__()
        self.V, self.d = V, d
        self.gru = nn.GRUCell(msg_dim, d)
        self.decay = decay
        self.log_gamma = nn.Parameter(torch.tensor(-3.0))
        self.register_buffer("mem", torch.zeros(V, d))
        self.register_buffer("last_t", torch.zeros(V))

    def reset(self):
        self.mem.zero_(); self.last_t.zero_()

    def read(self, idx, t_now):
        idx = idx.cpu() if hasattr(idx,'device') else idx
        m = self.mem[idx]
        if self.decay:
            dt = (t_now - self.last_t[idx]).clamp_min(0.0)
            gamma = torch.sigmoid(self.log_gamma) * 0.5 + 0.5
            m = m * gamma.pow(dt).unsqueeze(-1)
        return m

    @torch.no_grad()
    def write(self, idx, new_mem, t_now):
        self.mem[idx] = new_mem.detach()
        self.last_t[idx] = t_now

    def update(self, idx, msg, t_now):
        cur = self.read(idx, t_now)
        new = self.gru(msg, cur)
        self.write(idx, new, t_now)
        return new


class NeighbourhoodAttn(nn.Module):
    def __init__(self, d, heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(2 * d, heads, batch_first=True)
        self.proj = nn.Linear(2 * d, d)

    def forward(self, q, ctx, mask):
        out, _ = self.attn(q.unsqueeze(1), ctx, ctx,
                           key_padding_mask=mask, need_weights=False)
        return self.proj(out.squeeze(1))


class PosResEmbed(nn.Module):
    def __init__(self, pos_id, res_id, n_pos, n_res, d):
        super().__init__()
        self.register_buffer("pos_id", pos_id)
        self.register_buffer("res_id", res_id)
        self.pos = nn.Embedding(n_pos, d)
        self.res = nn.Embedding(n_res, d)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.res.weight, std=0.02)

    def forward(self, idx):
        idx = idx.to(self.pos_id.device)
        return self.pos(self.pos_id[idx]) + self.res(self.res_id[idx])


class TimeMixer(nn.Module):
    """SFCNTSP temporal mixer over the last M population states."""
    def __init__(self, d, M=8):
        super().__init__()
        self.M = M
        self.W = nn.Parameter(torch.zeros(M))
        with torch.no_grad():
            self.W[-1] = 1.0
        self.register_buffer("buf", torch.zeros(M, d))
        self.register_buffer("n", torch.zeros(1))

    def reset(self):
        self.buf.zero_(); self.n.zero_()

    def push(self, u):
        with torch.no_grad():
            self.buf = torch.roll(self.buf, -1, dims=0)
            self.buf[-1] = u.detach()
            self.n += 1

    def forward(self, u_now):
        if float(self.n) < 2:
            return u_now
        w = torch.softmax(self.W, dim=0)
        return u_now + (w.unsqueeze(-1) * self.buf).sum(0)


class PopEncoder(nn.Module):
    def __init__(self, d, heads=2):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(d, d), nn.Tanh())

    def forward(self, X, w):
        u = (w.to(X.device).unsqueeze(-1) * X).sum(0)
        return self.out(torch.nan_to_num(u))


class TransitionModel(nn.Module):
    """
    Node representations + dot-product transition scorer.

    The scorer factors as z_B · v_m, so ALL V additions for ALL circulating
    variants are covered by one matmul. No enumeration, no sampling.
    """

    def __init__(self, V, d=64, heads=2, n_recent=10, posres=None,
                 decay=True, M=8):
        super().__init__()
        self.V, self.d, self.N = V, d, n_recent
        self.time = FourierTime(d)
        self.mem = Memory(V, d, 3 * d, decay=decay)
        self.nbr = NeighbourhoodAttn(d, heads)
        self.posres = posres
        self.pop_enc = PopEncoder(d, heads)
        self.mixer = TimeMixer(d, M)
        # background encoder: maps a variant to z_B
        self.bg_enc = nn.Sequential(
            nn.Linear(d + 1, 2 * d), nn.Tanh(), nn.Linear(2 * d, d))
        # horizon-dependent fitness for existing variants
        self.fitness = nn.Sequential(
            nn.Linear(d + 1, 2 * d), nn.Tanh(), nn.Linear(2 * d, 1))
        nn.init.zeros_(self.fitness[-1].weight)
        nn.init.zeros_(self.fitness[-1].bias)
        # mass budget for new variants
        self.budget = nn.Sequential(nn.Linear(1, 8), nn.Tanh(),
                                    nn.Linear(8, 1))
        nn.init.zeros_(self.budget[-1].weight)
        self.new_bias = nn.Parameter(torch.tensor(-3.0))
        self.W_s = nn.Linear(d, d)
        self.W_n = nn.Linear(d, d)
        self.b_v = nn.Parameter(torch.zeros(d))
        self.register_buffer("nbr_vec", torch.zeros(V, n_recent, d))
        self.register_buffer("nbr_t", torch.zeros(V, n_recent))
        self.register_buffer("nbr_cnt", torch.zeros(V, dtype=torch.long))
        self.register_buffer("mem_ok", torch.ones(V, 1))
        self._cache_t = None
        self._cache_v = None
        self._pending = None
        self._override = None

    def reset_state(self):
        self.mem.reset(); self.mixer.reset()
        self.nbr_vec.zero_(); self.nbr_t.zero_(); self.nbr_cnt.zero_()
        self._cache_t = None; self._cache_v = None
        self._pending = None; self._override = None

    def flush_pending(self, t_now):
        self._override = None
        if self._pending is None:
            return
        idx, msg = self._pending
        new = self.mem.update(idx, msg, t_now)
        self._override = (idx, new)
        self._pending = None
        self._cache_t = None

    def all_node_repr(self, t_now):
        idx = torch.arange(self.V, device=_DEVICE)
        m = self.mem.read(idx, t_now)
        if self._override is not None:
            oi, ov = self._override
            m = m.index_copy(0, oi, ov)
        m = m * self.mem_ok
        dt = (t_now - self.nbr_t).clamp_min(0.0)
        ctx = torch.cat([self.nbr_vec, self.time(dt)], dim=-1)
        ar = torch.arange(self.N, device=_DEVICE).unsqueeze(0)
        mask = ar >= self.nbr_cnt.clamp(max=self.N).unsqueeze(1)
        mask[mask.all(dim=1), 0] = False
        q = torch.cat([m, self.time(torch.zeros(self.V, device=_DEVICE))], dim=-1)
        nb = self.nbr(q, ctx, mask)
        v = self.W_s(m) + self.W_n(nb)
        if self.posres is not None:
            v = v + self.posres(idx)
        return torch.tanh(v + self.b_v)

    def node_repr_cached(self, t_now, rebuild=False):
        if rebuild or self._cache_t != t_now or self._cache_v is None:
            self._cache_v = self.all_node_repr(t_now)
            self._cache_t = t_now
        return self._cache_v

    def observe(self, variants, t_now, max_k=64):
        seen = sorted({m for s in variants for m in s})
        if not seen:
            return
        V_rep = self.node_repr_cached(t_now)
        idx = torch.tensor(seen, dtype=torch.long)  # CPU: indexes memory buffers
        v = V_rep[idx.to(_DEVICE)]
        pos = {m: i for i, m in enumerate(seen)}
        agg = torch.zeros(len(seen), self.d)
        cnt = torch.zeros(len(seen), 1)
        for s in variants:
            ms = list(s)[:max_k]
            if not ms:
                continue
            rows = _t([pos[m] for m in ms], dtype=torch.long)
            ctx = V_rep[_t(ms, dtype=torch.long)].mean(0, keepdim=True)
            agg.index_add_(0, rows, ctx.expand(len(rows), -1))
            cnt.index_add_(0, rows, torch.ones(len(rows), 1, device=_DEVICE))
        agg = agg / cnt.clamp_min(1.0)
        dt = (t_now - self.mem.last_t[idx]).clamp_min(0.0)
        msg = torch.cat([v, agg, self.time(dt)], dim=-1)
        self._pending = (idx, msg.detach())
        with torch.no_grad():
            slot = (self.nbr_cnt[idx] % self.N).cpu()
            self.nbr_vec[idx.cpu(), slot] = agg.detach().cpu()
            self.nbr_t[idx.cpu(), slot] = t_now
            self.nbr_cnt[idx.cpu()] += 1
        self._cache_t = None

    def background_repr(self, variants, t_now, dt):
        """z_B for each background: pooled member repr + dt."""
        table = self.node_repr_cached(t_now)
        dev = table.device
        reps = []
        for v in variants:
            ms = list(v)
            if not ms:
                reps.append(torch.zeros(self.d, device=dev))
                continue
            reps.append(table[torch.tensor(ms, device=dev)].mean(0))
        X = torch.stack(reps)                              # (n, d)
        feat = torch.cat([X, torch.full((len(variants), 1), float(dt), device=_DEVICE)],
                         dim=-1)
        return self.bg_enc(feat)                           # (n, d)

    def pop_state(self, variants, mass, t_now):
        table = self.node_repr_cached(t_now)
        dev = table.device
        reps = []
        for v in variants:
            ms = list(v)
            reps.append(table[torch.tensor(ms, device=dev)].mean(0) if ms
                        else torch.zeros(self.d, device=dev))
        X = torch.stack(reps)
        return self.mixer(self.pop_enc(X, mass))

    def budget_logit(self, dt):
        return self.new_bias + self.budget(
            _t([[float(dt)]])).squeeze()

    def score_all_additions(self, circ_mass, t_now, dt):
        """Score every possible addition to every circulating variant.

        Returns:
          allv   list of (parent_idx, mutation) pairs  -- the new variants
          logits (n_circ * V_possible,) raw scores before softmax
          The first n_circ entries are the existing variants (logit from mass).
        """
        circ = [v for v, _ in circ_mass]
        mass = _t([m for _, m in circ_mass], dtype=torch.float32)
        n = len(circ)
        table = self.node_repr_cached(t_now)               # (V, d)

        # existing: log mass + dt * fitness
        logm = torch.log(mass.clamp_min(1e-9))
        dev = table.device
        X = torch.stack([table[torch.tensor(list(v), device=dev)].mean(0)
                         if v else torch.zeros(self.d, device=dev) for v in circ])
        feat = torch.cat([X, torch.full((n, 1), float(dt), device=_DEVICE)], dim=-1)
        fit = self.fitness(feat).squeeze(-1)
        logit_exist = logm + dt * fit                      # (n,)

        # new variants: z_B · v_m for all (B, m) where m not in B
        zB = self.background_repr(circ, t_now, dt)         # (n, d)
        scores = zB @ table.T                              # (n, V)

        # mask out mutations already in the background
        for i, v in enumerate(circ):
            for m in v:
                if m < self.V:
                    scores[i, m] = -1e9

        return logit_exist, scores, circ, mass

    def predict(self, circ_mass, t_now, dt, device=None):
        """Full predicted distribution at T+h.

        LOCAL normalisation per background, then weighted by background mass.

        The global softmax over n_circ * V ~ 1M terms was the failure:
        every entry gets probability ~1e-6, so the gradient from any one
        correct entry is diluted by a factor of 1M. The model never moves.

        Instead: for each background B, softmax over its V neighbours
        independently, giving a local distribution over "which mutation
        joins B." Then the probability of a new variant B+m is:

            p(B+m) = mass(B) * p(m | B) * b

        This is a mixture of local distributions, weighted by background mass
        and the budget b. Gradient per correct entry is now ~1/V instead of
        ~1/(n*V). The normalisation is still principled: each background
        contributes proportionally to its mass.
        """
        logit_exist, scores, circ, mass = self.score_all_additions(
            circ_mass, t_now, dt)
        b = torch.sigmoid(self.budget_logit(dt)).clamp(1e-6, 1 - 1e-6)

        # existing: weighted by current mass, then scaled by (1-b)
        if device is not None:
            logit_exist = logit_exist.to(device)
            scores = scores.to(device)
            mass = mass.to(device)
        lp_exist = torch.log_softmax(logit_exist, dim=0) + torch.log1p(-b)

        # new: local softmax per background, weighted by background mass
        lp_local = torch.log_softmax(scores, dim=1)       # (n, V)
        log_mass = torch.log(mass.clamp_min(1e-9))        # (n,)
        # p(B+m) = b * mass(B) * p(m|B); in log space:
        lp_new = lp_local + log_mass.unsqueeze(1) + torch.log(b)
        # normalise the new-variant part so it sums to b
        lp_new_flat = lp_new.reshape(-1)
        lp_new_flat = lp_new_flat - torch.logsumexp(lp_new_flat, 0)                       + torch.log(b)

        logp = torch.cat([lp_exist, lp_new_flat])
        return logp, circ, scores.shape


# ======================================================================
# LOSS AND EVALUATION
# ======================================================================

def transition_loss(model, circ_mass, obs_mass, t, dt, a, train=True, device=None):
    circ = [v for v, _ in circ_mass]
    if len(circ) < 2 or not obs_mass:
        return None, None

    # restrict training target to mass-carrying variants
    ranked = sorted(obs_mass.items(), key=lambda kv: -kv[1])
    tot_all = sum(v for _, v in ranked) or 1.0
    keep, run = [], 0.0
    for v, w in ranked:
        keep.append((v, w)); run += w / tot_all
        if run >= a.obs_frac or len(keep) >= a.obs_top:
            break
    obs_fit = dict(keep)

    logp, circ_out, shape = model.predict(circ_mass, float(t), float(dt), device=device)
    n_c, V = shape

    # build index: existing variants
    ix = {v: i for i, v in enumerate(circ_out)}

    # build index: new variants (parent_idx, mutation) -> index in flat
    # flat index = n_c + parent_idx * V + mutation_id
    def new_idx(v, circ_set):
        """Find the index in logp for a new variant, or None."""
        for pi, parent in enumerate(circ_set):
            diff = v - parent
            if len(diff) == 1 and not (parent - v):
                m = next(iter(diff))
                if m < V:
                    return n_c + pi * V + m
        return None

    tgt = obs_fit if train else obs_mass
    tot = sum(tgt.values()) or 1.0
    I, W = [], []
    for v, w in tgt.items():
        if v in ix:
            I.append(ix[v]); W.append(w / tot)
        else:
            j = new_idx(v, circ_out)
            if j is not None:
                I.append(j); W.append(w / tot)

    if not I:
        return None, None
    _I = torch.tensor(I, dtype=torch.long)
    _W = torch.tensor(W, dtype=torch.float32)
    if device is not None:
        _I = _I.to(device); _W = _W.to(device)
        logp = logp.to(device)
    loss = -(_W * logp[_I]).sum() \
        / max(sum(W), 1e-9)
    if device is not None:
        loss = loss.to("cpu") if not loss.device.type == "cpu" else loss
    covered = sum(W) * tot / tot_all
    meta = {"n_target": len(obs_fit), "target_mass": run,
            "covered": covered,
            "budget": float(torch.sigmoid(
                model.budget_logit(dt)).detach())}
    return loss, meta


def score_population(logp, circ, shape, obs_mass):
    n_c, V = shape
    ix = {v: i for i, v in enumerate(circ)}
    logp_np = logp.detach().cpu().numpy() if hasattr(logp, 'detach') else logp
    p = np.exp(logp_np)
    tot = sum(obs_mass.values()) or 1.0
    q = np.zeros_like(p)
    for v, m in obs_mass.items():
        w = m / tot
        if v in ix:
            q[ix[v]] += w
        else:
            for pi, parent in enumerate(circ):
                diff = v - parent
                if len(diff) == 1 and not (parent - v):
                    mm = next(iter(diff))
                    if mm < V:
                        j = n_c + pi * V + mm
                        q[j] += w
                        break
    overlap = float(np.minimum(p, q).sum())
    denom = float(np.maximum(p, q).sum())
    jac = overlap / denom if denom > 0 else float("nan")
    ce = float(-np.sum(q * np.log(p.clip(1e-300))))
    mass_new = float(q[n_c:].sum())
    mass_exist = float(q[:n_c].sum())
    cov = float((q > 0).any())
    return {"overlap": overlap, "jaccard": jac, "ce": ce,
            "mass_new": mass_new, "mass_exist": mass_exist,
            "coverage": float(q.sum())}


def persistence_scores(circ_mass, obs_mass, shape):
    circ = [v for v, _ in circ_mass]
    n_c, V = shape
    ix = {v: i for i, v in enumerate(circ)}
    mass = np.array([m for _, m in circ_mass], dtype=float)
    mass = mass / mass.sum()
    p = np.zeros(n_c + n_c * V)
    p[:n_c] = mass
    return p


# ======================================================================
# RUN
# ======================================================================

def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    global _DEVICE; _DEVICE = device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    D = load_events(a.events, a.vocab)
    events, days, V = D["events"], D["days"], D["V"]
    by_day = group_by_day(events)
    mass_map = mass_by_day(by_day)
    all_days = sorted(by_day)

    n_tr = int(len(all_days) * a.train_frac)
    train_days = all_days[:n_tr]
    test_days = all_days[n_tr:]
    print(f"split: train {len(train_days)} ({days[train_days[0]]}.."
          f"{days[train_days[-1]]})  test {len(test_days)}"
          f" ({days[test_days[0]]}..{days[test_days[-1]]})")

    posres = None
    if a.posres and a.vocab:
        p_, r_, npos, nres, nok = parse_posres(a.vocab, V)
        print(f"posres: {nok}/{V} -> {npos} positions {nres} residues")
        posres = PosResEmbed(p_, r_, npos, nres, a.d)

    model = TransitionModel(V, d=a.d, heads=a.heads, n_recent=a.n_recent,
                            posres=posres, decay=not a.no_decay, M=a.M).to(device)
    model.mem_ok.copy_(load_first_seen(
        a.vocab, V, days[train_days[-1]]).float().unsqueeze(-1))
    model = model.to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}  device: {device}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def circ_at(t):
        """Aggregate mass over the last --window days, cap at pop_support.

        Using only today's circulating variants misses parents that have
        declined but whose descendants are appearing now. A rolling window
        keeps those parents in scope without changing the radius or the matmul
        size -- the aggregated set is still capped at pop_support.
        """
        agg = defaultdict(float)
        for u in all_days:
            if u > t:
                break
            if t - u > a.window:
                continue
            for v, w in mass_map.get(u, {}).items():
                agg[v] += w
        if not agg:
            return []
        tot = sum(agg.values()) or 1.0
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:a.pop_support]
        s = sum(w for _, w in ranked) or 1.0
        return [(v, w / s) for v, w in ranked]

    def target(t, h):
        nxt = [u for u in all_days if u >= t + 30 * h]
        return mass_map.get(nxt[0], {}) if nxt else {}

    for ep in range(a.epochs):
        losses, info, n_bad = [], [], 0
        for t in train_days[::max(1, a.stride)]:
            model.flush_pending(float(t))
            cm = circ_at(t)
            if len(cm) < 2:
                model.observe([s for s, _ in by_day[t]], float(t))
                continue
            with torch.no_grad():
                model.mixer.push(model.pop_state(
                    [v for v, _ in cm],
                    _t([m for _, m in cm], dtype=torch.float32),
                    float(t)))
            total = None
            for h in a.horizons:
                obs = target(t, h)
                if not obs:
                    continue
                l, meta = transition_loss(model, cm, obs, t,
                                          float(30 * h), a, train=True, device=device)
                if l is not None and torch.isfinite(l):
                    total = l if total is None else total + l
                    info.append(meta)
            if total is not None:
                opt.zero_grad(); total.backward()
                bad = any(not torch.isfinite(p.grad).all()
                          for p in model.parameters()
                          if p.grad is not None)
                if bad:
                    n_bad += 1; opt.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    losses.append(float(total.detach()))
            model.observe([s for s, _ in by_day[t]], float(t))

        f = lambda k: float(np.nanmean([i[k] for i in info])) if info \
            else float("nan")
        print(f"epoch {ep+1}/{a.epochs}  loss {np.mean(losses):.4f}"
              f"  budget {f('budget'):.3f}"
              f"  target {f('n_target'):.0f} ({f('target_mass'):.2f})"
              f"  cov {f('covered'):.3f}"
              + (f"  [{n_bad} skipped]" if n_bad else ""), flush=True)

    # ---- evaluation ---------------------------------------------------
    origins = test_days[::max(1, len(test_days) // a.n_origins)][:a.n_origins]
    rows = []
    for T in origins:
        model.reset_state()
        for t in all_days:
            if t > T:
                break
            model.flush_pending(float(t))
            model.observe([s for s, _ in by_day[t]], float(t))
            _cm = sorted(mass_map.get(t, {}).items(),
                         key=lambda kv: -kv[1])[:a.pop_support]
            if _cm:
                with torch.no_grad():
                    model.mixer.push(model.pop_state(
                        [v for v, _ in _cm],
                        torch.tensor([m for _, m in _cm],
                                     dtype=torch.float32), float(t)))
        cm = circ_at(T)
        if len(cm) < 2:
            continue
        for h in a.horizons:
            obs = target(T, h)
            if not obs:
                continue
            with torch.no_grad():
                logp, circ_out, shape = model.predict(
                    cm, float(T), float(30 * h), device=device)
            r = {"origin": days[T], "h": h}
            r["model"] = score_population(logp, circ_out, shape, obs)
            # persistence baseline
            p_p = persistence_scores(cm, obs, shape)
            r["persistence"] = score_population(
                np.log(p_p.clip(1e-300)), circ_out, shape, obs)
            rows.append(r)

    report(rows)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"))
        print(f"wrote {a.out}")
    return rows


def report(rows):
    if not rows:
        print("no results"); return
    print("\n" + "=" * 66)
    print("POPULATION FORECAST  overlap = Σ min(pred, obs), higher better")
    methods = ["model", "persistence"]
    print(f"\n{'horizon':<10}{'n':>4}" +
          "".join(f"{m:>14}" for m in methods))
    for h in sorted({r["h"] for r in rows}):
        sub = [r for r in rows if r["h"] == h]
        line = f"{str(h) + ' months':<10}{len(sub):>4}"
        for m in methods:
            line += f"{np.mean([r[m]['overlap'] for r in sub]):>14.4f}"
        print(line)

    print(f"\n{'horizon':<10}{'n':>4}" +
          "".join(f"{m:>14}" for m in methods) + "  (Jaccard)")
    for h in sorted({r["h"] for r in rows}):
        sub = [r for r in rows if r["h"] == h]
        line = f"{str(h) + ' months':<10}{len(sub):>4}"
        for m in methods:
            line += f"{np.mean([r[m]['jaccard'] for r in sub]):>14.4f}"
        print(line)

    print("\nsplit by where mass sits:")
    print(f"{'horizon':<10}{'mass exist':>12}{'mass new':>10}"
          f"{'covered':>10}")
    for h in sorted({r["h"] for r in rows}):
        sub = [r for r in rows if r["h"] == h]
        print(f"{str(h) + ' months':<10}"
              f"{np.mean([r['model']['mass_exist'] for r in sub]):>12.3f}"
              f"{np.mean([r['model']['mass_new'] for r in sub]):>10.3f}"
              f"{np.mean([r['model']['coverage'] for r in sub]):>10.3f}")
    print("\nOne seed. Run seeds 1-4 before treating any number as a result.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--vocab", default=None)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--n-recent", type=int, default=10, dest="n_recent")
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--posres", action="store_true")
    p.add_argument("--no-decay", action="store_true", dest="no_decay")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 6])
    p.add_argument("--pop-support", type=int, default=500, dest="pop_support")
    p.add_argument("--obs-frac", type=float, default=0.8, dest="obs_frac")
    p.add_argument("--obs-top", type=int, default=200, dest="obs_top")
    p.add_argument("--stride", type=int, default=7)
    p.add_argument("--train-frac", type=float, default=0.7, dest="train_frac")
    p.add_argument("--n-origins", type=int, default=3, dest="n_origins")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=30,
                   help="days of history to aggregate for circulating variants")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
