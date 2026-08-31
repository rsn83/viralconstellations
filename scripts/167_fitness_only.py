#!/usr/bin/env python3
"""
164_faithful.py -- Faithful implementation of DHyperNodeTPP
(Gracious, Gupta, Dukkipati, AAAI-25), undirected variant.

EXACTLY what the paper does:
  - Per-node GRU memory with neighbourhood attention
  - MLP_t: predict inter-event time (lognormal), LL_t loss
  - MLP_a: predict adjacency vector per node, LL_a loss  
  - MLP_k: predict hyperedge size per node, LL_k loss
  - Candidate assembly: firing nodes → top-k_i neighbours → intersection
  - HyperSAGNN scorer: (d_i - s_i)^2, LL_h loss
  - All four losses trained jointly

Nothing added, nothing simplified. This is the baseline.
Evaluated by population overlap at frozen origins (our protocol).
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# DATA (same as 163)
# ======================================================================

def load_events(path, verbose=True):
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
        print(f"variant size: median {int(np.median(sizes))}"
              f"  unique {len({s for s,_,_ in events}):,}")
    return {"events": events, "days": days, "V": V}


def parse_posres(vocab_path, V):
    import re
    names = {}
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path) as f:
            for line in f:
                p = line.rstrip("\n").split()
                if len(p) >= 2 and p[0].isdigit():
                    names[int(p[0])] = p[1]
    else:
        if vocab_path:
            raise FileNotFoundError(f"vocab not found: {vocab_path}")
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
            pos.append(-(i+1)); res.append("?")
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
            p = line.rstrip("\n").split()
            if len(p) >= 3 and p[0].isdigit():
                i = int(p[0])
                if i < V and p[2] > cutoff:
                    seen[i] = False; n_late += 1
    print(f"  {n_late:,} mutations first appear after {cutoff}; "
          "memory suppressed, posres only")
    return seen


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
        out[t] = {s: v/tot for s, v in agg.items()}
    return out


# ======================================================================
# MODEL -- faithful to Section 3 of the paper
# ======================================================================

class FourierTime(nn.Module):
    """ψ(Δt) -- Fourier time encoding from the paper."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(
            1.0 / 10 ** np.linspace(0, 3, d)).float())
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, dt):
        return torch.cos(dt.unsqueeze(-1) * self.w + self.b)


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


class DHyperNodeTPP(nn.Module):
    """
    Faithful undirected DHyperNodeTPP.

    Section 3.2 of the paper:
        v_i(t) = tanh(W_s Mem_i + W_r v_i^r + b_v)
        where v_i^r = neighbourhood attention over last N hyperedges

    Four prediction heads (Section 3.3):
        MLP_t: inter-event time (lognormal)
        MLP_a: adjacency vector (Bernoulli per node)
        MLP_k: hyperedge size (categorical)
        HyperSAGNN: hyperedge score via (d_i - s_i)^2
    """

    def __init__(self, V, d=64, heads=2, N=10, max_size=64,
                 posres=None, decay=True):
        super().__init__()
        self.V, self.d, self.N = V, d, N
        self.max_size = max_size

        # time encoding ψ(Δt)
        self.psi = FourierTime(d)

        # GRU memory per node
        msg_dim = 3 * d
        self.gru = nn.GRUCell(msg_dim, d)
        self.decay = decay
        self.log_gamma = nn.Parameter(torch.tensor(-3.0))

        # neighbourhood attention (Section 3.2)
        self.W_r = nn.Linear(2 * d, d)
        self.nbr_attn = nn.MultiheadAttention(2*d, heads, batch_first=True)

        # node representation projection
        self.W_s = nn.Linear(d, d)
        self.b_v = nn.Parameter(torch.zeros(d))

        self.posres = posres

        # fitness head: predicts differential growth of existing variants
        # conditioned on variant repr + horizon h
        # this is the ONLY thing we train first
        self.fitness = nn.Sequential(
            nn.Linear(d + d, 2*d), nn.Tanh(), nn.Linear(2*d, 1))
        nn.init.zeros_(self.fitness[-1].weight)
        nn.init.zeros_(self.fitness[-1].bias)  # starts at persistence

        # MLP_t: predict log mean inter-event time (lognormal)
        self.MLP_t = nn.Sequential(
            nn.Linear(d, 2*d), nn.Tanh(), nn.Linear(2*d, 1))
        self.log_sigma_t = nn.Parameter(torch.tensor(0.0))

        # MLP_a: predict adjacency vector -- which nodes will co-occur
        self.MLP_a = nn.Sequential(
            nn.Linear(d, 2*d), nn.Tanh(), nn.Linear(2*d, d))
        self.adj_bias = nn.Parameter(torch.zeros(V))

        # MLP_k: predict hyperedge size distribution
        self.MLP_k = nn.Sequential(
            nn.Linear(d, 2*d), nn.Tanh(), nn.Linear(2*d, max_size))

        # HyperSAGNN scorer: (d_i - s_i)^2
        self.SAGNNattn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.W_sasgnn = nn.Linear(d, d, bias=False)
        self.W_out = nn.Linear(d, 1)
        nn.init.zeros_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

        # persistent buffers
        self.register_buffer("mem", torch.zeros(V, d))
        self.register_buffer("last_t", torch.zeros(V))
        self.register_buffer("nbr_vec", torch.zeros(V, N, d))
        self.register_buffer("nbr_t_buf", torch.zeros(V, N))
        self.register_buffer("nbr_cnt", torch.zeros(V, dtype=torch.long))
        self.register_buffer("mem_ok", torch.ones(V, 1))
        self._pending = None
        self._cache_t = None
        self._cache_v = None

    # ---- memory --------------------------------------------------------

    def reset_state(self):
        self.mem.zero_(); self.last_t.zero_()
        self.nbr_vec.zero_(); self.nbr_t_buf.zero_(); self.nbr_cnt.zero_()
        self._pending = None; self._cache_t = None; self._cache_v = None

    def _read_mem(self, idx, t_now):
        m = self.mem[idx]
        if self.decay:
            dt = (t_now - self.last_t[idx]).clamp_min(0.0)
            g = torch.sigmoid(self.log_gamma) * 0.5 + 0.5
            m = m * g.pow(dt).unsqueeze(-1)
        return m * self.mem_ok[idx]

    def flush_pending(self, t_now):
        if self._pending is None:
            return
        idx, msg = self._pending
        cur = self._read_mem(idx, t_now)
        new = self.gru(msg, cur)
        with torch.no_grad():
            self.mem[idx] = new.detach()
            self.last_t[idx] = t_now
        self._pending = None
        self._cache_t = None

    # ---- node representation (Section 3.2) ----------------------------

    def all_node_repr(self, t_now):
        dev = self.mem.device
        idx = torch.arange(self.V, device=dev)
        m = self._read_mem(idx, t_now)
        # neighbourhood attention over last N hyperedges per node
        dt = (t_now - self.nbr_t_buf).clamp_min(0.0)
        ctx = torch.cat([self.nbr_vec, self.psi(dt)], dim=-1)  # (V,N,2d)
        mask = torch.arange(self.N, device=dev).unsqueeze(0) >= \
               self.nbr_cnt.clamp(max=self.N).unsqueeze(1)
        mask[mask.all(dim=1), 0] = False
        q = torch.cat([m, self.psi(torch.zeros(self.V, device=dev))], dim=-1)
        nbr, _ = self.nbr_attn(q.unsqueeze(1), ctx, ctx,
                               key_padding_mask=mask, need_weights=False)
        nbr = self.W_r(nbr.squeeze(1))
        v = self.W_s(m) + nbr
        if self.posres is not None:
            v = v + self.posres(idx)
        return torch.tanh(v + self.b_v)

    def node_repr(self, t_now):
        if self._cache_t != t_now or self._cache_v is None:
            self._cache_v = self.all_node_repr(t_now)
            self._cache_t = t_now
        return self._cache_v

    # ---- observe: update memory with observed hyperedge ---------------

    def observe(self, variants, t_now, max_k=64):
        """Update node memories with observed hyperedges (Section 3.2)."""
        seen = sorted({m for s in variants for m in s if m < self.V})
        if not seen:
            return
        dev = self.mem.device
        V_rep = self.node_repr(t_now)
        idx = torch.tensor(seen, dtype=torch.long, device=dev)
        v = V_rep[idx]
        pos = {m: i for i, m in enumerate(seen)}
        agg = torch.zeros(len(seen), self.d, device=dev)
        cnt = torch.zeros(len(seen), 1, device=dev)
        for s in variants:
            ms = [m for m in list(s)[:max_k] if m < self.V]
            if not ms:
                continue
            rows = torch.tensor([pos[m] for m in ms
                                  if m in pos], dtype=torch.long, device=dev)
            ctx = V_rep[torch.tensor(ms, dtype=torch.long, device=dev)
                       ].mean(0, keepdim=True)
            agg.index_add_(0, rows, ctx.expand(len(rows), -1))
            cnt.index_add_(0, rows, torch.ones(len(rows), 1, device=dev))
        agg = agg / cnt.clamp_min(1.0)
        dt = (t_now - self.last_t[idx]).clamp_min(0.0)
        msg = torch.cat([v.detach(), agg.detach(),
                         self.psi(dt)], dim=-1)
        self._pending = (idx, msg.detach())
        with torch.no_grad():
            slot = self.nbr_cnt[idx] % self.N
            self.nbr_vec[idx, slot] = agg.detach()
            self.nbr_t_buf[idx, slot] = t_now
            self.nbr_cnt[idx] += 1
        self._cache_t = None

    # ---- four prediction heads ----------------------------------------

    def predict_timing(self, active_idx, t_now):
        """MLP_t: log mean inter-event time per node."""
        v = self.node_repr(t_now)[active_idx]
        return self.MLP_t(v).squeeze(-1)               # (K,)

    def predict_adjacency(self, active_idx, t_now):
        """MLP_a: Bernoulli logits over V per active node."""
        v = self.node_repr(t_now)[active_idx]          # (K, d)
        q = self.MLP_a(v)                              # (K, d)
        table = self.node_repr(t_now)
        return q @ table.T + self.adj_bias.unsqueeze(0)  # (K, V)

    def predict_size(self, active_idx, t_now):
        """MLP_k: size distribution per active node."""
        v = self.node_repr(t_now)[active_idx]
        return self.MLP_k(v)                            # (K, max_size)

    def score_hyperedge(self, variants, t_now):
        """HyperSAGNN: score each candidate via (d_i - s_i)^2."""
        if not variants:
            return torch.zeros(0)
        table = self.node_repr(t_now)
        B = len(variants)
        K = max(1, min(64, max(len(s) for s in variants)))
        idx = torch.zeros(B, K, dtype=torch.long, device=table.device)
        mask = torch.ones(B, K, dtype=torch.bool, device=table.device)
        for b, s in enumerate(variants):
            ms = list(s)[:K]
            if not ms:
                mask[b, 0] = False; continue
            idx[b, :len(ms)] = torch.tensor(ms, device=table.device)
            mask[b, :len(ms)] = False
        X = table[idx]                                 # (B, K, d)
        dyn, _ = self.SAGNNattn(X, X, X,
                                 key_padding_mask=mask,
                                 need_weights=False)
        stat = self.W_sasgnn(X)
        per = self.W_out((dyn - stat) ** 2).squeeze(-1)  # (B, K)
        valid = (~mask).float()
        return (per * valid).sum(1) / valid.sum(1).clamp_min(1.0)

    # ---- candidate assembly (Section 3.3) ----------------------------

    def assemble_candidates(self, t_now, n_fire=50, exclude=None):
        """
        Their exact assembly:
        1. Predict firing nodes via MLP_t (top-n_fire by rate)
        2. For each firing node, predict top-k_i neighbours via MLP_a
           where k_i comes from MLP_k
        3. Intersect neighbour sets across firing nodes
        4. Result: candidate hyperedges
        """
        exclude = exclude or set()
        dev = self.mem.device

        # step 1: predict which nodes fire (highest rate = lowest inter-event time)
        all_idx = torch.arange(self.V, device=dev)
        mu_t = self.predict_timing(all_idx, t_now)     # (V,)
        # lower mu_t = shorter inter-event time = fires sooner
        _, fire_idx = torch.topk(-mu_t, k=min(n_fire, self.V))

        # step 2: for each firing node, predict size and top-k neighbours
        adj_logits = self.predict_adjacency(fire_idx, t_now)  # (n_fire, V)
        size_logits = self.predict_size(fire_idx, t_now)      # (n_fire, max_size)

        candidates = []
        for i in range(len(fire_idx)):
            node = int(fire_idx[i])
            # predicted size
            k_i = int(torch.argmax(size_logits[i])) + 1
            k_i = max(2, min(k_i, 64))
            # top-k_i neighbours by adjacency probability
            _, top_j = torch.topk(adj_logits[i], k=k_i)
            neighbours = set(top_j.cpu().tolist())
            neighbours.add(node)
            cand = frozenset(neighbours)
            if cand and cand not in exclude:
                candidates.append(cand)

        # step 3: intersection -- find mutations predicted by multiple nodes
        # build a count of how many firing nodes predict each mutation
        vote = defaultdict(int)
        for i in range(len(fire_idx)):
            _, top_j = torch.topk(adj_logits[i],
                                   k=min(32, self.V))
            for j in top_j.cpu().tolist():
                vote[j] += 1

        # mutations with >= 2 votes are "agreed upon" co-members
        agreed = {m for m, c in vote.items() if c >= 2}
        if agreed:
            # build one consensus candidate from the agreed mutations
            consensus = frozenset(agreed)
            if consensus and consensus not in exclude:
                candidates.append(consensus)

        return candidates


# ======================================================================
# LOSSES (four terms from Section 3.3)
# ======================================================================

def compute_losses(model, circ_mass, obs_mass, t, dt, a, train=True):
    """Fitness-only loss: predict which existing variants grow vs decline.

    No assembly. No budget. No new variants.
    Budget = 0 always -- persistence on new variants.

    The fitness head takes each circulating variant's node representation
    + Fourier encoding of dt, outputs a scalar growth correction.

    Loss: cross-entropy of predicted existing-variant distribution
    vs observed existing-variant mass at T+h.

    If this beats persistence, the node representations carry signal
    about which variants will grow. That is the first real result.
    """
    circ = [v for v, _ in circ_mass]
    if len(circ) < 2 or not obs_mass:
        return None, {}

    dev = model.mem.device
    table = model.node_repr(float(t))

    # variant representations: mean of member node reprs
    reps = []
    for v in circ:
        ms = [m for m in v if m < model.V]
        if ms:
            reps.append(table[torch.tensor(ms, device=dev, dtype=torch.long)].mean(0))
        else:
            reps.append(torch.zeros(model.d, device=dev))
    X = torch.stack(reps)                                     # (n, d)

    # horizon conditioning via Fourier time encoding
    psi_h = model.psi(torch.tensor([float(dt)], device=dev))  # (1, d)
    psi_h = psi_h.expand(len(circ), -1)                       # (n, d)

    # fitness: correction to log mass
    feat = torch.cat([X, psi_h], dim=-1)                      # (n, 2d)
    fit  = model.fitness(feat).squeeze(-1)                     # (n,)

    # predicted log distribution over existing variants
    mass = torch.tensor([w for _, w in circ_mass],
                        dtype=torch.float32, device=dev)
    logm = torch.log(mass.clamp_min(1e-9))
    lp   = torch.log_softmax(logm + fit, dim=0)               # (n,)

    # target: observed mass on existing variants only
    ix   = {v: i for i, v in enumerate(circ)}
    tot  = sum(obs_mass.values()) or 1.0
    I, W = [], []
    for v, w in obs_mass.items():
        j = ix.get(v)
        if j is not None:
            I.append(j); W.append(w / tot)
    if not I:
        return None, {}

    loss = -(torch.tensor(W, device=dev)
             * lp[torch.tensor(I, dtype=torch.long, device=dev)]
             ).sum() / max(sum(W), 1e-9)

    cov = sum(W)   # fraction of future mass on existing variants
    meta = {'cov': cov, 'll_pop': float(loss.detach())}
    return loss, meta


# ======================================================================
# EVALUATION
# ======================================================================

def score_population(model, circ_mass, obs_mass, t, n_fire):
    """Evaluate: fitness head predicts existing variant weights.
    Budget = 0. No new variant predictions. Pure fitness baseline."""
    dev = model.mem.device
    circ = [v for v, _ in circ_mass]
    mass = torch.tensor([w for _, w in circ_mass],
                        dtype=torch.float32, device=dev)
    table = model.node_repr(float(t))
    reps = []
    for v in circ:
        ms = [m for m in v if m < model.V]
        if ms:
            reps.append(table[torch.tensor(ms, device=dev, dtype=torch.long)].mean(0))
        else:
            reps.append(torch.zeros(model.d, device=dev))
    X = torch.stack(reps)
    # evaluate at each horizon in EVAL_H
    results = {}
    for h in [1, 2, 3, 6]:
        dt = float(h * 30)
        psi_h = model.psi(torch.tensor([dt], device=dev)).expand(len(circ), -1)
        feat  = torch.cat([X, psi_h], dim=-1)
        fit   = model.fitness(feat).squeeze(-1)
        logm  = torch.log(mass.clamp_min(1e-9))
        lp    = torch.log_softmax(logm + fit, dim=0)
        p     = torch.exp(lp).detach().cpu().numpy()
        ix    = {v: i for i, v in enumerate(circ)}
        tot   = sum(obs_mass.values()) or 1.0
        q     = np.zeros_like(p)
        for v, w in obs_mass.items():
            j = ix.get(v)
            if j is not None:
                q[j] += w / tot
        overlap = float(np.minimum(p, q).sum())
        denom   = float(np.maximum(p, q).sum())
        results[h] = {
            'overlap':    overlap,
            'jaccard':    overlap / denom if denom > 0 else float('nan'),
            'coverage':   float(q.sum()),
            'mass_exist': float(q.sum()),
            'mass_new':   0.0,
        }
    return results


def persistence_overlap(circ_mass, obs_mass):
    ix = {v: i for i, v in enumerate(v for v, _ in circ_mass)}
    mass = np.array([w for _, w in circ_mass])
    p = mass / mass.sum()
    tot = sum(obs_mass.values()) or 1.0
    q = np.zeros_like(p)
    for v, w in obs_mass.items():
        j = ix.get(v)
        if j is not None:
            q[j] += w / tot
    overlap = float(np.minimum(p, q).sum())
    denom = float(np.maximum(p, q).sum())
    return {"overlap": overlap,
            "jaccard": overlap/denom if denom > 0 else float("nan")}


# ======================================================================
# TRAINING
# ======================================================================

def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    D = load_events(a.events)
    events, days, V = D["events"], D["days"], D["V"]
    by_day = group_by_day(events)
    mass_map = mass_by_day(by_day)
    all_days = sorted(by_day)

    n_tr = int(len(all_days) * a.train_frac)
    train_days = all_days[:n_tr]
    test_days = all_days[n_tr:]
    print(f"train {len(train_days)} ({days[train_days[0]]}.."
          f"{days[train_days[-1]]})  test {len(test_days)}"
          f" ({days[test_days[0]]}..{days[test_days[-1]]})")

    posres = None
    if a.posres and a.vocab:
        p_, r_, npos, nres, nok = parse_posres(a.vocab, V)
        print(f"posres: {nok}/{V} -> {npos} positions {nres} residues")
        posres = PosResEmbed(p_, r_, npos, nres, a.d)

    model = DHyperNodeTPP(V, d=a.d, heads=a.heads, N=a.n_recent,
                          max_size=a.max_size, posres=posres,
                          decay=not a.no_decay).to(device)
    model.mem_ok.copy_(load_first_seen(
        a.vocab, V, days[train_days[-1]]).float().unsqueeze(-1).to(device))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}"
          f"  device: {device}")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def circ_at(t):
        agg = defaultdict(float)
        for u in all_days:
            if u > t: break
            if t - u > a.window: continue
            for v, w in mass_map.get(u, {}).items():
                agg[v] += w
        if not agg: return []
        tot = sum(agg.values()) or 1.0
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:a.pop_support]
        s = sum(w for _, w in ranked) or 1.0
        return [(v, w/s) for v, w in ranked]

    def target(t, h):
        nxt = [u for u in all_days if u >= t + 30*h]
        return mass_map.get(nxt[0], {}) if nxt else {}

    for ep in range(a.epochs):
        losses, infos, n_bad = [], [], 0
        for t in train_days[::max(1, a.stride)]:
            model.flush_pending(float(t))
            cm = circ_at(t)
            if len(cm) < 2:
                model.observe([s for s, _ in by_day[t]], float(t))
                continue
            total = None
            for h in a.horizons:
                obs = target(t, h)
                if not obs: continue
                l, info = compute_losses(model, cm, obs, t,
                                         float(30*h), a, train=True)
                if l is not None and torch.isfinite(l):
                    total = l if total is None else total + l
                    infos.append(info)
            if total is not None:
                opt.zero_grad()
                try:
                    total.backward()
                    bad = any(not torch.isfinite(p.grad).all()
                              for p in model.parameters() if p.grad is not None)
                    if bad:
                        n_bad += 1; opt.zero_grad()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                        losses.append(float(total.detach()))
                except RuntimeError:
                    n_bad += 1; opt.zero_grad()
            model.observe([s for s, _ in by_day[t]], float(t))

        def f(k): return float(np.nanmean([i.get(k, float('nan'))
                                            for i in infos])) if infos else float('nan')
        print(f"epoch {ep+1}/{a.epochs}"
              f"  loss {np.mean(losses) if losses else float('nan'):.3f}"
              f"  ll_t {f('ll_t'):.3f}  ll_a {f('ll_a'):.3f}"
              f"  ll_k {f('ll_k'):.3f}  ll_h {f('ll_h'):.3f}"
              f"  ll_pop {f('ll_pop'):.3f}  cov {f('cov'):.3f}"
              + (f"  [{n_bad} skipped]" if n_bad else ""), flush=True)

    # ---- evaluation ---------------------------------------------------
    origins = test_days[::max(1, len(test_days)//a.n_origins)][:a.n_origins]
    rows = []
    for T in origins:
        model.reset_state()
        for t in all_days:
            if t > T: break
            model.flush_pending(float(t))
            model.observe([s for s, _ in by_day[t]], float(t))
        cm = circ_at(T)
        if len(cm) < 2: continue
        for h in a.horizons:
            obs = target(T, h)
            if not obs: continue
            r = {"origin": days[T], "h": h}
            model_scores = score_population(model, cm, obs, T, a.n_fire)
            r["model"] = model_scores.get(h, model_scores.get(1, {}))
            r["persistence"] = persistence_overlap(cm, obs)
            rows.append(r)

    print("\n" + "="*66)
    print("POPULATION FORECAST  overlap = Σ min(pred,obs), higher better")
    for h in sorted({r["h"] for r in rows}):
        sub = [r for r in rows if r["h"]==h]
        mo = np.mean([r["model"]["overlap"] for r in sub])
        po = np.mean([r["persistence"]["overlap"] for r in sub])
        cov = np.mean([r["model"]["coverage"] for r in sub])
        mn = np.mean([r["model"]["mass_new"] for r in sub])
        print(f"  h={h}m  model {mo:.4f}  persistence {po:.4f}"
              f"  covered {cov:.3f}  mass_new {mn:.3f}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"))
        print(f"wrote {a.out}")
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Faithful DHyperNodeTPP implementation")
    p.add_argument("--events", required=True)
    p.add_argument("--vocab", default=None)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--n-recent", type=int, default=10, dest="n_recent")
    p.add_argument("--max-size", type=int, default=64, dest="max_size")
    p.add_argument("--posres", action="store_true")
    p.add_argument("--no-decay", action="store_true", dest="no_decay")
    p.add_argument("--horizons", type=int, nargs="+", default=[1,2,3,6])
    p.add_argument("--pop-support", type=int, default=500, dest="pop_support")
    p.add_argument("--n-active", type=int, default=200, dest="n_active",
                   help="active mutations for LL_a and LL_k supervision")
    p.add_argument("--n-fire", type=int, default=50, dest="n_fire",
                   help="firing nodes for candidate assembly")
    p.add_argument("--n-pos", type=int, default=32, dest="n_pos",
                   help="positive hyperedges for LL_h")
    p.add_argument("--n-neg", type=int, default=64, dest="n_neg",
                   help="negative candidates for LL_h")
    p.add_argument("--obs-top", type=int, default=200, dest="obs_top")
    p.add_argument("--window", type=int, default=90)
    p.add_argument("--w-t", type=float, default=1.0, dest="w_t")
    p.add_argument("--w-a", type=float, default=1.0, dest="w_a")
    p.add_argument("--w-k", type=float, default=1.0, dest="w_k")
    p.add_argument("--w-h", type=float, default=1.0, dest="w_h")
    p.add_argument("--w-pop", type=float, default=1.0, dest="w_pop")
    p.add_argument("--stride", type=int, default=7)
    p.add_argument("--train-frac", type=float, default=0.7, dest="train_frac")
    p.add_argument("--n-origins", type=int, default=3, dest="n_origins")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
