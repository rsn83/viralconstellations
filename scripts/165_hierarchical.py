#!/usr/bin/env python3
"""
165_hierarchical.py -- Hierarchical Hypergraph Transition Model

DESIGN
------
Two levels of representation, both persistent:

  Node level:    v_i(t) ∈ R^d   per-mutation GRU memory (from 164)
  Cluster level: h_c(t) ∈ R^d   per-cluster GRU-HG state (from EvolveHypergraph)

The hypergraph is represented at both levels:
  - Node-to-cluster: soft assignments p(c|v_i(t)), learned MLP, dynamic
  - Cluster-to-variant: within-cluster activation p(activate|v_i,h_c(t))

A new variant = mutations activated across clusters, assembled by sampling
within-cluster activations conditioned on cluster hidden states.

SPECIAL CASES
-------------
K=1:  cluster level collapses to single population state u_t → recovers 164
K=V:  each mutation is its own cluster → recovers 163 dot product

SPARSITY REGULARIZATION (from EvolveHypergraph)
Encourages the model to use fewer active clusters via entropy regularization.
Effective K emerges from training rather than being preset.

SMOOTHNESS REGULARIZATION (from EvolveHypergraph)  
Penalises abrupt cluster reassignments. Biologically motivated: lineage
structure changes gradually. Also stabilises training.

LOSSES
------
LL_t:    timing supervision per mutation (from 164)
LL_a:    within-cluster adjacency supervision (from 164, restricted to cluster)
LL_k:    size prediction (from 164)
LL_h:    hyperedge scoring via HyperSAGNN (from 164)
LL_pop:  population forecast cross-entropy (our protocol)
L_SP:    sparsity regularization on cluster usage
L_SM:    smoothness regularization on cluster evolution
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
# DATA (identical to 164)
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
    if vocab_path:
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab not found: {vocab_path}")
        with open(vocab_path) as f:
            for line in f:
                p = line.rstrip("\n").split()
                if len(p) >= 2 and p[0].isdigit():
                    names[int(p[0])] = p[1]
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


class HierarchicalHypergraph(nn.Module):
    """
    Two-level hypergraph representation.

    Node level:    v_i(t) from GRU memory + neighbourhood attention
    Cluster level: h_c(t) from GRU-HG evolving cluster hidden states

    Special cases:
      K=1 → h_c = u_t (population state), recovers 164
      K=V → trivial clusters, recovers 163
    """

    def __init__(self, V, K, d=64, heads=2, N=10, max_size=64,
                 posres=None, decay=True):
        super().__init__()
        self.V, self.K, self.d, self.N = V, K, d, N
        self.max_size = max_size

        # ---- node level (identical to 164) ----------------------------
        self.psi = FourierTime(d)
        msg_dim = 3 * d
        self.gru = nn.GRUCell(msg_dim, d)
        self.decay = decay
        self.log_gamma = nn.Parameter(torch.tensor(-3.0))
        self.W_r = nn.Linear(2*d, d)
        self.nbr_attn = nn.MultiheadAttention(2*d, heads, batch_first=True)
        self.W_s = nn.Linear(d, d)
        self.b_v = nn.Parameter(torch.zeros(d))
        self.posres = posres

        # ---- cluster level (from EvolveHypergraph) --------------------
        # Soft cluster assignments: p(c | v_i(t))
        # Zero-init output so at start all mutations equally assigned
        self.cluster_assign = nn.Sequential(
            nn.Linear(d, 2*d), nn.Tanh(), nn.Linear(2*d, K))
        nn.init.zeros_(self.cluster_assign[-1].weight)
        nn.init.zeros_(self.cluster_assign[-1].bias)

        # GRU-HG: evolves cluster hidden states
        # Input: current cluster representation (mean of member v_i)
        # Hidden: persistent cluster state h_c(t)
        self.gru_hg = nn.GRUCell(d, d)

        # ---- prediction heads (from 164) ------------------------------
        # MLP_t: timing -- conditioned on node AND cluster state
        self.MLP_t = nn.Sequential(
            nn.Linear(2*d, 2*d), nn.Tanh(), nn.Linear(2*d, 1))

        # MLP_a: within-cluster adjacency -- conditioned on cluster state
        # This is the key improvement over 164: adjacency is predicted
        # within cluster context, not over all V independently
        self.MLP_a = nn.Sequential(
            nn.Linear(2*d, 2*d), nn.Tanh(), nn.Linear(2*d, d))
        self.adj_bias = nn.Parameter(torch.zeros(V))

        # MLP_k: size prediction -- conditioned on cluster state
        self.MLP_k = nn.Sequential(
            nn.Linear(2*d, 2*d), nn.Tanh(), nn.Linear(2*d, max_size))

        # HyperSAGNN scorer (from 164): (d_i - s_i)^2
        self.SAGNNattn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.W_sasgnn = nn.Linear(d, d, bias=False)
        self.W_out = nn.Linear(d, 1)
        nn.init.zeros_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

        # ---- persistent buffers ---------------------------------------
        self.register_buffer("mem", torch.zeros(V, d))
        self.register_buffer("last_t", torch.zeros(V))
        self.register_buffer("nbr_vec", torch.zeros(V, N, d))
        self.register_buffer("nbr_t_buf", torch.zeros(V, N))
        self.register_buffer("nbr_cnt", torch.zeros(V, dtype=torch.long))
        self.register_buffer("mem_ok", torch.ones(V, 1))
        # Cluster hidden states -- the persistent hypergraph state
        self.register_buffer("h_cluster", torch.zeros(K, d))
        # Previous cluster assignments for smoothness regularization
        self.register_buffer("prev_assign", torch.zeros(V, K))

        self._pending = None
        self._cache_t = None
        self._cache_v = None
        self._cache_assign = None   # cached cluster assignments

    # ---- state management --------------------------------------------

    def reset_state(self):
        self.mem.zero_(); self.last_t.zero_()
        self.nbr_vec.zero_(); self.nbr_t_buf.zero_(); self.nbr_cnt.zero_()
        self.h_cluster.zero_(); self.prev_assign.zero_()
        self._pending = None
        self._cache_t = None; self._cache_v = None; self._cache_assign = None

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
        self._cache_assign = None

    # ---- node representations (from 164) ----------------------------

    def all_node_repr(self, t_now):
        dev = self.mem.device
        idx = torch.arange(self.V, device=dev)
        m = self._read_mem(idx, t_now)
        dt = (t_now - self.nbr_t_buf).clamp_min(0.0)
        ctx = torch.cat([self.nbr_vec, self.psi(dt)], dim=-1)
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

    # ---- cluster assignments (Level 1) --------------------------------

    def cluster_assignments(self, t_now):
        """Soft cluster assignments p(c|v_i(t)) for all mutations.

        Returns (V, K) softmax -- row i is mutation i's cluster distribution.
        Cached per timestep since it's used multiple times.
        """
        if self._cache_assign is not None and self._cache_t == t_now:
            return self._cache_assign
        v = self.node_repr(t_now)                    # (V, d)
        logits = self.cluster_assign(v)              # (V, K)
        self._cache_assign = torch.softmax(logits, dim=-1)
        return self._cache_assign

    # ---- cluster state update (GRU-HG, Level 2) ---------------------

    def update_clusters(self, t_now, active_mutations=None):
        """Update cluster hidden states h_c(t) via GRU-HG.

        For each cluster c, its representation is the assignment-weighted
        mean of member node representations. GRU-HG then evolves h_c.

        This is the persistent hypergraph state -- it accumulates how each
        cluster has been evolving, not just what its current composition is.
        """
        v = self.node_repr(t_now)                    # (V, d)
        assign = self.cluster_assignments(t_now)     # (V, K)

        # cluster representation = weighted mean of member nodes
        # assign.T @ v gives (K, d) -- each cluster's weighted mean
        cluster_rep = assign.T @ v                   # (K, d)
        # normalise by total assignment weight per cluster
        cluster_rep = cluster_rep / assign.sum(0).unsqueeze(-1).clamp_min(1e-9)

        # GRU-HG evolution -- persistent hypergraph state
        new_h = self.gru_hg(cluster_rep, self.h_cluster)
        with torch.no_grad():
            self.prev_assign.copy_(assign.detach())
            self.h_cluster.copy_(new_h.detach())
        self._cache_assign = None   # assignments may shift after cluster update
        return new_h   # return grad-carrying version for loss

    # ---- conditioned node representations (node + cluster context) ---

    def conditioned_repr(self, idx, t_now):
        """v_i(t) concatenated with its cluster-weighted h_c(t).

        This is the key: each mutation's representation now includes its
        cluster context. A mutation in a trending cluster gets different
        predictions than one in a declining cluster -- even if their
        individual histories are similar.

        When K=1: all mutations share the same h_c = u_t, recovering 164.
        """
        v = self.node_repr(t_now)[idx]               # (n, d)
        assign = self.cluster_assignments(t_now)[idx]  # (n, K)
        # weighted combination of cluster hidden states
        h_weighted = assign @ self.h_cluster         # (n, d)
        return torch.cat([v, h_weighted], dim=-1)    # (n, 2d)

    # ---- observe: update node memory ---------------------------------

    def observe(self, variants, t_now, max_k=64):
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
            ms = [m for m in list(s)[:max_k] if m < self.V and m in pos]
            if not ms:
                continue
            rows = torch.tensor([pos[m] for m in ms],
                                 dtype=torch.long, device=dev)
            ctx = V_rep[torch.tensor(ms, dtype=torch.long, device=dev)
                       ].mean(0, keepdim=True)
            agg.index_add_(0, rows, ctx.expand(len(rows), -1))
            cnt.index_add_(0, rows, torch.ones(len(rows), 1, device=dev))
        agg = agg / cnt.clamp_min(1.0)
        dt = (t_now - self.last_t[idx.cpu()].to(dev)).clamp_min(0.0)
        msg = torch.cat([v.detach(), agg.detach(), self.psi(dt)], dim=-1)
        self._pending = (idx, msg.detach())
        with torch.no_grad():
            slot = self.nbr_cnt[idx.cpu()] % self.N
            self.nbr_vec[idx.cpu(), slot] = agg.detach().cpu()
            self.nbr_t_buf[idx.cpu(), slot] = t_now
            self.nbr_cnt[idx.cpu()] += 1
        self._cache_t = None
        self._cache_assign = None

    # ---- prediction heads (conditioned on cluster state) ------------

    def predict_timing(self, active_idx, t_now):
        """MLP_t conditioned on node + cluster state."""
        feat = self.conditioned_repr(active_idx, t_now)
        return self.MLP_t(feat).squeeze(-1)

    def predict_adjacency(self, active_idx, t_now):
        """MLP_a conditioned on node + cluster state.

        Key improvement over 164: adjacency is predicted within cluster
        context. Mutations in the same cluster get higher co-occurrence
        predictions -- the cluster structure constrains the search space.
        """
        feat = self.conditioned_repr(active_idx, t_now)
        q = self.MLP_a(feat)                         # (n, d)
        table = self.node_repr(t_now)
        return q @ table.T + self.adj_bias.unsqueeze(0)  # (n, V)

    def predict_size(self, active_idx, t_now):
        feat = self.conditioned_repr(active_idx, t_now)
        return self.MLP_k(feat)

    def score_hyperedge(self, variants, t_now):
        """HyperSAGNN: (d_i - s_i)^2 -- identical to 164."""
        variants = [v for v in variants if v]   # remove empty sets
        if not variants:
            return torch.zeros(0)
        table = self.node_repr(t_now)
        B = len(variants)
        K_size = max(1, min(64, max(len(s) for s in variants)))
        idx = torch.zeros(B, K_size, dtype=torch.long, device=table.device)
        mask = torch.ones(B, K_size, dtype=torch.bool, device=table.device)
        for b, s in enumerate(variants):
            ms = list(s)[:K_size]
            if not ms:
                mask[b, 0] = False; continue
            idx[b, :len(ms)] = torch.tensor(ms, device=table.device)
            mask[b, :len(ms)] = False
        X = table[idx]
        dyn, _ = self.SAGNNattn(X, X, X,
                                 key_padding_mask=mask, need_weights=False)
        stat = self.W_sasgnn(X)
        per = self.W_out((dyn - stat) ** 2).squeeze(-1)
        valid = (~mask).float()
        return (per * valid).sum(1) / valid.sum(1).clamp_min(1.0)

    def assemble_candidates(self, t_now, n_fire=50, exclude=None):
        """Hierarchical candidate assembly.

        Level 1: predict firing nodes (MLP_t)
        Level 2: for each firing node, predict top-k within-cluster
                 neighbours (MLP_a conditioned on cluster state)
        Intersection: mutations predicted by multiple firing nodes

        The cluster conditioning means mutations in the same cluster
        preferentially predict each other -- reducing the effective
        search space and improving precision.

        K=1 special case: reduces to 164's assembly exactly.
        """
        exclude = exclude or set()
        dev = self.mem.device
        all_idx = torch.arange(self.V, device=dev)
        mu_t = self.predict_timing(all_idx, t_now)
        _, fire_idx = torch.topk(-mu_t, k=min(n_fire, self.V))

        adj_logits = self.predict_adjacency(fire_idx, t_now)
        size_logits = self.predict_size(fire_idx, t_now)

        candidates = []
        vote = defaultdict(int)
        for i in range(len(fire_idx)):
            node = int(fire_idx[i])
            k_i = max(2, min(int(torch.argmax(size_logits[i])) + 1, 64))
            _, top_j = torch.topk(adj_logits[i], k=k_i)
            neighbours = set(top_j.cpu().tolist())
            neighbours.add(node)
            cand = frozenset(neighbours)
            if cand and cand not in exclude:
                candidates.append(cand)
            _, top_j2 = torch.topk(adj_logits[i], k=min(32, self.V))
            for j in top_j2.cpu().tolist():
                vote[j] += 1

        agreed = {m for m, c in vote.items() if c >= 2}
        if agreed:
            consensus = frozenset(agreed)
            if consensus and consensus not in exclude:
                candidates.append(consensus)

        return candidates


# ======================================================================
# REGULARIZATION (from EvolveHypergraph)
# ======================================================================

def sparsity_loss(assign):
    mean_assign = assign.detach().mean(0).clamp(1e-9, 1-1e-9)
    # use detached mean for stability; gradient still flows through assign
    entropy = -(assign.mean(0) * torch.log(mean_assign)).sum()
    return entropy.clamp(max=20.0)


def smoothness_loss(assign_now, assign_prev):
    p = assign_now
    q = assign_prev.clamp_min(1e-9)
    kl = (p * torch.log(p.clamp_min(1e-9) / q)).sum(1)
    return kl.clamp(0.0, 5.0).mean()


# ======================================================================
# LOSSES
# ======================================================================

def compute_losses(model, circ_mass, obs_mass, t, dt, a, train=True,
                   rng=None):
    circ = [v for v, _ in circ_mass]
    if not circ or not obs_mass:
        return None, {}

    dev = model.mem.device
    act = {}
    for v, w in circ_mass:
        for m in v:
            if m < model.V:
                act[m] = act.get(m, 0.0) + w
    active = sorted(act, key=lambda m: -act[m])[:a.n_active]
    if not active:
        return None, {}
    idx = torch.tensor(active, dtype=torch.long, device=dev)

    loss = torch.tensor(0.0, device=dev)
    info = {}

    # cluster state already updated in training loop before this call
    # use current h_cluster directly
    assign = model.cluster_assignments(float(t))  # (V, K)

    # regularization handled in training loop

    # ---- LL_t -------------------------------------------------------
    mu_t = model.predict_timing(idx, float(t))
    last = model.last_t[idx.cpu()].to(dev)
    delta = (float(t) - last).clamp_min(1.0)
    log_dt = torch.log(delta)
    sigma = torch.exp(model.log_sigma_t).clamp_min(0.1) \
        if hasattr(model, 'log_sigma_t') else torch.tensor(1.0, device=dev)
    ll_t = -((log_dt - mu_t)**2) / (2*sigma**2) - torch.log(sigma) - log_dt
    loss_t = -ll_t.mean()
    if torch.isfinite(loss_t):
        loss = loss + a.w_t * loss_t
        info['ll_t'] = float(loss_t.detach())

    # ---- LL_a (within-cluster adjacency) ----------------------------
    adj_logits = model.predict_adjacency(idx, float(t))
    pos_adj = torch.zeros(len(active), model.V, device=dev)
    w_adj = torch.zeros(len(active), device=dev)
    a2i = {m: i for i, m in enumerate(active)}
    tot_o = sum(obs_mass.values()) or 1.0
    for v, wv in obs_mass.items():
        ms = [m for m in v if m in a2i and m < model.V]
        if len(ms) < 2:
            continue
        for m_i in ms:
            ri = a2i[m_i]
            for m_j in ms:
                if m_j != m_i:
                    pos_adj[ri, m_j] = 1.0
            w_adj[ri] += wv / tot_o
    live = w_adj > 0
    if live.any():
        ce_a = F.binary_cross_entropy_with_logits(
            adj_logits[live], pos_adj[live], reduction='none').mean(1)
        loss_a = (w_adj[live] * ce_a).sum() / w_adj[live].sum().clamp_min(1e-9)
        if torch.isfinite(loss_a):
            loss = loss + a.w_a * loss_a
            info['ll_a'] = float(loss_a.detach())

    # ---- LL_k -------------------------------------------------------
    size_logits = model.predict_size(idx, float(t))
    sz_target = torch.zeros(len(active), dtype=torch.long, device=dev)
    w_sz = torch.zeros(len(active), device=dev)
    for v, wv in obs_mass.items():
        ms = [m for m in v if m in a2i]
        if not ms:
            continue
        sz = min(len(v)-1, model.max_size-1)
        for m_i in ms:
            sz_target[a2i[m_i]] = sz
            w_sz[a2i[m_i]] += wv / tot_o
    live_sz = w_sz > 0
    if live_sz.any():
        ce_k = F.cross_entropy(size_logits[live_sz],
                               sz_target[live_sz], reduction='none')
        loss_k = (w_sz[live_sz] * ce_k).sum() / \
                 w_sz[live_sz].sum().clamp_min(1e-9)
        if torch.isfinite(loss_k):
            loss = loss + a.w_k * loss_k
            info['ll_k'] = float(loss_k.detach())

    # ---- LL_h (random negatives during training) --------------------
    pos_v = [v for v in obs_mass][:a.n_pos]
    all_muts = list(range(model.V))
    neg_v = [frozenset(random.sample(all_muts, random.randint(2, 64)))
             for _ in range(a.n_neg)]
    all_v = pos_v + neg_v
    if all_v:
        sc = model.score_hyperedge(all_v, float(t))
        np_ = len(pos_v)
        y = torch.cat([torch.ones(np_, device=dev),
                       torch.zeros(len(neg_v), device=dev)])
        w_pos = torch.tensor([obs_mass.get(v, 0.0)/tot_o
                               for v in pos_v], device=dev)
        w_neg = torch.ones(len(neg_v), device=dev) / max(len(neg_v), 1)
        w_h = torch.cat([w_pos, w_neg])
        loss_h = (w_h * F.binary_cross_entropy_with_logits(
            sc, y, reduction='none')).sum() / w_h.sum().clamp_min(1e-9)
        if torch.isfinite(loss_h):
            loss = loss + a.w_h * loss_h
            info['ll_h'] = float(loss_h.detach())

    # ---- LL_pop (population forecast) --------------------------------
    mass = torch.tensor([w for _, w in circ_mass],
                        dtype=torch.float32, device=dev)
    logm = torch.log(mass.clamp_min(1e-9))
    sc_exist = model.score_hyperedge(circ, float(t))
    lp_exist = torch.log_softmax(logm + sc_exist, dim=0)
    ix = {v: i for i, v in enumerate(circ)}
    tgt = dict(sorted(obs_mass.items(),
                      key=lambda kv: -kv[1])[:a.obs_top])
    tot_t = sum(tgt.values()) or 1.0
    I, W = [], []
    for v, w in tgt.items():
        j = ix.get(v)
        if j is not None:
            I.append(j); W.append(w/tot_t)
    if I:
        loss_pop = -(torch.tensor(W, device=dev)
                     * lp_exist[torch.tensor(I, dtype=torch.long, device=dev)]
                     ).sum() / max(sum(W), 1e-9)
        if torch.isfinite(loss_pop):
            loss = loss + a.w_pop * loss_pop
            info['ll_pop'] = float(loss_pop.detach())
    info['cov'] = sum(W) if W else 0.0

    # active cluster count (effective K)
    with torch.no_grad():
        usage = assign.mean(0)
        info['k_eff'] = float((usage > 0.01).sum())

    return loss if loss.item() != 0 or loss.requires_grad else None, info


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate(model, circ_mass, obs_mass, t, n_fire):
    dev = model.mem.device
    circ = [v for v, _ in circ_mass]
    mass = torch.tensor([w for _, w in circ_mass],
                        dtype=torch.float32, device=dev)
    cands = model.assemble_candidates(float(t), n_fire=n_fire,
                                      exclude=set(circ))
    with torch.no_grad():
        sc_exist = model.score_hyperedge(circ, float(t))
        logm = torch.log(mass.clamp_min(1e-9))
        lp_exist = torch.log_softmax(logm + sc_exist, dim=0)
        if cands:
            sc_new = model.score_hyperedge(cands, float(t))
            b = torch.tensor(0.3, device=dev)
            logp = torch.cat([lp_exist + torch.log1p(-b),
                              torch.log_softmax(sc_new, 0) + torch.log(b)])
            allv = circ + cands
        else:
            logp, allv = lp_exist, circ

    p = torch.exp(logp).cpu().numpy()
    ix = {v: i for i, v in enumerate(allv)}
    tot = sum(obs_mass.values()) or 1.0
    q = np.zeros_like(p)
    for v, w in obs_mass.items():
        j = ix.get(v)
        if j is not None:
            q[j] += w/tot
    overlap = float(np.minimum(p, q).sum())
    denom = float(np.maximum(p, q).sum())
    return {"overlap": overlap,
            "jaccard": overlap/denom if denom > 0 else float("nan"),
            "coverage": float(q.sum()),
            "mass_exist": float(q[:len(circ)].sum()),
            "mass_new": float(q[len(circ):].sum())}


def persistence_overlap(circ_mass, obs_mass):
    circ = [v for v, _ in circ_mass]
    mass = np.array([w for _, w in circ_mass])
    p = mass / mass.sum()
    ix = {v: i for i, v in enumerate(circ)}
    tot = sum(obs_mass.values()) or 1.0
    q = np.zeros_like(p)
    for v, w in obs_mass.items():
        j = ix.get(v)
        if j is not None:
            q[j] += w/tot
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

    model = HierarchicalHypergraph(
        V, K=a.K, d=a.d, heads=a.heads, N=a.n_recent,
        max_size=a.max_size, posres=posres, decay=not a.no_decay
    ).to(device)
    # add log_sigma_t for timing loss
    model.log_sigma_t = nn.Parameter(torch.tensor(0.0, device=device))

    model.mem_ok.copy_(load_first_seen(
        a.vocab, V, days[train_days[-1]]).float().unsqueeze(-1).to(device))
    print(f"K={a.K}  parameters: {sum(p.numel() for p in model.parameters()):,}"
          f"  device: {device}")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    m_sp = sparsity_loss
    m_sm = smoothness_loss
    train = True

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
            # update cluster states once per day (GRU-HG)
            cluster_loss = torch.tensor(0.0)
            if train and len(cm) >= 2:
                assign_now = model.cluster_assignments(float(t))
                if a.w_sp > 0:
                    l_sp = m_sp(assign_now)
                    if torch.isfinite(l_sp):
                        cluster_loss = cluster_loss + a.w_sp * l_sp
                if a.w_sm > 0 and float(model.prev_assign.sum()) > 0.1:
                    l_sm = m_sm(assign_now, model.prev_assign)
                    if torch.isfinite(l_sm):
                        cluster_loss = cluster_loss + a.w_sm * l_sm
                with torch.no_grad():
                    model.update_clusters(float(t))

            total = cluster_loss if cluster_loss.item() > 0 else None
            for h in a.horizons:
                obs = target(t, h)
                if not obs: continue
                l, info = compute_losses(model, cm, obs, t,
                                         float(30*h), a, train=True, rng=rng)
                if l is not None and torch.isfinite(l) and float(l.detach()) < 1000:
                    total = l if total is None else total + l
                    infos.append(info)
            if total is not None:
                opt.zero_grad()
                try:
                    total.backward()
                    bad = any(not torch.isfinite(p.grad).all()
                              for p in model.parameters()
                              if p.grad is not None)
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
              f"  K_eff {f('k_eff'):.1f}"
              f"  sp {f('l_sp'):.3f}  sm {f('l_sm'):.3f}"
              + (f"  [{n_bad} skip]" if n_bad else ""), flush=True)

    # ---- evaluation --------------------------------------------------
    origins = test_days[::max(1, len(test_days)//a.n_origins)][:a.n_origins]
    rows = []
    for T in origins:
        model.reset_state()
        for t in all_days:
            if t > T: break
            model.flush_pending(float(t))
            model.observe([s for s, _ in by_day[t]], float(t))
            cm_t = sorted(mass_map.get(t, {}).items(),
                          key=lambda kv: -kv[1])[:a.pop_support]
            if cm_t:
                with torch.no_grad():
                    model.update_clusters(float(t))
        cm = circ_at(T)
        if len(cm) < 2: continue
        for h in a.horizons:
            obs = target(T, h)
            if not obs: continue
            r = {"origin": days[T], "h": h}
            r["model"] = evaluate(model, cm, obs, T, a.n_fire)
            r["persistence"] = persistence_overlap(cm, obs)
            rows.append(r)

    print("\n" + "="*66)
    print("POPULATION FORECAST  overlap = Σ min(pred,obs), higher better")
    print(f"{'horizon':<10}{'model':>10}{'persist':>10}{'covered':>10}"
          f"{'mass_new':>10}")
    for h in sorted({r["h"] for r in rows}):
        sub = [r for r in rows if r["h"]==h]
        mo = np.mean([r["model"]["overlap"] for r in sub])
        po = np.mean([r["persistence"]["overlap"] for r in sub])
        cov = np.mean([r["model"]["coverage"] for r in sub])
        mn = np.mean([r["model"]["mass_new"] for r in sub])
        print(f"  h={h}m    {mo:>10.4f}{po:>10.4f}{cov:>10.3f}{mn:>10.3f}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"))
        print(f"wrote {a.out}")
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Hierarchical Hypergraph -- 164 is K=1 special case")
    p.add_argument("--events", required=True)
    p.add_argument("--vocab", default=None)
    p.add_argument("--K", type=int, default=20,
                   help="max clusters; K=1 recovers 164, K=V recovers 163")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--n-recent", type=int, default=10, dest="n_recent")
    p.add_argument("--max-size", type=int, default=64, dest="max_size")
    p.add_argument("--posres", action="store_true")
    p.add_argument("--no-decay", action="store_true", dest="no_decay")
    p.add_argument("--horizons", type=int, nargs="+", default=[1,2,3,6])
    p.add_argument("--pop-support", type=int, default=500, dest="pop_support")
    p.add_argument("--n-active", type=int, default=200, dest="n_active")
    p.add_argument("--n-fire", type=int, default=50, dest="n_fire")
    p.add_argument("--n-pos", type=int, default=32, dest="n_pos")
    p.add_argument("--n-neg", type=int, default=64, dest="n_neg")
    p.add_argument("--obs-top", type=int, default=200, dest="obs_top")
    p.add_argument("--window", type=int, default=90)
    p.add_argument("--w-t", type=float, default=1.0, dest="w_t")
    p.add_argument("--w-a", type=float, default=1.0, dest="w_a")
    p.add_argument("--w-k", type=float, default=1.0, dest="w_k")
    p.add_argument("--w-h", type=float, default=1.0, dest="w_h")
    p.add_argument("--w-pop", type=float, default=1.0, dest="w_pop")
    p.add_argument("--w-sp", type=float, default=0.1, dest="w_sp",
                   help="sparsity regularization weight")
    p.add_argument("--w-sm", type=float, default=0.1, dest="w_sm",
                   help="smoothness regularization weight")
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
