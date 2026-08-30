#!/usr/bin/env python3
"""
157_hypergraph_tpp.py -- variant forecasting as an evolving hypergraph.

FORMULATION
-----------
node        a mutation (V ~ 1359)
hyperedge   a variant: the set of mutations carried by a sequence
event       a variant observed on a given day, with a weight (count)

This replaces the monthly seq2seq framing. That framing aggregated a month's
whole population into ONE training example (23 in total); here every observed
variant is an example, which is a ~10^4 increase in effective sample size from
the same data.

ADAPTED FROM DHyperNodeTPP (Gracious, Gupta, Dukkipati, AAAI-25).
Kept:     memory module, neighbourhood attention, HyperSAGNN-style scorer,
          negative-sampling objective, MRR evaluation.
Dropped:  the directed left/right split (variants are undirected sets), and
          the learned adjacency/size candidate generator (replaced below).

FOUR ADAPTATIONS, each justified by a measurement in this project:

1. CANDIDATE GENERATION BY ENUMERATION (script 150).
   Their own limitations section names candidate recall as the bottleneck.
   We measured that a newly appearing variant sits a median of 1 mutation from
   the nearest already-circulating variant (mean 0.29 at h=1 rising to 1.55 at
   h=6, ~0.25 mutations/month). So candidates are enumerated as circulating
   variants +/- a small number of currently active mutations. No learning is
   needed for this stage; capacity goes to scoring instead.

2. HARD NEGATIVES.
   Their negatives are random nodes and sizes, which are trivially separable
   and inflate MRR. Ours are candidates from the same generator that did NOT
   appear -- the discrimination that actually matters.

3. FREQUENCY-WEIGHTED EVENTS.
   Their events are unweighted. A variant at 30% and one at 0.001% are not
   equally informative, so positives are weighted by observed count.

4. TIME-DECAYED MEMORY.
   Their GRU memory has no forgetting, which suits a stationary email network.
   This population turns over completely at sweeps, so memory is decayed by
   gamma^(elapsed) before each update.

EVALUATION (extends their protocol)
   - MRR overall, and split into NEW vs REPEAT variants
   - MRR by forecast lead time, from a frozen origin (they report none)
   - MRR over calendar time, to expose regime shifts
   - three baselines they do not run: recency, frequency, and PROXIMITY
     (rank by distance to nearest circulating variant). Proximity is the
     baseline that matters here: our impossibility result showed any
     factorized mixture reduces to nearest-mode ranking, so a model that
     cannot beat proximity has learned a distance heuristic.

USAGE
    python scripts/157_hypergraph_tpp.py --events data/events.tsv --inspect
    python scripts/157_hypergraph_tpp.py --events data/events.tsv \
        --train-end 2022-12-31 --val-end 2023-06-30 --seed 0
"""

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ======================================================================
# DATA
# ======================================================================

def load_events(path, vocab_path=None, min_count=1, verbose=True):
    """Load daily variant events.

    DATA CONTRACT -- a TSV with one row per (variant, day):
        date <TAB> mutation_ids <TAB> count
        2021-03-04    3,17,402,915    12
    mutation_ids are comma-separated integer indices into the vocabulary.
    A 'count' column is optional and defaults to 1.

    Returns dict with:
        events  : list of (frozenset, day_index, weight), time-ordered
        days    : list of date strings, indexed by day_index
        V       : vocabulary size
    """
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if ln == 0 and not re.match(r"^\d{4}-\d{2}-\d{2}", parts[0]):
                continue                                   # header
            date = parts[0].strip()
            muts = parts[1].strip()
            cnt = float(parts[2]) if len(parts) > 2 else 1.0
            if cnt < min_count:
                continue
            s = frozenset(int(x) for x in muts.split(",") if x != "")
            rows.append((date, s, cnt))

    rows.sort(key=lambda r: r[0])
    days = sorted({r[0] for r in rows})
    day_ix = {d: i for i, d in enumerate(days)}
    events = [(s, day_ix[d], w) for d, s, w in rows]
    V = 1 + max((max(s) for s, _, _ in events if s), default=0)

    if verbose:
        sizes = [len(s) for s, _, _ in events]
        print(f"events {len(events):,}  days {len(days)}  "
              f"({days[0]} .. {days[-1]})  V {V}")
        print(f"variant size: median {int(np.median(sizes))} "
              f"[{min(sizes)}-{max(sizes)}]   "
              f"unique variants {len({s for s, _, _ in events}):,}")
    return {"events": events, "days": days, "V": V}


def events_from_monthly(data_dir, top=None, verbose=True):
    """Fallback: build events from the monthly *_occupied.pkl files.

    Loses daily resolution -- every variant in a month shares one timestamp --
    so the timing term carries no information. Useful only to get the pipeline
    running before the daily metadata is wired in.
    """
    import glob
    import pickle
    files = sorted(glob.glob(os.path.join(data_dir, "*_occupied.pkl")))
    rows, days = [], []
    for fp in files:
        m = re.search(r"(\d{4}-\d{2})_occupied\.pkl$", os.path.basename(fp))
        if not m:
            continue
        label = m.group(1) + "-15"
        with open(fp, "rb") as f:
            d = pickle.load(f)
        items = sorted(d.items(), key=lambda kv: -kv[1])
        if top:
            items = items[:top]
        for s, c in items:
            if s:
                rows.append((label, frozenset(s), float(c)))
        days.append(label)
    days = sorted(set(days))
    day_ix = {d: i for i, d in enumerate(days)}
    events = [(s, day_ix[d], w) for d, s, w in sorted(rows, key=lambda r: r[0])]
    V = 1 + max(max(s) for s, _, _ in events if s)
    if verbose:
        print(f"[monthly fallback] events {len(events):,}  months {len(days)}  V {V}")
    return {"events": events, "days": days, "V": V}


def parse_posres(vocab_path, V):
    """Split mutation names into (position, residue) ids for shared embeddings.

    Genuinely new mutations have no memory, so their score can only come from
    features. Position and residue are the features available.
    """
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
            pos.append(-(i + 1))
            res.append("?")
    up = {p: k for k, p in enumerate(sorted(set(pos)))}
    ur = {c: k for k, c in enumerate(sorted(set(res)))}
    return (torch.tensor([up[p] for p in pos]),
            torch.tensor([ur[c] for c in res]), len(up), len(ur), n_ok)


# ======================================================================
# MODEL
# ======================================================================

class FourierTime(nn.Module):
    """Learned Fourier features of an elapsed duration (Xu et al. 2020)."""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(
            1.0 / 10 ** np.linspace(0, 3, d)).float())
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, dt):
        return torch.cos(dt.unsqueeze(-1) * self.w + self.b)


class Memory(nn.Module):
    """Per-mutation state, GRU-updated in batches, with explicit forgetting.

    Their memory has no decay, which is right for a stationary network. This
    population turns over completely at sweeps, so state from before a sweep
    should not carry at full strength: memory is scaled by gamma^elapsed
    before each update, with gamma learned.
    """

    def __init__(self, V, d, msg_dim, decay=True):
        super().__init__()
        self.V, self.d = V, d
        self.gru = nn.GRUCell(msg_dim, d)
        self.decay = decay
        self.log_gamma = nn.Parameter(torch.tensor(-3.0))   # slow forgetting
        self.register_buffer("mem", torch.zeros(V, d))
        self.register_buffer("last_t", torch.zeros(V))

    def reset(self):
        self.mem.zero_()
        self.last_t.zero_()

    def read(self, idx, t_now):
        m = self.mem[idx]
        if self.decay:
            dt = (t_now - self.last_t[idx]).clamp_min(0.0)
            gamma = torch.sigmoid(self.log_gamma) * 0.5 + 0.5   # in (0.5, 1)
            m = m * gamma.pow(dt).unsqueeze(-1)
        return m

    @torch.no_grad()
    def write(self, idx, new_mem, t_now):
        self.mem[idx] = new_mem.detach()
        self.last_t[idx] = t_now

    def update(self, idx, msg, t_now):
        """Return the grad-carrying new memory; store a detached copy.

        The caller must USE the returned tensor in the current step's loss,
        otherwise the GRU receives no gradient at all -- writing to the buffer
        detaches, which cuts the graph. This is why messages are deferred by
        one step (see VariantTPP.flush_pending).
        """
        cur = self.read(idx, t_now)
        new = self.gru(msg, cur)
        self.write(idx, new, t_now)
        return new


class NeighborhoodAttention(nn.Module):
    """Attend over a mutation's recent co-occurring variants.

    Avoids memory staleness: a mutation with no recent event still gets a
    representation from the mutations it last appeared alongside.
    """

    def __init__(self, d, heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(2 * d, heads, batch_first=True)
        self.proj = nn.Linear(2 * d, d)

    def forward(self, q, ctx, mask):
        # q (B, 2d), ctx (B, N, 2d), mask (B, N) True where padded
        out, _ = self.attn(q.unsqueeze(1), ctx, ctx,
                           key_padding_mask=mask, need_weights=False)
        return self.proj(out.squeeze(1))


class PosResEmbed(nn.Module):
    """E[m] = pos_emb[position(m)] + res_emb[residue(m)].

    The only signal available for mutations seen for the first time.
    """

    def __init__(self, pos_id, res_id, n_pos, n_res, d):
        super().__init__()
        self.register_buffer("pos_id", pos_id)
        self.register_buffer("res_id", res_id)
        self.pos = nn.Embedding(n_pos, d)
        self.res = nn.Embedding(n_res, d)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.res.weight, std=0.02)

    def forward(self, idx):
        return self.pos(self.pos_id[idx]) + self.res(self.res_id[idx])


class HyperSAGNNScorer(nn.Module):
    """Score a variant from its member mutations.

        s_i = W_s v_i                       static: the mutation alone
        d_i = SelfAttention({v_j})           dynamic: the mutation in company
        score = mean_i W_o (d_i - s_i)^2 + b

    The score measures how much each mutation's representation is CHANGED by
    the set it sits in. A mutation that belongs looks the same in context as
    out of it. This is an interaction term -- it is exactly what an additive
    encoder cannot express.
    """

    def __init__(self, d, heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.W_s = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, 1)
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, X, mask):
        # X (B, K, d) member representations, mask (B, K) True where padded
        dyn, _ = self.attn(X, X, X, key_padding_mask=mask, need_weights=False)
        stat = self.W_s(X)
        per = self.W_o((dyn - stat) ** 2).squeeze(-1)        # (B, K)
        valid = (~mask).float()
        return (per * valid).sum(1) / valid.sum(1).clamp_min(1.0)


class VariantTPP(nn.Module):
    """Node state + set scorer.

    Node representations are recomputed ONCE per timestep and cached: every
    variant on the same day shares the same v_i(t), so recomputing per variant
    was pure waste. Recent-neighbour history is held in preallocated ring
    buffers so the context lookup is tensor indexing rather than a Python loop
    -- this is the difference between minutes and hours per epoch.
    """

    def __init__(self, V, d=64, heads=2, n_recent=10, posres=None,
                 decay=True):
        super().__init__()
        self.V, self.d, self.N = V, d, n_recent
        self.time = FourierTime(d)
        self.mem = Memory(V, d, 3 * d, decay=decay)
        self.nbr = NeighborhoodAttention(d, heads)
        self.posres = posres
        self.W_s = nn.Linear(d, d)
        self.W_n = nn.Linear(d, d)
        self.b_v = nn.Parameter(torch.zeros(d))
        self.scorer = HyperSAGNNScorer(d, heads)
        # variant-level heads: growth (selection) and death
        self.pool = nn.Linear(d, d)
        self.fitness = nn.Sequential(nn.Linear(d + 2, d), nn.Tanh(),
                                     nn.Linear(d, 1))
        self.death = nn.Sequential(nn.Linear(d + 2, d), nn.Tanh(),
                                   nn.Linear(d, 1))
        for h in (self.fitness, self.death):
            nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias)
        # ring buffers for recent co-occurrence context
        self.register_buffer("nbr_vec", torch.zeros(V, n_recent, d))
        self.register_buffer("nbr_t", torch.zeros(V, n_recent))
        self.register_buffer("nbr_cnt", torch.zeros(V, dtype=torch.long))
        self._cache_t = None
        self._cache_v = None
        self._pending = None          # (idx, msg) from the previous step
        self._override = None         # (idx, grad-carrying memory)

    def reset_state(self):
        self.mem.reset()
        self.nbr_vec.zero_(); self.nbr_t.zero_(); self.nbr_cnt.zero_()
        self._cache_t = None; self._cache_v = None
        self._pending = None; self._override = None

    def flush_pending(self, t_now):
        """Apply the previous step's messages inside THIS step's graph.

        TGN's raw-message-store trick. Messages generated at step t-1 update
        memory at step t, and the resulting tensor is used for step t's
        representations -- so the GRU receives gradient from step t's loss.
        Applying the update at step t-1 and reading it back at step t would
        detach and leave the GRU untrained.
        """
        self._override = None
        if self._pending is None:
            return
        idx, msg = self._pending
        new = self.mem.update(idx, msg, t_now)
        self._override = (idx, new)
        self._pending = None
        self._cache_t = None

    # ---- node representations -----------------------------------------
    def all_node_repr(self, t_now):
        """v_i(t) for every mutation, computed once per timestep."""
        idx = torch.arange(self.V)
        m = self.mem.read(idx, t_now)                       # (V, d)
        if self._override is not None:
            oi, ov = self._override
            m = m.index_copy(0, oi, ov)      # grad flows to the GRU
        dt = (t_now - self.nbr_t).clamp_min(0.0)            # (V, N)
        ctx = torch.cat([self.nbr_vec, self.time(dt)], dim=-1)   # (V, N, 2d)
        ar = torch.arange(self.N).unsqueeze(0)
        mask = ar >= self.nbr_cnt.clamp(max=self.N).unsqueeze(1)
        mask[mask.all(dim=1), 0] = False        # never fully mask a row
        q = torch.cat([m, self.time(torch.zeros(self.V))], dim=-1)
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

    # ---- scoring --------------------------------------------------------
    def score_variants(self, variants, t_now, max_k=64):
        """variants: list of frozensets -> (B,) scores."""
        Vrep = self.node_repr_cached(t_now)
        B = len(variants)
        K = max(1, min(max_k, max((len(s) for s in variants), default=1)))
        idx = torch.zeros(B, K, dtype=torch.long)
        mask = torch.ones(B, K, dtype=torch.bool)
        for b, s in enumerate(variants):
            ms = list(s)[:K]
            if not ms:
                mask[b, 0] = False
                continue
            idx[b, :len(ms)] = torch.tensor(ms)
            mask[b, :len(ms)] = False
        X = Vrep[idx]                                        # (B, K, d)
        return self.scorer(X, mask)

    # ---- memory update --------------------------------------------------
    def observe(self, variants, t_now, max_k=64):
        """Write a batch of observed variants into memory and history."""
        seen = sorted({m for s in variants for m in s})
        if not seen:
            return
        Vrep = self.node_repr_cached(t_now)
        idx = torch.tensor(seen)
        pos = {m: i for i, m in enumerate(seen)}
        v = Vrep[idx]
        agg = torch.zeros(len(seen), self.d)
        cnt = torch.zeros(len(seen), 1)
        for s in variants:
            ms = list(s)[:max_k]
            if not ms:
                continue
            rows = torch.tensor([pos[m] for m in ms])
            ctx = Vrep[torch.tensor(ms)].mean(0, keepdim=True)
            agg.index_add_(0, rows, ctx.expand(len(rows), -1))
            cnt.index_add_(0, rows, torch.ones(len(rows), 1))
        agg = agg / cnt.clamp_min(1.0)
        dt = (t_now - self.mem.last_t[idx]).clamp_min(0.0)
        msg = torch.cat([v, agg, self.time(dt)], dim=-1)
        # Detach the message: it belongs to this step's graph, and applying it
        # next step would backward through a freed graph. The GRU still trains,
        # since gradient w.r.t. its weights does not require the input to carry
        # grad -- only the OUTPUT must enter the next step's loss.
        self._pending = (idx, msg.detach())        # applied at the next step
        with torch.no_grad():
            slot = (self.nbr_cnt[idx] % self.N)
            self.nbr_vec[idx, slot] = agg.detach()
            self.nbr_t[idx, slot] = t_now
            self.nbr_cnt[idx] += 1
        self._cache_t = None          # state changed; representations stale
    # ---- selection: growth and death of circulating variants -----------
    def variant_repr(self, variants, t_now, max_k=64):
        """Pool member representations into one vector per variant."""
        Vrep = self.node_repr_cached(t_now)
        B = len(variants)
        K = max(1, min(max_k, max((len(s) for s in variants), default=1)))
        idx = torch.zeros(B, K, dtype=torch.long)
        m = torch.zeros(B, K, 1)
        for b, s in enumerate(variants):
            ms = list(s)[:K]
            idx[b, :len(ms)] = torch.tensor(ms) if ms else 0
            m[b, :len(ms)] = 1.0
        X = Vrep[idx] * m
        return torch.tanh(self.pool(X.sum(1) / m.sum(1).clamp_min(1.0)))

    def predict_shares(self, variants, mass, t_now, dt):
        """Predicted share of the population at t+dt, as a softmax.

        Competition is implicit and zero-sum: shares are a softmax over the
        circulating set, so one variant can only rise if others fall. This is
        the replicator structure without any pairwise interaction parameters --
        which we cannot identify from the handful of genuine co-circulation
        episodes in this data.

        The fitness head is zero-initialised, so at the start the prediction is
        exactly the current share: persistence, which the model then learns to
        deviate from.
        """
        z = self.variant_repr(variants, t_now)
        logm = torch.log(mass.clamp_min(1e-9)).unsqueeze(-1)
        feat = torch.cat([z, logm, torch.full_like(logm, float(dt))], dim=-1)
        fit = self.fitness(feat).squeeze(-1)
        return torch.log_softmax(logm.squeeze(-1) + dt * fit, dim=0), fit

    def predict_death(self, variants, mass, t_now, dt):
        z = self.variant_repr(variants, t_now)
        logm = torch.log(mass.clamp_min(1e-9)).unsqueeze(-1)
        feat = torch.cat([z, logm, torch.full_like(logm, float(dt))], dim=-1)
        return self.death(feat).squeeze(-1)


# ======================================================================
# CANDIDATE GENERATION  (adaptation 1)
# ======================================================================

def generate_candidates(circulating, active_mutations, radius=1,
                        max_cand=4000, rng=None, exclude=None):
    """Circulating variants perturbed by a few currently active mutations.

    Script 150 measured that a newly appearing variant is a median of 1
    mutation from the nearest circulating one. So the true variant is almost
    always inside this set, and no learned generator is needed -- unlike their
    adjacency predictor, whose recall they report as the bottleneck.
    """
    rng = rng or random.Random(0)
    exclude = exclude or set()
    out = set()
    circ = list(circulating)
    acts = list(active_mutations)
    if not circ or not acts:
        return []
    while len(out) < max_cand:
        base = circ[rng.randrange(len(circ))]
        cand = set(base)
        for _ in range(rng.randint(1, radius)):
            if cand and rng.random() < 0.35:
                cand.discard(rng.choice(list(cand)))
            else:
                cand.add(acts[rng.randrange(len(acts))])
        fs = frozenset(cand)
        if fs and fs not in exclude:
            out.add(fs)
        if len(out) >= max_cand:
            break
    return list(out)


# ======================================================================
# BASELINES  (adaptation: proximity, which they do not run)
# ======================================================================

class CirculatingIndex:
    """Sparse incidence matrix of the circulating set, rebuilt once per day.

    Nearest-neighbour distance for every candidate is then one sparse matmul:
        |s XOR c| = |s| + |c| - 2 |s AND c|
    rather than a Python double loop over candidates x circulating. Same
    numbers, and it is the difference between seconds and minutes per day at
    V ~ 5000 with hundreds of circulating variants.
    """

    def __init__(self, circulating, V):
        import scipy.sparse as sp
        rows, cols = [], []
        self.sizes = []
        for i, c in enumerate(circulating):
            self.sizes.append(len(c))
            for m in c:
                if m < V:
                    rows.append(i); cols.append(m)
        self.n = len(circulating)
        self.V = V
        self.M = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(max(self.n, 1), V))
        self.sizes = np.array(self.sizes, dtype=np.float32) \
            if self.sizes else np.zeros(1, dtype=np.float32)

    def nearest(self, variants):
        """Negative Hamming distance to the closest circulating variant."""
        import scipy.sparse as sp
        if self.n == 0:
            return np.full(len(variants), -1e9)
        rows, cols, sz = [], [], []
        for i, s in enumerate(variants):
            sz.append(len(s))
            for m in s:
                if m < self.V:
                    rows.append(i); cols.append(m)
        Q = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(variants), self.V))
        inter = np.asarray((Q @ self.M.T).todense())          # (B, n)
        dist = (np.array(sz, dtype=np.float32)[:, None]
                + self.sizes[None, :] - 2.0 * inter)
        return -dist.min(axis=1)


def baseline_scores(variants, circ_index, last_seen, freq, t_now):
    """Three reference rankings.

    recency   : how recently this exact variant was observed
    frequency : observed abundance of this exact variant
    proximity : negative distance to the nearest circulating variant.
                This is the one that matters -- our impossibility result
                showed any factorized mixture reduces to nearest-mode
                ranking, so a model that does not beat proximity has learned
                a distance heuristic with extra machinery.
    """
    rec = np.array([-(t_now - last_seen[s]) if s in last_seen else -1e9
                    for s in variants], dtype=float)
    frq = np.array([freq.get(s, 0.0) for s in variants], dtype=float)
    prox = circ_index.nearest(variants).astype(float)
    return rec, frq, prox


def mrr_from_scores(scores, true_index):
    """Reciprocal rank of the true item, ties given the AVERAGE rank.

    This matters more than it looks. A baseline that assigns every candidate
    the same score -- recency and frequency do exactly that when the true
    variant is new, since neither it nor any negative has been seen -- would
    otherwise be handed rank 1 and score a perfect MRR. Averaging over ties
    puts an uninformative ranker at chance, which is where it belongs.
    """
    s = np.asarray(scores, dtype=float)
    t = s[true_index]
    n_greater = int((s > t).sum())
    n_equal = int((s == t).sum())          # includes the true item itself
    rank = n_greater + (n_equal + 1) / 2.0
    return 1.0 / rank


# ======================================================================
# TRAIN / EVAL
# ======================================================================

def group_by_day(events):
    by_day = defaultdict(list)
    for s, t, w in events:
        by_day[t].append((s, w))
    return by_day


def mass_by_day(by_day):
    """Normalised abundance of each variant on each day."""
    out = {}
    for t, batch in by_day.items():
        agg = defaultdict(float)
        for s, w in batch:
            agg[s] += w
        tot = sum(agg.values()) or 1.0
        out[t] = {s: v / tot for s, v in agg.items()}
    return out


def selection_step(model, circ_mass, mass_map, t, t_tgt, death_thresh,
                   max_c=256, train=True, opt=None):
    """One growth+death step: predict shares and extinction at t_tgt.

    Returns (loss, records) where records hold predicted vs observed log
    growth and the death label, for evaluation against baselines.
    """
    items = sorted(circ_mass.items(), key=lambda kv: -kv[1])[:max_c]
    if len(items) < 2:
        return None, []
    variants = [s for s, _ in items]
    mass = torch.tensor([m for _, m in items], dtype=torch.float32)
    dt = max(1.0, float(t_tgt - t))

    tgt_map = mass_map.get(t_tgt, {})
    obs = torch.tensor([tgt_map.get(s, 0.0) for s in variants],
                       dtype=torch.float32)
    if obs.sum() <= 0:
        return None, []
    obs_share = obs / obs.sum()
    dead = (obs < death_thresh).float()

    log_pred, fit = model.predict_shares(variants, mass, float(t), dt)
    d_logit = model.predict_death(variants, mass, float(t), dt)

    # cross-entropy between observed and predicted share = KL up to a constant
    loss_growth = -(obs_share * log_pred).sum()
    loss_death = F.binary_cross_entropy_with_logits(d_logit, dead)
    loss = loss_growth + loss_death
    if train:
        # Return the loss TENSOR; the caller sums birth and every selection
        # horizon into a single backward per day. Backpropagating each horizon
        # separately would traverse the shared node-representation graph twice.
        return loss, []

    with torch.no_grad():
        pred_share = log_pred.exp().numpy()
        cur = mass.numpy()
        obs_s = obs_share.numpy()
        recs = []
        for j, s in enumerate(variants):
            recs.append({
                "t": t, "dt": dt, "size": len(s),
                "mass": float(cur[j]),
                "pred_share": float(pred_share[j]),
                "obs_share": float(obs_s[j]),
                "pred_logg": float(np.log(pred_share[j] + 1e-12)
                                   - np.log(cur[j] + 1e-12)),
                "obs_logg": float(np.log(obs_s[j] + 1e-12)
                                  - np.log(cur[j] + 1e-12)),
                "dead": float(dead[j]),
                "death_score": float(torch.sigmoid(d_logit[j])),
            })
    return None, recs


def run(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = random.Random(a.seed)

    D = (events_from_monthly(a.data_dir, top=a.top) if a.events is None
         else load_events(a.events, a.vocab))
    events, days, V = D["events"], D["days"], D["V"]
    by_day = group_by_day(events)
    mass_map = mass_by_day(by_day)
    all_days = sorted(by_day)

    n_train = int(len(all_days) * a.train_frac)
    n_val = int(len(all_days) * a.val_frac)
    train_days = all_days[:n_train]
    val_days = all_days[n_train:n_train + n_val]
    test_days = all_days[n_train + n_val:]
    print(f"split: train {len(train_days)} days ({days[train_days[0]]}.."
          f"{days[train_days[-1]]})  val {len(val_days)}  "
          f"test {len(test_days)} ({days[test_days[0]]}..{days[test_days[-1]]})")

    posres = None
    if a.posres:
        p, r, npos, nres, nok = parse_posres(a.vocab, V)
        print(f"posres: parsed {nok}/{V} -> {npos} positions, {nres} residues")
        posres = PosResEmbed(p, r, npos, nres, a.d)

    model = VariantTPP(V, d=a.d, heads=a.heads, n_recent=a.n_recent,
                       posres=posres, decay=not a.no_decay)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def stream(day_list, train=True, collect=None, sel_collect=None,
               window=a.window):
        """One chronological pass. Optionally collect evaluation records."""
        model.reset_state()
        circulating, last_seen, freq = deque(maxlen=a.circ_max), {}, {}
        circ_mass = {}
        seen_ever = set()
        # warm the state on everything before the first evaluated day
        for t in all_days:
            if t >= day_list[0] and not train:
                break
            batch = by_day[t]
            model.flush_pending(float(t))
            model.observe([s for s, _ in batch], float(t))
            for s, w in batch:
                circulating.append(s); last_seen[s] = t; freq[s] = w
                seen_ever.add(s)
            circ_mass = dict(mass_map.get(t, {}))
            if train and t >= day_list[-1]:
                break

        losses = []
        for t in day_list:
            model.flush_pending(float(t))
            batch = by_day[t]
            pos = [s for s, _ in batch]
            wts = np.array([w for _, w in batch], dtype=float)
            wts = wts / max(wts.sum(), 1.0)

            cands = generate_candidates(
                list(circulating), sorted({m for s in circulating for m in s}),
                radius=a.radius, max_cand=a.n_cand, rng=rng,
                exclude=set(pos))
            if not cands:
                continue
            k = min(len(pos), a.max_pos)
            sel = rng.sample(range(len(pos)), k)
            P = [pos[i] for i in sel]
            Pw = torch.tensor(wts[sel], dtype=torch.float32)
            Ncount = min(len(cands), a.n_neg * k)
            N = rng.sample(cands, Ncount)

            # ---- selection: growth and death of what is already here ----
            sel_losses = []
            if circ_mass:
                for dt in a.horizons:
                    nxt = [u for u in all_days if u >= t + dt]
                    if not nxt:
                        continue
                    t_tgt = nxt[0]
                    if train:
                        sl, _ = selection_step(
                            model, circ_mass, mass_map, t, t_tgt,
                            a.death_thresh, a.max_circ_sel, train=True)
                        if sl is not None:
                            sel_losses.append(sl)
                    elif sel_collect is not None:
                        _, rs = selection_step(model, circ_mass, mass_map, t,
                                               t_tgt, a.death_thresh,
                                               a.max_circ_sel, train=False)
                        sel_collect.extend(rs)

            sc = model.score_variants(P + N, float(t))
            lp, ln = sc[:k], sc[k:]
            if train:
                # adaptation 3: positives weighted by observed abundance
                loss = (F.binary_cross_entropy_with_logits(
                            lp, torch.ones_like(lp), reduction="none")
                        * (1.0 + a.w_scale * Pw)).mean()
                loss = loss + F.binary_cross_entropy_with_logits(
                    ln, torch.zeros_like(ln))
                for sl in sel_losses:            # birth + selection, one step
                    loss = loss + a.sel_weight * sl
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach()))
            elif collect is not None:
                circ_index = CirculatingIndex(list(circulating), V)
                with torch.no_grad():
                    for j, s in enumerate(P):
                        pool = [s] + N[:a.n_neg]
                        ms = model.score_variants(pool, float(t)).numpy()
                        rec, frq, prox = baseline_scores(
                            pool, circ_index, last_seen, freq, t)
                        collect.append({
                            "day": days[t], "t": t,
                            "new": s not in seen_ever,
                            "size": len(s),
                            "mrr_model": mrr_from_scores(ms, 0),
                            "mrr_recency": mrr_from_scores(rec, 0),
                            "mrr_freq": mrr_from_scores(frq, 0),
                            "mrr_prox": mrr_from_scores(prox, 0),
                        })
            # advance state
            model.observe(pos, float(t))
            for s, w in batch:
                circulating.append(s); last_seen[s] = t; freq[s] = w
                seen_ever.add(s)
            circ_mass = dict(mass_map.get(t, {}))
        return float(np.mean(losses)) if losses else float("nan")

    for ep in range(a.epochs):
        tr = stream(train_days, train=True)
        print(f"epoch {ep+1}/{a.epochs}  loss {tr:.4f}", flush=True)

    recs, sel = [], []
    stream(test_days, train=False, collect=recs, sel_collect=sel)
    report(recs, a)
    report_selection(sel, a)
    return recs, sel


def report(recs, a):
    if not recs:
        print("no test records"); return
    keys = ["mrr_model", "mrr_prox", "mrr_recency", "mrr_freq"]
    lbl = {"mrr_model": "MODEL", "mrr_prox": "proximity",
           "mrr_recency": "recency", "mrr_freq": "frequency"}

    def block(title, sub):
        if not sub:
            return
        print(f"\n{title}  (n={len(sub)})")
        print(f"  {'method':<12}{'MRR':>8}")
        for k in keys:
            print(f"  {lbl[k]:<12}{np.mean([r[k] for r in sub]):>8.4f}")

    print("\n" + "=" * 66)
    print(f"MRR against {a.n_neg} hard negatives per event "
          "(candidates from the same generator that did not appear)")
    block("ALL test events", recs)
    block("NEW variants (never seen before)", [r for r in recs if r["new"]])
    block("REPEAT variants", [r for r in recs if not r["new"]])

    # by calendar time -- exposes regime shifts, which their protocol hides
    print("\nMRR over calendar time")
    print(f"  {'month':<10}{'n':>6}{'MODEL':>9}{'proximity':>11}")
    bym = defaultdict(list)
    for r in recs:
        bym[r["day"][:7]].append(r)
    for mth in sorted(bym):
        sub = bym[mth]
        print(f"  {mth:<10}{len(sub):>6}"
              f"{np.mean([r['mrr_model'] for r in sub]):>9.4f}"
              f"{np.mean([r['mrr_prox'] for r in sub]):>11.4f}")

    # by lead time from the first test day
    t0 = min(r["t"] for r in recs)
    bins = [(0, 7), (8, 30), (31, 90), (91, 180), (181, 10 ** 9)]
    print("\nMRR by forecast lead time (days from first test day)")
    print(f"  {'lead':<12}{'n':>6}{'MODEL':>9}{'proximity':>11}")
    for lo, hi in bins:
        sub = [r for r in recs if lo <= r["t"] - t0 <= hi]
        if not sub:
            continue
        name = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        print(f"  {name:<12}{len(sub):>6}"
              f"{np.mean([r['mrr_model'] for r in sub]):>9.4f}"
              f"{np.mean([r['mrr_prox'] for r in sub]):>11.4f}")

    print("\nOne seed. Run several --seed values before treating any "
          "difference as a result.")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(recs, f)
        print(f"wrote {a.out}")


def report_selection(recs, a):
    """Growth and death, against the baselines that could beat them.

    Birth (which variants appear) is only part of forecasting. Most hyperedges
    persist; what changes is their mass. These two tasks measure whether the
    model tracks that -- and whether it beats simply assuming nothing changes.
    """
    if not recs:
        print("\nno selection records"); return
    print("\n" + "=" * 66)
    print("SELECTION: growth and death of already-circulating variants")

    for dt in sorted({r["dt"] for r in recs}):
        sub = [r for r in recs if r["dt"] == dt]
        pg = np.array([r["pred_logg"] for r in sub])
        og = np.array([r["obs_logg"] for r in sub])
        ok = np.isfinite(pg) & np.isfinite(og)
        pg, og = pg[ok], og[ok]
        if pg.size < 3:
            continue
        # persistence: predict zero log growth, i.e. share unchanged
        mae_model = float(np.abs(pg - og).mean())
        mae_pers = float(np.abs(og).mean())
        r_p = float(np.corrcoef(pg, og)[0, 1]) if pg.std() > 0 else float("nan")
        from scipy.stats import spearmanr
        try:
            r_s = float(spearmanr(pg, og).statistic)
        except Exception:
            r_s = float("nan")

        d = np.array([r["dead"] for r in sub])
        sc = np.array([r["death_score"] for r in sub])
        ms = np.array([r["mass"] for r in sub])
        auc = _auc(d, sc)
        auc_mass = _auc(d, -ms)      # baseline: rare variants die
        print(f"\n--- horizon {int(dt)} days   (n={len(sub)}, "
              f"{d.mean():.1%} died) ---")
        print(f"  growth  MAE log-growth   model {mae_model:8.4f}   "
              f"persistence {mae_pers:8.4f}")
        print(f"          corr(pred, obs)  pearson {r_p:7.3f}   "
              f"spearman {r_s:7.3f}")
        print(f"  death   AUC              model {auc:8.3f}   "
              f"rarity baseline {auc_mass:8.3f}")


def _auc(y, s):
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", default=None,
                   help="TSV: date <TAB> comma-separated mutation ids <TAB> count")
    p.add_argument("--data-dir", default=None,
                   help="fallback: directory of monthly *_occupied.pkl")
    p.add_argument("--vocab", default=None)
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--n-recent", type=int, default=10, dest="n_recent")
    p.add_argument("--posres", action="store_true")
    p.add_argument("--no-decay", action="store_true", dest="no_decay",
                   help="disable memory forgetting (their original)")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-cand", type=int, default=2000, dest="n_cand")
    p.add_argument("--n-neg", type=int, default=20, dest="n_neg")
    p.add_argument("--max-pos", type=int, default=64, dest="max_pos")
    p.add_argument("--circ-max", type=int, default=2000, dest="circ_max")
    p.add_argument("--w-scale", type=float, default=5.0, dest="w_scale")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=0.5, dest="train_frac")
    p.add_argument("--val-frac", type=float, default=0.25, dest="val_frac")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizons", type=int, nargs="+", default=[7, 30],
                   help="days ahead for growth/death prediction")
    p.add_argument("--death-thresh", type=float, default=1e-4,
                   dest="death_thresh",
                   help="mass below which a variant counts as dead; a variant "
                        "below detection is not necessarily extinct, so this "
                        "threshold must be stated with any death result")
    p.add_argument("--max-circ-sel", type=int, default=256,
                   dest="max_circ_sel")
    p.add_argument("--sel-weight", type=float, default=1.0, dest="sel_weight",
                   help="weight of the growth+death loss relative to birth")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if a.inspect:
        if a.events:
            load_events(a.events, a.vocab)
        else:
            events_from_monthly(a.data_dir, top=a.top)
        return
    run(a)


if __name__ == "__main__":
    main()
