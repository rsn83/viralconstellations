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


def load_first_seen(vocab_path, V, cutoff):
    """Mask of mutations already seen by the training cutoff.

    Mutations first appearing after the cutoff keep their vocabulary slot but
    must not carry learned memory -- their only signal is position/residue.
    """
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
                    seen[i] = False
                    n_late += 1
    print(f"  {n_late:,} mutations first appear after {cutoff}; "
          "memory suppressed, posres only")
    return seen


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
                    names[int(p[0])] = p[1]      # col 3, if present, is
                else:                            # the first-seen date
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


class PopulationEncoder(nn.Module):
    """Pool the circulating variants into one state u_t.

    157 had only per-mutation memory: the population was dissolved into node
    states and never reassembled, so nothing conditioned on the configuration
    of variants as a whole. This is the piece 156 had and 157 lost.
    """

    def __init__(self, d, heads=2, layers=1):
        super().__init__()
        if layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=2 * d,
                dropout=0.0, batch_first=True)
            self.attn = nn.TransformerEncoder(layer, num_layers=layers)
        else:
            self.attn = None
        self.out = nn.Sequential(nn.Linear(d, d), nn.Tanh())

    def forward(self, X, w):
        # X (n, d) variant representations, w (n,) masses
        x = X.unsqueeze(0)
        if self.attn is not None:
            xa = self.attn(x)
            x = xa if torch.isfinite(xa).all() else x
        x = x.squeeze(0)
        u = (w.unsqueeze(-1) * x).sum(0)
        return self.out(torch.nan_to_num(u, nan=0.0))


class Generator(nn.Module):
    """Fully learned candidate generation. No enumeration, no fixed radius.

    Three heads, all trained by the forecasting loss because every candidate's
    score contains the log-probabilities used to generate it:

      size      p(k | B, u, dt)              how many mutations to add
      add       p(m | B, added, u, dt)       autoregressive over the k additions
      free      p(m | u, dt)                 unanchored, population-wide

    The anchored path (size + add) preserves the joint structure of a
    background: a 26-mutation combination is carried, not reconstructed. The
    free path can assemble combinations that have never co-occurred, so reach
    is not capped by any radius -- which is what the anchored path alone cannot
    do, and what matters once the population has moved far from the origin.

    Which path earns mass is decided by the loss, not by a flag: both feed the
    same support and the same softmax. mix_logit sets the split and is
    conditioned on dt, since the anchored path should dominate at short
    horizons and lose ground as the population drifts.
    """

    def __init__(self, d, V, max_add=4):
        super().__init__()
        self.V, self.max_add = V, max_add
        cin = 3 * d + 1
        self.size_head = nn.Sequential(
            nn.Linear(cin, 2 * d), nn.Tanh(), nn.Linear(2 * d, max_add))
        self.add_head = nn.Sequential(
            nn.Linear(cin + d, 2 * d), nn.Tanh(), nn.Linear(2 * d, d))
        self.free_head = nn.Sequential(
            nn.Linear(d + 1, 2 * d), nn.Tanh(), nn.Linear(2 * d, d))
        self.free_size = nn.Sequential(
            nn.Linear(d + 1, 2 * d), nn.Tanh(), nn.Linear(2 * d, max_add))
        self.mut_bias = nn.Parameter(torch.zeros(V))
        self.mix = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
        nn.init.zeros_(self.mix[-1].weight); nn.init.zeros_(self.mix[-1].bias)

    def _ctx(self, zB, u, dt):
        n = zB.shape[0]
        return torch.cat([zB, u.unsqueeze(0).expand(n, -1),
                          zB * u.unsqueeze(0),
                          torch.full((n, 1), float(dt))], dim=-1)

    def size_logp(self, zB, u, dt):
        return torch.log_softmax(self.size_head(self._ctx(zB, u, dt)), dim=-1)

    def add_logp(self, zB, u, dt, added, table):
        """log p(next mutation | background, mutations added so far)."""
        q = self.add_head(torch.cat([self._ctx(zB, u, dt), added], dim=-1))
        return torch.log_softmax(q @ table.T + self.mut_bias.unsqueeze(0),
                                 dim=-1)

    def free_logp(self, u, dt, table):
        c = torch.cat([u, torch.tensor([float(dt)])]).unsqueeze(0)
        q = self.free_head(c)
        return (torch.log_softmax(q @ table.T + self.mut_bias.unsqueeze(0),
                                  dim=-1).squeeze(0),
                torch.log_softmax(self.free_size(c), dim=-1).squeeze(0))

    def anchored_logp(self, zB, u, dt, table, adds):
        """Exact log-probability of adding the SET `adds` to each background.

        Batched over backgrounds: the autoregressive chain is at most max_add
        steps, so this is max_add batched matmuls rather than one small matmul
        per candidate per step. Same numbers, orders of magnitude faster.

        Order-invariant: a set of k additions can be produced in k! orders, so
        the chain log-probability gets + log k!. Without it the model would
        prefer small sets for a reason unrelated to the data.
        """
        n, dev = zB.shape[0], zB.device
        ctx = self._ctx(zB, u, dt)                       # (n, 3d+1)
        slp = torch.log_softmax(self.size_head(ctx), dim=-1)

        ks = torch.tensor([len(x) for x in adds], dtype=torch.long)
        valid = (ks >= 1) & (ks <= self.max_add)
        out = torch.full((n,), -30.0, device=dev)
        if not bool(valid.any()):
            return out

        # sorted addition lists, padded to max_add
        seq = torch.zeros(n, self.max_add, dtype=torch.long)
        for i, x in enumerate(adds):
            if valid[i]:
                ms = sorted(x)[:self.max_add]
                seq[i, :len(ms)] = torch.tensor(ms, dtype=torch.long)

        lp = torch.where(valid, slp.gather(
            1, (ks.clamp(1, self.max_add) - 1).unsqueeze(1)).squeeze(1),
            torch.zeros(n, device=dev))

        added = torch.zeros(n, table.shape[1], device=dev)
        kf = ks.clamp_min(1).float().unsqueeze(-1)
        for j in range(self.max_add):
            live = valid & (ks > j)
            if not bool(live.any()):
                break
            q = self.add_head(torch.cat([ctx, added], dim=-1))
            step = torch.log_softmax(
                q @ table.T + self.mut_bias.unsqueeze(0), dim=-1)   # (n, V)
            got = step.gather(1, seq[:, j].unsqueeze(1)).squeeze(1)
            lp = lp + torch.where(live, got, torch.zeros_like(got))
            added = added + torch.where(
                live.unsqueeze(-1), table[seq[:, j]] / kf,
                torch.zeros_like(added))

        logfact = torch.tensor(
            [math.lgamma(max(int(k), 1) + 1) for k in ks], device=dev)
        return torch.where(valid, lp + logfact, out)

    def sample(self, zB, backgrounds, u, dt, table, n_anchor, n_free, rng):
        """Draw candidates from both paths, batched.

        All n_anchor draws advance through the autoregressive chain together,
        so this is max_add batched steps instead of one forward pass per
        mutation per candidate. Generation only -- no gradient here; the
        log-probabilities used in the loss come from anchored_logp.
        """
        cands = []
        with torch.no_grad():
            if n_anchor and len(backgrounds):
                pick = torch.randint(0, len(backgrounds), (n_anchor,))
                Z = zB[pick]                                  # (A, d)
                ctx = self._ctx(Z, u, dt)
                ks = torch.multinomial(
                    torch.log_softmax(self.size_head(ctx), -1).exp(), 1
                ).squeeze(1) + 1                               # (A,)
                added = torch.zeros(n_anchor, table.shape[1])
                chosen = torch.zeros(n_anchor, self.max_add, dtype=torch.long)
                kf = ks.float().unsqueeze(-1)
                for j in range(int(ks.max())):
                    live = ks > j
                    q = self.add_head(torch.cat([ctx, added], dim=-1))
                    p = torch.softmax(
                        q @ table.T + self.mut_bias.unsqueeze(0), dim=-1)
                    mm = torch.multinomial(p, 1).squeeze(1)
                    chosen[:, j] = mm
                    added = added + torch.where(
                        live.unsqueeze(-1), table[mm] / kf,
                        torch.zeros_like(added))
                for r in range(n_anchor):
                    i = int(pick[r]); k = int(ks[r])
                    add = {int(chosen[r, j]) for j in range(k)}
                    add -= set(backgrounds[i])
                    if not add:
                        continue
                    v = frozenset(set(backgrounds[i]) | add)
                    cands.append((v, i, frozenset(add)))
            if n_free:
                flp, fsz = self.free_logp(u, dt, table)
                fp, sp = flp.exp(), fsz.exp()
                base = int(np.median([len(b) for b in backgrounds])) \
                    if backgrounds else 20
                nz = int((fp > 0).sum())
                for _ in range(n_free):
                    k = min(base + int(torch.multinomial(sp, 1)), nz)
                    idx = torch.multinomial(fp, k, replacement=False)
                    cands.append((frozenset(int(x) for x in idx), None, None))
        return cands


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
        self.pop_enc = PopulationEncoder(d, heads, layers=1)
        self.gen = Generator(d, V, max_add=4)
        # horizon-dependent mass budget: 159 showed the share of future mass on
        # variants absent at the origin rises from ~0.5 at 3 months to ~0.9 at
        # 6, so a single scalar cannot be right at every horizon
        self.budget = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
        nn.init.zeros_(self.budget[-1].weight)
        # Identity features. The scorer sees only member mutations, so it has
        # no way to know it has met this exact variant before -- which is why
        # it scored repeats (0.285) no better than novel variants (0.291)
        # while plain recency scored 0.782 on repeats. Three cheap features
        # supply that: log recency, log abundance, and a seen-before flag.
        # how much total mass to allocate to never-seen variants, learned
        # Mass budget for never-seen variants, as sigmoid(new_bias). Starts
        # near 0.05 rather than saturated, so the gradient can move it: 159
        # showed ~50% of mass at 3 months and ~91% at 6 months sits on variants
        # absent at the origin, so this should learn to be LARGE.
        self.new_bias = nn.Parameter(torch.tensor(-3.0))   # budget intercept
        # Identity features: log recency, log abundance, seen-before flag.
        # The scorer sees only member mutations, so it cannot tell it has met
        # this exact variant before -- which is why it scored repeats (0.285)
        # no better than novel ones (0.291) while recency alone got 0.782.
        # Projected into d and added to every member representation BEFORE the
        # attention, not merely added to the final score: a scalar offset can
        # only shift a variant up or down, whereas the model needs to condition
        # set structure on history ("plausible combination AND it is growing").
        # Small random init, not zeros: a zero-initialised projection outputs
        # zero, and its own gradient is proportional to its output, so it would
        # never move. Same dead-initialisation trap as a multiplicative gate
        # started at zero.
        self.ident_proj = nn.Linear(3, d)
        nn.init.normal_(self.ident_proj.weight, std=0.01)
        nn.init.zeros_(self.ident_proj.bias)
        self.ident = nn.Linear(3, 1)
        nn.init.zeros_(self.ident.weight); nn.init.zeros_(self.ident.bias)
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
        self.register_buffer("mem_ok", torch.ones(V, 1))
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
        m = m * self.mem_ok      # zero for mutations unseen by the cutoff
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
    def score_variants(self, variants, t_now, max_k=64, ident=None):
        """variants: list of frozensets -> (B,) scores.

        ident: optional (B, 3) tensor of [log recency, log mass, seen flag].
        """
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
        if ident is not None:
            X = X + self.ident_proj(ident).unsqueeze(1)   # into the attention
        base = self.scorer(X, mask)
        if ident is not None:
            base = base + self.ident(ident).squeeze(-1)   # plus a direct path
        return base

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
    def mutation_table(self, t_now):
        return self.node_repr_cached(t_now)

    def pop_state(self, variants, mass, t_now):
        return self.pop_enc(self.variant_repr(variants, t_now), mass)

    def budget_logit(self, dt):
        return self.new_bias + self.budget(
            torch.tensor([[float(dt)]], dtype=torch.float32)).squeeze()

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

def ident_feats(variants, last_seen, freq, t_now):
    """[log(1+days since last seen), log mass, seen-before] per variant."""
    rows = []
    for s in variants:
        if s in last_seen:
            rows.append([math.log1p(max(0.0, t_now - last_seen[s])),
                         math.log(freq.get(s, 0.0) + 1e-9), 1.0])
        else:
            rows.append([math.log1p(1e4), math.log(1e-9), 0.0])
    return torch.tensor(rows, dtype=torch.float32)


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
                   max_c=256, train=True, opt=None, rng_sel=None):
    """One growth+death step: predict shares and extinction at t_tgt.

    Returns (loss, records) where records hold predicted vs observed log
    growth and the death label, for evaluation against baselines.
    """
    # Top-K by mass alone would report only how well the dominant variants
    # are tracked. Half the sample is drawn from the tail so growth and death
    # can be broken out by mass decile.
    ranked = sorted(circ_mass.items(), key=lambda kv: -kv[1])
    head = ranked[:max_c // 2]
    tail = ranked[max_c // 2:]
    if tail and rng_sel is not None:
        k = min(len(tail), max_c - len(head))
        items = head + [tail[i] for i in
                        rng_sel.sample(range(len(tail)), k)]
    else:
        items = ranked[:max_c]
    if len(items) < 2:
        return None, []
    rank_of = {s: i for i, (s, _) in enumerate(ranked)}
    n_circ = max(1, len(ranked))
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
                "mass_decile": int(10 * rank_of[s] / n_circ),
                "survived": float(obs_s[j] > 0),
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
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = random.Random(a.seed)

    D = load_events(a.events, a.vocab)
    events, days, V = D["events"], D["days"], D["V"]
    by_day = group_by_day(events)
    mass_map = mass_by_day(by_day)
    all_days = sorted(by_day)

    n_tr = int(len(all_days) * a.train_frac)
    train_days = all_days[:n_tr]
    test_days = all_days[n_tr:]
    print(f"split: train {len(train_days)} days "
          f"({days[train_days[0]]}..{days[train_days[-1]]})  "
          f"test {len(test_days)} ({days[test_days[0]]}..{days[test_days[-1]]})")

    posres = None
    if a.posres:
        p_, r_, npos, nres, nok = parse_posres(a.vocab, V)
        print(f"posres: {nok}/{V} parsed -> {npos} positions, {nres} residues")
        posres = PosResEmbed(p_, r_, npos, nres, a.d)

    model = VariantTPP(V, d=a.d, heads=a.heads, n_recent=a.n_recent,
                       posres=posres, decay=not a.no_decay)
    model.mem_ok.copy_(load_first_seen(
        a.vocab, V, days[train_days[-1]]).float().unsqueeze(-1))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def circ_at(t, k):
        return sorted(mass_map.get(t, {}).items(), key=lambda kv: -kv[1])[:k]

    def target(t, h):
        nxt = [u for u in all_days if u >= t + 30 * h]
        return mass_map.get(nxt[0], {}) if nxt else {}

    for ep in range(a.epochs):
        losses, info = [], []
        step_days = train_days[::max(1, a.stride)]
        for t in step_days:
            model.flush_pending(float(t))
            cm = circ_at(t, a.pop_support)
            if len(cm) < 2:
                model.observe([s for s, _ in by_day[t]], float(t)); continue
            total = None
            for h in a.horizons:
                obs = target(t, h)
                if not obs:
                    continue
                l, meta = transition_loss(model, cm, obs, t, float(30 * h),
                                          rng, a, train=True)
                if l is not None:
                    total = l if total is None else total + l
                    info.append(meta)
            if total is not None:
                opt.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(total.detach()))
            model.observe([s for s, _ in by_day[t]], float(t))
        f = lambda k: (np.mean([i[k] for i in info]) if info else float("nan"))
        print(f"epoch {ep+1}/{a.epochs}  loss {np.mean(losses):.4f}  "
              f"budget {f('budget'):.3f}  "
              f"target {f('n_target'):.0f} vars "
              f"({f('target_mass'):.2f} mass)  "
              f"missed/step {f('n_missed'):.1f}/{f('n_target'):.0f}",
              flush=True)

    # ---------------- evaluation: rolling origins -----------------------
    origins = test_days[::max(1, len(test_days) // a.n_origins)][:a.n_origins]
    rows = []
    for T in origins:
        model.reset_state()
        for t in all_days:
            if t > T:
                break
            model.flush_pending(float(t))
            model.observe([s for s, _ in by_day[t]], float(t))
        cm = circ_at(T, a.pop_support)
        if len(cm) < 2:
            continue
        for h in a.horizons:
            obs = target(T, h)
            if not obs:
                continue
            with torch.no_grad():
                _, meta, parts = transition_loss(
                    model, cm, obs, T, float(30 * h), rng, a,
                    train=False, return_parts=True)
            if parts is None:
                continue
            logp, allv, n_c = parts
            r = {"origin": days[T], "h": h,
                 "budget": meta["budget"], "mix": meta["mix_anchored"]}
            r["model"] = score_population(logp.numpy(), allv, n_c, obs)
            for k, lp in population_baselines(
                    cm, allv[n_c:], float(30 * h)).items():
                r[k] = score_population(lp, allv, n_c, obs)
            rows.append(r)

    report_forecast(rows)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"))
        print(f"wrote {a.out}")
    return rows


def report(recs, a):
    """Ranking diagnostics.

    The ALL row is the headline. A model that wins only on never-seen variants
    while losing on repeats is not useful: the question is which variants will
    be circulating, and the answer mixes both. The NEW/REPEAT split below is
    diagnosis of WHERE performance comes from, not the claim.
    """
    if not recs:
        print("no test records"); return
    keys = ["mrr_model", "mrr_prox", "mrr_recency", "mrr_freq"]
    lbl = {"mrr_model": "MODEL", "mrr_prox": "proximity",
           "mrr_recency": "recency", "mrr_freq": "frequency"}

    def block(title, sub, headline=False):
        if not sub:
            return
        w = np.array([r.get("count", 1.0) for r in sub], dtype=float)
        tot = w.sum()
        print(f"\n{title}  (n={len(sub)}, {tot:,.0f} sequences)")
        print(f"  {'method':<12}{'MRR':>9}{'MRR (mass-wtd)':>17}")
        best = None
        for k in keys:
            v = np.array([r[k] for r in sub], dtype=float)
            wm = float(np.average(v, weights=w)) if tot > 0 else float("nan")
            mark = ""
            if headline and k != "mrr_model":
                best = v.mean() if best is None else max(best, v.mean())
            print(f"  {lbl[k]:<12}{v.mean():>9.4f}{wm:>17.4f}{mark}")
        if headline and best is not None:
            mv = np.mean([r["mrr_model"] for r in sub])
            verdict = ("BEATS" if mv > best + 1e-6
                       else ("TIES" if abs(mv - best) <= 1e-6 else "LOSES TO"))
            print(f"  -> MODEL {verdict} the best baseline "
                  f"({mv:.4f} vs {best:.4f})")

    print("\n" + "=" * 66)
    print(f"RANKING DIAGNOSTIC: MRR against {a.n_neg} hard negatives per event")
    print("(candidates from the same generator that did not appear)")
    block("ALL test events  <- the headline", recs, headline=True)
    print("\n--- where it comes from (diagnosis, not the claim) ---")
    block("NEW variants (never seen before)", [r for r in recs if r["new"]])
    block("REPEAT variants", [r for r in recs if not r["new"]])

    def wavg(sub, key):
        w = np.array([r.get("count", 1.0) for r in sub], dtype=float)
        v = np.array([r[key] for r in sub], dtype=float)
        return float(np.average(v, weights=w)) if w.sum() > 0 else float("nan")

    print("\nMRR over calendar time")
    print(f"  {'month':<10}{'n':>7}{'MODEL':>9}{'wtd':>9}"
          f"{'best base':>11}{'wtd':>9}")
    bym = defaultdict(list)
    for r in recs:
        bym[r["day"][:7]].append(r)
    for mth in sorted(bym):
        sub = bym[mth]
        bb = max(np.mean([r[k] for r in sub])
                 for k in keys if k != "mrr_model")
        bbw = max(wavg(sub, k) for k in keys if k != "mrr_model")
        print(f"  {mth:<10}{len(sub):>7}"
              f"{np.mean([r['mrr_model'] for r in sub]):>9.4f}"
              f"{wavg(sub, 'mrr_model'):>9.4f}{bb:>11.4f}{bbw:>9.4f}")

    t0 = min(r["t"] for r in recs)
    bins = [(0, 7), (8, 30), (31, 90), (91, 180), (181, 10 ** 9)]
    print("\nMRR by forecast lead time (days from first test day)")
    print(f"  {'lead':<12}{'n':>7}{'MODEL':>9}{'wtd':>9}"
          f"{'best base':>11}{'wtd':>9}")
    for lo, hi in bins:
        sub = [r for r in recs if lo <= r["t"] - t0 <= hi]
        if not sub:
            continue
        name = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        bb = max(np.mean([r[k] for r in sub])
                 for k in keys if k != "mrr_model")
        bbw = max(wavg(sub, k) for k in keys if k != "mrr_model")
        print(f"  {name:<12}{len(sub):>7}"
              f"{np.mean([r['mrr_model'] for r in sub]):>9.4f}"
              f"{wavg(sub, 'mrr_model'):>9.4f}{bb:>11.4f}{bbw:>9.4f}")

    print("\nOne seed. Run several --seed values before treating any "
          "difference as a result.")


def predict_population(model, circ_mass, candidates, last_seen, freq,
                       t, dt, ident_fn):
    """One predicted distribution over the population at t+dt.

    The forecasting question is: given data up to t, what does the population
    look like at t+dt? The answer is a single distribution over variants, not
    a ranking against sampled negatives -- an MRR depends on how the negatives
    were drawn, which makes it unfalsifiable as a forecasting claim.

    Support = variants circulating at t (UPDATE: present, weight changes)
            + generated candidates      (NEW: absent at t, may appear)
    Logits  = log current mass + dt * fitness   for circulating
            + birth score                       for candidates
    One softmax over the union, so mass is conserved: a new variant can only
    take mass from existing ones.
    """
    circ = [s for s, _ in circ_mass]
    mass = torch.tensor([m for _, m in circ_mass], dtype=torch.float32)
    n_c = len(circ)
    if n_c == 0:
        return None

    with torch.no_grad():
        cands = list(candidates)
        logp, n_c, b = _split_logp(model, circ, mass, cands, t, dt, ident_fn,
                                   last_seen, freq)
        return circ + cands, logp.numpy(), n_c


def _split_logp(model, circ, mass, cands, t, dt, ident_fn,
                last_seen, freq, with_grad=False):
    """Predicted log-distribution, with the mass budget separated from
    composition.

    Two independent decisions, and mixing them was the defect:

      budget      how much total mass goes to variants that do not yet exist
      composition which candidates receive it, and how existing mass shifts

    A single softmax over (existing + candidates) renormalises, so merely
    PROPOSING candidates removes mass from every existing variant. The model
    then cannot express persistence at all -- it pays for its proposals whether
    or not they are any good, which is why it lost to persistence even with
    fitness identically zero.

    Split explicitly:
        b        = sigmoid(new_bias)                       one scalar
        existing = (1 - b) * softmax(log mass + dt*fitness)
        new      = b       * softmax(birth scores)

    Now b -> 0 reproduces persistence EXACTLY, so copying is the default and
    every unit of mass moved to new variants has to be earned. The composition
    heads are untouched: ranking candidates is still the birth scorer's job.
    """
    n_c = len(circ)
    logm = torch.log(mass.clamp_min(1e-9))
    z = model.variant_repr(circ, float(t))
    feat = torch.cat([z, logm.unsqueeze(-1),
                      torch.full((n_c, 1), float(dt))], dim=-1)
    fit = model.fitness(feat).squeeze(-1)
    lp_upd = torch.log_softmax(logm + dt * fit, dim=0)

    if not cands:
        return lp_upd, n_c, 0.0
    b = torch.sigmoid(model.new_bias)
    lp_new = torch.log_softmax(
        model.score_variants(cands, float(t),
                             ident=ident_fn(cands, last_seen, freq, t)), dim=0)
    return (torch.cat([lp_upd + torch.log1p(-b + 1e-9),
                       lp_new + torch.log(b + 1e-9)]),
            n_c, float(b.detach()))


def population_loss(model, circ_mass, candidates, obs_mass,
                    last_seen, freq, t, dt, ident_fn, max_cand=2000):
    """Train on the quantity that is reported.

    The population cross-entropy was previously evaluation-only: no parameter
    was ever optimised for it, so there was no reason for the model to beat
    persistence on it. Here it IS the objective. Persistence is the
    initialisation -- fitness starts at zero and new_bias low -- so any
    deviation that survives training is one that reduced the forecast loss.
    """
    circ = [s for s, _ in circ_mass]
    if not circ:
        return None
    mass = torch.tensor([m for _, m in circ_mass], dtype=torch.float32)
    cands = list(candidates)[:max_cand]
    logp, n_c, _ = _split_logp(model, circ, mass, cands, t, dt, ident_fn,
                               last_seen, freq)

    ix = {v: i for i, v in enumerate(circ + cands)}
    tot = sum(obs_mass.values()) or 1.0
    idx, wts = [], []
    for v, m in obs_mass.items():
        j = ix.get(v)
        if j is not None:
            idx.append(j); wts.append(m / tot)
    if not idx:
        return None
    # Unreachable mass is a constant w.r.t. the parameters, so it is excluded
    # from the loss. It is still charged in the reported metric -- the model is
    # trained on what it can affect and judged on everything.
    I = torch.tensor(idx, dtype=torch.long)
    W = torch.tensor(wts, dtype=torch.float32)
    return -(W * logp[I]).sum() / W.sum().clamp_min(1e-9)


def build_support(model, circ_mass, t, dt, rng, a, obs_mass=None,
                  train=True):
    """Support for the transition: circulating + generated + (training) observed.

    Candidates come only from the learned generator -- no enumeration and no
    fixed radius. Both of its paths are sampled here and scored below with the
    same parameters, so the forecasting loss trains generation directly.

    During TRAINING the observed variants are added even when the generator
    missed them: their score contains the generator's own log-probability, so a
    missed variant produces a large loss term whose gradient raises the
    probability of generating it. At EVALUATION they are excluded, so the
    reported number reflects what the model would actually have produced.
    """
    circ = [v for v, _ in circ_mass]
    mass = torch.tensor([m for _, m in circ_mass], dtype=torch.float32)
    u = model.pop_state(circ, mass, float(t))
    n_bg = min(len(circ), a.n_backgrounds)
    bg = circ[:n_bg]
    zB = model.variant_repr(bg, float(t))
    table = model.node_repr_cached(float(t))

    seen = set(circ)
    pairs = []                      # (variant, background index, added set)
    for v, bi, adds in model.gen.sample(zB, bg, u, dt, table,
                                        a.n_anchor, a.n_free, rng):
        if v and v not in seen:
            seen.add(v); pairs.append((v, bi, adds))

    n_missed = 0
    if train and obs_mass:
        for v in obs_mass:
            if v not in seen:
                seen.add(v)
                bi, adds = _attach(v, bg)
                pairs.append((v, bi, adds)); n_missed += 1
    return u, bg, zB, table, pairs, n_missed


def _attach(v, backgrounds):
    """Nearest background and the mutations that extend it, if it is a superset."""
    best, bi = None, None
    for i, b in enumerate(backgrounds):
        d = len(v ^ b)
        if best is None or d < best:
            best, bi = d, i
    if bi is None:
        return None, None
    extra = v - backgrounds[bi]
    if extra and not (backgrounds[bi] - v):
        return bi, extra
    return None, None


def transition_loss(model, circ_mass, obs_mass, t, dt, rng, a, train=True,
                    return_parts=False):
    """Cross-entropy of the observed population under the predicted one.

    Persistence is the zero-initialised default: budget -> 0 and fitness = 0
    reproduce it exactly, so training can only move away from persistence where
    that lowers the loss on observed transitions.
    """
    circ = [v for v, _ in circ_mass]
    if len(circ) < 2 or not obs_mass:
        return (None, None) if not return_parts else (None, None, None)

    # Restrict the target to the variants that carry the mass.
    # A real day has thousands of distinct variants, most seen once, so
    # scoring against all of them makes the loss a contest over singletons --
    # much of which is sequencing noise -- and no generator can propose them
    # all. The forecasting question is which variants will DOMINATE, so the
    # target is the heaviest ones covering obs_frac of the observed mass,
    # capped at obs_top. Reported metrics still use the full population.
    ranked = sorted(obs_mass.items(), key=lambda kv: -kv[1])
    tot_all = sum(v for _, v in ranked) or 1.0
    keep, run = [], 0.0
    for v, w in ranked:
        keep.append((v, w)); run += w / tot_all
        if run >= a.obs_frac or len(keep) >= a.obs_top:
            break
    obs_fit = dict(keep)

    mass = torch.tensor([m for _, m in circ_mass], dtype=torch.float32)
    u, bg, zB, table, pairs, n_missed = build_support(
        model, circ_mass, t, dt, rng, a, obs_fit, train)

    logm = torch.log(mass.clamp_min(1e-9))
    z = model.variant_repr(circ, float(t))
    feat = torch.cat([z, logm.unsqueeze(-1),
                      torch.full((len(circ), 1), float(dt))], dim=-1)
    fit = model.fitness(feat).squeeze(-1)
    lp_upd = torch.log_softmax(logm + dt * fit, dim=0)

    if pairs:
        cand = [p[0] for p in pairs]
        sc = model.score_variants(cand, float(t))

        # generation log-probability enters the score, so the forecasting loss
        # backpropagates into the generator itself
        anch = [k for k, p in enumerate(pairs) if p[1] is not None]
        gl = torch.zeros(len(pairs))
        if anch:
            sel = torch.tensor([pairs[k][1] for k in anch], dtype=torch.long)
            lp_a = model.gen.anchored_logp(
                zB[sel], u, dt, table, [pairs[k][2] for k in anch])
            gl = gl.index_copy(0, torch.tensor(anch), lp_a)
        free = [k for k, p in enumerate(pairs) if p[1] is None]
        if free:
            flp, _ = model.gen.free_logp(u, dt, table)
            fv = torch.stack([flp[list(pairs[k][0])].sum() for k in free])
            gl = gl.index_copy(0, torch.tensor(free), fv)
        mix = torch.sigmoid(model.gen.mix(
            torch.tensor([[float(dt)]], dtype=torch.float32)).squeeze())
        w_path = torch.where(
            torch.tensor([p[1] is not None for p in pairs]),
            torch.log(mix + 1e-9), torch.log1p(-mix + 1e-9))

        sc = sc + a.gen_weight * (gl + w_path)
        b = torch.sigmoid(model.budget_logit(dt))
        logp = torch.cat([lp_upd + torch.log1p(-b + 1e-9),
                          torch.log_softmax(sc, dim=0) + torch.log(b + 1e-9)])
        allv = circ + cand
    else:
        logp, allv, mix = lp_upd, circ, torch.tensor(0.5)

    ix = {v: i for i, v in enumerate(allv)}
    tgt = obs_fit if train else obs_mass
    tot = sum(tgt.values()) or 1.0
    I, W = [], []
    for v, m in tgt.items():
        j = ix.get(v)
        if j is not None:
            I.append(j); W.append(m / tot)
    if not I:
        return (None, None) if not return_parts else (None, None, None)
    Iw = torch.tensor(I, dtype=torch.long)
    Ww = torch.tensor(W, dtype=torch.float32)
    loss = -(Ww * logp[Iw]).sum() / Ww.sum().clamp_min(1e-9)
    meta = {"n_support": len(allv), "n_missed": n_missed,
            "n_target": len(obs_fit), "target_mass": run,
            "budget": float(torch.sigmoid(model.budget_logit(dt)).detach()),
            "mix_anchored": float(mix.detach())}
    if return_parts:
        return loss, meta, (logp, allv, len(circ))
    return loss, meta


def population_baselines(circ_mass, candidates, dt):
    """Full-population baselines. Each returns log-probabilities over the
    same support as the model, so the numbers are directly comparable.

    persistence : today's population, unchanged
    drift       : today's population scaled by its recent growth rate
    proximity   : persistence, with a small share spread over candidates
                  in proportion to closeness to the current population
    """
    circ = [s for s, _ in circ_mass]
    mass = np.array([m for _, m in circ_mass], dtype=float)
    n_c, n_n = len(circ), len(candidates)
    out = {}

    p = np.concatenate([mass, np.zeros(n_n)])
    out["persistence"] = np.log(np.clip(p / max(p.sum(), 1e-12), 1e-12, None))

    # proximity: 10% of mass onto candidates, weighted by 1/(1+distance)
    if n_n:
        idx = CirculatingIndex(circ, max(
            (max(s) for s in circ + list(candidates) if s), default=1) + 1)
        d = -idx.nearest(list(candidates))
        w = 1.0 / (1.0 + np.maximum(d, 0.0))
        w = w / max(w.sum(), 1e-12)
        q = np.concatenate([0.9 * mass / max(mass.sum(), 1e-12), 0.1 * w])
    else:
        q = p / max(p.sum(), 1e-12)
    out["prox"] = np.log(np.clip(q, 1e-12, None))
    return out


def score_population(logp, allv, n_circ, obs_mass):
    """Cross-entropy of the observed population under a predicted one.

    Reported in nats per sequence -- lower is better. Also split by whether
    the observed mass sits on a variant that was already present (UPDATE) or
    one that was not (NEW), which is the decomposition that says what the
    model is actually getting right.
    """
    ix = {s: i for i, s in enumerate(allv)}
    tot = sum(obs_mass.values()) or 1.0
    ce = ce_u = ce_n = 0.0
    m_u = m_n = m_cov = 0.0
    for s, m in obs_mass.items():
        w = m / tot
        j = ix.get(s)
        lp = logp[j] if j is not None else math.log(1e-12)
        ce += -w * lp
        if j is not None:
            m_cov += w
        if j is not None and j < n_circ:
            ce_u += -w * lp; m_u += w
        else:
            ce_n += -w * lp; m_n += w
    # Direct comparison of the two populations as weighted sets. Variants are
    # exact discrete sets, so predicted and observed match exactly -- there is
    # no assignment problem and no need for a ranking metric.
    #   overlap = sum_i min(p_i, q_i)   the share of the population predicted
    #                                   correctly; 1.0 is perfect
    #   jaccard = sum min / sum max     the same, penalising over-prediction
    # Both are bounded in [0, 1], so unreachable mass costs at most its own
    # weight instead of the log(1e-12) floor that made the cross-entropy
    # columns indistinguishable.
    p = np.exp(np.asarray(logp, dtype=float))
    q = np.zeros_like(p)
    for s, m in obs_mass.items():
        j = ix.get(s)
        if j is not None:
            q[j] += m / tot
    overlap = float(np.minimum(p, q).sum())
    denom = float(np.maximum(p, q).sum())
    jac = overlap / denom if denom > 0 else float("nan")
    return {"ce": ce, "ce_update": ce_u / m_u if m_u > 0 else float("nan"),
            "ce_new": ce_n / m_n if m_n > 0 else float("nan"),
            "mass_update": m_u, "mass_new": m_n, "coverage": m_cov,
            "overlap": overlap, "jaccard": jac}


def report_forecast(rows):
    if not rows:
        print("\nno forecast rows"); return
    print("\n" + "=" * 74)
    print("POPULATION FORECAST -- cross-entropy of the observed population")
    print("under the predicted one, nats per sequence. LOWER IS BETTER.")
    methods = ["model", "persistence", "prox"]
    lbl = {"model": "MODEL", "persistence": "persistence", "prox": "persist+prox"}
    print("\nOVERLAP = share of the future population predicted correctly "
          "(sum of min(pred, obs)).\nHIGHER IS BETTER. 1.0 is perfect, and "
          "it is bounded, so unreachable mass\ncosts only its own weight "
          "rather than a log-floor penalty.")
    print(f"\n{'horizon':<10}{'n':>4}" +
          "".join(f"{lbl[m]:>15}" for m in methods))
    for h in sorted({r["h"] for r in rows}):
        sub = [r for r in rows if r["h"] == h]
        line = f"{str(h) + ' months':<10}{len(sub):>4}"
        for mth in methods:
            line += f"{np.mean([r[mth]['overlap'] for r in sub]):>15.4f}"
        print(line)

    print("\nWeighted Jaccard (overlap normalised by the union)")
    print(f"{'horizon':<10}{'n':>4}" +
          "".join(f"{lbl[m]:>15}" for m in methods))
    for h in sorted({r["h"] for r in rows}):
        sub = [r for r in rows if r["h"] == h]
        line = f"{str(h) + ' months':<10}{len(sub):>4}"
        for mth in methods:
            line += f"{np.mean([r[mth]['jaccard'] for r in sub]):>15.4f}"
        print(line)

    print("\nCross-entropy, nats per sequence (lower better). Dominated by "
          "unreachable\nmass, so read the overlap table first.")
    print(f"{'horizon':<10}{'n':>4}" +
          "".join(f"{lbl[m]:>15}" for m in methods))
    for h in sorted({r["h"] for r in rows}):
        sub = [r for r in rows if r["h"] == h]
        line = f"{str(h) + ' months':<10}{len(sub):>4}"
        for mth in methods:
            line += f"{np.mean([r[mth]['ce'] for r in sub]):>15.4f}"
        print(line)

    print("\nsplit by where the observed mass sits")
    print(f"{'horizon':<10}{'mass on':>10}{'mass on':>10}"
          f"{'MODEL ce':>11}{'MODEL ce':>11}{'covered':>9}")
    print(f"{'':<10}{'existing':>10}{'new':>10}{'existing':>11}{'new':>11}{'':>9}")
    for h in sorted({r["h"] for r in rows}):
        sub = [r for r in rows if r["h"] == h]
        f = lambda k: np.nanmean([r["model"][k] for r in sub])
        print(f"{str(h) + ' months':<10}{f('mass_update'):>10.3f}"
              f"{f('mass_new'):>10.3f}{f('ce_update'):>11.3f}"
              f"{f('ce_new'):>11.3f}{f('coverage'):>9.3f}")
    print("\n'covered' is the share of observed mass that appears anywhere in "
          "the\npredicted support -- an upper bound on what any scoring "
          "method could get.")


def report_selection(recs, a):
    """Growth and death, against the baselines that could beat them.

    Growth is computed on SURVIVORS ONLY. When most variants go to zero, a
    log-growth MAE over everything is dominated by log(0) floor terms and
    measures extinction, not growth -- which is what the death AUC is for.
    Spearman is the honest growth number: it asks whether the model orders
    variants by how fast they actually grew.
    """
    if not recs:
        print("\nno selection records"); return
    print("\n" + "=" * 70)
    print("SELECTION: growth and death of already-circulating variants")

    for dt in sorted({r["dt"] for r in recs}):
        sub = [r for r in recs if r["dt"] == dt]
        d = np.array([r["dead"] for r in sub])
        sc = np.array([r["death_score"] for r in sub])
        ms = np.array([r["mass"] for r in sub])
        auc = _auc(d, sc)
        auc_mass = _auc(d, -ms)          # baseline: rare variants die
        print(f"\n--- horizon {int(dt)} days   (n={len(sub)}, "
              f"{d.mean():.1%} below threshold at t+dt) ---")

        surv = [r for r in sub if r.get("survived", 0.0) > 0]
        if len(surv) >= 5:
            pg = np.array([r["pred_logg"] for r in surv])
            og = np.array([r["obs_logg"] for r in surv])
            ok = np.isfinite(pg) & np.isfinite(og)
            pg, og = pg[ok], og[ok]
            mae_m = float(np.abs(pg - og).mean())
            mae_p = float(np.abs(og).mean())     # persistence: no change
            r_p = (float(np.corrcoef(pg, og)[0, 1])
                   if pg.std() > 0 and og.std() > 0 else float("nan"))
            try:
                from scipy.stats import spearmanr
                r_s = float(spearmanr(pg, og).statistic)
            except Exception:
                r_s = float("nan")
            print(f"  growth (survivors only, n={len(pg)})")
            print(f"    MAE log-growth   model {mae_m:8.4f}   "
                  f"persistence {mae_p:8.4f}")
            print(f"    corr(pred, obs)  pearson {r_p:7.3f}   "
                  f"spearman {r_s:7.3f}   <- the growth number")
        else:
            print("  growth: too few survivors to score")

        print(f"  death   AUC   model {auc:7.3f}   "
              f"rarity baseline {auc_mass:7.3f}")

        # by mass decile: does the model track dominant variants, the tail,
        # or neither? a single pooled number cannot say.
        if any("mass_decile" in r for r in sub):
            print(f"    {'decile':<8}{'n':>7}{'died':>8}"
                  f"{'deathAUC':>10}{'spearman':>10}")
            for dec in range(10):
                g = [r for r in sub if r.get("mass_decile") == dec]
                if len(g) < 20:
                    continue
                gd = np.array([r["dead"] for r in g])
                gs = np.array([r["death_score"] for r in g])
                sv = [r for r in g if r.get("survived", 0.0) > 0]
                rs = float("nan")
                if len(sv) >= 5:
                    try:
                        from scipy.stats import spearmanr
                        rs = float(spearmanr(
                            [r["pred_logg"] for r in sv],
                            [r["obs_logg"] for r in sv]).statistic)
                    except Exception:
                        pass
                print(f"    {dec:<8}{len(g):>7}{gd.mean():>8.1%}"
                      f"{_auc(gd, gs):>10.3f}{rs:>10.3f}")


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
    p.add_argument("--events", required=True)
    p.add_argument("--vocab", default=None)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--n-recent", type=int, default=10, dest="n_recent")
    p.add_argument("--posres", action="store_true")
    p.add_argument("--no-decay", action="store_true", dest="no_decay")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 6],
                   help="forecast horizons in months")
    p.add_argument("--pop-support", type=int, default=1000, dest="pop_support",
                   help="circulating variants kept, heaviest first")
    p.add_argument("--n-backgrounds", type=int, default=200,
                   dest="n_backgrounds",
                   help="backgrounds the anchored path may extend")
    p.add_argument("--n-anchor", type=int, default=400, dest="n_anchor",
                   help="candidates sampled from the anchored path")
    p.add_argument("--n-free", type=int, default=100, dest="n_free",
                   help="candidates sampled from the unanchored path")
    p.add_argument("--obs-frac", type=float, default=0.8, dest="obs_frac",
                   help="fit against the heaviest observed variants covering "
                        "this share of mass; the rest is a singleton tail no "
                        "generator can enumerate")
    p.add_argument("--obs-top", type=int, default=200, dest="obs_top",
                   help="hard cap on the number of target variants")
    p.add_argument("--gen-weight", type=float, default=1.0, dest="gen_weight",
                   help="weight of the generation log-probability in a "
                        "candidate's score")
    p.add_argument("--stride", type=int, default=7,
                   help="train on every Nth day")
    p.add_argument("--train-frac", type=float, default=0.7, dest="train_frac")
    p.add_argument("--n-origins", type=int, default=6, dest="n_origins")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
