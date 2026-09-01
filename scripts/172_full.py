#!/usr/bin/env python3
"""
172_full.py -- HGNN + TPP loss (faithful to HGDHE 2023) + MADE generation.

Three components:
1. Node representations via HGNN + exponential decay (eq 5 from HGDHE)
2. TPP loss: f(v_members) for observed variants + survival via negative sampling
   f is applied to SPECIFIC member node representations (not pooled)
   Gradient flows specifically into member mutations' representations
3. MADE for generation: per-position v_k conditioning + autoregressive joint
   Skip connections fix vanishing gradient through K sequential steps

Training: weekly batches, per-variant gradient updates
Evaluation: project v_i to T+h via decay, sample from MADE

Usage:
  python scripts/172_full.py \
    --events data/processed/events_v3.tsv \
    --train-end 2022-06 --test-month 2022-07 \
    --d 64 --d-hidden 128 --K-made 50 \
    --epochs 5 --n-neg 20 --n-samples 500 --seed 0
"""
import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from datetime import date as dobj

# ── data ──────────────────────────────────────────────────────────────

def load_weekly(path, end_ym, test_ym):
    by_week = defaultdict(lambda: defaultdict(float))
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if ln == 0 and not parts[0][:4].isdigit(): continue
            date_s, muts = parts[0].strip(), parts[1].strip()
            cnt = float(parts[2]) if len(parts) > 2 else 1.0
            ym = date_s[:7]
            if ym > test_ym: continue
            s = frozenset(int(x) for x in muts.split(',') if x)
            if not s: continue
            try:
                d = dobj.fromisoformat(date_s)
                wk = f"{d.isocalendar()[0]:04d}-{d.isocalendar()[1]:02d}"
                by_week[wk][s] += cnt
            except ValueError:
                continue

    weeks = sorted(by_week.keys())
    week2ym = {}
    for wk in weeks:
        y, w = int(wk[:4]), int(wk[5:])
        monday = dobj.fromisocalendar(y, w, 1)
        week2ym[wk] = monday.strftime('%Y-%m')

    var_mass = {}
    for wk in weeks:
        tot = sum(by_week[wk].values()) or 1.0
        var_mass[wk] = {s: v/tot for s, v in by_week[wk].items()}

    all_muts = sorted({m for wk in weeks for v in var_mass[wk] for m in v})
    mut2idx = {m: i for i, m in enumerate(all_muts)}
    V = len(all_muts)
    train_wks = [w for w in weeks if week2ym[w] <= end_ym]
    print(f"loaded {len(weeks)} weeks  V={V}  "
          f"train: {week2ym[train_wks[0]]} to {week2ym[train_wks[-1]]}")
    return var_mass, weeks, week2ym, mut2idx, V

def month_population(path, test_ym):
    agg = defaultdict(float)
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if ln == 0 and not parts[0][:4].isdigit(): continue
            date_s, muts = parts[0].strip(), parts[1].strip()
            cnt = float(parts[2]) if len(parts) > 2 else 1.0
            if date_s[:7] == test_ym:
                s = frozenset(int(x) for x in muts.split(',') if x)
                if s: agg[s] += cnt
    tot = sum(agg.values()) or 1.0
    return {s: v/tot for s, v in agg.items()}

# ── modules ───────────────────────────────────────────────────────────

class FourierTime(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(
            1.0/10**np.linspace(0,3,d)).float())
        self.b = nn.Parameter(torch.zeros(d))
    def forward(self, dt):
        if not torch.is_tensor(dt): dt = torch.tensor(float(dt))
        if dt.dim() == 0: dt = dt.unsqueeze(0)
        out = torch.cos(dt.unsqueeze(-1)*self.w + self.b)
        return out.squeeze(0) if out.shape[0]==1 else out

class HGNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
    def forward(self, H, X):
        deg_v = H.sum(1).clamp_min(1)
        deg_e = H.sum(0).clamp_min(1)
        E = (H.T @ X) / deg_e.unsqueeze(1)
        X_new = (H @ E) / deg_v.unsqueeze(1)
        return self.norm(F.relu(self.W(X_new)))

class IntensityMLP(nn.Module):
    """Their f: scores a hyperedge from member node representations.
    
    Faithfully implements eq 3 from HGDHE 2023:
    lambda_h(t) = f(v_1(t), ..., v_k(t))
    
    Takes mean of member representations then MLP + softplus.
    Gradient flows ONLY into member mutations' representations.
    """
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.Tanh(),
            nn.Linear(d, 1), nn.Softplus())

    def forward(self, v_members):
        """v_members: (k, d) representations of variant members.
        Returns scalar lambda_h.
        Gradient flows into v_members -- their specific node reprs.
        """
        h = v_members.mean(0)  # (d,) -- mean of members
        return self.net(h).squeeze(-1)  # scalar

class MADE(nn.Module):
    """Autoregressive joint over K core positions.
    
    Each position k conditioned on:
    - v_k: that mutation's own node representation (per-node signal)
    - x_{<k}: previously sampled positions (autoregressive joint)
    - h_gru: accumulated population context
    
    Skip connections fix vanishing gradient through K steps.
    Gradient flows: loss -> made -> v_k -> HGNN -> node_emb
    """
    def __init__(self, K, d, d_hidden=128):
        super().__init__()
        self.K = K
        self.d = d
        self.d_hidden = d_hidden
        self.gru = nn.GRUCell(d, d_hidden)
        nn.init.ones_(self.gru.bias_ih[d_hidden:2*d_hidden])
        nn.init.ones_(self.gru.bias_hh[d_hidden:2*d_hidden])

        # deep path: h_gru + v_k + prev_k
        self.deep = nn.Sequential(
            nn.Linear(d_hidden + d + K, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, 1))
        # skip connections -- direct gradient paths
        self.skip_h = nn.Linear(d_hidden, 1, bias=False)
        self.skip_v = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.deep[-1].weight)
        nn.init.zeros_(self.deep[-1].bias)
        nn.init.zeros_(self.skip_h.weight)
        nn.init.zeros_(self.skip_v.weight)

        self.register_buffer('h', torch.zeros(d_hidden))

    def reset(self): self.h.zero_()

    def update(self, ctx):
        """Update GRU state with population context. Detached."""
        self.h = self.gru(ctx.unsqueeze(0),
                         self.h.detach().unsqueeze(0)).squeeze(0).detach()

    def log_prob(self, x, v_core, h_diff=None):
        """Log joint probability of binary vector x.

        x:      (B, K) binary
        v_core: (K, d) per-position node representations -- NOT pooled
        h_diff: (d_hidden,) differentiable GRU state for gradient flow

        Each position k sees its OWN v_core[k] -- per-node conditioning.
        Returns (B,) log probabilities.
        """
        B = x.shape[0]
        h_use = h_diff if h_diff is not None else self.h
        h_exp = h_use.unsqueeze(0).expand(B, -1)

        lp = torch.zeros(B, device=x.device)
        prev = torch.zeros(B, self.K, device=x.device)

        for k in range(self.K):
            # per-position: each step sees mutation k's OWN representation
            v_k = v_core[k].unsqueeze(0).expand(B, -1)  # (B, d)
            inp = torch.cat([h_exp, v_k, prev], dim=-1)
            # deep + skip -- gradient flows through both paths
            logit = (self.deep(inp).squeeze(-1) +
                    self.skip_h(h_exp).squeeze(-1) +
                    self.skip_v(v_k).squeeze(-1))
            lp = lp - F.binary_cross_entropy_with_logits(
                logit, x[:, k], reduction='none')
            prev = prev.clone()
            prev[:, k] = x[:, k]

        return lp

    def sample(self, v_core, n=500):
        """Sample n binary vectors using per-position v_core."""
        dev = self.h.device
        h_exp = self.h.unsqueeze(0).expand(n, -1)
        prev = torch.zeros(n, self.K, device=dev)
        with torch.no_grad():
            for k in range(self.K):
                v_k = v_core[k].unsqueeze(0).expand(n, -1)
                inp = torch.cat([h_exp, v_k, prev], dim=-1)
                logit = (self.deep(inp).squeeze(-1) +
                        self.skip_h(h_exp).squeeze(-1) +
                        self.skip_v(v_k).squeeze(-1))
                prev[:, k] = torch.bernoulli(torch.sigmoid(logit))
        return prev

# ── full model ────────────────────────────────────────────────────────

class FullModel(nn.Module):
    def __init__(self, V, K_made, d=64, d_hidden=128, mut2idx=None):
        super().__init__()
        self.V = V
        self.K_made = K_made
        self.d = d
        self._mut2idx = mut2idx or {}

        # node embeddings
        self.node_emb = nn.Embedding(V, d)
        nn.init.normal_(self.node_emb.weight, 0, 0.1)

        # exponential decay -- learned
        self.log_gamma = nn.Parameter(torch.tensor(-2.0))

        # Fourier time
        self.psi = FourierTime(d)

        # HGNN
        self.hgnn = HGNN(d)

        # their f -- intensity MLP for hyperedge scoring
        self.f_mlp = IntensityMLP(d)

        # context projection for GRU
        self.ctx_proj = nn.Linear(d, d)

        # MADE for generation with per-position conditioning
        self.made = MADE(K_made, d, d_hidden)

        # state
        self.register_buffer('last_seen_week', torch.zeros(V))
        self._current_week = 0

    def reset(self):
        self.last_seen_week.zero_()
        self.made.reset()
        self._current_week = 0

    def get_node_reprs(self, H, week_idx=None):
        """Node representations with exponential decay.
        v_i(t) = tanh(HGNN(emb_i * gamma^dt, H) + Phi(dt*7))
        """
        dev = self.node_emb.weight.device
        gamma = torch.sigmoid(self.log_gamma)*0.5 + 0.5
        t = float(week_idx if week_idx is not None else self._current_week)
        dt = (t - self.last_seen_week).clamp_min(0)
        decay = gamma.pow(dt)
        X = self.node_emb.weight * decay.unsqueeze(1)
        X_hgnn = self.hgnn(H, X)
        phi = self.psi(dt * 7.0)
        return torch.tanh(X_hgnn + phi)

    def get_context(self, node_reprs, active_idxs):
        if not active_idxs:
            return torch.zeros(self.d, device=node_reprs.device)
        idx = torch.tensor(active_idxs, dtype=torch.long,
                          device=node_reprs.device)
        return torch.tanh(self.ctx_proj(node_reprs[idx].mean(0)))

    def forward(self, H, var_mass_week, core_muts, neg_variants, week_idx):
        """TPP loss for one week.

        Positive: f(v_members) for each observed variant -- their exact eq 3
        Survival: f(v_members) for negative variants -- approximates integral
        MADE: log_prob(variant | v_core per position) -- joint generation

        Gradient:
        - f_mlp: flows into member v_i specifically (not all mutations)
        - made: flows into v_core[k] per position k
        - Both flow into HGNN and node_emb
        """
        dev = self.node_emb.weight.device
        core_set = set(core_muts)
        c2k = {m: k for k, m in enumerate(core_muts)}
        core_t = torch.tensor(core_muts, dtype=torch.long, device=dev)

        # node representations WITH gradient
        node_reprs = self.get_node_reprs(H, week_idx)
        v_core = node_reprs[core_t]  # (K, d) -- per-position, NOT pooled

        # compute h WITH gradient for MADE
        active = list({self._mut2idx[m]
                      for v in var_mass_week for m in v
                      if m in self._mut2idx and self._mut2idx[m] < self.V})
        ctx = self.get_context(node_reprs, active)
        h_diff = self.made.gru(ctx.unsqueeze(0),
                              self.made.h.detach().unsqueeze(0)).squeeze(0)

        # ── positive: f on observed variants (their eq 3) ─────────────
        loss_pos = torch.tensor(0.0, device=dev)
        loss_made = torch.tensor(0.0, device=dev)
        total_w = 0.0

        variants = list(var_mass_week.items())
        X_pos = torch.zeros(len(variants), self.K_made, device=dev)
        W_pos = torch.zeros(len(variants), device=dev)

        for bi, (v, w) in enumerate(variants):
            # f: applied to SPECIFIC members (their exact formulation)
            member_idxs = [self._mut2idx[m] for m in v
                          if m in self._mut2idx and self._mut2idx[m] < self.V]
            if member_idxs:
                m_t = torch.tensor(member_idxs, dtype=torch.long, device=dev)
                v_members = node_reprs[m_t]  # only member representations
                lam = self.f_mlp(v_members)
                loss_pos = loss_pos - w * torch.log(lam + 1e-9)

            # MADE binary vector
            for m in v:
                mi = self._mut2idx.get(m, -1)
                if mi in core_set: X_pos[bi, c2k[mi]] = 1.0
            W_pos[bi] = w
            total_w += w

        if total_w > 0:
            W_pos = W_pos / total_w
            # MADE: per-position v_core -- NOT pooled
            lp_made = self.made.log_prob(X_pos, v_core, h_diff=h_diff)
            loss_made = -(W_pos * lp_made).sum()

        # ── survival: f on negative variants ──────────────────────────
        loss_surv = torch.tensor(0.0, device=dev)
        for neg in neg_variants:
            member_idxs = [self._mut2idx[m] for m in neg
                          if m in self._mut2idx and self._mut2idx[m] < self.V]
            if member_idxs:
                m_t = torch.tensor(member_idxs, dtype=torch.long, device=dev)
                v_neg = node_reprs[m_t]
                lam_neg = self.f_mlp(v_neg)
                loss_surv = loss_surv + lam_neg  # minimize intensity

        loss_surv = loss_surv / max(len(neg_variants), 1)

        total = loss_pos + loss_made + 0.01 * loss_surv
        return total, float(loss_pos.detach()), \
               float(loss_made.detach()), float(loss_surv.detach())

    def update_state(self, H, var_mass_week, week_idx):
        """Update GRU and last_seen after gradient step."""
        with torch.no_grad():
            node_reprs = self.get_node_reprs(H, week_idx)
            active = list({self._mut2idx[m]
                          for v in var_mass_week for m in v
                          if m in self._mut2idx and self._mut2idx[m] < self.V})
            if active:
                ctx = self.get_context(node_reprs, active)
                self.made.update(ctx)
            for v in var_mass_week:
                for m in v:
                    mi = self._mut2idx.get(m, -1)
                    if 0 <= mi < self.V:
                        self.last_seen_week[mi] = float(week_idx)
            self._current_week = week_idx

    def generate(self, H, core_muts, week_idx, h_weeks, n_samples):
        """Project to T+h, sample from MADE with per-position v_core."""
        dev = self.node_emb.weight.device
        core_t = torch.tensor(core_muts, dtype=torch.long, device=dev)
        idx2mut = {v: k for k, v in self._mut2idx.items()}

        with torch.no_grad():
            gamma = torch.sigmoid(self.log_gamma)*0.5 + 0.5
            future = week_idx + h_weeks
            dt = (future - self.last_seen_week[core_t]).clamp_min(0)
            decay = gamma.pow(dt)
            X = self.node_emb.weight[core_t] * decay.unsqueeze(1)
            H_core = H[core_t]
            # project: use core-only HGNN approximation
            phi = self.psi(dt * 7.0)
            # full HGNN needs full V -- use projected embeddings directly
            v_core_proj = torch.tanh(X + phi)  # (K, d) projected

            samples = self.made.sample(v_core_proj, n=n_samples)

        candidates = []
        for i in range(n_samples):
            muts = frozenset(
                idx2mut[core_muts[k]]
                for k in range(self.K_made)
                if samples[i,k].item() > 0.5
                and idx2mut.get(core_muts[k]) is not None)
            if muts: candidates.append(muts)
        return candidates

    def score_variant(self, v, node_reprs):
        """Score a variant using f_mlp on its member representations."""
        dev = node_reprs.device
        member_idxs = [self._mut2idx[m] for m in v
                      if m in self._mut2idx and self._mut2idx[m] < self.V]
        if not member_idxs: return 0.0
        m_t = torch.tensor(member_idxs, dtype=torch.long, device=dev)
        with torch.no_grad():
            lam = self.f_mlp(node_reprs[m_t])
        return float(lam.item())

# ── helpers ───────────────────────────────────────────────────────────

def sample_negatives(var_mass_week, all_muts, n_neg):
    observed = list(var_mass_week.keys())
    negs = []
    for _ in range(n_neg):
        base = list(random.choice(observed))
        n_flip = random.randint(2, 4)
        for _ in range(n_flip):
            if random.random() < 0.5 and base:
                base.pop(random.randint(0, len(base)-1))
            else:
                base.append(random.choice(all_muts))
        negs.append(frozenset(base))
    return negs

def build_H(var_mass_week, mut2idx, V, device):
    variants = list(var_mass_week.keys())
    K = max(len(variants), 1)
    H = torch.zeros(V, K, device=device)
    for ki, v in enumerate(variants):
        w = var_mass_week[v]
        for m in v:
            if m in mut2idx: H[mut2idx[m], ki] = w
    return H

# ── run ───────────────────────────────────────────────────────────────

def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    var_mass, weeks, week2ym, mut2idx, V = load_weekly(
        a.events, a.train_end, a.test_month)
    all_muts_list = list(mut2idx.keys())

    train_weeks = [w for w in weeks if week2ym[w] <= a.train_end]
    print(f"train weeks: {len(train_weeks)}")

    # core mutations
    freq = defaultdict(float)
    for wk in train_weeks:
        for v, w in var_mass[wk].items():
            for m in v:
                if m in mut2idx: freq[m] += w
    core_muts_raw = sorted(freq, key=lambda m: -freq[m])[:a.K_made]
    core_muts = [mut2idx[m] for m in core_muts_raw]
    core_set = set(core_muts)
    c2k = {m: k for k, m in enumerate(core_muts)}
    print(f"core mutations: {len(core_muts)}")

    model = FullModel(V, len(core_muts), d=a.d,
                     d_hidden=a.d_hidden, mut2idx=mut2idx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # training
    model.reset()
    for ep in range(a.epochs):
        tot = pos = mad = sur = 0.0
        for wi, wk in enumerate(train_weeks):
            H = build_H(var_mass[wk], mut2idx, V, device)
            vm = var_mass[wk]
            if not vm: continue
            negs = sample_negatives(vm, all_muts_list, a.n_neg)
            loss, lp, lm, ls = model(H, vm, core_muts, negs, wi)
            if torch.isfinite(loss):
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += loss.item(); pos += lp
                mad += lm; sur += ls
            model.update_state(H, vm, wi)

        n = len(train_weeks)
        print(f"ep {ep+1}/{a.epochs}  loss {tot/n:.3f}  "
              f"pos {pos/n:.3f}  made {mad/n:.3f}  "
              f"surv {sur/n:.3f}  "
              f"gamma {torch.sigmoid(model.log_gamma).item()*0.5+0.5:.3f}",
              flush=True)

    # evaluation
    print("\nevaluating...")
    last_wk = train_weeks[-1]
    last_wi = len(train_weeks) - 1
    H_last = build_H(var_mass[last_wk], mut2idx, V, device)

    # circulating at train_end
    circ_raw = defaultdict(float)
    for wk in train_weeks:
        if week2ym[wk] == a.train_end:
            for v, w in var_mass[wk].items(): circ_raw[v] += w
    circ_tot = sum(circ_raw.values()) or 1.0
    circ = {v: w/circ_tot for v, w in circ_raw.items()}

    obs = month_population(a.events, a.test_month)
    obs_tot = sum(obs.values()) or 1.0
    obs_norm = {v: w/obs_tot for v, w in obs.items()}

    # generate candidates
    h_weeks = a.horizon * 4
    candidates = model.generate(H_last, core_muts, last_wi,
                                h_weeks, a.n_samples)
    print(f"generated {len(candidates)}, unique {len(set(candidates))}")

    # get projected node reprs for scoring
    with torch.no_grad():
        node_reprs_eval = model.get_node_reprs(H_last, last_wi)

    # score and build predicted distribution
    new_cands = {}
    for v in set(candidates):
        if v not in circ:
            new_cands[v] = model.score_variant(v, node_reprs_eval)
    nc_tot = sum(new_cands.values()) or 1.0

    pred = {v: (1-a.budget)*w for v, w in circ.items()}
    for v, s in new_cands.items():
        pred[v] = pred.get(v,0) + a.budget * s/nc_tot

    def overlap(p, q):
        return sum(min(p.get(v,0), q.get(v,0)) for v in set(p)|set(q))

    ov_m = overlap(pred, obs_norm)
    ov_p = overlap(circ, obs_norm)
    new_vars = set(obs_norm) - set(circ)
    mass_new = sum(pred.get(v,0) for v in new_vars)

    print(f"\n{'='*50}")
    print(f"train ..{a.train_end}  predict {a.test_month}  h={a.horizon}m")
    print(f"model       overlap {ov_m:.4f}  mass_new {mass_new:.4f}")
    print(f"persistence overlap {ov_p:.4f}  mass_new 0.0000")
    print(f"gain        {ov_m-ov_p:+.4f}")
    print(f"gamma: {torch.sigmoid(model.log_gamma).item()*0.5+0.5:.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events',    required=True)
    p.add_argument('--train-end', default='2022-06', dest='train_end')
    p.add_argument('--test-month',default='2022-07', dest='test_month')
    p.add_argument('--d',         type=int, default=64)
    p.add_argument('--d-hidden',  type=int, default=128, dest='d_hidden')
    p.add_argument('--K-made',    type=int, default=50, dest='K_made')
    p.add_argument('--epochs',    type=int, default=5)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--n-neg',     type=int, default=20, dest='n_neg')
    p.add_argument('--n-samples', type=int, default=500, dest='n_samples')
    p.add_argument('--budget',    type=float, default=0.1)
    p.add_argument('--horizon',   type=int, default=1)
    p.add_argument('--seed',      type=int, default=0)
    run(p.parse_args())

if __name__ == '__main__':
    main()
