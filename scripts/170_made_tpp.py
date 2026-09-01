#!/usr/bin/env python3
"""
170_made_tpp.py -- HGNN + MADE for population forecasting.

Architecture:
  1. Build incidence matrix H(t) from month t variants
  2. HGNN convolution over H(t) -> node representations v_i(t)
  3. Population context = mean(v_i(t) for active mutations)
  4. MADE models p(B | context) for each observed variant B
  5. Loss = -sum_B w_B * log p(B | context) [weighted by variant mass]
  6. Gradient flows through MADE -> context -> v_i -> HGNN weights

At test time: sample from MADE conditioned on context(T) to generate
candidate new variants. Score + evaluate population overlap.

Train: months 1-6, one gradient update per month (5 total)
Test:  predict month 7

Usage:
  python scripts/170_made_tpp.py \
    --events data/processed/events_v3.tsv \
    --train-end 2022-06 --test-month 2022-07 \
    --M 6 --d 64 --epochs 20 --seed 0
"""
import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# ── data ──────────────────────────────────────────────────────────────

def load_monthly(path, start_ym, end_ym, test_ym):
    by_month = defaultdict(lambda: defaultdict(float))
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if ln == 0 and not parts[0][:4].isdigit(): continue
            date, muts = parts[0].strip(), parts[1].strip()
            cnt = float(parts[2]) if len(parts) > 2 else 1.0
            s = frozenset(int(x) for x in muts.split(',') if x)
            ym = date[:7]
            if start_ym <= ym <= test_ym and s:
                by_month[ym][s] += cnt
    months = sorted(m for m in by_month if start_ym <= m <= test_ym)
    var_mass = {}
    for ym in months:
        tot = sum(by_month[ym].values()) or 1.0
        var_mass[ym] = {s: v/tot for s, v in by_month[ym].items()}
    # build global mutation vocabulary
    all_muts = sorted({m for ym in months for v in var_mass[ym] for m in v})
    mut2idx = {m: i for i, m in enumerate(all_muts)}
    V = len(all_muts)
    print(f"months {len(months)}, V={V} active mutations")
    return var_mass, months, mut2idx, V

# ── HGNN ──────────────────────────────────────────────────────────────

class HGNN(nn.Module):
    """Hypergraph neural network convolution.
    
    Given incidence matrix H (V x K) and node features X (V x d):
    1. Hyperedge aggregation: E = H^T X  (K x d) -- mean of member features
    2. Node update: X' = H E / degree     (V x d) -- weighted sum back to nodes
    
    This is the standard HGNN from Bai et al. 2021.
    Gradient flows through both steps into X and W.
    """
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, H, X):
        """H: (V, K) incidence matrix, X: (V, d) node features."""
        # degree matrices
        deg_v = H.sum(1).clamp_min(1)   # (V,) node degree
        deg_e = H.sum(0).clamp_min(1)   # (K,) edge degree
        # hyperedge features: mean of member node features
        E = (H.T @ X) / deg_e.unsqueeze(1)   # (K, d)
        # node update: weighted sum of hyperedge features
        X_new = (H @ E) / deg_v.unsqueeze(1)  # (V, d)
        return self.norm(F.relu(self.W(X_new)))


# ── MADE ──────────────────────────────────────────────────────────────

class MADE(nn.Module):
    """Masked Autoregressive Density Estimator over K positions.
    
    Models joint p(x_1,...,x_K | context) autoregressively.
    Each position x_k ~ Bernoulli(sigmoid(f(context, x_1,...,x_{k-1}))).
    
    Fully differentiable. Gradient flows through log_prob into context
    and into all MADE parameters.
    """
    def __init__(self, K, d_ctx, d_hidden=256):
        super().__init__()
        self.K = K
        # input: context (d_ctx) + previous K positions
        self.net = nn.Sequential(
            nn.Linear(d_ctx + K, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def log_prob(self, x, ctx):
        """Log probability of binary vectors x given context ctx.
        x:   (B, K) binary
        ctx: (B, d_ctx) or (d_ctx,)
        Returns (B,) log probabilities.
        Gradient flows into ctx and all MADE parameters.
        """
        B = x.shape[0]
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0).expand(B, -1)
        lp = torch.zeros(B, device=x.device)
        prev = torch.zeros(B, self.K, device=x.device)
        for k in range(self.K):
            inp = torch.cat([ctx, prev], dim=-1)
            logit = self.net(inp).squeeze(-1)
            lp = lp - F.binary_cross_entropy_with_logits(
                logit, x[:, k], reduction='none')
            prev = prev.clone()
            prev[:, k] = x[:, k]
        return lp

    def sample(self, ctx, n=1):
        """Sample n binary vectors given context ctx."""
        dev = ctx.device
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0).expand(n, -1)
        prev = torch.zeros(n, self.K, device=dev)
        with torch.no_grad():
            for k in range(self.K):
                inp = torch.cat([ctx, prev], dim=-1)
                logit = self.net(inp).squeeze(-1)
                prev[:, k] = torch.bernoulli(torch.sigmoid(logit))
        return prev


# ── main model ────────────────────────────────────────────────────────

class HGNNMADEModel(nn.Module):
    def __init__(self, V, K_made, d=64, d_hidden=256, mut2idx=None):
        super().__init__()
        self.V = V
        self.K_made = K_made
        self.d = d

        # node embeddings -- learned, updated by HGNN
        self.node_emb = nn.Embedding(V, d)
        nn.init.normal_(self.node_emb.weight, 0, 0.1)

        # HGNN for hypergraph convolution
        self.hgnn = HGNN(d)

        # context projection: mean of node reprs -> context vector
        self.ctx_proj = nn.Linear(d, d)

        # MADE: joint distribution over top-K mutation positions
        self.made = MADE(K_made, d, d_hidden)
        self._mut2idx = mut2idx or {}

    def get_node_reprs(self, H):
        """H: (V, K) incidence matrix.
        Returns (V, d) node representations after HGNN convolution.
        Gradient flows through HGNN and node_emb.
        """
        X = self.node_emb.weight   # (V, d)
        return self.hgnn(H, X)     # (V, d)

    def get_context(self, node_reprs, active_muts):
        """Population context = mean of active mutation representations."""
        if not active_muts:
            return torch.zeros(self.d, device=node_reprs.device)
        idx = torch.tensor(active_muts, dtype=torch.long,
                          device=node_reprs.device)
        return torch.tanh(self.ctx_proj(node_reprs[idx].mean(0)))

    def forward(self, H, variants_mass, core_muts):
        """Compute MADE log-likelihood for observed variants.
        
        H: (V, K_variants) incidence matrix of current month
        variants_mass: list of (variant_frozenset, mass)
        core_muts: list of top-K_made mutation indices for MADE
        
        Returns scalar loss (negative weighted log-likelihood).
        Gradient flows through MADE -> context -> HGNN -> node_emb.
        """
        dev = self.node_emb.weight.device

        # get node representations via HGNN
        node_reprs = self.get_node_reprs(H)  # (V, d)

        # population context -- active is already local indices (0..V-1)
        active = list(set(
            k for v, _ in variants_mass
            for m in v
            for k in [self._mut2idx.get(m, -1)]
            if k >= 0 and k < self.V))
        ctx = self.get_context(node_reprs, active)  # (d,)

        # build binary vectors for observed variants over core_muts
        core_set = set(core_muts)
        c2k = {m: k for k, m in enumerate(core_muts)}
        X = torch.zeros(len(variants_mass), self.K_made, device=dev)
        W = torch.zeros(len(variants_mass), device=dev)
        for bi, (v, w) in enumerate(variants_mass):
            for m in v:
                if m in core_set:
                    X[bi, c2k[m]] = 1.0
            W[bi] = w

        # MADE log-likelihood -- gradient flows here
        lp = self.made.log_prob(X, ctx.unsqueeze(0).expand(len(variants_mass), -1))

        # weighted negative log-likelihood
        W = W / W.sum().clamp_min(1e-9)
        loss = -(W * lp).sum()
        return loss

    def generate(self, H, active_muts, core_muts, n_samples=100):
        """Generate candidate variants by sampling from MADE."""
        dev = self.node_emb.weight.device
        with torch.no_grad():
            node_reprs = self.get_node_reprs(H)
            ctx = self.get_context(node_reprs, active_muts)
            samples = self.made.sample(ctx, n=n_samples)  # (n, K_made)
        candidates = []
        for i in range(n_samples):
            muts = frozenset(core_muts[k] for k in range(self.K_made)
                           if samples[i, k].item() > 0.5)
            if muts:
                candidates.append(muts)
        return candidates


# ── training ──────────────────────────────────────────────────────────

def build_incidence(var_mass_ym, mut2idx, V):
    """Build incidence matrix H (V x K) for a month."""
    variants = list(var_mass_ym.keys())
    K = len(variants)
    if K == 0:
        return torch.zeros(V, 1), variants
    H = torch.zeros(V, K)
    for ki, v in enumerate(variants):
        for m in v:
            if m in mut2idx:
                H[mut2idx[m], ki] = 1.0
    return H, variants

def get_core_muts(var_mass_ym, mut2idx, K_made):
    """Top-K mutations by total mass in this month."""
    freq = defaultdict(float)
    for v, w in var_mass_ym.items():
        for m in v:
            if m in mut2idx:
                freq[mut2idx[m]] += w
    return [idx for idx, _ in sorted(freq.items(),
                                     key=lambda x: -x[1])[:K_made]]

def run(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # compute start month
    y, m2 = int(a.train_end[:4]), int(a.train_end[5:7])
    for _ in range(a.M - 1):
        m2 -= 1
        if m2 < 1: m2 = 12; y -= 1
    start_ym = f"{y:04d}-{m2:02d}"

    var_mass, months, mut2idx, V = load_monthly(
        a.events, start_ym, a.train_end, a.test_month)

    train_months = [m for m in months if m <= a.train_end]
    test_month   = a.test_month

    print(f"train: {train_months[0]}..{train_months[-1]} ({len(train_months)} months)")
    print(f"test:  {test_month}")

    # get core mutations from full training data
    all_freq = defaultdict(float)
    for ym in train_months:
        for v, w in var_mass[ym].items():
            for m in v:
                if m in mut2idx:
                    all_freq[mut2idx[m]] += w
    core_muts = [idx for idx, _ in sorted(all_freq.items(),
                                          key=lambda x: -x[1])[:a.K_made]]
    print(f"core mutations for MADE: {len(core_muts)}")

    model = HGNNMADEModel(V, len(core_muts), d=a.d, mut2idx=mut2idx).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=a.lr)

    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # training: one gradient update per consecutive month pair
    for ep in range(a.epochs):
        total_loss = 0.0
        for ti in range(len(train_months) - 1):
            t_ym  = train_months[ti]
            t1_ym = train_months[ti + 1]

            # build H from current month
            H, _ = build_incidence(var_mass[t_ym], mut2idx, V)
            H = H.to(device)

            # target: next month's variants
            variants_mass = sorted(var_mass[t1_ym].items(),
                                  key=lambda x: -x[1])

            if not variants_mass: continue

            loss = model(H, variants_mass, core_muts)

            if torch.isfinite(loss):
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total_loss += loss.item()

        print(f"ep {ep+1}/{a.epochs}  loss {total_loss:.4f}", flush=True)

    # evaluation
    print("\nevaluating...")
    last_train = train_months[-1]
    H_test, _ = build_incidence(var_mass[last_train], mut2idx, V)
    H_test = H_test.to(device)

    # circulating variants at train_end
    circ = var_mass[last_train]
    active_muts = list(set(mut2idx[m] for v in circ for m in v if m in mut2idx))

    # observed population at test month
    obs = var_mass.get(test_month, {})
    if not obs:
        print("no test data"); return

    # generate candidates
    candidates = model.generate(H_test, active_muts, core_muts,
                               n_samples=a.n_samples)
    print(f"generated {len(candidates)} candidates")
    print(f"unique candidates: {len(set(candidates))}")

    # score all candidates + circulating via MADE likelihood
    model.eval()
    with torch.no_grad():
        node_reprs = model.get_node_reprs(H_test)
        ctx = model.get_context(node_reprs, active_muts)
        core_set = set(core_muts)
        c2k = {m: k for k, m in enumerate(core_muts)}

        def score_variant(v):
            x = torch.zeros(1, len(core_muts), device=device)
            for m in v:
                mi = mut2idx.get(m)
                if mi is not None and mi in core_set:
                    x[0, c2k[mi]] = 1.0
            return model.made.log_prob(x, ctx.unsqueeze(0)).item()

        # score circulating
        circ_scores = {v: np.exp(score_variant(v)) for v in circ}
        # score candidates
        cand_scores = {v: np.exp(score_variant(v)) for v in set(candidates)}

    # predicted distribution:
    # existing variants keep persistence mass, adjusted by MADE score
    # new candidates get small budget proportional to MADE score
    budget = a.budget
    circ_tot2 = sum(circ.values()) or 1.0
    exist_pred = {v: (1-budget) * w/circ_tot2 for v, w in circ.items()}

    # new candidates only
    new_cands = {v: s for v, s in cand_scores.items() if v not in circ}
    new_tot = sum(new_cands.values()) or 1.0
    cand_pred = {v: budget * s/new_tot for v, s in new_cands.items()}

    pred = {**exist_pred, **cand_pred}

    # persistence
    circ_tot = sum(circ.values()) or 1.0
    persist  = {v: w/circ_tot for v, w in circ.items()}

    # compute overlaps
    obs_tot = sum(obs.values()) or 1.0
    obs_norm = {v: w/obs_tot for v, w in obs.items()}

    def overlap(p, q):
        return sum(min(p.get(v, 0), q.get(v, 0)) for v in set(p)|set(q))

    ov_model = overlap(pred, obs_norm)
    ov_persist = overlap(persist, obs_norm)

    # mass on new variants
    new_vars = set(obs_norm) - set(circ)
    mass_new_model = sum(pred.get(v, 0) for v in new_vars)
    mass_new_persist = 0.0

    print(f"\n{'='*50}")
    print(f"train {start_ym}..{a.train_end}  predict {test_month}")
    print(f"model       overlap {ov_model:.4f}  mass_new {mass_new_model:.4f}")
    print(f"persistence overlap {ov_persist:.4f}  mass_new {mass_new_persist:.4f}")
    print(f"gain        {ov_model - ov_persist:+.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events',     required=True)
    p.add_argument('--train-end',  default='2022-06', dest='train_end')
    p.add_argument('--test-month', default='2022-07', dest='test_month')
    p.add_argument('--M',          type=int, default=6)
    p.add_argument('--d',          type=int, default=64)
    p.add_argument('--K-made',     type=int, default=50, dest='K_made')
    p.add_argument('--epochs',     type=int, default=20)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--n-samples',  type=int, default=200, dest='n_samples')
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--budget',     type=float, default=0.1,
                   help='fraction of mass for new candidates')
    run(p.parse_args())

if __name__ == '__main__':
    main()
