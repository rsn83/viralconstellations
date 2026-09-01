#!/usr/bin/env python3
"""
172_full.py -- Full model with all three missing components.

1. Survival term: MADE log_prob IS log lambda. Negative sampling
   approximates the survival integral. Pushes probability away
   from non-occurring combinations.

2. Per-mutation intensity lambda_i(t): scalar head per mutation,
   trained via TPP loss. Teaches which mutations are rising vs declining.
   Gradient flows into node representations v_i(t).

3. Exponential decay gamma^dt: recent co-occurrence dominates old.
   Node embeddings decayed by gamma^(t - last_seen) before HGNN.
   Gamma is a learned parameter.

Training: weekly batches within M training months.
Each week: one gradient update with positive + negative sampling.

Usage:
  python scripts/172_full.py \
    --events data/processed/events_v3.tsv \
    --train-end 2022-06 --test-month 2022-07 \
    --M 24 --d 64 --K-made 50 --epochs 5 \
    --n-neg 20 --n-samples 500 --seed 0
"""
import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# ── data ──────────────────────────────────────────────────────────────

def load_weekly(path, start_ym, end_ym, test_ym):
    """Load events aggregated by ISO week."""
    by_week = defaultdict(lambda: defaultdict(float))
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if ln == 0 and not parts[0][:4].isdigit(): continue
            date, muts = parts[0].strip(), parts[1].strip()
            cnt = float(parts[2]) if len(parts) > 2 else 1.0
            ym = date[:7]
            if start_ym <= ym <= test_ym:
                s = frozenset(int(x) for x in muts.split(',') if x)
                if s:
                    # week key: YYYY-WW
                    from datetime import date as dobj
                    try:
                        d = dobj.fromisoformat(date)
                        wk = f"{d.isocalendar()[0]:04d}-{d.isocalendar()[1]:02d}"
                    except ValueError:
                        continue
                    by_week[wk][s] += cnt

    weeks = sorted(by_week.keys())
    # which weeks belong to which month
    week2ym = {}
    from datetime import date as dobj
    import datetime
    for wk in weeks:
        y, w = int(wk[:4]), int(wk[5:])
        # get date of monday of that week
        monday = dobj.fromisocalendar(y, w, 1)
        week2ym[wk] = monday.strftime('%Y-%m')

    var_mass = {}
    for wk in weeks:
        tot = sum(by_week[wk].values()) or 1.0
        var_mass[wk] = {s: v/tot for s, v in by_week[wk].items()}

    all_muts = sorted({m for wk in weeks for v in var_mass[wk] for m in v})
    mut2idx = {m: i for i, m in enumerate(all_muts)}
    V = len(all_muts)
    print(f"weeks {len(weeks)}  V {V}  ({weeks[0]}..{weeks[-1]})")
    return var_mass, weeks, week2ym, mut2idx, V

def month_population(path, test_ym, mut2idx):
    by_month = defaultdict(lambda: defaultdict(float))
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if ln == 0 and not parts[0][:4].isdigit(): continue
            date, muts = parts[0].strip(), parts[1].strip()
            cnt = float(parts[2]) if len(parts) > 2 else 1.0
            ym = date[:7]
            if ym == test_ym:
                s = frozenset(int(x) for x in muts.split(',') if x)
                if s: by_month[ym][s] += cnt
    tot = sum(by_month[test_ym].values()) or 1.0
    return {s: v/tot for s, v in by_month[test_ym].items()}

# ── model ─────────────────────────────────────────────────────────────

class FourierTime(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(
            1.0/10**np.linspace(0,3,d)).float())
        self.b = nn.Parameter(torch.zeros(d))
    def forward(self, dt):
        if not torch.is_tensor(dt): dt = torch.tensor(float(dt))
        if dt.dim()==0: dt = dt.unsqueeze(0)
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

class FullModel(nn.Module):
    def __init__(self, V, K_made, d=64, d_hidden=256, mut2idx=None):
        super().__init__()
        self.V = V
        self.K_made = K_made
        self.d = d
        self._mut2idx = mut2idx or {}

        # node embeddings
        self.node_emb = nn.Embedding(V, d)
        nn.init.normal_(self.node_emb.weight, 0, 0.1)

        # COMPONENT 3: learned exponential decay
        # gamma in (0.5, 1.0) via sigmoid
        self.log_gamma = nn.Parameter(torch.tensor(-2.0))

        # Fourier time encoding
        self.psi = FourierTime(d)

        # HGNN for co-occurrence structure
        self.hgnn = HGNN(d)

        # COMPONENT 2: per-mutation intensity lambda_i(t)
        # scalar head: v_i -> lambda_i
        self.intensity_head = nn.Sequential(
            nn.Linear(d, d), nn.Tanh(),
            nn.Linear(d, 1), nn.Softplus())

        # context projection
        self.ctx_proj = nn.Linear(d, d)

        # COMPONENT 1: ParallelJointScorer as lambda_B
        # skip connections ensure gradient flows
        self.d_hidden = d_hidden
        self.gru = nn.GRUCell(d, d_hidden)
        # init forget gate to 1 -- avoids vanishing gradient
        nn.init.ones_(self.gru.bias_ih[d_hidden:2*d_hidden])
        nn.init.ones_(self.gru.bias_hh[d_hidden:2*d_hidden])
        # pool v_core (K*d) -> d before scorer -- saves params
        self.v_pool = nn.Linear(d, d, bias=False)
        self.scorer_deep = nn.Sequential(
            nn.Linear(d_hidden + d, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, K_made))
        self.scorer_skip_h = nn.Linear(d_hidden, K_made, bias=False)
        self.scorer_skip_v = nn.Linear(d, K_made, bias=False)
        nn.init.zeros_(self.scorer_deep[-1].weight)
        nn.init.zeros_(self.scorer_deep[-1].bias)
        nn.init.zeros_(self.scorer_skip_h.weight)
        nn.init.zeros_(self.scorer_skip_v.weight)

        # persistent state
        self.register_buffer('h_gru', torch.zeros(d_hidden))
        self.register_buffer('last_seen_week', torch.zeros(V))
        self._current_week = 0

    def reset(self):
        self.h_gru.zero_()
        self.last_seen_week.zero_()
        self._current_week = 0

    def get_node_reprs(self, H, week_idx=None):
        """Node representations with exponential decay.

        v_i(t) = tanh(HGNN(decay(emb_i), H) + psi(dt_i))

        decay(emb_i) = emb_i * gamma^(t - last_seen_i)
        dt_i = time since mutation i last appeared
        """
        dev = self.node_emb.weight.device
        gamma = torch.sigmoid(self.log_gamma) * 0.5 + 0.5  # in (0.5, 1.0)

        t = float(week_idx if week_idx is not None else self._current_week)
        dt = (t - self.last_seen_week).clamp_min(0)  # (V,) weeks since last seen

        # COMPONENT 3: decay node embeddings by gamma^dt
        decay = gamma.pow(dt)                              # (V,)
        X_decayed = self.node_emb.weight * decay.unsqueeze(1)  # (V, d)

        # HGNN on decayed embeddings
        X_hgnn = self.hgnn(H, X_decayed)                 # (V, d)

        # temporal drift via Fourier encoding
        phi = self.psi(dt * 7.0)                         # (V, d) -- dt in days
        return torch.tanh(X_hgnn + phi)                  # (V, d)

    def get_context(self, node_reprs, active_muts):
        if not active_muts:
            return torch.zeros(self.d, device=node_reprs.device)
        idx = torch.tensor(active_muts, dtype=torch.long,
                          device=node_reprs.device)
        return torch.tanh(self.ctx_proj(node_reprs[idx].mean(0)))

    def log_lambda_B(self, x, v_core, h):
        """log lambda_B = log_prob of binary vector x.

        COMPONENT 1: MADE/scorer IS the intensity function.
        x:      (B, K) binary
        v_core: (K, d) per-position node representations
        h:      (d_hidden,) GRU state
        Returns (B,) log intensities.
        """
        B = x.shape[0]
        h_exp = h.unsqueeze(0).expand(B, -1)
        v_flat = v_core.reshape(1, -1).expand(B, -1)
        # pool v_core to d dims
        v_pooled = torch.tanh(self.v_pool(v_core)).mean(0).unsqueeze(0).expand(B,-1)
        inp = torch.cat([h_exp, v_pooled], dim=-1)
        logits = (self.scorer_deep(inp) +
                  self.scorer_skip_h(h_exp) +
                  self.scorer_skip_v(v_pooled))
        return -F.binary_cross_entropy_with_logits(
            logits, x, reduction='none').sum(-1)

    def forward(self, H, var_mass_week, core_muts, neg_variants,
                week_idx, delta_weeks):
        """Full TPP loss for one week.

        L = -sum_B w_B * log_lambda_B(t)   [positive: observed variants]
          + sum_neg exp(log_lambda_neg(t))  [survival: random negatives]
          + lambda_i loss                   [per-mutation intensity]

        All terms flow gradient into node_emb, hgnn, scorer, intensity_head.
        """
        dev = self.node_emb.weight.device
        core_set = set(core_muts)
        c2k = {m: k for k, m in enumerate(core_muts)}

        # node representations WITH gradient
        node_reprs = self.get_node_reprs(H, week_idx)

        # compute h WITH gradient for this step
        active = list({m for v in var_mass_week for m in v
                      if self._mut2idx.get(m,-1) >= 0
                      and self._mut2idx.get(m,-1) < self.V})
        ctx = self.get_context(node_reprs, [self._mut2idx[m]
                               for m in active if m in self._mut2idx])
        h_diff = self.gru(ctx.unsqueeze(0),
                         self.h_gru.detach().unsqueeze(0)).squeeze(0)

        # core node representations
        core_t = torch.tensor(core_muts, dtype=torch.long, device=dev)
        v_core = node_reprs[core_t]  # (K, d) -- in gradient graph

        # ── COMPONENT 1: positive log_lambda ──────────────────────────
        variants = list(var_mass_week.items())
        X_pos = torch.zeros(len(variants), self.K_made, device=dev)
        W_pos = torch.zeros(len(variants), device=dev)
        for bi, (v, w) in enumerate(variants):
            for m in v:
                mi = self._mut2idx.get(m,-1)
                if mi in core_set: X_pos[bi, c2k[mi]] = 1.0
            W_pos[bi] = w
        W_pos = W_pos / W_pos.sum().clamp_min(1e-9)
        lp_pos = self.log_lambda_B(X_pos, v_core, h_diff)
        loss_pos = -(W_pos * lp_pos).sum()

        # ── COMPONENT 1: survival term (negative sampling) ────────────
        if neg_variants:
            X_neg = torch.zeros(len(neg_variants), self.K_made, device=dev)
            for ni, neg in enumerate(neg_variants):
                for m in neg:
                    mi = self._mut2idx.get(m,-1)
                    if mi in core_set: X_neg[ni, c2k[mi]] = 1.0
            lp_neg = self.log_lambda_B(X_neg, v_core, h_diff)
            # survival: sum exp(log_lambda) * delta_t (delta in weeks)
            loss_survival = torch.exp(lp_neg).sum() * float(delta_weeks)
        else:
            loss_survival = torch.tensor(0.0, device=dev)

        # ── COMPONENT 2: per-mutation intensity ───────────────────────
        # observed mutations should have high lambda_i
        # unobserved (in negatives) should have low lambda_i
        obs_muts = list({self._mut2idx[m] for v,_ in variants
                        for m in v if m in self._mut2idx
                        and self._mut2idx[m] < self.V})
        neg_muts = list({self._mut2idx[m] for neg in neg_variants
                        for m in neg if m in self._mut2idx
                        and self._mut2idx[m] < self.V
                        and self._mut2idx[m] not in set(obs_muts)})[:50]

        lambda_loss = torch.tensor(0.0, device=dev)
        if obs_muts:
            obs_t = torch.tensor(obs_muts, dtype=torch.long, device=dev)
            lam_obs = self.intensity_head(node_reprs[obs_t]).squeeze(-1)
            lambda_loss = lambda_loss - torch.log(lam_obs + 1e-9).mean()
        if neg_muts:
            neg_t = torch.tensor(neg_muts, dtype=torch.long, device=dev)
            lam_neg = self.intensity_head(node_reprs[neg_t]).squeeze(-1)
            lambda_loss = lambda_loss + lam_neg.mean() * float(delta_weeks)

        total = loss_pos + 0.01 * loss_survival + 0.01 * lambda_loss.clamp(-5, 5)
        return total, float(loss_pos.detach()), \
               float(loss_survival.detach()), float(lambda_loss.detach())

    def update_state(self, H, var_mass_week, week_idx):
        """Update GRU state and last_seen after gradient step."""
        with torch.no_grad():
            node_reprs = self.get_node_reprs(H, week_idx)
            active = list({self._mut2idx[m]
                          for v in var_mass_week
                          for m in v if m in self._mut2idx
                          and self._mut2idx[m] < self.V})
            if active:
                ctx = self.get_context(node_reprs, active)
                self.h_gru = self.gru(
                    ctx.unsqueeze(0),
                    self.h_gru.detach().unsqueeze(0)).squeeze(0).detach()
            # update last seen
            for v in var_mass_week:
                for m in v:
                    mi = self._mut2idx.get(m,-1)
                    if 0 <= mi < self.V:
                        self.last_seen_week[mi] = float(week_idx)
            self._current_week = week_idx

    def predict(self, H, core_muts, week_idx, h_weeks=4):
        """Predict at T+h_weeks by projecting node representations."""
        dev = self.node_emb.weight.device
        core_t = torch.tensor(core_muts, dtype=torch.long, device=dev)
        with torch.no_grad():
            # project to T+h: decay by additional h_weeks
            gamma = torch.sigmoid(self.log_gamma)*0.5 + 0.5
            future_week = week_idx + h_weeks
            dt = (future_week - self.last_seen_week).clamp_min(0)
            decay = gamma.pow(dt)
            X_proj = self.node_emb.weight * decay.unsqueeze(1)
            X_hgnn = self.hgnn(H, X_proj)
            phi = self.psi(dt * 7.0)
            node_reprs_proj = torch.tanh(X_hgnn + phi)
            v_core = node_reprs_proj[core_t]
            # sample
            B = 1
            h_exp = self.h_gru.unsqueeze(0)
            v_flat = v_core.reshape(1,-1)
            v_pooled2 = torch.tanh(self.v_pool(v_core)).mean(0).unsqueeze(0)
            inp = torch.cat([h_exp, v_pooled2], dim=-1)
            logits = (self.scorer_deep(inp) +
                     self.scorer_skip_h(h_exp) +
                     self.scorer_skip_v(v_pooled2))
        return logits.squeeze(0), node_reprs_proj, v_core

    def sample_variants(self, logits, core_muts, n=500):
        """Sample candidate variants from scorer logits."""
        idx2mut = {v: k for k, v in self._mut2idx.items()}
        probs = torch.sigmoid(logits)
        candidates = []
        for _ in range(n):
            sample = torch.bernoulli(probs)
            muts = frozenset(
                idx2mut[core_muts[k]]
                for k in range(self.K_made)
                if sample[k].item() > 0.5
                and idx2mut.get(core_muts[k]) is not None)
            if muts: candidates.append(muts)
        return candidates

# ── negative sampling ──────────────────────────────────────────────────

def sample_negatives(var_mass_week, all_muts, n_neg):
    """Sample negative variants by corrupting observed ones."""
    observed = list(var_mass_week.keys())
    negs = []
    for _ in range(n_neg):
        base = random.choice(observed)
        members = list(base)
        # flip 2-4 random positions
        n_flip = random.randint(2, 4)
        for _ in range(n_flip):
            if random.random() < 0.5 and members:
                members.pop(random.randint(0, len(members)-1))
            else:
                members.append(random.choice(all_muts))
        negs.append(frozenset(members))
    return negs

# ── run ───────────────────────────────────────────────────────────────

def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    var_mass, weeks, week2ym, mut2idx, V = load_weekly(
        a.events, '2000-01', a.train_end, a.test_month)
    all_muts_list = list(mut2idx.keys())

    train_weeks = [w for w in weeks if week2ym[w] <= a.train_end]
    print(f"train weeks: {len(train_weeks)} ({week2ym[train_weeks[0]]}..{week2ym[train_weeks[-1]]})")

    # core mutations: top-K by total mass
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

    def build_H(wk):
        vm = var_mass[wk]
        variants = list(vm.keys())
        K = max(len(variants), 1)
        H = torch.zeros(V, K, device=device)
        for ki, v in enumerate(variants):
            w = vm[v]
            for m in v:
                if m in mut2idx: H[mut2idx[m], ki] = w
        return H

    # training -- reset once before training, not between epochs
    model.reset()
    for ep in range(a.epochs):
        total_loss = pos_loss = surv_loss = lam_loss = 0.0

        for wi, wk in enumerate(train_weeks):
            H = build_H(wk)
            vm = var_mass[wk]
            if not vm: continue

            # negative sampling
            negs = sample_negatives(vm, all_muts_list, a.n_neg)

            # delta_weeks: time gap to next week
            delta = (float(wi+1)/len(train_weeks)) if wi < len(train_weeks)-1 else 1.0

            loss, lp, ls, ll = model(H, vm, core_muts, negs,
                                     week_idx=wi,
                                     delta_weeks=delta)
            if torch.isfinite(loss):
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total_loss += loss.item()
                pos_loss += lp; surv_loss += ls; lam_loss += ll

            # update state AFTER gradient step
            model.update_state(H, vm, wi)

        n = len(train_weeks)
        print(f"ep {ep+1}/{a.epochs}  "
              f"loss {total_loss/n:.3f}  "
              f"pos {pos_loss/n:.3f}  "
              f"surv {surv_loss/n:.3f}  "
              f"lam {lam_loss/n:.3f}  "
              f"gamma {torch.sigmoid(model.log_gamma).item()*0.5+0.5:.3f}",
              flush=True)

    # evaluation
    print("\nevaluating...")
    last_wk = train_weeks[-1]
    last_wi = len(train_weeks) - 1
    H_last = build_H(last_wk)

    # state already built from training -- no need to rebuild

    # circulating at train_end
    last_ym = a.train_end
    circ_raw = defaultdict(float)
    for wk in train_weeks:
        if week2ym[wk] == last_ym:
            for v, w in var_mass[wk].items(): circ_raw[v] += w
    circ_tot = sum(circ_raw.values()) or 1.0
    circ = {v: w/circ_tot for v, w in circ_raw.items()}

    # observed population at test month
    obs = month_population(a.events, a.test_month, mut2idx)
    obs_tot = sum(obs.values()) or 1.0
    obs_norm = {v: w/obs_tot for v, w in obs.items()}

    # predict at T+h
    h_weeks = a.horizon * 4  # ~4 weeks per month
    logits, node_reprs_proj, v_core = model.predict(
        H_last, core_muts, last_wi, h_weeks)
    candidates = model.sample_variants(logits, core_muts, n=a.n_samples)
    print(f"generated {len(candidates)} candidates, "
          f"unique {len(set(candidates))}")

    # score candidates
    with torch.no_grad():
        def score(v):
            x = torch.zeros(1, len(core_muts), device=device)
            for m in v:
                mi = mut2idx.get(m,-1)
                if mi in core_set: x[0,c2k[mi]] = 1.0
            lp = model.log_lambda_B(x, v_core,
                                    model.h_gru)
            return float(torch.exp(lp).item())

    new_cands = {v: score(v) for v in set(candidates) if v not in circ}
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
    print(f"train {train_weeks[0]}..{a.train_end}  predict {a.test_month}  h={a.horizon}m")
    print(f"model       overlap {ov_m:.4f}  mass_new {mass_new:.4f}")
    print(f"persistence overlap {ov_p:.4f}  mass_new 0.0000")
    print(f"gain        {ov_m-ov_p:+.4f}")
    print(f"gamma (decay): {torch.sigmoid(model.log_gamma).item()*0.5+0.5:.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events',    required=True)
    p.add_argument('--train-end', default='2022-06', dest='train_end')
    p.add_argument('--test-month',default='2022-07', dest='test_month')
    p.add_argument('--d',         type=int, default=64)
    p.add_argument('--d-hidden',  type=int, default=256, dest='d_hidden')
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
