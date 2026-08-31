#!/usr/bin/env python3
"""
169_clean.py -- Their node representation + our horizon head. Nothing else.

Train: full event stream up to --train-end, GRU memory updated each day.
Test:  freeze at train-end, predict population at train-end + h months.
       Compare overlap vs persistence.

No windowing. No pop-support cutoff. No assembly. No budget.
Just: can their node representations predict which variants grow?

Usage:
  python scripts/169_clean.py \
    --events data/processed/events_v3.tsv \
    --vocab data/processed/vocab_v3.tsv \
    --train-end 2022-06 \
    --horizon 1 \
    --epochs 10 \
    --seed 0
"""
import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# ── data ──────────────────────────────────────────────────────────────

def load_events(path):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if ln == 0 and not parts[0][:4].isdigit(): continue
            date, muts = parts[0].strip(), parts[1].strip()
            cnt = float(parts[2]) if len(parts) > 2 else 1.0
            s = frozenset(int(x) for x in muts.split(',') if x)
            if s: rows.append((date, s, cnt))
    rows.sort(key=lambda r: r[0])
    days = sorted({r[0] for r in rows})
    day_ix = {d: i for i, d in enumerate(days)}
    events = [(s, day_ix[d], float(cnt)) for d, s, cnt in rows]
    V = 1 + max(max(s) for s, _, _ in events)
    print(f"events {len(events):,}  days {len(days)}  ({days[0]}..{days[-1]})  V {V}")
    return events, days, V

def load_first_seen(vocab_path, V, cutoff_date):
    ok = torch.ones(V, dtype=torch.bool)
    if not (vocab_path and os.path.exists(vocab_path)): return ok
    with open(vocab_path) as f:
        for line in f:
            p = line.rstrip().split()
            if len(p) >= 3 and p[0].isdigit():
                i = int(p[0])
                if i < V and p[2] > cutoff_date:
                    ok[i] = False
    return ok

def group_by_day(events):
    by_day = defaultdict(list)
    for s, t, w in events:
        by_day[t].append((s, w))
    return by_day

def month_population(events, days, ym):
    """Normalised variant mass for a given year-month string."""
    agg = defaultdict(float)
    for s, ti, w in events:
        if days[ti][:7] == ym:
            agg[s] += w
    tot = sum(agg.values()) or 1.0
    return {s: v/tot for s, v in agg.items()}

# ── model ─────────────────────────────────────────────────────────────

class FourierTime(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(
            1.0 / 10**np.linspace(0, 3, d)).float())
        self.b = nn.Parameter(torch.zeros(d))
    def forward(self, dt):
        return torch.cos(dt.unsqueeze(-1) * self.w + self.b)

class NodeModel(nn.Module):
    """
    Their node representation (Section 3.2) + our fitness head.

    Node repr v_i(t):
      - GRU memory updated at each event
      - Neighbourhood attention over last N hyperedges involving node i
      - v_i(t) = tanh(W_s Mem_i + W_r v^r(t) + b_v)

    Our addition:
      - fitness(mean(v_i for i in variant), psi(h)) -> growth scalar
      - softmax(log_mass + fitness) -> predicted distribution at T+h
      - zero-initialized -> starts exactly at persistence
    """
    def __init__(self, V, d=64, N=10):
        super().__init__()
        self.V, self.d, self.N = V, d, N
        self.psi = FourierTime(d)

        # their GRU memory
        self.gru = nn.GRUCell(3*d, d)
        self.log_gamma = nn.Parameter(torch.tensor(-3.0))

        # their neighbourhood attention (eq 5)
        self.nbr_attn = nn.MultiheadAttention(2*d, 2, batch_first=True)
        self.W_r = nn.Linear(2*d, d)
        self.W_s = nn.Linear(d, d)
        self.b_v = nn.Parameter(torch.zeros(d))

        # our fitness head: variant repr (d) + horizon (d) -> scalar
        self.fitness = nn.Sequential(
            nn.Linear(2*d, 2*d), nn.Tanh(), nn.Linear(2*d, 1))
        nn.init.zeros_(self.fitness[-1].weight)
        nn.init.zeros_(self.fitness[-1].bias)

        # persistent state
        self.register_buffer('mem',     torch.zeros(V, d))
        self.register_buffer('last_t',  torch.zeros(V))
        self.register_buffer('nbr_vec', torch.zeros(V, N, d))
        self.register_buffer('nbr_tb',  torch.zeros(V, N))
        self.register_buffer('nbr_cnt', torch.zeros(V, dtype=torch.long))
        self.register_buffer('mem_ok',  torch.ones(V, 1))
        self._cache_t = None
        self._cache_v = None
        self._pending = None

    def reset(self):
        for buf in [self.mem, self.last_t, self.nbr_vec, self.nbr_tb]:
            buf.zero_()
        self.nbr_cnt.zero_()
        self._cache_t = self._cache_v = self._pending = None

    def flush(self, t_now):
        if self._pending is None: return
        idx, msg = self._pending
        dev = self.mem.device
        cur = self.mem[idx] * self.mem_ok[idx]
        dt  = (t_now - self.last_t[idx]).clamp_min(0)
        g   = torch.sigmoid(self.log_gamma) * 0.5 + 0.5
        cur = cur * g.pow(dt).unsqueeze(-1)
        new = self.gru(msg.to(dev), cur)
        with torch.no_grad():
            self.mem[idx] = new.detach()
            self.last_t[idx] = t_now
        self._pending = None
        self._cache_t = None

    def node_repr(self, t_now):
        if self._cache_t == t_now and self._cache_v is not None:
            return self._cache_v
        dev = self.mem.device
        m   = self.mem * self.mem_ok
        dt  = (t_now - self.last_t).clamp_min(0)
        g   = torch.sigmoid(self.log_gamma) * 0.5 + 0.5
        m   = m * g.pow(dt).unsqueeze(-1)
        dt2 = (t_now - self.nbr_tb).clamp_min(0)
        ctx = torch.cat([self.nbr_vec, self.psi(dt2)], dim=-1)
        mask = (torch.arange(self.N, device=dev).unsqueeze(0) >=
                self.nbr_cnt.clamp(max=self.N).unsqueeze(1))
        mask[mask.all(1), 0] = False
        q   = torch.cat([m, self.psi(torch.zeros(self.V, device=dev))], dim=-1)
        nbr, _ = self.nbr_attn(q.unsqueeze(1), ctx, ctx,
                               key_padding_mask=mask, need_weights=False)
        v = torch.tanh(self.W_s(m) + self.W_r(nbr.squeeze(1)) + self.b_v)
        self._cache_v = v
        self._cache_t = t_now
        return v

    def observe(self, variants, t_now, max_k=64):
        """Update memory with observed hyperedges -- their Section 3.2."""
        seen = sorted({m for s in variants for m in s if m < self.V})
        if not seen: return
        dev   = self.mem.device
        V_rep = self.node_repr(t_now)
        idx   = torch.tensor(seen, dtype=torch.long)
        v     = V_rep[idx.to(dev)]
        pos   = {m: i for i, m in enumerate(seen)}
        agg   = torch.zeros(len(seen), self.d, device=dev)
        cnt   = torch.zeros(len(seen), 1,      device=dev)
        for s in variants:
            ms = [m for m in list(s)[:max_k] if m < self.V and m in pos]
            if not ms: continue
            rows = torch.tensor([pos[m] for m in ms], dtype=torch.long, device=dev)
            ctx  = V_rep[torch.tensor(ms, dtype=torch.long, device=dev)].mean(0, keepdim=True)
            agg.index_add_(0, rows, ctx.expand(len(rows), -1))
            cnt.index_add_(0, rows, torch.ones(len(rows), 1, device=dev))
        agg  = agg / cnt.clamp_min(1.0)
        dt   = (t_now - self.last_t[idx]).clamp_min(0).to(dev)
        msg  = torch.cat([v.detach(), agg.detach(), self.psi(dt)], dim=-1)
        self._pending = (idx, msg.detach())
        with torch.no_grad():
            slot = self.nbr_cnt[idx] % self.N
            self.nbr_vec[idx, slot] = agg.detach()
            self.nbr_tb [idx, slot] = t_now
            self.nbr_cnt[idx] += 1
        self._cache_t = None

    def predict(self, variants_mass, t_now, h_days):
        """Predicted distribution at T + h_days."""
        dev  = self.mem.device
        vars = [v for v, _ in variants_mass]
        mass = torch.tensor([w for _, w in variants_mass],
                            dtype=torch.float32, device=dev)
        table = self.node_repr(t_now)
        reps  = []
        for v in vars:
            ms = [m for m in v if m < self.V]
            reps.append(table[torch.tensor(ms, dtype=torch.long, device=dev)].mean(0)
                        if ms else torch.zeros(self.d, device=dev))
        X    = torch.stack(reps)
        psi_h = self.psi(torch.tensor([float(h_days)], device=dev)).expand(len(vars), -1)
        feat  = torch.cat([X, psi_h], dim=-1)
        fit   = self.fitness(feat).squeeze(-1)
        logm  = torch.log(mass.clamp_min(1e-9))
        lp    = torch.log_softmax(logm + fit, dim=0)
        return lp, vars

# ── train / eval ───────────────────────────────────────────────────────

def run(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    events, days, V = load_events(a.events)
    by_day = group_by_day(events)
    all_days = sorted(by_day.keys())

    # split by date
    train_days = [t for t in all_days if days[t] <= a.train_end]
    test_month  = a.test_month   # year-month string e.g. "2022-07"
    print(f"train days: {len(train_days)} (up to {a.train_end})")
    print(f"predicting: {test_month}  horizon h={a.horizon}m")

    # observed population for training target and test
    # training target = a.horizon months after each training window end
    # for simplicity: one training target = the test month itself
    target_pop = month_population(events, days, test_month)
    if not target_pop:
        print(f"ERROR: no data for {test_month}"); return

    model = NodeModel(V, d=a.d, N=a.n_recent).to(device)
    if a.vocab:
        model.mem_ok.copy_(
            load_first_seen(a.vocab, V, a.train_end).float().unsqueeze(-1).to(device))
        print(f"vocab: {(~model.mem_ok.squeeze().bool()).sum()} mutations suppressed")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    h_days = float(a.horizon * 30)

    # build training pairs: origins spaced ~30 days apart within training data
    # each origin: circulating at that day, target = month h months later
    h_gap = int(h_days)
    train_origins = []
    for t in train_days[::30]:   # one origin per ~month
        target_date_min = days[t][:7]
        # find day h months later (approximate)
        future_days = [u for u in all_days
                       if days[u][:7] > target_date_min
                       and days[u] <= a.train_end]
        if not future_days: continue
        # target month = approximately h months after origin
        import calendar
        ym = days[t][:7]
        y, m = int(ym[:4]), int(ym[5:7])
        m += a.horizon
        if m > 12: y += m//12; m = m%12 or 12
        tgt_ym = f"{y:04d}-{m:02d}"
        tgt_pop = month_population(events, days, tgt_ym)
        if tgt_pop:
            train_origins.append((t, tgt_pop, tgt_ym))

    print(f"training origins: {len(train_origins)}")

    for ep in range(a.epochs):
        model.reset()
        losses = []
        for t in train_days:
            model.flush(float(t))
            model.observe([s for s, _ in by_day[t]], float(t))
            # is this a training origin?
            for orig_t, tgt_pop_i, tgt_ym in train_origins:
                if orig_t != t: continue
                circ_raw_i = defaultdict(float)
                for u in train_days[max(0,train_days.index(t)-30):train_days.index(t)+1]:
                    for s, w in by_day[u]:
                        circ_raw_i[s] += w
                tot_i = sum(circ_raw_i.values()) or 1.0
                cm_i = sorted(circ_raw_i.items(), key=lambda kv: -kv[1])[:500]
                cm_i = [(s, w/tot_i) for s, w in cm_i]
                if len(cm_i) < 2: continue
                lp_i, vars_i = model.predict(cm_i, float(t), h_days)
                ix_i  = {v: j for j, v in enumerate(vars_i)}
                tot2_i = sum(tgt_pop_i.values()) or 1.0
                I_i, W_i = [], []
                for v, w in tgt_pop_i.items():
                    j = ix_i.get(v)
                    if j is not None: I_i.append(j); W_i.append(w/tot2_i)
                if not I_i: continue
                loss_i = -(torch.tensor(W_i, device=device)
                           * lp_i[torch.tensor(I_i, dtype=torch.long, device=device)]
                           ).sum() / max(sum(W_i), 1e-9)
                if torch.isfinite(loss_i):
                    opt.zero_grad(); loss_i.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step(); losses.append(float(loss_i.detach()))
        print(f"ep {ep+1}/{a.epochs}  loss {np.mean(losses) if losses else float('nan'):.4f}", flush=True)

    # circulating at train_end for evaluation
    last_t = train_days[-1]
    circ_raw = defaultdict(float)
    for t in train_days[-30:]:
        for s, w in by_day[t]:
            circ_raw[s] += w
    tot = sum(circ_raw.values()) or 1.0
    circ_mass = sorted(circ_raw.items(), key=lambda kv: -kv[1])
    circ_mass = [(s, w/tot) for s, w in circ_mass]

    # ── evaluation ────────────────────────────────────────────────────
    model.reset()
    for t in train_days:
        model.flush(float(t))
        model.observe([s for s, _ in by_day[t]], float(t))

    with torch.no_grad():
        lp, vars = model.predict(circ_mass, float(last_t), h_days)

    p   = torch.exp(lp).detach().cpu().numpy()
    ix  = {v: i for i, v in enumerate(vars)}
    tot2 = sum(target_pop.values()) or 1.0
    q   = np.zeros_like(p)
    for v, w in target_pop.items():
        j = ix.get(v)
        if j is not None: q[j] += w/tot2

    # persistence
    mass_arr = np.array([w for _, w in circ_mass])
    p_per    = mass_arr / mass_arr.sum()
    q_per    = np.zeros(len(circ_mass))
    ix2 = {v: i for i, v in enumerate(v for v, _ in circ_mass)}
    for v, w in target_pop.items():
        j = ix2.get(v)
        if j is not None: q_per[j] += w/tot2

    ov_m = float(np.minimum(p, q).sum())
    ov_p = float(np.minimum(p_per, q_per).sum())
    cov  = float(q.sum())
    mn   = float(q[len(circ_mass):].sum()) if len(p) > len(circ_mass) else 0.0

    print(f"\n{'='*50}")
    print(f"train end: {a.train_end}  predict: {test_month}  h={a.horizon}m")
    print(f"model       overlap {ov_m:.4f}")
    print(f"persistence overlap {ov_p:.4f}")
    print(f"gain        {ov_m - ov_p:+.4f}")
    print(f"covered     {cov:.3f}  mass_new {mn:.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events',     required=True)
    p.add_argument('--vocab',      default=None)
    p.add_argument('--train-end',  default='2022-06', dest='train_end',
                   help='last training month e.g. 2022-06')
    p.add_argument('--test-month', default='2022-07', dest='test_month',
                   help='month to predict e.g. 2022-07')
    p.add_argument('--horizon',    type=int, default=1,
                   help='horizon in months (should match train-end to test-month gap)')
    p.add_argument('--d',          type=int, default=64)
    p.add_argument('--n-recent',   type=int, default=10, dest='n_recent')
    p.add_argument('--epochs',     type=int, default=10)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--seed',       type=int, default=0)
    run(p.parse_args())

if __name__ == '__main__':
    main()
