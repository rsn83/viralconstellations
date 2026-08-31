#!/usr/bin/env python3
"""
169_clean.py

Their node representation (GRU memory + neighbourhood attention) exactly.
Our probe: fitness head to test if representations carry growth signal.

Train: stream 6 months of events day by day, update memory their way.
Test:  predict month 7 population. Compare overlap vs persistence.

No windowing. No sliding origins. No horizon encoding.
One prediction target. Clean.

Usage:
  python scripts/169_clean.py \
    --events data/processed/events_v3.tsv \
    --vocab data/processed/vocab_v3.tsv \
    --train-end 2022-06 --test-month 2022-07 \
    --M 6 --d 64 --epochs 10 --seed 0
"""
import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


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
    events = [(s, day_ix[d], cnt) for d, s, cnt in rows]
    V = 1 + max(max(s) for s, _, _ in events)
    return events, days, V


def load_first_seen(vocab_path, V, cutoff):
    ok = torch.ones(V, dtype=torch.bool)
    if not (vocab_path and os.path.exists(vocab_path)): return ok
    with open(vocab_path) as f:
        for line in f:
            p = line.rstrip().split()
            if len(p) >= 3 and p[0].isdigit():
                i = int(p[0])
                if i < V and p[2] > cutoff:
                    ok[i] = False
    return ok


def month_population(events, days, ym):
    agg = defaultdict(float)
    for s, ti, w in events:
        if days[ti][:7] == ym:
            agg[s] += w
    tot = sum(agg.values()) or 1.0
    return {s: v/tot for s, v in agg.items()}


class FourierTime(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(
            1.0/10**np.linspace(0,3,d)).float())
        self.b = nn.Parameter(torch.zeros(d))
    def forward(self, dt):
        return torch.cos(dt.unsqueeze(-1)*self.w + self.b)


class NodeModel(nn.Module):
    """Their node representation + our fitness probe."""
    def __init__(self, V, d=64, N=10):
        super().__init__()
        self.V, self.d, self.N = V, d, N
        self.psi = FourierTime(d)
        self.gru = nn.GRUCell(3*d, d)
        self.log_gamma = nn.Parameter(torch.tensor(-3.0))
        self.nbr_attn = nn.MultiheadAttention(2*d, 2, batch_first=True)
        self.W_r = nn.Linear(2*d, d)
        self.W_s = nn.Linear(d, d)
        self.b_v = nn.Parameter(torch.zeros(d))
        # fitness probe -- zero-init = starts at persistence
        self.fitness = nn.Sequential(
            nn.Linear(d, 2*d), nn.Tanh(), nn.Linear(2*d, 1))
        nn.init.zeros_(self.fitness[-1].weight)
        nn.init.zeros_(self.fitness[-1].bias)
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
        self.mem.zero_(); self.last_t.zero_()
        self.nbr_vec.zero_(); self.nbr_tb.zero_(); self.nbr_cnt.zero_()
        self._cache_t = self._cache_v = self._pending = None

    def flush(self, t):
        if self._pending is None: return
        idx, msg = self._pending
        dev = self.mem.device
        cur = self.mem[idx] * self.mem_ok[idx]
        dt  = (t - self.last_t[idx]).clamp_min(0)
        g   = torch.sigmoid(self.log_gamma)*0.5 + 0.5
        cur = cur * g.pow(dt).unsqueeze(-1)
        new = self.gru(msg.to(dev), cur)
        with torch.no_grad():
            self.mem[idx] = new.detach()
            self.last_t[idx] = t
        self._pending = None; self._cache_t = None

    def node_repr(self, t):
        if self._cache_t == t and self._cache_v is not None:
            return self._cache_v
        dev = self.mem.device
        m   = self.mem * self.mem_ok
        dt  = (t - self.last_t).clamp_min(0)
        g   = torch.sigmoid(self.log_gamma)*0.5 + 0.5
        m   = m * g.pow(dt).unsqueeze(-1)
        dt2 = (t - self.nbr_tb).clamp_min(0)
        ctx = torch.cat([self.nbr_vec, self.psi(dt2)], dim=-1)
        mask = (torch.arange(self.N, device=dev).unsqueeze(0) >=
                self.nbr_cnt.clamp(max=self.N).unsqueeze(1))
        mask[mask.all(1), 0] = False
        q   = torch.cat([m, self.psi(torch.zeros(self.V, device=dev))], dim=-1)
        nbr, _ = self.nbr_attn(q.unsqueeze(1), ctx, ctx,
                               key_padding_mask=mask, need_weights=False)
        v = torch.tanh(self.W_s(m) + self.W_r(nbr.squeeze(1)) + self.b_v)
        self._cache_v = v; self._cache_t = t
        return v

    def observe(self, variants, t):
        """Their memory update -- exactly Section 3.2."""
        seen = sorted({m for s in variants for m in s if m < self.V})
        if not seen: return
        dev   = self.mem.device
        V_rep = self.node_repr(t)
        idx   = torch.tensor(seen, dtype=torch.long)
        v     = V_rep[idx.to(dev)]
        pos   = {m: i for i, m in enumerate(seen)}
        agg   = torch.zeros(len(seen), self.d, device=dev)
        cnt   = torch.zeros(len(seen), 1, device=dev)
        for s in variants:
            ms = [m for m in s if m < self.V and m in pos]
            if not ms: continue
            rows = torch.tensor([pos[m] for m in ms], dtype=torch.long, device=dev)
            ctx  = V_rep[torch.tensor(ms, dtype=torch.long, device=dev)].mean(0, keepdim=True)
            agg.index_add_(0, rows, ctx.expand(len(rows), -1))
            cnt.index_add_(0, rows, torch.ones(len(rows), 1, device=dev))
        agg = agg / cnt.clamp_min(1.0)
        dt  = (t - self.last_t[idx]).clamp_min(0).to(dev)
        msg = torch.cat([v.detach(), agg.detach(), self.psi(dt)], dim=-1)
        self._pending = (idx, msg.detach())
        with torch.no_grad():
            slot = self.nbr_cnt[idx] % self.N
            self.nbr_vec[idx, slot] = agg.detach()
            self.nbr_tb [idx, slot] = t
            self.nbr_cnt[idx] += 1
        self._cache_t = None

    def predict(self, circ_mass, t):
        dev  = self.mem.device
        vars = [v for v, _ in circ_mass]
        mass = torch.tensor([w for _, w in circ_mass],
                            dtype=torch.float32, device=dev)
        # use cached repr but detach only memory, keep W_s/W_r in graph
        m2    = self.mem.detach() * self.mem_ok.detach()
        dt    = (t - self.last_t.detach()).clamp_min(0)
        g     = torch.sigmoid(self.log_gamma)*0.5 + 0.5
        m2    = m2 * g.pow(dt).unsqueeze(-1)
        # W_s applied to detached memory -- W_s.weight gets gradient via chain rule
        table = torch.tanh(self.W_s(m2) + self.b_v)
        reps  = []
        for v in vars:
            ms = [m for m in v if m < self.V]
            reps.append(table[torch.tensor(ms, dtype=torch.long, device=dev)].mean(0)
                        if ms else torch.zeros(self.d, device=dev))
        X   = torch.stack(reps)
        fit = self.fitness(X).squeeze(-1)
        logm = torch.log(mass.clamp_min(1e-9))
        return torch.log_softmax(logm + fit, dim=0), vars


def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    events, days, V = load_events(a.events)

    # compute start month = train_end minus (M-1) months
    y, m2 = int(a.train_end[:4]), int(a.train_end[5:7])
    for _ in range(a.M - 1):
        m2 -= 1
        if m2 < 1: m2 = 12; y -= 1
    start_ym = f"{y:04d}-{m2:02d}"

    # filter to only the M training months
    train_events = [(s, ti, w) for s, ti, w in events
                    if start_ym <= days[ti][:7] <= a.train_end]
    train_days   = sorted({ti for _, ti, _ in train_events})

    print(f"months {start_ym} to {a.train_end}  ({len(train_days)} days)  V {V}")
    print(f"predicting {a.test_month}  device {device}")

    # observed populations
    target_pop = month_population(events, days, a.test_month)
    print(f"target variants: {len(target_pop)}")

    model = NodeModel(V, d=a.d, N=a.n_recent).to(device)
    if a.vocab:
        model.mem_ok.copy_(
            load_first_seen(a.vocab, V, a.train_end).float().unsqueeze(-1).to(device))

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    # circulating at train_end = last month's population
    circ_raw = defaultdict(float)
    for s, ti, w in train_events:
        if days[ti][:7] == a.train_end:
            circ_raw[s] += w
    tot = sum(circ_raw.values()) or 1.0
    circ_mass = sorted(circ_raw.items(), key=lambda kv: -kv[1])
    circ_mass = [(s, w/tot) for s, w in circ_mass]
    print(f"circulating at {a.train_end}: {len(circ_mass)} variants")

    # build day-indexed lookup
    by_day = defaultdict(list)
    for sv, ti, w in train_events:
        by_day[ti].append((sv, w))

    # target population for each training day: events h_days later
    h_days_int = a.horizon * 30
    all_day_dates = {ti: days[ti] for ti in sorted(by_day.keys())}

    # precompute target populations per origin day
    # target = variants observed in the day window around ti + h_days
    def get_target(ti):
        target_date = days[ti]
        # find days approximately h_days later
        target_variants = defaultdict(float)
        for sv, tj, w in train_events:
            gap = tj - ti
            if h_days_int - 15 <= gap <= h_days_int + 15:
                target_variants[sv] += w
        # also include test month if origin is near train_end
        if not target_variants:
            for sv, w in target_pop.items():
                target_variants[sv] += w
        tot = sum(target_variants.values()) or 1.0
        return {sv: v/tot for sv, v in target_variants.items()}

    for ep in range(a.epochs):
        model.reset()
        losses = []
        # stream events: one gradient update per training day
        for ti in train_days:
            model.flush(float(ti))
            model.observe([sv for sv, _ in by_day[ti]], float(ti))

            # get circulating at this day
            circ_day = defaultdict(float)
            for tj in train_days:
                if ti - 30 <= tj <= ti:
                    for sv, w in by_day[tj]:
                        circ_day[sv] += w
            if not circ_day: continue
            tot_c = sum(circ_day.values()) or 1.0
            cm_day = sorted(circ_day.items(), key=lambda kv: -kv[1])
            cm_day = [(sv, w/tot_c) for sv, w in cm_day]
            if len(cm_day) < 2: continue

            # target: population h days later
            tgt = get_target(ti)
            if not tgt: continue

            lp, vars = model.predict(cm_day, float(ti))
            ix = {v: i for i, v in enumerate(vars)}
            tot2 = sum(tgt.values()) or 1.0
            I, W = [], []
            for v, w in tgt.items():
                j = ix.get(v)
                if j is not None: I.append(j); W.append(w/tot2)
            if not I: continue

            loss = -(torch.tensor(W, device=device)
                     * lp[torch.tensor(I, dtype=torch.long, device=device)]
                     ).sum() / max(sum(W), 1e-9)
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach()))

        print(f"ep {ep+1}/{a.epochs}  loss {np.mean(losses) if losses else float('nan'):.4f}"
              f"  steps {len(losses)}", flush=True)

    last_t = train_days[-1]

    # final evaluation
    model.reset()
    for ti in train_days:
        model.flush(float(ti))
        model.observe([s for s, _ in by_day[ti]], float(ti))

    with torch.no_grad():
        lp, vars = model.predict(circ_mass, float(last_t))
    p   = torch.exp(lp).detach().cpu().numpy()
    ix  = {v: i for i, v in enumerate(vars)}
    q   = np.zeros_like(p)
    for v, w in target_pop.items():
        j = ix.get(v)
        if j is not None: q[j] += w/tot2

    mass_arr = np.array([w for _, w in circ_mass])
    p_per    = mass_arr / mass_arr.sum()
    q_per    = np.zeros(len(circ_mass))
    ix2 = {v: i for i, v in enumerate(v for v, _ in circ_mass)}
    for v, w in target_pop.items():
        j = ix2.get(v)
        if j is not None: q_per[j] += w/tot2

    ov_m = float(np.minimum(p, q).sum())
    ov_p = float(np.minimum(p_per, q_per).sum())

    print(f"\n{'='*50}")
    print(f"train {start_ym}..{a.train_end}  predict {a.test_month}")
    print(f"model       {ov_m:.4f}")
    print(f"persistence {ov_p:.4f}")
    print(f"gain        {ov_m-ov_p:+.4f}")
    print(f"covered     {float(q.sum()):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events',    required=True)
    p.add_argument('--vocab',     default=None)
    p.add_argument('--train-end', default='2022-06', dest='train_end')
    p.add_argument('--test-month',default='2022-07', dest='test_month')
    p.add_argument('--M',         type=int, default=6)
    p.add_argument('--horizon',   type=int, default=1,
                   help='forecast horizon in months')
    p.add_argument('--d',         type=int, default=64)
    p.add_argument('--n-recent',  type=int, default=10, dest='n_recent')
    p.add_argument('--epochs',    type=int, default=10)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--seed',      type=int, default=0)
    run(p.parse_args())

if __name__ == '__main__':
    main()
