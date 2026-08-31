#!/usr/bin/env python3
"""
168_fitness_fast.py -- Fast fitness diagnostic.

6 months in → predict month 7 (or T+h).
One horizon per run. No assembly. No generation. No budget.
Fitness head only: which existing variants grow vs decline?

Zero-initialized → starts at persistence. Any gain is real signal.

Usage:
  python scripts/168_fitness_fast.py \
    --events data/processed/events_v3.tsv \
    --vocab data/processed/vocab_v3.tsv \
    --horizon 1 --M 6 --seeds 0 1 2 3 4
"""
import argparse, os, random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# ======================================================================
# DATA
# ======================================================================

def load_monthly(path, vocab_path=None, verbose=True):
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
            if s: rows.append((date[:7], s, cnt))
    rows.sort(key=lambda r: r[0])

    # aggregate by month
    by_month = defaultdict(lambda: defaultdict(float))
    for ym, s, cnt in rows:
        by_month[ym][s] += cnt
    months = sorted(by_month.keys())

    # normalise
    var_mass = {}
    mut_freq = {}
    for ym in months:
        tot = sum(by_month[ym].values()) or 1.0
        var_mass[ym] = {s: v/tot for s, v in by_month[ym].items()}
        mf = defaultdict(float)
        for s, w in by_month[ym].items():
            for m in s: mf[m] += w/tot
        mut_freq[ym] = dict(mf)

    V = 1 + max(m for ym in months for s in by_month[ym] for m in s)
    if verbose:
        print(f"months {len(months)}  ({months[0]}..{months[-1]})  V {V}")
    return var_mass, mut_freq, months, V


def load_first_seen(vocab_path, V, cutoff):
    seen = torch.ones(V, dtype=torch.bool)
    if not (vocab_path and os.path.exists(vocab_path) and cutoff):
        return seen
    n_late = 0
    with open(vocab_path) as f:
        for line in f:
            p = line.rstrip().split()
            if len(p) >= 3 and p[0].isdigit():
                i = int(p[0])
                if i < V and p[2] > cutoff:
                    seen[i] = False; n_late += 1
    print(f"  {n_late} mutations suppressed after {cutoff}")
    return seen


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


class FitnessModel(nn.Module):
    """
    Node GRU memory + neighbourhood attention + horizon-conditioned fitness.

    For each circulating variant B, represents it as mean of member node
    representations, then predicts a growth correction conditioned on h.

    Starts at persistence (zero-initialized fitness head).
    """
    def __init__(self, V, d=64, N=10, decay=True):
        super().__init__()
        self.V, self.d, self.N = V, d, N
        self.psi = FourierTime(d)

        # node memory
        self.gru = nn.GRUCell(3*d, d)
        self.decay = decay
        self.log_gamma = nn.Parameter(torch.tensor(-3.0))
        self.nbr_attn = nn.MultiheadAttention(2*d, 2, batch_first=True)
        self.W_r = nn.Linear(2*d, d)
        self.W_s = nn.Linear(d, d)
        self.b_v = nn.Parameter(torch.zeros(d))

        # fitness: variant repr (d) + horizon encoding (d) → scalar
        self.fitness = nn.Sequential(
            nn.Linear(2*d, 2*d), nn.Tanh(), nn.Linear(2*d, 1))
        nn.init.zeros_(self.fitness[-1].weight)
        nn.init.zeros_(self.fitness[-1].bias)

        # buffers
        self.register_buffer('mem',      torch.zeros(V, d))
        self.register_buffer('last_t',   torch.zeros(V))
        self.register_buffer('nbr_vec',  torch.zeros(V, N, d))
        self.register_buffer('nbr_t',    torch.zeros(V, N))
        self.register_buffer('nbr_cnt',  torch.zeros(V, dtype=torch.long))
        self.register_buffer('mem_ok',   torch.ones(V, 1))
        self._cache_t = None
        self._cache_v = None
        self._pending = None

    def reset(self):
        self.mem.zero_(); self.last_t.zero_()
        self.nbr_vec.zero_(); self.nbr_t.zero_(); self.nbr_cnt.zero_()
        self._cache_t = None; self._cache_v = None; self._pending = None

    def flush(self, t_now):
        if self._pending is None: return
        idx, msg = self._pending
        dev = self.mem.device
        cur = self.mem[idx]
        if self.decay:
            dt = (t_now - self.last_t[idx]).clamp_min(0)
            g = torch.sigmoid(self.log_gamma) * 0.5 + 0.5
            cur = cur * g.pow(dt).unsqueeze(-1)
        cur = cur * self.mem_ok[idx]
        new = self.gru(msg.to(dev), cur)
        with torch.no_grad():
            self.mem[idx] = new.detach()
            self.last_t[idx] = t_now
        self._pending = None; self._cache_t = None

    def node_repr(self, t_now):
        if self._cache_t == t_now and self._cache_v is not None:
            return self._cache_v
        dev = self.mem.device
        idx = torch.arange(self.V, device=dev)
        m = self.mem * self.mem_ok
        if self.decay:
            dt = (t_now - self.last_t).clamp_min(0)
            g = torch.sigmoid(self.log_gamma) * 0.5 + 0.5
            m = m * g.pow(dt).unsqueeze(-1)
        dt2 = (t_now - self.nbr_t).clamp_min(0)
        ctx = torch.cat([self.nbr_vec, self.psi(dt2)], dim=-1)
        mask = torch.arange(self.N, device=dev).unsqueeze(0) >= \
               self.nbr_cnt.clamp(max=self.N).unsqueeze(1)
        mask[mask.all(1), 0] = False
        q = torch.cat([m, self.psi(torch.zeros(self.V, device=dev))], dim=-1)
        nbr, _ = self.nbr_attn(q.unsqueeze(1), ctx, ctx,
                               key_padding_mask=mask, need_weights=False)
        v = torch.tanh(self.W_s(m) + self.W_r(nbr.squeeze(1)) + self.b_v)
        self._cache_v = v; self._cache_t = t_now
        return v

    def observe(self, variants, t_now):
        seen = sorted({m for s in variants for m in s if m < self.V})
        if not seen: return
        dev = self.mem.device
        V_rep = self.node_repr(t_now)
        idx = torch.tensor(seen, dtype=torch.long)
        v = V_rep[idx.to(dev)]
        pos = {m: i for i, m in enumerate(seen)}
        agg = torch.zeros(len(seen), self.d, device=dev)
        cnt = torch.zeros(len(seen), 1, device=dev)
        for s in variants:
            ms = [m for m in s if m < self.V and m in pos]
            if not ms: continue
            rows = torch.tensor([pos[m] for m in ms], dtype=torch.long, device=dev)
            ctx = V_rep[torch.tensor(ms, dtype=torch.long, device=dev)].mean(0, keepdim=True)
            agg.index_add_(0, rows, ctx.expand(len(rows), -1))
            cnt.index_add_(0, rows, torch.ones(len(rows), 1, device=dev))
        agg = agg / cnt.clamp_min(1.0)
        dt = (t_now - self.last_t[idx]).clamp_min(0).to(dev)
        msg = torch.cat([v.detach(), agg.detach(), self.psi(dt)], dim=-1)
        self._pending = (idx, msg.detach())
        with torch.no_grad():
            slot = self.nbr_cnt[idx] % self.N
            self.nbr_vec[idx, slot] = agg.detach()
            self.nbr_t[idx, slot] = t_now
            self.nbr_cnt[idx] += 1
        self._cache_t = None

    def predict(self, circ_mass, t_now, h_days):
        """Predicted distribution over circulating variants at T+h."""
        dev = self.mem.device
        circ = [v for v, _ in circ_mass]
        mass = torch.tensor([w for _, w in circ_mass], dtype=torch.float32, device=dev)
        table = self.node_repr(t_now)
        reps = []
        for v in circ:
            ms = [m for m in v if m < self.V]
            reps.append(table[torch.tensor(ms, device=dev, dtype=torch.long)].mean(0)
                        if ms else torch.zeros(self.d, device=dev))
        X = torch.stack(reps)
        psi_h = self.psi(torch.tensor([float(h_days)], device=dev)).expand(len(circ), -1)
        feat = torch.cat([X, psi_h], dim=-1)
        fit  = self.fitness(feat).squeeze(-1)
        logm = torch.log(mass.clamp_min(1e-9))
        return torch.log_softmax(logm + fit, dim=0), circ


# ======================================================================
# TRAIN / EVAL
# ======================================================================

def make_windows(months, M, h, train_frac):
    """Sliding windows: M months in, predict month at +h."""
    n = len(months)
    n_train = int(n * train_frac)
    train_w, test_w = [], []
    for i in range(M, n - h):
        window  = months[i-M:i]
        target  = months[i+h-1]
        if i < n_train:
            train_w.append((window, target))
        else:
            test_w.append((window, target))
    return train_w, test_w


def overlap(p_np, q_np):
    mn = np.minimum(p_np, q_np).sum()
    mx = np.maximum(p_np, q_np).sum()
    return float(mn), float(mn/mx) if mx > 0 else float('nan')


def run_seed(a, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    var_mass, mut_freq, months, V = load_monthly(a.events, a.vocab)

    n_train = int(len(months) * a.train_frac)
    cutoff  = months[n_train - 1]
    train_w, test_w = make_windows(months, a.M, a.horizon, a.train_frac)

    model = FitnessModel(V, d=a.d, N=a.n_recent, decay=True).to(device)
    if a.vocab:
        model.mem_ok.copy_(load_first_seen(a.vocab, V, cutoff)
                           .float().unsqueeze(-1).to(device))

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    print(f"\nseed {seed}  |  train windows {len(train_w)}  test {len(test_w)}"
          f"  horizon h={a.horizon}m  device {device}")

    model.reset()
    for ep in range(a.epochs):
        losses = []
        # replay ALL months in order; train on windows as they appear
        all_months = sorted(var_mass.keys())
        for mi, ym in enumerate(all_months):
            model.flush(float(mi))
            model.observe(list(var_mass[ym].keys()), float(mi))

            # is this a training origin?
            for window, target in train_w:
                if window[-1] != ym: continue
                t_idx = all_months.index(ym)
                cm = sorted(var_mass[ym].items(), key=lambda kv: -kv[1])[:a.pop_support]
                obs = var_mass.get(target, {})
                if len(cm) < 2 or not obs: continue

                lp, circ = model.predict(cm, float(t_idx),
                                         float(a.horizon * 30))
                ix  = {v: i for i, v in enumerate(circ)}
                tot = sum(obs.values()) or 1.0
                I, W = [], []
                for v, w in obs.items():
                    j = ix.get(v)
                    if j is not None:
                        I.append(j); W.append(w/tot)
                if not I: continue
                loss = -(torch.tensor(W, device=device)
                         * lp[torch.tensor(I, dtype=torch.long, device=device)]
                         ).sum() / max(sum(W), 1e-9)
                if torch.isfinite(loss):
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step(); losses.append(float(loss.detach()))

        print(f"  epoch {ep+1}/{a.epochs}  loss {np.mean(losses):.4f}", flush=True)

    # evaluation
    model.reset()
    all_months = sorted(var_mass.keys())
    for mi, ym in enumerate(all_months):
        model.flush(float(mi))
        model.observe(list(var_mass[ym].keys()), float(mi))

    mdl_ov, per_ov = [], []
    for window, target in test_w:
        ym = window[-1]
        t_idx = all_months.index(ym)
        cm  = sorted(var_mass[ym].items(), key=lambda kv: -kv[1])[:a.pop_support]
        obs = var_mass.get(target, {})
        if len(cm) < 2 or not obs: continue

        with torch.no_grad():
            lp, circ = model.predict(cm, float(t_idx), float(a.horizon * 30))
        p = torch.exp(lp).detach().cpu().numpy()
        ix  = {v: i for i, v in enumerate(circ)}
        tot = sum(obs.values()) or 1.0
        q   = np.zeros_like(p)
        for v, w in obs.items():
            j = ix.get(v)
            if j is not None: q[j] += w/tot

        # persistence
        mass_arr = np.array([w for _, w in cm])
        p_per    = mass_arr / mass_arr.sum()
        q_per    = np.zeros_like(p_per)
        ix2 = {v: i for i, v in enumerate(v for v, _ in cm)}
        for v, w in obs.items():
            j = ix2.get(v)
            if j is not None: q_per[j] += w/tot

        ov_m, _ = overlap(p, q)
        ov_p, _ = overlap(p_per, q_per)
        mdl_ov.append(ov_m); per_ov.append(ov_p)

    print(f"  h={a.horizon}m  model {np.mean(mdl_ov):.4f}"
          f"  persistence {np.mean(per_ov):.4f}"
          f"  gain {np.mean(mdl_ov)-np.mean(per_ov):+.4f}"
          f"  (n={len(mdl_ov)})")
    return np.mean(mdl_ov), np.mean(per_ov)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events',      required=True)
    p.add_argument('--vocab',       default=None)
    p.add_argument('--horizon',     type=int, default=1)
    p.add_argument('--M',           type=int, default=6,
                   help='months of history per window')
    p.add_argument('--d',           type=int, default=64)
    p.add_argument('--n-recent',    type=int, default=10, dest='n_recent')
    p.add_argument('--pop-support', type=int, default=200, dest='pop_support')
    p.add_argument('--epochs',      type=int, default=5)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--train-frac',  type=float, default=0.7, dest='train_frac')
    p.add_argument('--seeds',       type=int, nargs='+', default=[0,1,2])
    a = p.parse_args()

    all_mdl, all_per = [], []
    for seed in a.seeds:
        m, per = run_seed(a, seed)
        all_mdl.append(m); all_per.append(per)

    print(f"\n{'='*50}")
    print(f"h={a.horizon}m  seeds={a.seeds}")
    print(f"model       {np.mean(all_mdl):.4f} ± {np.std(all_mdl):.4f}")
    print(f"persistence {np.mean(all_per):.4f} ± {np.std(all_per):.4f}")
    print(f"gain        {np.mean(all_mdl)-np.mean(all_per):+.4f}")

if __name__ == '__main__':
    main()
