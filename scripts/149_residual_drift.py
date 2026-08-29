#!/usr/bin/env python
"""
149_residual_drift.py  --  Residual + drift set prediction.

Key design:
  drift    = mean(mu_{t+1} - mu_t) over training pairs -- analytic, not learned
  centered = s - mu_t removes absolute mutation level

  Group 1 -- Seen (constellations present at both t and t+h):
    target  = (s_{t+h} - mu_{t+h}) - (s_t - mu_t) - drift*h  <- centered residual
    model   = small MLP per constellation: phi_i + ctx -> delta_i
    output  = s_t + drift*h + delta_i
    at init = delta_i ~ 0 -> output ~ s_t + drift = persistence + drift

  Group 2 -- Novel (absent at t, appear at t+h):
    N_novel learned queries cross-attend to phi_i
    MADE per slot -> new binary vector
    at init ~ population mean

Loss: Chamfer(seen_pred + novel_pred, actual top-500 at t+h)
      Novel constellations upweighted 10x

Evaluation: exact recall@k, novel recall@k vs persistence

Run:
  python scripts/149_residual_drift.py \
    --data-dir data/processed/full_data_graphs_withdel \
    --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
    --months 2020-06:2023-06 --test-start 2022-06 \
    --M 6 --N-novel 100 --J 200 \
    --d 64 --heads 4 --layers 2 --d-lstm 64 --d-made 32 \
    --epochs 500 --lr 1e-3 --top 500 --upweight 10.0 --top-k 20
"""

import argparse, importlib.util, sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ENGINE = "scripts/110_hierarchical_birthdeath_v2_fixed.py"
EPS    = 1e-6

# ---------------------------------------------------------------- engine ---
def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m    = importlib.util.module_from_spec(spec)
    sys.modules["engine"] = m; spec.loader.exec_module(m); return m

# ------------------------------------------------------------------ data ---
def recs_to_matrix(recs, V, top=None):
    if top: recs = sorted(recs, key=lambda x: -x[1])[:top]
    S = np.zeros((len(recs), V), dtype=np.float32)
    for i, (s, _) in enumerate(recs):
        for v in s:
            if v < V: S[i, v] = 1.0
    w = np.array([float(c) for _, c in recs], dtype=np.float32)
    w /= w.sum()
    return S, w

def load_all(E, data_dir, months, V, top):
    print("loading...", flush=True)
    out = []
    for ym in months:
        recs = E.load_month(data_dir, ym)
        if not recs: out.append(None); continue
        S, w  = recs_to_matrix(recs, V, top)
        sets  = {frozenset(s) for s, _ in recs}
        mu    = (w[:, None] * S).sum(0)
        out.append({"S": S, "w": w, "sets": sets, "mu": mu, "ym": ym})
    print(f"  loaded {sum(1 for x in out if x)}/{len(months)} months")
    return out

def top_j_positions(clouds, train_idx, J):
    freq = freq2 = None; total = 0.0
    for i in train_idx:
        c  = clouds[i]
        f  = (c["w"][:, None] * c["S"]).sum(0)
        f2 = (c["w"][:, None] * c["S"]**2).sum(0)
        freq  = f  if freq  is None else freq  + f
        freq2 = f2 if freq2 is None else freq2 + f2
        total += 1.0
    mu  = freq / total
    var = freq2 / total - mu**2
    top = np.sort(np.argsort(var)[::-1][:J])
    print(f"  top-{J} positions, mean var={var[top].mean():.4f}")
    return top

# --------------------------------------------------------- Jaccard --------
def jaccard_matrix(S_bin):
    dot = S_bin @ S_bin.T; sz = S_bin.sum(1)
    return torch.nan_to_num(
        dot / (sz.unsqueeze(1) + sz.unsqueeze(0) - dot + EPS), nan=0.0)

# ----------------------------------------------------------------- MADE ---
class MADE(nn.Module):
    def __init__(self, J, d_cond, d_hidden=32, seed=0):
        super().__init__()
        self.J = J
        rng     = np.random.default_rng(seed)
        degrees = rng.integers(1, max(J, 2), size=d_hidden)
        v_range = torch.arange(J).float()
        d_range = torch.tensor(degrees).float()
        self.register_buffer('mask_W1',
            (d_range.unsqueeze(1) >= v_range.unsqueeze(0)).float())
        self.register_buffer('mask_W2',
            (v_range.unsqueeze(1) > d_range.unsqueeze(0)).float())
        self.W1        = nn.Parameter(torch.randn(d_hidden, J) * 0.01)
        self.b1        = nn.Parameter(torch.zeros(d_hidden))
        self.W2        = nn.Parameter(torch.randn(J, d_hidden) * 0.01)
        self.b2        = nn.Parameter(torch.zeros(J))
        self.cond_proj = nn.Linear(d_cond, d_hidden, bias=False)

    def soft_output_batch(self, C):
        """C: (N, d). Returns (N, J) soft binary in [0,1]. One forward pass."""
        N = C.shape[0]
        s = torch.zeros(N, self.J, device=C.device)
        h = torch.relu(s @ (self.W1*self.mask_W1).T
                       + self.b1
                       + C @ self.cond_proj.weight.T)
        return torch.sigmoid(h @ (self.W2*self.mask_W2).T + self.b2)

# --------------------------------------------------------------- model ----

class SetEncoder(nn.Module):
    """Returns phi_i (N, d) per-constellation AND u_t (d) pooled."""
    def __init__(self, V, d, heads, n_layers):
        super().__init__()
        self.proj = nn.Linear(V, d)
        self.lam  = nn.Parameter(torch.zeros(1))
        if n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=d*2,
                dropout=0.1, batch_first=True)
            self.attn = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.attn = None
        self.pool_out = nn.Sequential(nn.Linear(d, d), nn.Tanh())

    def forward(self, S_centered, S_bin, w):
        x = self.proj(S_centered)
        if self.attn is not None:
            J = jaccard_matrix(S_bin)
            b = torch.clamp(self.lam * J, -10., 10.)
            xa = self.attn(x.unsqueeze(0), mask=b).squeeze(0)
            x  = xa if torch.isfinite(xa).all() else x
        phi_i = x
        u_t   = (w.unsqueeze(-1) * phi_i).sum(0)
        return phi_i, self.pool_out(torch.nan_to_num(u_t, nan=0.0))


class TemporalEncoder(nn.Module):
    def __init__(self, d, d_lstm):
        super().__init__()
        self.lstm   = nn.LSTM(d, d_lstm, batch_first=True)
        self.attn_w = nn.Linear(d_lstm, 1)

    def forward(self, us):
        hs, (h_n, _) = self.lstm(us.unsqueeze(0))
        hs      = hs.squeeze(0)
        weights = torch.softmax(self.attn_w(hs).squeeze(-1), dim=0)
        context = (weights.unsqueeze(-1) * hs).sum(0)
        return h_n.squeeze(0).squeeze(0), context


class ResidualDriftDecoder(nn.Module):
    """
    Group 1 -- Seen residual:
      For each input constellation phi_i + ctx -> small delta_i in [0,1]^V
      output_i = clamp(s_i + drift*h + delta_i, 0, 1)
      At init: delta_i ~ 0 -> output ~ s_i + drift = persistence + drift

    Group 2 -- Novel generation:
      N_novel learned queries cross-attend to phi_i
      MADE -> new binary constellations
      At init: near population mean
    """
    def __init__(self, d, d_lstm, N_novel, V, J, d_made, top_j_idx):
        super().__init__()
        self.N_novel = N_novel
        self.V       = V
        self.J       = J
        V_rest       = V - J
        self.register_buffer('top_j_idx',
                             torch.tensor(top_j_idx, dtype=torch.long))

        ctx_d = d_lstm * 2
        self.ctx_proj = nn.Linear(ctx_d, d)

        # --- Group 1: seen residual ---
        # phi_i + ctx -> delta_i in V dimensions
        # small residual: how much each position changes beyond drift
        # ZERO INIT: delta ~ 0 at init -> persistence + drift baseline
        self.residual_mlp = nn.Sequential(
            nn.Linear(d + d, d*2), nn.Tanh(),
            nn.Linear(d*2, V))
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

        # weight correction per seen slot
        self.seen_w_mlp = nn.Sequential(
            nn.Linear(d + d, d), nn.Tanh(),
            nn.Linear(d, 1))
        nn.init.zeros_(self.seen_w_mlp[-1].weight)
        nn.init.zeros_(self.seen_w_mlp[-1].bias)

        # --- Group 2: novel generation ---
        self.novel_queries = nn.Parameter(
            torch.randn(N_novel, d) * 0.01)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=max(1, d//16),
            batch_first=True)
        self.made = MADE(J, d_cond=d, d_hidden=d_made)
        self.fact_net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, V_rest))
        nn.init.zeros_(self.fact_net[-1].weight)
        nn.init.zeros_(self.fact_net[-1].bias)
        self.novel_w_mlp = nn.Linear(d, 1)
        nn.init.zeros_(self.novel_w_mlp.weight)
        nn.init.zeros_(self.novel_w_mlp.bias)

    def _rest_idx(self, device):
        all_idx = torch.arange(self.V, device=device)
        mask    = torch.ones(self.V, dtype=torch.bool, device=device)
        mask[self.top_j_idx] = False
        return all_idx[mask]

    def forward(self, phi_i, w_i, h_t, context,
                S_input, drift_h):
        """
        phi_i:   (N_in, d) input constellation embeddings
        w_i:     (N_in,) input frequencies
        h_t:     (d_lstm,) temporal hidden
        context: (d_lstm,) temporal context
        S_input: (N_in, V) input binary constellations (NOT centered)
        drift_h: (V,) drift * h -- population mean shift

        Returns:
          S_seen:   (N_in, V) soft predicted seen constellations
          w_seen:   (N_in,) predicted seen weights
          S_novel:  (N_novel, V) soft predicted novel constellations
          w_novel:  (N_novel,) predicted novel weights
        """
        N_in     = phi_i.shape[0]
        ctx      = torch.tanh(self.ctx_proj(
            torch.cat([h_t, context])))                    # (d,)
        ctx_exp  = ctx.unsqueeze(0).expand(N_in, -1)      # (N_in, d)
        rest_idx = self._rest_idx(phi_i.device)

        # ---- Group 1: seen residual ----
        inp      = torch.cat([phi_i, ctx_exp], dim=1)     # (N_in, 2d)
        delta    = torch.tanh(self.residual_mlp(inp))     # (N_in, V) in [-1,1]
        # output = input + drift + small residual, clamped to [0,1]
        S_seen   = (S_input + drift_h.unsqueeze(0)
                    + delta * 0.1).clamp(0, 1)            # (N_in, V)
        # weight correction
        w_delta  = self.seen_w_mlp(inp).squeeze(-1)       # (N_in,)
        w_seen   = torch.softmax(
            torch.log(w_i + EPS) + w_delta, dim=0)        # (N_in,)

        # ---- Group 2: novel generation ----
        phi_w    = phi_i * w_i.unsqueeze(-1)              # (N_in, d)
        Q        = (self.novel_queries
                    + ctx.unsqueeze(0))                    # (N_novel, d)
        e_novel, _ = self.cross_attn(
            Q.unsqueeze(0),
            phi_w.unsqueeze(0),
            phi_i.unsqueeze(0))
        e_novel  = e_novel.squeeze(0)                     # (N_novel, d)

        # novel constellation: MADE for J, factorized for rest
        S_novel  = torch.zeros(
            self.N_novel, self.V, device=phi_i.device)
        S_novel[:, rest_idx]       = torch.sigmoid(
            self.fact_net(e_novel))
        S_novel[:, self.top_j_idx] = self.made.soft_output_batch(
            e_novel)                                       # (N_novel, J)

        # add drift to novel too
        S_novel  = (S_novel + drift_h.unsqueeze(0)).clamp(0, 1)

        # novel weights
        w_novel  = torch.softmax(
            self.novel_w_mlp(e_novel).squeeze(-1), dim=0) # (N_novel,)

        return S_seen, w_seen, S_novel, w_novel


class ResidualDriftModel(nn.Module):
    def __init__(self, V, d, heads, n_layers, d_lstm,
                 N_novel, J, d_made, top_j_idx, M):
        super().__init__()
        self.V = V; self.M = M
        self.enc  = SetEncoder(V, d, heads, n_layers)
        self.temp = TemporalEncoder(d, d_lstm)
        self.dec  = ResidualDriftDecoder(
            d, d_lstm, N_novel, V, J, d_made, top_j_idx)
        self.register_buffer('mean_drift', torch.zeros(V))

    def set_mean_drift(self, drift):
        self.mean_drift.copy_(torch.tensor(drift, dtype=torch.float32))

    def encode_month(self, S_bin, w, mu):
        return self.enc(S_bin - mu, S_bin, w)

    def _encode_window(self, window):
        phis = []; us = []
        for data in window:
            phi, u = self.encode_month(
                data["S_bin"], data["w"], data["mu"])
            phis.append(phi); us.append(u)
        h_t, ctx = self.temp(torch.stack(us))
        return phis[-1], window[-1]["w"], h_t, ctx

    def forward(self, window, S_input, w_input, h=1):
        phi_i, w_i, h_t, ctx = self._encode_window(window)
        drift_h = self.mean_drift * h                     # (V,) drift for h steps

        S_seen, w_seen, S_novel, w_novel = self.dec(
            phi_i, w_i, h_t, ctx, S_input, drift_h)

        # combine seen + novel
        S_all = torch.cat([S_seen, S_novel], dim=0)      # (N_in+N_novel, V)
        w_all = torch.softmax(
            torch.cat([torch.log(w_seen + EPS),
                       torch.log(w_novel + EPS)]), dim=0) # renormalize jointly

        return S_all, w_all

    def predict_binary(self, window, S_input, w_input, h=1):
        """At test time: threshold to binary."""
        S_all, w_all = self.forward(window, S_input, w_input, h)
        return (S_all > 0.5).float(), w_all

# ------------------------------------------------------------ loss --------

def hamming_cost(A, B):
    return (A.sum(1, keepdim=True) + B.sum(1, keepdim=True).T
            - 2 * A @ B.T) / A.shape[1]

def chamfer_loss(S_pred, w_pred, S_true, w_true,
                 S_true_cpu, present_sets, upweight):
    w_adj = w_true.clone()
    for i, row in enumerate(S_true_cpu):
        fs = frozenset(torch.nonzero(row).squeeze(-1).tolist())
        if fs not in present_sets:
            w_adj[i] = w_adj[i] * upweight
    w_adj = w_adj / w_adj.sum()
    C   = hamming_cost(S_pred, S_true)
    fwd = (w_pred * C.min(1).values).sum()
    bwd = (w_adj  * C.min(0).values).sum()
    return fwd + bwd

# ------------------------------------------------ evaluation metrics ----

def recall_at_k(S_pred, w_pred, S_true, w_true,
                present_sets, top_k=20):
    pred_sets  = [frozenset(np.flatnonzero(r).tolist()) for r in S_pred]
    true_sets  = [frozenset(np.flatnonzero(r).tolist()) for r in S_true]
    top_k_pred = {pred_sets[i]
                  for i in np.argsort(w_pred)[::-1][:top_k]}
    top_k_true = {true_sets[i]
                  for i in np.argsort(w_true)[::-1][:top_k]}
    recall     = len(top_k_pred & top_k_true) / max(len(top_k_true), 1)
    novel_true = {s for s in top_k_true if s not in present_sets}
    novel_pred = {s for s in top_k_pred if s not in present_sets}
    nov_recall = (len(novel_pred & novel_true) / max(len(novel_true), 1)
                  if novel_true else float('nan'))
    return recall, nov_recall

def persistence_recall(S_t, w_t, S_true, w_true,
                       present_sets, top_k=20):
    true_sets  = [frozenset(np.flatnonzero(r).tolist()) for r in S_true]
    top_k_t    = {frozenset(np.flatnonzero(S_t[i]).tolist())
                  for i in np.argsort(w_t)[::-1][:top_k]}
    top_k_true = {true_sets[i]
                  for i in np.argsort(w_true)[::-1][:top_k]}
    recall     = len(top_k_t & top_k_true) / max(len(top_k_true), 1)
    novel_true = {s for s in top_k_true if s not in present_sets}
    nov_recall = 0.0 if novel_true else float('nan')
    return recall, nov_recall

# --------------------------------------------------------- windows -------
def make_windows(clouds, M, horizons, start, end):
    wins = []
    for i in range(start, end):
        for h in horizons:
            t      = i + h
            inputs = list(range(i-M+1, i+1))
            if inputs[0] < 0: continue
            if any(clouds[j] is None for j in inputs): continue
            if t >= len(clouds) or clouds[t] is None: continue
            wins.append((inputs, t, h))
    return wins

# --------------------------------------------------------------- train ---
def train(model, clouds, train_wins, a, device):
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-3)
    for epoch in range(a.epochs):
        model.train(); total = 0.0
        for inputs, target, h in [train_wins[i] for i in
                                   np.random.permutation(len(train_wins))]:
            ct_t = clouds[inputs[-1]]
            ct_h = clouds[target]
            window = [{"S_bin": torch.tensor(clouds[i]["S"],
                       dtype=torch.float32).to(device),
                       "w":    torch.tensor(clouds[i]["w"],
                       dtype=torch.float32).to(device),
                       "mu":   torch.tensor(clouds[i]["mu"],
                       dtype=torch.float32).to(device)}
                      for i in inputs]
            S_inp  = torch.tensor(ct_t["S"],
                                  dtype=torch.float32).to(device)
            w_inp  = torch.tensor(ct_t["w"],
                                  dtype=torch.float32).to(device)
            S_true = torch.tensor(ct_h["S"],
                                  dtype=torch.float32).to(device)
            w_true = torch.tensor(ct_h["w"],
                                  dtype=torch.float32).to(device)
            S_true_cpu = S_true.cpu().round()

            S_pred, w_pred = model(window, S_inp, w_inp, h)
            loss = chamfer_loss(
                S_pred, w_pred, S_true, w_true,
                S_true_cpu, ct_t["sets"], a.upweight)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        if (epoch+1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}  "
                  f"chamfer/win {total/len(train_wins):.5f}  "
                  f"lam={model.enc.lam.item():.3f}")

# ------------------------------------------------------------- evaluate --
def evaluate(model, clouds, test_wins, months,
             eval_h, top_k, device):
    model.eval()
    print(f"\n{'Input window':>22} | {'Test':>8} | {'h':>3} | "
          f"{'rec@k':>6} {'nov_rec':>8} | "
          f"{'per_rec':>7} {'per_nov':>8} | "
          f"{'gain':>6} {'nov_gain':>8}")
    print("-" * 95)
    by_h = {h: [] for h in eval_h}

    with torch.no_grad():
        for inputs, target, h in test_wins:
            if h not in eval_h: continue
            ct_t = clouds[inputs[-1]]
            ct_h = clouds[target]
            window = [{"S_bin": torch.tensor(clouds[i]["S"],
                       dtype=torch.float32).to(device),
                       "w":    torch.tensor(clouds[i]["w"],
                       dtype=torch.float32).to(device),
                       "mu":   torch.tensor(clouds[i]["mu"],
                       dtype=torch.float32).to(device)}
                      for i in inputs]
            S_inp = torch.tensor(ct_t["S"],
                                 dtype=torch.float32).to(device)
            w_inp = torch.tensor(ct_t["w"],
                                 dtype=torch.float32).to(device)

            S_out, w_out = model.predict_binary(
                window, S_inp, w_inp, h)
            S_np = S_out.cpu().numpy()
            w_np = w_out.cpu().numpy()

            rec, nov = recall_at_k(
                S_np, w_np, ct_h["S"], ct_h["w"],
                ct_t["sets"], top_k=top_k)
            per_rec, per_nov = persistence_recall(
                ct_t["S"], ct_t["w"],
                ct_h["S"], ct_h["w"],
                ct_t["sets"], top_k=top_k)

            gain    = rec - per_rec
            nov_gain= ((nov-(per_nov if np.isfinite(per_nov) else 0.0))
                       if np.isfinite(nov) else float('nan'))

            iw   = f"{months[inputs[0]]}..{months[inputs[-1]]}"
            nstr = f"{nov:.3f}"  if np.isfinite(nov)     else "  nan"
            pnstr= f"{per_nov:.3f}" if np.isfinite(per_nov) else "  nan"
            ngstr= f"{nov_gain:+.3f}" if np.isfinite(nov_gain) else "   nan"
            print(f"{iw:>22} | {months[target]:>8} | {h:>3} | "
                  f"{rec:6.3f} {nstr:>8} | "
                  f"{per_rec:7.3f} {pnstr:>8} | "
                  f"{gain:+6.3f} {ngstr:>8}")
            by_h[h].append((rec,
                            nov  if np.isfinite(nov)     else np.nan,
                            per_rec,
                            per_nov if np.isfinite(per_nov) else np.nan))

    print(f"\n{'='*70}")
    print(f"=== Summary (top-{top_k}) ===")
    print(f"{'h':>4} {'rec@k':>7} {'nov_rec':>9} | "
          f"{'per_rec':>8} | "
          f"{'rec_gain':>9} {'nov_gain':>9}")
    for h in eval_h:
        if not by_h[h]: continue
        R    = np.array(by_h[h])
        rec  = np.nanmean(R[:,0]); nov  = np.nanmean(R[:,1])
        prec = np.nanmean(R[:,2])
        print(f"  h={h:2d} {rec:7.3f} {nov:9.3f} | "
              f"{prec:8.3f} | "
              f"{rec-prec:+9.3f} {nov:+9.3f}")

# ----------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True)
    p.add_argument("--vocab",       required=True)
    p.add_argument("--months",      required=True)
    p.add_argument("--test-start",  required=True)
    p.add_argument("--engine",      default=ENGINE)
    p.add_argument("--M",       type=int,   default=6)
    p.add_argument("--N-novel", type=int,   default=100, dest="N_novel")
    p.add_argument("--J",       type=int,   default=200)
    p.add_argument("--d",       type=int,   default=64)
    p.add_argument("--heads",   type=int,   default=4)
    p.add_argument("--layers",  type=int,   default=2)
    p.add_argument("--d-lstm",  type=int,   default=64, dest="d_lstm")
    p.add_argument("--d-made",  type=int,   default=32, dest="d_made")
    p.add_argument("--epochs",  type=int,   default=500)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--top",     type=int,   default=500)
    p.add_argument("--upweight",type=float, default=10.0)
    p.add_argument("--top-k",   type=int,   default=20, dest="top_k")
    p.add_argument("--horizons",type=int, nargs="+", default=[1,2,3,6])
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    E = load_engine(a.engine)
    idx2name, V = E.load_names(a.vocab)
    print(f"V={V}")
    months = E.months_in_range(a.months)
    print(f"{len(months)} months: {months[0]} .. {months[-1]}")
    if a.test_start not in months:
        print("not found"); sys.exit(1)
    ts = months.index(a.test_start)

    clouds = load_all(E, a.data_dir, months, V, a.top)
    train_idx  = list(range(ts))
    train_wins = make_windows(clouds, a.M, a.horizons, a.M-1, ts)
    test_wins  = make_windows(clouds, a.M, a.horizons, ts, len(months))
    print(f"train: {len(train_wins)}  test: {len(test_wins)}")

    # analytic drift -- mean per-position change per month, fit on training only
    drifts = [clouds[i+1]["mu"] - clouds[i]["mu"]
               for i in range(ts-1) if clouds[i] and clouds[i+1]]
    mean_drift = np.stack(drifts).mean(0) if drifts else np.zeros(V)
    print(f"drift magnitude: {np.abs(mean_drift).mean():.5f}/position/month")

    print(f"top-{a.J} positions...")
    top_j = top_j_positions(clouds, train_idx, a.J)

    train_sets = set()
    for i in train_idx:
        if clouds[i]: train_sets |= clouds[i]["sets"]
    print(f"training constellations: {len(train_sets)}")

    model = ResidualDriftModel(
        V=V, d=a.d, heads=a.heads, n_layers=a.layers,
        d_lstm=a.d_lstm, N_novel=a.N_novel, J=a.J,
        d_made=a.d_made, top_j_idx=top_j, M=a.M)
    model.set_mean_drift(mean_drift)
    model = model.to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\ntraining...")
    train(model, clouds, train_wins, a, device)

    print("\nevaluating...")
    evaluate(model, clouds, test_wins, months,
             a.horizons, a.top_k, device)

    os.makedirs("results", exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "mean_drift": mean_drift, "top_j": top_j},
               "results/149_model.pt")
    print("saved results/149_model.pt")

if __name__ == "__main__":
    main()
