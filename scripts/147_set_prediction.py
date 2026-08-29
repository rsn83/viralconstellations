#!/usr/bin/env python
"""
147_set_prediction.py  --  Direct 500->500 constellation set prediction.

Input:  top-500 constellations at month t with weights {(s_i, w_i)}
Output: top-500 constellations at month t+1 with weights {(s_j, w_j)}

Architecture:
  Encoder:  {(s_i, w_i)} -> phi_i (500, d) per-constellation embeddings
                          -> u_t (d) pooled for LSTM
  LSTM:     u_t over M months -> h_t temporal context
  Decoder:  500 output queries cross-attend to phi_i
            -> e_j (500, d) output slot embeddings
            -> MADE per slot: p(s_v | s_{<v}, e_j) for J positions
            -> factorized for V-J positions
  Output:   500 binary constellations + 500 weights

Mean centering:
  Input:  S_t - mu_t         (known at test time)
  Output: S_{t+1} - mu_{t+1} (known at train time, estimated at test)
  Test:   add mu_t + mean_drift back

Loss: weighted Chamfer in centered space
  Novel upweighting: novel constellations get upweight x more Chamfer weight

Evaluation: recall@k, novel recall@k vs persistence baseline

Run:
  python scripts/147_set_prediction.py \
    --data-dir data/processed/full_data_graphs_withdel \
    --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
    --months 2020-06:2023-06 --test-start 2022-06 \
    --M 6 --N 500 --J 200 --d 64 --heads 4 --layers 2 \
    --d-lstm 64 --d-made 32 --epochs 500 --lr 1e-3 \
    --top 500 --upweight 10.0 --top-k 20
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

# ------------------------------------------------------ top-J positions ---
def top_j_positions(clouds, train_idx, J):
    freq = freq2 = None; total = 0.0
    for i in train_idx:
        c = clouds[i]
        f  = (c["w"][:, None] * c["S"]).sum(0)
        f2 = (c["w"][:, None] * c["S"]**2).sum(0)
        freq  = f  if freq  is None else freq  + f
        freq2 = f2 if freq2 is None else freq2 + f2
        total += 1.0
    mu  = freq / total
    var = freq2 / total - mu**2
    top = np.sort(np.argsort(var)[::-1][:J])
    print(f"  top-{J} positions selected, mean var={var[top].mean():.4f}")
    return top

# --------------------------------------------------------- Jaccard --------
def jaccard_matrix(S_bin):
    dot = S_bin @ S_bin.T; sz = S_bin.sum(1)
    return torch.nan_to_num(
        dot / (sz.unsqueeze(1) + sz.unsqueeze(0) - dot + EPS), nan=0.0)

# ----------------------------------------------------------------- MADE ---
class MADE(nn.Module):
    """Autoregressive over J positions conditioned on slot embedding e_j."""
    def __init__(self, J, d_cond, d_hidden=32, seed=0):
        super().__init__()
        self.J = J
        rng     = np.random.default_rng(seed)
        degrees = rng.integers(1, max(J, 2), size=d_hidden)
        v_range = torch.arange(J).float()
        d_range = torch.tensor(degrees).float()
        mask_W1 = (d_range.unsqueeze(1) >= v_range.unsqueeze(0)).float()
        mask_W2 = (v_range.unsqueeze(1) > d_range.unsqueeze(0)).float()
        self.register_buffer('mask_W1', mask_W1)
        self.register_buffer('mask_W2', mask_W2)
        self.W1        = nn.Parameter(torch.randn(d_hidden, J) * 0.01)
        self.b1        = nn.Parameter(torch.zeros(d_hidden))
        self.W2        = nn.Parameter(torch.randn(J, d_hidden) * 0.01)
        self.b2        = nn.Parameter(torch.zeros(J))
        self.cond_proj = nn.Linear(d_cond, d_hidden, bias=False)

    def forward(self, s_J, c):
        """s_J: (N, J), c: (d,). Returns logits (N, J)."""
        h = torch.relu(s_J @ (self.W1 * self.mask_W1).T
                       + self.b1 + self.cond_proj(c))
        return h @ (self.W2 * self.mask_W2).T + self.b2

    def sample(self, c, n=1):
        """Sample n binary vectors from p(s|c). Returns (n, J)."""
        s = torch.zeros(n, self.J, device=c.device)
        for v in range(self.J):
            logits = self.forward(s, c)[:, v]
            s[:, v] = torch.bernoulli(torch.sigmoid(logits))
        return s

    def greedy_decode(self, c):
        """MAP: argmax at each position. Deterministic, no noise. Returns (J,)."""
        s = torch.zeros(self.J, device=c.device)
        for v in range(self.J):
            logit_v = self.forward(s.unsqueeze(0), c)[0, v]
            s[v] = (logit_v > 0).float()
        return s

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
        phi_i = x                                        # (N, d) KEPT
        u_t   = (w.unsqueeze(-1) * phi_i).sum(0)        # (d,) pooled
        u_t   = self.pool_out(torch.nan_to_num(u_t, nan=0.0))
        return phi_i, u_t


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


class SlotDecoder(nn.Module):
    """N output slots cross-attend to phi_i (N_in, d).
    Each slot -> embedding e_j -> MADE + factorized -> output constellation.

    Initialization:
      - queries initialized to zero -> at init Q = temporal context only
      - freq_head zero-init -> uniform weights at init
      - MADE/fact zero-init -> output ~ 0 -> add mu_next -> persistence

    Seen prediction:  slot j attends heavily to ONE phi_i -> modifies it
    Novel prediction: slot j attends to MULTIPLE phi_i -> cross-haplotype
    """
    def __init__(self, d, d_lstm, N, V, J, d_made, top_j_idx):
        super().__init__()
        self.N = N; self.V = V; self.J = J
        V_rest = V - J
        self.register_buffer('top_j_idx',
                             torch.tensor(top_j_idx, dtype=torch.long))

        # N learned queries -- one per output slot
        # initialized small so temporal context dominates initially
        self.queries = nn.Parameter(torch.randn(N, d) * 0.01)

        # temporal context: shifts all queries by same amount
        # -> tells all slots where population is heading
        self.ctx_proj = nn.Linear(d_lstm * 2, d)

        # cross-attention: N queries attend to N_in input embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=max(1, d // 16),
            batch_first=True)

        # MADE per slot: e_j -> p(s_v | s_{<v}, e_j) for J positions
        self.made = MADE(J, d_cond=d, d_hidden=d_made)

        # factorized for remaining V-J positions
        self.fact_net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, V_rest))
        nn.init.zeros_(self.fact_net[-1].weight)
        nn.init.zeros_(self.fact_net[-1].bias)

        # frequency head: e_j -> scalar weight
        self.freq_head = nn.Linear(d, 1)
        nn.init.zeros_(self.freq_head.weight)
        nn.init.zeros_(self.freq_head.bias)

    def _rest_idx(self, device):
        all_idx = torch.arange(self.V, device=device)
        mask    = torch.ones(self.V, dtype=torch.bool, device=device)
        mask[self.top_j_idx] = False
        return all_idx[mask]

    def forward(self, phi_i, w_i, h_t, context, mu_next,
                S_true_cen=None):
        """
        phi_i:     (N_in, d) input constellation embeddings
        w_i:       (N_in,) input frequencies
        h_t:       (d_lstm,) temporal hidden state
        context:   (d_lstm,) temporal context
        mu_next:   (V,) predicted population mean at t+h
        S_true_cen:(N_true, V) centered actual constellations
                   if provided: return log-probs for loss
                   if None: return sampled binary constellations

        Returns at train time:
          S_out_cen (N, V) centered predicted constellations
          w_out     (N,) predicted weights
          e_j       (N, d) slot embeddings (for MADE scoring)

        Returns at test time:
          S_out_bin (N, V) binary thresholded constellations
          w_out     (N,) predicted weights
        """
        # temporal context shifts all queries equally
        ctx   = torch.tanh(self.ctx_proj(
            torch.cat([h_t, context])))              # (d,)
        Q     = self.queries + ctx.unsqueeze(0)       # (N, d)

        # frequency-weight keys for cross-attention
        phi_w = phi_i * w_i.unsqueeze(-1)            # (N_in, d)

        # cross-attention: N output queries attend to N_in input embeddings
        e_j, attn = self.cross_attn(
            Q.unsqueeze(0),            # (1, N, d) queries
            phi_w.unsqueeze(0),        # (1, N_in, d) keys (freq-weighted)
            phi_i.unsqueeze(0))        # (1, N_in, d) values (unweighted)
        e_j = e_j.squeeze(0)          # (N, d)

        # slot frequencies
        w_out = torch.softmax(
            self.freq_head(e_j).squeeze(-1), dim=0)  # (N,)

        rest_idx = self._rest_idx(phi_i.device)

        # factorized output for non-MADE positions
        fact_out = torch.sigmoid(self.fact_net(e_j))  # (N, V_rest)

        # assemble full centered output
        S_cen = torch.zeros(self.N, self.V, device=phi_i.device)
        S_cen[:, rest_idx] = fact_out

        # MADE for J positions -- sample or score
        if S_true_cen is None:
            # test time: greedy MAP decode (deterministic, no sampling noise)
            for j in range(self.N):
                s_J = self.made.greedy_decode(e_j[j])            # (J,)
                S_cen[j, self.top_j_idx] = s_J
            # add mu_next back to get actual scale
            S_out = S_cen + mu_next.unsqueeze(0)     # (N, V)
            S_bin = (S_out > 0.5).float()
            return S_bin, w_out
        else:
            # train time: return e_j for MADE scoring in loss
            return S_cen, w_out, e_j

    def made_logprob(self, S_true_J, e_j):
        """Score actual constellations under MADE per slot.
        S_true_J: (N_true, J), e_j: (N_slots, d)
        Returns: (N_true, N_slots) log probs
        """
        return torch.stack([
            (S_true_J * F.logsigmoid(self.made(S_true_J, e_j[j]))
             + (1-S_true_J) * F.logsigmoid(-self.made(S_true_J, e_j[j]))
             ).sum(1)
            for j in range(self.N)
        ], dim=1)  # (N_true, N_slots)


class SetPredictionModel(nn.Module):
    def __init__(self, V, d, heads, n_layers, d_lstm,
                 N, J, d_made, top_j_idx, M):
        super().__init__()
        self.V = V; self.N = N; self.M = M
        self.enc  = SetEncoder(V, d, heads, n_layers)
        self.temp = TemporalEncoder(d, d_lstm)
        self.dec  = SlotDecoder(d, d_lstm, N, V, J,
                                d_made, top_j_idx)
        self.register_buffer('mean_drift', torch.zeros(V))

    def set_mean_drift(self, drift):
        self.mean_drift.copy_(torch.tensor(drift, dtype=torch.float32))

    def encode_month(self, S_bin, w, mu):
        return self.enc(S_bin - mu, S_bin, w)  # phi_i, u_t

    def forward_train(self, window, S_true, w_true, mu_t, mu_t1, h):
        """Training: score actual t+h constellations."""
        phis = []; us = []
        for data in window:
            phi, u = self.encode_month(
                data["S_bin"], data["w"], data["mu"])
            phis.append(phi); us.append(u)

        h_t, ctx  = self.temp(torch.stack(us))
        phi_last  = phis[-1]
        w_last    = window[-1]["w"]
        mu_next   = mu_t + self.mean_drift * h

        # centered actual output -- use ACTUAL mu_{t+h} at train time
        S_true_cen = S_true - mu_t1.unsqueeze(0)   # (N_true, V)

        S_cen, w_out, e_j = self.dec(
            phi_last, w_last, h_t, ctx,
            mu_next, S_true_cen=S_true_cen)

        return S_cen, w_out, e_j, mu_next, S_true_cen

    def forward_test(self, window, mu_t, h):
        """Test: sample 500 output constellations."""
        phis = []; us = []
        for data in window:
            phi, u = self.encode_month(
                data["S_bin"], data["w"], data["mu"])
            phis.append(phi); us.append(u)

        h_t, ctx = self.temp(torch.stack(us))
        phi_last = phis[-1]
        w_last   = window[-1]["w"]
        mu_next  = mu_t + self.mean_drift * h

        S_bin, w_out = self.dec(
            phi_last, w_last, h_t, ctx,
            mu_next, S_true_cen=None)

        return S_bin, w_out

# ------------------------------------------------------------ loss --------

def hamming_cost(A, B):
    """(N, M) normalized Hamming via dot product trick."""
    return (A.sum(1, keepdim=True) + B.sum(1, keepdim=True).T
            - 2 * A @ B.T) / A.shape[1]

def chamfer_upweighted(S_pred_cen, w_pred, S_true_cen, w_true,
                       S_true_bin, present_sets, upweight,
                       model, e_j):
    """Weighted Chamfer loss with novel upweighting.
    Uses MADE log-prob for J positions + factorized for rest
    as the 'distance' between predicted slot and actual constellation.

    For MADE positions: distance = -log p(s_J | e_j)  (negative log-prob)
    For factorized positions: squared difference in [0,1] space

    Novel constellations upweighted by upweight factor.
    """
    rest_idx = model.dec._rest_idx(S_pred_cen.device)
    top_j    = model.dec.top_j_idx

    # factorized distance for rest positions: (N_pred, N_true)
    # S_pred_cen[:, rest_idx] is factorized output in centered space
    # use sigmoid to get [0,1], then squared difference
    pred_rest = torch.sigmoid(S_pred_cen[:, rest_idx])  # (N_pred, V_rest)
    true_rest = S_true_bin[:, rest_idx].float()       # (N_true, V_rest) binary
    # squared diff: (N_pred, N_true)
    fact_dist = ((pred_rest.unsqueeze(1) -
                  true_rest.unsqueeze(0))**2).mean(-1)

    # MADE distance: -log p(s_J | e_j) for each (pred_slot, true_constellation)
    # e_j: (N_pred, d), S_true_J: (N_true, J)
    S_true_J  = S_true_bin[:, top_j].float()          # (N_true, J)
    made_lp   = model.dec.made_logprob(S_true_J, e_j) # (N_true, N_pred)
    made_dist = -made_lp.T                             # (N_pred, N_true)

    # total distance: (N_pred, N_true)
    C = fact_dist + made_dist

    # novel upweighting -- move to CPU for frozenset ops
    w_true_adj = w_true.clone()
    S_true_bin_cpu = S_true_bin.cpu()
    for i, row in enumerate(S_true_bin_cpu):
        fs = frozenset(torch.nonzero(row).squeeze(-1).tolist())
        if fs not in present_sets:
            w_true_adj[i] = w_true_adj[i] * upweight
    w_true_adj = w_true_adj / w_true_adj.sum()

    # weighted Chamfer
    fwd = (w_pred      * C.min(1).values).sum()  # each pred -> nearest true
    bwd = (w_true_adj  * C.min(0).values).sum()  # each true -> nearest pred
    return fwd + bwd

# ------------------------------------------------ evaluation metrics ----

def hamming_recall_at_k(S_pred_bin, w_pred, S_true_bin, w_true,
                        present_sets, top_k=20, max_dist=3):
    """Hit if predicted is within max_dist Hamming of any actual top-k."""
    top_k_pred_idx = np.argsort(w_pred)[::-1][:top_k]
    top_k_true_idx = np.argsort(w_true)[::-1][:top_k]
    pred_S = S_pred_bin[top_k_pred_idx]   # (top_k, V)
    true_S = S_true_bin[top_k_true_idx]   # (top_k, V)

    # for each predicted, min Hamming to any actual
    C = np.abs(pred_S[:, None, :] - true_S[None, :, :]).sum(-1)  # (k, k)
    hits = (C.min(1) <= max_dist).mean()

    # novel recall: actual constellations absent at t
    true_sets  = [frozenset(np.flatnonzero(r).tolist()) for r in S_true_bin]
    novel_mask = np.array([true_sets[i] not in present_sets
                           for i in top_k_true_idx])
    if novel_mask.sum() == 0:
        return float(hits), float('nan')
    novel_true_S = true_S[novel_mask]
    C_nov = np.abs(pred_S[:, None, :] - novel_true_S[None, :, :]).sum(-1)
    novel_hits = (C_nov.min(1) <= max_dist).mean()
    return float(hits), float(novel_hits)

def recall_at_k(S_pred_bin, w_pred, S_true_bin, w_true,
                present_sets, top_k=20):
    pred_sets    = [frozenset(np.flatnonzero(r).tolist())
                    for r in S_pred_bin]
    true_sets    = [frozenset(np.flatnonzero(r).tolist())
                    for r in S_true_bin]
    top_k_pred   = {pred_sets[i]
                    for i in np.argsort(w_pred)[::-1][:top_k]}
    top_k_true   = {true_sets[i]
                    for i in np.argsort(w_true)[::-1][:top_k]}
    recall       = len(top_k_pred & top_k_true) / max(len(top_k_true), 1)
    novel_true   = {s for s in top_k_true if s not in present_sets}
    novel_pred   = {s for s in top_k_pred if s not in present_sets}
    novel_recall = (len(novel_pred & novel_true) / max(len(novel_true), 1)
                    if novel_true else float('nan'))
    return recall, novel_recall

def persistence_recall(S_t, w_t, S_true, w_true, present_sets, top_k=20):
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

            mu_t   = torch.tensor(ct_t["mu"],
                                  dtype=torch.float32).to(device)
            mu_t1  = torch.tensor(ct_h["mu"],   # ACTUAL mu at t+h
                                  dtype=torch.float32).to(device)
            S_true = torch.tensor(ct_h["S"],
                                  dtype=torch.float32).to(device)
            w_true = torch.tensor(ct_h["w"],
                                  dtype=torch.float32).to(device)

            S_cen, w_out, e_j, mu_next, S_true_cen = \
                model.forward_train(window, S_true, w_true,
                                    mu_t, mu_t1, h)

            loss = chamfer_upweighted(
                S_cen, w_out, S_true_cen, w_true,
                S_true, ct_t["sets"], a.upweight, model, e_j)

            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        if (epoch+1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}  "
                  f"chamfer/win {total/len(train_wins):.5f}  "
                  f"lam={model.enc.lam.item():.3f}")

# ------------------------------------------------------------- evaluate --
def evaluate(model, clouds, test_wins, months,
             eval_h, top_k, device, max_dist=3):
    model.eval()
    print(f"\n{'Input window':>22} | {'Test':>8} | {'h':>3} | "
          f"{'rec@k':>6} {'nov_rec':>8} | {'per_rec':>7} {'per_nov':>8}")
    print("-" * 80)

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
            mu_t = torch.tensor(ct_t["mu"],
                                dtype=torch.float32).to(device)

            # sample 500 output constellations
            S_bin, w_out = model.forward_test(window, mu_t, h)

            S_pred_np = S_bin.cpu().numpy()
            w_pred_np = w_out.cpu().numpy()

            rec, nov = recall_at_k(
                S_pred_np, w_pred_np,
                ct_h["S"], ct_h["w"],
                ct_t["sets"], top_k=top_k)

            per_rec, per_nov = persistence_recall(
                ct_t["S"], ct_t["w"],
                ct_h["S"], ct_h["w"],
                ct_t["sets"], top_k=top_k)

            iw   = f"{months[inputs[0]]}..{months[inputs[-1]]}"
            nstr = f"{nov:.3f}"  if np.isfinite(nov)     else "  nan"
            pnstr= f"{per_nov:.3f}" if np.isfinite(per_nov) else "  nan"
            print(f"{iw:>22} | {months[target]:>8} | {h:>3} | "
                  f"{rec:6.3f} {nstr:>8} | "
                  f"{per_rec:7.3f} {pnstr:>8}")
            by_h[h].append((rec,
                            nov  if np.isfinite(nov)  else np.nan,
                            per_rec,
                            per_nov if np.isfinite(per_nov) else np.nan))

    print(f"\n{'='*60}")
    print(f"=== Summary (top-{top_k}) ===")
    print(f"{'h':>4} {'rec@k':>7} {'nov_rec':>9} | "
          f"{'per_rec':>8} {'per_nov':>9} | "
          f"{'rec_gain':>9} {'nov_gain':>9}")
    for h in eval_h:
        if not by_h[h]: continue
        R    = np.array(by_h[h])
        rec  = np.nanmean(R[:,0]); nov  = np.nanmean(R[:,1])
        prec = np.nanmean(R[:,2]); pnov = np.nanmean(R[:,3])
        print(f"  h={h:2d} {rec:7.3f} {nov:9.3f} | "
              f"{prec:8.3f} {pnov:9.3f} | "
              f"{rec-prec:+9.3f} {nov-pnov:+9.3f}")

# ----------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True)
    p.add_argument("--vocab",       required=True)
    p.add_argument("--months",      required=True)
    p.add_argument("--test-start",  required=True)
    p.add_argument("--engine",      default=ENGINE)
    p.add_argument("--M",       type=int,   default=6)
    p.add_argument("--N",       type=int,   default=500)
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
    p.add_argument("--top-k",    type=int,   default=20,  dest="top_k")
    p.add_argument("--max-dist", type=int,   default=3,   dest="max_dist")
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
    print(f"train: {len(train_wins)} windows  test: {len(test_wins)} windows")

    drifts     = [clouds[i+1]["mu"] - clouds[i]["mu"]
                  for i in range(ts-1) if clouds[i] and clouds[i+1]]
    mean_drift = np.stack(drifts).mean(0) if drifts else np.zeros(V)

    print(f"top-{a.J} positions...")
    top_j = top_j_positions(clouds, train_idx, a.J)

    train_sets = set()
    for i in train_idx:
        if clouds[i]: train_sets |= clouds[i]["sets"]
    print(f"training constellations: {len(train_sets)}")

    model = SetPredictionModel(
        V=V, d=a.d, heads=a.heads, n_layers=a.layers,
        d_lstm=a.d_lstm, N=a.N, J=a.J,
        d_made=a.d_made, top_j_idx=top_j, M=a.M)
    model.set_mean_drift(mean_drift)
    model = model.to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\ntraining...")
    train(model, clouds, train_wins, a, device)

    print("\nevaluating...")
    evaluate(model, clouds, test_wins, months,
             a.horizons, a.top_k, device, a.max_dist)

    os.makedirs("results", exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "mean_drift": mean_drift, "top_j": top_j},
               "results/147_model.pt")
    print("saved results/147_model.pt")

if __name__ == "__main__":
    main()
