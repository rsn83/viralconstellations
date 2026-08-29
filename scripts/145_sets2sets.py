#!/usr/bin/env python
"""
145_sets2sets.py  --  Direct weighted set to weighted set prediction.

Input:  top-500 constellations at month t with weights {(s_i, w_i)}
Output: top-500 constellations at month t+h with weights {(s_j, w_j)}

Both input and output are weighted point clouds over {0,1}^1359.
Same type, same size. Direct set-to-set map.

Key design:
  1. Mean centering: center constellations by population mean mu_t
     removes growing mutation count artifact
  2. Residual: decoder predicts CHANGE from input, not full output
     gate ~ 0 at init => output ~ input => persistence baseline
  3. Sinkhorn loss: differentiable Wasserstein between predicted and actual
     handles correspondence implicitly, no matching needed
  4. Set Transformer encoder + LSTM temporal model (from 143_v2)

Evaluation:
  - Wasserstein distance between predicted and actual weighted sets
  - Recall@k: actual top-k constellations in predicted top-k
  - Novel recall: same restricted to constellations absent at t

Run:
  python scripts/145_sets2sets.py \
    --data-dir data/processed/full_data_graphs_withdel \
    --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
    --months 2020-06:2023-06 --test-start 2022-06 \
    --M 6 --N 500 --K-out 500 --d 32 --heads 2 --layers 1 \
    --d-lstm 64 --epochs 500 --lr 1e-3 --top 500
"""

import argparse, importlib.util, sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logsumexp

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
        S, w = recs_to_matrix(recs, V, top)
        sets = {frozenset(s) for s, _ in recs}
        mu   = (w[:, None] * S).sum(0)      # (V,) population mean
        out.append({"S": S, "w": w, "sets": sets, "mu": mu, "ym": ym})
    print(f"  loaded {sum(1 for x in out if x)}/{len(months)} months")
    return out

# --------------------------------------------------------- Jaccard --------
def jaccard_matrix(S_bin):
    dot = S_bin @ S_bin.T; sz = S_bin.sum(1)
    return torch.nan_to_num(
        dot / (sz.unsqueeze(1) + sz.unsqueeze(0) - dot + EPS), nan=0.0)

# --------------------------------------------------------------- model ----

class SetEncoder(nn.Module):
    """Level 1: within-month Set Transformer.
    Input: centered constellations (N, V), frequencies (N,)
    Output: u (d,) -- month embedding
    """
    def __init__(self, V, d, heads, n_layers):
        super().__init__()
        self.proj = nn.Linear(V, d)
        self.lam  = nn.Parameter(torch.zeros(1))
        if n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=d*2,
                dropout=0.0, batch_first=True)
            self.attn = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.attn = None
        self.out = nn.Sequential(nn.Linear(d, d), nn.Tanh())

    def forward(self, S_centered, S_bin, w):
        """S_centered: (N,V) mean-centered, S_bin: (N,V) binary for Jaccard."""
        x = self.proj(S_centered)
        if self.attn is not None:
            J = jaccard_matrix(S_bin)
            b = torch.clamp(self.lam * J, -10., 10.)
            xa = self.attn(x.unsqueeze(0), mask=b).squeeze(0)
            x  = xa if torch.isfinite(xa).all() else x
        u = (w.unsqueeze(-1) * x).sum(0)
        return self.out(torch.nan_to_num(u, nan=0.0))


class TemporalEncoder(nn.Module):
    """Level 2: LSTM over monthly embeddings."""
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


class GatedResidualDecoder(nn.Module):
    """Set-to-set decoder. Takes input constellations and predicts output
    constellations via gated residual. Operates on centered constellations.

    For each input constellation s_i:
      gate_i = sigmoid(MLP(h_t, s̃_i)) in [0,1]^V   -- what to change
      new_i  = sigmoid(MLP(h_t, s̃_i)) in [0,1]^V   -- new values
      s̃_i_out = (1-gate)*s̃_i + gate*new_i           -- gated update
      δ_i    = MLP(h_t, s̃_i) scalar                  -- frequency correction

    At init: gate~0 => output~input => persistence
    """
    def __init__(self, V, d, d_lstm):
        super().__init__()
        self.V = V
        # context projection: (d_lstm*2,) -> (d,)
        self.ctx_proj = nn.Linear(d_lstm * 2, d)

        # per-constellation processing: takes [s_centered, context] -> gate, new, delta
        # shared MLP applied to each constellation independently
        self.gate_mlp = nn.Sequential(
            nn.Linear(V + d, d*2), nn.Tanh(),
            nn.Linear(d*2, V))
        self.new_mlp  = nn.Sequential(
            nn.Linear(V + d, d*2), nn.Tanh(),
            nn.Linear(d*2, V))
        self.freq_mlp = nn.Sequential(
            nn.Linear(V + d, d), nn.Tanh(),
            nn.Linear(d, 1))

        # initialize gate and freq MLPs to near-zero output -> persistence at init
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        nn.init.zeros_(self.freq_mlp[-1].weight)
        nn.init.zeros_(self.freq_mlp[-1].bias)

    def forward(self, S_centered, w, h_t, context):
        """
        S_centered: (N, V) mean-centered input constellations
        w:          (N,) input frequencies
        h_t:        (d_lstm,) LSTM hidden state
        context:    (d_lstm,) attention context
        Returns: S_out (N, V) in [0,1], w_out (N,) normalized
        """
        # project temporal context to d dimensions
        ctx = torch.tanh(self.ctx_proj(torch.cat([h_t, context])))  # (d,)

        # broadcast context to all constellations
        ctx_expanded = ctx.unsqueeze(0).expand(S_centered.shape[0], -1)  # (N, d)

        # concatenate each constellation with context
        inp = torch.cat([S_centered, ctx_expanded], dim=1)  # (N, V+d)

        # gated residual update
        gate  = torch.sigmoid(self.gate_mlp(inp))           # (N, V) in [0,1]
        new   = torch.sigmoid(self.new_mlp(inp))            # (N, V) in [0,1]
        S_out = (1 - gate) * S_centered + gate * new        # (N, V) gated update

        # frequency correction
        delta = self.freq_mlp(inp).squeeze(-1)              # (N,)
        w_out = torch.softmax(torch.log(w + EPS) + delta, dim=0)  # (N,)

        return S_out, w_out


class Sets2Sets(nn.Module):
    """Full set-to-set model.
    Input:  weighted constellation cloud at month t (centered)
    Output: weighted constellation cloud at month t+h (centered)
    """
    def __init__(self, V, d, heads, n_layers, d_lstm, M):
        super().__init__()
        self.V = V; self.M = M
        self.set_enc  = SetEncoder(V, d, heads, n_layers)
        self.temp_enc = TemporalEncoder(d, d_lstm)
        self.decoder  = GatedResidualDecoder(V, d, d_lstm)
        self.register_buffer("mean_drift", torch.zeros(V))

    def set_mean_drift(self, drift):
        self.mean_drift.copy_(torch.tensor(drift, dtype=torch.float32))

    def encode_month(self, S_bin, w, mu_t):
        """Encode one month. Centers by mu_t before encoding."""
        S_cen = S_bin - mu_t                     # center by population mean
        return self.set_enc(S_cen, S_bin, w)

    def forward(self, window, S_bin_t, w_t, mu_t, h=1):
        """
        window:  list of M dicts with S_bin, w, mu per month
        S_bin_t: (N, V) input constellations at last window month
        w_t:     (N,) input frequencies
        mu_t:    (V,) population mean at t
        h:       prediction horizon
        Returns: S_out (N, V) in [0,1], w_out (N,) normalized
        """
        # encode all window months
        us = torch.stack([
            self.encode_month(d["S_bin"], d["w"], d["mu"])
            for d in window])                    # (M, d)

        # temporal encoding
        h_t, ctx = self.temp_enc(us)             # (d_lstm,), (d_lstm,)

        # center input constellations
        S_cen = S_bin_t - mu_t                   # (N, V)

        # gated residual decode
        S_out, w_out = self.decoder(S_cen, w_t, h_t, ctx)  # (N, V), (N,)

        # add predicted mu_{t+h} back to get actual scale
        mu_next = mu_t + self.mean_drift * h     # (V,) linear drift extrapolation
        S_final = torch.sigmoid(
            S_out + torch.logit(mu_next.clamp(1e-3, 1-1e-3)).unsqueeze(0))  # (N, V)

        return S_final, w_out

# ------------------------------------------------------ Sinkhorn loss ----

def sinkhorn(a, b, C, reg=0.1, n_iter=20):
    """Sinkhorn algorithm for differentiable Wasserstein distance.
    a: (N,) source weights
    b: (M,) target weights
    C: (N, M) cost matrix
    Returns scalar Wasserstein distance.
    """
    K = torch.exp(-C / reg)
    u = torch.ones_like(a)
    for _ in range(n_iter):
        v = b / (K.T @ u + EPS)
        u = a / (K @ v + EPS)
    T = u.unsqueeze(1) * K * v.unsqueeze(0)    # (N, M) transport plan
    return (T * C).sum()

def hamming_cost(S_pred, S_true):
    """Pairwise Hamming distance matrix.
    S_pred: (N, V) predicted in [0,1]
    S_true: (M, V) actual binary
    Returns: (N, M) cost matrix
    """
    # expected Hamming distance under Bernoulli relaxation
    # E[|s_pred - s_true|] = pred*(1-true) + (1-pred)*true
    # = pred + true - 2*pred*true
    return (S_pred.unsqueeze(1) + S_true.unsqueeze(0)
            - 2 * S_pred.unsqueeze(1) * S_true.unsqueeze(0)).sum(-1)  # (N, M)

def sets2sets_loss(S_pred, w_pred, S_true, w_true, reg=0.05):
    """Sinkhorn loss between predicted and actual weighted sets."""
    C = hamming_cost(S_pred, S_true) / S_pred.shape[1]  # normalize by V
    return sinkhorn(w_pred, w_true, C, reg=reg)

# ------------------------------------------------ evaluation metrics ----

def evaluate_predictions(S_pred_bin, w_pred, S_true_bin, w_true,
                          S_t_sets, top_k=20):
    """
    S_pred_bin: (N, V) binary thresholded predictions
    w_pred:     (N,) predicted frequencies
    S_true_bin: (M, V) actual binary constellations at t+h
    w_true:     (M,) actual frequencies
    S_t_sets:   set of frozensets at month t (to identify novel)
    top_k:      k for recall@k
    """
    # convert to frozensets
    pred_sets = [frozenset(np.flatnonzero(row).tolist())
                 for row in S_pred_bin]
    true_sets = [frozenset(np.flatnonzero(row).tolist())
                 for row in S_true_bin]

    # top-k by predicted weight
    top_k_idx   = np.argsort(w_pred)[::-1][:top_k]
    top_k_preds = {pred_sets[i] for i in top_k_idx}

    # top-k actual by true weight
    top_k_true_idx = np.argsort(w_true)[::-1][:top_k]
    top_k_true     = {true_sets[i] for i in top_k_true_idx}

    # recall@k overall
    recall_k = len(top_k_preds & top_k_true) / max(len(top_k_true), 1)

    # novel: true constellations absent at t
    novel_true = {s for s in top_k_true if s not in S_t_sets}
    novel_pred = {pred_sets[i] for i in top_k_idx
                  if pred_sets[i] not in S_t_sets}
    novel_recall = (len(novel_pred & novel_true) / max(len(novel_true), 1)
                    if novel_true else float('nan'))

    # Wasserstein (approximate, use top-100 for speed)
    n = min(100, len(pred_sets), len(true_sets))
    S_p = torch.tensor(S_pred_bin[:n], dtype=torch.float32)
    S_t_th = torch.tensor(S_true_bin[:n], dtype=torch.float32)
    w_p = torch.tensor(w_pred[:n] / w_pred[:n].sum(), dtype=torch.float32)
    w_t = torch.tensor(w_true[:n] / w_true[:n].sum(), dtype=torch.float32)
    with torch.no_grad():
        C   = hamming_cost(S_p, S_t_th) / S_p.shape[1]
        wass = sinkhorn(w_p, w_t, C).item()

    return recall_k, novel_recall, wass

def persistence_recall(S_t_bin, w_t, S_true_bin, w_true,
                       S_t_sets, top_k=20):
    """Persistence baseline: predict top-k at t as top-k at t+h."""
    true_sets      = [frozenset(np.flatnonzero(row).tolist())
                      for row in S_true_bin]
    top_k_t_idx    = np.argsort(w_t)[::-1][:top_k]
    top_k_t        = {frozenset(np.flatnonzero(S_t_bin[i]).tolist())
                      for i in top_k_t_idx}
    top_k_true_idx = np.argsort(w_true)[::-1][:top_k]
    top_k_true     = {true_sets[i] for i in top_k_true_idx}
    recall_k       = len(top_k_t & top_k_true) / max(len(top_k_true), 1)
    novel_true     = {s for s in top_k_true if s not in S_t_sets}
    novel_recall   = (len(set() & novel_true) / max(len(novel_true), 1)
                      if novel_true else float('nan'))
    return recall_k, novel_recall

# --------------------------------------------------------- windows -------
def make_windows(clouds, M, horizons, start, end):
    wins = []
    for i in range(start, end):
        for h in horizons:
            target = i + h
            inputs = list(range(i-M+1, i+1))
            if inputs[0] < 0: continue
            if any(clouds[j] is None for j in inputs): continue
            if target >= len(clouds) or clouds[target] is None: continue
            wins.append((inputs, target, h))
    return wins

# --------------------------------------------------------------- train ---
def train(model, clouds, train_wins, a):
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-3)

    for epoch in range(a.epochs):
        model.train(); total = 0.0
        for inputs, target, h in [train_wins[i] for i in
                                   np.random.permutation(len(train_wins))]:
            ct_t   = clouds[inputs[-1]]
            ct_h   = clouds[target]
            window = [{"S_bin": torch.tensor(clouds[i]["S"], dtype=torch.float32),
                       "w":     torch.tensor(clouds[i]["w"], dtype=torch.float32),
                       "mu":    torch.tensor(clouds[i]["mu"], dtype=torch.float32)}
                      for i in inputs]
            S_bin_t = torch.tensor(ct_t["S"], dtype=torch.float32)
            w_t     = torch.tensor(ct_t["w"], dtype=torch.float32)
            mu_t    = torch.tensor(ct_t["mu"], dtype=torch.float32)
            S_true  = torch.tensor(ct_h["S"], dtype=torch.float32)
            w_true  = torch.tensor(ct_h["w"], dtype=torch.float32)

            S_pred, w_pred = model(window, S_bin_t, w_t, mu_t, h=h)
            loss = sets2sets_loss(S_pred, w_pred, S_true, w_true, reg=a.reg)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        if (epoch+1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}  "
                  f"sinkhorn/win {total/len(train_wins):.4f}  "
                  f"lam={model.set_enc.lam.item():.3f}")

# ------------------------------------------------------------- evaluate --
def evaluate(model, clouds, test_wins, months, eval_h, top_k=20):
    model.eval()

    print(f"\n{'Input window':>22} | {'Test':>8} | "
          f"{'h':>3} | {'rec@k':>6} {'nov_rec':>8} {'wass':>7} | "
          f"{'per_rec':>8} {'per_nov':>8}")
    print("-" * 90)

    by_h = {h: [] for h in eval_h}

    with torch.no_grad():
        for inputs, target, h in test_wins:
            if h not in eval_h: continue
            ct_t   = clouds[inputs[-1]]
            ct_h   = clouds[target]
            window = [{"S_bin": torch.tensor(clouds[i]["S"], dtype=torch.float32),
                       "w":     torch.tensor(clouds[i]["w"], dtype=torch.float32),
                       "mu":    torch.tensor(clouds[i]["mu"], dtype=torch.float32)}
                      for i in inputs]
            S_bin_t = torch.tensor(ct_t["S"], dtype=torch.float32)
            w_t     = torch.tensor(ct_t["w"], dtype=torch.float32)
            mu_t    = torch.tensor(ct_t["mu"], dtype=torch.float32)

            S_pred, w_pred = model(window, S_bin_t, w_t, mu_t, h=h)

            # threshold to binary
            S_pred_bin = (S_pred.numpy() > 0.5).astype(np.float32)
            w_pred_np  = w_pred.numpy()

            rec_k, nov_rec, wass = evaluate_predictions(
                S_pred_bin, w_pred_np,
                ct_h["S"], ct_h["w"],
                ct_t["sets"], top_k=top_k)

            per_rec, per_nov = persistence_recall(
                ct_t["S"], ct_t["w"],
                ct_h["S"], ct_h["w"],
                ct_t["sets"], top_k=top_k)

            iw = f"{months[inputs[0]]}..{months[inputs[-1]]}"
            print(f"{iw:>22} | {months[target]:>8} | "
                  f"{h:>3} | {rec_k:6.3f} {nov_rec:8.3f} {wass:7.3f} | "
                  f"{per_rec:8.3f} {per_nov:8.3f}")

            by_h[h].append((rec_k, nov_rec, wass, per_rec, per_nov))

    print(f"\n{'='*60}")
    print(f"=== Summary (mean across test windows, top-{top_k}) ===")
    print(f"{'h':>4} {'rec@k':>8} {'nov_rec':>10} {'wass':>8} | "
          f"{'per_rec':>8} {'per_nov':>10}")
    for h in eval_h:
        if not by_h[h]: continue
        R = np.array(by_h[h])
        # handle NaN in novel recall
        nov = R[:, 1]; nov = nov[np.isfinite(nov)]
        print(f"  h={h:2d} "
              f"{R[:,0].mean():8.3f} "
              f"{nov.mean() if len(nov) else float('nan'):10.3f} "
              f"{R[:,2].mean():8.3f} | "
              f"{R[:,3].mean():8.3f} "
              f"{R[:,4].mean():10.3f}")
        print(f"       rec gain: {R[:,0].mean()-R[:,3].mean():+.3f}  "
              f"novel gain: "
              f"{(nov.mean()-R[:,4].mean()) if len(nov) else float('nan'):+.3f}")

# ----------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True)
    p.add_argument("--vocab",       required=True)
    p.add_argument("--months",      required=True)
    p.add_argument("--test-start",  required=True)
    p.add_argument("--engine",      default=ENGINE)
    p.add_argument("--M",     type=int,   default=6)
    p.add_argument("--d",     type=int,   default=32)
    p.add_argument("--heads", type=int,   default=2)
    p.add_argument("--layers",type=int,   default=1)
    p.add_argument("--d-lstm",type=int,   default=64, dest="d_lstm")
    p.add_argument("--epochs",type=int,   default=500)
    p.add_argument("--lr",    type=float, default=1e-3)
    p.add_argument("--top",   type=int,   default=500)
    p.add_argument("--reg",   type=float, default=0.05,
                   help="Sinkhorn regularization")
    p.add_argument("--top-k", type=int,   default=20, dest="top_k")
    p.add_argument("--horizons", type=int, nargs="+", default=[1,2,3,6])
    a = p.parse_args()

    E = load_engine(a.engine)
    idx2name, V = E.load_names(a.vocab)
    print(f"V={V}")
    months = E.months_in_range(a.months)
    print(f"{len(months)} months: {months[0]} .. {months[-1]}")
    if a.test_start not in months:
        print("--test-start not found"); sys.exit(1)
    ts = months.index(a.test_start)

    clouds = load_all(E, a.data_dir, months, V, a.top)

    train_idx   = list(range(ts))
    train_wins  = make_windows(clouds, a.M, a.horizons, a.M-1, ts)
    test_wins   = make_windows(clouds, a.M, a.horizons, ts, len(months))
    print(f"train windows: {len(train_wins)}  test windows: {len(test_wins)}")

    # compute mean drift from training pairs
    drifts = []
    for i in range(ts-1):
        if clouds[i] and clouds[i+1]:
            drifts.append(clouds[i+1]["mu"] - clouds[i]["mu"])
    mean_drift = np.stack(drifts).mean(0) if drifts else np.zeros(V)
    print(f"mean drift magnitude: {np.abs(mean_drift).mean():.5f}/position")

    model = Sets2Sets(V=V, d=a.d, heads=a.heads,
                      n_layers=a.layers, d_lstm=a.d_lstm, M=a.M)
    model.set_mean_drift(mean_drift)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\ntraining...")
    train(model, clouds, train_wins, a)

    print("\nevaluating...")
    evaluate(model, clouds, test_wins, months, a.horizons, a.top_k)

    os.makedirs("results", exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "mean_drift": mean_drift},
               "results/145_model.pt")
    print("saved results/145_model.pt")

if __name__ == "__main__":
    main()
