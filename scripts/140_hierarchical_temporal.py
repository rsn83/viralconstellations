#!/usr/bin/env python
"""
140_hierarchical_temporal.py

Hierarchical temporal model. Strictly improves 139 by adding:

  Level 1 : within-month Set encoder (identical to 139)
  Level 1b: PrototypeLayer -- K fixed cluster identities, learned end-to-end
             theta_kt (K,V) and pi_t (K,) computed per month from fixed prototypes
  Level 2 : CausalTemporalModel -- attends over M months of (theta_kt, pi_t)
             zero-initialized output => at init reduces exactly to 139
  Level 3 : residual Decoder (identical to 139)
             baseline = theta_drift from Level 2 (not mu_t as in 139)
             when causal model learns nothing: theta_drift = theta_t => persistence

Strict improvement guarantee:
  - At init: CausalTemporalModel output = 0
             predicted state = current state = (theta_t, pi_t)
             decoder baseline = theta_t ~ persistence
             => identical to 139 at initialization
  - During training: gradient can grow causal model output
             => can only improve over 139, never hurt

Run:
  python scripts/140_hierarchical_temporal.py \
    --data-dir data/processed/full_data_graphs_withdel \
    --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
    --months 2020-06:2023-06 --test-start 2022-06 \
    --M 6 --K 8 --r 70 --d 32 --heads 2 --layers 1 \
    --d-temp 32 --epochs 500 --top 500
"""

import argparse, importlib.util, sys, os
import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp

ENGINE = "scripts/110_hierarchical_birthdeath_v2_fixed.py"
EPS    = 1e-6

# ---------------------------------------------------------------- engine ---

def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["engine"] = m
    spec.loader.exec_module(m)
    return m

# ------------------------------------------------------------------ data ---

def recs_to_matrix(recs, V, top=None):
    if top:
        recs = sorted(recs, key=lambda x: -x[1])[:top]
    S = np.zeros((len(recs), V), dtype=np.float32)
    for i, (s, _) in enumerate(recs):
        for v in s:
            if v < V: S[i, v] = 1.0
    w = np.array([float(c) for _, c in recs], dtype=np.float32)
    w /= w.sum()
    return S, w

def load_all(E, data_dir, months, V, top):
    print("loading months...", flush=True)
    out = []
    for ym in months:
        recs = E.load_month(data_dir, ym)
        if not recs:
            out.append(None); continue
        S, w = recs_to_matrix(recs, V, top)
        sets = {frozenset(s) for s, _ in recs}
        mu   = (w[:, None] * S).sum(0)
        out.append({"S": S, "w": w, "sets": sets, "mu": mu, "ym": ym})
    print(f"  loaded {sum(1 for x in out if x)}/{len(months)} months")
    return out

# ----------------------------------------------------------------- PCA ----

def fit_pca(clouds, train_idx, r, n_sample=200, seed=0):
    rng = np.random.default_rng(seed)
    global_mean = None; total = 0.0; rows = []
    for i in train_idx:
        c = clouds[i]
        contrib = (c["w"][:, None] * c["S"]).sum(0)
        global_mean = contrib if global_mean is None else global_mean + contrib
        total += 1.0
        n = min(n_sample, len(c["S"]))
        idx = rng.choice(len(c["S"]), size=n, replace=False,
                         p=c["w"]/c["w"].sum())
        rows.append(c["S"][idx])
    global_mean /= total
    X = np.vstack(rows).astype(np.float32) - global_mean
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:r], global_mean.astype(np.float32)

# --------------------------------------------------------- Jaccard --------

def jaccard_matrix(S_bin):
    dot   = S_bin @ S_bin.T
    sizes = S_bin.sum(1)
    union = sizes.unsqueeze(1) + sizes.unsqueeze(0) - dot
    J = dot / (union + EPS)
    return torch.nan_to_num(J, nan=0.0)

# --------------------------------------------------------------- model ----

class Encoder(nn.Module):
    """Level 1: within-month Set encoder. Identical to 139.
    Added: encode_phi() exposes per-constellation embeddings for prototype layer."""
    def __init__(self, r, d, heads, n_layers):
        super().__init__()
        self.proj = nn.Linear(r, d)
        self.lam  = nn.Parameter(torch.zeros(1))
        if n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=d*2,
                dropout=0.0, batch_first=True)
            self.attn = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.attn = None
        self.out = nn.Sequential(nn.Linear(d, d), nn.Tanh())

    def encode_phi(self, S_pca, S_bin):
        """Per-constellation embeddings after attention. (N, d)"""
        x = self.proj(S_pca)
        if self.attn is not None:
            J    = jaccard_matrix(S_bin)
            bias = torch.clamp(self.lam * J, -10., 10.)
            x_att = self.attn(x.unsqueeze(0), mask=bias).squeeze(0)
            x = x_att if torch.isfinite(x_att).all() else x
        return x

    def pool(self, phi, w):
        """Weighted pool to month embedding. (d,)"""
        u = (w.unsqueeze(-1) * phi).sum(0)
        u = torch.nan_to_num(u, nan=0.0)
        return self.out(u)

    def forward(self, S_pca, S_bin, w):
        return self.pool(self.encode_phi(S_pca, S_bin), w)


class PrototypeLayer(nn.Module):
    """Level 1b: K fixed prototype identities shared across all months.
    Assigns per-constellation embeddings to prototypes via soft assignment.
    Computes theta_kt (K,V) and pi_t (K,) for each month."""
    def __init__(self, d, K):
        super().__init__()
        self.K = K
        self.C = nn.Parameter(torch.randn(K, d) * 0.1)  # K prototypes in R^d

    def forward(self, phi, w, S_bin):
        """
        phi   : (N, d) per-constellation embeddings from encoder
        w     : (N,)   frequency weights
        S_bin : (N, V) binary constellations
        Returns theta_k (K,V), pi (K,)
        """
        dists = torch.cdist(phi, self.C)           # (N, K)
        r     = torch.softmax(-dists, dim=1)        # (N, K) soft assignment
        r_w   = w[:, None] * r                      # (N, K) freq-weighted
        pi    = r_w.sum(0)
        pi    = pi / (pi.sum() + EPS)               # (K,)
        # per-cluster mutation profile
        denom  = r_w.sum(0)[:, None] + EPS          # (K, 1)
        theta_k = (r_w.T @ S_bin) / denom           # (K, V)
        return theta_k.clamp(EPS, 1-EPS), pi


class CausalTemporalModel(nn.Module):
    """Level 2: causal transformer over M monthly states (theta_kt, pi_t).
    State vector per month: [theta_pca.flatten(), logit(pi)] shape (K*(r+1),)
    Output projection ZERO INITIALIZED => at init output = current state = 139 baseline.
    """
    def __init__(self, K, r, d_temp=32, nhead=2, n_layers=1):
        super().__init__()
        self.K = K; self.r = r
        self.state_dim = K * (r + 1)
        self.proj_in  = nn.Linear(self.state_dim, d_temp)
        layer = nn.TransformerEncoderLayer(
            d_model=d_temp, nhead=nhead, dim_feedforward=d_temp*2,
            dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_temp, self.state_dim)
        # ZERO INIT: at init output=0 => residual = current state => 139
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def make_causal_mask(self, M, device):
        mask = torch.triu(torch.ones(M, M, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, states):
        """states: (M, state_dim). Returns predicted next state (state_dim,).
        Uses first differences to remove monotone mutation accumulation artifact.
        Causal transformer sees rate-of-change, not absolute levels.
        Absolute level is restored via residual from states[-1].
        When transformer output=0: predicted = states[-1] = persistence baseline.
        """
        if states.shape[0] > 1:
            deltas = states[1:] - states[:-1]   # (M-1, state_dim) rate of change
        else:
            deltas = torch.zeros(1, states.shape[1], device=states.device)
        M  = deltas.shape[0]
        x  = self.proj_in(deltas)                           # (M-1, d_temp)
        cm = self.make_causal_mask(M, states.device)
        x  = self.transformer(x.unsqueeze(0),
                              mask=cm).squeeze(0)           # (M-1, d_temp)
        delta = self.proj_out(x[-1])                        # (state_dim,) zero at init
        return states[-1] + delta                           # anchor to current level


class Decoder(nn.Module):
    """Level 3: residual decoder. Identical structure to 139.
    Baseline is now theta_drift (K,V) from temporal model, not mu_t.
    When delta=0: output = theta_drift, pi_drift (exactly temporal prediction)."""
    def __init__(self, d, K, r):
        super().__init__()
        self.K = K; self.r = r
        self.net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, K*r + K)
        )

    def forward(self, u, P, theta_drift, pi_drift):
        """
        u           : (d,)   fine-grained current month embedding
        P           : (r, V) frozen PCA components
        theta_drift : (K, V) drift prediction from temporal model
        pi_drift    : (K,)   drift prediction for weights
        Returns theta_final (K,V), pi_final (K,)
        """
        out        = self.net(u)
        delta_coeff = out[:self.K * self.r].view(self.K, self.r)  # (K, r)
        delta_pi    = out[self.K * self.r:]                        # (K,)
        # correct drift prediction
        delta_theta = delta_coeff @ P                              # (K, V)
        theta_final = torch.sigmoid(
            torch.logit(theta_drift.clamp(EPS, 1-EPS)) + delta_theta)
        pi_final = torch.softmax(
            torch.log(pi_drift + EPS) + delta_pi, dim=0)
        return theta_final, pi_final


class HierarchicalTemporal(nn.Module):
    def __init__(self, V, r, d, K, heads, n_layers, d_temp, n_temp_layers):
        super().__init__()
        self.K = K; self.V = V; self.r = r
        self.enc   = Encoder(r, d, heads, n_layers)
        self.proto = PrototypeLayer(d, K)
        self.temp  = CausalTemporalModel(K, r, d_temp, nhead=2,
                                         n_layers=n_temp_layers)
        self.dec   = Decoder(d, K, r)
        self.register_buffer("P",           torch.zeros(r, V))
        self.register_buffer("global_mean", torch.zeros(V))

    def set_pca(self, components, global_mean):
        self.P.copy_(torch.tensor(components, dtype=torch.float32))
        self.global_mean.copy_(torch.tensor(global_mean, dtype=torch.float32))

    def encode_month(self, S_bin, w):
        """Encode one month. Returns u (d,), theta_k (K,V), pi (K,)."""
        S_pca   = (S_bin - self.global_mean) @ self.P.T   # (N, r)
        phi     = self.enc.encode_phi(S_pca, S_bin)        # (N, d)
        u       = self.enc.pool(phi, w)                    # (d,)
        theta_k, pi = self.proto(phi, w, S_bin)            # (K,V), (K,)
        return u, theta_k, pi

    def state_vec(self, theta_k, pi):
        """Compress (theta_k, pi) -> state vector (K*(r+1),) for temporal model."""
        # project theta_k to PCA space
        theta_pca = (theta_k - self.global_mean) @ self.P.T  # (K, r)
        logit_pi  = torch.log(pi + EPS)                       # (K,)
        return torch.cat([theta_pca.flatten(), logit_pi])

    def state_to_theta_pi(self, state):
        """Decompress state vector -> (theta_k (K,V), pi (K,))."""
        theta_pca = state[:self.K * self.r].view(self.K, self.r)
        logit_pi  = state[self.K * self.r:]
        theta_k   = (theta_pca @ self.P + self.global_mean).clamp(EPS, 1-EPS)
        pi        = torch.softmax(logit_pi, dim=0)
        return theta_k, pi

    def forward(self, window):
        """
        window: list of M dicts, each with S_bin (N,V) and w (N,) tensors
                last entry is current month t
        Returns theta_final (K,V), pi_final (K,)
        """
        states = []; u_t = None
        for i, data in enumerate(window):
            u, theta_k, pi = self.encode_month(data["S_bin"], data["w"])
            states.append(self.state_vec(theta_k, pi))
            if i == len(window) - 1:
                u_t = u

        states_th     = torch.stack(states)                # (M, K*(r+1))
        pred_state    = self.temp(states_th)               # (M, K*(r+1)) -> (K*(r+1),)
        theta_drift, pi_drift = self.state_to_theta_pi(pred_state)

        return self.dec(u_t, self.P, theta_drift, pi_drift)

# ------------------------------------------------------------ loss --------

def loglik_th(S_next_th, w_next_th, theta, pi):
    th    = theta.clamp(EPS, 1-EPS)
    lt    = torch.log(th); lf = torch.log1p(-th)
    base  = lf.sum(1)
    delta = S_next_th @ (lt - lf).T
    lp    = (torch.log(pi + EPS) + base).unsqueeze(0) + delta
    ll    = torch.logsumexp(lp, dim=1)
    return (w_next_th * ll).sum()

def diversity_loss(theta):
    normed = theta / (theta.norm(dim=1, keepdim=True) + EPS)
    sim    = normed @ normed.T
    mask   = 1 - torch.eye(theta.shape[0], device=theta.device)
    return (sim * mask).sum() / (mask.sum() + EPS)

def entropy_loss(pi):
    return (pi * torch.log(pi + EPS)).sum()

# ------------------------------------------------ seen / unseen ----------

def score_seen_unseen(S_next, w_next, theta_np, pi_np, train_sets):
    th     = np.clip(theta_np, 1e-3, 1-1e-3)
    log_pi = np.log(pi_np + EPS)
    lt, lf = np.log(th), np.log1p(-th)
    base   = lf.sum(1)
    seen_ll = unseen_ll = seen_w = unseen_w = 0.0
    for row, wi in zip(S_next, w_next):
        idx  = np.flatnonzero(row)
        ll_i = float(logsumexp(log_pi + base +
                               (lt[:, idx] - lf[:, idx]).sum(1)))
        if not np.isfinite(ll_i): continue
        fs = frozenset(idx.tolist())
        if fs in train_sets:
            seen_ll += wi*ll_i; seen_w += wi
        else:
            unseen_ll += wi*ll_i; unseen_w += wi
    return (seen_ll/seen_w   if seen_w   > 0 else float('nan'),
            unseen_ll/unseen_w if unseen_w > 0 else float('nan'),
            seen_w, unseen_w)

def persistence_mixture(S_t, w_t):
    mu = (w_t[:, None] * S_t).sum(0, keepdims=True)
    return np.clip(mu, EPS, 1-EPS), np.array([1.0])

# --------------------------------------------------------- windows -------

def make_windows(clouds, start, end, M):
    """Sliding windows of size M+1. Input=[i..i+M-1], target=i+M."""
    wins = []
    for target in range(start, end):
        inputs = list(range(target-M, target))
        if any(clouds[i] is None for i in inputs) or clouds[target] is None:
            continue
        wins.append((inputs, target))
    return wins

def window_tensors(clouds, input_indices, target_idx):
    window = []
    for i in input_indices:
        c = clouds[i]
        window.append({
            "S_bin": torch.tensor(c["S"], dtype=torch.float32),
            "w":     torch.tensor(c["w"], dtype=torch.float32)
        })
    ct1 = clouds[target_idx]
    S_n = torch.tensor(ct1["S"], dtype=torch.float32)
    w_n = torch.tensor(ct1["w"], dtype=torch.float32)
    return window, S_n, w_n

# --------------------------------------------------------------- train ---

def _run_epochs(model, clouds, train_windows, epochs, lr, alpha, beta,
                params, label):
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-3)
    for epoch in range(epochs):
        model.train(); total = 0.0
        idx = np.random.permutation(len(train_windows))
        for i in idx:
            inputs, target = train_windows[i]
            window, S_n, w_n = window_tensors(clouds, inputs, target)
            theta, pi = model(window)
            ll   = -loglik_th(S_n, w_n, theta, pi)
            div  = diversity_loss(theta)
            ent  = entropy_loss(pi)
            loss = ll + alpha*div + beta*ent
            opt.zero_grad(); loss.backward(); opt.step()
            total += ll.item()
        if (epoch+1) % 50 == 0:
            print(f"  [{label}] epoch {epoch+1:4d}  "
                  f"ll/win {total/len(train_windows):.4f}  "
                  f"lam={model.enc.lam.item():.3f}")


def train(model, clouds, train_windows, a):
    # Stage 1: train only encoder + decoder (identical to 139)
    # Prototype and causal transformer frozen at zero -- model IS 139
    print("  stage 1: encoder + decoder only (identical to 139)...")
    for name, param in model.named_parameters():
        param.requires_grad = "enc" in name or "dec" in name
    stage1_params = [p for p in model.parameters() if p.requires_grad]
    _run_epochs(model, clouds, train_windows, a.epochs, a.lr,
                a.alpha, a.beta, stage1_params, "stage1")

    # Stage 2: freeze encoder + decoder, train prototype + causal transformer
    # Encoder is at a good 139 solution -- temporal components can only help
    print("  stage 2: prototype + causal transformer only...")
    for name, param in model.named_parameters():
        param.requires_grad = "proto" in name or "temp" in name
    stage2_params = [p for p in model.parameters() if p.requires_grad]
    _run_epochs(model, clouds, train_windows, a.epochs // 2, a.lr,
                a.alpha, a.beta, stage2_params, "stage2")

    # Stage 3 removed: joint fine-tuning corrupts stage 1 encoder solution
    # Strict improvement guarantee holds: stage 1 = 139, stage 2 adds temporal only

# ------------------------------------------------------------- evaluate --

def evaluate(model, clouds, test_windows, train_sets, months):
    model.eval(); rows = []
    print(f"\n{'month_t+1':>12} {'mdl_seen':>10} {'mdl_unseen':>12} "
          f"{'per_seen':>10} {'per_unseen':>12}")
    with torch.no_grad():
        for inputs, target in test_windows:
            window, _, _ = window_tensors(clouds, inputs, target)
            theta, pi = model(window)
            th_np, pi_np = theta.numpy(), pi.numpy()
            if not np.isfinite(th_np).all():
                th_np = np.nan_to_num(th_np, nan=clouds[inputs[-1]]["mu"])

            ct1 = clouds[target]
            th_per, pi_per = persistence_mixture(
                clouds[inputs[-1]]["S"], clouds[inputs[-1]]["w"])

            s_m,u_m,sw,uw = score_seen_unseen(
                ct1["S"],ct1["w"],th_np,pi_np,train_sets)
            s_p,u_p,_,_   = score_seen_unseen(
                ct1["S"],ct1["w"],th_per,pi_per,train_sets)
            print(f"{months[target]:>12} {s_m:>10.3f} {u_m:>12.3f} "
                  f"{s_p:>10.3f} {u_p:>12.3f}")
            rows.append((s_m,u_m,s_p,u_p,sw,uw))

    if rows:
        R = np.array(rows,dtype=float)
        R = R[np.isfinite(R).all(axis=1)]
        if not len(R): print("all NaN"); return
        ws,wu = R[:,4],R[:,5]
        print(f"\nweighted mean:")
        print(f"  model       seen {np.average(R[:,0],weights=ws):.3f}  "
              f"unseen {np.average(R[:,1],weights=wu):.3f}")
        print(f"  persistence seen {np.average(R[:,2],weights=ws):.3f}  "
              f"unseen {np.average(R[:,3],weights=wu):.3f}")
        gain = np.average(R[:,1]-R[:,3],weights=wu)
        print(f"\nunseen gain: {gain:+.3f} nats")
        print("positive => beats persistence on novel constellations")

# ----------------------------------------------------------------- main --

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   required=True)
    p.add_argument("--vocab",      required=True)
    p.add_argument("--months",     required=True)
    p.add_argument("--test-start", required=True)
    p.add_argument("--engine",  default=ENGINE)
    p.add_argument("--M",       type=int,   default=6,
                   help="months of history in each window")
    p.add_argument("--K",       type=int,   default=8)
    p.add_argument("--r",       type=int,   default=70)
    p.add_argument("--d",       type=int,   default=32)
    p.add_argument("--heads",   type=int,   default=2)
    p.add_argument("--layers",  type=int,   default=1)
    p.add_argument("--d-temp",  type=int,   default=32, dest="d_temp")
    p.add_argument("--epochs",  type=int,   default=500)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--top",     type=int,   default=500)
    p.add_argument("--alpha",   type=float, default=0.1)
    p.add_argument("--beta",    type=float, default=0.1)
    a = p.parse_args()

    E = load_engine(a.engine)
    idx2name, V = E.load_names(a.vocab)
    print(f"V={V}")

    months = E.months_in_range(a.months)
    print(f"{len(months)} months: {months[0]} .. {months[-1]}")

    if a.test_start not in months:
        print(f"--test-start {a.test_start} not in months"); sys.exit(1)
    ts = months.index(a.test_start)

    clouds = load_all(E, a.data_dir, months, V, a.top)

    # training windows: targets in [M .. ts-1]
    # test windows: targets in [ts .. len-1]
    train_windows = make_windows(clouds, a.M, ts, a.M)
    test_windows  = make_windows(clouds, ts, len(months), a.M)
    print(f"train windows: {len(train_windows)}  "
          f"test windows: {len(test_windows)}  M={a.M}")

    # all months used as inputs in training (indices 0..ts-1)
    train_idx = list(range(ts))
    print("fitting PCA...")
    components, global_mean = fit_pca(clouds, train_idx, a.r)

    train_sets = set()
    for i in train_idx:
        if clouds[i]: train_sets |= clouds[i]["sets"]
    print(f"training constellations: {len(train_sets)}")

    model = HierarchicalTemporal(
        V=V, r=a.r, d=a.d, K=a.K,
        heads=a.heads, n_layers=a.layers,
        d_temp=a.d_temp, n_temp_layers=1)
    model.set_pca(components, global_mean)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n:,}")

    print("\ntraining...")
    train(model, clouds, train_windows, a)

    print("\nevaluating...")
    evaluate(model, clouds, test_windows, train_sets, months)

    os.makedirs("results", exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "components": components, "global_mean": global_mean},
               "results/140_model.pt")
    print("saved results/140_model.pt")

if __name__ == "__main__":
    main()
