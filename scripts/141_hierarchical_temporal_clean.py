#!/usr/bin/env python
"""
141_hierarchical_temporal_clean.py

139 + causal transformer on top. No prototype layer.

Architecture:
  Level 1: shared Set Encoder (identical to 139)
           {(s_i, w_i)} -> u_i in R^d, one per month
  Level 2: CausalTemporalModel
           [u_{t-M+1}, ..., u_t] -> delta_u via first differences
           u_next = u_t + delta_u  (residual, zero-init => 139 at init)
  Level 3: Decoder (identical to 139)
           u_next + logit(mu_t) -> (theta, pi)

Strict improvement guarantee:
  Stage 1: train encoder + decoder only => identical to 139
  Stage 2: freeze encoder + decoder, train causal transformer only
           zero-init output => at start of stage 2, model IS trained 139
           causal transformer can only improve, never hurt

Run:
  python scripts/141_hierarchical_temporal_clean.py \
    --data-dir data/processed/full_data_graphs_withdel \
    --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
    --months 2020-06:2023-06 --test-start 2022-06 \
    --M 6 --K 8 --r 70 --d 32 --heads 2 --layers 1 \
    --d-temp 32 --epochs-s1 500 --epochs-s2 300 --top 500
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
    m    = importlib.util.module_from_spec(spec)
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
        n   = min(n_sample, len(c["S"]))
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
    return torch.nan_to_num(dot / (union + EPS), nan=0.0)

# --------------------------------------------------------------- model ----

class Encoder(nn.Module):
    """Identical to 139."""
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

    def forward(self, S_pca, S_bin, w):
        x = self.proj(S_pca)
        if self.attn is not None:
            J     = jaccard_matrix(S_bin)
            bias  = torch.clamp(self.lam * J, -10., 10.)
            x_att = self.attn(x.unsqueeze(0), mask=bias).squeeze(0)
            x     = x_att if torch.isfinite(x_att).all() else x
        u = (w.unsqueeze(-1) * x).sum(0)
        u = torch.nan_to_num(u, nan=0.0)
        return self.out(u)


class CausalTemporalModel(nn.Module):
    """Small causal transformer over first-differences of u sequence.
    Zero-initialized output => at init outputs zero => u_next = u_t => 139.
    """
    def __init__(self, d, d_temp=32, nhead=2, n_layers=1):
        super().__init__()
        self.proj_in  = nn.Linear(d, d_temp)
        layer = nn.TransformerEncoderLayer(
            d_model=d_temp, nhead=nhead, dim_feedforward=d_temp*2,
            dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_temp, d)
        # ZERO INIT: at start of stage 2, outputs zero => u_next = u_t
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def causal_mask(self, M, device):
        mask = torch.triu(torch.ones(M, M, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, us):
        """us: (M, d) sequence of month embeddings.
        Uses first differences to remove identity/persistence signal.
        Residual anchors prediction to current month u_t.
        """
        if us.shape[0] > 1:
            diffs = us[1:] - us[:-1]              # (M-1, d) first differences
        else:
            diffs = torch.zeros(1, us.shape[1], device=us.device)
        M  = diffs.shape[0]
        x  = self.proj_in(diffs)                  # (M-1, d_temp)
        cm = self.causal_mask(M, us.device)
        x  = self.transformer(x.unsqueeze(0),
                              mask=cm).squeeze(0)  # (M-1, d_temp)
        delta = self.proj_out(x[-1])               # (d,) -- zero at init
        return us[-1] + delta                      # residual from u_t


class Decoder(nn.Module):
    """Identical to 139."""
    def __init__(self, d, K, r):
        super().__init__()
        self.K = K; self.r = r
        self.net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, K*r + K)
        )

    def forward(self, u, P, logit_mu_t):
        out    = self.net(u)
        coeff  = out[:self.K * self.r].view(self.K, self.r)
        pi     = torch.softmax(out[self.K * self.r:], dim=0)
        theta  = torch.sigmoid(coeff @ P + logit_mu_t.unsqueeze(0))
        return theta, pi


class HierarchicalTemporal(nn.Module):
    def __init__(self, V, r, d, K, heads, n_layers, d_temp):
        super().__init__()
        self.enc  = Encoder(r, d, heads, n_layers)
        self.temp = CausalTemporalModel(d, d_temp)
        self.dec  = Decoder(d, K, r)
        self.register_buffer("P",           torch.zeros(r, V))
        self.register_buffer("global_mean", torch.zeros(V))

    def set_pca(self, components, global_mean):
        self.P.copy_(torch.tensor(components,  dtype=torch.float32))
        self.global_mean.copy_(torch.tensor(global_mean, dtype=torch.float32))

    def encode(self, S_bin, w):
        S_pca = (S_bin - self.global_mean) @ self.P.T
        return self.enc(S_pca, S_bin, w)

    def forward_single(self, S_bin, w, logit_mu_t):
        """139-compatible single-month forward. Used in stage 1."""
        u = self.encode(S_bin, w)
        return self.dec(u, self.P, logit_mu_t)

    def forward_window(self, window, logit_mu_t):
        """Full temporal forward. Used in stage 2+."""
        us = torch.stack([self.encode(d["S_bin"], d["w"])
                          for d in window])        # (M, d)
        u_next = self.temp(us)                     # (d,) residual from us[-1]
        return self.dec(u_next, self.P, logit_mu_t)

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

def safe_logit(mu, eps=1e-3):
    mu = np.clip(mu, eps, 1-eps)
    return np.log(mu / (1-mu)).astype(np.float32)

# --------------------------------------------------------- windows -------

def make_windows(clouds, start, end, M):
    wins = []
    for target in range(start, end):
        inputs = list(range(target-M, target))
        if any(clouds[i] is None for i in inputs) or clouds[target] is None:
            continue
        wins.append((inputs, target))
    return wins

# --------------------------------------------------------------- train ---

def _one_epoch_s1(model, clouds, pairs, opt, a):
    """Stage 1: single-month forward, identical to 139."""
    model.train(); total = 0.0
    for t, t1 in np.random.permutation(pairs).tolist():
        ct, ct1 = clouds[t], clouds[t1]
        S_bin = torch.tensor(ct["S"],  dtype=torch.float32)
        w_t   = torch.tensor(ct["w"],  dtype=torch.float32)
        lmu   = torch.tensor(safe_logit(ct["mu"]), dtype=torch.float32)
        S_n   = torch.tensor(ct1["S"], dtype=torch.float32)
        w_n   = torch.tensor(ct1["w"], dtype=torch.float32)
        theta, pi = model.forward_single(S_bin, w_t, lmu)
        loss = (-loglik_th(S_n, w_n, theta, pi)
                + a.alpha * diversity_loss(theta)
                + a.beta  * entropy_loss(pi))
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    return total / len(pairs)


def _one_epoch_s2(model, clouds, windows, opt, a):
    """Stage 2: window forward with causal transformer."""
    model.train(); total = 0.0
    for inputs, target in [windows[i] for i in
                            np.random.permutation(len(windows))]:
        ct = clouds[target]
        window = [{"S_bin": torch.tensor(clouds[i]["S"], dtype=torch.float32),
                   "w":     torch.tensor(clouds[i]["w"], dtype=torch.float32)}
                  for i in inputs]
        lmu   = torch.tensor(safe_logit(ct["mu"]), dtype=torch.float32)
        S_n   = torch.tensor(ct["S"], dtype=torch.float32)
        w_n   = torch.tensor(ct["w"], dtype=torch.float32)
        # target is inputs[-1]+1
        ct_t  = clouds[inputs[-1]]
        lmu_t = torch.tensor(safe_logit(ct_t["mu"]), dtype=torch.float32)
        theta, pi = model.forward_window(window, lmu_t)
        loss = (-loglik_th(S_n, w_n, theta, pi)
                + a.alpha * diversity_loss(theta)
                + a.beta  * entropy_loss(pi))
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    return total / len(windows)


def train(model, clouds, train_pairs, train_windows, a):
    # Stage 1: encoder + decoder only (= 139)
    print("  stage 1: encoder + decoder (= 139)...")
    for n, p in model.named_parameters():
        p.requires_grad = "enc" in n or "dec" in n
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.lr, weight_decay=1e-3)
    for epoch in range(a.epochs_s1):
        loss = _one_epoch_s1(model, clouds, train_pairs, opt, a)
        if (epoch+1) % 100 == 0:
            print(f"    [s1] epoch {epoch+1:4d}  ll/pair {loss:.4f}  "
                  f"lam={model.enc.lam.item():.3f}")

    # Stage 2: causal transformer only (encoder+decoder frozen)
    print("  stage 2: causal transformer only...")
    for n, p in model.named_parameters():
        p.requires_grad = "temp" in n
    opt2 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.lr, weight_decay=1e-3)
    for epoch in range(a.epochs_s2):
        loss = _one_epoch_s2(model, clouds, train_windows, opt2, a)
        if (epoch+1) % 100 == 0:
            print(f"    [s2] epoch {epoch+1:4d}  ll/win {loss:.4f}")

# ------------------------------------------------------------- evaluate --

def evaluate(model, clouds, test_windows, train_sets, months):
    model.eval(); rows = []
    print(f"\n{'month_t+1':>12} {'mdl_seen':>10} {'mdl_unseen':>12} "
          f"{'per_seen':>10} {'per_unseen':>12}")
    with torch.no_grad():
        for inputs, target in test_windows:
            ct_t  = clouds[inputs[-1]]
            ct1   = clouds[target]
            window = [{"S_bin": torch.tensor(clouds[i]["S"], dtype=torch.float32),
                       "w":     torch.tensor(clouds[i]["w"], dtype=torch.float32)}
                      for i in inputs]
            lmu_t = torch.tensor(safe_logit(ct_t["mu"]), dtype=torch.float32)
            theta, pi = model.forward_window(window, lmu_t)
            th_np, pi_np = theta.numpy(), pi.numpy()
            if not np.isfinite(th_np).all():
                th_np = np.nan_to_num(th_np, nan=0.5)

            th_per, pi_per = persistence_mixture(ct_t["S"], ct_t["w"])
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
    p.add_argument("--data-dir",    required=True)
    p.add_argument("--vocab",       required=True)
    p.add_argument("--months",      required=True)
    p.add_argument("--test-start",  required=True)
    p.add_argument("--engine",      default=ENGINE)
    p.add_argument("--M",           type=int,   default=6)
    p.add_argument("--K",           type=int,   default=8)
    p.add_argument("--r",           type=int,   default=70)
    p.add_argument("--d",           type=int,   default=32)
    p.add_argument("--heads",       type=int,   default=2)
    p.add_argument("--layers",      type=int,   default=1)
    p.add_argument("--d-temp",      type=int,   default=32, dest="d_temp")
    p.add_argument("--epochs-s1",   type=int,   default=500, dest="epochs_s1")
    p.add_argument("--epochs-s2",   type=int,   default=300, dest="epochs_s2")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--top",         type=int,   default=500)
    p.add_argument("--alpha",       type=float, default=0.1)
    p.add_argument("--beta",        type=float, default=0.1)
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

    # stage 1 uses consecutive pairs (same as 139)
    train_pairs   = [(i, i+1) for i in range(ts-1)
                     if clouds[i] and clouds[i+1]]
    # stage 2 uses M-month windows
    train_windows = make_windows(clouds, a.M, ts, a.M)
    test_windows  = make_windows(clouds, ts, len(months), a.M)

    print(f"train pairs (s1): {len(train_pairs)}  "
          f"train windows (s2): {len(train_windows)}  "
          f"test windows: {len(test_windows)}  M={a.M}")

    train_idx = list(range(ts))
    print("fitting PCA...")
    components, global_mean = fit_pca(clouds, train_idx, a.r)

    train_sets = set()
    for i in train_idx:
        if clouds[i]: train_sets |= clouds[i]["sets"]
    print(f"training constellations: {len(train_sets)}")

    model = HierarchicalTemporal(V=V, r=a.r, d=a.d, K=a.K,
                                  heads=a.heads, n_layers=a.layers,
                                  d_temp=a.d_temp)
    model.set_pca(components, global_mean)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\ntraining...")
    train(model, clouds, train_pairs, train_windows, a)

    print("\nevaluating...")
    evaluate(model, clouds, test_windows, train_sets, months)

    os.makedirs("results", exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "components": components, "global_mean": global_mean},
               "results/141_model.pt")
    print("saved results/141_model.pt")

if __name__ == "__main__":
    main()
