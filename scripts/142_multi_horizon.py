#!/usr/bin/env python
"""
142_multi_horizon.py

141 + multi-horizon training (h=1,2,3,6) + M=12 input window.
One model, one encoder, h-conditioned decoder.

Changes from 141:
  - Decoder conditioned on horizon h via learned h-embedding
  - Training at h=1,2,3,6 simultaneously
  - M=12 months of input history
  - Evaluation table per horizon

Run:
  python scripts/142_multi_horizon.py \
    --data-dir data/processed/full_data_graphs_withdel \
    --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
    --months 2020-06:2023-06 --test-start 2022-06 \
    --M 12 --K 8 --r 70 --d 32 --heads 2 --layers 1 \
    --d-temp 32 --epochs-s1 500 --epochs-s2 300 --top 500
"""

import argparse, importlib.util, sys, os
import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp

ENGINE  = "scripts/110_hierarchical_birthdeath_v2_fixed.py"
EPS     = 1e-6
HORIZONS = [1, 2, 3, 6]

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
    """Identical to 141."""
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
        return self.out(torch.nan_to_num(u, nan=0.0))


class CausalTemporalModel(nn.Module):
    """Identical to 141. Zero-initialized output."""
    def __init__(self, d, d_temp=32, nhead=2, n_layers=1):
        super().__init__()
        self.proj_in  = nn.Linear(d, d_temp)
        layer = nn.TransformerEncoderLayer(
            d_model=d_temp, nhead=nhead, dim_feedforward=d_temp*2,
            dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_temp, d)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def causal_mask(self, M, device):
        mask = torch.triu(torch.ones(M, M, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, us):
        if us.shape[0] > 1:
            diffs = us[1:] - us[:-1]
        else:
            diffs = torch.zeros(1, us.shape[1], device=us.device)
        M  = diffs.shape[0]
        x  = self.proj_in(diffs)
        cm = self.causal_mask(M, us.device)
        x  = self.transformer(x.unsqueeze(0), mask=cm).squeeze(0)
        return us[-1] + self.proj_out(x[-1])


class Decoder(nn.Module):
    """H-conditioned decoder. Learns different output shape per horizon."""
    def __init__(self, d, K, r, horizons=HORIZONS):
        super().__init__()
        self.K = K; self.r = r
        self.h_to_idx = {h: i for i, h in enumerate(horizons)}
        # learned horizon embedding -- shifts u before decoding
        self.h_embed  = nn.Embedding(len(horizons), d)
        nn.init.zeros_(self.h_embed.weight)   # zero init: at start h has no effect
        self.net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, K*r + K)
        )

    def forward(self, u, P, logit_mu_t, h):
        h_idx  = torch.tensor(self.h_to_idx[h], device=u.device)
        u_h    = u + self.h_embed(h_idx)          # condition on horizon
        out    = self.net(u_h)
        coeff  = out[:self.K * self.r].view(self.K, self.r)
        pi     = torch.softmax(out[self.K * self.r:], dim=0)
        theta  = torch.sigmoid(coeff @ P + logit_mu_t.unsqueeze(0))
        return theta, pi


class MultiHorizonModel(nn.Module):
    def __init__(self, V, r, d, K, heads, n_layers, d_temp):
        super().__init__()
        self.enc  = Encoder(r, d, heads, n_layers)
        self.temp = CausalTemporalModel(d, d_temp)
        self.dec  = Decoder(d, K, r)
        self.register_buffer("P",           torch.zeros(r, V))
        self.register_buffer("global_mean", torch.zeros(V))

    def set_pca(self, components, global_mean):
        self.P.copy_(torch.tensor(components,   dtype=torch.float32))
        self.global_mean.copy_(torch.tensor(global_mean, dtype=torch.float32))

    def encode(self, S_bin, w):
        S_pca = (S_bin - self.global_mean) @ self.P.T
        return self.enc(S_pca, S_bin, w)

    def forward_single(self, S_bin, w, logit_mu_t, h=1):
        """Stage 1: single month, h=1."""
        u = self.encode(S_bin, w)
        return self.dec(u, self.P, logit_mu_t, h)

    def forward_window(self, window, logit_mu_t, h=1):
        """Stage 2+: M-month window, any h."""
        us     = torch.stack([self.encode(d["S_bin"], d["w"])
                              for d in window])
        u_next = self.temp(us)
        return self.dec(u_next, self.P, logit_mu_t, h)

# ------------------------------------------------------------ loss --------

def loglik_th(S_th, w_th, theta, pi):
    th    = theta.clamp(EPS, 1-EPS)
    lt    = torch.log(th); lf = torch.log1p(-th)
    base  = lf.sum(1)
    delta = S_th @ (lt - lf).T
    lp    = (torch.log(pi + EPS) + base).unsqueeze(0) + delta
    return (w_th * torch.logsumexp(lp, dim=1)).sum()

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

def make_windows(clouds, M, horizons, start, end):
    """Returns list of (input_indices, target_idx, h)."""
    wins = []
    for target in range(start, end):
        for h in horizons:
            i = target - h          # last input month
            inputs = list(range(i - M + 1, i + 1))
            if inputs[0] < 0: continue
            if any(clouds[j] is None for j in inputs): continue
            if clouds[target] is None: continue
            wins.append((inputs, target, h))
    return wins

# --------------------------------------------------------------- train ---

def _step(model, clouds, inputs, target, h, opt, a, use_window=True):
    ct_t = clouds[inputs[-1]]
    ct1  = clouds[target]
    S_bin = torch.tensor(ct_t["S"], dtype=torch.float32)
    w_t   = torch.tensor(ct_t["w"], dtype=torch.float32)
    lmu   = torch.tensor(safe_logit(ct_t["mu"]), dtype=torch.float32)
    S_n   = torch.tensor(ct1["S"], dtype=torch.float32)
    w_n   = torch.tensor(ct1["w"], dtype=torch.float32)

    if use_window:
        window = [{"S_bin": torch.tensor(clouds[i]["S"], dtype=torch.float32),
                   "w":     torch.tensor(clouds[i]["w"], dtype=torch.float32)}
                  for i in inputs]
        theta, pi = model.forward_window(window, lmu, h)
    else:
        theta, pi = model.forward_single(S_bin, w_t, lmu, h=1)

    loss = (-loglik_th(S_n, w_n, theta, pi)
            + a.alpha * diversity_loss(theta)
            + a.beta  * entropy_loss(pi))
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def train(model, clouds, train_wins_s1, train_wins_s2, a):
    # Stage 1: encoder + decoder, h=1 only (= 141/139)
    print("  stage 1: encoder + decoder, h=1 only...")
    for n, p in model.named_parameters():
        p.requires_grad = "enc" in n or "dec" in n
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=a.lr, weight_decay=1e-3)
    for epoch in range(a.epochs_s1):
        total = 0.0
        for inputs, target, h in [train_wins_s1[i] for i in
                                   np.random.permutation(len(train_wins_s1))]:
            total += _step(model, clouds, inputs, target, h, opt, a,
                           use_window=False)
        if (epoch+1) % 100 == 0:
            print(f"    [s1] epoch {epoch+1:4d}  "
                  f"ll/pair {total/len(train_wins_s1):.4f}  "
                  f"lam={model.enc.lam.item():.3f}")

    # Stage 2: causal transformer + h-embedding, all horizons
    print("  stage 2: causal transformer + h-embedding, h=1,2,3,6...")
    for n, p in model.named_parameters():
        p.requires_grad = "temp" in n or "h_embed" in n
    opt2 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=1e-3)
    for epoch in range(a.epochs_s2):
        total = 0.0
        for inputs, target, h in [train_wins_s2[i] for i in
                                   np.random.permutation(len(train_wins_s2))]:
            total += _step(model, clouds, inputs, target, h, opt2, a,
                           use_window=True)
        if (epoch+1) % 100 == 0:
            print(f"    [s2] epoch {epoch+1:4d}  "
                  f"ll/win {total/len(train_wins_s2):.4f}")

# ------------------------------------------------------------- evaluate --

def evaluate(model, clouds, test_wins, train_sets, months):
    model.eval()
    # group by horizon
    by_h = {h: [] for h in HORIZONS}
    for inputs, target, h in test_wins:
        by_h[h].append((inputs, target))

    print(f"\n{'h':>4} {'month_t+h':>12} {'mdl_seen':>10} {'mdl_unseen':>12} "
          f"{'per_seen':>10} {'per_unseen':>12}")
    summary = {}
    with torch.no_grad():
        for h in HORIZONS:
            rows = []
            for inputs, target in by_h[h]:
                ct_t = clouds[inputs[-1]]
                ct1  = clouds[target]
                window = [{"S_bin": torch.tensor(clouds[i]["S"],
                           dtype=torch.float32),
                           "w":     torch.tensor(clouds[i]["w"],
                           dtype=torch.float32)}
                          for i in inputs]
                lmu = torch.tensor(safe_logit(ct_t["mu"]),
                                   dtype=torch.float32)
                theta, pi = model.forward_window(window, lmu, h)
                th_np = theta.numpy(); pi_np = pi.numpy()
                if not np.isfinite(th_np).all():
                    th_np = np.nan_to_num(th_np, nan=0.5)
                th_per, pi_per = persistence_mixture(ct_t["S"], ct_t["w"])
                s_m,u_m,sw,uw = score_seen_unseen(
                    ct1["S"],ct1["w"],th_np,pi_np,train_sets)
                s_p,u_p,_,_   = score_seen_unseen(
                    ct1["S"],ct1["w"],th_per,pi_per,train_sets)
                print(f"{h:>4} {months[target]:>12} "
                      f"{s_m:>10.3f} {u_m:>12.3f} "
                      f"{s_p:>10.3f} {u_p:>12.3f}")
                rows.append((s_m,u_m,s_p,u_p,sw,uw))

            if rows:
                R = np.array(rows,dtype=float)
                R = R[np.isfinite(R).all(axis=1)]
                if len(R):
                    ws,wu = R[:,4],R[:,5]
                    gain  = np.average(R[:,1]-R[:,3],weights=wu)
                    print(f"  h={h} weighted mean: "
                          f"model unseen {np.average(R[:,1],weights=wu):.3f}  "
                          f"persistence unseen {np.average(R[:,3],weights=wu):.3f}  "
                          f"gain {gain:+.3f}")
                    summary[h] = gain
            print()

    print("=== unseen gain summary ===")
    for h in HORIZONS:
        g = summary.get(h, float('nan'))
        print(f"  h={h}: {g:+.3f} nats  "
              f"{'BEATS persistence' if g > 0 else 'loses to persistence'}")

# ----------------------------------------------------------------- main --

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True)
    p.add_argument("--vocab",       required=True)
    p.add_argument("--months",      required=True)
    p.add_argument("--test-start",  required=True)
    p.add_argument("--engine",      default=ENGINE)
    p.add_argument("--M",           type=int,   default=12)
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

    # stage 1: h=1 only, M=1 (single month, identical to 139)
    train_wins_s1 = make_windows(clouds, 1, [1], a.M, ts)
    # stage 2: all horizons, M=12
    train_wins_s2 = make_windows(clouds, a.M, HORIZONS, a.M, ts)
    # test: all horizons, M=12
    test_wins     = make_windows(clouds, a.M, HORIZONS, ts, len(months))

    print(f"train windows s1 (h=1, M=1): {len(train_wins_s1)}")
    print(f"train windows s2 (h=1,2,3,6, M={a.M}): {len(train_wins_s2)}")
    print(f"test windows (all h, M={a.M}): {len(test_wins)}")

    train_idx = list(range(ts))
    print("fitting PCA...")
    components, global_mean = fit_pca(clouds, train_idx, a.r)

    train_sets = set()
    for i in train_idx:
        if clouds[i]: train_sets |= clouds[i]["sets"]
    print(f"training constellations: {len(train_sets)}")

    model = MultiHorizonModel(V=V, r=a.r, d=a.d, K=a.K,
                               heads=a.heads, n_layers=a.layers,
                               d_temp=a.d_temp)
    model.set_pca(components, global_mean)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\ntraining...")
    train(model, clouds, train_wins_s1, train_wins_s2, a)

    print("\nevaluating...")
    evaluate(model, clouds, test_wins, train_sets, months)

    os.makedirs("results", exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "components": components, "global_mean": global_mean},
               "results/142_model.pt")
    print("saved results/142_model.pt")

if __name__ == "__main__":
    main()
