#!/usr/bin/env python
"""
139_cloud_to_cloud.py

Input  : weighted binary constellations {(s_i, w_i)} at month t
Output : K smooth profiles {(theta_k, pi_k)} predicting month t+1
Loss   : held-out log-likelihood on t+1 sequences, seen/unseen split

Key design decisions:
- Encoder input is raw binary S (no per-month centering -- binary vectors
  cannot be meaningfully centered per month)
- PCA computed on (S - global_train_mean) across all training constellations
  global_mean is fit on training only and frozen
- Decoder baseline is logit(mu_t): when coeff=0, theta=mu_t=persistence
  This is where centering lives -- on the continuous output theta, not
  the binary input
- Jaccard attention bias uses original binary S

Run:
  python scripts/139_cloud_to_cloud.py \
    --data-dir data/processed/full_data_graphs_withdel \
    --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
    --months 2020-06:2023-06 \
    --test-start 2022-06 \
    --r 70 --K 8 --d 32 --heads 2 --layers 1 --epochs 500 --top 500
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
            if v < V:
                S[i, v] = 1.0
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
        S, w  = recs_to_matrix(recs, V, top)
        sets  = {frozenset(s) for s, _ in recs}
        mu    = (w[:, None] * S).sum(0)    # (V,) population mean this month
        out.append({"S": S, "w": w, "sets": sets, "mu": mu, "ym": ym})
    n = sum(1 for x in out if x is not None)
    print(f"  loaded {n}/{len(months)} months")
    return out

# ----------------------------------------------------------------- PCA ----

def fit_pca(clouds, train_pairs, r, n_sample=200, seed=0):
    """PCA on (S - global_train_mean) across training constellations.
    global_mean is the weighted mean constellation across ALL training months.
    Frozen after fitting -- applied to both train and test at encode time.
    Returns: components (r, V), global_mean (V,)
    """
    rng = np.random.default_rng(seed)

    # compute global training mean (weighted across all training months)
    total_w = 0.0
    global_mean = None
    for t, _ in train_pairs:
        c = clouds[t]
        contrib = (c["w"][:, None] * c["S"]).sum(0)
        global_mean = contrib if global_mean is None else global_mean + contrib
        total_w += 1.0
    global_mean /= total_w    # (V,)

    # sample centered constellations for SVD
    rows = []
    for t, _ in train_pairs:
        c   = clouds[t]
        n   = min(n_sample, len(c["S"]))
        idx = rng.choice(len(c["S"]), size=n, replace=False,
                         p=c["w"] / c["w"].sum())
        rows.append(c["S"][idx] - global_mean)
    X = np.vstack(rows).astype(np.float32)
    print(f"  PCA input: {X.shape[0]} x {X.shape[1]}")
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:r]    # (r, V)
    print(f"  PCA components: {components.shape}")
    return components, global_mean.astype(np.float32)

# --------------------------------------------------------- Jaccard --------

def jaccard_matrix(S_bin):
    """S_bin: (N, V) binary float32 tensor -> J: (N, N).
    Zero-length constellations get J=0 (no similarity to anything)."""
    dot   = S_bin @ S_bin.T
    sizes = S_bin.sum(1)
    union = sizes.unsqueeze(1) + sizes.unsqueeze(0) - dot
    J     = dot / (union + EPS)
    J     = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
    return J

# --------------------------------------------------------------- model ----

class Encoder(nn.Module):
    """Input: (N, r) PCA-projected constellations.
       Jaccard bias computed from original binary S."""
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
        x = self.proj(S_pca)                              # (N, d)
        if self.attn is not None:
            J    = jaccard_matrix(S_bin)                  # (N, N)
            # clamp bias to prevent attention logits going to -inf
            bias = torch.clamp(self.lam * J, -10.0, 10.0)
            x_att = self.attn(x.unsqueeze(0),
                              mask=bias).squeeze(0)
            # if attention collapsed to NaN fall back to unattended x
            x = x_att if torch.isfinite(x_att).all() else x
        u = (w.unsqueeze(-1) * x).sum(0)                 # (d,)
        u = torch.nan_to_num(u, nan=0.0)                 # final guard
        return self.out(u)


class Decoder(nn.Module):
    """u -> K PCA coefficient vectors + K weights.
    Baseline: logit(mu_t) so zero coefficients = persistence."""
    def __init__(self, d, K, r):
        super().__init__()
        self.K = K; self.r = r
        self.net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, K*r + K)
        )

    def forward(self, u, P, logit_mu_t):
        """
        P          : (r, V) frozen PCA components
        logit_mu_t : (V,)   logit of current month's population mean
                             baseline: coeff=0 => theta=mu_t=persistence
        """
        out   = self.net(u)
        coeff = out[:self.K * self.r].view(self.K, self.r)  # (K, r)
        pi    = torch.softmax(out[self.K * self.r:], dim=0) # (K,)
        # reconstruct profiles: deviation in PCA space + persistence baseline
        theta_dev = coeff @ P                                # (K, V)
        theta     = torch.sigmoid(theta_dev +
                                  logit_mu_t.unsqueeze(0))  # (K, V)
        return theta, pi


class CloudToCloud(nn.Module):
    def __init__(self, V, r, d, K, heads, n_layers):
        super().__init__()
        self.enc = Encoder(r, d, heads, n_layers)
        self.dec = Decoder(d, K, r)
        self.register_buffer("P",           torch.zeros(r, V))
        self.register_buffer("global_mean", torch.zeros(V))

    def set_pca(self, components, global_mean):
        self.P.copy_(torch.tensor(components, dtype=torch.float32))
        self.global_mean.copy_(torch.tensor(global_mean, dtype=torch.float32))

    def forward(self, S_bin, w, logit_mu_t):
        """
        S_bin      : (N, V) raw binary constellations
        w          : (N,)   frequencies
        logit_mu_t : (V,)   logit of current month's population mean
        """
        # project to PCA space using frozen global_mean and P
        S_pca = (S_bin - self.global_mean) @ self.P.T   # (N, r)
        u     = self.enc(S_pca, S_bin, w)
        return self.dec(u, self.P, logit_mu_t)

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
    """Penalize pairwise cosine similarity between profiles.
    Forces decoder to use all K components distinctly. theta: (K, V)"""
    normed = theta / (theta.norm(dim=1, keepdim=True) + EPS)
    sim    = normed @ normed.T
    mask   = 1 - torch.eye(theta.shape[0], device=theta.device)
    return (sim * mask).sum() / (mask.sum() + EPS)

def entropy_loss(pi):
    """Negative entropy of pi. Minimizing pushes mass to spread across K."""
    return (pi * torch.log(pi + EPS)).sum()

def nearest_constellation_loss(theta, S_t):
    """Pull each predicted profile toward its nearest observed constellation at t.
    Prevents decoder outputting arbitrary continuous profiles disconnected from data.
    theta: (K, V) predicted profiles
    S_t:   (N, V) observed binary constellations at t
    Returns mean distance from each component to its nearest constellation.
    """
    dists    = torch.cdist(theta, S_t)           # (K, N) Euclidean distances
    min_dist = dists.min(dim=1).values           # (K,) nearest per component
    return min_dist.mean()

# ------------------------------------------------ seen / unseen ----------

def score_seen_unseen(S_next, w_next, theta_np, pi_np, train_sets):
    th     = np.clip(theta_np, 1e-3, 1-1e-3)   # harder clamp to kill NaNs
    log_pi = np.log(pi_np + EPS)
    lt, lf = np.log(th), np.log1p(-th)
    base   = lf.sum(1)
    seen_ll = unseen_ll = seen_w = unseen_w = 0.0
    for row, wi in zip(S_next, w_next):
        idx  = np.flatnonzero(row)
        ll_i = float(logsumexp(log_pi + base +
                               (lt[:, idx] - lf[:, idx]).sum(1)))
        if not np.isfinite(ll_i):
            continue
        fs = frozenset(idx.tolist())
        if fs in train_sets:
            seen_ll   += wi * ll_i; seen_w   += wi
        else:
            unseen_ll += wi * ll_i; unseen_w += wi
    seen   = seen_ll   / seen_w   if seen_w   > 0 else float('nan')
    unseen = unseen_ll / unseen_w if unseen_w > 0 else float('nan')
    return seen, unseen, seen_w, unseen_w

def persistence_mixture(S_t, w_t):
    """Single-component mixture = marginal frequencies at t."""
    mu = (w_t[:, None] * S_t).sum(0, keepdims=True)
    return np.clip(mu, EPS, 1-EPS), np.array([1.0])

def safe_logit(mu, eps=1e-3):
    """Clamp at 1e-3 to keep logit in [-6.9, +6.9] -- no overflow."""
    mu = np.clip(mu, eps, 1-eps)
    return np.log(mu / (1-mu)).astype(np.float32)

# --------------------------------------------------------------- train ---

def train(model, clouds, train_pairs, a):
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-3)
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for t, t1 in train_pairs:
            ct, ct1 = clouds[t], clouds[t1]
            S_bin      = torch.tensor(ct["S"],           dtype=torch.float32)
            w_t        = torch.tensor(ct["w"],           dtype=torch.float32)
            logit_mu_t = torch.tensor(safe_logit(ct["mu"]), dtype=torch.float32)
            S_n        = torch.tensor(ct1["S"],          dtype=torch.float32)
            w_n        = torch.tensor(ct1["w"],          dtype=torch.float32)

            theta, pi = model(S_bin, w_t, logit_mu_t)
            ll   = -loglik_th(S_n, w_n, theta, pi)
            div  = diversity_loss(theta)
            ent  = entropy_loss(pi)
            nn   = nearest_constellation_loss(theta, S_bin)
            loss = ll + a.alpha * div + a.beta * ent + a.gamma * nn
            opt.zero_grad(); loss.backward(); opt.step()
            total += ll.item()          # track only likelihood, not penalties

        if (epoch+1) % 50 == 0:
            with torch.no_grad():
                # compute diversity and entropy on last batch for display
                d_val = diversity_loss(theta).item()
                e_val = entropy_loss(pi).item()
            with torch.no_grad():
                nn_val = nearest_constellation_loss(theta, S_bin).item()
            print(f"  epoch {epoch+1:4d}  "
                  f"ll/pair {total/len(train_pairs):.4f}  "
                  f"div={d_val:.3f}  ent={e_val:.3f}  "
                  f"nn={nn_val:.3f}  "
                  f"lam={model.enc.lam.item():.3f}")

# ------------------------------------------------------------- evaluate --

def evaluate(model, clouds, test_pairs, train_sets, months):
    model.eval()
    rows = []
    print(f"\n{'month_t+1':>12} {'mdl_seen':>10} {'mdl_unseen':>12} "
          f"{'per_seen':>10} {'per_unseen':>12}")
    with torch.no_grad():
        for t, t1 in test_pairs:
            ct, ct1 = clouds[t], clouds[t1]
            S_bin      = torch.tensor(ct["S"],              dtype=torch.float32)
            w_t        = torch.tensor(ct["w"],              dtype=torch.float32)
            logit_mu_t = torch.tensor(safe_logit(ct["mu"]), dtype=torch.float32)

            theta, pi  = model(S_bin, w_t, logit_mu_t)
            th_np, pi_np = theta.numpy(), pi.numpy()
            if not np.isfinite(th_np).all():
                n_nan = (~np.isfinite(th_np)).sum()
                print(f"  WARNING {months[t1]}: {n_nan} NaN in theta "
                      f"-- replacing with mu_t")
                mu_t = ct["mu"][None, :]             # (1, V) fallback
                th_np = np.where(np.isfinite(th_np), th_np,
                                 np.clip(mu_t, 1e-3, 1-1e-3))
            th_per, pi_per = persistence_mixture(ct["S"], ct["w"])

            s_m, u_m, sw, uw = score_seen_unseen(
                ct1["S"], ct1["w"], th_np,  pi_np,  train_sets)
            s_p, u_p,  _,  _ = score_seen_unseen(
                ct1["S"], ct1["w"], th_per, pi_per, train_sets)
            print(f"{months[t1]:>12} {s_m:>10.3f} {u_m:>12.3f} "
                  f"{s_p:>10.3f} {u_p:>12.3f}")
            rows.append((s_m, u_m, s_p, u_p, sw, uw))

    if rows:
        R    = np.array(rows, dtype=float)
        good = np.isfinite(R).all(axis=1)
        R    = R[good]
        if len(R) == 0:
            print("all NaN"); return
        ws, wu = R[:,4], R[:,5]
        print(f"\nweighted mean (finite months):")
        print(f"  model       seen {np.average(R[:,0],weights=ws):.3f}  "
              f"unseen {np.average(R[:,1],weights=wu):.3f}")
        print(f"  persistence seen {np.average(R[:,2],weights=ws):.3f}  "
              f"unseen {np.average(R[:,3],weights=wu):.3f}")
        gain = np.average(R[:,1]-R[:,3], weights=wu)
        print(f"\nunseen gain (model - persistence): {gain:+.3f} nats")
        print("positive => model beats persistence on novel constellations")

# ----------------------------------------------------------------- main --

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   required=True)
    p.add_argument("--vocab",      required=True)
    p.add_argument("--months",     required=True)
    p.add_argument("--test-start", required=True)
    p.add_argument("--engine", default=ENGINE)
    p.add_argument("--r",      type=int,   default=70)
    p.add_argument("--K",      type=int,   default=8)
    p.add_argument("--d",      type=int,   default=32)
    p.add_argument("--heads",  type=int,   default=2)
    p.add_argument("--layers", type=int,   default=1)
    p.add_argument("--epochs", type=int,   default=500)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--top",    type=int,   default=500)
    p.add_argument("--alpha",  type=float, default=0.1,
                   help="diversity loss weight")
    p.add_argument("--beta",   type=float, default=0.1,
                   help="entropy loss weight")
    p.add_argument("--gamma",  type=float, default=0.01,
                   help="nearest constellation regularization weight")
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

    train_pairs = [(i, i+1) for i in range(ts-1)
                   if clouds[i] and clouds[i+1]]
    test_pairs  = [(i, i+1) for i in range(ts-1, len(months)-1)
                   if clouds[i] and clouds[i+1]]
    print(f"train pairs: {len(train_pairs)}  test pairs: {len(test_pairs)}")

    print("fitting PCA...")
    components, global_mean = fit_pca(clouds, train_pairs, r=a.r)

    train_sets = set()
    for i in range(ts):
        if clouds[i]:
            train_sets |= clouds[i]["sets"]
    print(f"training constellations: {len(train_sets)}")

    model = CloudToCloud(V=V, r=a.r, d=a.d, K=a.K,
                         heads=a.heads, n_layers=a.layers)
    model.set_pca(components, global_mean)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")

    print("\ntraining...")
    train(model, clouds, train_pairs, a)

    print("\nevaluating...")
    evaluate(model, clouds, test_pairs, train_sets, months)

    os.makedirs("results", exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "components": components, "global_mean": global_mean},
               "results/139_model.pt")
    print("saved results/139_model.pt")

if __name__ == "__main__":
    main()
