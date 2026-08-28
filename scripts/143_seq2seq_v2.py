#!/usr/bin/env python
"""
143_seq2seq_v2.py  --  revised 143 with:
  1. Novel upweighting: novel constellations get upweight x more gradient
  2. Non-frozen weights: all params trained jointly (stage 1 = warmup, stage 2 = all)
  3. Rolling evaluation: 12-month input window -> 6-month test window
  4. Clean output: training window | test month | h | metrics

Run:
  python scripts/143_seq2seq_v2.py \
    --data-dir data/processed/full_data_graphs_withdel \
    --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
    --months 2020-06:2023-06 --test-start 2022-06 \
    --M 12 --l 6 --K 8 --r 70 --d 32 --heads 2 --layers 1 \
    --d-lstm 64 --epochs-s1 300 --epochs-s2 300 \
    --upweight 10.0 --top 500
"""

import argparse, importlib.util, sys, os
import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp

ENGINE   = "scripts/110_hierarchical_birthdeath_v2_fixed.py"
EPS      = 1e-6
EVAL_H   = [1, 2, 3, 6]

# ---------------------------------------------------------------- engine ---
def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m    = importlib.util.module_from_spec(spec)
    sys.modules["engine"] = m
    spec.loader.exec_module(m)
    return m

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
    gm  = None; total = 0.0; rows = []
    for i in train_idx:
        c = clouds[i]
        gm = (c["w"][:, None] * c["S"]).sum(0) if gm is None \
             else gm + (c["w"][:, None] * c["S"]).sum(0)
        total += 1.0
        n   = min(n_sample, len(c["S"]))
        idx = rng.choice(len(c["S"]), size=n, replace=False,
                         p=c["w"]/c["w"].sum())
        rows.append(c["S"][idx])
    gm /= total
    _, _, Vt = np.linalg.svd(
        np.vstack(rows).astype(np.float32) - gm, full_matrices=False)
    return Vt[:r], gm.astype(np.float32)

# --------------------------------------------------------- Jaccard --------
def jaccard_matrix(S_bin):
    dot = S_bin @ S_bin.T
    sz  = S_bin.sum(1)
    return torch.nan_to_num(
        dot / (sz.unsqueeze(1) + sz.unsqueeze(0) - dot + EPS), nan=0.0)

# --------------------------------------------------------------- model ----
class SetEncoder(nn.Module):
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
            J = jaccard_matrix(S_bin)
            b = torch.clamp(self.lam * J, -10., 10.)
            xa = self.attn(x.unsqueeze(0), mask=b).squeeze(0)
            x  = xa if torch.isfinite(xa).all() else x
        u = (w.unsqueeze(-1) * x).sum(0)
        return self.out(torch.nan_to_num(u, nan=0.0))

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

class AutoregressiveDecoder(nn.Module):
    def __init__(self, d, d_lstm):
        super().__init__()
        self.init_h = nn.Linear(d_lstm * 2, d_lstm)
        self.init_c = nn.Linear(d_lstm * 2, d_lstm)
        self.lstm   = nn.LSTM(d, d_lstm, batch_first=True)
        self.proj   = nn.Linear(d_lstm, d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h_t, context, u_t, l, teacher_us=None):
        hc  = torch.cat([h_t, context])
        h_0 = torch.tanh(self.init_h(hc)).unsqueeze(0).unsqueeze(0)
        c_0 = torch.tanh(self.init_c(hc)).unsqueeze(0).unsqueeze(0)
        us_pred = []; inp = u_t; h, c = h_0, c_0
        for s in range(l):
            out, (h, c) = self.lstm(inp.unsqueeze(0).unsqueeze(0), (h, c))
            u_next = self.proj(out.squeeze())
            us_pred.append(u_next)
            inp = (teacher_us[s] if teacher_us is not None else u_next)
        return torch.stack(us_pred)

class MLPDecoder(nn.Module):
    def __init__(self, d, K, r):
        super().__init__()
        self.K = K; self.r = r
        self.net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, K*r + K))

    def forward(self, u, P, logit_mu_t):
        out   = self.net(u)
        coeff = out[:self.K * self.r].view(self.K, self.r)
        pi    = torch.softmax(out[self.K * self.r:], dim=0)
        return torch.sigmoid(coeff @ P + logit_mu_t.unsqueeze(0)), pi

class Seq2SeqSets(nn.Module):
    def __init__(self, V, r, d, K, heads, n_layers, d_lstm):
        super().__init__()
        self.d        = d
        self.set_enc  = SetEncoder(r, d, heads, n_layers)
        self.temp_enc = TemporalEncoder(d, d_lstm)
        self.dec_lstm = AutoregressiveDecoder(d, d_lstm)
        self.dec_mlp  = MLPDecoder(d, K, r)
        self.register_buffer("P",           torch.zeros(r, V))
        self.register_buffer("global_mean", torch.zeros(V))

    def set_pca(self, components, gm):
        self.P.copy_(torch.tensor(components, dtype=torch.float32))
        self.global_mean.copy_(torch.tensor(gm, dtype=torch.float32))

    def encode_month(self, S_bin, w):
        S_pca = (S_bin - self.global_mean) @ self.P.T
        return self.set_enc(S_pca, S_bin, w)

    def forward_single(self, S_bin, w, logit_mu_t):
        return self.dec_mlp(self.encode_month(S_bin, w), self.P, logit_mu_t)

    def forward_seq2seq(self, window, logit_mu_t, l, teacher_us=None):
        us       = torch.stack([self.encode_month(d["S_bin"], d["w"])
                                for d in window])
        h_t, ctx = self.temp_enc(us)
        us_pred  = self.dec_lstm(h_t, ctx, us[-1], l, teacher_us)
        return [self.dec_mlp(us_pred[s], self.P, logit_mu_t)
                for s in range(l)]

# ------------------------------------------------------------ loss --------
def loglik_th(S_th, w_th, theta, pi):
    th    = theta.clamp(EPS, 1-EPS)
    lt    = torch.log(th); lf = torch.log1p(-th)
    base  = lf.sum(1)
    delta = S_th @ (lt - lf).T
    lp    = (torch.log(pi + EPS) + base).unsqueeze(0) + delta
    return (w_th * torch.logsumexp(lp, dim=1)).sum()

def loglik_upweighted(S_next_th, w_next_th, theta, pi,
                      present_sets, upweight):
    """Upweight novel constellations (absent at input month) in the loss."""
    w_adj = w_next_th.clone()
    for i, row in enumerate(S_next_th):
        fs = frozenset(torch.nonzero(row).squeeze(-1).tolist())
        if fs not in present_sets:
            w_adj[i] = w_adj[i] * upweight
    w_adj = w_adj / w_adj.sum()
    return loglik_th(S_next_th, w_adj, theta, pi)

def diversity_loss(theta):
    n = theta / (theta.norm(dim=1, keepdim=True) + EPS)
    s = n @ n.T
    m = 1 - torch.eye(theta.shape[0], device=theta.device)
    return (s * m).sum() / (m.sum() + EPS)

def entropy_loss(pi):
    return (pi * torch.log(pi + EPS)).sum()

# ------------------------------------------------ seen / unseen ----------
def score_seen_unseen(S_next, w_next, theta_np, pi_np, train_sets):
    th     = np.clip(theta_np, 1e-3, 1-1e-3)
    log_pi = np.log(pi_np + EPS)
    lt, lf = np.log(th), np.log1p(-th)
    base   = lf.sum(1)
    sl = ul = sw = uw = 0.0
    for row, wi in zip(S_next, w_next):
        idx  = np.flatnonzero(row)
        ll_i = float(logsumexp(log_pi + base +
                               (lt[:, idx] - lf[:, idx]).sum(1)))
        if not np.isfinite(ll_i): continue
        fs = frozenset(idx.tolist())
        if fs in train_sets:
            sl += wi*ll_i; sw += wi
        else:
            ul += wi*ll_i; uw += wi
    return (sl/sw if sw > 0 else float('nan'),
            ul/uw if uw > 0 else float('nan'), sw, uw)

def persistence_mixture(S_t, w_t):
    mu = (w_t[:, None] * S_t).sum(0, keepdims=True)
    return np.clip(mu, EPS, 1-EPS), np.array([1.0])

def safe_logit(mu, eps=1e-3):
    mu = np.clip(mu, eps, 1-eps)
    return np.log(mu / (1-mu)).astype(np.float32)

# --------------------------------------------------------- windows -------
def make_windows(clouds, M, l, start, end):
    wins = []
    for i in range(start, end):
        inputs  = list(range(i-M+1, i+1))
        targets = list(range(i+1, i+l+1))
        if inputs[0] < 0: continue
        if any(clouds[j] is None for j in inputs): continue
        if any(j >= len(clouds) or clouds[j] is None for j in targets):
            continue
        wins.append((inputs, targets))
    return wins

# --------------------------------------------------------------- train ---
def train(model, clouds, train_pairs, train_wins, a):
    all_params = list(model.parameters())
    opt = torch.optim.Adam(all_params, lr=a.lr, weight_decay=1e-3)

    # Stage 1: all params, h=1 pairs (warmup)
    print("  stage 1: all params, h=1 warmup...")
    for epoch in range(a.epochs_s1):
        total = 0.0
        for t, t1 in [train_pairs[i] for i in
                      np.random.permutation(len(train_pairs))]:
            ct, ct1 = clouds[t], clouds[t1]
            S_b = torch.tensor(ct["S"],  dtype=torch.float32)
            w_t = torch.tensor(ct["w"],  dtype=torch.float32)
            lmu = torch.tensor(safe_logit(ct["mu"]), dtype=torch.float32)
            S_n = torch.tensor(ct1["S"], dtype=torch.float32)
            w_n = torch.tensor(ct1["w"], dtype=torch.float32)
            pres = ct["sets"]
            theta, pi = model.forward_single(S_b, w_t, lmu)
            ll = -loglik_upweighted(S_n, w_n, theta, pi, pres, a.upweight)
            loss = ll + a.alpha*diversity_loss(theta) + a.beta*entropy_loss(pi)
            opt.zero_grad(); loss.backward(); opt.step()
            total += ll.item()
        if (epoch+1) % 100 == 0:
            print(f"    [s1] epoch {epoch+1:4d}  "
                  f"ll/pair {total/len(train_pairs):.4f}  "
                  f"lam={model.set_enc.lam.item():.3f}")

    # Stage 2: all params, all horizons + novel upweighting
    print("  stage 2: all params, h=1,2,3,6 + novel upweighting...")
    opt2 = torch.optim.Adam(all_params, lr=a.lr*0.3, weight_decay=1e-3)
    for epoch in range(a.epochs_s2):
        total = 0.0; n_wins = 0
        for inputs, targets in [train_wins[i] for i in
                                 np.random.permutation(len(train_wins))]:
            ct_t = clouds[inputs[-1]]
            lmu  = torch.tensor(safe_logit(ct_t["mu"]), dtype=torch.float32)
            pres = ct_t["sets"]
            window = [{"S_bin": torch.tensor(clouds[i]["S"], dtype=torch.float32),
                       "w":     torch.tensor(clouds[i]["w"], dtype=torch.float32)}
                      for i in inputs]
            l = len(targets)
            with torch.no_grad():
                teacher_us = torch.stack([
                    model.encode_month(
                        torch.tensor(clouds[t]["S"], dtype=torch.float32),
                        torch.tensor(clouds[t]["w"], dtype=torch.float32))
                    for t in targets])
            outputs = model.forward_seq2seq(window, lmu, l,
                                            teacher_us=teacher_us)
            loss = sum(
                -loglik_upweighted(
                    torch.tensor(clouds[targets[s]]["S"], dtype=torch.float32),
                    torch.tensor(clouds[targets[s]]["w"], dtype=torch.float32),
                    theta, pi, pres, a.upweight)
                + a.alpha*diversity_loss(theta) + a.beta*entropy_loss(pi)
                for s, (theta, pi) in enumerate(outputs))
            opt2.zero_grad(); loss.backward(); opt2.step()
            total += loss.item() / l; n_wins += 1
        if (epoch+1) % 100 == 0:
            print(f"    [s2] epoch {epoch+1:4d}  ll/step {total/n_wins:.4f}")

# ------------------------------------------------------------- evaluate --
def evaluate(model, clouds, test_wins, train_sets, months, eval_h):
    model.eval()
    # header
    h_cols = " | ".join(f"h={h:2d} mdl_u  per_u   gain" for h in eval_h)
    print(f"\n{'Input window':>22} | {'Test month':>12} | {h_cols}")
    print("-" * (24 + 16 + len(h_cols) + 10))

    gains_by_h = {h: [] for h in eval_h}

    with torch.no_grad():
        for inputs, targets in test_wins:
            ct_t   = clouds[inputs[-1]]
            lmu_t  = torch.tensor(safe_logit(ct_t["mu"]), dtype=torch.float32)
            window = [{"S_bin": torch.tensor(clouds[i]["S"], dtype=torch.float32),
                       "w":     torch.tensor(clouds[i]["w"], dtype=torch.float32)}
                      for i in inputs]
            l       = len(targets)
            outputs = model.forward_seq2seq(window, lmu_t, l,
                                            teacher_us=None)  # free running

            # training window label
            tw_label = f"{months[inputs[0]]}..{months[inputs[-1]]}"  # LSTM context, not train data

            # collect results for all h at this window
            results_by_h = {}
            for s, (theta, pi) in enumerate(outputs):
                h = s + 1
                if h not in eval_h: continue
                target = targets[s]
                if target >= len(clouds) or clouds[target] is None:
                    continue
                ct1   = clouds[target]
                th_np = theta.numpy(); pi_np = pi.numpy()
                if not np.isfinite(th_np).all():
                    th_np = np.nan_to_num(th_np, nan=0.5)
                th_per, pi_per = persistence_mixture(ct_t["S"], ct_t["w"])
                s_m,u_m,sw,uw = score_seen_unseen(
                    ct1["S"],ct1["w"],th_np,pi_np,train_sets)
                s_p,u_p,_,_   = score_seen_unseen(
                    ct1["S"],ct1["w"],th_per,pi_per,train_sets)
                gain = u_m - u_p if (np.isfinite(u_m) and np.isfinite(u_p)) \
                       else float('nan')
                results_by_h[h] = (months[target], u_m, u_p, gain, sw, uw)
                if np.isfinite(gain):
                    gains_by_h[h].append((gain, uw))

            # print one row per test window
            if results_by_h:
                first_h   = eval_h[0]
                test_month = results_by_h.get(first_h, (None,))[0] or ""
                h_vals     = ""
                for h in eval_h:
                    if h in results_by_h:
                        _, u_m, u_p, gain, _, _ = results_by_h[h]
                        h_vals += f" | {u_m:+7.2f} {u_p:+7.2f} {gain:+6.2f}"
                    else:
                        h_vals += " |    n/a     n/a    n/a"
                print(f"{tw_label:>22} | {test_month:>12} |{h_vals}")

    # summary
    print(f"\n{'='*60}")
    print("=== Unseen gain summary (weighted mean across test windows) ===")
    print(f"{'h':>4} {'gain':>8}  {'verdict':>25}")
    for h in eval_h:
        if not gains_by_h[h]:
            print(f"  h={h}: no results"); continue
        gains = np.array([g for g, _ in gains_by_h[h]])
        wts   = np.array([w for _, w in gains_by_h[h]])
        wm    = np.average(gains, weights=wts)
        v     = "BEATS persistence" if wm > 0 else "loses to persistence"
        print(f"  h={h:2d}: {wm:+8.3f} nats  {v}")

# ----------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True)
    p.add_argument("--vocab",       required=True)
    p.add_argument("--months",      required=True)
    p.add_argument("--test-start",  required=True)
    p.add_argument("--engine",      default=ENGINE)
    p.add_argument("--M",     type=int,   default=12)
    p.add_argument("--l",     type=int,   default=6)
    p.add_argument("--K",     type=int,   default=8)
    p.add_argument("--r",     type=int,   default=70)
    p.add_argument("--d",     type=int,   default=32)
    p.add_argument("--heads", type=int,   default=2)
    p.add_argument("--layers",type=int,   default=1)
    p.add_argument("--d-lstm",type=int,   default=64, dest="d_lstm")
    p.add_argument("--epochs-s1", type=int, default=300, dest="epochs_s1")
    p.add_argument("--epochs-s2", type=int, default=300, dest="epochs_s2")
    p.add_argument("--lr",    type=float, default=1e-3)
    p.add_argument("--top",   type=int,   default=500)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta",  type=float, default=0.1)
    p.add_argument("--upweight", type=float, default=10.0)
    a = p.parse_args()

    E = load_engine(a.engine)
    idx2name, V = E.load_names(a.vocab)
    print(f"V={V}")
    months = E.months_in_range(a.months)
    print(f"{len(months)} months: {months[0]} .. {months[-1]}")
    if a.test_start not in months:
        print(f"--test-start not found"); sys.exit(1)
    ts = months.index(a.test_start)

    clouds = load_all(E, a.data_dir, months, V, a.top)

    # training pairs (h=1) and windows (all h)
    train_pairs = [(i, i+1) for i in range(ts-1)
                   if clouds[i] and clouds[i+1]]
    train_wins  = make_windows(clouds, a.M, a.l, a.M-1, ts)
    test_wins   = make_windows(clouds, a.M, a.l, ts, len(months)-a.l)

    print(f"stage 1 pairs: {len(train_pairs)}")
    print(f"stage 2 windows: {len(train_wins)}  test windows: {len(test_wins)}")

    train_idx = list(range(ts))
    print("fitting PCA...")
    components, gm = fit_pca(clouds, train_idx, a.r)

    train_sets = set()
    for i in train_idx:
        if clouds[i]: train_sets |= clouds[i]["sets"]
    print(f"training constellations: {len(train_sets)}")

    model = Seq2SeqSets(V=V, r=a.r, d=a.d, K=a.K,
                        heads=a.heads, n_layers=a.layers,
                        d_lstm=a.d_lstm)
    model.set_pca(components, gm)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\ntraining...")
    train(model, clouds, train_pairs, train_wins, a)

    print("\nevaluating...")
    evaluate(model, clouds, test_wins, train_sets, months, EVAL_H)

    os.makedirs("results", exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "components": components, "global_mean": gm},
               "results/143v2_model.pt")
    print("saved results/143v2_model.pt")

if __name__ == "__main__":
    main()
