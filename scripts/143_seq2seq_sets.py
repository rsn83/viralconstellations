#!/usr/bin/env python
"""
143_seq2seq_sets.py

Hierarchical Set Encoder + Seq2Seq LSTM Decoder.

Level 1 (within-month): Set Transformer with Jaccard attention
         {(s_i, w_i)} -> u_t in R^d   [permutation invariant, shared weights]

Level 2 (across-month): Encoder LSTM over u_1,...,u_t
         -> hidden state h_t capturing population trajectory
         -> attention over all past h_i (what mattered in the past)

Decoder: Autoregressive LSTM in u-space
         h_t + context -> u_{t+1} -> u_{t+2} -> ... -> u_{t+l}
         each u_{t+s} decoded to (theta_{t+s}, pi_{t+s})

Multi-horizon is FREE from rollout:
         evaluate at s=1,2,3,6 without any h-conditioning

Two-stage training:
  Stage 1: set encoder + decoder MLP only (= 139, strict baseline)
  Stage 2: freeze stage 1 weights, train encoder LSTM + decoder LSTM
           LSTM output zero-initialized => at start of s2, model = 139

Run:
  python scripts/143_seq2seq_sets.py \
    --data-dir data/processed/full_data_graphs_withdel \
    --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
    --months 2020-06:2023-06 --test-start 2022-06 \
    --M 12 --l 6 --K 8 --r 70 --d 32 --heads 2 --layers 1 \
    --d-lstm 64 --epochs-s1 500 --epochs-s2 300 --top 500
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
        global_mean = (contrib if global_mean is None
                       else global_mean + contrib)
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

class SetEncoder(nn.Module):
    """Level 1: within-month Set Transformer. Identical to 139."""
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


class TemporalEncoder(nn.Module):
    """Level 2: LSTM over monthly embeddings u_1,...,u_t.
    Captures population trajectory with recurrent hidden state.
    Forget gate learns what to carry forward vs discard -- biologically:
    what mutation background persists vs what lineage has been replaced.
    """
    def __init__(self, d, d_lstm):
        super().__init__()
        self.lstm = nn.LSTM(d, d_lstm, batch_first=True)
        # attention over hidden states
        self.attn_w = nn.Linear(d_lstm, 1)

    def forward(self, us):
        """us: (M, d) sequence of monthly embeddings.
        Returns: h_t (d_lstm,) last hidden, context (d_lstm,) attention summary.
        """
        hs, (h_n, _) = self.lstm(us.unsqueeze(0))  # hs: (1, M, d_lstm)
        hs = hs.squeeze(0)                           # (M, d_lstm)
        # attention over past hidden states
        scores  = self.attn_w(hs).squeeze(-1)        # (M,)
        weights = torch.softmax(scores, dim=0)       # (M,)
        context = (weights.unsqueeze(-1) * hs).sum(0)  # (d_lstm,)
        h_t     = h_n.squeeze(0).squeeze(0)          # (d_lstm,)
        return h_t, context


class AutoregressiveDecoder(nn.Module):
    """Decoder LSTM in u-space. Autoregressively generates
    u_{t+1}, u_{t+2}, ..., u_{t+l} from (h_t, context).
    Each u_{t+s} decoded to (theta, pi) via MLP decoder.
    Zero-initialized output projection => at init u_{t+s} = 0
    => decoder MLP sees logit(mu_t) baseline = persistence.
    """
    def __init__(self, d, d_lstm):
        super().__init__()
        # project (h_t, context) to decoder initial state
        self.init_h = nn.Linear(d_lstm * 2, d_lstm)
        self.init_c = nn.Linear(d_lstm * 2, d_lstm)
        # decoder LSTM: input is previous u, state is d_lstm
        self.lstm   = nn.LSTM(d, d_lstm, batch_first=True)
        # project decoder hidden to u-space
        self.proj   = nn.Linear(d_lstm, d)
        # ZERO INIT: at start of stage 2, proj outputs zero
        # => u_{t+s} = 0 => decoder MLP gets pure logit(mu_t) baseline
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h_t, context, u_t, l, teacher_us=None):
        """
        h_t, context : (d_lstm,) from temporal encoder
        u_t          : (d,) current month embedding -- first decoder input
        l            : number of steps to unroll
        teacher_us   : (l, d) ground truth embeddings for teacher forcing
                       if None: free running (test time)
        Returns: us_pred (l, d) predicted embeddings
        """
        # initialize decoder state from encoder
        hc  = torch.cat([h_t, context])               # (2*d_lstm,)
        h_0 = torch.tanh(self.init_h(hc)).unsqueeze(0).unsqueeze(0)
        c_0 = torch.tanh(self.init_c(hc)).unsqueeze(0).unsqueeze(0)

        us_pred = []
        inp     = u_t                                  # (d,) first input
        h, c    = h_0, c_0

        for s in range(l):
            out, (h, c) = self.lstm(inp.unsqueeze(0).unsqueeze(0),
                                    (h, c))            # out: (1,1,d_lstm)
            u_next = self.proj(out.squeeze())          # (d,) -- zero at init
            us_pred.append(u_next)
            # teacher forcing at train time, free running at test time
            inp = (teacher_us[s] if teacher_us is not None
                   else u_next)

        return torch.stack(us_pred)                   # (l, d)


class MLPDecoder(nn.Module):
    """Maps u + logit(mu_t) -> (theta K x V, pi K).
    Identical to 139's decoder. Baseline = persistence when u=0.
    """
    def __init__(self, d, K, r):
        super().__init__()
        self.K = K; self.r = r
        self.net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, K*r + K)
        )

    def forward(self, u, P, logit_mu_t):
        out   = self.net(u)
        coeff = out[:self.K * self.r].view(self.K, self.r)
        pi    = torch.softmax(out[self.K * self.r:], dim=0)
        theta = torch.sigmoid(coeff @ P + logit_mu_t.unsqueeze(0))
        return theta, pi


class Seq2SeqSets(nn.Module):
    """Full hierarchical set encoder + seq2seq LSTM decoder."""
    def __init__(self, V, r, d, K, heads, n_layers, d_lstm):
        super().__init__()
        self.d = d
        self.set_enc  = SetEncoder(r, d, heads, n_layers)
        self.temp_enc = TemporalEncoder(d, d_lstm)
        self.dec_lstm = AutoregressiveDecoder(d, d_lstm)
        self.dec_mlp  = MLPDecoder(d, K, r)
        self.register_buffer("P",           torch.zeros(r, V))
        self.register_buffer("global_mean", torch.zeros(V))

    def set_pca(self, components, global_mean):
        self.P.copy_(torch.tensor(components,   dtype=torch.float32))
        self.global_mean.copy_(
            torch.tensor(global_mean, dtype=torch.float32))

    def encode_month(self, S_bin, w):
        S_pca = (S_bin - self.global_mean) @ self.P.T
        return self.set_enc(S_pca, S_bin, w)

    def forward_single(self, S_bin, w, logit_mu_t):
        """Stage 1: single month, identical to 139."""
        u = self.encode_month(S_bin, w)
        return self.dec_mlp(u, self.P, logit_mu_t)

    def forward_seq2seq(self, window_data, logit_mu_t, l,
                        teacher_us=None):
        """Full seq2seq forward.
        window_data: list of M dicts with S_bin, w
        l: decode steps (max horizon)
        teacher_us: (l, d) for teacher forcing at train time
        Returns: list of l (theta, pi) tuples
        """
        # encode all months in window
        us = torch.stack([self.encode_month(d["S_bin"], d["w"])
                          for d in window_data])         # (M, d)

        # temporal encoder: LSTM + attention over monthly embeddings
        h_t, context = self.temp_enc(us)                # (d_lstm,), (d_lstm,)

        # autoregressive decoder
        us_pred = self.dec_lstm(h_t, context, us[-1],
                                l, teacher_us)           # (l, d)

        # decode each predicted embedding to (theta, pi)
        outputs = []
        for s in range(l):
            theta, pi = self.dec_mlp(us_pred[s], self.P, logit_mu_t)
            outputs.append((theta, pi))

        return outputs

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
    return (seen_ll/seen_w    if seen_w   > 0 else float('nan'),
            unseen_ll/unseen_w if unseen_w > 0 else float('nan'),
            seen_w, unseen_w)

def persistence_mixture(S_t, w_t):
    mu = (w_t[:, None] * S_t).sum(0, keepdims=True)
    return np.clip(mu, EPS, 1-EPS), np.array([1.0])

def safe_logit(mu, eps=1e-3):
    mu = np.clip(mu, eps, 1-eps)
    return np.log(mu / (1-mu)).astype(np.float32)

# --------------------------------------------------------- windows -------

def make_windows(clouds, M, l, start, end):
    """Windows for seq2seq: M input months, l target months.
    Returns (input_indices, target_indices) where targets=[i+1,...,i+l].
    """
    wins = []
    for i in range(start, end):
        inputs  = list(range(i-M+1, i+1))
        targets = list(range(i+1, i+l+1))
        if inputs[0] < 0: continue
        if any(clouds[j] is None for j in inputs): continue
        if any(j >= len(clouds) or clouds[j] is None
               for j in targets): continue
        wins.append((inputs, targets))
    return wins

# --------------------------------------------------------------- train ---

def train(model, clouds, train_pairs_s1, train_wins_s2, a):
    # ---- Stage 1: set encoder + MLP decoder only (= 139) ----
    print("  stage 1: set encoder + MLP decoder (= 139)...")
    for n, p in model.named_parameters():
        p.requires_grad = "set_enc" in n or "dec_mlp" in n
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.lr, weight_decay=1e-3)

    for epoch in range(a.epochs_s1):
        total = 0.0
        for t, t1 in [train_pairs_s1[i] for i in
                      np.random.permutation(len(train_pairs_s1))]:
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
        if (epoch+1) % 100 == 0:
            print(f"    [s1] epoch {epoch+1:4d}  "
                  f"ll/pair {total/len(train_pairs_s1):.4f}  "
                  f"lam={model.set_enc.lam.item():.3f}")

    # ---- Stage 2: temporal encoder + decoder LSTM (encoder+mlp frozen) ----
    print("  stage 2: temporal encoder + decoder LSTM...")
    for n, p in model.named_parameters():
        p.requires_grad = "temp_enc" in n or "dec_lstm" in n
    opt2 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.lr, weight_decay=1e-3)

    for epoch in range(a.epochs_s2):
        total = 0.0; n_wins = 0
        for inputs, targets in [train_wins_s2[i] for i in
                                 np.random.permutation(len(train_wins_s2))]:
            ct_t  = clouds[inputs[-1]]
            lmu_t = torch.tensor(safe_logit(ct_t["mu"]), dtype=torch.float32)
            window = [{"S_bin": torch.tensor(clouds[i]["S"],
                                             dtype=torch.float32),
                       "w":     torch.tensor(clouds[i]["w"],
                                             dtype=torch.float32)}
                      for i in inputs]
            l = len(targets)

            # teacher forcing: encode actual future months
            with torch.no_grad():
                teacher_us = torch.stack([
                    model.encode_month(
                        torch.tensor(clouds[t]["S"], dtype=torch.float32),
                        torch.tensor(clouds[t]["w"], dtype=torch.float32))
                    for t in targets])               # (l, d)

            outputs = model.forward_seq2seq(window, lmu_t, l,
                                            teacher_us=teacher_us)

            # loss at every step -- richer training signal
            loss = sum(
                -loglik_th(
                    torch.tensor(clouds[targets[s]]["S"], dtype=torch.float32),
                    torch.tensor(clouds[targets[s]]["w"], dtype=torch.float32),
                    theta, pi)
                + a.alpha * diversity_loss(theta)
                + a.beta  * entropy_loss(pi)
                for s, (theta, pi) in enumerate(outputs))

            opt2.zero_grad(); loss.backward(); opt2.step()
            total += loss.item() / l
            n_wins += 1

        if (epoch+1) % 100 == 0:
            print(f"    [s2] epoch {epoch+1:4d}  "
                  f"ll/step {total/n_wins:.4f}")

# ------------------------------------------------------------- evaluate --

def evaluate(model, clouds, test_wins, train_sets, months, eval_h):
    model.eval()
    print(f"\n{'h':>4} {'month_t+h':>12} {'mdl_seen':>10} "
          f"{'mdl_unseen':>12} {'per_seen':>10} {'per_unseen':>12}")
    summary = {}

    with torch.no_grad():
        for inputs, targets in test_wins:
            ct_t  = clouds[inputs[-1]]
            lmu_t = torch.tensor(safe_logit(ct_t["mu"]), dtype=torch.float32)
            window = [{"S_bin": torch.tensor(clouds[i]["S"],
                                             dtype=torch.float32),
                       "w":     torch.tensor(clouds[i]["w"],
                                             dtype=torch.float32)}
                      for i in inputs]

            # free-running rollout to max horizon
            l       = len(targets)
            outputs = model.forward_seq2seq(window, lmu_t, l,
                                            teacher_us=None)  # free running

            for s, (theta, pi) in enumerate(outputs):
                h = s + 1
                if h not in eval_h: continue
                target = targets[s]
                if target >= len(clouds) or clouds[target] is None:
                    continue

                ct1    = clouds[target]
                th_np  = theta.numpy(); pi_np = pi.numpy()
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

                if h not in summary: summary[h] = []
                summary[h].append((s_m,u_m,s_p,u_p,sw,uw))

    print("\n=== unseen gain summary ===")
    for h in eval_h:
        if h not in summary: continue
        R = np.array(summary[h], dtype=float)
        R = R[np.isfinite(R).all(axis=1)]
        if not len(R): print(f"  h={h}: no finite results"); continue
        ws, wu = R[:,4], R[:,5]
        gain   = np.average(R[:,1]-R[:,3], weights=wu)
        print(f"  h={h:2d}: model unseen {np.average(R[:,1],weights=wu):.3f}  "
              f"persistence unseen {np.average(R[:,3],weights=wu):.3f}  "
              f"gain {gain:+.3f} nats  "
              f"{'BEATS persistence' if gain > 0 else 'loses'}")

# ----------------------------------------------------------------- main --

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True)
    p.add_argument("--vocab",       required=True)
    p.add_argument("--months",      required=True)
    p.add_argument("--test-start",  required=True)
    p.add_argument("--engine",      default=ENGINE)
    p.add_argument("--M",  type=int, default=12,
                   help="months of input history")
    p.add_argument("--l",  type=int, default=6,
                   help="decoder rollout steps")
    p.add_argument("--K",       type=int,   default=8)
    p.add_argument("--r",       type=int,   default=70)
    p.add_argument("--d",       type=int,   default=32)
    p.add_argument("--heads",   type=int,   default=2)
    p.add_argument("--layers",  type=int,   default=1)
    p.add_argument("--d-lstm",  type=int,   default=64, dest="d_lstm")
    p.add_argument("--epochs-s1", type=int, default=500, dest="epochs_s1")
    p.add_argument("--epochs-s2", type=int, default=300, dest="epochs_s2")
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

    # stage 1: consecutive h=1 pairs
    train_pairs_s1 = [(i, i+1) for i in range(ts-1)
                      if clouds[i] and clouds[i+1]]

    # stage 2: M-month windows with l rollout steps
    train_wins_s2  = make_windows(clouds, a.M, a.l, a.M-1, ts)

    # test: same window format
    test_wins      = make_windows(clouds, a.M, a.l, ts, len(months)-a.l)

    print(f"stage 1 pairs: {len(train_pairs_s1)}")
    print(f"stage 2 windows: {len(train_wins_s2)}  "
          f"(M={a.M}, l={a.l})")
    print(f"test windows: {len(test_wins)}")

    train_idx = list(range(ts))
    print("fitting PCA...")
    components, global_mean = fit_pca(clouds, train_idx, a.r)

    train_sets = set()
    for i in train_idx:
        if clouds[i]: train_sets |= clouds[i]["sets"]
    print(f"training constellations: {len(train_sets)}")

    model = Seq2SeqSets(V=V, r=a.r, d=a.d, K=a.K,
                        heads=a.heads, n_layers=a.layers,
                        d_lstm=a.d_lstm)
    model.set_pca(components, global_mean)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total_p:,}")

    print("\ntraining...")
    train(model, clouds, train_pairs_s1, train_wins_s2, a)

    print("\nevaluating...")
    evaluate(model, clouds, test_wins, train_sets, months, EVAL_H)

    os.makedirs("results", exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "components": components, "global_mean": global_mean},
               "results/143_model.pt")
    print("saved results/143_model.pt")

if __name__ == "__main__":
    main()
