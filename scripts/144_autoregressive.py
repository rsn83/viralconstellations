#!/usr/bin/env python
"""
144_autoregressive.py  --  143_seq2seq_v2 with autoregressive emission.

Key change: MLPDecoder replaced with AutoregressiveMixtureDecoder.
  - MADE (Masked Autoencoder) over top-J most variable positions
  - Factorized Bernoulli for remaining V-J positions
  - Directly breaks all three impossibility results for the J positions
  - J=200 for computational feasibility on CPU

Why top-J positions:
  Novel constellations differ from existing ones at specific positions.
  Top-J by variance = positions where constellations actually differ.
  The impossibility results matter most at these positions.

Run:
  python scripts/144_autoregressive.py \
    --data-dir data/processed/full_data_graphs_withdel \
    --vocab data/processed/full_data_graphs_withdel/posres_vocab_withdel.tsv \
    --months 2020-06:2023-06 --test-start 2022-06 \
    --M 12 --l 6 --K 8 --r 70 --d 32 --heads 2 --layers 1 \
    --d-lstm 64 --J 200 --d-made 32 \
    --epochs-s1 300 --epochs-s2 300 \
    --upweight 10.0 --top 500
"""

import argparse, importlib.util, sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logsumexp

ENGINE   = "scripts/110_hierarchical_birthdeath_v2_fixed.py"
EPS      = 1e-6
EVAL_H   = [1, 2, 3, 6]

# ---------------------------------------------------------------- engine ---
def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m    = importlib.util.module_from_spec(spec); sys.modules["engine"] = m
    spec.loader.exec_module(m); return m

# ------------------------------------------------------------------ data ---
def recs_to_matrix(recs, V, top=None):
    if top: recs = sorted(recs, key=lambda x: -x[1])[:top]
    S = np.zeros((len(recs), V), dtype=np.float32)
    for i, (s, _) in enumerate(recs):
        for v in s:
            if v < V: S[i, v] = 1.0
    w = np.array([float(c) for _, c in recs], dtype=np.float32)
    w /= w.sum(); return S, w

def load_all(E, data_dir, months, V, top):
    print("loading...", flush=True)
    out = []
    for ym in months:
        recs = E.load_month(data_dir, ym)
        if not recs: out.append(None); continue
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
        c   = clouds[i]
        gm  = (c["w"][:, None] * c["S"]).sum(0) if gm is None \
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

# ------------------------------------------------------- top-J positions --
def top_j_positions(clouds, train_idx, J):
    """Top-J positions by weighted variance across training constellations.
    These are where constellations differ most -- where MADE adds most value.
    """
    freq  = None; freq2 = None; total_w = 0.0
    for i in train_idx:
        c = clouds[i]; S = c["S"]; w = c["w"]
        f = (w[:, None] * S).sum(0)
        f2 = (w[:, None] * S**2).sum(0)
        freq  = f  if freq  is None else freq  + f
        freq2 = f2 if freq2 is None else freq2 + f2
        total_w += 1.0
    mu  = freq / total_w
    var = freq2 / total_w - mu**2
    top = np.argsort(var)[::-1][:J]
    print(f"  top-{J} positions: mean var {var[top].mean():.4f} "
          f"vs overall mean var {var.mean():.4f}")
    return np.sort(top)   # sorted for consistent ordering

# --------------------------------------------------------- Jaccard --------
def jaccard_matrix(S_bin):
    dot = S_bin @ S_bin.T; sz = S_bin.sum(1)
    return torch.nan_to_num(
        dot / (sz.unsqueeze(1) + sz.unsqueeze(0) - dot + EPS), nan=0.0)

# ----------------------------------------------------------------- MADE ---
class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation over J binary positions.
    Computes log p(s_J | c) autoregressively in one forward pass.
    c is a component conditioning vector (d_cond,).

    Breaks all three impossibility results:
      - Two equidistant sets CAN get different log-prob (non-additive)
      - Ranking is NOT nearest-mode ranking (context-dependent)
      - Unseen mutations get context-specific cost (not -9.21 uniformly)
    """
    def __init__(self, J, d_cond, d_hidden=32, seed=0):
        super().__init__()
        self.J = J

        # assign degrees to hidden units: uniformly spaced in {1,...,J-1}
        rng = np.random.default_rng(seed)
        degrees = rng.integers(1, max(J, 2), size=d_hidden)  # (d_hidden,)

        # mask_W1[h, v] = 1 if degrees[h] >= v  (hidden h sees input v)
        v_range = torch.arange(J).float()
        d_range = torch.tensor(degrees).float()
        mask_W1 = (d_range.unsqueeze(1) >= v_range.unsqueeze(0)).float()  # (H, J)

        # mask_W2[v, h] = 1 if v > degrees[h]  (output v sees hidden h)
        mask_W2 = (v_range.unsqueeze(1) > d_range.unsqueeze(0)).float()   # (J, H)

        self.register_buffer('mask_W1', mask_W1)
        self.register_buffer('mask_W2', mask_W2)

        self.W1       = nn.Parameter(torch.randn(d_hidden, J) * 0.01)
        self.b1       = nn.Parameter(torch.zeros(d_hidden))
        self.W2       = nn.Parameter(torch.randn(J, d_hidden) * 0.01)
        self.b2       = nn.Parameter(torch.zeros(J))
        self.cond_proj = nn.Linear(d_cond, d_hidden, bias=False)

    def log_prob(self, s_J, c):
        """s_J: (N, J) binary, c: (d_cond,). Returns log_probs (N,)."""
        W1_m   = self.W1 * self.mask_W1                             # (H, J)
        W2_m   = self.W2 * self.mask_W2                             # (J, H)
        h      = torch.relu(s_J @ W1_m.T + self.b1
                            + self.cond_proj(c))                    # (N, H)
        logits = h @ W2_m.T + self.b2                              # (N, J)
        lp     = (s_J * F.logsigmoid(logits)
                  + (1 - s_J) * F.logsigmoid(-logits))        # (N, J)
        return lp.sum(1)                                            # (N,)


# ---------------------------------------- autoregressive mixture decoder --
class AutoregressiveMixtureDecoder(nn.Module):
    """K-component mixture:
      - MADE emission for top-J positions (autoregressive, breaks impossibilities)
      - Factorized Bernoulli for remaining V-J positions (same as before)
    Component embeddings + pi produced by MLP from u.
    """
    def __init__(self, d, K, r, V, J, d_made=32, top_j_idx=None):
        super().__init__()
        self.K = K; self.r = r; self.J = J; self.V = V
        self.d_comp = d

        # MLP: u -> K component embeddings + K logits for pi
        self.net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, K*d + K))

        # shared MADE with component conditioning for top-J positions
        self.made = MADE(J, d_cond=d, d_hidden=d_made)

        # factorized decoder for remaining positions (PCA-based, same as 139)
        self.fact_net = nn.Sequential(
            nn.Linear(d, d*2), nn.Tanh(),
            nn.Linear(d*2, K * r))  # K*r PCA coefficients for factorized part

        if top_j_idx is not None:
            self.register_buffer('top_j_idx',
                                 torch.tensor(top_j_idx, dtype=torch.long))

        # fixed alpha schedule based on empirical finding:
        # factorized works better short-term, MADE works better long-term
        # h=1: pure factorized, h=6: pure MADE, h=2,3: interpolated
        self.alpha_schedule = {1: 0.0, 2: 0.2, 3: 0.5, 6: 1.0}

    def forward(self, u, P, logit_mu_t, S_query, h=1):
        """
        u           : (d,) LSTM output
        P           : (r, V) PCA buffer
        logit_mu_t  : (V,) persistence baseline
        S_query     : (N, V) constellations to score
        Returns: log_mix (N,) mixture log-likelihood, pi (K,)
        """
        out    = self.net(u)
        c_ks   = out[:self.K * self.d_comp].view(self.K, self.d_comp)  # (K, d)
        pi     = torch.softmax(out[self.K * self.d_comp:], dim=0)       # (K,)
        log_pi = torch.log(pi + EPS)

        # factorized theta for remaining positions
        f_out     = self.fact_net(u)
        coeff     = f_out.view(self.K, self.r)                          # (K, r)
        theta_all = torch.sigmoid(coeff @ P + logit_mu_t.unsqueeze(0)) # (K, V)

        # top-J positions from S_query
        S_J = S_query[:, self.top_j_idx]                               # (N, J)

        # remaining positions (complement of top_j_idx)
        all_idx  = torch.arange(self.V, device=u.device)
        mask     = torch.ones(self.V, dtype=torch.bool, device=u.device)
        mask[self.top_j_idx] = False
        rest_idx = all_idx[mask]                                        # (V-J,)
        S_rest   = S_query[:, rest_idx]                                 # (N, V-J)

        # fixed alpha from schedule: 0=factorized, 1=MADE
        alpha = self.alpha_schedule.get(h, 0.5)

        # full factorized log-prob over ALL positions (for short horizons)
        S_all_J   = S_query[:, self.top_j_idx]                         # (N, J)
        S_all_rest = S_query[:, rest_idx]                               # (N, V-J)

        # log p(s|k) = alpha * MADE + (1-alpha) * factorized
        log_p_s_k = torch.stack([
            alpha       * self.made.log_prob(S_J, c_ks[k])
            + (1-alpha) * self._factorized_lp(S_all_J,
                                              theta_all[k, self.top_j_idx])
            + self._factorized_lp(S_all_rest, theta_all[k, rest_idx])
            for k in range(self.K)
        ], dim=1)                                                        # (N, K)

        log_mix = torch.logsumexp(log_pi.unsqueeze(0) + log_p_s_k,
                                  dim=1)                                # (N,)
        return log_mix, pi

    def _factorized_lp(self, S_rest, theta_rest):
        """Factorized Bernoulli log-prob for non-MADE positions."""
        th = theta_rest.clamp(EPS, 1-EPS)
        lt = torch.log(th); lf = torch.log1p(-th)
        return (S_rest * lt + (1 - S_rest) * lf).sum(1)                # (N,)


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

class AutoregressiveLSTMDecoder(nn.Module):
    def __init__(self, d, d_lstm):
        super().__init__()
        self.init_h = nn.Linear(d_lstm*2, d_lstm)
        self.init_c = nn.Linear(d_lstm*2, d_lstm)
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

class Seq2SeqAutoregressive(nn.Module):
    def __init__(self, V, r, d, K, heads, n_layers, d_lstm,
                 J, d_made, top_j_idx):
        super().__init__()
        self.d = d
        self.set_enc  = SetEncoder(r, d, heads, n_layers)
        self.temp_enc = TemporalEncoder(d, d_lstm)
        self.dec_lstm = AutoregressiveLSTMDecoder(d, d_lstm)
        self.dec_ar   = AutoregressiveMixtureDecoder(
            d, K, r, V, J, d_made, top_j_idx)
        self.register_buffer("P",           torch.zeros(r, V))
        self.register_buffer("global_mean", torch.zeros(V))

    def set_pca(self, components, gm):
        self.P.copy_(torch.tensor(components, dtype=torch.float32))
        self.global_mean.copy_(torch.tensor(gm, dtype=torch.float32))

    def encode_month(self, S_bin, w):
        S_pca = (S_bin - self.global_mean) @ self.P.T
        return self.set_enc(S_pca, S_bin, w)

    def score(self, u, S_query, logit_mu_t, h=1):
        """Score constellations under the autoregressive mixture."""
        return self.dec_ar(u, self.P, logit_mu_t, S_query, h=h)

    def forward_single(self, S_bin, w, logit_mu_t, S_query):
        """Stage 1: single month, returns (log_mix, pi)."""
        u = self.encode_month(S_bin, w)
        return self.score(u, S_query, logit_mu_t, h=1)

    def forward_seq2seq(self, window, logit_mu_t, l,
                        S_targets, teacher_us=None):
        """Full seq2seq. S_targets: list of l S_bin tensors (target constellations)."""
        us       = torch.stack([self.encode_month(d["S_bin"], d["w"])
                                for d in window])
        h_t, ctx = self.temp_enc(us)
        us_pred  = self.dec_lstm(h_t, ctx, us[-1], l, teacher_us)
        return [self.score(us_pred[s], S_targets[s], logit_mu_t, h=s+1)
                for s in range(l)]

# ------------------------------------------------------------ loss --------
def mixture_nll(log_mix, w, S_query, present_sets, upweight):
    """Upweighted negative log-likelihood. Novel constellations weighted more."""
    w_adj = w.clone()
    for i, row in enumerate(S_query):
        fs = frozenset(torch.nonzero(row).squeeze(-1).tolist())
        if fs not in present_sets:
            w_adj[i] = w_adj[i] * upweight
    w_adj = w_adj / w_adj.sum()
    return -(w_adj * log_mix).sum()

def diversity_loss_pi(pi):
    return (pi * torch.log(pi + EPS)).sum()

# ------------------------------------------------ seen / unseen ----------
def score_seen_unseen_ar(model, S_next, w_next, logit_mu_t,
                         train_sets, is_model=True):
    """Score using MADE-based mixture. Returns seen, unseen LL."""
    with torch.no_grad():
        S_th  = torch.tensor(S_next, dtype=torch.float32)
        w_th  = torch.tensor(w_next, dtype=torch.float32)
        if is_model:
            # u_t already set by caller -- pass dummy u=0, will be overridden
            # Actually we need u from outside; use stored last u
            pass
    # Simpler: just use numpy logsumexp on stored log_mix
    # Called with pre-computed log_mix_np
    raise NotImplementedError("use score_from_logmix")

def score_from_logmix(log_mix_np, w_next, S_next, train_sets):
    seen_ll = unseen_ll = seen_w = unseen_w = 0.0
    for i, (ll_i, wi, row) in enumerate(zip(log_mix_np, w_next, S_next)):
        if not np.isfinite(ll_i): continue
        fs = frozenset(np.flatnonzero(row).tolist())
        if fs in train_sets:
            seen_ll += wi*ll_i; seen_w += wi
        else:
            unseen_ll += wi*ll_i; unseen_w += wi
    return (seen_ll/seen_w    if seen_w   > 0 else float('nan'),
            unseen_ll/unseen_w if unseen_w > 0 else float('nan'),
            seen_w, unseen_w)

def persistence_logmix(S_next, S_t, w_t):
    """Factorized Bernoulli persistence baseline: log p(s|mu_t)."""
    mu  = (w_t[:, None] * S_t).sum(0)
    mu  = np.clip(mu, 1e-3, 1-1e-3)
    lt  = np.log(mu); lf = np.log1p(-mu)
    lps = []
    for row in S_next:
        lps.append((row * lt + (1-row) * lf).sum())
    return np.array(lps)

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
        if any(j >= len(clouds) or clouds[j] is None for j in targets): continue
        wins.append((inputs, targets))
    return wins

# --------------------------------------------------------------- train ---
def train(model, clouds, train_pairs, train_wins, a):
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-3)

    print("  stage 1: all params, h=1 warmup...")
    for epoch in range(a.epochs_s1):
        total = 0.0
        for t, t1 in [train_pairs[i] for i in
                      np.random.permutation(len(train_pairs))]:
            ct, ct1 = clouds[t], clouds[t1]
            S_b  = torch.tensor(ct["S"],  dtype=torch.float32)
            w_t  = torch.tensor(ct["w"],  dtype=torch.float32)
            lmu  = torch.tensor(safe_logit(ct["mu"]), dtype=torch.float32)
            S_n  = torch.tensor(ct1["S"], dtype=torch.float32)
            w_n  = torch.tensor(ct1["w"], dtype=torch.float32)
            pres = ct["sets"]
            log_mix, pi = model.forward_single(S_b, w_t, lmu, S_n)
            loss = (mixture_nll(log_mix, w_n, S_n, pres, a.upweight)
                    + a.beta * diversity_loss_pi(pi))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (epoch+1) % 100 == 0:
            print(f"    [s1] epoch {epoch+1:4d}  "
                  f"nll/pair {total/len(train_pairs):.4f}  "
                  f"alpha schedule {model.dec_ar.alpha_schedule}  "
                  f"lam={model.set_enc.lam.item():.3f}")

    print("  stage 2: all params, all horizons...")
    opt2 = torch.optim.Adam(model.parameters(),
                             lr=a.lr*0.3, weight_decay=1e-3)
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
            S_targets = [torch.tensor(clouds[t]["S"], dtype=torch.float32)
                         for t in targets]
            w_targets  = [torch.tensor(clouds[t]["w"], dtype=torch.float32)
                          for t in targets]
            l = len(targets)
            with torch.no_grad():
                teacher_us = torch.stack([
                    model.encode_month(S_targets[s], w_targets[s])
                    for s in range(l)])
            outputs = model.forward_seq2seq(window, lmu, l,
                                            S_targets, teacher_us)
            loss = sum(
                mixture_nll(log_mix, w_targets[s], S_targets[s],
                            pres, a.upweight)
                + a.beta * diversity_loss_pi(pi)
                for s, (log_mix, pi) in enumerate(outputs))
            opt2.zero_grad(); loss.backward(); opt2.step()
            total += loss.item() / l; n_wins += 1
        if (epoch+1) % 100 == 0:
            print(f"    [s2] epoch {epoch+1:4d}  "
                  f"nll/step {total/n_wins:.4f}")

# ------------------------------------------------------------- evaluate --
def evaluate(model, clouds, test_wins, train_sets, months, eval_h):
    model.eval()
    h_cols = " | ".join(f"h={h} mdl_u  per_u   gain" for h in eval_h)
    print(f"\nModel trained on fixed window, input window slides at test time.")
    print(f"\n{'Input window':>22} | {'Test month':>12} | {h_cols}")
    print("-" * (24 + 16 + len(h_cols) + 10))

    gains_by_h = {h: [] for h in eval_h}

    with torch.no_grad():
        for inputs, targets in test_wins:
            ct_t  = clouds[inputs[-1]]
            lmu_t = torch.tensor(safe_logit(ct_t["mu"]), dtype=torch.float32)
            window = [{"S_bin": torch.tensor(clouds[i]["S"], dtype=torch.float32),
                       "w":     torch.tensor(clouds[i]["w"], dtype=torch.float32)}
                      for i in inputs]
            l         = len(targets)
            S_targets = [torch.tensor(clouds[t]["S"], dtype=torch.float32)
                         for t in targets]
            outputs   = model.forward_seq2seq(window, lmu_t, l,
                                              S_targets, teacher_us=None)

            iw_label = f"{months[inputs[0]]}..{months[inputs[-1]]}"
            results  = {}
            for s, (log_mix, pi) in enumerate(outputs):
                h = s + 1
                if h not in eval_h: continue
                target = targets[s]
                if target >= len(clouds) or clouds[target] is None: continue
                ct1    = clouds[target]
                lm_np  = log_mix.numpy()
                # model scores
                sm, um, sw, uw = score_from_logmix(
                    lm_np, ct1["w"], ct1["S"], train_sets)
                # persistence scores
                per_lm = persistence_logmix(ct1["S"], ct_t["S"], ct_t["w"])
                sp, up, _, _  = score_from_logmix(
                    per_lm, ct1["w"], ct1["S"], train_sets)
                gain = um - up if (np.isfinite(um) and np.isfinite(up)) \
                       else float('nan')
                results[h] = (months[target], um, up, gain, uw)
                if np.isfinite(gain):
                    gains_by_h[h].append((gain, uw))

            if results:
                first_h    = eval_h[0]
                test_month = results.get(first_h, ("?",))[0]
                h_vals = ""
                for h in eval_h:
                    if h in results:
                        _, u_m, u_p, gain, _ = results[h]
                        h_vals += f" | {u_m:+7.2f} {u_p:+7.2f} {gain:+6.2f}"
                    else:
                        h_vals += " |    n/a     n/a    n/a"
                print(f"{iw_label:>22} | {test_month:>12} |{h_vals}")

    print(f"\n{'='*60}")
    print("=== Unseen gain summary ===")
    for h in eval_h:
        if not gains_by_h[h]: print(f"  h={h}: no results"); continue
        gains = np.array([g for g, _ in gains_by_h[h]])
        wts   = np.array([w for _, w in gains_by_h[h]])
        wm    = np.average(gains, weights=wts)
        print(f"  h={h:2d}: {wm:+8.3f} nats  "
              f"{'BEATS persistence' if wm > 0 else 'loses'}")

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
    p.add_argument("--J",     type=int,   default=200,
                   help="MADE positions (top-J by variance)")
    p.add_argument("--d-made",type=int,   default=32, dest="d_made")
    p.add_argument("--epochs-s1", type=int, default=300, dest="epochs_s1")
    p.add_argument("--epochs-s2", type=int, default=300, dest="epochs_s2")
    p.add_argument("--lr",    type=float, default=1e-3)
    p.add_argument("--top",   type=int,   default=500)
    p.add_argument("--beta",  type=float, default=0.1)
    p.add_argument("--upweight", type=float, default=10.0)
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
    train_pairs = [(i, i+1) for i in range(ts-1)
                   if clouds[i] and clouds[i+1]]
    train_wins  = make_windows(clouds, a.M, a.l, a.M-1, ts)
    test_wins   = make_windows(clouds, a.M, a.l, ts, len(months)-a.l)
    print(f"s1 pairs: {len(train_pairs)}  s2 windows: {len(train_wins)}"
          f"  test windows: {len(test_wins)}")

    print("fitting PCA...")
    components, gm = fit_pca(clouds, train_idx, a.r)

    print(f"computing top-{a.J} positions...")
    top_j = top_j_positions(clouds, train_idx, a.J)

    train_sets = set()
    for i in train_idx:
        if clouds[i]: train_sets |= clouds[i]["sets"]
    print(f"training constellations: {len(train_sets)}")

    model = Seq2SeqAutoregressive(
        V=V, r=a.r, d=a.d, K=a.K,
        heads=a.heads, n_layers=a.layers,
        d_lstm=a.d_lstm, J=a.J, d_made=a.d_made,
        top_j_idx=top_j)
    model.set_pca(components, gm)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\ntraining...")
    train(model, clouds, train_pairs, train_wins, a)

    print("\nevaluating...")
    evaluate(model, clouds, test_wins, train_sets, months, EVAL_H)

    os.makedirs("results", exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(a),
                "components": components, "global_mean": gm,
                "top_j_idx": top_j},
               "results/144_model.pt")
    print("saved results/144_model.pt")

if __name__ == "__main__":
    main()
