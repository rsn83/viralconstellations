#!/usr/bin/env python3
"""
Toy CTMC with learned background-conditioned rates.
Pure numpy — no torch needed.  Runs on laptop in minutes.

What this tests (one thing only):
  Does conditioning on mutational background improve novel constellation
  prediction over unconditional parent-weighting (M4 baseline)?

Architecture
  State  : binary vector x ∈ {0,1}^K  (mutation present/absent)
  Network: x -> [K] -> relu -> [H] -> relu -> [H] -> softmax -> rates[K]
  Rates  : u(z|x) = network output at position z, = 0 if z already in x
  
Training (discrete flow matching style, weak coupling)
  For each consecutive pair (T, T+1):
    1. Sample x0 from pop(T), x1 from pop(T+1) -- RANDOM pairing
    2. x1 must be a superset of x0 (else skip -- 99.8% of cases pass)
    3. Sample t ~ U[0,1]
    4. Build xt: mutations in (x1 minus x0) each included with prob t
    5. Loss: cross-entropy of rates toward mutations still missing from xt

Inference (Gillespie, one step per month)
  For each constellation x in pop(T):
    - compute rates u(z|x) for all z not in x
    - scale by lambda (mean additions per sequence per month)
    - sample number of additions from Poisson(lambda * sum(rates))
    - sample which mutations, proportional to rates
  Aggregate new constellations -> predicted pop(T+1)
  
Evaluation: same log-score harness as forecast_baseline.py
"""

import argparse
import csv
import datetime as dt
import sys
from collections import Counter, defaultdict
import math

import numpy as np


def growth_predict(prev, cur, shrink=0.5, clip=1.5):
    """Extrapolate each constellation frequency one step forward."""
    n_cur = sum(cur.values())
    n_prev = sum(prev.values()) if prev else 0
    a = 1.0 / max(n_cur, 1)
    pred = {}
    for x, k in cur.items():
        p_now = k / n_cur
        if n_prev:
            p_before = prev.get(x, 0) / n_prev
            r = math.log((p_now + a) / (p_before + a))
            r = max(-clip, min(clip, r)) * shrink
        else:
            r = 0.0
        pred[x] = p_now * math.exp(r)
    s = sum(pred.values())
    return {x: v / s for x, v in pred.items()}


# --------------------------------------------------------------- data

def load_windows(path, window_days, stride_days, min_seqs):
    dates, cons_raw = [], []
    with open(path) as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        i_d, i_c = h.index("date"), h.index("constellation")
        for row in r:
            if len(row) <= i_c:
                continue
            dates.append(dt.date.fromisoformat(row[i_d]))
            cons_raw.append(row[i_c])
    order = np.argsort(dates)
    dates = [dates[i] for i in order]
    cons_raw = [cons_raw[i] for i in order]
    out, start, j = [], dates[0], 0
    while start <= dates[-1]:
        end = start + dt.timedelta(days=window_days)
        c = Counter()
        k = j
        while k < len(dates) and dates[k] < end:
            if dates[k] >= start:
                c[cons_raw[k]] += 1
            k += 1
        if sum(c.values()) >= min_seqs:
            out.append(c)
        start += dt.timedelta(days=stride_days)
        while j < len(dates) and dates[j] < start:
            j += 1
    return out


def build_vocab(windows, K):
    freq = Counter()
    for w in windows:
        for x_str in w:
            for m in x_str.split(","):
                freq[m] += 1
    vocab = [m for m, _ in freq.most_common(K)]
    v2i = {m: i for i, m in enumerate(vocab)}
    return vocab, v2i


def encode(x_str, v2i, K):
    vec = np.zeros(K, dtype=np.float32)
    for m in x_str.split(","):
        if m in v2i:
            vec[v2i[m]] = 1.0
    return vec


def encode_window(w, v2i, K):
    """Returns list of (binary_vec, count) for a window."""
    out = []
    for x_str, cnt in w.items():
        out.append((encode(x_str, v2i, K), cnt, x_str))
    return out


# --------------------------------------------------------------- MLP

class MLP:
    def __init__(self, K, H, rng):
        scale = lambda n_in: np.sqrt(2.0 / n_in)
        self.W1 = rng.normal(0, scale(K), (K, H)).astype(np.float32)
        self.b1 = np.zeros(H, dtype=np.float32)
        self.W2 = rng.normal(0, scale(H), (H, H)).astype(np.float32)
        self.b2 = np.zeros(H, dtype=np.float32)
        self.W3 = rng.normal(0, scale(H), (H, K)).astype(np.float32)
        self.b3 = np.zeros(K, dtype=np.float32)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.grads = [np.zeros_like(p) for p in self.params]

    def forward(self, x):
        """x: (B, K) -> logits (B, K), cache for backward"""
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return logits, (x, h1, h2)

    def backward(self, d_logits, cache):
        x, h1, h2 = cache
        dW3 = h2.T @ d_logits
        db3 = d_logits.sum(0)
        dh2 = d_logits @ self.W3.T
        dh2 *= (h2 > 0)
        dW2 = h1.T @ dh2
        db2 = dh2.sum(0)
        dh1 = dh2 @ self.W2.T
        dh1 *= (h1 > 0)
        dW1 = x.T @ dh1
        db1 = dh1.sum(0)
        for g, d in zip(self.grads,
                        [dW1, db1, dW2, db2, dW3, db3]):
            g += d

    def zero_grad(self):
        for g in self.grads:
            g[:] = 0.0

    def step(self, lr):
        for p, g in zip(self.params, self.grads):
            p -= lr * g


def softmax_masked(logits, mask):
    """Softmax over positions where mask==1; zero elsewhere. (B,K)"""
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits) * mask
    s = e.sum(1, keepdims=True) + 1e-9
    return e / s


# --------------------------------------------------------------- training

def make_batches(W_enc, t_idx, v2i, K, rng, n_pairs, batch):
    """Yield training batches from windows up to t_idx."""
    pairs_collected = []
    for t in range(1, t_idx + 1):
        cur = W_enc[t - 1]
        nxt = W_enc[t]
        # build frequency-weighted lists
        cur_list = [(vec, s) for vec, _, s in cur]
        cur_cnt = np.array([c for _, c, _ in cur], dtype=float)
        cur_cnt /= cur_cnt.sum()
        nxt_list = [(vec, s) for vec, _, s in nxt]
        nxt_cnt = np.array([c for _, c, _ in nxt], dtype=float)
        nxt_cnt /= nxt_cnt.sum()
        for _ in range(min(n_pairs, 500)):
            i0 = rng.choice(len(cur_list), p=cur_cnt)
            i1 = rng.choice(len(nxt_list), p=nxt_cnt)
            x0, x1 = cur_list[i0][0], nxt_list[i1][0]
            # only keep if x1 is superset of x0 in vocab positions
            added = (x1 - x0).clip(0)
            if added.sum() == 0:
                continue
            pairs_collected.append((x0, x1, added))

    rng.shuffle(pairs_collected)
    for i in range(0, len(pairs_collected), batch):
        chunk = pairs_collected[i:i + batch]
        if not chunk:
            break
        yield chunk


def train_epoch(net, W_enc, t_idx, v2i, K, rng, lr, n_pairs, batch):
    total_loss, n_batches = 0.0, 0
    for chunk in make_batches(W_enc, t_idx, v2i, K, rng, n_pairs, batch):
        B = len(chunk)
        X0 = np.stack([c[0] for c in chunk])     # (B, K)
        X1 = np.stack([c[1] for c in chunk])     # (B, K)
        added = np.stack([c[2] for c in chunk])  # (B, K) -- mutations to add

        # interpolate: each added mutation present in xt with prob t
        t = rng.uniform(0, 1, size=(B, 1))
        include = rng.random((B, K)) < t
        Xt = X0 + added * include                # (B, K)

        # mask: positions absent from xt are candidates
        mask = (1.0 - Xt)                        # (B, K)

        net.zero_grad()
        logits, cache = net.forward(Xt)
        probs = softmax_masked(logits, mask)      # (B, K)

        # target: mutations still to be added (in x1 but not xt)
        target = (X1 - Xt).clip(0)              # (B, K)
        # normalise target to sum to 1 per row
        tsum = target.sum(1, keepdims=True) + 1e-9
        target_p = target / tsum

        # cross-entropy loss
        loss = -(target_p * np.log(probs + 1e-9)).sum(1).mean()
        d_logits = (probs - target_p) / B
        net.backward(d_logits, cache)
        net.step(lr)

        total_loss += loss
        n_batches += 1

    return total_loss / max(n_batches, 1)


# --------------------------------------------------------------- inference

def gillespie_novel(net, W_enc_t, lam, K, rng, n_samples, seen_keys):
    """
    Simulate ONLY novel constellations (not seen before).
    Returns Counter of index-key -> count for novel proposals only.
    Incumbents are handled separately by the growth model.
    """
    seqs = [(vec, cnt, s) for vec, cnt, s in W_enc_t]
    counts = np.array([c for _, c, _ in seqs], dtype=float)
    counts /= counts.sum()

    novel = Counter()
    attempts = 0
    while sum(novel.values()) < n_samples and attempts < n_samples * 10:
        attempts += 1
        idx = rng.choice(len(seqs), p=counts)
        x = seqs[idx][0].copy()

        logits, _ = net.forward(x[None])
        mask = (1.0 - x)[None]
        rates = softmax_masked(logits, mask)[0] * mask[0]
        total_rate = rates.sum()
        if total_rate < 1e-9:
            continue

        n_add = max(1, rng.poisson(lam))   # at least 1 addition
        p = rates / (total_rate + 1e-9)
        avail = int(mask.sum())
        if avail == 0:
            continue
        chosen = rng.choice(K, size=min(n_add, avail),
                            replace=False, p=p)
        x[chosen] = 1.0
        key = ",".join(str(i) for i in np.where(x > 0.5)[0])
        if key not in seen_keys:
            novel[key] += 1

    return novel


# --------------------------------------------------------------- scoring

def score_window(pred, floor, obs_enc, obs_w, seen_before_strs,
                 K, v2i, vocab):
    """
    pred: Counter of index-key -> mass (already normalised)
    obs_enc: list of (vec, cnt, str) for next window
    """
    tot = sum(c for _, c, _ in obs_enc)
    ll_all = ll_inc = ll_nov = 0.0
    n_inc = n_nov = hit_nov = 0

    pred_total = sum(pred.values()) + 1e-9

    for vec, cnt, s in obs_enc:
        key = ",".join(str(i) for i in np.where(vec > 0.5)[0])
        p = pred.get(key, 0) / pred_total
        p = max(p, floor)
        lp = math.log(p) * cnt
        ll_all += lp
        if s in seen_before_strs:
            ll_inc += lp
            n_inc += cnt
        else:
            ll_nov += lp
            n_nov += cnt
            if p > floor:
                hit_nov += cnt

    return {
        "ll_all": ll_all / tot,
        "ll_inc": ll_inc / n_inc if n_inc else float("nan"),
        "ll_nov": ll_nov / n_nov if n_nov else float("nan"),
        "novel_frac": n_nov / tot,
        "novel_recall": hit_nov / n_nov if n_nov else float("nan"),
    }


# --------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--stride-days", type=int, default=30)
    ap.add_argument("--min-seqs", type=int, default=50)
    ap.add_argument("--K", type=int, default=150,
                    help="mutation vocabulary size")
    ap.add_argument("--H", type=int, default=256,
                    help="hidden layer size")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-pairs", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-samples", type=int, default=3000,
                    help="Gillespie samples per window")
    ap.add_argument("--lam", type=float, default=1.5,
                    help="mean mutations added per step (from data)")
    ap.add_argument("--floor-space", type=float, default=1e7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("loading...", file=sys.stderr)
    windows = load_windows(args.data, args.window_days,
                           args.stride_days, args.min_seqs)
    print(f"{len(windows)} windows", file=sys.stderr)

    vocab, v2i = build_vocab(windows, args.K)
    K = len(vocab)
    print(f"vocab size: {K}", file=sys.stderr)

    W_enc = [encode_window(w, v2i, K) for w in windows]

    net = MLP(K, args.H, rng)
    floor = 1.0 / args.floor_space

    results = defaultdict(list)
    seen_strs = set()

    for t in range(len(windows) - 1):
        if t < 2:
            for _, _, s in W_enc[t]:
                seen_strs.add(s)
            continue

        # train on all pairs up to t
        for ep in range(args.epochs):
            loss = train_epoch(net, W_enc, t, v2i, K, rng,
                               args.lr, args.n_pairs, args.batch)
            if args.verbose:
                print(f"  w{t} ep{ep} loss={loss:.4f}", file=sys.stderr)

        # --- incumbent part: growth model (same as M4 baseline) --------
        cur_w = windows[t]
        prev_w = windows[t - 1] if t > 0 else None
        g = growth_predict(prev_w, cur_w)   # str -> freq
        n_cur = sum(cur_w.values())
        cur_freq = {s: k / n_cur for s, k in cur_w.items()}

        # eps: trailing mean novel fraction
        eps_hist = []
        seen_tmp = set()
        for ww in windows[:t + 1]:
            tot = sum(ww.values())
            nv = sum(k for s, k in ww.items() if s not in seen_tmp)
            eps_hist.append(nv / tot)
            seen_tmp.update(ww.keys())
        eps = float(np.mean(eps_hist[-5:])) if eps_hist else 0.05
        eps = min(max(eps, 1e-4), 0.5)

        # incumbent prediction (index-key based)
        inc_pred = {}
        for s, p in g.items():
            vec = encode(s, v2i, K)
            key = ",".join(str(i) for i in np.where(vec > 0.5)[0])
            inc_pred[key] = (1.0 - eps) * p

        # M4 novel (parent-weighted enumeration) for fair comparison
        add_c = Counter()
        for ww in windows[:t + 1]:
            for s in ww:
                for m in s.split(","):
                    add_c[m] += 1
        vocab_local = [m for m, _ in add_c.most_common(args.K)]
        m4_novel = {}
        for s, p in sorted(cur_freq.items(), key=lambda kv: -kv[1])[:3000]:
            xs = frozenset(s.split(","))
            for z in vocab_local:
                if z in xs:
                    continue
                child_set = xs | {z}
                child_str = ",".join(sorted(child_set))
                vec = encode(child_str, v2i, K)
                ckey = ",".join(str(i) for i in np.where(vec > 0.5)[0])
                m4_novel[ckey] = m4_novel.get(ckey, 0) + p
        s4 = sum(m4_novel.values()) + 1e-9
        m4_novel = {k: eps * v / s4 for k, v in m4_novel.items()}

        # CTMC novel (Gillespie)
        seen_keys = set(inc_pred.keys())
        ctmc_novel = gillespie_novel(net, W_enc[t], args.lam, K,
                                     rng, args.n_samples, seen_keys)
        sc = sum(ctmc_novel.values()) + 1e-9
        ctmc_novel = {k: eps * v / sc for k, v in ctmc_novel.items()}

        # score both
        obs_w = W_enc[t + 1]

        r_m4 = score_window({**inc_pred, **m4_novel}, floor,
                             obs_w, windows[t + 1], seen_strs, K, v2i, vocab)
        r_ctmc = score_window({**inc_pred, **ctmc_novel}, floor,
                              obs_w, windows[t + 1], seen_strs, K, v2i, vocab)
        r = r_ctmc
        for k, v in r_ctmc.items():
            if not math.isnan(v):
                results["ctmc_" + k].append(v)
        for k, v in r_m4.items():
            if not math.isnan(v):
                results["m4_" + k].append(v)

        if args.verbose:
            print(f"  w{t} ctmc recall={r_ctmc['novel_recall']:.3f} "
                  f"ll_nov={r_ctmc['ll_nov']:.3f} | "
                  f"m4 recall={r_m4['novel_recall']:.3f} "
                  f"ll_nov={r_m4['ll_nov']:.3f}", file=sys.stderr)

        for _, _, s in W_enc[t]:
            seen_strs.add(s)

    print(f"\n{'metric':<16} {'CTMC':>10} {'M4_enum':>10}")
    print("-" * 38)
    for k in ("ll_all", "ll_inc", "ll_nov", "novel_recall"):
        vc = np.mean(results["ctmc_" + k]) if results["ctmc_" + k] else float("nan")
        vm = np.mean(results["m4_" + k]) if results["m4_" + k] else float("nan")
        better = "<-" if vc > vm else ""
        print(f"{k:<16} {vc:>10.4f} {vm:>10.4f}  {better}")
    print("\n(log scores: higher is better; <- means CTMC wins)")


if __name__ == "__main__":
    main()
