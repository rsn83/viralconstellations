#!/usr/bin/env python3
"""
CTMC v2: background-conditioned rates with population-level features.

Network input per constellation:
  [binary_K | log_freq | freq_trend | log_persistence]

Three models compared on same Gillespie samples:
  CTMC_uncond  : marginal q(z), no network, no conditioning
  CTMC_cond    : network conditioned on binary vector only (v1)
  CTMC_pop     : network conditioned on binary + population features (v2)

All composed with same growth model for incumbents.
"""

import argparse, csv, datetime as dt, sys, math
from collections import Counter, defaultdict
import numpy as np


# ----------------------------------------------------------------- data

def growth_predict(prev, cur, shrink=0.5, clip=1.5):
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
            cons_raw.append(",".join(sorted(row[i_c].split(","))))
    order = np.argsort(dates)
    dates  = [dates[i]    for i in order]
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
    v2i   = {m: i for i, m in enumerate(vocab)}
    return vocab, v2i


def encode(x_str, v2i, K):
    vec = np.zeros(K, dtype=np.float32)
    for m in x_str.split(","):
        if m in v2i:
            vec[v2i[m]] = 1.0
    return vec


def make_pop_features(freq, trend, persistence):
    """Three scalar population features, log-scaled."""
    lf  = math.log(max(freq, 1e-6))
    lt  = math.log(max(trend, 1e-4))
    lp  = math.log(max(persistence, 1))
    return np.array([lf, lt, lp], dtype=np.float32)


def encode_window_with_pop(w, prev_w, persistence_map, v2i, K):
    """
    Returns list of (aug_vec, binary_vec, count, str) per constellation.
    aug_vec = [binary_K | log_freq | freq_trend | log_persistence]
    """
    tot      = max(sum(w.values()), 1)
    tot_prev = max(sum(prev_w.values()), 1) if prev_w else 1
    out = []
    for x_str, cnt in w.items():
        bvec = encode(x_str, v2i, K)
        freq  = cnt / tot
        prev_freq = prev_w.get(x_str, 0) / tot_prev if prev_w else freq
        trend = freq / max(prev_freq, 1e-6)
        pers = persistence_map.get(x_str, 1)
        pfeat = make_pop_features(freq, trend, pers)
        aug   = np.concatenate([bvec, pfeat])
        out.append((aug, bvec, cnt, x_str))
    return out


# ----------------------------------------------------------------- MLP

class MLP:
    def __init__(self, D_in, H, D_out, rng):
        s = lambda n: np.sqrt(2.0 / n)
        self.W1 = rng.normal(0, s(D_in), (D_in, H)).astype(np.float32)
        self.b1 = np.zeros(H,     dtype=np.float32)
        self.W2 = rng.normal(0, s(H),    (H, H)).astype(np.float32)
        self.b2 = np.zeros(H,     dtype=np.float32)
        self.W3 = rng.normal(0, s(H),    (H, D_out)).astype(np.float32)
        self.b3 = np.zeros(D_out, dtype=np.float32)
        self.params = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
        self.grads  = [np.zeros_like(p) for p in self.params]

    def forward(self, x):
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return logits, (x, h1, h2)

    def backward(self, d_logits, cache):
        x, h1, h2 = cache
        dW3 = h2.T @ d_logits;  db3 = d_logits.sum(0)
        dh2 = d_logits @ self.W3.T * (h2 > 0)
        dW2 = h1.T @ dh2;        db2 = dh2.sum(0)
        dh1 = dh2 @ self.W2.T * (h1 > 0)
        dW1 = x.T @ dh1;         db1 = dh1.sum(0)
        for g, d in zip(self.grads,[dW1,db1,dW2,db2,dW3,db3]):
            g += d

    def zero_grad(self): [g.__setitem__(slice(None),0) for g in self.grads]

    def step(self, lr):
        for p, g in zip(self.params, self.grads):
            p -= lr * g


def softmax_masked(logits, mask):
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits) * mask
    return e / (e.sum(1, keepdims=True) + 1e-9)


# --------------------------------------------------------------- training

def train_epoch(net, use_pop_feat, W_enc, t_idx, K, rng, lr, n_pairs, batch):
    pairs = []
    for t in range(1, t_idx + 1):
        cur = W_enc[t - 1]; nxt = W_enc[t]
        cur_cnt = np.array([c for _,_,c,_ in cur], dtype=float)
        cur_cnt /= cur_cnt.sum()
        nxt_cnt = np.array([c for _,_,c,_ in nxt], dtype=float)
        nxt_cnt /= nxt_cnt.sum()
        for _ in range(min(n_pairs, 300)):
            i0 = rng.choice(len(cur), p=cur_cnt)
            i1 = rng.choice(len(nxt), p=nxt_cnt)
            aug0, bv0, _, _ = cur[i0]
            aug1, bv1, _, _ = nxt[i1]
            added = (bv1 - bv0).clip(0)
            if added.sum() == 0:
                continue
            pairs.append((aug0 if use_pop_feat else bv0,
                          bv1, added))

    rng.shuffle(pairs)
    total_loss, nb = 0.0, 0
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i+batch]
        if not chunk: break
        B = len(chunk)
        X0    = np.stack([c[0] for c in chunk])
        X1_bv = np.stack([c[1] for c in chunk])
        added = np.stack([c[2] for c in chunk])

        t_samp = rng.uniform(0, 1, (B, 1))
        include = rng.random((B, K)) < t_samp
        # for pop features, keep aug features fixed, only interpolate binary
        if use_pop_feat:
            Xt_bin = X0[:, :K] + added * include
            Xt = np.concatenate([Xt_bin, X0[:, K:]], axis=1)
        else:
            Xt = X0 + added * include

        mask   = 1.0 - (Xt[:, :K] if use_pop_feat else Xt)
        target = (X1_bv - (Xt[:, :K] if use_pop_feat else Xt)).clip(0)
        tsum   = target.sum(1, keepdims=True) + 1e-9

        net.zero_grad()
        logits, cache = net.forward(Xt)
        probs  = softmax_masked(logits, mask)
        loss   = -(target / tsum * np.log(probs + 1e-9)).sum(1).mean()
        net.backward((probs - target / tsum) / B, cache)
        net.step(lr)
        total_loss += loss; nb += 1

    return total_loss / max(nb, 1)


# --------------------------------------------------------------- Gillespie

def gillespie_novel(rate_fn, W_enc_t, lam, K, rng, n_samples, seen_keys):
    seqs   = W_enc_t
    counts = np.array([c for _,_,c,_ in seqs], dtype=float)
    counts /= counts.sum()
    novel  = Counter()
    attempts = 0
    while sum(novel.values()) < n_samples and attempts < n_samples * 15:
        attempts += 1
        idx  = rng.choice(len(seqs), p=counts)
        aug, bvec, _, _ = seqs[idx]
        x    = bvec.copy()
        rates = rate_fn(aug)          # unconditional uses aug=bvec
        mask  = 1.0 - x
        rates = rates * mask
        total = rates.sum()
        if total < 1e-9: continue
        n_add = max(1, rng.poisson(lam))
        p     = rates / total
        avail = int(mask.sum())
        if avail == 0: continue
        chosen = rng.choice(K, size=min(n_add, avail), replace=False, p=p)
        x[chosen] = 1.0
        key = ",".join(str(i) for i in np.where(x > 0.5)[0])
        if key not in seen_keys:
            novel[key] += 1
    return novel


# --------------------------------------------------------------- scoring

def score_window(inc_pred, nov_pred, floor, W_enc_nxt, seen_strs):
    tot = sum(c for _,_,c,_ in W_enc_nxt)
    nov_total = sum(nov_pred.values()) + 1e-9
    inc_total = sum(inc_pred.values()) + 1e-9
    ll_all = ll_inc = ll_nov = 0.0
    n_inc = n_nov = hit_nov = 0
    for aug, bvec, cnt, s in W_enc_nxt:
        key = ",".join(str(i) for i in np.where(bvec > 0.5)[0])
        p = inc_pred.get(key, 0)/inc_total + nov_pred.get(key, 0)/nov_total * 0.0
        # compose: (1-eps)*inc + eps*nov already normalised externally
        p_inc = inc_pred.get(key, 0)
        p_nov = nov_pred.get(key, 0)
        p = max(p_inc + p_nov, floor)
        lp = math.log(p) * cnt
        ll_all += lp
        if s in seen_strs:
            ll_inc += lp; n_inc += cnt
        else:
            ll_nov += lp; n_nov += cnt
            if p_nov > 0: hit_nov += cnt
    return {
        "ll_all":        ll_all / tot,
        "ll_inc":        ll_inc / n_inc if n_inc else float("nan"),
        "ll_nov":        ll_nov / n_nov if n_nov else float("nan"),
        "novel_frac":    n_nov  / tot,
        "novel_recall":  hit_nov/ n_nov if n_nov else float("nan"),
    }


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--window-days",  type=int,   default=30)
    ap.add_argument("--stride-days",  type=int,   default=30)
    ap.add_argument("--min-seqs",     type=int,   default=50)
    ap.add_argument("--K",            type=int,   default=150)
    ap.add_argument("--H",            type=int,   default=256)
    ap.add_argument("--epochs",       type=int,   default=10)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--n-pairs",      type=int,   default=2000)
    ap.add_argument("--batch",        type=int,   default=128)
    ap.add_argument("--n-samples",    type=int,   default=2000)
    ap.add_argument("--lam",          type=float, default=1.5)
    ap.add_argument("--floor-space",  type=float, default=1e7)
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("-v","--verbose", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print("loading...", file=sys.stderr)
    windows = load_windows(args.data, args.window_days,
                           args.stride_days, args.min_seqs)
    print(f"{len(windows)} windows", file=sys.stderr)

    vocab, v2i = build_vocab(windows, args.K)
    K = len(vocab)
    print(f"vocab size: {K}", file=sys.stderr)

    # build encoded windows with population features
    persistence = {}   # str -> int (windows seen)
    W_enc = []
    for i, w in enumerate(windows):
        prev_w = windows[i-1] if i > 0 else None
        for s in w: persistence[s] = persistence.get(s, 0) + 1
        W_enc.append(encode_window_with_pop(w, prev_w, persistence, v2i, K))

    D_pop   = K + 3   # binary + 3 pop features
    net_cond = MLP(K,     args.H, K, rng)   # binary only
    net_pop  = MLP(D_pop, args.H, K, rng)   # binary + pop features

    # unconditional rates: marginal add_counts
    add_counts = np.zeros(K, dtype=np.float32)

    floor   = 1.0 / args.floor_space
    results = defaultdict(list)
    seen_strs = set()
    eps_hist  = []

    for t in range(len(windows) - 1):
        # update add_counts from current window
        for _, bvec, cnt, s in W_enc[t]:
            if s not in seen_strs:
                add_counts += bvec
        seen_before = set(seen_strs)
        seen_strs.update(s for _,_,_,s in W_enc[t])

        if t < 2:
            continue

        # train both networks
        for ep in range(args.epochs):
            l1 = train_epoch(net_cond, False, W_enc, t, K, rng,
                             args.lr, args.n_pairs, args.batch)
            l2 = train_epoch(net_pop,  True,  W_enc, t, K, rng,
                             args.lr, args.n_pairs, args.batch)
            if args.verbose:
                print(f"  w{t} ep{ep} cond={l1:.3f} pop={l2:.3f}",
                      file=sys.stderr)

        # eps
        cur_w = windows[t]; prev_w = windows[t-1]
        tot = sum(cur_w.values())
        nv  = sum(k for s,k in cur_w.items() if s not in seen_before)
        eps_hist.append(nv/tot)
        eps = float(np.mean(eps_hist[-5:]))
        eps = min(max(eps, 1e-4), 0.5)

        # incumbent predictions (growth model, index-key based)
        g = growth_predict(prev_w, cur_w)
        inc_pred = {}
        for s, p in g.items():
            bvec = encode(s, v2i, K)
            key  = ",".join(str(i) for i in np.where(bvec>0.5)[0])
            inc_pred[key] = (1.0-eps) * p

        seen_keys = set(inc_pred.keys())

        # unconditional CTMC rates
        uc_rates = add_counts / (add_counts.sum() + 1e-9)
        def rate_uncond(aug): return uc_rates.copy()

        # conditioned (binary only)
        def rate_cond(aug):
            bvec = aug[:K]
            logits, _ = net_cond.forward(bvec[None])
            mask = (1.0-bvec)[None]
            return softmax_masked(logits, mask)[0] * mask[0]

        # population-featured
        def rate_pop(aug):
            logits, _ = net_pop.forward(aug[None])
            bvec = aug[:K]
            mask = (1.0-bvec)[None]
            return softmax_masked(logits, mask)[0] * mask[0]

        nov_uc   = gillespie_novel(rate_uncond, W_enc[t], args.lam,
                                   K, rng, args.n_samples, seen_keys)
        nov_cond = gillespie_novel(rate_cond,   W_enc[t], args.lam,
                                   K, rng, args.n_samples, seen_keys)
        nov_pop  = gillespie_novel(rate_pop,    W_enc[t], args.lam,
                                   K, rng, args.n_samples, seen_keys)

        def normalise_nov(nov):
            sc = sum(nov.values()) + 1e-9
            return {k: eps*v/sc for k,v in nov.items()}

        obs = W_enc[t+1]
        for name, nov in [("uncond", nov_uc),
                           ("cond",   nov_cond),
                           ("pop",    nov_pop)]:
            r = score_window(inc_pred, normalise_nov(nov),
                             floor, obs, seen_before)
            for k, v in r.items():
                if not math.isnan(v):
                    results[f"{name}_{k}"].append(v)

        if args.verbose:
            for name in ("uncond","cond","pop"):
                rec = results[f"{name}_novel_recall"]
                print(f"  w{t} {name} recall={np.mean(rec):.3f}",
                      file=sys.stderr)

    print(f"\n{'metric':<16} {'uncond':>9} {'cond':>9} {'pop+freq':>9}")
    print("-" * 47)
    for k in ("ll_all","ll_inc","ll_nov","novel_recall"):
        vs = []
        for name in ("uncond","cond","pop"):
            v = results.get(f"{name}_{k}", [])
            vs.append(f"{np.mean(v):9.4f}" if v else "       NA")
        best = np.argmax([float(v) for v in vs])
        markers = ["  ","  ","  "]; markers[best] = "<-"
        print(f"{k:<16} {vs[0]}{markers[0]} {vs[1]}{markers[1]}"
              f" {vs[2]}{markers[2]}")
    print("\n(log scores: higher is better; <- marks best per row)")


if __name__ == "__main__":
    main()
