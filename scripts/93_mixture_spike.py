"""
93_mixture_spike.py

Bernoulli mixture model with time-varying weights on full-length spike.

Model:
    P(mutation at position j | t) = Σ_k π_k(t) · p_{j|k}

    p_{j|k}  : per-position mutation probability for component k  (learned by EM)
    π_k(t)   : softmax( a_k + v_k · t )                          (learned by logistic)

This is the lineage-aware generalization of 92_spike_timemue_v2.py.
The key question: does conditioning on latent lineage identity beat persistence,
where the single per-position logistic (script 92) could not?

Same table format as script 92 for direct comparison.

Usage:
    python scripts/93_mixture_spike.py \
        --data  data/processed/full_data_graphs_posres \
        --vocab data/processed/full_data_graphs_posres/posres_vocab.tsv \
        --train-end 2022-12 --n-per-month 2000 --K 20

    python scripts/93_mixture_spike.py --synthetic --K 3
"""
import argparse, glob, os, pickle, sys, time
import numpy as np
from collections import Counter

# ----------------------------------------------------------------- data
def read_vocab(path):
    m = {}
    with open(path) as f:
        hdr = {c: i for i, c in enumerate(f.readline().rstrip("\n").split("\t"))}
        for line in f:
            p = line.rstrip("\n").split("\t")
            m[int(p[hdr["node_idx"]])] = (int(p[hdr["aa_pos"]]),
                                           p[hdr["residue"]])
    return m


def load_month(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        if all(isinstance(k, (frozenset, set, tuple)) for k in list(obj)[:5]):
            out = []
            for k, v in obj.items():
                out.extend([frozenset(k)] * int(v))
            return out
        for key in ("sets", "constellations", "occupied"):
            if key in obj:
                return [frozenset(s) for s in obj[key]]
    return [frozenset(s) for s in obj]


def load_all(data_dir, node_map, L, n_per_month, seed=0):
    rng = np.random.default_rng(seed)
    months, Y_by_month = [], {}
    for p in sorted(glob.glob(os.path.join(data_dir, "*_occupied.pkl"))):
        ym = os.path.basename(p).split("_")[0]
        sets = load_month(p)
        if not sets:
            continue
        if n_per_month and len(sets) > n_per_month:
            sets = [sets[i] for i in
                    rng.choice(len(sets), n_per_month, replace=False)]
        months.append(ym)
        rows = np.zeros((len(sets), L), dtype=np.float32)
        for i, s in enumerate(sets):
            for n in s:
                hit = node_map.get(n)
                if hit and 1 <= hit[0] <= L:
                    rows[i, hit[0] - 1] = 1.0
        Y_by_month[ym] = rows
    return months, Y_by_month


# -------------------------------------------------------------- synthetic
def synthetic(seed=0):
    """Three true lineages (blocks), same wave dynamics as script 92."""
    rng = np.random.default_rng(seed)
    L = 1273
    blocks = {
        0: rng.choice(L, 8,  replace=False),  # Alpha-like
        1: rng.choice(L, 15, replace=False),  # Delta-like
        2: rng.choice(L, 30, replace=False),  # Omicron-like
    }
    months = [f"{y}-{m:02d}" for y in range(2020, 2025) for m in range(1, 13)]
    Y_by_month = {}
    for i, ym in enumerate(months):
        t = i / len(months)
        wA = max(0.0, np.exp(-((t - 0.25)**2) / 0.01))
        wB = max(0.0, np.exp(-((t - 0.50)**2) / 0.01))
        wC = max(0.0, 2.5 * (t - 0.65)) if t > 0.65 else 0.0
        ws = [wA, wB, wC]; tot = sum(ws) + 0.1
        rows = []
        for _ in range(200):
            v = np.zeros(L, np.float32)
            u, acc = rng.random() * tot, 0.0
            for k, w in enumerate(ws):
                acc += w
                if u < acc:
                    keep = rng.random(len(blocks[k])) < 0.9
                    v[blocks[k][keep]] = 1.0
                    break
            rows.append(v)
        Y_by_month[ym] = np.stack(rows)
    return months, Y_by_month


# ----------------------------------------------------------- EM for BMM
def fit_bmm(Y, K, max_iter=100, tol=1e-4, seed=0, reg=1e-3):
    """
    Bernoulli Mixture Model via EM.
    Y : (N, L) binary float32
    Returns:
        P : (K, L)  per-component per-position mutation probabilities
        pi: (K,)    mixing weights
        z : (N, K)  soft assignments (responsibilities)
    """
    rng = np.random.default_rng(seed)
    N, L = Y.shape
    # init: K random cluster centers from data
    idx = rng.choice(N, K, replace=False)
    P = Y[idx].astype(np.float64) + rng.random((K, L)) * 0.1
    P = np.clip(P, reg, 1 - reg)
    pi = np.ones(K) / K

    prev_ll = -np.inf
    for it in range(max_iter):
        # E step: responsibilities
        logP = np.log(P)        # (K, L)
        log1P = np.log(1 - P)  # (K, L)
        # log p(y_i | k) = y_i·logP_k + (1-y_i)·log(1-P_k)
        ll_k = Y @ logP.T + (1 - Y) @ log1P.T  # (N, K)
        ll_k += np.log(pi)[None, :]
        # stable softmax
        ll_k -= ll_k.max(1, keepdims=True)
        z = np.exp(ll_k)
        z /= z.sum(1, keepdims=True)            # (N, K)

        # M step
        Nk = z.sum(0) + 1e-10                  # (K,)
        pi = Nk / Nk.sum()
        P = (z.T @ Y) / Nk[:, None]            # (K, L)
        P = np.clip(P, reg, 1 - reg)

        # log likelihood
        ll = float((z * ll_k).sum())
        if abs(ll - prev_ll) < tol * abs(prev_ll + 1):
            break
        prev_ll = ll

    return P, pi, z


# ------------------------------------------- time-varying mixture weights
def fit_mixing_weights(z_by_month, month_indices, t_scaled, K, wd=1e-2):
    """
    For each month, soft responsibilities z (N_m, K).
    Fit multinomial logistic: log π_k(t) = a_k + v_k · t
    Returns a (K,), v (K,).
    """
    # aggregate to monthly mean responsibilities
    t_m = np.array([t_scaled[month_indices == i].mean()
                    for i in range(int(month_indices.max()) + 1)
                    if (month_indices == i).any()])
    pi_m = np.array([z_by_month[i].mean(0)
                     for i in sorted(z_by_month.keys())])  # (n_months, K)

    # fit K-1 logistic regressors (vs last class)
    a = np.zeros(K); v = np.zeros(K)
    X = np.stack([np.ones_like(t_m), t_m], 1)  # (n_months, 2)
    for k in range(K - 1):
        y = pi_m[:, k]
        w = np.array([0.0, 0.0])
        for _ in range(80):
            p = np.clip(1 / (1 + np.exp(-X @ w)), 1e-7, 1 - 1e-7)
            g = X.T @ (y - p) - wd * w
            H = -(X.T * (p * (1 - p))) @ X - wd * np.eye(2)
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        a[k], v[k] = w[0], w[1]
    return a, v


def mixing_weights(a, v, t_scalar):
    """π_k(t) via softmax. Shape (K,)."""
    logits = a + v * t_scalar
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def predict_p(P, a, v, t_scalar):
    """
    Marginal per-position mutation probability at time t.
    P : (K, L), returns (L,)
    """
    pi = mixing_weights(a, v, t_scalar)   # (K,)
    return (pi[:, None] * P).sum(0)       # (L,)


# -------------------------------------------------------------- metrics
def eval_month(P, a, v, t_m, Y_m, Y_persist):
    p_mix  = predict_p(P, a, v, t_m)
    p_pers = Y_persist.mean(0)
    p_act  = Y_m.mean(0)
    var    = (p_act > 0.01) | (p_mix > 0.01)

    def mae(p): return float(np.abs(p[var] - p_act[var]).mean())
    def top_overlap(p, k):
        return len(set(np.argsort(-p)[:k]) &
                   set(np.argsort(-p_act)[:k])) / k

    return dict(
        mae_mix   = mae(p_mix),
        mae_pers  = mae(p_pers),
        top20     = top_overlap(p_mix, 20),
        top20_p   = top_overlap(p_pers, 20),
        pred_setsz= float(p_mix.sum()),
        act_setsz = float(p_act.sum()),
    )


def print_table(rows, prev_mae_time=None):
    """Same format as script 92 for direct comparison.
    prev_mae_time: dict {month: mae_time} from script 92 if available."""
    hdr = (f"{'month':<10}{'MAE_mix':>9}{'MAE_92':>9}{'MAE_pers':>10}"
           f"{'top20':>7}{'top20_p':>8}{'pred_sz':>9}{'act_sz':>8}{'beats':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        prev = f"{prev_mae_time[r['month']]:.4f}" \
               if prev_mae_time and r['month'] in prev_mae_time else "  -   "
        beats = "✓" if r["mae_mix"] < r["mae_pers"] else "✗"
        improve = ""
        if prev_mae_time and r['month'] in prev_mae_time:
            delta = prev_mae_time[r['month']] - r['mae_mix']
            improve = f"({'+' if delta>0 else ''}{delta:.4f})"
        print(f"{r['month']:<10}{r['mae_mix']:>9.4f}{prev:>9}"
              f"{r['mae_pers']:>10.4f}{r['top20']:>7.2f}{r['top20_p']:>8.2f}"
              f"{r['pred_setsz']:>9.1f}{r['act_setsz']:>8.1f}{beats:>7}  {improve}")

    print("""
COLUMNS
  MAE_mix   mixture model (this script)
  MAE_92    logistic from script 92 (pasted via --prev-results or absent)
  MAE_pers  persistence
  beats     ✓ if MAE_mix < MAE_pers
  (delta)   how much better mixture is vs script-92 logistic (positive = improvement)
""")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--vocab")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--train-end", default="2022-12")
    ap.add_argument("--n-per-month", type=int, default=2000)
    ap.add_argument("--K", type=int, default=20,
                    help="number of mixture components (lineages)")
    ap.add_argument("--em-iter", type=int, default=100)
    a = ap.parse_args()

    if a.synthetic:
        months, Y_by_month = synthetic(); L = 1273
    else:
        if not all([a.data, a.vocab]):
            sys.exit("need --data and --vocab (or --synthetic)")
        node_map = read_vocab(a.vocab)
        L = max(pos for pos, _ in node_map.values())
        print(f"spike length {L}, vocab {len(node_map)} nodes")
        months, Y_by_month = load_all(a.data, node_map, L, a.n_per_month)

    print(f"loaded {len(months)} months: {months[0]}..{months[-1]}")

    tr_m = [m for m in months if m <= a.train_end]
    te_m = [m for m in months if m > a.train_end]
    print(f"train {tr_m[0]}..{tr_m[-1]} ({len(tr_m)} mo)  "
          f"test {te_m[0]}..{te_m[-1]} ({len(te_m)} mo)")

    mi = {m: i for i, m in enumerate(months)}
    mu = np.mean([mi[m] for m in tr_m])
    sd = np.std( [mi[m] for m in tr_m])
    t_sc = lambda m: (mi[m] - mu) / sd

    # stack training data
    Y_tr = np.concatenate([Y_by_month[m] for m in tr_m])
    t_tr = np.array([t_sc(m) for m in tr_m
                     for _ in range(len(Y_by_month[m]))])
    mi_tr = np.array([mi[m] for m in tr_m
                      for _ in range(len(Y_by_month[m]))])

    # --- EM ---
    print(f"\nfitting Bernoulli mixture  K={a.K}  N={len(Y_tr)} ...",
          end=" ", flush=True)
    t0 = time.time()
    P, pi, z = fit_bmm(Y_tr, a.K, a.em_iter)
    print(f"{time.time()-t0:.1f}s")

    # soft assignments per training month
    z_by_month = {}
    for m in tr_m:
        mask = mi_tr == mi[m]
        z_by_month[mi[m]] = z[mask]

    # --- time-varying mixing weights ---
    print("fitting mixing weights ...", end=" ", flush=True)
    t0 = time.time()
    a_w, v_w = fit_mixing_weights(z_by_month, mi_tr, t_tr, a.K)
    print(f"{time.time()-t0:.1f}s")

    # --- evaluate ---
    Y_persist = Y_by_month[tr_m[-1]]
    rows = []
    for m in te_m:
        r = eval_month(P, a_w, v_w, t_sc(m), Y_by_month[m], Y_persist)
        r["month"] = m
        rows.append(r)

    # script-92 MAE_time values for comparison
    prev = {
        "2023-01":0.0907,"2023-02":0.1356,"2023-03":0.1586,"2023-04":0.1710,
        "2023-05":0.1727,"2023-06":0.1690,"2023-07":0.1763,"2023-08":0.1797,
        "2023-09":0.1730,"2023-10":0.1634,"2023-11":0.1805,"2023-12":0.2399,
        "2024-01":0.3054,"2024-02":0.3380,"2024-03":0.3414,"2024-04":0.3441,
        "2024-05":0.3522,"2024-06":0.3465,"2024-07":0.3561,"2024-08":0.3612,
        "2024-09":0.3593,"2024-10":0.3649,"2024-11":0.3661,"2024-12":0.3644,
        "2025-01":0.3721,"2025-02":0.3722,"2025-03":0.3781,"2025-04":0.3744,
        "2025-05":0.3925,"2025-06":0.3926,"2025-07":0.3855,"2025-08":0.3776,
        "2025-09":0.3796,"2025-10":0.3827,"2025-11":0.3757,"2025-12":0.3746,
        "2026-01":0.3828,"2026-02":0.3827,"2026-03":0.4016,"2026-04":0.3843,
        "2026-05":0.3901,
    }

    print(f"\n=== MIXTURE MODEL K={a.K}  train<={a.train_end} ===")
    print_table(rows, prev)

    # component summary
    print(f"\nComponent summary (K={a.K}):")
    print(f"  mean mutations per component: "
          f"{P.sum(1).min():.1f} - {P.sum(1).max():.1f}")
    print(f"  mixing weights at train-end: "
          f"{mixing_weights(a_w, v_w, t_sc(tr_m[-1])).round(3)}")


if __name__ == "__main__":
    main()
