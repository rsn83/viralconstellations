"""
93_mixture_spike_v2.py

Bernoulli mixture model on FULL reconstructed amino acid sequences.

Each component k has a (L, D) emission matrix over 20 amino acids.
Mixing weights π_k(t) = softmax(a_k + v_k · t) vary with collection month.

Model:
    P(residue r at pos j | t) = Σ_k π_k(t) · P_k[j, r]

Fit by EM:
    E step: soft assignment of each sequence to components
    M step: update P_k from weighted residue counts, update π via logistic

Same table format as 92_spike_timemue_v3.py for direct comparison.

Usage:
    python scripts/93_mixture_spike_v2.py \
        --data  data/processed/full_data_graphs_posres \
        --vocab data/processed/full_data_graphs_posres/posres_vocab.tsv \
        --ref   data/raw/spike_reference.fasta \
        --train-end 2022-12 --n-per-month 500 --K 20

    python scripts/93_mixture_spike_v2.py --synthetic --K 3
"""
import argparse, glob, os, pickle, sys, time
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
D = len(AA)


# ------------------------------------------------------------------ data
def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip())
    return "".join(seq)


def read_vocab(path):
    m = {}
    with open(path) as f:
        hdr = {c: i for i, c in enumerate(f.readline().rstrip("\n").split("\t"))}
        for line in f:
            p = line.rstrip("\n").split("\t")
            node = int(p[hdr["node_idx"]])
            pos  = int(p[hdr["aa_pos"]])
            res  = p[hdr["residue"]]
            if res in AA_IDX:
                m[node] = (pos, res)
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


def reconstruct(ref, node_map, s):
    seq = list(ref)
    for n in s:
        hit = node_map.get(n)
        if hit and 1 <= hit[0] <= len(ref):
            seq[hit[0] - 1] = hit[1]
    return "".join(seq)


def encode(seq):
    return np.array([AA_IDX.get(c, 0) for c in seq], dtype=np.int32)


def load_all(data_dir, ref, node_map, n_per_month, seed=0):
    rng = np.random.default_rng(seed)
    months, Y_by_month = [], {}
    L = len(ref)
    for p in sorted(glob.glob(os.path.join(data_dir, "*_occupied.pkl"))):
        ym = os.path.basename(p).split("_")[0]
        sets = load_month(p)
        if not sets:
            continue
        if n_per_month and len(sets) > n_per_month:
            sets = [sets[i] for i in
                    rng.choice(len(sets), n_per_month, replace=False)]
        months.append(ym)
        seqs = np.stack([encode(reconstruct(ref, node_map, s)) for s in sets])
        Y_by_month[ym] = seqs
    return months, Y_by_month, L


# -------------------------------------------------------------- synthetic
def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    L = 100
    ref_seq = np.zeros(L, dtype=np.int32)
    blocks = {
        0: (rng.choice(L, 5,  replace=False), rng.integers(1, D, 5)),
        1: (rng.choice(L, 10, replace=False), rng.integers(1, D, 10)),
        2: (rng.choice(L, 20, replace=False), rng.integers(1, D, 20)),
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
        for _ in range(150):
            seq = ref_seq.copy()
            u, acc = rng.random() * tot, 0.0
            for k, w in enumerate(ws):
                acc += w
                if u < acc:
                    pos, res = blocks[k]
                    keep = rng.random(len(pos)) < 0.9
                    seq[pos[keep]] = res[keep]
                    break
            rows.append(seq)
        Y_by_month[ym] = np.stack(rows)
    return months, Y_by_month, L


# -------------------------------------------------------------------  EM
def fit_mixture(Y_cat, K, max_iter=80, tol=1e-4, reg=1e-3, seed=0):
    """
    EM for multinomial mixture model on integer-encoded sequences.
    Y_cat : (N, L) int32   residue index per position
    Returns:
        P  : (K, L, D)  per-component per-position residue distributions
        pi : (K,)       mixing weights
        z  : (N, K)     soft assignments
    """
    rng = np.random.default_rng(seed)
    N, L = Y_cat.shape

    # initialise: K random sequences from data as cluster centres
    idx = rng.choice(N, K, replace=False)
    P = np.full((K, L, D), reg)
    for k, i in enumerate(idx):
        P[k, np.arange(L), Y_cat[i]] += 1.0
    P /= P.sum(-1, keepdims=True)

    pi = np.ones(K) / K
    prev_ll = -np.inf

    for it in range(max_iter):
        # ---- E step: log p(y_i | k) = sum_j log P_k[j, y_ij] ----
        # build (N, K) log-likelihood matrix efficiently
        ll_k = np.zeros((N, K))
        for k in range(K):
            # log P_k[j, y_ij] for each i,j
            ll_k[:, k] = np.log(np.clip(
                P[k, np.arange(L)[None, :], Y_cat],
                1e-12, 1)).sum(1)
        ll_k += np.log(pi)[None, :]

        # stable softmax to get responsibilities
        ll_k -= ll_k.max(1, keepdims=True)
        z = np.exp(ll_k)
        z /= z.sum(1, keepdims=True)       # (N, K)

        # ---- M step ----
        Nk = z.sum(0) + 1e-10              # (K,)
        pi = Nk / Nk.sum()

        # weighted residue counts per component
        P = np.full((K, L, D), reg)
        for k in range(K):
            wk = z[:, k]                   # (N,)
            for d in range(D):
                mask = Y_cat == d          # (N, L) bool
                P[k, :, d] += (wk[:, None] * mask).sum(0)
        P /= P.sum(-1, keepdims=True)

        ll = float((z * ll_k).sum())
        if abs(ll - prev_ll) < tol * (abs(prev_ll) + 1):
            print(f"  EM converged at iter {it}")
            break
        prev_ll = ll

    return P, pi, z


# ----------------------------------------- time-varying mixing weights
def fit_mixing_weights(z, t_tr, K, wd=1e-2):
    """
    Fit π_k(t) = softmax(a_k + v_k·t) from soft assignments.
    Returns a (K,), v (K,).
    """
    from scipy.optimize import minimize
    N = len(t_tr)
    X = np.stack([np.ones(N), t_tr], 1)   # (N, 2)
    a = np.zeros(K); v = np.zeros(K)

    def obj(params):
        theta = params.reshape(2, K)       # (2, K)
        logits = X @ theta                 # (N, K)
        logits -= logits.max(1, keepdims=True)
        pi_n = np.exp(logits)
        pi_n /= pi_n.sum(1, keepdims=True)
        nll = -(z * np.log(np.clip(pi_n, 1e-12, 1))).sum() / N
        reg = 0.5 * wd * (theta**2).sum()
        grad = -(X.T @ (z - pi_n)) / N + wd * theta
        return float(nll + reg), grad.ravel()

    res = minimize(obj, np.zeros(2 * K), jac=True, method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-10})
    theta = res.x.reshape(2, K)
    return theta[0], theta[1]


def mixing_weights(a, v, t_scalar):
    logits = a + v * t_scalar
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def marginal_probs(P, a, v, t_scalar):
    """
    Marginal per-position residue distribution at time t.
    Returns (L, D).
    """
    pi = mixing_weights(a, v, t_scalar)   # (K,)
    return (pi[:, None, None] * P).sum(0) # (L, D)


# ---------------------------------------------------------------- metrics
def perplexity(P_ld, Y_cat):
    """
    P_ld : (L, D)  predicted residue probs
    Y_cat: (N, L)  true residues
    """
    lp = np.log(np.clip(
        P_ld[np.arange(Y_cat.shape[1])[None, :], Y_cat],
        1e-12, 1)).mean()
    return float(np.exp(-lp))


def mae_freq(P_ld, Y_cat):
    """MAE between predicted and actual per-position residue frequencies."""
    act = np.zeros_like(P_ld)
    for d in range(D):
        act[:, d] = (Y_cat == d).mean(0)
    var = act.max(1) < 0.99
    return float(np.abs(P_ld[var] - act[var]).mean())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--vocab"); ap.add_argument("--ref")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--train-end", default="2022-12")
    ap.add_argument("--n-per-month", type=int, default=500)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--em-iter", type=int, default=80)
    a = ap.parse_args()

    if a.synthetic:
        months, Y_by_month, L = synthetic()
    else:
        if not all([a.data, a.vocab, a.ref]):
            sys.exit("need --data --vocab --ref  (or --synthetic)")
        ref      = read_fasta(a.ref)
        node_map = read_vocab(a.vocab)
        print(f"reference length {len(ref)} aa, vocab {len(node_map)} nodes")
        months, Y_by_month, L = load_all(a.data, ref, node_map, a.n_per_month)

    print(f"loaded {len(months)} months: {months[0]}..{months[-1]}  L={L}")

    tr_m = [m for m in months if m <= a.train_end]
    te_m = [m for m in months if m >  a.train_end]
    print(f"train {tr_m[0]}..{tr_m[-1]} ({len(tr_m)} mo)  "
          f"test  {te_m[0]}..{te_m[-1]} ({len(te_m)} mo)")

    mi   = {m: i for i, m in enumerate(months)}
    mu   = np.mean([mi[m] for m in tr_m])
    sd   = np.std( [mi[m] for m in tr_m])
    t_sc = lambda m: (mi[m] - mu) / sd

    Y_tr = np.concatenate([Y_by_month[m] for m in tr_m])
    t_tr = np.array([t_sc(m) for m in tr_m
                     for _ in range(len(Y_by_month[m]))])

    # persistence emission from last training month
    Y_persist = Y_by_month[tr_m[-1]]
    P_persist = np.zeros((L, D))
    for d in range(D):
        P_persist[:, d] = (Y_persist == d).mean(0)
    P_persist = np.clip(P_persist, 1e-8, 1)

    # EM
    print(f"\nfitting mixture  K={a.K}  N={len(Y_tr)} seqs  L={L} ...",
          flush=True)
    t0 = time.time()
    P, pi, z = fit_mixture(Y_tr, a.K, a.em_iter)
    print(f"  EM done  {time.time()-t0:.1f}s")

    # time-varying weights
    print("fitting mixing weights ...", end=" ", flush=True)
    t0 = time.time()
    a_w, v_w = fit_mixing_weights(z, t_tr, a.K)
    print(f"{time.time()-t0:.1f}s")

    # script-92v3 numbers for comparison (from your run)
    ppl92 = {
        "2023-01":1.03453,"2023-02":1.04648,"2023-03":1.05473,"2023-04":1.06104,
        "2023-05":1.06438,"2023-06":1.06798,"2023-07":1.07224,"2023-08":1.07621,
        "2023-09":1.08001,"2023-10":1.08322,"2023-11":1.09276,"2023-12":1.11860,
        "2024-01":1.14204,"2024-02":1.14755,"2024-03":1.15412,"2024-04":1.16141,
        "2024-05":1.17067,"2024-06":1.17538,"2024-07":1.17464,"2024-08":1.17778,
        "2024-09":1.18091,"2024-10":1.18503,"2024-11":1.18778,"2024-12":1.19409,
        "2025-01":1.19284,"2025-02":1.20220,"2025-03":1.21080,"2025-04":1.21532,
        "2025-05":1.22176,"2025-06":1.22959,"2025-07":1.23425,"2025-08":1.23943,
        "2025-09":1.24292,"2025-10":1.25035,"2025-11":1.25402,"2025-12":1.25700,
        "2026-01":1.25475,"2026-02":1.25525,"2026-03":1.27627,"2026-04":1.26710,
        "2026-05":1.28230,
    }

    print(f"\n=== MIXTURE K={a.K}  train<={a.train_end} ===")
    print(f"\n{'month':<10}{'ppl_mix':>9}{'ppl_92':>9}{'ppl_pers':>10}"
          f"{'delta_92':>10}{'mix>p':>7}{'92>p':>6}")
    print("-" * 62)

    for m in te_m:
        Ym   = Y_by_month[m]
        tm   = t_sc(m)
        Pmix = marginal_probs(P, a_w, v_w, tm)   # (L, D)
        pm   = perplexity(Pmix,    Ym)
        pp   = perplexity(P_persist, Ym)
        p92  = ppl92.get(m, float("nan"))
        delta = p92 - pm                          # positive = mixture better
        mix_beats = "✓" if pm < pp else "✗"
        p92_beats = "✓" if p92 < pp else "✗"
        print(f"{m:<10}{pm:>9.5f}{p92:>9.5f}{pp:>10.5f}"
              f"{delta:>+10.5f}{mix_beats:>7}{p92_beats:>6}")

    print(f"""
COLUMNS
  ppl_mix   per-residue perplexity, multinomial mixture K={a.K}
  ppl_92    per-residue perplexity, time-varying logistic (script 92v3)
  ppl_pers  per-residue perplexity, persistence
  delta_92  ppl_92 - ppl_mix  (positive = mixture better than logistic)
  mix>p     ✓ if mixture beats persistence
  92>p      ✓ if logistic beats persistence
""")

    print("Component summary:")
    print(f"  mean mutations/component: "
          f"{(P.argmax(-1) != 0).sum(-1).min()} - "
          f"{(P.argmax(-1) != 0).sum(-1).max()}")
    print(f"  mixing weights at train-end: "
          f"{mixing_weights(a_w, v_w, t_sc(tr_m[-1])).round(3)}")


if __name__ == "__main__":
    main()
