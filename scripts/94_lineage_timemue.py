"""
94_lineage_timemue.py

Full time-varying lineage mixture model on reconstructed spike sequences.

Model:
    z_i ~ Categorical(π(t_i))                     latent lineage per sequence
    π_k(t) = softmax(a_k + v_k · t)               time-varying mixing weights
    P(residue r at pos j | z=k, t) =
        softmax(B_k[j,r] + W_k[j,r] · t)          time-varying per-lineage emissions

This combines:
    - Lineage structure (script 93): correlations via shared latent z
    - Time-varying emissions (script 92v3): within-lineage drift

Fit by EM:
    E step : soft assignment z given current params
    M step : update B_k, W_k by L-BFGS-B per component per position
             update a_k, v_k by multinomial logistic on soft assignments

Same table as 92v3 for direct comparison.

Usage:
    python scripts/94_lineage_timemue.py \
        --data  data/processed/full_data_graphs_posres \
        --vocab data/processed/full_data_graphs_posres/posres_vocab.tsv \
        --ref   data/raw/spike_reference.fasta \
        --train-end 2022-12 --n-per-month 500 --K 5

    python scripts/94_lineage_timemue.py --synthetic --K 3
"""
import argparse, glob, os, pickle, sys, time
import numpy as np
from scipy.optimize import minimize

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
    L = 80
    ref_seq = np.zeros(L, dtype=np.int32)
    blocks = {
        0: (rng.choice(L, 5,  replace=False), rng.integers(1, D, 5)),
        1: (rng.choice(L, 12, replace=False), rng.integers(1, D, 12)),
        2: (rng.choice(L, 25, replace=False), rng.integers(1, D, 25)),
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


# ----------------------------------------------------------------- model
def softmax_2d(logits):
    """logits: (..., D) → probabilities"""
    l = logits - logits.max(-1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(-1, keepdims=True)


def emission_probs(B_logprob, W_shared, t_scalar, t_train_end=0.0):
    """
    B_logprob  : (K, L, D)  log prob per component at training end
    W_shared   : (L, D)     shared slope in logit space
    t_train_end: scalar     scaled time of last training month

    At t = t_train_end: logits = B_logprob (component profile, correct)
    At future t:        logits = B_logprob + W*(t - t_train_end) (drift correction)
    """
    dt = t_scalar - t_train_end
    logits = B_logprob + W_shared[None, :, :] * dt
    logits = np.clip(logits, -10.0, 10.0)
    return softmax_2d(logits)


def mixing_weights(a, v, t_scalar):
    """a, v: (K,) → π: (K,)"""
    logits = a + v * t_scalar
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def marginal_probs(B, W_shared, a, v, t_scalar, t_train_end=0.0):
    """Marginal P(residue r at pos j | t). Returns (L, D)."""
    pi  = mixing_weights(a, v, t_scalar)
    P_k = emission_probs(B, W_shared, t_scalar, t_train_end)
    return (pi[:, None, None] * P_k).sum(0)


# ------------------------------------------------------------------- EM
def e_step(B, W_shared, a, v, Y_cat, t_arr, t_train_end=0.0):
    """
    Compute soft assignments z: (N, K).
    Y_cat: (N, L), t_arr: (N,)
    """
    N, L = Y_cat.shape
    K = B.shape[0]
    log_z = np.zeros((N, K))

    unique_t = np.unique(t_arr)
    for t_val in unique_t:
        mask = t_arr == t_val
        pi  = mixing_weights(a, v, t_val)
        P_k = emission_probs(B, W_shared, t_val, t_train_end)  # (K, L, D)
        for k in range(K):
            log_z[mask, k] = (
                np.log(np.clip(P_k[k, np.arange(L), Y_cat[mask]], 1e-12, 1))
                .sum(1) + np.log(pi[k])
            )

    log_z -= log_z.max(1, keepdims=True)
    z = np.exp(log_z)
    z /= z.sum(1, keepdims=True)
    return z


def m_step_emissions(B_logprob, W_shared, z, Y_cat, t_arr, wd=0.1):
    """
    Update B_k and W_shared.

    W_shared: fit as global time trend using ALL sequences pooled.
              This is the same as script 92v3 logistic — stable because
              all N sequences contribute to every position.

    B_k: per-component log frequency of residues in sequences assigned
         to component k (weighted by z). Captures lineage-specific profile
         on top of the global trend.
    """
    from scipy.optimize import minimize as sp_minimize
    N, L = Y_cat.shape
    K = B_logprob.shape[0]
    X = np.stack([np.ones(N), t_arr], 1)   # (N, 2)
    B_new = B_logprob.copy()
    W_new = W_shared.copy()

    # --- W_shared: pooled logistic on all sequences (stable, N data points) ---
    for j in range(L):
        yj = Y_cat[:, j]
        if np.bincount(yj, minlength=D).max() == N:
            W_new[j] = 0.0
            continue

        def obj(params):
            theta = params.reshape(2, D)
            logits = X @ theta
            logits -= logits.max(1, keepdims=True)
            probs = softmax_2d(logits)
            probs = np.clip(probs, 1e-12, 1)
            nll = -np.log(probs[np.arange(N), yj]).mean()
            reg = 0.5 * wd * (theta**2).sum()
            Y_oh = np.zeros((N, D)); Y_oh[np.arange(N), yj] = 1.0
            grad = -(X.T @ (Y_oh - probs)) / N + wd * theta
            return float(nll + reg), grad.ravel()

        theta0 = np.stack([B_new.mean(0)[j], W_shared[j]], 0)
        res = sp_minimize(obj, theta0.ravel(), jac=True, method="L-BFGS-B",
                         options={"maxiter": 100, "ftol": 1e-8},
                         bounds=[(-5,5)]*(2*D))
        theta = res.x.reshape(2, D)
        # only update W; intercept absorbed into B_k below
        W_new[j] = theta[1]

    # --- B_k: weighted residue frequencies per component ---
    # B_k captures the lineage-specific deviation from the global trend
    for k in range(K):
        wk = z[:, k]
        if wk.sum() < 1.0:
            continue
        freq = np.zeros((L, D))
        for d in range(D):
            freq[:, d] = (wk[:, None] * (Y_cat == d)).sum(0) / (wk.sum() + 1e-10)
        # store as log prob — when used as logit, softmax recovers the freq
        B_new[k] = np.log(np.clip(freq, 1e-12, 1))

    return B_new, W_new


def m_step_weights(z, t_arr, K, wd=1e-2):
    """Update a, v via multinomial logistic on soft assignments."""
    N = len(t_arr)
    X = np.stack([np.ones(N), t_arr], 1)

    def obj(params):
        theta = params.reshape(2, K)
        logits = X @ theta
        logits -= logits.max(1, keepdims=True)
        pi_n = softmax_2d(logits)
        nll = -(z * np.log(np.clip(pi_n, 1e-12, 1))).sum() / N
        reg = 0.5 * wd * (theta**2).sum()
        grad = -(X.T @ (z - pi_n)) / N + wd * theta
        return float(nll + reg), grad.ravel()

    res = minimize(obj, np.zeros(2 * K), jac=True, method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-10})
    theta = res.x.reshape(2, K)
    return theta[0], theta[1]


def init_params(Y_cat, t_arr, K, seed=0):
    """
    Time-based initialisation: divide training window into K equal eras,
    initialise component k from sequences in era k.
    Biologically: early / mid / late training eras.
    Falls back to random if K > number of unique time points.
    """
    N, L = Y_cat.shape
    unique_t = np.unique(t_arr)
    B = np.zeros((K, L, D))
    W_shared = np.zeros((L, D))   # shared slope, initialised to zero
    a = np.zeros(K)
    v = np.zeros(K)

    if K <= len(unique_t):
        # split training time into K equal eras
        boundaries = np.array_split(unique_t, K)
        for k, era_t in enumerate(boundaries):
            mask = np.isin(t_arr, era_t)
            if mask.sum() == 0:
                continue
            Y_era = Y_cat[mask]
            # per-position residue frequencies in this era
            for d in range(D):
                B[k, :, d] = (Y_era == d).mean(0)
            B[k] = np.log(np.clip(B[k], 1e-8, 1))  # log for logit space
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, K, replace=False)
        for k, i in enumerate(idx):
            B[k, np.arange(L), Y_cat[i]] = 2.0

    return B, W_shared, a, v


# -------------------------------------------------------------- metrics
def perplexity(P_ld, Y_cat):
    lp = np.log(np.clip(
        P_ld[np.arange(Y_cat.shape[1])[None, :], Y_cat],
        1e-12, 1)).mean()
    return float(np.exp(-lp))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--vocab"); ap.add_argument("--ref")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--train-end", default="2022-12")
    ap.add_argument("--n-per-month", type=int, default=500)
    ap.add_argument("--K", type=int, default=5,
                    help="number of lineage components (keep small: 3-10)")
    ap.add_argument("--em-iter", type=int, default=10)
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

    # persistence
    Y_persist = Y_by_month[tr_m[-1]]
    P_persist = np.zeros((L, D))
    for d in range(D):
        P_persist[:, d] = (Y_persist == d).mean(0)
    P_persist = np.clip(P_persist, 1e-8, 1)

    # EM
    print(f"\nfitting time-varying lineage mixture  K={a.K}  "
          f"N={len(Y_tr)}  L={L}")
    print(f"NOTE: K must stay small (3-10). M-step fits L-BFGS-B "
          f"per position per component.")
    B, W_shared, av, vv = init_params(Y_tr, t_tr, a.K)

    t_train_end = float(t_sc(tr_m[-1]))   # W drift anchored to last train month
    for it in range(a.em_iter):
        t0 = time.time()
        print(f"\n--- EM iteration {it+1}/{a.em_iter} ---")

        print("  E step ...", end=" ", flush=True)
        z = e_step(B, W_shared, av, vv, Y_tr, t_tr, t_train_end)
        print(f"done  component sizes: {z.sum(0).round(0)}")

        print("  M step weights ...", end=" ", flush=True)
        av, vv = m_step_weights(z, t_tr, a.K)
        print("done")

        print("  M step emissions ...", end=" ", flush=True)
        B, W_shared = m_step_emissions(B, W_shared, z, Y_tr, t_tr)
        print(f"done  {time.time()-t0:.1f}s")

    # script 92v3 numbers for comparison
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

    print(f"\n=== TIME-VARYING LINEAGE MIXTURE K={a.K}  "
          f"train<={a.train_end} ===")
    print(f"\n{'month':<10}{'ppl_mix':>9}{'ppl_92':>9}{'ppl_pers':>10}"
          f"{'delta_92':>10}{'mix>p':>7}{'92>p':>6}")
    print("-" * 62)

    for m in te_m:
        Ym  = Y_by_month[m]
        tm  = t_sc(m)
        Pm  = marginal_probs(B, W_shared, av, vv, tm, t_train_end)
        pm  = perplexity(Pm, Ym)
        pp  = perplexity(P_persist, Ym)
        p92 = ppl92.get(m, float("nan"))
        delta = p92 - pm
        print(f"{m:<10}{pm:>9.5f}{p92:>9.5f}{pp:>10.5f}"
              f"{delta:>+10.5f}"
              f"{'✓' if pm<pp else '✗':>7}"
              f"{'✓' if p92<pp else '✗':>6}")

    print(f"""
COLUMNS
  ppl_mix   time-varying lineage mixture K={a.K}
  ppl_92    time-varying logistic (script 92v3, no lineage structure)
  ppl_pers  persistence
  delta_92  ppl_92 - ppl_mix  positive = mixture better than logistic
  mix>p     ✓ if mixture beats persistence on perplexity
  92>p      ✓ if logistic beats persistence

NOTE ON K: keep K small (3-10). Each EM iteration fits
K×L L-BFGS-B optimizations. K=5, L=1273 = 6365 fits per iteration.
At 500 seqs/month: expect ~5-10 min per EM iteration on CPU.
Biologically K=3 (pre-Omicron / Omicron / post-Omicron) is defensible.
""")


if __name__ == "__main__":
    main()
