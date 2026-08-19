"""
92_spike_timemue_v3.py

Time-varying per-position model on FULL reconstructed amino acid sequences.

This is the correct implementation of Weinstein & Marks (ICML 2021) RegressMuE
in the no-indel limit on SARS-CoV-2 spike.

Reconstruction:
    reference (1273 aa) + substitutions from posres vocab → full amino acid string
    This is exact — reference-anchored alignment handles indels upstream.

Model (per position, per residue):
    logit P(residue r at position j | t) = b_{j,r} + W_{j,r} · t
    (multinomial logistic, D=20 amino acids)

Metric:
    per-residue perplexity = exp( -mean log P(correct residue) / L )
    directly comparable to Weinstein & Marks: 1.32 (static) → 1.24 (time-varying)

Same table as scripts 92/93 for MAE comparison, PLUS perplexity.

Usage:
    python scripts/92_spike_timemue_v3.py \
        --data  data/processed/full_data_graphs_posres \
        --vocab data/processed/full_data_graphs_posres/posres_vocab.tsv \
        --ref   data/raw/spike_reference.fasta \
        --train-end 2022-12 --n-per-month 500

    python scripts/92_spike_timemue_v3.py --synthetic
"""
import argparse, glob, os, pickle, sys, time
import numpy as np
from collections import Counter

# ------------------------------------------------------------------ alphabet
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
D = len(AA)   # 20


# ------------------------------------------------------------------ data io
def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip())
    return "".join(seq)


def read_vocab(path):
    """posres_vocab.tsv -> {node_idx: (aa_pos_1based, residue)}"""
    m = {}
    with open(path) as f:
        hdr = {c: i for i, c in enumerate(f.readline().rstrip("\n").split("\t"))}
        for line in f:
            p = line.rstrip("\n").split("\t")
            node = int(p[hdr["node_idx"]])
            pos  = int(p[hdr["aa_pos"]])
            res  = p[hdr["residue"]]
            if res in AA_IDX:               # keep only standard amino acids
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
    """
    frozenset of node_ids -> full amino acid string of length L.
    Copy reference, overwrite substituted positions.
    """
    seq = list(ref)
    for n in s:
        hit = node_map.get(n)
        if hit and 1 <= hit[0] <= len(ref):
            seq[hit[0] - 1] = hit[1]
    return "".join(seq)


def encode(seq):
    """amino acid string -> integer array of length L"""
    return np.array([AA_IDX.get(c, 0) for c in seq], dtype=np.int32)


def load_all(data_dir, ref, node_map, n_per_month, seed=0):
    """Load monthly data, reconstruct full sequences, encode as integers."""
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
        # reconstruct full sequences
        seqs = np.stack([encode(reconstruct(ref, node_map, s)) for s in sets])
        Y_by_month[ym] = seqs          # (N_month, L) int32
    return months, Y_by_month, L


# --------------------------------------------------------------- synthetic
def synthetic(seed=0):
    """Three lineages, wave dynamics, returns integer-encoded sequences."""
    rng = np.random.default_rng(seed)
    L = 100; ref = np.zeros(L, dtype=np.int32)   # all residue 0
    blocks = {
        0: (rng.choice(L, 5, replace=False),  rng.integers(1, D, 5)),
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
            seq = ref.copy()
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


# ------------------------------------------------------------------ model
def fit_static(Y_cat, D):
    """
    Per-position empirical residue frequencies.
    Y_cat: (N, L) int32
    Returns P: (L, D) float64  (rows sum to 1)
    """
    N, L = Y_cat.shape
    P = np.zeros((L, D))
    for d in range(D):
        P[:, d] = (Y_cat == d).mean(0)
    return np.clip(P, 1e-8, 1)


def fit_time_varying(Y_cat, t, D, wd=1e-2):
    """
    Per-position multinomial logistic regression on time.
    Uses scipy L-BFGS-B per position — properly converged.
    Returns B: (L, D), W: (L, D)
    """
    from scipy.optimize import minimize
    N, L = Y_cat.shape
    X = np.stack([np.ones(N), t], axis=1)   # (N, 2)
    B = np.zeros((L, D)); W = np.zeros((L, D))

    def obj(theta_flat, yj):
        theta = theta_flat.reshape(2, D)
        logits = X @ theta                              # (N, D)
        logits -= logits.max(1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(1, keepdims=True)
        # negative log-likelihood + L2
        nll = -np.log(probs[np.arange(N), yj]).mean()
        reg = 0.5 * wd * (theta**2).sum()
        # gradient
        Y_oh = np.zeros((N, D)); Y_oh[np.arange(N), yj] = 1.0
        grad = -(X.T @ (Y_oh - probs)) / N + wd * theta
        return float(nll + reg), grad.ravel()

    for j in range(L):
        yj = Y_cat[:, j]
        counts = np.bincount(yj, minlength=D)
        if counts.max() == N:
            B[j, counts.argmax()] = 10.0
            continue
        res = minimize(obj, np.zeros(2 * D), args=(yj,),
                      jac=True, method="L-BFGS-B",
                      options={"maxiter": 200, "ftol": 1e-10})
        theta = res.x.reshape(2, D)
        B[j] = theta[0]; W[j] = theta[1]
    return B, W


def predict_probs_static(P, j=None):
    """P: (L, D). Returns (L, D) or (D,) for position j."""
    return P if j is None else P[j]


def predict_probs_time(B, W, t_scalar):
    """Returns (L, D) softmax probabilities at time t."""
    logits = B + W * t_scalar
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


# ----------------------------------------------------------------- metrics
def perplexity_from_probs(P_seq, Y_cat):
    """
    P_seq: (N, L, D) predicted probs per sequence
    Y_cat: (N, L) int32 true residues
    Returns scalar perplexity (per residue).
    """
    N, L, _ = P_seq.shape
    lp = 0.0
    for i in range(N):
        for j in range(L):
            lp += np.log(max(P_seq[i, j, Y_cat[i, j]], 1e-12))
    return float(np.exp(-lp / (N * L)))


def perplexity_static(P, Y_cat):
    """P: (L, D) static. Score all sequences."""
    lp = np.log(np.clip(P[np.arange(Y_cat.shape[1])[None, :],
                           Y_cat], 1e-12, 1)).mean()
    return float(np.exp(-lp))


def perplexity_time(B, W, Y_cat, t_scalar):
    """Score all sequences at a single time point."""
    P = predict_probs_time(B, W, t_scalar)          # (L, D)
    return perplexity_static(P, Y_cat)


def mae_freq(P_pred, Y_cat):
    """
    MAE between predicted per-position residue frequencies and actual.
    P_pred: (L, D), Y_cat: (N, L)
    Returns scalar MAE over variable positions.
    """
    act = np.zeros_like(P_pred)
    for d in range(P_pred.shape[1]):
        act[:, d] = (Y_cat == d).mean(0)
    var = act.max(1) < 0.99        # variable positions
    return float(np.abs(P_pred[var] - act[var]).mean())


def mae_pers(Y_persist, Y_test):
    """MAE of persistence predictor."""
    P_pers = np.zeros((Y_persist.shape[1], D))
    for d in range(D):
        P_pers[:, d] = (Y_persist == d).mean(0)
    return mae_freq(P_pers, Y_test)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--vocab"); ap.add_argument("--ref")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--train-end", default="2022-12")
    ap.add_argument("--n-per-month", type=int, default=500)
    
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

    mi = {m: i for i, m in enumerate(months)}
    mu = np.mean([mi[m] for m in tr_m])
    sd = np.std( [mi[m] for m in tr_m])
    t_sc = lambda m: (mi[m] - mu) / sd

    Y_tr = np.concatenate([Y_by_month[m] for m in tr_m])
    t_tr = np.array([t_sc(m) for m in tr_m
                     for _ in range(len(Y_by_month[m]))])

    # fit models
    print("fitting static model ...", end=" ", flush=True)
    t0 = time.time(); P_static = fit_static(Y_tr, D)
    print(f"{time.time()-t0:.1f}s")

    print("fitting time-varying model ...", end=" ", flush=True)
    t0 = time.time(); B, W = fit_time_varying(Y_tr, t_tr, D)
    print(f"{time.time()-t0:.1f}s")

    Y_persist = Y_by_month[tr_m[-1]]
    # persistence emission: empirical residue freq from last train month
    P_persist = fit_static(Y_persist, D)   # (L, D)

    # perplexity on train
    ppl_s_tr = perplexity_static(P_static, Y_tr)
    # time-varying train perplexity: average over training months
    ppl_t_tr = np.mean([perplexity_time(B, W, Y_by_month[m], t_sc(m))
                        for m in tr_m])

    print(f"\n{'model':<26}{'train ppl':>12}  "
          f"(paper H3N2: static=1.32, time=1.24)")
    print("-" * 50)
    print(f"{'static (no time)':<26}{ppl_s_tr:>12.5f}")
    print(f"{'time-varying':<26}{ppl_t_tr:>12.5f}")

    # monthly test evaluation
    print(f"\n{'month':<10}{'ppl_time':>10}{'ppl_static':>11}{'ppl_pers':>10}"
          f"{'MAE_time':>10}{'MAE_static':>11}{'MAE_pers':>10}"
          f"{'ppl>p':>8}{'mae>p':>8}")
    print("-" * 85)

    for m in te_m:
        Ym = Y_by_month[m]
        tm = t_sc(m)
        pt  = perplexity_time(B, W, Ym, tm)
        ps  = perplexity_static(P_static, Ym)
        pp  = perplexity_static(P_persist, Ym)
        mt  = mae_freq(predict_probs_time(B, W, tm), Ym)
        ms  = mae_freq(P_static, Ym)
        mp  = mae_pers(Y_persist, Ym)
        ppl_beats = "✓" if pt < pp else "✗"
        mae_beats = "✓" if mt < mp else "✗"
        print(f"{m:<10}{pt:>10.5f}{ps:>11.5f}{pp:>11.5f}"
              f"{mt:>10.4f}{ms:>11.4f}{mp:>10.4f}"
              f"{ppl_beats:>8}{mae_beats:>8}")

    print("""
LEGEND
  ppl_time    per-residue perplexity, time-varying model (paper H3N2: 1.24)
  ppl_static  per-residue perplexity, static model      (paper H3N2: 1.32)
  ppl_pers    per-residue perplexity, persistence (last train month frequencies)
  MAE_time    mean |predicted - actual| mutation frequency over variable positions
  MAE_static  same, static model
  MAE_pers    same, persistence
  ppl>p       ✓ if ppl_time < ppl_pers  (time-varying is a better sequence model than persistence)
  mae>p       ✓ if MAE_time < MAE_pers
""")


if __name__ == "__main__":
    main()
