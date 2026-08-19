"""
92_spike_timemue.py

Time-varying per-position model on full-length spike (1273 aa).
Equivalent to Weinstein & Marks RegressMuE in the no-indel limit.

    logit P(mutation at position j | t) = b_j + W_j * t

TWO MODES:

  --single   train on months <= --train-end, predict all months after
  --rolling  train on increasing windows, predict next K months each time
             this shows whether more history helps or hurts

Usage:
  python scripts/92_spike_timemue.py \
      --data  data/processed/full_data_graphs_posres \
      --vocab data/processed/posres_vocab.tsv \
      --ref   data/raw/spike_reference.fasta \
      --single --train-end 2021-11

  python scripts/92_spike_timemue.py \
      --data  data/processed/full_data_graphs_posres \
      --vocab data/processed/posres_vocab.tsv \
      --ref   data/raw/spike_reference.fasta \
      --rolling --horizon 3

  python scripts/92_spike_timemue.py --synthetic --rolling --horizon 3
"""
import argparse, glob, os, pickle, sys, time
import numpy as np
from collections import Counter

# ----------------------------------------------------------------- data
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
    """
    Three lineages with block structure, rising and falling like real variants.
    Lineage A = Alpha-like (rises, then replaced)
    Lineage B = Delta-like (dominates mid-period)
    Lineage C = Omicron-like (emerges late, takes over)
    """
    rng = np.random.default_rng(seed)
    L = 1273
    blocks = {
        "A": rng.choice(L, 8,  replace=False),
        "B": rng.choice(L, 15, replace=False),
        "C": rng.choice(L, 30, replace=False),
    }
    months = [f"{y}-{m:02d}" for y in range(2020, 2025) for m in range(1, 13)]
    Y_by_month = {}
    for i, ym in enumerate(months):
        t = i / len(months)
        wA = max(0.0, np.exp(-((t - 0.25)**2) / 0.01))   # peaks ~2021-04
        wB = max(0.0, np.exp(-((t - 0.50)**2) / 0.01))   # peaks ~2022-07
        wC = max(0.0, 2.5 * (t - 0.65)) if t > 0.65 else 0.0
        tot = wA + wB + wC + 0.1
        rows = []
        for _ in range(200):
            v = np.zeros(L, np.float32)
            u, acc = rng.random() * tot, 0.0
            for k, w in (("A", wA), ("B", wB), ("C", wC)):
                acc += w
                if u < acc:
                    keep = rng.random(len(blocks[k])) < 0.9
                    v[blocks[k][keep]] = 1.0
                    break
            rows.append(v)
        Y_by_month[ym] = np.stack(rows)
    return months, Y_by_month


# ---------------------------------------------------------------- model
def fit(Y, t, steps=80, wd=1e-3):
    """
    Per-position logistic regression of mutation presence on time.
    Y : (N, L)   binary float32
    t : (N,)     scaled time  (mean 0, std 1 over TRAINING data)
    Returns b (L,), W (L,).
    """
    L = Y.shape[1]
    X = np.stack([np.ones_like(t), t], axis=1)
    b = np.zeros(L, np.float64)
    W = np.zeros(L, np.float64)
    Yf = Y.astype(np.float64)
    for j in range(L):
        yj = Yf[:, j]
        if yj.sum() < 2:
            b[j] = -10.0   # never seen -> p~0
            continue
        w = np.zeros(2)
        for _ in range(steps):
            p = 1.0 / (1.0 + np.exp(-X @ w))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            g = X.T @ (yj - p) - wd * w
            H = -(X.T * (p * (1 - p))) @ X - wd * np.eye(2)
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        b[j], W[j] = w[0], w[1]
    return b, W


def predict_p(b, W, t_scalar):
    return 1.0 / (1.0 + np.exp(-(b + W * t_scalar)))


def perplexity(b, W, Y, t_scaled, L):
    lp = 0.0
    for i, ti in enumerate(t_scaled):
        p = np.clip(predict_p(b, W, ti), 1e-7, 1 - 1e-7)
        lp += (Y[i] * np.log(p) + (1 - Y[i]) * np.log(1 - p)).sum()
    return float(np.exp(-lp / (len(Y) * L)))


# -------------------------------------------------------------- metrics
def eval_month(b_time, b_static, W, t_m, Y_m, Y_persist, L):
    """
    Returns dict of metrics for one test month.
    Compared against: static model (no W), persistence (last train month).
    """
    p_time   = predict_p(b_time,   W,            t_m)
    p_static = predict_p(b_static, np.zeros(L),  0.0)
    p_pers   = Y_persist.mean(0)
    p_act    = Y_m.mean(0)

    var = (p_act > 0.01) | (p_time > 0.01)

    def mae(p): return float(np.abs(p[var] - p_act[var]).mean())
    def top_overlap(p, k):
        return len(set(np.argsort(-p)[:k]) &
                   set(np.argsort(-p_act)[:k])) / k

    return dict(
        mae_time   = mae(p_time),
        mae_static = mae(p_static),
        mae_pers   = mae(p_pers),
        top20      = top_overlap(p_time, 20),
        top20_pers = top_overlap(p_pers, 20),
        pred_setsz = float(p_time.sum()),
        act_setsz  = float(p_act.sum()),
        var_pos    = int(var.sum()),
        n_seqs     = len(Y_m),
    )


# ----------------------------------------------------------------- runs
def run_single(months, Y_by_month, train_end, L, verbose=True):
    tr_m = [m for m in months if m <= train_end]
    te_m = [m for m in months if m >  train_end]
    if not te_m:
        sys.exit("no test months after --train-end")

    mi = {m: i for i, m in enumerate(months)}
    tr_idx = np.array([mi[m] for m in tr_m], dtype=float)
    mu, sd = tr_idx.mean(), tr_idx.std()

    Y_tr = np.concatenate([Y_by_month[m] for m in tr_m])
    t_tr = np.concatenate([np.full(len(Y_by_month[m]), mi[m]) for m in tr_m])
    t_tr_s = (t_tr - mu) / sd

    t0 = time.time()
    b_s, _ = fit(Y_tr, t_tr_s * 0)        # static: W zeroed
    b_t, W  = fit(Y_tr, t_tr_s)           # time-varying
    elapsed = time.time() - t0

    if verbose:
        print(f"  fit time {elapsed:.1f}s  "
              f"train {len(tr_m)} mo ({len(Y_tr)} seqs)  "
              f"test {len(te_m)} mo")

    Y_persist = Y_by_month[tr_m[-1]]
    rows = []
    for m in te_m:
        t_m = (mi[m] - mu) / sd
        r = eval_month(b_t, b_s, W, t_m, Y_by_month[m], Y_persist, L)
        r["month"] = m
        r["n_train_months"] = len(tr_m)
        rows.append(r)
    return rows


def run_rolling(months, Y_by_month, horizon, min_train, L):
    """
    For each possible train-end (starting from min_train months in),
    predict the next `horizon` months.
    This shows whether MORE training data = better predictions.
    """
    all_rows = []
    cutoffs = months[min_train - 1: -horizon]
    print(f"rolling: {len(cutoffs)} windows, horizon={horizon} months")
    for cut in cutoffs:
        rows = run_single(months, Y_by_month, cut, L, verbose=False)
        rows = rows[:horizon]   # only next `horizon` months
        all_rows.extend(rows)
    return all_rows


# -------------------------------------------------------------- printing
def print_table(rows, mode="single"):
    if mode == "single":
        hdr = (f"{'month':<10}{'MAE_time':>10}{'MAE_static':>11}"
               f"{'MAE_pers':>10}{'top20':>7}{'top20_p':>8}"
               f"{'pred_sz':>9}{'act_sz':>8}")
        print(hdr); print("-" * len(hdr))
        for r in rows:
            beat = "✓" if r["mae_time"] < r["mae_pers"] else "✗"
            print(f"{r['month']:<10}{r['mae_time']:>10.4f}"
                  f"{r['mae_static']:>11.4f}{r['mae_pers']:>10.4f}"
                  f"{r['top20']:>7.2f}{r['top20_pers']:>8.2f}"
                  f"{r['pred_setsz']:>9.1f}{r['act_setsz']:>8.1f}  {beat}")
    else:
        # rolling: summarise mae_time vs mae_pers by n_train_months
        from collections import defaultdict
        by_n = defaultdict(list)
        for r in rows:
            by_n[r["n_train_months"]].append(
                (r["mae_time"], r["mae_pers"]))
        print(f"\n{'n_train_months':>16}{'MAE_time':>12}{'MAE_pers':>12}"
              f"{'beats_pers':>12}")
        print("-" * 55)
        for n in sorted(by_n):
            mt = np.mean([x[0] for x in by_n[n]])
            mp = np.mean([x[1] for x in by_n[n]])
            print(f"{n:>16}{mt:>12.4f}{mp:>12.4f}"
                  f"{'YES' if mt < mp else 'NO':>12}")

    print("""
LEGEND
  MAE_time   |predicted - actual| per variable position, time-varying model
  MAE_static same, static model (no time covariate)
  MAE_pers   same, persistence (predict next month = last train month)
  top20      fraction of top-20 mutated positions the model gets right
  top20_p    same for persistence
  pred_sz    predicted mean mutations/sequence
  act_sz     actual mean mutations/sequence (your data rises ~1->55)
  ✓          time-varying model beats persistence  ✗ it doesn't
""")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--vocab")
    ap.add_argument("--ref", default=None, help="optional spike reference fasta")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--single",    action="store_true")
    ap.add_argument("--rolling",   action="store_true")
    ap.add_argument("--train-end", default="2021-11")
    ap.add_argument("--horizon",   type=int, default=3,
                    help="months to predict ahead in rolling mode")
    ap.add_argument("--min-train", type=int, default=12,
                    help="minimum training months before first prediction")
    ap.add_argument("--n-per-month", type=int, default=500)
    a = ap.parse_args()

    if not (a.single or a.rolling):
        a.single = True     # default to single

    if a.synthetic:
        months, Y_by_month = synthetic()
        L = 1273
    else:
        if not all([a.data, a.vocab]):
            sys.exit("need --data and --vocab  (or --synthetic)")
        node_map = read_vocab(a.vocab)
        if a.ref:
            L = len(read_fasta(a.ref))
        else:
            L = max(pos for pos, res in node_map.values())
            print(f"no --ref given, inferring spike length from vocab: L={L}")
        print(f"spike length {L}, vocab {len(node_map)} nodes")
        months, Y_by_month = load_all(a.data, node_map, L, a.n_per_month)

    print(f"loaded {len(months)} months: {months[0]} .. {months[-1]}")

    if a.single:
        print(f"\n=== SINGLE SPLIT  train <= {a.train_end} ===")
        rows = run_single(months, Y_by_month, a.train_end, L)
        print_table(rows, "single")

    if a.rolling:
        print(f"\n=== ROLLING EVAL  horizon={a.horizon} months ===")
        rows = run_rolling(months, Y_by_month, a.horizon, a.min_train, L)
        print_table(rows, "rolling")

    # save checkpoint for downstream use
    if a.single:
        tr_m = [m for m in months if m <= a.train_end]
        mi   = {m: i for i, m in enumerate(months)}
        tnum = np.array([mi[m] for m in tr_m], dtype=float)
        mu, sd = tnum.mean(), tnum.std()
        Y_tr = np.concatenate([Y_by_month[m] for m in tr_m])
        t_tr_s = (tnum.repeat([len(Y_by_month[m]) for m in tr_m]) - mu) / sd
        b, W = fit(Y_tr, t_tr_s)
        with open("regressmue_ckpt.pkl", "wb") as f:
            pickle.dump(dict(b=b, W=W, months=months,
                             mu=mu, sd=sd, L=L,
                             train_end=a.train_end), f)
        print("wrote regressmue_ckpt.pkl")


if __name__ == "__main__":
    main()
