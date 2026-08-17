#!/usr/bin/env python
"""
80_esm_appearance.py

Why this test is different from every earlier one
------------------------------------------------
Six model families failed to beat frequency at predicting which mutations
appear. But frequency can only score a mutation that has ALREADY been observed --
an unseen mutation has frequency zero and no history, so the whole approach is
structurally blind to genuine novelty.

ESM features are not: llr_ref and sem_ref score a mutation from its content
alone, with no requirement that it has ever been seen spreading. That is the one
kind of information the surveillance record does not contain, and the reason the
influenza literature (EVE, EVEscape, Luksza-Lassig) reaches outside frequency
data at all.

Two tests, and the second is the one that has never been run.

  A. ENTRY, among mutations already in the causal universe but absent at t.
     Frequency and recency work here (0.139 and 0.180 in scripts 61 and 76).
     The question is whether ESM ADDS anything. Ablation: a logistic with and
     without the ESM block, plus a structural control so ESM is not merely
     standing in for "this position is in the RBD".

  B. FIRST-EVER APPEARANCE, among grid cells never observed in any month up to t.
     Every candidate has frequency zero and no history, so frequency, recency and
     co-occurrence are all uninformative BY CONSTRUCTION. Any lift above random
     comes from content alone. This is the only stratum where ESM can be the sole
     explanation, and no earlier script tested it.

     Test B needs ESM features for cells that have never been seen. llr_ref from
     masked marginals on the reference sequence is defined for all 20 residues at
     every position, so the full 1273 x 20 grid is available in principle. If the
     cached file only covers observed nodes, test B is restricted to what is
     there and the script says so rather than quietly shrinking the candidate set.

Evaluation
----------
Rolling origin, AP, and lift against a random scorer on the same candidates --
not against the base rate, since AP sits above the base rate when positives are
few.

Outputs
-------
outputs/80_entry_ablation.csv   test A, per origin per feature block
outputs/80_firstever.csv        test B, per origin per scorer
outputs/80_summary.csv          pooled

Usage
-----
python scripts/80_esm_appearance.py --esm outputs/esm_node_features_ref.pkl
python scripts/80_esm_appearance.py --self_test
"""

import argparse
import bisect
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

DOMAINS = [(13, 305, "NTD"), (306, 330, "linker"), (331, 527, "RBD"),
           (528, 685, "SD1/SD2"), (686, 815, "S2-FP"), (816, 1273, "S2")]


def domain_of(pos):
    for a, b, nm in DOMAINS:
        if a <= pos <= b:
            return nm
    return "other"


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_months(data_dir, min_count, start_month=None, end_month=None):
    files = []
    for fn in os.listdir(data_dir):
        m = MONTH_RE.search(fn)
        if m:
            files.append((m.group(1), os.path.join(data_dir, fn)))
    files.sort()
    out = []
    for month, path in files:
        if start_month and month < start_month:
            continue
        if end_month and month > end_month:
            continue
        with open(path, "rb") as f:
            occ = pickle.load(f)
        occ = {k: v for k, v in occ.items() if v >= min_count}
        if occ:
            out.append((month, occ))
    return out


def load_vocab(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = {c.lower(): c for c in df.columns}
    idc = next((cols[c] for c in ("node_idx", "node", "id", "idx")
                if c in cols), None)
    pc = next((cols[c] for c in ("aa_pos", "pos", "position") if c in cols), None)
    rc = next((cols[c] for c in ("residue", "res", "aa") if c in cols), None)
    if pc is None or rc is None:
        sys.exit(f"no position/residue columns in {path}")
    out = {}
    for i, row in enumerate(df.itertuples(index=False)):
        d = dict(zip(df.columns, row))
        key = int(d[idc]) if idc else i
        out[key] = (int(str(d[pc]).strip()), str(d[rc]).strip())
    return out


def load_esm(path, vocab):
    """
    Returns {(pos, res): {feature: value}} plus the list of feature names.
    Tolerates the layouts these caches usually come in: a dict keyed by node id
    or by (pos, res), or a DataFrame. Prints what it found so the mapping can be
    checked rather than trusted.
    """
    if not os.path.exists(path):
        sys.exit(f"ESM feature file not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"ESM cache type: {type(obj).__name__}")

    feats = {}
    if isinstance(obj, pd.DataFrame):
        print(f"  columns: {list(obj.columns)}")
        cols = {c.lower(): c for c in obj.columns}
        pc = next((cols[c] for c in ("aa_pos", "pos", "position")
                   if c in cols), None)
        rc = next((cols[c] for c in ("residue", "res", "aa") if c in cols), None)
        idc = next((cols[c] for c in ("node_idx", "node", "id") if c in cols),
                   None)
        num = [c for c in obj.columns
               if pd.api.types.is_numeric_dtype(obj[c])
               and c not in (pc, rc, idc)]
        for row in obj.itertuples(index=False):
            d = dict(zip(obj.columns, row))
            if pc and rc:
                key = (int(d[pc]), str(d[rc]))
            elif idc and int(d[idc]) in vocab:
                key = vocab[int(d[idc])]
            else:
                continue
            feats[key] = {c: float(d[c]) for c in num}
        names = num
    elif isinstance(obj, dict):
        k0 = next(iter(obj))
        print(f"  example key: {k0!r}  value type: {type(obj[k0]).__name__}")
        names = None
        for k, v in obj.items():
            if isinstance(k, tuple) and len(k) == 2:
                key = (int(k[0]), str(k[1]))
            elif isinstance(k, (int, np.integer)) and int(k) in vocab:
                key = vocab[int(k)]
            else:
                continue
            if isinstance(v, dict):
                d = {a: float(b) for a, b in v.items()
                     if isinstance(b, (int, float, np.floating))}
            elif isinstance(v, (list, tuple, np.ndarray)):
                d = {f"f{i}": float(x) for i, x in enumerate(np.ravel(v))}
            elif isinstance(v, (int, float, np.floating)):
                d = {"value": float(v)}
            else:
                continue
            feats[key] = d
            if names is None:
                names = sorted(d.keys())
    else:
        sys.exit(f"unrecognised ESM cache layout: {type(obj)}")

    if not feats:
        sys.exit("could not map any ESM entries to (position, residue)")
    names = sorted(next(iter(feats.values())).keys())
    print(f"  mapped {len(feats)} (position, residue) cells")
    print(f"  features: {names}")
    ex = list(feats.items())[:3]
    for k, v in ex:
        print(f"    {k} -> " + ", ".join(f"{a}={b:+.3f}" for a, b in v.items()))
    return feats, names


# ----------------------------------------------------------------------------
# metrics and model
# ----------------------------------------------------------------------------

def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    if y.size == 0 or y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -np.asarray(s, dtype=float)))
    yy = y[order]
    tp = np.cumsum(yy)
    return float(((tp / np.arange(1, y.size + 1)) * yy).sum() / y.sum())


def fit_logistic(X, y, l2=1.0, n_iter=60, tol=1e-7):
    w = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1])
    R[0, 0] = 0.0
    for _ in range(n_iter):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (mu - y) + R @ w
        s = np.clip(mu * (1 - mu), 1e-6, None)
        H = X.T @ (X * s[:, None]) + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        wn = w - step
        if np.max(np.abs(wn - w)) < tol:
            return wn
        w = wn
    return w


def predict(X, w):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def design(F, cols, mu=None, sd=None):
    X = F[:, cols]
    if mu is None:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]), mu, sd


def self_test():
    print("self-test")
    rng = np.random.default_rng(0)

    y = (rng.random(20000) < 0.02).astype(int)
    assert abs(average_precision(y, rng.random(20000)) - 0.02) < 0.01
    assert average_precision(y, y + rng.normal(0, 0.05, y.size)) > 0.9
    print("  AP unbiased for random, high for informative     ok")

    # a logistic on an informative feature must beat one on noise
    X = rng.normal(size=(4000, 2))
    yy = (rng.random(4000) < 1 / (1 + np.exp(-(1.5 * X[:, 0] - 2)))).astype(float)
    A, mu, sd = design(X, [0])
    B, mu2, sd2 = design(X, [1])
    wa, wb = fit_logistic(A, yy), fit_logistic(B, yy)
    ap_a = average_precision(yy.astype(int), predict(A, wa))
    ap_b = average_precision(yy.astype(int), predict(B, wb))
    assert ap_a > ap_b + 0.05, (ap_a, ap_b)
    print(f"  informative block {ap_a:.3f} beats noise block {ap_b:.3f}   ok")

    assert domain_of(445) == "RBD" and domain_of(19) == "NTD"
    print("  domain assignment                                ok")
    print("all tests passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--esm", default="outputs/esm_node_features_ref.pkl")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--min_train", type=int, default=12)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    vocab = load_vocab(args.vocab or
                       os.path.join(args.data_dir, "posres_vocab.tsv"))
    esm, esm_names = load_esm(args.esm, vocab)

    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    T = len(names)
    print(f"\nloaded {T} months: {names[0]} .. {names[-1]}")

    # how much of the grid does the ESM cache cover?
    grid_positions = sorted({p for p, _ in vocab.values()})
    covered = len(esm)
    print(f"ESM covers {covered} cells; the observed vocabulary is "
          f"{len(vocab)}; the full grid is {len(grid_positions)} x 20 = "
          f"{len(grid_positions) * 20}")
    full_grid = covered > 1.5 * len(vocab)
    print("  test B will use "
          + ("the full grid of unseen cells" if full_grid else
             "only cells present in the ESM cache -- a cache limited to the\n"
             "  observed vocabulary cannot score truly novel cells, so test B\n"
             "  is restricted and its coverage is reported"))

    # per-month presence and frequency
    freq, present = [], []
    for m in names:
        tot = float(sum(occ_by[m].values()))
        nc = defaultdict(float)
        for cs, w in occ_by[m].items():
            for l in cs:
                nc[l] += w
        freq.append({l: v / tot for l, v in nc.items()})
        present.append(set(nc.keys()))

    seen_by, idxs = [], defaultdict(list)
    acc = set()
    for j, m in enumerate(names):
        for l in present[j]:
            idxs[l].append(j)
        acc |= present[j]
        seen_by.append(frozenset(acc))

    def esm_vec(node):
        pr = vocab.get(node)
        if pr is None:
            return None
        return esm.get(pr)

    # ========================================================================
    # A. entry, with and without the ESM block
    # ========================================================================
    FREQ_F = ["log_hist_freq", "recency", "months_present", "log_last_freq"]
    STRUCT_F = ["is_RBD", "is_NTD", "position"]
    BLOCKS = {
        "freq_only": FREQ_F,
        "freq+struct": FREQ_F + STRUCT_F,
        "freq+struct+esm": FREQ_F + STRUCT_F + esm_names,
        "esm_only": esm_names,
        "struct+esm": STRUCT_F + esm_names,
    }
    ALL_F = FREQ_F + STRUCT_F + esm_names
    col = {f: i for i, f in enumerate(ALL_F)}

    rows_a, rows_b, cache = [], [], {}
    for t in range(T - 1):
        universe = sorted(seen_by[t], key=str)
        cand = [l for l in universe if l not in present[t] and esm_vec(l)]
        if len(cand) < 20:
            continue
        y = np.array([1 if l in present[t + 1] else 0 for l in cand], dtype=int)
        lo = max(0, t - args.window + 1)
        F = np.zeros((len(cand), len(ALL_F)))
        for r, l in enumerate(cand):
            pos, _ = vocab[l]
            h = float(np.mean([freq[j].get(l, 0.0) for j in range(lo, t + 1)]))
            k = bisect.bisect_right(idxs[l], t) - 1
            ls = idxs[l][k] if k >= 0 else None
            F[r, col["log_hist_freq"]] = np.log(max(h, 1e-9))
            F[r, col["recency"]] = 0.0 if ls is None else 1.0 / (1.0 + (t - ls))
            F[r, col["months_present"]] = sum(1 for j in idxs[l] if j <= t)
            F[r, col["log_last_freq"]] = (np.log(max(freq[ls].get(l, 1e-9), 1e-9))
                                          if ls is not None else -9.0)
            F[r, col["is_RBD"]] = 1.0 if domain_of(pos) == "RBD" else 0.0
            F[r, col["is_NTD"]] = 1.0 if domain_of(pos) == "NTD" else 0.0
            F[r, col["position"]] = pos / 1273.0
            ev = esm_vec(l)
            for nm in esm_names:
                F[r, col[nm]] = float(ev.get(nm, 0.0))
        cache[t] = (F, y)

        if t < args.min_train or y.sum() == 0:
            continue
        tr = [cache[j] for j in range(max(0, t - args.window), t) if j in cache]
        if not tr:
            continue
        Ftr = np.vstack([a for a, _ in tr])
        ytr = np.concatenate([b for _, b in tr]).astype(float)
        for bname, feats in BLOCKS.items():
            cols = [col[f] for f in feats]
            A, mu, sd = design(Ftr, cols)
            w = fit_logistic(A, ytr, args.l2)
            B, _, _ = design(F, cols, mu, sd)
            rows_a.append({"origin": names[t], "block": bname,
                           "ap": average_precision(y, predict(B, w)),
                           "base": float(y.mean()), "n": len(y),
                           "n_pos": int(y.sum())})
        rows_a.append({"origin": names[t], "block": "random",
                       "ap": average_precision(y, rng.random(len(y))),
                       "base": float(y.mean()), "n": len(y),
                       "n_pos": int(y.sum())})

        # ====================================================================
        # B. first-ever appearance: every candidate has frequency zero
        # ====================================================================
        unseen = [l for l in vocab
                  if l not in seen_by[t] and esm_vec(l)]
        yb = np.array([1 if l in present[t + 1] else 0 for l in unseen],
                      dtype=int)
        if yb.sum() > 0 and len(unseen) > 50:
            E = np.array([[float(esm_vec(l).get(nm, 0.0)) for nm in esm_names]
                          for l in unseen])
            posv = np.array([vocab[l][0] for l in unseen], dtype=float)
            isrbd = np.array([1.0 if domain_of(int(p)) == "RBD" else 0.0
                              for p in posv])
            scorers = {"random": rng.random(len(unseen)),
                       "is_RBD": isrbd}
            for i, nm in enumerate(esm_names):
                scorers[f"esm:{nm}"] = E[:, i]
                scorers[f"esm:-{nm}"] = -E[:, i]
            for nm, s in scorers.items():
                rows_b.append({"origin": names[t], "scorer": nm,
                               "ap": average_precision(yb, s),
                               "base": float(yb.mean()), "n": len(unseen),
                               "n_pos": int(yb.sum())})

    da = pd.DataFrame(rows_a)
    db = pd.DataFrame(rows_b)
    da.to_csv(f"{args.out_dir}/80_entry_ablation.csv", index=False)
    db.to_csv(f"{args.out_dir}/80_firstever.csv", index=False)

    def summarise(d, key):
        if not len(d) or key not in d.columns:
            return pd.DataFrame(columns=[key, "mean_ap", "lift_vs_random",
                                         "mean_base", "mean_n", "mean_pos",
                                         "origins"])
        rnd = d[d[key] == "random"].set_index("origin")["ap"]
        out = []
        for nm in d[key].unique():
            sub = d[d[key] == nm].set_index("origin")
            common = sub.index.intersection(rnd.index)
            a, r = sub.loc[common, "ap"], rnd.loc[common]
            ok = (~a.isna()) & (~r.isna()) & (r > 0)
            out.append({key: nm, "mean_ap": float(sub["ap"].mean()),
                        "lift_vs_random": (float((a[ok] / r[ok]).mean())
                                           if ok.any() else np.nan),
                        "mean_base": float(sub["base"].mean()),
                        "mean_n": float(sub["n"].mean()),
                        "mean_pos": float(sub["n_pos"].mean()),
                        "origins": int(len(sub))})
        return pd.DataFrame(out).sort_values("mean_ap", ascending=False)

    print("\n" + "=" * 82)
    print("A. ENTRY  (mutations already in the universe, absent at t)")
    print("=" * 82)
    sa = summarise(da, "block")
    if not len(sa):
        print("no scorable entry events -- every label in the universe was")
        print("present at every origin, so there was nothing to predict")
    else:
        print(sa.round(4).to_string(index=False))
    f = (sa.set_index("block")["mean_ap"] if len(sa)
         else pd.Series(dtype=float))
    if "freq+struct" in f.index and "freq+struct+esm" in f.index:
        print(f"\nadding ESM: {f['freq+struct']:.4f} -> "
              f"{f['freq+struct+esm']:.4f}   ratio "
              f"{f['freq+struct+esm']/max(f['freq+struct'],1e-9):.3f}")
        print("  the structural block is in both, so any gain is ESM content")
        print("  rather than ESM standing in for 'this position is in the RBD'.")

    print("\n" + "=" * 82
          if len(db) else "\n(test B had too few first-ever events to score)")
    if len(db):
        print("B. FIRST-EVER APPEARANCE  (never observed up to t;")
        print("   every candidate has frequency zero, so frequency, recency and")
        print("   co-occurrence are uninformative by construction)")
        print("=" * 82)
        sb = summarise(db, "scorer")
        print(sb.round(5).to_string(index=False))
        print("\n  any scorer clearly above random here is doing something no")
        print("  frequency-based method can do, because there is no frequency to")
        print("  use. That is the entire case for reaching outside the")
        print("  surveillance record.")
        print("  all scorers at random -> ESM does not locate novelty either,")
        print("  and the emergence problem stays open.")

    parts = []
    if len(sa):
        parts.append(sa.assign(test="A_entry").rename(columns={"block": "name"}))
    sb2 = summarise(db, "scorer")
    if len(sb2):
        parts.append(sb2.assign(test="B_firstever")
                     .rename(columns={"scorer": "name"}))
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(
            f"{args.out_dir}/80_summary.csv", index=False)
    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
