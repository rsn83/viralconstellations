#!/usr/bin/env python
"""
72_copyforward_plus.py

The idea
--------
Copy-forward is the model, not the baseline. Every architecture tried in scripts
63-66 either competed with it from zero or, in the case of 64, started at it and
then overfitted the correction: training loss fell 12x while every held-out
metric got worse than initialisation. Roughly 30 months of real signal cannot
support a network in the residual.

So the correction here has THREE parameters. It is not a network, it cannot
overfit, and it is the smallest thing that could beat copying.

The model
---------
Copy-forward says: a label stays if it is there, stays out if it is not. The
correction asks, separately for each direction, whether population frequency
moves it. TWO parameters per direction:

    ADD   logit P(label l joins c)   = b_add  * log rho_t(l) + d_add
    DROP  logit P(label l leaves c)  = b_drop * log rho_t(l) + d_drop

Fitted by maximum likelihood on months <= t, evaluated on t+1. The predicted set
is then the surviving members plus the added ones.

An earlier version fitted ONE logistic over both directions with an extra
"is it already in c" feature. That does not work: the copy coefficient fits to
6-17, the sigmoid saturates, thresholding returns exactly the current set, and
the shared likelihood is dominated by the in-set rows -- on synthetic data where
additions were frequency-driven by construction, the frequency weight came out
at 0.03. Splitting the two directions fixes both problems.

  b  how much population frequency pulls a label in (or pushes it out). This is
     the only information copy-forward structurally cannot see, and script 67
     found population frequency is what governs which labels move.
  d  an intercept setting the base rate of change in that direction.

Why frequency and nothing else
------------------------------
Script 67's ablation: co-occurrence features added nothing over frequency
(full/no_pmi 1.185 against a measured null of 1.32). Script 66's learned
transport put +3.0 on "do not add" and 0.1 on the co-occurrence terms. Script 65
lost to the marginal. Three independent results say the movable part is governed
by population frequency, so that is the only term the correction gets.

Evaluation
----------
Rolling origin. For every constellation circulating at t:
  ADD    among labels absent from c, which are present at t+1?
  DROP   among labels in c, which are absent at t+1?
  SET    Jaccard of the predicted set against the observed one
Copy-forward scores at the base rate on ADD and DROP by construction -- it
predicts no change -- so those two are where a gain would have to appear.

Membership at t+1 is defined without any coupling: a label counts as present if
the set c u {l} or c itself is observed at t+1. Concretely, ADD asks whether
c u {l} appears; DROP asks whether c disappears while c \\ {l} is present. No
sequence-level pairing is imputed anywhere.

Outputs
-------
outputs/72_by_month.csv
outputs/72_summary.csv
outputs/72_params.csv

Usage
-----
python scripts/72_copyforward_plus.py --min_count 3 --end_month 2024-12
python scripts/72_copyforward_plus.py --self_test
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")


# ----------------------------------------------------------------------------
# data
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


def rho_of(occ):
    """Population frequency of each label: share of sequences carrying it."""
    tot = float(sum(occ.values()))
    nc = defaultdict(float)
    for cs, w in occ.items():
        for l in cs:
            nc[l] += w
    return {l: v / tot for l, v in nc.items()}


def build_examples(occ_t, occ_n, rho_t, labels, max_sets):
    """
    One row per (constellation, candidate label). No coupling is imputed.

    in_c   1 if the label is already in c
    rho    population frequency of the label at t
    y      1 if the label is in c's successor at t+1, where
             for a label in c      -> c itself still observed at t+1
             for a label not in c  -> c u {label} observed at t+1
    """
    sets = sorted(occ_t.items(), key=lambda kv: -kv[1])[:max_sets]
    H_n = set(occ_n.keys())
    in_c, rho, y, cid = [], [], [], []
    for k, (c, _) in enumerate(sets):
        stays = c in H_n
        for l in labels:
            present = l in c
            in_c.append(1.0 if present else 0.0)
            rho.append(rho_t.get(l, 0.0))
            if present:
                y.append(1 if stays else 0)
            else:
                y.append(1 if frozenset(c | {l}) in H_n else 0)
            cid.append(k)
    return (np.array(in_c), np.array(rho, dtype=float),
            np.array(y, dtype=int), np.array(cid), [c for c, _ in sets])


# ----------------------------------------------------------------------------
# the three-parameter model
# ----------------------------------------------------------------------------

def design(rho):
    """[log rho, 1] -- log frequency because rho spans several decades."""
    rho = np.asarray(rho, dtype=float)
    return np.column_stack([np.log(np.clip(rho, 1e-6, None)),
                            np.ones_like(rho)])


def fit(X, y, l2=1e-3, n_iter=100, tol=1e-9):
    w = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1])
    for _ in range(n_iter):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (mu - y) + R @ w
        s = np.clip(mu * (1 - mu), 1e-9, None)
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


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def average_precision(y, s):
    y = np.asarray(y, dtype=int)
    if y.size == 0 or y.sum() == 0:
        return np.nan
    order = np.lexsort((np.arange(y.size), -np.asarray(s, dtype=float)))
    yy = y[order]
    tp = np.cumsum(yy)
    return float(((tp / np.arange(1, y.size + 1)) * yy).sum() / y.sum())


def self_test():
    print("self-test")

    rng = np.random.default_rng(0)
    rho = rng.random(8000) * 0.4 + 1e-4
    X = design(rho)

    # copy-forward means never add: representable by a very low intercept
    p = predict(X, np.array([0.0, -20.0]))
    assert p.max() < 1e-6
    print("  copy-forward (never change) representable        ok")

    # frequency-driven truth -> clearly positive weight, even at a low base rate
    y = (rng.random(8000) < np.clip(0.02 * rho / rho.mean(), 0, 1)).astype(float)
    w = fit(X, y)
    assert w[0] > 0.5, w
    print(f"  frequency-driven truth, base {y.mean():.3f} -> b = "
          f"{w[0]:+.2f}   ok")

    # frequency-free truth at the same base rate -> weight near zero
    y2 = (rng.random(8000) < 0.02).astype(float)
    w2 = fit(X, y2)
    assert abs(w2[0]) < 0.3, w2
    print(f"  frequency-free truth,   base {y2.mean():.3f} -> b = "
          f"{w2[0]:+.2f}   ok")

    # AP unbiased
    yy = (rng.random(20000) < 0.05).astype(int)
    assert abs(average_precision(yy, rng.random(20000)) - 0.05) < 0.02
    print("  AP unbiased for a random scorer                  ok")

    print("all tests passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed/full_data_graphs_posres")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", default=None)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--max_sets", type=int, default=400)
    ap.add_argument("--pool", type=int, default=150,
                    help="candidate labels: the most frequent at month t")
    ap.add_argument("--min_train", type=int, default=12)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    months = load_months(args.data_dir, args.min_count,
                         args.start_month, args.end_month)
    names = [m for m, _ in months]
    occ = {m: o for m, o in months}
    T = len(names)
    print(f"loaded {T} months: {names[0]} .. {names[-1]}")

    cache, rows, prows = {}, [], []
    for t in range(T - 1):
        rt = rho_of(occ[names[t]])
        labels = [l for l, _ in sorted(rt.items(), key=lambda kv: -kv[1])
                  ][:args.pool]
        cache[t] = build_examples(occ[names[t]], occ[names[t + 1]], rt,
                                  labels, args.max_sets)

        if t < args.min_train:
            continue
        tr = [cache[j] for j in range(max(0, t - args.window), t) if j in cache]
        if not tr:
            continue

        # fit the two directions SEPARATELY -- a shared fit is dominated by the
        # in-set rows and drowns the addition signal
        w_dir, p_dir = {}, {}
        in_c, rho, y, cid, sets = cache[t]
        for nm, want_in in (("add", False), ("drop", True)):
            Xs, ys = [], []
            for a_, b_, c_, _, _ in tr:
                m = (a_ > 0.5) if want_in else (a_ < 0.5)
                if not m.any():
                    continue
                Xs.append(design(b_[m]))
                # the event is joining (add) or leaving (drop)
                ys.append(c_[m].astype(float) if nm == "add"
                          else 1.0 - c_[m].astype(float))
            if not Xs:
                continue
            w_dir[nm] = fit(np.vstack(Xs), np.concatenate(ys))
            m_te = (in_c > 0.5) if want_in else (in_c < 0.5)
            p_dir[nm] = predict(design(rho[m_te]), w_dir[nm])

        if "add" not in w_dir or "drop" not in w_dir:
            continue
        prows.append({"origin": names[t],
                      "b_add": w_dir["add"][0], "d_add": w_dir["add"][1],
                      "b_drop": w_dir["drop"][0], "d_drop": w_dir["drop"][1]})

        add = in_c < 0.5
        drp = in_c > 0.5
        for nm, mask in (("add", add), ("drop", drp)):
            yy = y[mask]
            if yy.size == 0 or yy.sum() == 0 or yy.sum() == yy.size:
                continue
            target = yy if nm == "add" else 1 - yy
            score = p_dir[nm]
            rows.append({
                "origin": names[t], "target": names[t + 1], "stratum": nm,
                "model": "copyforward_plus",
                "ap": average_precision(target, score),
                "base": float(target.mean()), "n": int(target.size),
                "n_pos": int(target.sum()),
            })
            rows.append({
                "origin": names[t], "target": names[t + 1], "stratum": nm,
                "model": "marginal",
                "ap": average_precision(target,
                                        rho[mask] if nm == "add" else -rho[mask]),
                "base": float(target.mean()), "n": int(target.size),
                "n_pos": int(target.sum()),
            })
            rows.append({
                "origin": names[t], "target": names[t + 1], "stratum": nm,
                "model": "copyforward",
                "ap": float(target.mean()),      # predicts no change: base rate
                "base": float(target.mean()), "n": int(target.size),
                "n_pos": int(target.sum()),
            })

        # SET stratum: Jaccard of the predicted membership against observed
        js_model, js_copy = [], []
        lab_arr = np.array(labels, dtype=object)
        p_full = np.zeros(len(in_c))
        p_full[in_c < 0.5] = p_dir["add"]
        p_full[in_c > 0.5] = 1.0 - p_dir["drop"]     # survival = 1 - leaving
        for k, c in enumerate(sets):
            sel = cid == k
            # rows for set k are one per label, in the order of `labels`,
            # so the mask indexes lab_arr directly rather than the full row set
            assert sel.sum() == len(lab_arr)
            pred = set(lab_arr[p_full[sel] > 0.5])
            true = set(lab_arr[y[sel] > 0.5])
            cur = set(l for l in labels if l in c)
            for store, s in ((js_model, pred), (js_copy, cur)):
                u = len(s | true)
                store.append(len(s & true) / u if u else 1.0)
        rows.append({"origin": names[t], "target": names[t + 1],
                     "stratum": "set_jaccard", "model": "copyforward_plus",
                     "ap": float(np.mean(js_model)), "base": np.nan,
                     "n": len(sets), "n_pos": np.nan})
        rows.append({"origin": names[t], "target": names[t + 1],
                     "stratum": "set_jaccard", "model": "copyforward",
                     "ap": float(np.mean(js_copy)), "base": np.nan,
                     "n": len(sets), "n_pos": np.nan})

        print(f"  {names[t]}: add b={w_dir['add'][0]:+.3f} "
              f"d={w_dir['add'][1]:+.2f} | drop b={w_dir['drop'][0]:+.3f} "
              f"d={w_dir['drop'][1]:+.2f}")
        for j in list(cache):
            if j < t - args.window:
                del cache[j]

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/72_by_month.csv", index=False)
    pdf = pd.DataFrame(prows)
    pdf.to_csv(f"{args.out_dir}/72_params.csv", index=False)

    print("\n" + "=" * 74)
    print("FITTED PARAMETERS")
    print("=" * 74)
    print(pdf[["b_add", "d_add", "b_drop", "d_drop"]].describe()
          .loc[["mean", "std", "min", "max"]].round(4).to_string())
    print("\n  b_add  > 0 : common labels are more likely to join a set")
    print("  b_drop < 0 : common labels are less likely to leave")
    print("  b near zero in either direction means population frequency adds")
    print("  nothing there and the model has collapsed to copy-forward.")
    print("  CALIBRATION: on synthetic data with a 2% base rate, a")
    print("  frequency-driven truth gave b = +0.94 and a frequency-free truth")
    print("  gave b = +0.25. So the separation is real but not clean: treat")
    print("  b below about +0.4 as no frequency effect.")

    print("\n" + "=" * 74)
    print("HELD-OUT PERFORMANCE")
    print("=" * 74)
    summ = df.groupby(["stratum", "model"]).agg(
        mean_ap=("ap", "mean"), mean_base=("base", "mean"),
        origins=("ap", "count")).reset_index()
    summ["lift_over_base"] = summ["mean_ap"] / summ["mean_base"]
    summ.to_csv(f"{args.out_dir}/72_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))

    for st in ("add", "drop"):
        s = df[df["stratum"] == st]
        if not len(s):
            continue
        a = s[s["model"] == "copyforward_plus"].set_index("origin")["ap"]
        b = s[s["model"] == "copyforward"].set_index("origin")["ap"]
        c = s[s["model"] == "marginal"].set_index("origin")["ap"]
        i1 = a.index.intersection(b.index)
        i2 = a.index.intersection(c.index)
        print(f"\n{st}: beats copyforward on {(a[i1] > b[i1]).sum()}/{len(i1)}"
              f" origins, beats marginal on {(a[i2] > c[i2]).sum()}/{len(i2)}")

    sj = df[df["stratum"] == "set_jaccard"]
    if len(sj):
        a = sj[sj["model"] == "copyforward_plus"].set_index("origin")["ap"]
        b = sj[sj["model"] == "copyforward"].set_index("origin")["ap"]
        i = a.index.intersection(b.index)
        print(f"\nset jaccard: model {a.mean():.4f} vs copyforward {b.mean():.4f}"
              f"   model wins {(a[i] > b[i]).sum()}/{len(i)} origins")
        print("  EXPECT THESE TO BE IDENTICAL, and it is not a bug. Change")
        print("  events have a base rate of 1-2%, so a calibrated probability")
        print("  never crosses 0.5, so the predicted set always equals the")
        print("  current one. ANY set-level metric with a 0.5 threshold is")
        print("  forced to equal copy-forward exactly. That is arithmetic, and")
        print("  it is why the set-level comparisons in scripts 63-66 could")
        print("  not be won. Read add and drop instead; those are the only")
        print("  strata where a difference can appear at all.")

    print(f"\nwrote 3 files to {args.out_dir}/")


if __name__ == "__main__":
    main()
