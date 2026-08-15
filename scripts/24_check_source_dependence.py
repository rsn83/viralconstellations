#!/usr/bin/env python
"""
24_check_source_dependence.py

The gate before building an edit model. No model, no GPU, counts only.

QUESTION
--------
A new constellation appears at t+1 as s + {m}, where s is a set in H_t (a
SOURCE -- a geometric relation at edit distance 1, NOT an ancestry claim).
Does the identity of s change WHICH m gets added?

WHY IT DECIDES EVERYTHING
-------------------------
An edit model learns a rule f(s, m) -> does s+{m} appear. For that rule to
beat the frequency baseline it must depend on the PAIR. If the same mutations
are added to every source regardless of composition, f(s, m) collapses to
f(m) -- "m is hot, add it everywhere" -- which IS the independence-assumed
baseline, rebuilt with more machinery.

HOW IT IS MEASURED
------------------
Two ranking predictors over all valid (source, mutation) pairs:

  MARGINAL       score = rate(m), the overall fraction of sources that added m.
                 Ignores s completely. This is the independence baseline.

  MARGINAL+SRC   score = rate(m) * exp(z(affinity(s, m))), where affinity is
                 the cosine between s's composition and the centroid of the
                 OTHER sources that added m (leave-one-out).

AUC_gain = AUC(MARGINAL+SRC) - AUC(MARGINAL).

The gain is the number that matters, not either AUC alone. A marginal-only
AUC is high whenever add-rates vary across mutations, which says nothing
about source dependence -- an earlier version of this script used it and a
unit test showed it scored 0.82 both when the source mattered and when it did
not. The gain isolates the pair-level contribution.

Validated on synthetic data with known structure:
  purely marginal generative process -> gain -0.010
  source-dependent process           -> gain +0.039

Usage
-----
  python scripts/24_check_source_dependence.py
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def log(m):
    print(m, flush=True)


def constellations_of(occ):
    out = {}
    for c, v in occ.items():
        out[frozenset(c)] = v if isinstance(v, (int, float)) else 1
    return out


def auc(y, sc):
    """Tie-corrected AUC via average ranks."""
    o = np.argsort(sc, kind="stable")
    r = np.empty(len(sc), dtype=float)
    r[o] = np.arange(1, len(sc) + 1)
    ss, rs = sc[o], r[o]
    k = 0
    while k < len(ss):
        j = k
        while j + 1 < len(ss) and ss[j + 1] == ss[k]:
            j += 1
        if j > k:
            rs[k:j + 1] = rs[k:j + 1].mean()
        k = j + 1
    r[o] = rs
    p, n = int(y.sum()), int((~y).sum())
    if p == 0 or n == 0:
        return float("nan")
    return (r[y].sum() - p * (p + 1) / 2) / (p * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--start_month", type=str, default=None)
    ap.add_argument("--end_month", type=str, default="2024-12")
    ap.add_argument("--max_set_size", type=int, default=30)
    ap.add_argument("--max_sources", type=int, default=1500)
    ap.add_argument("--max_muts", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "24_source_dependence.csv"))
    args = ap.parse_args()

    gd = Path(args.graphs_dir)
    months = sorted(pd.read_csv(gd / "index.tsv", sep="\t")["month"].tolist())
    if args.start_month:
        months = [m for m in months if m >= args.start_month]
    if args.end_month:
        months = [m for m in months if m <= args.end_month]
    N = len(pd.read_csv(gd / "posres_vocab.tsv", sep="\t"))
    log(f"{len(months)} months: {months[0]} .. {months[-1]}  N={N}  "
        f"min_count={args.min_count}\n")

    cache = {}

    def H(mo):
        if mo not in cache:
            with open(gd / f"{mo}_occupied.pkl", "rb") as fh:
                raw = constellations_of(pickle.load(fh))
            cache[mo] = {c: v for c, v in raw.items()
                         if v >= args.min_count and 1 <= len(c) <= args.max_set_size}
        return cache[mo]

    rows = []
    for i in range(len(months) - 1):
        mt, mth = months[i], months[i + 1]
        Ht, Hth = H(mt), H(mth)
        if len(Ht) < 30 or not Hth:
            continue

        srcs = [c for c, _ in sorted(Ht.items(), key=lambda kv: -kv[1])
                if len(c) < args.max_set_size][:args.max_sources]
        mass = Counter()
        for c, v in Ht.items():
            for m in c:
                mass[m] += v
        muts = [m for m, _ in mass.most_common(args.max_muts)]
        if len(srcs) < 30 or len(muts) < 10:
            continue

        occ_th = set(Hth.keys())
        ns, nm = len(srcs), len(muts)
        add = np.zeros((ns, nm), dtype=bool)
        valid = np.zeros((ns, nm), dtype=bool)
        for a, s in enumerate(srcs):
            for b, m in enumerate(muts):
                if m in s:
                    continue
                valid[a, b] = True
                if frozenset(set(s) | {m}) in occ_th:
                    add[a, b] = True

        n_add = int(add.sum())
        if n_add < 30:
            continue

        # source composition: binary over the full mutation vocabulary
        S = np.zeros((ns, N), dtype=np.float32)
        for a, s in enumerate(srcs):
            for m in s:
                if 0 <= m < N:
                    S[a, m] = 1.0
        Sn = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-9)

        col_add = add.sum(axis=0).astype(float)
        col_val = valid.sum(axis=0).astype(float)
        rate_m = np.divide(col_add, np.maximum(col_val, 1))
        rm2d = np.broadcast_to(rate_m[None, :], add.shape)

        # leave-one-out source affinity
        aff = np.zeros(add.shape, dtype=np.float32)
        for b in range(nm):
            idx = np.where(add[:, b])[0]
            if len(idx) < 2:
                continue
            tot = Sn[idx].sum(0)
            base = Sn @ tot
            base[idx] -= (Sn[idx] * Sn[idx]).sum(1)     # leave-one-out
            k = np.where(add[:, b], len(idx) - 1, len(idx))
            aff[:, b] = base / np.maximum(k, 1)
        z = (aff - aff.mean()) / max(aff.std(), 1e-9)
        combo = rm2d * np.exp(z)

        y = add[valid]
        a_marg = auc(y, rm2d[valid])
        a_comb = auc(y, combo[valid])
        gain = a_comb - a_marg

        top10 = np.sort(col_add)[::-1][:10].sum() / max(n_add, 1)

        rows.append(dict(
            month_t=mt, month_th=mth, n_sources=ns, n_muts=nm,
            n_valid_pairs=int(valid.sum()), n_additions=n_add,
            pair_rate=n_add / max(int(valid.sum()), 1),
            n_distinct_muts_added=int((col_add > 0).sum()),
            top10_share=top10,
            auc_marginal=a_marg, auc_marginal_plus_source=a_comb, auc_gain=gain,
            frac_sources_spawn=float((add.sum(1) > 0).mean()),
        ))
        log(f"  {mt}->{mth} | src={ns:5d} adds={n_add:6d} "
            f"muts_used={int((col_add>0).sum()):4d} top10={top10:.3f} | "
            f"AUC marg={a_marg:.3f} +src={a_comb:.3f} gain={gain:+.4f}")

    if not rows:
        log("no usable month pairs")
        return

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    g = df["auc_gain"]
    log("\n" + "=" * 74)
    log(f"SUMMARY over {len(df)} month pairs")
    log("=" * 74)
    for c in ["n_sources", "n_additions", "pair_rate", "n_distinct_muts_added",
              "top10_share", "auc_marginal", "auc_marginal_plus_source",
              "auc_gain", "frac_sources_spawn"]:
        log(f"  {c:<28} {df[c].mean():.4f}")
    log(f"  auc_gain > 0 in {(g > 0).sum()}/{len(g)} months")

    log("\n" + "-" * 74)
    log("VERDICT")
    log("-" * 74)
    mg = g.mean()
    log(f"  mean AUC gain from adding source information: {mg:+.4f}")
    log(f"  (synthetic reference: -0.010 when the source is irrelevant,")
    log(f"   +0.039 when the generative process is source-dependent)")
    log("")
    if mg > 0.02 and (g > 0).mean() > 0.7:
        log("  -> Source identity carries real, consistent signal beyond marginal")
        log("     frequency. An edit model f(s, m) has something to learn that the")
        log("     independence baseline cannot express. BUILD IT.")
    elif mg < 0.005:
        log("  -> Source identity adds essentially nothing over marginal frequency.")
        log("     f(s, m) would collapse to f(m). An edit model would be the")
        log("     frequency baseline with more machinery. DO NOT BUILD IT.")
        log("     This is a reportable negative result about the central hypothesis.")
    else:
        log("  -> Weak or inconsistent. Some signal, not much. Check whether the")
        log("     gain concentrates in particular months (high-turnover periods)")
        log("     rather than being uniform -- that would still be worth pursuing,")
        log("     but as a regime-specific effect, not a general rule.")
    log(f"\n  top-10 mutations account for {df['top10_share'].mean():.1%} of additions")
    log(f"  {df['frac_sources_spawn'].mean():.1%} of sources spawn anything")
    log(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
