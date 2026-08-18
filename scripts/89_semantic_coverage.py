#!/usr/bin/env python
"""
89_semantic_coverage.py

The mechanism being tested
--------------------------
Each new variant must be more semantically distant from the previous dominant
to escape accumulated immunity. When the maximum semantic distance achievable
by one-step grammar-guided additions to the incumbent falls below the distance
to the actual new variant, the jump was necessary -- not random arrival.

For each known transition month, this script computes:

  1. The semantic position of the current dominant cluster (centroid of its
     sets in dir_pc embedding space).

  2. The semantic positions of all one-step candidates: sets reachable by
     adding one mutation to a currently circulating set. Weighted by the
     parent set's frequency and the mutation's grammar (llr_ref).

  3. The maximum semantic distance from the dominant achievable by one-step
     grammar-guided search -- the local semantic reachability.

  4. The semantic position of the actual new dominant (the cluster that took
     over next month).

  5. The distance from the old dominant to the new dominant.

If distance(old dominant, new dominant) > max one-step reachability, the
transition required a jump outside the local grammar neighbourhood.

The prediction: this gap increases over time, as the accumulated immune
memory forces each new variant to seek greater semantic divergence than
grammar-guided single-step search can provide.

Also computed: the semantic coverage radius C_t -- the convex hull of
dominant clusters from all previous months -- and whether the new dominant
falls inside or outside it.

Usage
-----
python scripts/89_semantic_coverage.py --esm outputs/esm_node_features_ref.pkl
python scripts/89_semantic_coverage.py --self_test
"""

import argparse
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"(\d{4}-\d{2})_occupied\.pkl$")

TRANSITIONS = {
    "2021-01": "Alpha",
    "2021-06": "Delta",
    "2021-12": "Omicron_BA1",
    "2022-03": "BA2",
    "2022-06": "BA5",
    "2023-02": "XBB",
    "2023-12": "JN1",
}


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_esm(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    F = np.asarray(obj["features"])
    names = [str(x) for x in obj["names"]]
    return F, names


def load_vocab(data_dir):
    path = os.path.join(data_dir, "posres_vocab.tsv")
    idx2pr, pr2idx = {}, {}
    with open(path) as f:
        header = f.readline().strip().split("\t")
        cols = {c.lower(): i for i, c in enumerate(header)}
        id_col = next(cols[c] for c in ("node_idx", "id", "node") if c in cols)
        pos_col = next(cols[c] for c in ("aa_pos", "pos", "position") if c in cols)
        res_col = next(cols[c] for c in ("residue", "res", "aa") if c in cols)
        for line in f:
            parts = line.strip().split("\t")
            nid = int(parts[id_col])
            pos = int(parts[pos_col])
            res = str(parts[res_col])
            idx2pr[nid] = (pos, res)
            pr2idx[(pos, res)] = nid
    return idx2pr, pr2idx


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


# ----------------------------------------------------------------------------
# embedding functions
# ----------------------------------------------------------------------------

def set_emb(s, F, pr2idx, pc_cols):
    """Mean of per-mutation dir_pc vectors, unit-normalised."""
    vecs = [F[pr2idx[m], pc_cols] for m in s if m in pr2idx]
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cosine_dist(a, b):
    if a is None or b is None:
        return np.nan
    return float(1.0 - np.dot(a, b))


def grammar(s, F, pr2idx, llr_col):
    vals = [float(F[pr2idx[m], llr_col]) for m in s if m in pr2idx]
    return float(np.mean(vals)) if vals else 0.0


def dominant_emb(occ, F, pr2idx, pc_cols, top_n=50):
    """
    Embedding of the dominant cluster: weighted mean of top-N sets' embeddings,
    weighted by frequency.
    """
    items = sorted(occ.items(), key=lambda kv: -kv[1])[:top_n]
    tot = sum(w for _, w in items)
    acc = None
    for s, w in items:
        e = set_emb(s, F, pr2idx, pc_cols)
        if e is None:
            continue
        acc = e * w / tot if acc is None else acc + e * w / tot
    if acc is None:
        return None
    n = np.linalg.norm(acc)
    return acc / n if n > 1e-12 else acc


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    print("self-test")

    # cosine distance sanity
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_dist(a, a)) < 1e-9
    assert abs(cosine_dist(a, b) - 1.0) < 1e-9
    print("  cosine distance: identical 0, orthogonal 1       ok")

    # dominant embedding: weighted mean of two sets
    rng = np.random.default_rng(0)
    F = rng.normal(size=(4, 3))
    pr2idx = {("p1", "A"): 0, ("p2", "B"): 1,
              ("p1", "C"): 2, ("p2", "D"): 3}
    pc_cols = [0, 1, 2]
    occ = {frozenset({("p1", "A"), ("p2", "B")}): 900,
           frozenset({("p1", "C"), ("p2", "D")}): 100}
    emb = dominant_emb(occ, F, pr2idx, pc_cols, top_n=5)
    assert emb is not None and abs(np.linalg.norm(emb) - 1.0) < 1e-6
    print("  dominant embedding is unit-normalised             ok")

    # set_emb returns None for empty set
    assert set_emb(frozenset(), F, pr2idx, pc_cols) is None
    print("  empty set returns None                           ok")

    print("all checks passed\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="data/processed/full_data_graphs_posres")
    ap.add_argument("--esm", default="outputs/esm_node_features_ref.pkl")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--end_month", default="2024-12")
    ap.add_argument("--top_dominant", type=int, default=50,
                    help="top sets used to compute dominant embedding")
    ap.add_argument("--max_candidates", type=int, default=2000,
                    help="one-step candidates to score per month")
    ap.add_argument("--grammar_threshold", type=float, default=0.0,
                    help="minimum llr_ref to include a candidate mutation")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    os.makedirs(args.out_dir, exist_ok=True)

    F, feat_names = load_esm(args.esm)
    idx2pr, pr2idx = load_vocab(args.data_dir)
    llr_col = feat_names.index("llr_ref")
    pc_cols = [i for i, n in enumerate(feat_names) if n.startswith("dir_pc")]
    print(f"ESM: {F.shape[0]} cells, {len(pc_cols)} dir_pc dimensions")

    months = load_months(args.data_dir, args.min_count,
                         end_month=args.end_month)
    names = [m for m, _ in months]
    occ_by = {m: o for m, o in months}
    print(f"loaded {len(names)} months: {names[0]} .. {names[-1]}\n")

    # precompute dominant embedding per month
    dom_emb = {}
    for m in names:
        dom_emb[m] = dominant_emb(occ_by[m], F, pr2idx, pc_cols,
                                  args.top_dominant)

    # accumulate the hull of past dominant embeddings as a list of vectors
    past_doms = []

    rows = []
    for i, m in enumerate(names[:-1]):
        m_next = names[i + 1]
        occ_t = occ_by[m]
        occ_n = occ_by[m_next]
        d_cur = dom_emb[m]
        d_nxt = dom_emb[m_next]

        if d_cur is None or d_nxt is None:
            past_doms.append(d_cur)
            continue

        # distance from current dominant to next dominant
        dist_to_next = cosine_dist(d_cur, d_nxt)

        # one-step candidates: take top sets, add every mutation in vocab
        top_sets = sorted(occ_t.items(), key=lambda kv: -kv[1])
        top_sets = top_sets[:min(100, len(top_sets))]
        tot = float(sum(occ_t.values()))

        cand_dists = []
        n_tested = 0
        for cs, w in top_sets:
            freq = w / tot
            for mut, midx in pr2idx.items():
                if mut in cs:
                    continue
                llr = float(F[midx, llr_col])
                if llr < args.grammar_threshold:
                    continue
                new_set = frozenset(cs | {mut})
                e = set_emb(new_set, F, pr2idx, pc_cols)
                if e is None:
                    continue
                d = cosine_dist(d_cur, e)
                cand_dists.append((d, llr, freq))
                n_tested += 1
                if n_tested >= args.max_candidates:
                    break
            if n_tested >= args.max_candidates:
                break

        if not cand_dists:
            past_doms.append(d_cur)
            continue

        dists = np.array([x[0] for x in cand_dists])
        max_one_step = float(dists.max())
        mean_one_step = float(dists.mean())
        p95_one_step = float(np.percentile(dists, 95))

        # semantic coverage: mean distance from current dominant to all
        # past dominant embeddings (how spread out the hull is)
        if past_doms:
            past_valid = [v for v in past_doms if v is not None]
            if past_valid:
                cov_dists = [cosine_dist(d_cur, v) for v in past_valid]
                coverage_radius = float(np.mean(cov_dists))
            else:
                coverage_radius = np.nan
        else:
            coverage_radius = np.nan

        # gap: how much further the new dominant is than one-step search can reach
        gap = dist_to_next - max_one_step

        is_transition = m_next in TRANSITIONS
        variant_name = TRANSITIONS.get(m_next, "")

        rows.append({
            "month": m,
            "next_month": m_next,
            "variant": variant_name,
            "is_transition": is_transition,
            "dist_to_next_dominant": dist_to_next,
            "max_one_step_semantic": max_one_step,
            "p95_one_step_semantic": p95_one_step,
            "mean_one_step_semantic": mean_one_step,
            "gap_next_minus_max": gap,
            "semantic_coverage_radius": coverage_radius,
            "n_candidates_tested": n_tested,
        })

        tag = f"  ** {variant_name}" if is_transition else ""
        print(f"  {m} -> {m_next}: dist_to_next={dist_to_next:.4f}  "
              f"max_one_step={max_one_step:.4f}  gap={gap:+.4f}{tag}")

        past_doms.append(d_cur)

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/89_semantic_coverage.csv", index=False)

    print("\n" + "=" * 80)
    print("SEMANTIC REACHABILITY AT EACH TRANSITION")
    print("=" * 80)
    trans = df[df["is_transition"]].copy()
    if len(trans):
        print(trans[["next_month", "variant", "dist_to_next_dominant",
                      "max_one_step_semantic", "gap_next_minus_max",
                      "semantic_coverage_radius"]].round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("NON-TRANSITION MONTHS (baseline)")
    print("=" * 80)
    non = df[~df["is_transition"]]
    print(f"  mean dist_to_next_dominant  : "
          f"{non['dist_to_next_dominant'].mean():.4f}")
    print(f"  mean max_one_step_semantic  : "
          f"{non['max_one_step_semantic'].mean():.4f}")
    print(f"  mean gap                    : "
          f"{non['gap_next_minus_max'].mean():+.4f}")

    print("\n" + "=" * 80)
    print("READING")
    print("=" * 80)
    print("""
  dist_to_next_dominant: how far the new dominant is from the current one
     in embedding space. Large = the new variant is semantically distant.

  max_one_step_semantic: the furthest any one-step grammar-guided addition
     can reach from the current dominant. This is the local reachability.

  gap = dist_to_next - max_one_step:
     POSITIVE -> the new dominant is FURTHER than one-step search can reach.
     The transition required a jump outside the grammar neighbourhood.
     NEGATIVE -> the new dominant was reachable by one-step search.
     The transition was a grammar-guided step.

  If the gap is positive and large at transitions but near zero at
  non-transition months: variant emergence requires semantic jumps that
  single-step grammar-guided search cannot explain. This supports the
  mechanism: variants accumulate until the local semantic neighbourhood
  is exhausted, then a large jump occurs.

  If the gap increases across transitions (Alpha < Delta < Omicron < ...):
  each successive variant requires a larger semantic jump, consistent with
  the hypothesis that accumulated immunity forces increasingly distant escapes.
""")

    if len(trans) > 1:
        print("gap across transitions (should increase if mechanism holds):")
        for _, row in trans.iterrows():
            print(f"  {row['variant']:15s} ({row['next_month']}): "
                  f"gap = {row['gap_next_minus_max']:+.4f}  "
                  f"dist = {row['dist_to_next_dominant']:.4f}")

    print(f"\nwrote outputs/89_semantic_coverage.csv")


if __name__ == "__main__":
    main()
