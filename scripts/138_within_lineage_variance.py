#!/usr/bin/env python3
"""
138_within_lineage_variance.py

GO/NO-GO CHECK for the coupled-emission (pairwise J) route.

Question
--------
A global coupling matrix J learns that mutations u and v go together. But two
mutations co-occur for two very different reasons:

    (a) COMPATIBILITY  - they genuinely work well together (what we want)
    (b) CO-INHERITANCE - they are both lineage-defining for BA.2 (nuisance)

If every strong pair is lineage-defining, then J is a re-encoding of Pango
identity and the "background structure" result is vacuous. This script measures
how much pair association survives after conditioning on lineage.

Method
------
For each Pango lineage L with enough sequences, and each mutation pair (u,v)
that is POLYMORPHIC within L (both present in some sequences and absent in
others), build the 2x2 table:

              v=1    v=0
      u=1      a      b
      u=0      c      d

Pool across lineages with the Mantel-Haenszel odds ratio:

      OR_MH = sum_L (a_L d_L / n_L) / sum_L (b_L c_L / n_L)

OR_MH is the within-lineage association. Compare against the marginal OR
computed on the pooled population (which contains co-inheritance).

Outputs
-------
  n_usable_pairs   pairs with within-lineage variation in >= MIN_LINEAGES
  |log OR_MH|      size of the surviving within-lineage association
  spearman(marginal, MH)  how much marginal association is co-inheritance

DECISION RULE (set before looking):
  n_usable_pairs < 200                      -> NO-GO. Nothing for J to fit.
  median |log OR_MH| < 0.2 over usable      -> NO-GO. Association is all lineage.
  otherwise                                 -> GO, and n_usable_pairs is the
                                               effective parameter budget for J.

Usage
-----
  python 138_within_lineage_variance.py \
      --metadata data/raw/metadata.tsv.zst \
      --train-end 2022-06 \
      --out results/138_within_lineage.npz

NOTE ON SCHEMA: column names below are guesses based on the standard GISAID
metadata dump. Run with --list-columns first to see what your file actually
has, then set --lineage-col / --date-col / --subs-col accordingly.
"""

import argparse
import io
import re
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    import zstandard as zstd
except ImportError:
    sys.exit("need zstandard: pip install zstandard")


# ----------------------------------------------------------------------------
# Parameters that define the design. Change these deliberately, not casually.
# ----------------------------------------------------------------------------
MIN_SEQS_PER_LINEAGE = 100    # lineage needs this many seqs to contribute
MIN_MUT_COUNT        = 50     # mutation needs this many occurrences overall
POLY_LO, POLY_HI     = 0.05, 0.95   # "polymorphic within lineage" band
MIN_LINEAGES         = 3      # pair needs variation in this many lineages
MAX_MUTATIONS        = 1000   # cap vocabulary for memory


def open_maybe_zst(path):
    if str(path).endswith(".zst"):
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8",
                                errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_spike_subs(field):
    """Extract spike substitutions from an AA-substitution field.

    Handles the common GISAID format: (Spike_D614G,NSP3_A1892T,...)
    Returns a list of tokens like 'S:D614G'. Deletions kept as-is.
    """
    if not field or field in ("?", "", "NA"):
        return []
    toks = re.findall(r"Spike_([A-Za-z0-9_*-]+)", field)
    if not toks:
        # fall back: maybe already in 'S:D614G' form, comma separated
        toks = [t.split(":", 1)[1] for t in field.replace("(", "")
                .replace(")", "").split(",")
                if t.startswith("S:") and ":" in t]
    return [f"S:{t}" for t in toks]


def month_of(datestr):
    """YYYY-MM from a date field, or None if not resolvable to a month."""
    if not datestr or len(datestr) < 7:
        return None
    m = datestr[:7]
    return m if re.fullmatch(r"\d{4}-\d{2}", m) else None


def load(path, lineage_col, date_col, subs_col, train_end, list_columns=False):
    """Single streaming pass. Returns (rows, vocab_counter).

    rows: list of (lineage, [mutation tokens])
    """
    rows = []
    vocab = Counter()
    with open_maybe_zst(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if list_columns:
            for i, c in enumerate(header):
                print(f"  [{i}] {c}")
            sys.exit(0)

        try:
            i_lin = header.index(lineage_col)
            i_dat = header.index(date_col)
            i_sub = header.index(subs_col)
        except ValueError as e:
            sys.exit(f"column not found: {e}\nrun with --list-columns to inspect")

        for n, line in enumerate(fh):
            if n % 500_000 == 0 and n:
                print(f"  ...{n:,} rows", file=sys.stderr)
            f = line.rstrip("\n").split("\t")
            if len(f) <= max(i_lin, i_dat, i_sub):
                continue

            mo = month_of(f[i_dat])
            if mo is None or mo > train_end:
                continue          # LEAKAGE GUARD: training window only

            lin = f[i_lin].strip()
            if not lin or lin in ("None", "Unassigned", "?"):
                continue

            muts = parse_spike_subs(f[i_sub])
            if not muts:
                continue

            vocab.update(set(muts))
            rows.append((lin, muts))

    return rows, vocab


def build_matrices(rows, vocab):
    """Vocabulary -> index map, and per-lineage binary presence matrices."""
    keep = [m for m, c in vocab.most_common(MAX_MUTATIONS) if c >= MIN_MUT_COUNT]
    keep.sort()
    idx = {m: i for i, m in enumerate(keep)}
    V = len(keep)
    print(f"vocabulary: {V} mutations (>= {MIN_MUT_COUNT} occurrences)")

    by_lineage = defaultdict(list)
    for lin, muts in rows:
        v = np.zeros(V, dtype=np.uint8)
        for m in muts:
            j = idx.get(m)
            if j is not None:
                v[j] = 1
        by_lineage[lin].append(v)

    mats = {}
    for lin, vecs in by_lineage.items():
        if len(vecs) >= MIN_SEQS_PER_LINEAGE:
            mats[lin] = np.stack(vecs)
    print(f"lineages: {len(mats)} with >= {MIN_SEQS_PER_LINEAGE} sequences")
    return keep, mats


def mantel_haenszel(mats, V):
    """Accumulate MH numerator/denominator and count contributing lineages.

    Only pairs polymorphic within a lineage contribute from that lineage --
    this is the whole point. A pair fixed in BA.2 tells us nothing about
    compatibility, only about BA.2.
    """
    num = np.zeros((V, V))
    den = np.zeros((V, V))
    nlin = np.zeros((V, V), dtype=np.int32)

    for li, (lin, S) in enumerate(mats.items()):
        n = S.shape[0]
        S = S.astype(np.float64)
        freq = S.mean(axis=0)
        poly = (freq > POLY_LO) & (freq < POLY_HI)
        if poly.sum() < 2:
            continue

        cnt = S.sum(axis=0)                  # per-mutation counts
        a = S.T @ S                          # both present
        b = cnt[:, None] - a                 # u only
        c = cnt[None, :] - a                 # v only
        d = n - a - b - c                    # neither

        mask = np.outer(poly, poly)
        np.fill_diagonal(mask, False)

        num += np.where(mask, a * d / n, 0.0)
        den += np.where(mask, b * c / n, 0.0)
        nlin += mask.astype(np.int32)

        if li % 25 == 0:
            print(f"  lineage {li}/{len(mats)}: {lin} "
                  f"(n={n}, poly={int(poly.sum())})", file=sys.stderr)

    return num, den, nlin


def marginal_or(mats, V):
    """Marginal OR on the pooled population -- includes co-inheritance."""
    tot_a = np.zeros((V, V))
    tot_cnt = np.zeros(V)
    N = 0
    for S in mats.values():
        S = S.astype(np.float64)
        tot_a += S.T @ S
        tot_cnt += S.sum(axis=0)
        N += S.shape[0]
    a = tot_a
    b = tot_cnt[:, None] - a
    c = tot_cnt[None, :] - a
    d = N - a - b - c
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--train-end", required=True,
                    help="YYYY-MM; rows after this are dropped (leakage guard)")
    ap.add_argument("--lineage-col", default="Pango lineage")
    ap.add_argument("--date-col", default="Collection date")
    ap.add_argument("--subs-col", default="AA Substitutions")
    ap.add_argument("--out", default="results/138_within_lineage.npz")
    ap.add_argument("--list-columns", action="store_true")
    args = ap.parse_args()

    print(f"loading (train window <= {args.train_end})")
    rows, vocab = load(args.metadata, args.lineage_col, args.date_col,
                       args.subs_col, args.train_end, args.list_columns)
    print(f"  {len(rows):,} sequences, {len(vocab)} distinct spike mutations")

    keep, mats = build_matrices(rows, vocab)
    V = len(keep)
    if V < 2 or not mats:
        sys.exit("not enough data after filtering -- check column names")

    print("accumulating within-lineage 2x2 tables")
    num, den, nlin = mantel_haenszel(mats, V)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_or_mh = np.log((num + 0.5) / (den + 0.5))
    log_or_marg = marginal_or(mats, V)

    iu = np.triu_indices(V, k=1)
    usable = nlin[iu] >= MIN_LINEAGES
    n_usable = int(usable.sum())
    n_total = len(iu[0])

    mh_u = log_or_mh[iu][usable]
    mg_u = log_or_marg[iu][usable]

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    print(f"total pairs in vocabulary        {n_total:,}")
    print(f"usable (polymorphic in >={MIN_LINEAGES} lin)  {n_usable:,} "
          f"({100 * n_usable / max(n_total, 1):.1f}%)")

    if n_usable == 0:
        print("\nNO-GO: no pair varies within any lineage.")
        print("A global J would encode Pango identity and nothing else.")
        return

    med = float(np.median(np.abs(mh_u)))
    print(f"\nwithin-lineage |log OR|")
    print(f"  median   {med:.3f}")
    print(f"  90th pct {float(np.percentile(np.abs(mh_u), 90)):.3f}")
    print(f"  max      {float(np.abs(mh_u).max()):.3f}")
    print(f"  |log OR| > 0.5 : {int((np.abs(mh_u) > 0.5).sum()):,} pairs")
    print(f"  |log OR| > 1.0 : {int((np.abs(mh_u) > 1.0).sum()):,} pairs")

    try:
        from scipy.stats import spearmanr
        rho = spearmanr(mg_u, mh_u).correlation
        print(f"\nspearman(marginal, within-lineage) = {rho:.3f}")
        print("  high rho -> marginal association survives conditioning")
        print("  low rho  -> marginal association was mostly co-inheritance")
    except ImportError:
        pass

    print("\n" + "-" * 66)
    if n_usable < 200:
        print("NO-GO: too few pairs with within-lineage variation.")
    elif med < 0.2:
        print("NO-GO: within-lineage association is negligible.")
    else:
        print(f"GO: {n_usable:,} pairs carry within-lineage association.")
        print(f"    Use this as the parameter budget for J -- restrict J to")
        print(f"    these pairs rather than all {n_total:,}.")
    print("-" * 66)

    np.savez_compressed(
        args.out,
        mutations=np.array(keep),
        log_or_mh=log_or_mh,
        log_or_marginal=log_or_marg,
        n_lineages=nlin,
        train_end=args.train_end,
    )
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
