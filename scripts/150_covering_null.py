#!/usr/bin/env python3
"""
150_covering_null.py -- Minimum-source covering measurement with a
frequency-rank-matched null.

QUESTION
--------
For a constellation C that first appears at month t+h, how many haplotypes
present at month <= t are needed so that their union contains C?

If that number is small, novel constellations are COMPOSITIONAL: assembled
from material spread across several co-present haplotypes.

If the same number is just as small for size- and rarity-matched random
constellations, then it reflects the density of a 500-haplotype population
over V=1359 mutations, and says nothing about evolution.

The measurement is the COMPARISON, not the number.

WHAT IS AND IS NOT CLAIMED
--------------------------
Claimed:      C's composition is reachable by unioning material co-present at t.
Not claimed:  C descends from those haplotypes.
Not claimed:  any mutation was acquired at any particular time.
Not claimed:  the minimal cover is the true decomposition. It is usually not
              unique; n_min_covers records that ambiguity explicitly.

DATA CONTRACT
-------------
load_months() must return an ordered list of dicts, one per month:

    {
      "month": "2020-06",                  # str, chronological
      "sets":  [frozenset({3, 17, 402}),   # top-500 constellations,
                frozenset({3, 17, 55}),    #   mutation indices in [0, V)
                ...],
      "freq":  np.array([0.31, 0.12, ...]) # same length as "sets"
    }

Nothing else is required.

USAGE
    python 150_covering_null.py --data <path> --horizon 6 --n-null 100 \
        --dev-windows 0:20 --out results/
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------
# DATA LOADING -- replace the body with your own loader.
# ----------------------------------------------------------------------

SET_KEYS = ["sets", "constellations", "haplotypes", "X", "matrix", "occupied",
            "binary", "data", "rows"]
FREQ_KEYS = ["freq", "freqs", "frequency", "frequencies", "weights", "w",
             "counts", "count", "n", "mass", "prop"]


def _to_sets(obj):
    """Coerce whatever is in the pickle into a list of frozensets."""
    import scipy.sparse as sp

    if sp.issparse(obj):
        csr = obj.tocsr()
        return [frozenset(csr.indices[csr.indptr[i]:csr.indptr[i + 1]].tolist())
                for i in range(csr.shape[0])]
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        return [frozenset(np.flatnonzero(r).tolist()) for r in obj]
    if isinstance(obj, (list, tuple)) and obj:
        first = obj[0]
        if isinstance(first, (set, frozenset)):
            return [frozenset(s) for s in obj]
        if isinstance(first, (list, tuple, np.ndarray)):
            arr = np.asarray(obj)
            if arr.ndim == 2 and set(np.unique(arr)).issubset({0, 1}):
                return [frozenset(np.flatnonzero(r).tolist()) for r in arr]
            return [frozenset(np.asarray(s).ravel().tolist()) for s in obj]
    raise TypeError(f"cannot coerce {type(obj)} to a list of sets")


def _find(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k in lower:
            return lower[k]
    return None


def load_months(path, month_range=None, top=500, verbose=False):
    """Load one *_occupied.pkl per month from `path`.

    Auto-detects the common container shapes. Run with --inspect first to see
    what it found before trusting any numbers.
    """
    import glob
    import pickle
    import re

    files = sorted(glob.glob(os.path.join(path, "*_occupied.pkl")))
    if not files:
        raise FileNotFoundError(f"no *_occupied.pkl under {path}")

    lo = hi = None
    if month_range:
        lo, hi = month_range.split(":")

    months = []
    for fp in files:
        m = re.search(r"(\d{4}-\d{2})_occupied\.pkl$", os.path.basename(fp))
        if not m:
            continue
        label = m.group(1)
        if lo and (label < lo or label > hi):
            continue
        with open(fp, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            keyset = set(map(str, obj.keys()))
            if keyset & set(SET_KEYS) or keyset & set(FREQ_KEYS):
                raw_sets = _find(obj, SET_KEYS)
                raw_freq = _find(obj, FREQ_KEYS)
                if raw_sets is None:
                    raise KeyError(f"{fp}: no set-like key in {sorted(keyset)}")
                sets = _to_sets(raw_sets)
                freq = (np.asarray(raw_freq, dtype=float)
                        if raw_freq is not None else np.ones(len(sets)))
            else:
                # dict mapping constellation -> count/frequency
                items = list(obj.items())
                sets = [frozenset(k) for k, _ in items]
                freq = np.asarray([float(v) for _, v in items])
        elif isinstance(obj, tuple) and len(obj) == 2:
            sets, freq = _to_sets(obj[0]), np.asarray(obj[1], dtype=float)
        else:
            sets, freq = _to_sets(obj), None
            freq = np.ones(len(sets))

        if freq.sum() > 0:
            freq = freq / freq.sum()
        if top and len(sets) > top:
            keep = np.argsort(-freq)[:top]
            sets = [sets[i] for i in keep]
            freq = freq[keep]
        months.append({"month": label, "sets": sets, "freq": freq})

        if verbose:
            sizes = [len(s) for s in sets]
            print(f"{label}: {len(sets)} sets, "
                  f"size median {int(np.median(sizes))} "
                  f"[{min(sizes)}-{max(sizes)}], "
                  f"mass {freq.sum():.3f}, "
                  f"max idx {max(max(s) for s in sets if s)}")

    months.sort(key=lambda d: d["month"])
    return months


def inspect(path):
    """Print the raw structure of one pickle, then a parsed summary."""
    import glob
    import pickle

    fp = sorted(glob.glob(os.path.join(path, "*_occupied.pkl")))[0]
    with open(fp, "rb") as f:
        obj = pickle.load(f)
    print(f"--- raw structure of {os.path.basename(fp)} ---")
    print(f"type: {type(obj)}")
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:12]:
            shape = getattr(v, "shape", None)
            print(f"  {k!r}: {type(v).__name__}"
                  f"{f' shape={shape}' if shape is not None else ''}"
                  f"{f' len={len(v)}' if hasattr(v, '__len__') and shape is None else ''}")
    elif isinstance(obj, (list, tuple)):
        print(f"  len={len(obj)}, first element type={type(obj[0])}")
        print(f"  first element: {str(obj[0])[:200]}")
    else:
        print(f"  shape={getattr(obj, 'shape', None)}")
    print("\n--- parsed ---")
    load_months(path, top=500, verbose=True)


# ----------------------------------------------------------------------
# MINIMUM SET COVER (exact, branch and bound)
# ----------------------------------------------------------------------

def _popcount(x):
    try:
        return x.bit_count()          # Python 3.10+
    except AttributeError:
        return bin(x).count("1")


def project_and_prune(target, sources, prune_max=120):
    """Project sources onto target, dedupe, drop dominated masks.

    Returns (masks, n_elems, elem_order, coverable_mask).
    Dropping a mask that is a subset of another never changes the minimum
    cover size, and it shrinks the search space by a large factor.
    """
    elem_order = sorted(target)
    idx = {m: i for i, m in enumerate(elem_order)}
    n = len(elem_order)

    raw = set()
    coverable = 0
    for s in sources:
        inter = s & target
        if inter:
            mask = 0
            for m in inter:
                mask |= 1 << idx[m]
            raw.add(mask)
            coverable |= mask

    # Dedupe is the big win: projecting hundreds of sources onto a few dozen
    # mutations collapses most of them to identical masks. Domination pruning
    # is O(m^2) and only pays off once the deduped set is small, so skip it
    # above prune_max. Neither step changes the minimum cover.
    if len(raw) <= prune_max:
        ordered = sorted(raw, key=_popcount, reverse=True)
        keep = []
        for m in ordered:
            if not any((m | k) == k for k in keep):
                keep.append(m)
    else:
        keep = sorted(raw, key=_popcount, reverse=True)

    return keep, n, elem_order, coverable


def greedy_cover(masks, full):
    covered, k = 0, 0
    while covered != full:
        best = max(masks, key=lambda m: _popcount(m & ~covered))
        gain = _popcount(best & ~covered)
        if gain == 0:
            return None
        covered |= best
        k += 1
    return k


class SourceIndex:
    """Sparse index over source haplotypes for fast projection onto a target.

    The per-source Python loop is the bottleneck once the pool grows past a few
    thousand sets, so intersections are computed as a single sparse column
    slice, deduped as packed bytes, and only the unique masks are converted to
    Python ints.
    """

    def __init__(self, sets, V):
        import scipy.sparse as sp
        rows, cols = [], []
        for i, s in enumerate(sets):
            for m in s:
                rows.append(i)
                cols.append(m)
        self.M = sp.csc_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(len(sets), V))
        self.n = len(sets)

    def project(self, target):
        """Return (masks, n_elems, coverable) with bits indexed by sorted(target)."""
        elem_order = sorted(target)
        n = len(elem_order)
        sub = self.M[:, elem_order].toarray()
        sub = sub[sub.any(axis=1)]
        if sub.size == 0:
            return [], n, 0
        packed = np.packbits(sub, axis=1)
        packed = np.unique(packed, axis=0)
        masks, coverable = [], 0
        for row in packed:
            v = int.from_bytes(row.tobytes(), "big")
            masks.append(v)
            coverable |= v
        masks.sort(key=_popcount, reverse=True)
        return masks, n, coverable



def greedy_cover_masks(masks, full):
    """Greedy cover, returning the chosen masks."""
    covered, chosen = 0, []
    while covered != full:
        best = max(masks, key=lambda m: _popcount(m & ~covered))
        if _popcount(best & ~covered) == 0:
            return chosen
        covered |= best
        chosen.append(best)
    return chosen


def exact_min_cover(masks, full, count_optima=False, cap=12,
                    optima_cap=1000, node_budget=200000):
    """Exact minimum cover of `full` using `masks`.

    Returns (size, n_optimal_covers, exact_flag, cover_masks).

    Counting ALL optimal covers is exponential -- with hundreds of sources over
    a 40-mutation target there can be millions -- so optima counting is off by
    default and capped when on. If the search exceeds node_budget, the greedy
    upper bound is returned with exact_flag False.
    """
    if full == 0:
        return 0, 1, True, []
    ub = greedy_cover(masks, full)
    if ub is None:
        return None, 0, True, []

    # If the greedy solution already matches the trivial lower bound, it is
    # optimal and no search is needed. This short-circuits most large targets.
    biggest = max(_popcount(m & full) for m in masks)
    if biggest and ub <= -(-_popcount(full) // biggest):
        return ub, 0, True, greedy_cover_masks(masks, full)

    best = [ub]
    best_cov = [greedy_cover_masks(masks, full)]
    n_opt = [0]
    nodes = [0]
    aborted = [False]

    covers_elem = defaultdict(list)
    for m in masks:
        mm = m & full
        while mm:
            low = mm & -mm
            covers_elem[low.bit_length() - 1].append(m)
            mm ^= low

    def rec(covered, depth, chosen):
        if aborted[0]:
            return
        nodes[0] += 1
        if nodes[0] > node_budget:
            aborted[0] = True
            return
        if covered == full:
            if depth < best[0]:
                best[0] = depth
                best_cov[0] = list(chosen)
                n_opt[0] = 1
            elif depth == best[0] and count_optima and n_opt[0] < optima_cap:
                n_opt[0] += 1
            return
        if depth + 1 > best[0] or depth >= cap:
            return
        rem = full & ~covered

        # pivot: least-covered uncovered element. Scan elements, not all masks.
        pivot, fewest, maxgain = None, None, 0
        mm = rem
        while mm:
            low = mm & -mm
            e = low.bit_length() - 1
            cands = [m for m in covers_elem[e] if m & rem]
            if fewest is None or len(cands) < fewest:
                pivot, fewest, pivot_cands = e, len(cands), cands
            for m in cands:
                g = _popcount(m & rem)
                if g > maxgain:
                    maxgain = g
            mm ^= low
        if maxgain == 0:
            return
        if depth + -(-_popcount(rem) // maxgain) > best[0]:
            return
        # try high-gain candidates first so the bound tightens early
        for m in sorted(pivot_cands, key=lambda x: -_popcount(x & rem)):
            rec(covered | m, depth + 1, chosen + [m])

    rec(0, 0, [])
    if aborted[0]:
        return ub, 0, False, greedy_cover_masks(masks, full)
    return best[0], (n_opt[0] if count_optima else 0), True, best_cov[0]


# ----------------------------------------------------------------------
# FREQUENCY-RANK-MATCHED NULL
# ----------------------------------------------------------------------

def mutation_frequency(month):
    """Population frequency of each mutation at this month."""
    V_local = defaultdict(float)
    for s, f in zip(month["sets"], month["freq"]):
        for m in s:
            V_local[m] += f
    return V_local


def build_rank_bands(freq_map, n_bands=20):
    """Split mutations present at t into equal-count bands by frequency rank.

    The band is the unit of matching. Sampling a replacement from the same band
    preserves how common a mutation is -- and therefore how many haplotypes
    contain it -- while destroying which mutations co-occur.
    """
    muts = sorted(freq_map.keys(), key=lambda m: -freq_map[m])
    bands = np.array_split(np.array(muts), min(n_bands, max(1, len(muts))))
    mut2band = {}
    for b, arr in enumerate(bands):
        for m in arr:
            mut2band[int(m)] = b
    return [list(map(int, b)) for b in bands], mut2band


def sample_matched(target, bands, mut2band, rng, max_tries=50):
    """Draw a synthetic constellation matching target's size and rank profile."""
    out = set()
    for m in target:
        b = mut2band.get(m)
        if b is None:
            continue
        pool = bands[b]
        for _ in range(max_tries):
            cand = int(rng.choice(pool))
            if cand not in out:
                out.add(cand)
                break
    return frozenset(out)


# ----------------------------------------------------------------------
# MAIN MEASUREMENT
# ----------------------------------------------------------------------

def analyse_window(months, t_idx, horizon, n_null, rng,
                   source_window=None, n_bands=20, count_optima=False,
                   max_targets=None):
    """One (t -> t+h) window. Returns a list of per-constellation records."""
    if t_idx + horizon >= len(months):
        return []

    src_lo = 0 if source_window is None else max(0, t_idx - source_window + 1)
    source_sets, seen = [], set()
    for mo in months[src_lo:t_idx + 1]:
        for s in mo["sets"]:
            if s not in seen:
                seen.add(s)
                source_sets.append(s)

    # "Novel" = absent from every month up to and including t (not just the
    # source window), so a constellation that merely resurfaces is not novel.
    ever_seen = set()
    for mo in months[:t_idx + 1]:
        ever_seen.update(mo["sets"])

    present_muts = set(mutation_frequency(months[t_idx]).keys())
    bands, mut2band = build_rank_bands(
        mutation_frequency(months[t_idx]), n_bands=n_bands)
    min_band = min(len(b) for b in bands) if bands else 0
    n_present = len(present_muts)
    n_sources = len(source_sets)
    V = 1 + max((max(s) for s in source_sets if s), default=0)
    V = max(V, 1 + max((max(s) for s in months[t_idx + horizon]["sets"] if s),
                       default=0))
    index = SourceIndex(source_sets, V)

    records = []
    targets = [C for C in months[t_idx + horizon]["sets"] if C not in ever_seen]
    if max_targets and len(targets) > max_targets:
        pick = rng.choice(len(targets), max_targets, replace=False)
        targets = [targets[i] for i in pick]
    for C in targets:
        # De novo mutations cannot be covered by construction. Separate them
        # out and report them; they are the epsilon term, not a cover failure.
        C_cov = frozenset(C & present_muts)
        n_denovo = len(C) - len(C_cov)
        if not C_cov:
            continue

        masks, n_el, coverable = index.project(C_cov)
        if not masks:
            continue
        n_uncoverable = n_el - _popcount(coverable)
        k_real, n_opt, exact, cov_masks = exact_min_cover(
            masks, coverable, count_optima=count_optima)

        # Granularity: how much of C does the single largest source supply?
        # A value near 1.0 means "one near-parent plus scraps" (accumulative),
        # not genuine combination of material from distinct haplotypes.
        # Granularity. Raw mask sizes overlap, so also record DISJOINT marginal
        # contributions in cover order, and the residual left by the single
        # best source. The latter is the decisive quantity: if one haplotype
        # already covers nearly all of C, novelty is extension of a near-parent,
        # not combination of material from distinct haplotypes.
        n_cov = _popcount(coverable)
        best_single = max(_popcount(m) for m in masks)
        best_single_frac = best_single / n_cov if n_cov else float("nan")
        n_residual = n_cov - best_single

        if cov_masks and n_cov:
            contribs = sorted((_popcount(m & coverable) for m in cov_masks),
                              reverse=True)
            top_frac = contribs[0] / n_cov
            second_frac = (contribs[1] / n_cov) if len(contribs) > 1 else 0.0
            marg, seen_bits = [], 0
            for m in sorted(cov_masks, key=_popcount, reverse=True):
                marg.append(_popcount(m & ~seen_bits) / n_cov)
                seen_bits |= m
            second_marginal = marg[1] if len(marg) > 1 else 0.0
        else:
            top_frac = second_frac = second_marginal = float("nan")

        k_null = []
        for _ in range(n_null):
            Ct = sample_matched(C_cov, bands, mut2band, rng)
            if not Ct:
                continue
            m2, n2, cov2 = index.project(Ct)
            if not m2:
                continue
            k2, _, _, _ = exact_min_cover(m2, cov2, count_optima=False)
            if k2 is not None:
                k_null.append(k2)

        if not k_null:
            continue
        records.append({
            "t": months[t_idx]["month"],
            "t_plus_h": months[t_idx + horizon]["month"],
            "size": len(C),
            "size_coverable": n_el - n_uncoverable,
            "n_denovo": n_denovo,
            "k_real": k_real,
            "k_null_mean": float(np.mean(k_null)),
            "k_null_sd": float(np.std(k_null)),
            "delta": k_real - float(np.mean(k_null)),
            "n_min_covers": n_opt,
            "exact": bool(exact),
            "top_frac": float(top_frac),
            "second_frac": float(second_frac),
            "n_contribs": len(cov_masks) if cov_masks else 0,
            "best_single_frac": float(best_single_frac),
            "n_residual": int(n_residual),
            "second_marginal": float(second_marginal),
            "n_sources": n_sources,
            "n_present_muts": n_present,
            "min_band_size": min_band,
        })
    return records


def summarise(records):
    if not records:
        return {}
    d = np.array([r["delta"] for r in records])
    kr = np.array([r["k_real"] for r in records])
    kn = np.array([r["k_null_mean"] for r in records])
    boot = [np.mean(rng_) for rng_ in
            (np.random.default_rng(0).choice(d, (2000, len(d))))]
    return {
        "n": len(records),
        "mean_k_real": float(kr.mean()),
        "mean_k_null": float(kn.mean()),
        "mean_delta": float(d.mean()),
        "delta_ci95": [float(np.percentile(boot, 2.5)),
                       float(np.percentile(boot, 97.5))],
        "frac_real_below_null": float((d < 0).mean()),
        "frac_multi_source_real": float((kr > 1).mean()),
        "frac_multi_source_null": float((kn > 1).mean()),
        "median_n_min_covers": float(np.median(
            [r["n_min_covers"] for r in records])),
        "frac_exact": float(np.mean([r.get("exact", True) for r in records])),
        "mean_top_frac": float(np.nanmean([r["top_frac"] for r in records])),
        "median_top_frac": float(np.nanmedian([r["top_frac"] for r in records])),
        "frac_top_over_90pct": float(np.nanmean(
            [float(r["top_frac"] > 0.9) for r in records])),
        "mean_second_frac": float(np.nanmean(
            [r["second_frac"] for r in records])),
        # sanity: mean cover size implied by the recorded contributions must
        # match mean_k_real. If these disagree, the cover profile is wrong.
        "mean_n_contribs": float(np.mean(
            [r["n_contribs"] for r in records])),
        # decisive statistics
        "median_best_single_frac": float(np.nanmedian(
            [r["best_single_frac"] for r in records])),
        "median_n_residual": float(np.median(
            [r["n_residual"] for r in records])),
        "mean_n_residual": float(np.mean(
            [r["n_residual"] for r in records])),
        "frac_residual_le_2": float(np.mean(
            [r["n_residual"] <= 2 for r in records])),
        "mean_second_marginal": float(np.nanmean(
            [r["second_marginal"] for r in records])),
    }


# ----------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------

def make_figure(records, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kr = np.array([r["k_real"] for r in records])
    kn = np.array([r["k_null_mean"] for r in records])
    d = np.array([r["delta"] for r in records])
    sz = np.array([r["size_coverable"] for r in records])
    dn = np.array([r["n_denovo"] for r in records])

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))

    # A: distribution of minimum source count
    bins = np.arange(0.5, max(6, kr.max() + 1.5))
    ax[0, 0].hist([kr, kn], bins=bins, label=["real", "null"], density=True)
    ax[0, 0].set_xlabel("minimum number of source haplotypes")
    ax[0, 0].set_ylabel("fraction of novel constellations")
    ax[0, 0].set_title("A. Source multiplicity")
    ax[0, 0].legend()

    # B: paired difference -- the actual result
    ax[0, 1].hist(d, bins=40)
    ax[0, 1].axvline(0, color="k", lw=1)
    ax[0, 1].axvline(d.mean(), color="r", lw=1.5,
                     label=f"mean {d.mean():.2f}")
    ax[0, 1].set_xlabel("k_real - mean(k_null),  per constellation")
    ax[0, 1].set_title("B. Paired difference (< 0 = structure)")
    ax[0, 1].legend()

    # C: stratified by size -- rules out the size confound
    edges = np.unique(np.percentile(sz, [0, 20, 40, 60, 80, 100]))
    ctr, mr, mn, er, en = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sz >= lo) & (sz <= hi)
        if sel.sum() < 5:
            continue
        ctr.append(0.5 * (lo + hi))
        mr.append(kr[sel].mean()); er.append(kr[sel].std() / np.sqrt(sel.sum()))
        mn.append(kn[sel].mean()); en.append(kn[sel].std() / np.sqrt(sel.sum()))
    ax[1, 0].errorbar(ctr, mr, yerr=er, marker="o", label="real")
    ax[1, 0].errorbar(ctr, mn, yerr=en, marker="s", label="null")
    ax[1, 0].set_xlabel("constellation size (coverable mutations)")
    ax[1, 0].set_ylabel("mean minimum sources")
    ax[1, 0].set_title("C. Stratified by size")
    ax[1, 0].legend()

    # D: de novo residual
    ax[1, 1].hist(dn, bins=np.arange(-0.5, max(4, dn.max() + 1.5)))
    ax[1, 1].set_xlabel("mutations absent from population at t")
    ax[1, 1].set_title(
        f"D. De novo residual ({(dn > 0).mean():.1%} have >= 1)")

    # E/F: period structure -- essential here because constellation size grows
    # roughly 30x across the range, so a pooled number would mostly track size.
    tm = np.array([r["t"] for r in records])
    umonths = sorted(set(tm))
    xs = np.arange(len(umonths))
    md, mr2, mn2, msz = [], [], [], []
    for u in umonths:
        sel = tm == u
        md.append(d[sel].mean())
        mr2.append(kr[sel].mean())
        mn2.append(kn[sel].mean())
        msz.append(sz[sel].mean())
    ax[0, 2].plot(xs, md, marker=".")
    ax[0, 2].axhline(0, color="k", lw=1)
    ax[0, 2].set_title("E. Effect over time")
    ax[0, 2].set_ylabel("mean delta")
    ax[1, 2].plot(xs, mr2, marker=".", label="real")
    ax[1, 2].plot(xs, mn2, marker=".", label="null")
    ax2 = ax[1, 2].twinx()
    ax2.plot(xs, msz, color="grey", ls=":", label="size")
    ax2.set_ylabel("mean size", color="grey")
    ax[1, 2].set_title("F. Sources and size over time")
    ax[1, 2].legend(loc="upper left")
    for a_ in (ax[0, 2], ax[1, 2]):
        step = max(1, len(umonths) // 6)
        a_.set_xticks(xs[::step])
        a_.set_xticklabels([umonths[i] for i in range(0, len(umonths), step)],
                           rotation=45, ha="right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--n-null", type=int, default=100)
    p.add_argument("--n-bands", type=int, default=20)
    p.add_argument("--source-window", type=int, default=None,
                   help="months of history in the source pool; default all")
    p.add_argument("--dev-windows", default=None,
                   help="LO:HI index range of t to use. Set this and keep the "
                        "rest held out.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results_150")
    p.add_argument("--months", default="2020-06:2023-06")
    p.add_argument("--top", type=int, default=500)
    p.add_argument("--inspect", action="store_true",
                   help="print pickle structure and per-month summary, then exit")
    p.add_argument("--max-targets", type=int, default=None,
                   help="subsample novel constellations per window (runtime)")
    a = p.parse_args()

    if a.inspect:
        inspect(a.data)
        return

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    months = load_months(a.data, month_range=a.months, top=a.top, verbose=True)

    if a.dev_windows:
        lo, hi = (int(x) for x in a.dev_windows.split(":"))
    else:
        lo, hi = 0, len(months) - a.horizon
    print(f"windows t in [{lo}, {hi}), horizon {a.horizon}")

    records = []
    for t in range(lo, min(hi, len(months) - a.horizon)):
        t0 = time.time()
        r = analyse_window(months, t, a.horizon, a.n_null, rng,
                           source_window=a.source_window, n_bands=a.n_bands,
                           max_targets=a.max_targets)
        dt = time.time() - t0
        per = dt / max(1, len(r))
        print(f"  t={months[t]['month']}  novel={len(r)}  "
              f"{dt:.1f}s  ({per:.2f}s/target)", flush=True)
        records.extend(r)

    tag = f"h{a.horizon}"
    with open(os.path.join(a.out, f"records_{tag}.json"), "w") as f:
        json.dump(records, f)
    summary = summarise(records)
    with open(os.path.join(a.out, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    if records:
        make_figure(records, os.path.join(a.out, f"covering_{tag}.png"))


if __name__ == "__main__":
    main()
