#!/usr/bin/env python3
"""
85_profilehmm_vs_logistic.py

Empirically test the degeneracy claim:
    profile HMM (reference-anchored, gapless)  ==  per-position multinomial logistic

Two comparison levels:
  (A) PER-POSITION : HMMER match emissions  vs  logistic softmax probs
  (B) FULL-SEQUENCE: per-sequence log-likelihood on a held-out month, both models

Models compared:
  M0  profile HMM (pyhmmer/HMMER3), no time
  M1a intercept-only multinomial logistic  == per-position empirical frequency
  M1b time-varying multinomial logistic:  softmax(beta_pos + gamma_pos * t)

Usage:
    python 85_profilehmm_vs_logistic.py \
        --data-dir data/processed/full_data_graphs_posres \
        --vocab    data/processed/posres_vocab.tsv \
        --ref      data/raw/spike_reference.fasta \
        --train    2020-03:2020-12 \
        --test     2021-01
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_reference(path):
    seq = "".join(l.strip() for l in open(path) if not l.startswith(">"))
    return seq.upper()


def load_vocab(path):
    """posres_vocab.tsv: node_idx, aa_pos, residue, raw_count"""
    import csv
    node2pr = {}
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            node2pr[int(row["node_idx"])] = (int(row["aa_pos"]), row["residue"].strip())
    return node2pr


def load_month(data_dir, ym):
    """Returns list of (frozenset_of_node_ids, count).

    Handles several plausible pickle layouts; prints what it found so you can
    confirm the loader matched your actual format.
    """
    p = Path(data_dir) / f"{ym}_occupied.pkl"
    obj = pickle.load(open(p, "rb"))

    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and isinstance(vals[0], (int, np.integer)):      # {frozenset: count}
            return [(frozenset(k), int(v)) for k, v in obj.items()]
        for key in ("sets", "occupied", "constellations"):        # {'sets': [...]}
            if key in obj:
                return [(frozenset(s), 1) for s in obj[key]]
        return [(frozenset(v), 1) for v in vals]                  # {id: frozenset}
    if isinstance(obj, (list, tuple, set)):
        items = list(obj)
        if items and isinstance(items[0], tuple) and len(items[0]) == 2:
            return [(frozenset(s), int(c)) for s, c in items]
        return [(frozenset(s), 1) for s in items]
    raise ValueError(f"unrecognised pickle layout in {p}: {type(obj)}")


def months_in_range(spec):
    """'2020-03:2020-12' -> ['2020-03', ..., '2020-12']"""
    if ":" not in spec:
        return [spec]
    a, b = spec.split(":")
    ya, ma = map(int, a.split("-")); yb, mb = map(int, b.split("-"))
    out = []
    y, m = ya, ma
    while (y, m) <= (yb, mb):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13: m, y = 1, y + 1
    return out


# ----------------------------------------------------------------------------
# mutation sets  ->  full-length residue matrix
# ----------------------------------------------------------------------------
def sets_to_residue_matrix(records, node2pr, ref, verbose=True):
    """records: [(frozenset(node_ids), count)] -> (N, L) int8 matrix of AA indices,
    plus per-row weights. Applies each set's substitutions to the reference."""
    L = len(ref)
    base = np.array([AA_IDX.get(c, -1) for c in ref], dtype=np.int8)
    if (base < 0).any():
        bad = sorted({ref[i] for i in np.where(base < 0)[0]})
        print(f"  [warn] non-standard residues in reference: {bad} -> masked", file=sys.stderr)

    rows, wts, skipped = [], [], 0
    for s, c in records:
        v = base.copy()
        ok = True
        for n in s:
            pr = node2pr.get(n)
            if pr is None:
                skipped += 1; ok = False; break
            pos, res = pr
            ai = AA_IDX.get(res)
            if ai is None or not (1 <= pos <= L):   # deletions / stops / out-of-range
                skipped += 1; ok = False; break
            v[pos - 1] = ai
        if ok:
            rows.append(v); wts.append(c)
    if verbose:
        print(f"  {len(rows):,} sets kept, {skipped:,} skipped (unmapped node / non-AA residue)")
    return np.asarray(rows, dtype=np.int8), np.asarray(wts, dtype=np.float64)


# ----------------------------------------------------------------------------
# M0 : profile HMM
# ----------------------------------------------------------------------------
def fit_profile_hmm(mat, wts, prior=None, expand=True, max_seqs=50_000):
    """Build a profile HMM from a gapless MSA.

    prior=None   -> raw ML emissions (equals empirical frequency, exactly)
    prior='laplace' -> HMMER's Laplace prior
    expand=True  -> replicate each unique set `count` times (weights the MSA)
    """
    import pyhmmer
    from pyhmmer.easel import Alphabet, TextMSA, TextSequence
    from pyhmmer.plan7 import Builder, Background

    abc = Alphabet.amino()
    idx = np.arange(len(mat))
    if expand:
        target = wts.astype(float)
        if max_seqs and target.sum() > max_seqs:
            target = target * (max_seqs / target.sum())
        # stochastic rounding: unbiased, E[reps] = target exactly.
        # (the old max(1, int(.)) floor systematically overweighted rare sets)
        rng = np.random.default_rng(0)
        reps = np.floor(target).astype(int)
        reps += (rng.random(len(target)) < (target - np.floor(target))).astype(int)
        if reps.sum() == 0:
            reps = np.ones(len(target), dtype=int)
        idx = np.repeat(idx, reps)
        print(f"    MSA rows: {reps.sum():,} (from {wts.sum():,.0f} sequences)", flush=True)

    seqs = [TextSequence(name=f"s{i}".encode(),
                         sequence="".join(AA[a] for a in mat[j]))
            for i, j in enumerate(idx)]
    msa = TextMSA(name=b"train", sequences=seqs).digitize(abc)

    b = Builder(abc, weighting="none", effective_number="none", prior_scheme=prior)
    hmm, _, _ = b.build_msa(msa, Background(abc))
    emis = np.array(hmm.match_emissions)[1:, :20]      # (M, 20), drop dummy row
    return hmm, emis


# ----------------------------------------------------------------------------
# M1a / M1b : multinomial logistic over positions
# ----------------------------------------------------------------------------
def fit_logistic_intercept(mat, wts, L, pseudo=0.0):
    """Intercept-only multinomial logistic == weighted empirical frequency."""
    counts = np.full((L, 20), pseudo, dtype=np.float64)
    for a in range(20):
        counts[:, a] += (wts[:, None] * (mat == a)).sum(axis=0)
    return counts / counts.sum(axis=1, keepdims=True)


def aggregate_counts(mat, wts, tvec, L):
    """Sufficient statistics: (T, L, 20) weighted residue counts, one slice per month.
    t is constant within a month, so the time-varying logistic likelihood depends on
    the data only through these counts."""
    months = np.unique(tvec)
    C = np.zeros((len(months), L, 20))
    for k, t in enumerate(months):
        sel = tvec == t
        sub, w = mat[sel], wts[sel]
        for a in range(20):
            C[k, :, a] = (w[:, None] * (sub == a)).sum(axis=0)
    return C, months


def fit_logistic_time(mat, wts, tvec, L, beta_init=None, l2=1.0,
                      iters=3000, lr=1.0, tol=1e-7, verbose=True):
    """softmax(beta_pos + gamma_pos * t) on aggregated per-month counts.

    beta is warm-started at the intercept-only (M1a) solution, so M1b begins
    exactly where M1a ends and can only improve. Gradients are normalised by
    the per-position sequence total, making the step size scale-free.
    """
    C, months = aggregate_counts(mat, wts, tvec, L)          # (T, L, 20)
    t = (months - tvec.mean()) / (tvec.std() + 1e-9)
    Nk = C.sum(axis=2)                                        # (T, L)
    Ntot = Nk.sum(axis=0)[:, None]                            # (L, 1)

    beta = np.log(beta_init + 1e-12) if beta_init is not None else np.zeros((L, 20))
    beta -= beta.max(axis=1, keepdims=True)
    gamma = np.zeros((L, 20))

    prev = -np.inf
    for it in range(iters):
        eta = beta[None] + gamma[None] * t[:, None, None]
        eta -= eta.max(axis=2, keepdims=True)
        P = np.exp(eta); P /= P.sum(axis=2, keepdims=True)
        R = C - Nk[:, :, None] * P
        beta  += lr * (R.sum(axis=0)                  - l2 * beta ) / Ntot
        gamma += lr * ((R * t[:, None, None]).sum(0)  - l2 * gamma) / Ntot
        if (it + 1) % 100 == 0 or it == iters - 1:
            ll = (C * np.log(P + 1e-12)).sum() / C.sum()
            if verbose:
                print(f"    logistic-time iter {it+1}/{iters}  LL/residue = {ll:.6f}", flush=True)
            if ll - prev < tol:
                print(f"    converged at iter {it+1}", flush=True); break
            prev = ll
    return beta, gamma


def logistic_time_probs(beta, gamma, t_scalar):
    eta = beta + gamma * t_scalar
    eta -= eta.max(axis=1, keepdims=True)
    P = np.exp(eta)
    return P / P.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def seq_loglik(mat, probs, eps=1e-10):
    """Sum over positions of log p(residue). probs: (L, 20)."""
    lp = np.log(probs + eps)
    return lp[np.arange(mat.shape[1])[None, :], mat.astype(np.int64)].sum(axis=1)


def tvd(P, Q):
    return 0.5 * np.abs(P - Q).sum(axis=1)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--train", required=True, help="e.g. 2020-03:2020-12")
    ap.add_argument("--test", required=True, help="e.g. 2021-01")
    ap.add_argument("--max-seqs", type=int, default=50_000,
                    help="cap on MSA rows for the HMM build; 0 = use all sequences")
    ap.add_argument("--out", default="hmm_vs_logistic.npz")
    args = ap.parse_args()

    ref = load_reference(args.ref); L = len(ref)
    node2pr = load_vocab(args.vocab)
    print(f"reference length L = {L}   vocab nodes = {len(node2pr):,}")

    tr_months = months_in_range(args.train)
    te_months = months_in_range(args.test)

    print(f"\nloading train months {tr_months[0]} .. {tr_months[-1]}")
    tr_mats, tr_wts, tr_t = [], [], []
    for k, ym in enumerate(tr_months):
        recs = load_month(args.data_dir, ym)
        m, w = sets_to_residue_matrix(recs, node2pr, ref, verbose=False)
        if len(m) == 0:
            print(f"  {ym}: empty, skipped"); continue
        tr_mats.append(m); tr_wts.append(w); tr_t.append(np.full(len(m), k, float))
        print(f"  {ym}: {len(m):,} unique sets, {w.sum():,.0f} sequences", flush=True)
    Xtr = np.vstack(tr_mats); Wtr = np.concatenate(tr_wts); Ttr = np.concatenate(tr_t)

    print(f"\nloading test month(s) {te_months}")
    te_recs = []
    for ym in te_months:
        te_recs += load_month(args.data_dir, ym)
    Xte, Wte = sets_to_residue_matrix(te_recs, node2pr, ref)
    t_test = float(len(tr_months))          # next month index

    # ---- fit -----------------------------------------------------------
    print("\nfitting M0  profile HMM (no prior) ...", flush=True)
    _, E_hmm_ml = fit_profile_hmm(Xtr, Wtr, prior=None, max_seqs=args.max_seqs)
    print("fitting M0' profile HMM (laplace prior) ...", flush=True)
    _, E_hmm_lap = fit_profile_hmm(Xtr, Wtr, prior="laplace", max_seqs=args.max_seqs)

    print("fitting M1a intercept-only logistic ...", flush=True)
    P_freq = fit_logistic_intercept(Xtr, Wtr, L, pseudo=0.0)
    P_freq_ps = fit_logistic_intercept(Xtr, Wtr, L, pseudo=1.0)

    print("fitting M1b time-varying logistic ...", flush=True)
    beta, gamma = fit_logistic_time(Xtr, Wtr, Ttr, L, beta_init=P_freq_ps)
    P_time = logistic_time_probs(beta, gamma, (t_test - Ttr.mean()) / (Ttr.std() + 1e-9))

    # ---- (A) per-position comparison -----------------------------------
    print("\n" + "=" * 68)
    print("(A) PER-POSITION:  total variation distance between emission models")
    print("=" * 68)
    rows = [
        ("HMM(ML)     vs logistic(intercept)", tvd(E_hmm_ml,  P_freq)),
        ("HMM(laplace) vs logistic(+1 pseudo)", tvd(E_hmm_lap, P_freq_ps)),
        ("HMM(ML)     vs logistic(time@test)", tvd(E_hmm_ml,  P_time)),
    ]
    print(f"{'comparison':<38} {'mean TVD':>10} {'max TVD':>10} {'>0.01':>8}")
    for name, d in rows:
        print(f"{name:<38} {d.mean():>10.6f} {d.max():>10.6f} {(d>0.01).sum():>8}")

    # ---- (B) full-sequence comparison ----------------------------------
    print("\n" + "=" * 68)
    print(f"(B) FULL-SEQUENCE:  held-out {te_months}, {Wte.sum():,.0f} sequences")
    print("=" * 68)
    models = {
        "M0  profile HMM (laplace)": E_hmm_lap,
        "M1a logistic (intercept)":  P_freq_ps,
        "M1b logistic (+time)":      P_time,
    }
    from scipy.stats import spearmanr
    lls = {}
    print(f"{'model':<28} {'mean LL/seq':>13} {'per-res ppl':>13}")
    for name, P in models.items():
        ll = seq_loglik(Xte, P)
        lls[name] = ll
        mean_ll = np.average(ll, weights=Wte)
        ppl = np.exp(-mean_ll / L)
        print(f"{name:<28} {mean_ll:>13.3f} {ppl:>13.5f}")

    print("\nrank correlation of per-sequence log-likelihood (Spearman):")
    names = list(models)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = spearmanr(lls[names[i]], lls[names[j]]).statistic
            print(f"  {names[i]:<28} vs {names[j]:<28} rho = {r:.6f}")

    np.savez(args.out, E_hmm_ml=E_hmm_ml, E_hmm_lap=E_hmm_lap,
             P_freq=P_freq, P_freq_ps=P_freq_ps, P_time=P_time,
             beta=beta, gamma=gamma)
    print(f"\nsaved -> {args.out}")

    print("""
INTERPRETATION
  Row 1 of (A) should be ~0 to machine precision. That IS the degeneracy:
  with a fixed reference and no indels, the profile HMM's match emissions
  are exactly the per-position empirical frequencies, i.e. an intercept-only
  multinomial logistic. Any nonzero value means the loader introduced gaps
  or the MSA is not gapless -- investigate before proceeding.

  Row 2 isolates HMMER's Dirichlet prior as the only thing distinguishing
  the two implementations.

  In (B), all sequences have identical length L, so the HMMER null-model
  score is constant across sequences and bit-score ranking equals
  log-likelihood ranking. Spearman rho(M0, M1a) ~ 1.0 confirms the two
  models induce the same ordering on held-out data.

  M1b is the only model with a time term. Whatever it gains over M1a is
  what a time covariate buys you -- and the profile HMM cannot express it.
""")


if __name__ == "__main__":
    main()
