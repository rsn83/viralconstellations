#!/usr/bin/env python
"""
21_build_ref_anchored_node_features.py

Replaces the pooled per-month ESM node embedding with reference-anchored
per-mutation CONTRASTS.

WHY
---
The current feature is  E[ emb(sequence) | mutation m present ],  a count-
weighted mean over the constellations containing m in a given month. Which
constellations contain m is determined by lineage, so that quantity varies
mainly with lineage composition, not with the mutation. It is a LINEAGE
DESCRIPTOR indexed by a mutation. That is consistent with the observed
ablation behaviour: ESM helps when lineage identity happens to be predictive
and hurts when it is a confound (no_esm_context beat full_model on weight
correlation in Alpha, Gamma and Delta).

No PLM variant-effect method uses a pooled marginal. EVE (Frazer et al.) and
Shin et al. score a mutation by log p(mutant)/p(wildtype); Hie et al. use
grammaticality (probability of the mutant residue in context) and semantic
change (embedding distance to the original). All are DIFFERENCES against a
FIXED background.

WHAT IS KEPT AND WHAT IS DROPPED
--------------------------------
SEQUENCE context is kept in full. Every number below comes from ESM attending
over the entire spike, so structural neighbourhood and fold context are all
still in there.

POPULATION context is dropped -- not by the subtraction, but because no
circulating sequence is ever fed to ESM. ref and mut are both derived from the
Wuhan reference, so there is no lineage signal to leak. The subtraction does a
different job: it removes the ~99.9% of the representation that just says
"this is spike protein", which would otherwise swamp a one-residue change.

Population context is not lost from the system. It reaches the model through
node frequency features, hyperedge incidence, the GRU, and the new set-history
head -- places where the model can condition on it rather than receiving it
pre-averaged.

OUTPUT
------
outputs/esm_node_features_ref.pkl
  {"features": float32 (N, d), "names": [...], "dim": d, "meta": {...}}
Indexed by node index from posres_vocab.tsv. STATIC: one vector per
(position, residue), identical in every month. It is a vocabulary lookup,
like a word embedding -- you do not recompute GloVe vectors per sentence.

Usage
-----
  pip install fair-esm
  python scripts/21_build_ref_anchored_node_features.py
  python scripts/21_build_ref_anchored_node_features.py --esm_model esm2_t33_650M_UR50D
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
AAS = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy, net charge at pH 7, van der Waals volume (A^3)
HYDRO = dict(zip(AAS, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                       1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
CHARGE = {a: 0.0 for a in AAS}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})
VOL = dict(zip(AAS, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6,
                     166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0,
                     227.8, 193.6]))

# SARS-CoV-2 spike domains (1-indexed, approximate boundaries)
DOMAINS = [("NTD", 13, 305), ("RBD", 319, 541), ("RBM", 437, 508),
           ("FP", 788, 806), ("HR1", 912, 984), ("HR2", 1163, 1213)]


def log(m):
    print(m, flush=True)


def read_fasta(path: Path) -> str:
    seq = []
    for line in path.read_text().splitlines():
        if not line.startswith(">"):
            seq.append(line.strip())
    return "".join(seq)


def domain_onehot(pos1):
    v = np.zeros(len(DOMAINS) + 1, dtype=np.float32)
    hit = False
    for i, (_, lo, hi) in enumerate(DOMAINS):
        if lo <= pos1 <= hi:
            v[i] = 1.0
            hit = True
    if not hit:
        v[-1] = 1.0  # "other"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir",
                    default=str(ROOT / "data" / "processed" / "full_data_graphs_posres"))
    ap.add_argument("--reference", default=str(ROOT / "data" / "raw" / "spike_reference.fasta"))
    ap.add_argument("--esm_model", default="esm2_t30_150M_UR50D",
                    help="esm2_t30_150M_UR50D (640d), esm2_t33_650M_UR50D (1280d)")
    ap.add_argument("--window", type=int, default=7,
                    help="+/- residues around the mutated position used for the "
                         "local delta. Sequence-mean pooling divides a single "
                         "substitution's signal by ~1273 and buries it in noise.")
    ap.add_argument("--n_pca", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "esm_node_features_ref.pkl"))
    args = ap.parse_args()

    import esm as esm_lib

    # ---------------- vocab ----------------
    vocab = pd.read_csv(Path(args.graphs_dir) / "posres_vocab.tsv", sep="\t")
    log(f"posres_vocab.tsv columns: {list(vocab.columns)}")
    cols = {c.lower(): c for c in vocab.columns}
    pos_col = next((cols[c] for c in ("pos", "position", "site") if c in cols), None)
    res_col = next((cols[c] for c in ("res", "residue", "aa", "aa_to", "alt") if c in cols), None)
    if pos_col is None or res_col is None:
        raise SystemExit(f"could not find position/residue columns in {list(vocab.columns)}; "
                         f"edit the lookup above")
    N = len(vocab)
    log(f"N={N} (position,residue) nodes")

    ref = read_fasta(Path(args.reference))
    log(f"reference spike length {len(ref)}")

    # ---------------- ESM ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = getattr(esm_lib.pretrained, args.esm_model)()
    model = model.eval().to(device)
    bc = alphabet.get_batch_converter()
    layer = model.num_layers
    log(f"loaded {args.esm_model}, {layer} layers, device={device}")

    @torch.no_grad()
    def run(seqs):
        _, _, toks = bc([(f"s{i}", s) for i, s in enumerate(seqs)])
        out = model(toks.to(device), repr_layers=[layer], return_contacts=False)
        # strip BOS/EOS -> (B, L, d) aligned to sequence positions
        return out["logits"][:, 1:-1, :], out["representations"][layer][:, 1:-1, :]

    logits_ref, H_ref = run([ref])
    logits_ref, H_ref = logits_ref[0], H_ref[0]
    d_model = H_ref.shape[-1]
    log(f"esm hidden dim {d_model}")

    # ---------------- Pass A: masked marginals, one per POSITION ----------------
    positions = sorted({int(p) for p in vocab[pos_col]})
    log(f"Pass A: masked marginals at {len(positions)} distinct positions")
    mask_idx = alphabet.mask_idx
    llr_table = {}   # pos1 -> {aa: llr}
    for s0 in range(0, len(positions), args.batch):
        chunk = positions[s0:s0 + args.batch]
        seqs = []
        for p1 in chunk:
            i0 = p1 - 1
            seqs.append(ref[:i0] + "<mask>" + ref[i0 + 1:] if False else ref)
        # build token batch manually so we can insert the mask token
        _, _, toks = bc([(f"s{k}", ref) for k in range(len(chunk))])
        for k, p1 in enumerate(chunk):
            toks[k, p1] = mask_idx        # +1 offset for BOS
        with torch.no_grad():
            out = model(toks.to(device), repr_layers=[], return_contacts=False)
        lp = torch.log_softmax(out["logits"], dim=-1)
        for k, p1 in enumerate(chunk):
            row = lp[k, p1]
            wt = ref[p1 - 1]
            wt_lp = float(row[alphabet.get_idx(wt)]) if wt in AAS else 0.0
            llr_table[p1] = {a: float(row[alphabet.get_idx(a)]) - wt_lp for a in AAS}
        if s0 % (args.batch * 20) == 0:
            log(f"  {s0}/{len(positions)}")

    # ---------------- Pass B: representation deltas, one per MUTATION ----------
    log("Pass B: representation deltas")
    deltas = np.zeros((N, d_model), dtype=np.float32)
    scorable = np.zeros(N, dtype=bool)
    rows = [(i, int(r[pos_col]), str(r[res_col]).strip().upper())
            for i, (_, r) in enumerate(vocab.iterrows())]
    todo = [(i, p, a) for i, p, a in rows if a in AAS and 1 <= p <= len(ref) and ref[p - 1] != a]

    for s0 in range(0, len(todo), args.batch):
        chunk = todo[s0:s0 + args.batch]
        seqs = []
        for _, p1, aa in chunk:
            seqs.append(ref[:p1 - 1] + aa + ref[p1:])
        _, Hm = run(seqs)
        for k, (ni, p1, aa) in enumerate(chunk):
            D = Hm[k] - H_ref                                   # (L, d)
            lo = max(0, p1 - 1 - args.window)
            hi = min(D.shape[0], p1 + args.window)
            deltas[ni] = D[lo:hi].mean(dim=0).cpu().numpy()
            scorable[ni] = True
        if s0 % (args.batch * 50) == 0:
            log(f"  {s0}/{len(todo)}")

    log(f"scorable: {scorable.sum()}/{N} "
        f"({N - scorable.sum()} deletions / non-standard / same-as-reference)")

    # ---------------- assemble ----------------
    sem = np.linalg.norm(deltas, axis=1).astype(np.float32)

    # PCA of the DELTA DIRECTION. Direction is a context-free property of the
    # substitution: two mutations perturbing the representation the same way
    # are plausibly substitutable. Fit on the mutation vocabulary only -- these
    # vectors depend on (position, residue) and the reference, never on
    # observed counts or on anything from a future month.
    norm = np.linalg.norm(deltas, axis=1, keepdims=True)
    dirs = deltas / np.clip(norm, 1e-8, None)
    fit = dirs[scorable]
    n_pca = min(args.n_pca, fit.shape[0], fit.shape[1]) if fit.shape[0] > 1 else 0
    if n_pca > 0:
        mu = fit.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False)
        comps = Vt[:n_pca]
        dir_pca = ((dirs - mu) @ comps.T).astype(np.float32)
        dir_pca[~scorable] = 0.0
    else:
        dir_pca = np.zeros((N, 0), dtype=np.float32)

    feats, names = [], []

    llr = np.zeros(N, dtype=np.float32)
    for i, p1, aa in rows:
        if aa in AAS and p1 in llr_table:
            llr[i] = llr_table[p1][aa]
    feats.append(llr[:, None]);  names.append("llr_ref")
    feats.append(sem[:, None]);  names.append("sem_ref")
    feats.append(scorable.astype(np.float32)[:, None]); names.append("is_scorable")

    # cheap context-free extras: ESM gives one scalar and a direction; raw
    # position, domain and substitution chemistry are independent signals.
    chem = np.zeros((N, 4), dtype=np.float32)
    posn = np.zeros((N, 1), dtype=np.float32)
    dom = np.zeros((N, len(DOMAINS) + 1), dtype=np.float32)
    for i, p1, aa in rows:
        posn[i, 0] = p1 / max(len(ref), 1)
        dom[i] = domain_onehot(p1)
        wt = ref[p1 - 1] if 1 <= p1 <= len(ref) else None
        if aa in AAS and wt in AAS:
            chem[i] = [HYDRO[aa] - HYDRO[wt], CHARGE[aa] - CHARGE[wt],
                       (VOL[aa] - VOL[wt]) / 100.0, 1.0]
    feats.append(chem); names += ["d_hydro", "d_charge", "d_volume", "chem_valid"]
    feats.append(posn); names.append("pos_norm")
    feats.append(dom);  names += [f"dom_{n}" for n, _, _ in DOMAINS] + ["dom_other"]
    if dir_pca.shape[1]:
        feats.append(dir_pca); names += [f"dir_pc{k}" for k in range(dir_pca.shape[1])]

    X = np.concatenate(feats, axis=1).astype(np.float32)

    # standardise continuous columns (one-hots and flags left alone)
    skip = {"is_scorable", "chem_valid"} | {f"dom_{n}" for n, _, _ in DOMAINS} | {"dom_other"}
    for j, nm in enumerate(names):
        if nm in skip:
            continue
        col = X[:, j]
        sd = col.std()
        if sd > 1e-8:
            X[:, j] = (col - col.mean()) / sd

    out = {"features": X, "names": names, "dim": X.shape[1],
           "meta": {"esm_model": args.esm_model, "window": args.window,
                    "n_scorable": int(scorable.sum()), "N": N}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(out, fh)

    log(f"\nwrote {args.out}  shape={X.shape}")
    log(f"features: {names}")
    log("\nSANITY CHECK -- llr_ref should be strongly negative in conserved")
    log("structural regions and near zero at variable NTD loop sites. If it is")
    log("flat across the protein, the masking in Pass A is wrong.")
    ss = pd.DataFrame({"pos": [p for _, p, _ in rows], "llr": llr})
    ss["dom"] = ss["pos"].apply(lambda p: next((n for n, lo, hi in DOMAINS if lo <= p <= hi), "other"))
    log(ss.groupby("dom")["llr"].agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
