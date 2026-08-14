"""
scripts/20_build_structural_prior.py

Parses a real PDB structure (default: 6VXX, the closed-state
SARS-CoV-2 spike, Walls et al. 2020, Cell) into an N x N structural
proximity matrix aligned to posres_vocab.tsv's aa_pos numbering, and
saves it to outputs/structural_prior.pt for reuse.

This is genuinely a fixed PRIOR, not learned or data-dependent: it
reflects the folded 3D structure of the protein itself, identical
every month, computed once and reused across every training run.

Contact definition: two positions are "in contact" if their CA-CA
(alpha carbon) distance is below --contact_threshold Angstroms (8.0
default -- a standard, commonly-used CA-CA contact cutoff in structural
biology literature).

Coverage limitation, expected and reported: cryo-EM structures like
6VXX do not resolve every residue -- flexible loops and termini are
often missing density. Positions not resolved in the structure get
zero rows/columns (no structural signal available for them; this is
not a bug, it's what "the structure doesn't tell us about this region"
actually means). The script reports coverage % so you know how much of
your vocab this prior actually informs.

Usage (after downloading the PDB file locally, e.g. in Colab):
  !curl -s "https://files.rcsb.org/download/6VXX.pdb" -o data/raw/6vxx.pdb
  python scripts/20_build_structural_prior.py --pdb_path data/raw/6vxx.pdb
"""

import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--pdb_path", type=str, default="data/raw/6vxx.pdb")
parser.add_argument("--chain_id", type=str, default="A",
                     help="6VXX is a homotrimer (3 identical chains) -- using one chain's "
                          "intra-chain distances only; inter-chain (trimer interface) "
                          "contacts are not included in this pass")
parser.add_argument("--contact_threshold", type=float, default=8.0,
                     help="CA-CA distance (Angstroms) below which two positions are "
                          "considered structurally in contact")
parser.add_argument("--position_col", type=str, default="aa_pos")
args = parser.parse_args()

import numpy as np
import pandas as pd
import torch


def log(msg): print(msg, flush=True)


def parse_ca_coordinates(pdb_path: Path, chain_id: str) -> dict:
    """Returns {residue_number: (x, y, z)} for CA atoms in the given chain.
    Fixed-column PDB ATOM record parsing, per the standard PDB format spec."""
    coords = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21]
            if chain != chain_id:
                continue
            try:
                res_seq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            if res_seq not in coords:  # first occurrence only (ignore altloc duplicates)
                coords[res_seq] = (x, y, z)
    return coords


def main():
    pdb_path = ROOT / args.pdb_path
    if not pdb_path.exists():
        log(f"ERROR: {pdb_path} not found. Download it first, e.g.:\n"
            f'  curl -s "https://files.rcsb.org/download/6VXX.pdb" -o {pdb_path}')
        sys.exit(1)

    graphs_dir = ROOT / "data" / "processed" / "full_data_graphs_posres"
    vocab_df = pd.read_csv(graphs_dir / "posres_vocab.tsv", sep="\t")
    N = len(vocab_df)
    log(f"N={N} nodes")

    coords = parse_ca_coordinates(pdb_path, args.chain_id)
    log(f"Parsed {len(coords)} resolved CA positions from chain {args.chain_id}")

    positions = vocab_df[args.position_col].values
    resolved_mask = np.array([p in coords for p in positions])
    log(f"Vocab coverage: {resolved_mask.sum()}/{N} nodes have a resolved structural "
        f"position ({100*resolved_mask.sum()/N:.1f}%)")
    if resolved_mask.sum() < N * 0.3:
        log("WARNING: low coverage -- check --chain_id and that positions use the same "
            "numbering convention as this PDB structure (both should be standard "
            "Wuhan-Hu-1 spike numbering, but verify with a known contact, e.g. RBD "
            "residues ~438-506, before trusting this prior).")

    coord_arr = np.zeros((N, 3), dtype=np.float64)
    for i, p in enumerate(positions):
        if p in coords:
            coord_arr[i] = coords[p]

    diff = coord_arr[:, None, :] - coord_arr[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    struct = (dist <= args.contact_threshold).astype(np.float32)
    struct[~resolved_mask, :] = 0.0
    struct[:, ~resolved_mask] = 0.0
    np.fill_diagonal(struct, 0.0)  # no self-loops here -- handled inside the conv layer

    n_contacts = int(struct.sum())
    log(f"Contact threshold={args.contact_threshold} Angstroms -> {n_contacts} directed "
        f"contact entries ({n_contacts / max(resolved_mask.sum(), 1):.1f} avg contacts "
        f"per resolved node)")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "structural_prior.pt"
    torch.save({
        "struct": torch.tensor(struct, dtype=torch.float32),
        "resolved_mask": torch.tensor(resolved_mask, dtype=torch.bool),
        "chain_id": args.chain_id,
        "contact_threshold": args.contact_threshold,
        "pdb_path": str(args.pdb_path),
    }, out_path)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
