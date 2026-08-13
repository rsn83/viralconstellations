"""
Check B: does physical/structural proximity between two positions in the
folded spike protein predict appearance, independent of co-occurrence?

This is a genuinely different data source from everything else tested so
far -- it never touches your GISAID sequences or M_t matrices. It only
needs a solved spike protein structure.

WHERE TO GET A STRUCTURE
--------------------------
Download a SARS-CoV-2 spike protein PDB file, e.g.:
  - 6VXX (original Wuhan-Hu-1 spike, closed state): https://files.rcsb.org/download/6VXX.pdb
  - 7DF4 / 6M0J (spike RBD + ACE2): also usable, RBD region only
Save it locally, e.g. spike_6VXX.pdb, and pass its path below.

Note: PDB residue numbering may not exactly match your P=153 spike
position vocabulary from position_vocab.tsv (numbering offsets, missing
loops in the crystal structure, etc). You will likely need to align your
position indices to the PDB's residue numbers once -- check
position_vocab.tsv's 'aa_pos' column against the PDB SEQRES / residue
numbers for a handful of known landmark mutations (e.g. N501Y, E484K) to
confirm the offset before trusting the distances below.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, is_aa

from viralconstellations.checks import (
    correlation_and_vif, held_out_ap_comparison, held_out_ap_with_residual,
    frequency_matched_stratification, plot_stratification,
)


# ---------------------------------------------------------------------------
# Build a position -> position distance matrix from a PDB structure
# ---------------------------------------------------------------------------
def build_structural_distance_matrix(
    pdb_path: str,
    chain_id: str = "A",
    positions: list[int] | None = None,
    use_ca_only: bool = True,
) -> dict[int, dict[int, float]]:
    """
    Parse a PDB file and compute pairwise CA-CA (or min heavy-atom)
    distances in Angstroms between residues at the given PDB residue
    numbers. Returns {pos_i: {pos_j: distance_angstrom}}.

    `positions`: PDB residue numbers to include (e.g. your position_vocab
    aa_pos values, AFTER confirming numbering alignment). If None, uses
    all standard amino acid residues found in the chain.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("spike", pdb_path)
    model = structure[0]
    chain = model[chain_id]

    residues = {}
    for res in chain:
        if not is_aa(res, standard=True):
            continue
        resnum = res.id[1]
        if positions is not None and resnum not in positions:
            continue
        if use_ca_only:
            if "CA" not in res:
                continue
            residues[resnum] = res["CA"].coord
        else:
            residues[resnum] = np.array([a.coord for a in res])

    print(f"Loaded {len(residues)} residues from chain {chain_id} "
          f"(requested {len(positions) if positions else 'all'})")

    resnums = sorted(residues.keys())
    dist = {p: {} for p in resnums}
    for i, p in enumerate(resnums):
        for q in resnums[i:]:
            if use_ca_only:
                d = float(np.linalg.norm(residues[p] - residues[q]))
            else:
                # min heavy-atom distance (more accurate for contact, slower)
                d = float(np.min(np.linalg.norm(
                    residues[p][:, None, :] - residues[q][None, :, :], axis=-1
                )))
            dist[p][q] = d
            dist[q][p] = d
    return dist


def check_alignment(dist_matrix: dict, known_landmark_pairs: list[tuple[int, int, str]]):
    """
    Sanity check before trusting the distances: print distances for a few
    well-known functionally-linked position pairs (e.g. RBD contact
    residues) to eyeball whether numbering looks sane. Not a substitute
    for a real sequence alignment against the PDB SEQRES, just a smoke test.
    """
    print("\nLandmark pair distances (sanity check numbering alignment):")
    for p, q, label in known_landmark_pairs:
        if p in dist_matrix and q in dist_matrix.get(p, {}):
            print(f"  {label}: pos {p} <-> pos {q} = {dist_matrix[p][q]:.1f} A")
        else:
            print(f"  {label}: pos {p} or {q} not found in structure -- check numbering")


# ---------------------------------------------------------------------------
# Feature construction + Check B diagnostics
# ---------------------------------------------------------------------------
def add_structural_distance_feature(
    df: pd.DataFrame,
    candidates: list,          # list of frozenset candidates, aligned to df rows
    dist_matrix: dict,
    missing_value: float = 999.0,  # large distance = "not proximal" for missing pairs
) -> pd.DataFrame:
    """
    For each candidate, compute the mean pairwise structural distance
    (Angstroms) between its positions. Also adds a binary
    'any_contact_8A' feature (any pair within 8A, a common contact-map
    threshold) since proximity effects are often non-linear/thresholded.
    """
    mean_dist, any_contact = [], []
    for candidate in candidates:
        positions = list(candidate)
        if len(positions) < 2:
            mean_dist.append(missing_value)
            any_contact.append(0)
            continue
        pairs = [(positions[a], positions[b])
                 for a in range(len(positions))
                 for b in range(a + 1, len(positions))]
        ds = []
        for p, q in pairs:
            d = dist_matrix.get(p, {}).get(q, None)
            ds.append(d if d is not None else missing_value)
        mean_dist.append(float(np.mean(ds)))
        any_contact.append(int(any(d <= 8.0 for d in ds if d < missing_value)))

    out = df.copy()
    out["struct_mean_dist"] = mean_dist
    out["struct_any_contact_8A"] = any_contact
    return out


def run_check_B(df: pd.DataFrame, freq_cols: list[str]) -> None:
    """
    Four diagnostics on structural proximity, mirroring Check A / the
    original coo_support check. Uses inverse distance so 'larger = more
    proximal', consistent in direction with the other features (larger
    coo_support / g_t = more co-occurrence = expected positive effect).
    """
    df = df.copy()
    df["struct_proximity"] = 1.0 / (df["struct_mean_dist"] + 1.0)  # avoid div-by-0

    print("=" * 70); print("CHECK B: structural proximity"); print("=" * 70)

    print("\n1. Correlation / VIF"); print("-" * 40)
    correlation_and_vif(df, freq_cols + ["struct_proximity"])

    print("\n2. Held-out AP comparison (continuous proximity feature)"); print("-" * 40)
    held_out_ap_comparison(df, freq_cols, "struct_proximity")

    print("\n3. Residualized"); print("-" * 40)
    held_out_ap_with_residual(df, freq_cols, "struct_proximity")

    print("\n4. Frequency-matched stratification (binary contact feature)"); print("-" * 40)
    frequency_matched_stratification(df, freq_cols[0], "struct_any_contact_8A")


if __name__ == "__main__":
    print(__doc__)
    print("\nThis script requires a real PDB file -- no synthetic demo, since the")
    print("whole point is testing against a real solved structure.")
    print("\nExample usage once you have a PDB file and your candidate DataFrame:")
    print("""
    dist_matrix = build_structural_distance_matrix(
        "spike_6VXX.pdb", chain_id="A", positions=vocab_df["aa_pos"].tolist()
    )
    check_alignment(dist_matrix, [
        (501, 484, "N501Y <-> E484K (both RBD, known close)"),
    ])
    df = add_structural_distance_feature(df, candidates, dist_matrix)
    run_check_B(df, freq_cols=["max_parent_freq", "pred_freq_new_pos"])
    """)
