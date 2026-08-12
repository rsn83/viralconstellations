"""
Step 4: Build CATEGORICAL mutation matrices per month.

Each sequence is represented as a dense integer vector of length P_variable,
where P_variable = number of spike positions that showed any non-reference
amino acid across all training sequences.

Matrix entry [i, j]:
  0         = reference amino acid at position j (no mutation)
  1-20      = amino acid index (A=1, C=2, D=3, ..., Y=20)

Why categorical, not binary?
  Binary forces a separate column per named mutation event (N501Y, N501T, ...)
  and cannot represent unseen residues at known positions.
  Categorical: column j = position 501. Entry can be Y, T, S, or any AA —
  including ones never seen during training — because the model embeds
  (position, residue) factored into two separate sub-embeddings.

Position vocabulary:
  Saved to data/vocab/position_vocab.tsv:
    col 0: 1-indexed AA position in spike
    col 1: reference amino acid (single letter)
  Only positions with ≥ min_position_prevalence sequences showing any
  non-reference residue are included.

Also saved per month:
  YYYY-MM.npy        : (n_seq, P_variable) int8 categorical matrix
  YYYY-MM_posfreq.npy: (P_variable, 21) float32 residue frequency matrix
                       posfreq[j][k] = fraction of sequences with residue k
                       at position j. Used as conditioning signal.

Two-pass:
  Pass 1: collect all (position, residue) observations across all months
  Pass 2: encode categorical matrices using position vocabulary
"""

import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Amino acid encoding ────────────────────────────────────────────────────────

AA_LIST  = "ACDEFGHIKLMNPQRSTVWY"   # 20 standard AAs, alphabetical
AA_INDEX = {aa: i + 1 for i, aa in enumerate(AA_LIST)}  # A=1..Y=20, 0=reference
INDEX_AA = {i + 1: aa for i, aa in enumerate(AA_LIST)}
INDEX_AA[0] = "-"   # reference / no mutation

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K',
    'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W',
    'CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

def degrade_codon(c):
    return "".join(x if x in "ACGT" else "N" for x in c.upper())

def call_mutations_categorical(seq_spike, ref_spike):
    """
    Returns dict {aa_position (1-indexed): residue_index (1-20)}
    for all positions where this sequence differs from the reference.
    Stops codons and ambiguous codons are skipped.
    """
    mutations = {}
    ref_pos, ref_buf, seq_buf = 0, [], []
    for col in range(len(ref_spike)):
        r = ref_spike[col]
        s = seq_spike[col] if col < len(seq_spike) else "-"
        if r == "-":
            continue
        ref_buf.append(r); seq_buf.append(s); ref_pos += 1
        if ref_pos % 3 == 0:
            codon_idx = ref_pos // 3
            rc = degrade_codon("".join(ref_buf))
            sc = degrade_codon("".join(seq_buf))
            ref_buf, seq_buf = [], []
            if "N" in rc or "N" in sc or "-" in sc:
                continue
            ref_aa = CODON_TABLE.get(rc, "?")
            seq_aa = CODON_TABLE.get(sc, "?")
            if ref_aa in ("?", "*") or seq_aa in ("?", "*"):
                continue
            if ref_aa != seq_aa and seq_aa in AA_INDEX:
                mutations[codon_idx] = AA_INDEX[seq_aa]
    return mutations   # {pos: residue_idx}

def read_fasta_subset(fasta_path, allowed_ids):
    current_id, current_seq = None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id and current_id in allowed_ids:
                    yield current_id, "".join(current_seq)
                current_id = line[1:].split("|")[0].strip()
                current_seq = []
            else:
                current_seq.append(line)
    if current_id and current_id in allowed_ids:
        yield current_id, "".join(current_seq)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))

    spike_dir  = ROOT / cfg["paths"]["spike_fasta_dir"]
    sample_dir = ROOT / cfg["paths"]["monthly_sample_dir"]
    matrix_dir = ROOT / cfg["paths"]["matrix_dir"]
    vocab_dir  = ROOT / cfg["paths"]["vocab_dir"]
    matrix_dir.mkdir(parents=True, exist_ok=True)

    min_prev = cfg["mutations"]["min_position_prevalence"]
    max_muts = cfg["mutations"]["max_mutations_per_seq"]

    ref_spike = (vocab_dir / "reference_spike_aligned.txt").read_text().strip()
    print(f"Reference spike (aligned): {len(ref_spike)} columns")

    months = sorted(
        id_file.stem.replace("_ids", "")
        for id_file in sample_dir.glob("*_ids.txt")
        if (spike_dir / f"{id_file.stem.replace('_ids','')}.fasta").exists()
    )
    print(f"Months to process: {len(months)}")

    # ── Pass 1: collect position-level observations ───────────────────────
    print("\nPass 1: collecting per-position residue observations...")

    # position_obs[pos] = {residue_idx: count}
    position_obs = defaultdict(lambda: defaultdict(int))
    total_seqs = 0
    month_seq_muts = {}   # month -> list of dicts {pos: residue_idx}

    for month in tqdm(months, desc="Pass 1"):
        id_file = sample_dir / f"{month}_ids.txt"
        fasta   = spike_dir  / f"{month}.fasta"
        allowed = set(id_file.read_text().splitlines())
        seq_muts_list = []
        for sid, seq in read_fasta_subset(fasta, allowed):
            muts = call_mutations_categorical(seq, ref_spike)
            if len(muts) > max_muts:
                continue
            seq_muts_list.append(muts)
            for pos, res_idx in muts.items():
                position_obs[pos][res_idx] += 1
            total_seqs += 1
        month_seq_muts[month] = seq_muts_list

    print(f"Total sequences: {total_seqs:,}")
    print(f"Total variable positions seen: {len(position_obs)}")

    # Build position vocabulary: positions with enough variation
    threshold = int(min_prev * total_seqs)
    # A position qualifies if total non-reference observations >= threshold
    variable_positions = sorted(
        pos for pos, obs in position_obs.items()
        if sum(obs.values()) >= threshold
    )
    P = len(variable_positions)
    pos_to_col = {pos: col for col, pos in enumerate(variable_positions)}
    print(f"Position vocabulary (>={min_prev*100:.2f}% prevalence): {P} positions")

    # Extract reference AA at each variable position from the reference spike
    def get_ref_aa_at_pos(codon_pos_1indexed):
        start = (codon_pos_1indexed - 1) * 3
        ref_buf = ""
        ref_count = 0
        for c in ref_spike:
            if c != "-":
                ref_count += 1
                if ref_count > start and len(ref_buf) < 3:
                    ref_buf += c
            if len(ref_buf) == 3:
                break
        codon = degrade_codon(ref_buf)
        return CODON_TABLE.get(codon, "?")

    # Write position vocabulary TSV
    vocab_rows = []
    for col, pos in enumerate(variable_positions):
        ref_aa = get_ref_aa_at_pos(pos)
        total_alt = sum(position_obs[pos].values())
        vocab_rows.append({
            "col":     col,
            "aa_pos":  pos,
            "ref_aa":  ref_aa,
            "total_alt_obs": total_alt,
        })
    vocab_df = pd.DataFrame(vocab_rows)
    vocab_df.to_csv(vocab_dir / "position_vocab.tsv", sep="\t", index=False)
    print(f"Wrote: {vocab_dir / 'position_vocab.tsv'}")

    # Print top 20 most variable positions
    print("\nTop 20 most variable positions:")
    top20 = vocab_df.nlargest(20, "total_alt_obs")
    for _, r in top20.iterrows():
        print(f"  Spike AA {r['aa_pos']:4d}  (ref={r['ref_aa']})  "
              f"alt_obs={r['total_alt_obs']:,}")

    # ── Pass 2: encode categorical matrices ───────────────────────────────
    print("\nPass 2: encoding categorical matrices...")
    index_rows = []

    for month in tqdm(months, desc="Pass 2"):
        seq_muts_list = month_seq_muts[month]
        n = len(seq_muts_list)
        if n == 0:
            continue

        # Categorical matrix: (n, P) int8
        # Value 0 = reference at that position
        # Value 1-20 = alternate amino acid index
        mat = np.zeros((n, P), dtype=np.int8)
        for i, muts in enumerate(seq_muts_list):
            for pos, res_idx in muts.items():
                if pos in pos_to_col:
                    mat[i, pos_to_col[pos]] = res_idx

        # Per-position residue frequency matrix: (P, 21) float32
        # posfreq[j][k] = fraction of sequences with residue k at position j
        # k=0: reference, k=1-20: amino acids
        posfreq = np.zeros((P, 21), dtype=np.float32)
        for j in range(P):
            col_vals = mat[:, j]            # (n,) int8, values 0-20
            for k in range(21):
                posfreq[j, k] = (col_vals == k).sum() / n

        np.save(matrix_dir / f"{month}.npy",        mat)
        np.save(matrix_dir / f"{month}_posfreq.npy", posfreq)

        mean_muts = (mat > 0).sum(axis=1).mean()
        index_rows.append({
            "month":       month,
            "n_sequences": n,
            "mean_muts":   round(float(mean_muts), 2),
            "P_variable":  P,
        })

    pd.DataFrame(index_rows).to_csv(
        matrix_dir / "index.tsv", sep="\t", index=False
    )
    print(f"Wrote index: {matrix_dir / 'index.tsv'}")
    pd.DataFrame(index_rows).pipe(print)


if __name__ == "__main__":
    main()
