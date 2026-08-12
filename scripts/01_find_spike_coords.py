"""
Step 1: Find spike gene alignment columns from the reference sequence.

The aligned FASTA aligns every sequence (including the reference) to
the same coordinate system. Gaps (-) in the reference mark insertions
in other sequences relative to Wuhan. To extract the spike region, we
find which *alignment columns* correspond to reference positions 21563-25384.

This script reads the reference from data/raw/reference_txt, finds those
columns, and writes them to data/vocab/spike_coords.txt for use by Step 2.

Why spike only?
  - Spike is the primary target of immune selection, so most adaptive
    mutations occur here.
  - Spike = 3,822 nt (vs ~29,903 nt full genome), reducing sequence
    length ~8x and making the vocabulary of mutations much smaller.
  - Most published constellation analyses (Alpha, Delta, Omicron etc.)
    are defined by spike mutations.

Outputs:
  data/vocab/spike_coords.txt   -- two integers: start_col end_col (0-indexed)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

def main():
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    ref_path   = ROOT / cfg["paths"]["raw_dir"] / "reference_txt"
    vocab_dir  = ROOT / cfg["paths"]["vocab_dir"]
    vocab_dir.mkdir(parents=True, exist_ok=True)

    spike_start_nt = cfg["spike"]["start_nt"]   # 21563, 1-indexed
    spike_end_nt   = cfg["spike"]["end_nt"]     # 25384, 1-indexed

    # Read reference — may or may not have a FASTA header
    lines = ref_path.read_text().splitlines()
    ref_seq = "".join(l.strip() for l in lines if not l.startswith(">"))
    print(f"Reference length (alignment columns): {len(ref_seq)}")

    # Walk the alignment, counting reference positions (skip gap columns)
    start_col = None
    end_col   = None
    ref_pos   = 0  # 1-indexed reference position (gaps don't count)

    for col, base in enumerate(ref_seq):
        if base == "-":
            continue   # insertion in other sequences relative to reference; skip
        ref_pos += 1
        if ref_pos == spike_start_nt:
            start_col = col
        if ref_pos == spike_end_nt:
            end_col = col
            break

    if start_col is None or end_col is None:
        raise ValueError(
            f"Could not find spike coords. Reference length = {len(ref_seq)}, "
            f"last ref_pos reached = {ref_pos}. "
            f"Check that reference_txt covers positions {spike_start_nt}-{spike_end_nt}."
        )

    # Inclusive alignment column range
    n_cols = end_col - start_col + 1
    print(f"Spike gene: reference positions {spike_start_nt}-{spike_end_nt}")
    print(f"Alignment columns (0-indexed): {start_col} - {end_col}  ({n_cols} columns)")

    # Also extract the reference spike subsequence as a sanity check
    ref_spike = ref_seq[start_col : end_col + 1]
    ref_spike_no_gaps = ref_spike.replace("-", "")
    expected_len = spike_end_nt - spike_start_nt + 1
    print(f"Reference spike bases (no gaps): {len(ref_spike_no_gaps)} "
          f"(expected {expected_len})")

    if len(ref_spike_no_gaps) != expected_len:
        print("WARNING: length mismatch — check spike coordinates vs reference.")

    # Write coords
    coords_path = vocab_dir / "spike_coords.txt"
    coords_path.write_text(f"{start_col}\t{end_col}\n")
    print(f"\nWrote: {coords_path}")

    # Also write the reference spike (with alignment gaps) for use in mutation calling
    ref_spike_path = vocab_dir / "reference_spike_aligned.txt"
    ref_spike_path.write_text(ref_spike + "\n")
    print(f"Wrote: {ref_spike_path}")


if __name__ == "__main__":
    main()
