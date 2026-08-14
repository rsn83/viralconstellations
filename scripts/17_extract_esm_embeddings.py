"""
scripts/17_extract_esm_embeddings.py

Run ONCE, on a GPU (built for Colab T4 free tier), before any training
run. Produces a cache mapping each DISTINCT constellation observed
anywhere in 2020-2025 to a per-node ESM2 embedding, extracted from a
real reconstructed full spike sequence.

Why per-constellation, not per-raw-sequence: occupied.pkl already
deduplicates millions of raw sequences down to a much smaller number of
distinct mutation sets. One ESM forward pass per distinct constellation
gives you embeddings for ALL of that constellation's mutated nodes at
once (ESM outputs a hidden state per residue position in a single
pass) -- so the number of expensive GPU calls is bounded by the number
of distinct constellations, not by raw sequence count or by N nodes.

Output: outputs/esm_cache.pkl containing
  { "embeddings": {frozenset(node_indices): {node_idx: np.float32[esm_dim]}},
    "esm_dim": int }

Cache is keyed by the constellation itself (a frozenset), NOT by a
positional index into a sorted list. This matters: an earlier version
keyed by list-position, which silently breaks on resume if the
constellation set or --min_count filter ever changes between runs
(sort order shifts -> old cached ids point at the wrong constellation).
Keying by content is stable regardless of filtering choices across runs.

RESUME SUPPORT: free Colab sessions can disconnect. If esm_cache.pkl
already exists, already-computed constellations are skipped and the
cache is saved incrementally (every --save_every batches), so a
disconnect only costs you the current batch, not the whole run. Safe
to change --min_count between resumed runs now.

SPEED NOTES:
  - --batch_size 8 was conservative to guard against a since-fixed bug
    (wrong reference file producing 37,896-residue "sequences"). With
    the correct ~1273-residue spike sequence, T4 (15GB) can go much
    higher -- try 32 or 64 first.
  - --fp16 roughly halves memory and speeds up inference; use it to
    push batch size further.
  - --min_count filters out rare one-off constellations. The script
    prints the count distribution before filtering so you can pick a
    sensible threshold instead of guessing -- e.g. if the bottom 80%
    of constellations have count=1-2, --min_count 3 could cut runtime
    substantially with little information loss (each dropped
    constellation's nodes just fall back to whatever OTHER carriers
    they have -- see esm_embeddings.py miss handling).

ASSUMPTIONS TO VERIFY:
  - posres_vocab.tsv position/residue columns match --position_col/--residue_col.
  - --reference_path is a translated PROTEIN FASTA (~1273 aa), not the
    nucleotide genome.
  - Sequences are alignment-based (fixed length, no indels breaking the
    position numbering).
  - occupied.pkl constellation keys are iterables of node indices.

Usage:
  python scripts/17_extract_esm_embeddings.py \
      --model facebook/esm2_t30_150M_UR50D --batch_size 32 --fp16 \
      --position_col aa_pos --residue_col residue \
      --reference_path data/raw/spike_reference.fasta
"""

import sys, argparse, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="facebook/esm2_t30_150M_UR50D",
                     help="Free T4 (~15GB): esm2_t12_35M_UR50D or esm2_t30_150M_UR50D. "
                          "Only step up to t33_650M if you have more GPU memory.")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--fp16", action="store_true", default=False,
                     help="half-precision inference -- roughly 2x faster, half the "
                          "memory, negligible accuracy impact for this use case")
parser.add_argument("--min_count", type=int, default=1,
                     help="skip constellations that never reach this total sequence "
                          "count across all months -- reduces cache size/runtime by "
                          "dropping extremely rare one-off constellations. The count "
                          "distribution is printed before filtering to help you choose.")
parser.add_argument("--save_every", type=int, default=50,
                     help="save cache to disk every N batches (resume safety)")
parser.add_argument("--position_col", type=str, default="position")
parser.add_argument("--residue_col", type=str, default="residue")
parser.add_argument("--reference_path", type=str, default="data/raw/spike_reference.fasta",
                     help="path (relative to repo root) to the spike PROTEIN reference "
                          "FASTA -- NOT the nucleotide genome. aa_pos in your vocab is "
                          "protein-residue numbering (~1273 aa), so this must be a "
                          "translated protein sequence, e.g. NCBI YP_009724390.1.")
args = parser.parse_args()

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def log(msg):
    print(msg, flush=True)


def load_reference_sequence(path: Path) -> str:
    text = path.read_text().strip()
    if text.startswith(">"):
        text = "".join(text.split("\n")[1:])
    return text.replace("\n", "").replace(" ", "").upper()


def load_all_occupied(graphs_dir: Path, months: list[str]):
    """Union of distinct constellations across ALL months, with total
    count summed across months (for --min_count filtering; per-month
    weighting at train time uses each month's own count, not this sum)."""
    global_counts: dict[frozenset, int] = {}
    for m in months:
        with open(graphs_dir / f"{m}_occupied.pkl", "rb") as fh:
            occ = pickle.load(fh)
        for c, v in occ.items():
            key = frozenset(c)
            count = v if isinstance(v, (int, float)) else 1
            global_counts[key] = global_counts.get(key, 0) + int(count)
    return global_counts


def log_count_distribution(global_counts: dict):
    counts = np.array(list(global_counts.values()))
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(counts, percentiles)
    log("Constellation count distribution (total sequences carrying each, summed across months):")
    for p, v in zip(percentiles, vals):
        log(f"    p{p}: {v:.0f}")
    for thresh in [1, 2, 3, 5, 10]:
        n_kept = int((counts >= thresh).sum())
        log(f"    --min_count {thresh} would keep {n_kept}/{len(counts)} "
            f"({100*n_kept/len(counts):.1f}%)")


def reconstruct_sequence(constellation: frozenset, reference_seq: str,
                          vocab_df: pd.DataFrame, position_col: str, residue_col: str) -> str:
    seq = list(reference_seq)
    for node_idx in constellation:
        row = vocab_df.iloc[node_idx]
        pos = int(row[position_col])
        res = str(row[residue_col]).upper()
        if 1 <= pos <= len(seq):
            seq[pos - 1] = res
    return "".join(seq)


def main():
    graphs_dir = ROOT / "data" / "processed" / "full_data_graphs_posres"
    index_df = pd.read_csv(graphs_dir / "index.tsv", sep="\t")
    months = sorted(index_df["month"].tolist())
    vocab_df = pd.read_csv(graphs_dir / "posres_vocab.tsv", sep="\t")
    N = len(vocab_df)

    ref_path = ROOT / args.reference_path
    reference_seq = load_reference_sequence(ref_path)
    log(f"Reference length: {len(reference_seq)}, N nodes: {N}")

    max_pos = int(vocab_df[args.position_col].max())
    if len(reference_seq) < max_pos:
        log(f"WARNING: reference length ({len(reference_seq)}) is shorter than the max "
            f"{args.position_col} in your vocab ({max_pos}). Likely the wrong reference "
            f"file -- double check {ref_path} before proceeding.")

    log("Building union of distinct constellations across all months...")
    global_counts = load_all_occupied(graphs_dir, months)
    log(f"Total distinct constellations across all months: {len(global_counts)}")
    log_count_distribution(global_counts)

    if args.min_count > 1:
        global_counts = {c: v for c, v in global_counts.items() if v >= args.min_count}
        log(f"After min_count={args.min_count} filter: {len(global_counts)}")

    constellation_list = sorted(global_counts.keys(), key=lambda c: -global_counts[c])

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    cache_path = out_dir / "esm_cache.pkl"

    embeddings: dict[frozenset, dict[int, np.ndarray]] = {}
    esm_dim = None
    if cache_path.exists():
        log(f"Found existing cache at {cache_path}, resuming...")
        with open(cache_path, "rb") as fh:
            prev = pickle.load(fh)
        embeddings = prev.get("embeddings", {})
        esm_dim = prev.get("esm_dim")
        log(f"  {len(embeddings)} constellations already cached, skipping those "
            f"(cache is keyed by constellation content, so this is safe even if "
            f"--min_count changed since the last run)")

    todo = [c for c in constellation_list if c not in embeddings]
    log(f"{len(todo)} constellations remaining to embed")

    if not todo:
        log("Nothing to do -- cache already complete.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading {args.model} on {device} (fp16={args.fp16})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    if args.fp16 and device.type == "cuda":
        model = model.half()

    def save_cache():
        with open(cache_path, "wb") as fh:
            pickle.dump({"embeddings": embeddings, "esm_dim": esm_dim}, fh)
        log(f"  saved cache: {len(embeddings)}/{len(constellation_list)} constellations")

    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    for b_idx in range(n_batches):
        batch_constellations = todo[b_idx * args.batch_size:(b_idx + 1) * args.batch_size]
        batch_seqs = [
            reconstruct_sequence(c, reference_seq, vocab_df, args.position_col, args.residue_col)
            for c in batch_constellations
        ]

        with torch.no_grad():
            enc = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)
            out = model(**enc)
            hidden = out.last_hidden_state  # (batch, seq_len_padded, esm_dim)
            if esm_dim is None:
                esm_dim = hidden.shape[-1]

            special_mask = enc["special_tokens_mask"] if "special_tokens_mask" in enc else None
            for k, c in enumerate(batch_constellations):
                if special_mask is not None:
                    non_special = (special_mask[k] == 0).nonzero(as_tuple=True)[0]
                    offset = int(non_special[0].item())
                else:
                    offset = 1  # standard ESM2: single [CLS] at index 0
                node_embs = {}
                for node_idx in c:
                    pos = int(vocab_df.iloc[node_idx][args.position_col])  # 1-indexed
                    hidden_idx = offset + (pos - 1)
                    if hidden_idx < hidden.shape[1]:
                        node_embs[node_idx] = hidden[k, hidden_idx].float().cpu().numpy().astype(np.float32)
                embeddings[c] = node_embs

        if (b_idx + 1) % args.save_every == 0 or (b_idx + 1) == n_batches:
            save_cache()
            log(f"  batch {b_idx+1}/{n_batches}")

    save_cache()
    log("Done.")


if __name__ == "__main__":
    main()
