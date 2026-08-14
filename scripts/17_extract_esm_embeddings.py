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
  { "constellation_list": [frozenset(node_indices), ...],
    "embeddings": {constellation_id: {node_idx: np.float32[esm_dim]}},
    "esm_dim": int }

RESUME SUPPORT: free Colab sessions can disconnect. If esm_cache.pkl
already exists, already-computed constellation ids are skipped and the
cache is saved incrementally (every --save_every batches), so a
disconnect only costs you the current batch, not the whole run.

ASSUMPTIONS TO VERIFY:
  - posres_vocab.tsv has columns 'position' (1-indexed) and 'residue'.
  - reference_txt is either a plain sequence or single-record FASTA.
  - Sequences are alignment-based (fixed length, no indels breaking the
    position numbering) -- consistent with aligned.fasta.zst already
    in your pipeline.
  - occupied.pkl constellation keys are iterables of node indices
    (frozenset/tuple/etc) -- converted to frozenset here for a stable
    cache key.

Usage (in Colab, after `pip install transformers` and cloning the repo,
data mounted from Drive):
  python scripts/17_extract_esm_embeddings.py \
      --model facebook/esm2_t30_150M_UR50D --batch_size 8
"""

import sys, argparse, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="facebook/esm2_t30_150M_UR50D",
                     help="Free T4 (~15GB): esm2_t12_35M_UR50D or esm2_t30_150M_UR50D. "
                          "Only step up to t33_650M if you have more GPU memory.")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--min_count", type=int, default=1,
                     help="skip constellations that never reach this total sequence "
                          "count across all months -- reduces cache size/runtime by "
                          "dropping extremely rare one-off constellations")
parser.add_argument("--save_every", type=int, default=50,
                     help="save cache to disk every N batches (resume safety)")
parser.add_argument("--position_col", type=str, default="position")
parser.add_argument("--residue_col", type=str, default="residue")
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
    count summed across months (for --min_count filtering and for
    weighting, though the per-month weighting used later at train time
    uses each month's own count, not this global sum)."""
    global_counts: dict[frozenset, int] = {}
    for m in months:
        with open(graphs_dir / f"{m}_occupied.pkl", "rb") as fh:
            occ = pickle.load(fh)
        for c, v in occ.items():
            key = frozenset(c)
            count = v if isinstance(v, (int, float)) else 1
            global_counts[key] = global_counts.get(key, 0) + int(count)
    return global_counts


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

    ref_path = ROOT / "data" / "raw" / "reference_txt"
    reference_seq = load_reference_sequence(ref_path)
    log(f"Reference length: {len(reference_seq)}, N nodes: {N}")

    log("Building union of distinct constellations across all months...")
    global_counts = load_all_occupied(graphs_dir, months)
    log(f"Total distinct constellations across all months: {len(global_counts)}")

    if args.min_count > 1:
        global_counts = {c: v for c, v in global_counts.items() if v >= args.min_count}
        log(f"After min_count={args.min_count} filter: {len(global_counts)}")

    constellation_list = sorted(global_counts.keys(), key=lambda c: -global_counts[c])
    constellation_to_id = {c: i for i, c in enumerate(constellation_list)}

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    cache_path = out_dir / "esm_cache.pkl"

    embeddings: dict[int, dict[int, np.ndarray]] = {}
    esm_dim = None
    if cache_path.exists():
        log(f"Found existing cache at {cache_path}, resuming...")
        with open(cache_path, "rb") as fh:
            prev = pickle.load(fh)
        embeddings = prev.get("embeddings", {})
        esm_dim = prev.get("esm_dim")
        log(f"  {len(embeddings)} constellations already cached, skipping those")

    todo = [cid for cid in range(len(constellation_list)) if cid not in embeddings]
    log(f"{len(todo)} constellations remaining to embed")

    if not todo:
        log("Nothing to do -- cache already complete.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    def save_cache():
        with open(cache_path, "wb") as fh:
            pickle.dump({
                "constellation_list": constellation_list,
                "embeddings": embeddings,
                "esm_dim": esm_dim,
            }, fh)
        log(f"  saved cache: {len(embeddings)}/{len(constellation_list)} constellations")

    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    for b_idx in range(n_batches):
        batch_ids = todo[b_idx * args.batch_size:(b_idx + 1) * args.batch_size]
        batch_seqs, batch_constellations = [], []
        for cid in batch_ids:
            c = constellation_list[cid]
            seq = reconstruct_sequence(c, reference_seq, vocab_df, args.position_col, args.residue_col)
            batch_seqs.append(seq)
            batch_constellations.append(c)

        with torch.no_grad():
            enc = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)
            out = model(**enc)
            hidden = out.last_hidden_state  # (batch, seq_len_padded, esm_dim)
            if esm_dim is None:
                esm_dim = hidden.shape[-1]

            special_mask = enc["special_tokens_mask"] if "special_tokens_mask" in enc else None
            for k, cid in enumerate(batch_ids):
                c = batch_constellations[k]
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
                        node_embs[node_idx] = hidden[k, hidden_idx].cpu().numpy().astype(np.float32)
                embeddings[cid] = node_embs

        if (b_idx + 1) % args.save_every == 0 or (b_idx + 1) == n_batches:
            save_cache()
            log(f"  batch {b_idx+1}/{n_batches}")

    save_cache()
    log("Done.")


if __name__ == "__main__":
    main()
