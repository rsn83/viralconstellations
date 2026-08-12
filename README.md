# Viral Mutation Constellations

Learning structure over viral mutation sets from population-level sequencing data.

## Scientific question
Does the internal co-occurrence structure of mutations in the currently
circulating population predict which new mutation combinations will appear
in the next time window — beyond what a frequency-independence baseline predicts?

## Repository layout

```
viralconstellations/
├── data/
│   ├── raw/                  ← place your 3 input files here
│   │   ├── aligned.fasta.zst
│   │   ├── metadata.tsv.zst
│   │   └── reference_txt
│   ├── processed/
│   │   ├── spike_fasta/      ← per-month spike-region FASTA (Step 1 output)
│   │   ├── monthly_samples/  ← sampled sequence ID lists (Step 2 output)
│   │   └── mutation_matrices/← binary .npy arrays per month (Step 3 output)
│   └── vocab/
│       └── mutation_vocab.txt← global mutation vocabulary (Step 3 output)
├── src/viralconstellations/
│   ├── data/                 ← reusable data utilities
│   ├── model/                ← model components
│   └── eval/                 ← evaluation metrics
├── scripts/
│   ├── 01_find_spike_coords.py   ← find spike columns from reference
│   ├── 02_extract_spike.py       ← slice spike region, join metadata dates
│   ├── 03_sample_monthly.py      ← stratified monthly sampling
│   ├── 04_build_matrices.py      ← binary mutation matrices
│   ├── 05_train.py               ← train the model
│   └── 06_evaluate.py            ← run evaluation metrics
├── configs/
│   └── default.yaml              ← all hyperparameters in one place
└── tests/
    └── test_data_pipeline.py
```

## Quickstart

```bash
pip install -e .

# Place your 3 files in data/raw/ then run in order:
python scripts/01_find_spike_coords.py
python scripts/02_extract_spike.py
python scripts/03_sample_monthly.py
python scripts/04_build_matrices.py
python scripts/05_train.py
python scripts/06_evaluate.py
```

## Data notes

- `aligned.fasta.zst`: sequences already aligned to Wuhan reference, one sequence per
  entry. Headers contain strain IDs only — **no dates**. Dates come from metadata.
- `metadata.tsv.zst`: Nextstrain-format TSV with columns including `strain`, `date`,
  `country`, `region`. Joined to FASTA on strain ID.
- `reference_txt`: Wuhan-Hu-1 (NC_045512.2) sequence used as alignment reference.
  Spike gene occupies positions 21563–25384 (1-indexed).

## Sampling rationale

We sample 10,000 sequences per month, stratified by country (capped per country).
This is enough to get stable frequency estimates for mutations with prevalence ≥ 0.5%
(rule of thumb: need ~200 positive examples for a stable proportion estimate).
See `configs/default.yaml` for tunable parameters.
