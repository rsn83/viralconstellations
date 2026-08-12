# Viral Mutation Constellation Forecasting — Codebase Summary

## Scientific Question
Does population-level mutation background structure (co-occurrence, 
trajectory, lattice depth) predict which new mutation constellations 
will appear in future months, beyond a frequency-independence baseline?

A constellation = a frozenset of spike amino acid positions that are 
mutated (non-reference) in a given sequence. The Boolean lattice F(O_t) 
is all constellations reachable by adding one mutation to any currently 
occupied constellation.

---

## Repository Structure

```
viralconstellations/
├── configs/
│   ├── default.yaml                  # Full dataset config
│   ├── colab_2022_test.yaml          # Train 2020-2022, predict 2023
│   └── test.yaml                     # Laptop CPU test (2020 data only)
├── data/
│   ├── raw/                          # aligned.fasta.zst, metadata.tsv.zst
│   ├── processed/mutation_matrices/  # YYYY-MM.npy, YYYY-MM_posfreq.npy
│   └── vocab/position_vocab.tsv      # P=153 variable spike positions
├── scripts/
│   ├── 04_build_matrices_from_metadata.py
│   ├── 05_train.py
│   ├── 06_evaluate.py
│   ├── 07_walk_forward_eval.py
│   ├── 08_train_eval_discriminative.py
│   └── test_pipeline.py
├── src/viralconstellations/
│   ├── data/utils.py
│   ├── eval/metrics.py
│   ├── frontier/frontier.py
│   └── model/
│       ├── model.py
│       ├── trajectory.py
│       └── discriminative.py
└── tests/test_data_pipeline.py
```

---

## Data Pipeline

### `scripts/04_build_matrices_from_metadata.py`
- Input: `data/raw/metadata.tsv.zst` (Nextstrain format, 955MB compressed)
- Join key: `genbank_accession` column
- Mutation source: `aaSubstitutions` column (pre-computed by Nextclade)
- Filters: spike mutations only (prefix `S:`), drops stop codons (*)
- Sampling: 10,000 sequences/month, capped 1,500/country, stratified
- Output per month `YYYY-MM`:
  - `YYYY-MM.npy`: (n_seq, P) int8 categorical matrix
    - 0 = reference residue at that position
    - 1-20 = amino acid index (A=1...Y=20)
  - `YYYY-MM_posfreq.npy`: (P, 21) float32 per-position residue frequencies
- Output: `data/vocab/position_vocab.tsv` — the P=153 variable positions
- Coverage: 74 consecutive months 2020-02 to 2026-03, no gaps

### Data representation
- P = 153 variable spike positions (positions that showed any variation 
  across all 74 months above min_prevalence threshold)
- The vocabulary is FIXED across all months — sequences from 2026 use 
  the same 153 columns as 2020 sequences
- Limitation: positions never variable in training data are invisible to 
  the model

---

## Model Components

### `src/viralconstellations/model/model.py`

#### `MutationSetEncoder` (inner DeepSets φ)
- Input: (B, P) int categorical sequence
- Architecture: factored embeddings concat(pos_emb(j), res_emb(r)) → MLP
- Output: (B, d_model) per-sequence embedding
- Pool: mean over non-reference (mutated) positions only
- Note: DeepSets applied to SEQUENCES — aggregates across mutations within 
  one sequence

#### `PopulationEncoder` (outer DeepSets ρ)
- Input: full monthly matrix (n_seq, P) int8
- encode_population(): applies MutationSetEncoder to all sequences, 
  mean-pools across sequences, applies ρ MLP → (d_model,)
- forward(): applies ρ only to pre-pooled embedding (used at train time)
- Output: c_t — one population embedding per month
- Problem identified: mean-pooling across sequences loses within-sequence 
  co-occurrence structure. The resulting c_t cannot represent which 
  mutations appear TOGETHER in the same sequence.

#### `TrajectoryEmbeddingCache`
- Stores per-month population embeddings c_t and posfreq matrices
- Refreshed once per epoch as encoder weights update
- get_window(month_t, W): returns (W+1, d_model) stacked window embeddings
- get_posfreq(month): returns (P, 21) frequency matrix
- Key design: caches mean(φ(x_i)) before ρ, so ρ gradients flow at 
  training time

#### `LengthToGoHead`
- Input: (B, d_model) hidden state + (B,) horizon h
- Architecture: sinusoidal horizon embedding → concat with hidden → MLP
- Output: (B,) log(predicted mutation count) — NO Softplus activation
- At generation: exp(output) = predicted count
- Loss: Poisson NLL loss (F.poisson_nll_loss with log_input=True)
- Molecular clock: trained to predict that h=6 → more mutations than h=1
- Current status: disabled (length_weight=0.0) — was causing instability
  due to interaction with Poisson NLL

#### `FrequencyRegressionHead`
- Input: (B, d_model) hidden state h_{t+h}
- Architecture: Linear(d_model, d_model*2) → SiLU → Linear → reshape → softmax
- Output: (B, P, 21) predicted per-position residue probability distributions
- Loss: KL divergence (cross-entropy on soft labels) vs empirical posfreq
- Evaluation: Pearson r between predicted and real per-position mutation rates
- Result: r=0.99 at h=1, r=0.84 at h=6 (single context 2022-12)
- Walk-forward mean: r=0.66 at h=1, r=0.78 at h=6 (62 windows)
- NOTE: this is MARGINAL frequency prediction — each position independently
  No joint/co-occurrence information

#### `CooccurrenceRegressionHead` (experimental, currently disabled)
- Input: (B, d_model) hidden state
- Architecture: LOW-RANK factorization
  - net: Linear(d_model, d_model) → SiLU → Linear(d_model, P*rank)
  - reshape to (B, P, rank) factor matrix V
  - co-occurrence = Sigmoid(V @ V.T / sqrt(rank))
  - returns upper triangle: (B, n_pairs) where n_pairs = P*(P-1)//2
- Rank=16: predicts 153×16=2,448 values instead of 11,628
- Loss: weighted BCE, upweights pairs deviating from independence by 
  weight = 1 + alpha * |real_coo - freq_i*freq_j|
- Buffers: iu_row, iu_col — upper triangle indices (must be on same device as h)
- Status: diverges during training (cooc loss goes from 0.7 → 3.3)
- Problem: low-rank matrix cold-start instability + gradient competition 
  with denoising loss

#### `evaluate_cooccurrence(pred_coo, real_coo, indep_coo, P)`
- Computes TWO metrics:
  1. Absolute Pearson r: correlation between predicted and real co-occurrence
     → Independence wins here (r=0.999) because dominant lineage makes 
       independence approximately correct
  2. Residual Pearson r: correlation between predicted DEVIATION from 
     independence and real DEVIATION from independence
     → This is the right scientific metric — positive means model captures 
       structure independence misses
- Result: residual_pearson_r = -0.35 at h=1 — model worse than independence
  even on residuals

#### `ConstellationTransformer` (DILM-M diffusion)
- Input: (B, P) noisy categorical sequence + diffusion step t + horizon h + 
  context h_{t+h}
- Architecture: per-position embeddings (state + position) + sinusoidal 
  conditioning (t, h) + context → TransformerEncoder → Linear(d, 21)
- Forward process (training): reversion noising — each mutated position 
  reverts to reference independently with probability (1-rho(t))
  where rho(t) = 0.5*(1+cos(πt/T)) (cosine schedule)
- Reverse process (generation): T→1 denoising steps
- Conditioning: time_emb(t) + horizon_emb(h) + h_{t+h} broadcast to all 
  P positions
- Loss: cross-entropy per position (label smoothing=0.05)
- Known failure mode: cross-entropy sum over positions is mathematically 
  equivalent to assuming position independence given context. Transformer 
  CAN attend across positions but loss doesn't require it. Model learns 
  marginals, not joints.
- Constellation consistency loss (added, partially effective):
  computes pred_coo = mean_batch(p_mut_i * p_mut_j) from output probabilities,
  compares to real co-occurrence, deviation-weighted MSE
  → collapsed to 0.001 in epoch 2 (trivial solution via independence)

#### `generate_from_hidden(model, length_head, h_state, horizon, ...)`
- Three-step DILM-S generation:
  1. LengthToGoHead → exp(log_pred) = target mutation count k
  2. T→1 denoising → (n_samples, P) sampled residues
  3. Final forward pass → p_mutated per position → keep top-k by p_mutated
- Current status: generates 1-mutation sequences (length head untrained)

#### `independence_cooccurrence(posfreq)` 
- Returns (P, P) matrix where entry [i,j] = freq_i × freq_j
- The baseline every model must beat on co-occurrence prediction

### `src/viralconstellations/model/trajectory.py`

#### `GRUTrajectoryEncoder` (filter step)
- Input: (W+1, d_model) window of monthly embeddings
- Architecture: temporal position embeddings + GRU → projection → LayerNorm
- Output: (d_model,) filtered hidden state h_t
- Corresponds to Kalman filter UPDATE step: incorporate new observation

#### `TransitionModel` (prediction step)
- Architecture: learned step_token + GRUCell + residual + LayerNorm
- step(h): one month forward → h_{t+1}
- forward(h, k): k steps → returns (h_{t+k}, [h_t, h_{t+1}, ..., h_{t+k}])
- Does NOT see horizon h — multi-horizon comes from running step() k times
- Corresponds to Kalman filter PREDICT step: extrapolate without observation

#### `VelocityEncoder` (ablation alternative to GRU+Transition)
- Input: posfreq_t and posfreq_{t-1}
- Encodes one-step velocity only, no forward propagation capability

### `src/viralconstellations/model/discriminative.py`

#### `CandidateEncoder` (DeepSets on candidate constellation)
- Input: (B, P) candidate sequence + (B, d_model) population context h_{t+h}
- Architecture (DeepSets applied to MUTATIONS, not sequences):
  - φ: concat(pos_emb(j), res_emb(r), h_{t+h}) → MLP → mutation embedding
    h_{t+h} fed into φ so each mutation is conditioned on population trajectory
  - Pool: mean over MUTATED positions only (ignore reference positions)
  - ρ: concat(pooled, h_{t+h}) → LayerNorm → SiLU → Linear → emb(c)
- Output: (B, d_model) candidate constellation embedding
- Key difference from PopulationEncoder: aggregates over mutations within 
  one constellation, not over sequences. Preserves set structure of candidate.

#### `FrontierDiscriminator`
- Input: (B, d_model) emb(c) + (B, 7) hand_features
- Architecture: concat → Linear → LayerNorm → SiLU → Linear → SiLU → 
  Linear → Sigmoid
- Output: (B,) predicted P(candidate appears in O_{t+h})

#### `DiscriminativeModel`
- Wraps CandidateEncoder + FrontierDiscriminator
- score_candidates(seqs, hand, h_state, device): scores a batch of candidates
- 33,761 parameters total (tiny)

---

## Frontier Module

### `src/viralconstellations/frontier/frontier.py`

#### `compute_occupied(mat, top_k, min_freq)`
- Input: (n_seq, P) categorical matrix
- Output: {frozenset_of_mutated_positions: frequency} — top_k constellations
- Constellation = frozenset of POSITION INDICES (binary, ignores residue)

#### `compute_frontier(occupied, P)`
- Input: {constellation: frequency} dict
- Output: {candidate_constellation: {parents, parent_freqs, parent_depths, 
  n_parents, max_parent_freq}}
- F(O_t): all constellations = occupied_c ∪ {j} for each unmutated position j
- Excludes already-occupied constellations

#### `compute_new_constellations(mat_t, mat_th, min_freq)`
- Returns (occupied_t, new_in_th) where new_in_th = O_{t+h} \ O_t
- new_in_th = ground truth for what should be predicted

#### `frontier_coverage_benchmark(mat_t, mat_th, P, hamming_r, min_freq)`
- The first experiment: what fraction of new_in_th is in F(O_t)?
- hamming_r=1: strict one-mutation frontier
- hamming_r=2: within two mutations of any occupied constellation
- Results across 62 windows:
  - H=1: mean=54.6% at h=1, 20.2% at h=6
  - H=2: mean=86.8% at h=1, 43.6% at h=6
- Key finding: new constellations typically require 2 simultaneous mutations

#### `candidate_to_sequence(candidate, pred_posfreq, P)`
- Converts frozenset of positions → (P,) int8 categorical sequence
- At each mutated position: uses most probable non-reference residue from 
  pred_posfreq
- Used to convert frontier candidates to input format for models

#### `extract_features(candidate, info, pred_posfreq, prev_posfreq, P)`
- Returns (7,) float32 feature vector:
  1. pred_freq_new_pos: predicted frequency of new position (from FreqHead)
  2. max_parent_freq: frequency of most common parent constellation
  3. log_n_parents: log(1 + number of parent constellations)
  4. mean_parent_depth: mean mutation count of parents
  5. jaccard_best_parent: |c ∩ best_parent| / |c ∪ best_parent|
  6. freq_trend_new_pos: pred_freq - prev_freq at new position
  7. coo_support: mean product of predicted marginals for all pairs in c

#### `score_candidates_neural(model, candidates, pred_posfreq, h_state, horizon, P, device)`
- Scores frontier candidates using ConstellationTransformer log-likelihood
- For each candidate: convert to sequence, run model at t=1 (near-clean),
  compute sum of log P(correct residue at each position)
- Batched for efficiency

#### `LogisticFrontierScorer`
- collect(mat_t, mat_th, pred_posfreq, prev_posfreq, P): builds (X, y) training data
- fit(X, y): StandardScaler + LogisticRegression(class_weight='balanced')
- score(mat_t, pred_posfreq, prev_posfreq, P): returns ranked [(constellation, prob)]
- feature_importances(): returns {feature_name: coefficient}
- Walk-forward AP results: 0.027 at h=1 vs random 0.004 (7× improvement)

#### `evaluate_ranking(ranked, new_in_th, top_ks)`
- Computes precision@k, recall@k, AP, random_baseline_P
- Used to evaluate all three scorers uniformly

---

## Training Scripts

### `scripts/05_train.py`
- Usage: `python scripts/05_train.py --config configs/colab_2022_test.yaml`
- Training triple: (month_t, mat_{t+h}, h) — fundamental training unit
- With 35 training months and max_h=6: 189 triples
- Per-triple forward pass:
  1. GRU filter: window → h_t
  2. TransitionModel × h: h_t → h_{t+1},...,h_{t+h}
  3. Freq loss at each intermediate step (multi-step supervision)
  4. Denoising loss: corrupt sequences from mat_th → ConstellationTransformer
  5. Constellation consistency loss: pred_coo vs real_coo from model output
  6. Co-occurrence head loss (currently disabled, cooc_weight=0.0)
  7. Length head loss (currently disabled, length_weight=0.0)
- Active losses: denoise×1.0, freq×1.0, consistency×5.0
- Prints every epoch with flush=True (Colab compatible)
- Saves best checkpoint by validation loss

### Constellation consistency loss (in 05_train.py)
- After denoising forward pass: logits → probs → p_mut → pred_coo
- pred_coo[i,j] = mean_batch(p_mut[i] × p_mut[j])
- Deviation-weighted MSE: weight = 1 + alpha × |real_coo - freq_i×freq_j|
- Problem: collapsed to 0.001 in epoch 2 — trivial solution (predict 
  marginals, independence ≈ real for dominant lineage)
- Not learning cross-position dependencies

---

## Evaluation Scripts

### `scripts/06_evaluate.py`
- Usage: `python scripts/06_evaluate.py --config configs/colab_2022_test.yaml`
- Single context month evaluation (last training month)
- Per horizon h=1,2,3,6:
  - Frequency prediction: pred_posfreq vs real_posfreq, Pearson r
  - Co-occurrence: absolute r + residual r (model vs independence)
  - Frontier coverage H=1 and H=2
  - Neural frontier scoring: AP, precision@k
  - Generative evaluation: generate n_gen sequences, compare co-occ to baseline
- Key result: freq Pearson r=0.991 at h=1

### `scripts/07_walk_forward_eval.py`
- Usage: `python scripts/07_walk_forward_eval.py --config configs/colab_2022_test.yaml`
- 62 evaluation windows across all available months
- Per window × per horizon:
  - Frequency Pearson r
  - Frontier coverage H=1 and H=2
  - Three scorers: random, logistic, neural (ConstellationTransformer)
  - Progressive logistic scorer: refitted every 6 windows
- Key results across 62 windows:
  - Freq r: mean=0.66±0.23 at h=1, 0.78±0.14 at h=6
  - Frontier H=1: mean=54.6% at h=1, 20.2% at h=6
  - Frontier H=2: mean=86.8% at h=1, 43.6% at h=6
  - Logistic AP: 0.027 at h=1 (7× over random 0.004)
  - Neural AP (ConstellationTransformer): 0.0044 at h=1 (barely beats random)

### `scripts/08_train_eval_discriminative.py`
- Usage: `python scripts/08_train_eval_discriminative.py --config configs/colab_2022_test.yaml`
- Trains discriminative model on top of FROZEN trajectory encoder
- Walk-forward: train on first N-20 windows, evaluate on last 20
- build_window_examples(): for each window builds (seqs, hand_features, labels)
- train_discriminative(): weighted BCE (pos_weight = n_neg/n_pos)
- eval_window(): compares random vs logistic vs neural discriminative
- Reports AP table: neural vs logistic vs random per horizon

---

## Evaluation Metrics

### `src/viralconstellations/eval/metrics.py`

#### `all_metrics_categorical(generated, real, top_k, mmd_n_sub, hamming_r)`
- pos_frequency_correlation: Pearson r on per-position mutation rates
- pairwise_cooccurrence: Pearson r on top_k-site pairwise co-occurrence
- independence_baseline_coo: same metric for independence-sampled sequences
- mmd_hamming: MMD with RBF kernel on binary mutation matrices
- frontier_coverage: fraction of real sequences within hamming_r of generated
- mean_mut_count: mean mutations per sequence (model vs real)
- NaN-safe: returns warning dict if generated sequences are all zeros

---

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| Freq Pearson r h=1 (single) | 0.991 | Context 2022-12 |
| Freq Pearson r h=1 (walk-fwd) | 0.66±0.23 | 62 windows |
| Frontier H=1 coverage h=1 | 54.6%±18% | New constellations in F(O_t) |
| Frontier H=2 coverage h=1 | 86.8%±9% | Within 2 mutations |
| Logistic AP h=1 | 0.027 | vs random 0.004 |
| Neural AP h=1 (ConsTrans) | 0.0044 | Barely > random |
| Neural AP h=1 (discriminative) | TBD | Script 08 running |
| Co-occ residual r h=1 | -0.35 | Model worse than independence |

---

## Known Limitations and Failures

1. **Fixed vocabulary**: P=153 positions fixed at preprocessing time. Novel 
   positions not in training data are invisible to the model.

2. **DeepSets misapplied**: PopulationEncoder aggregates across sequences, 
   losing within-sequence co-occurrence. CandidateEncoder (discriminative.py) 
   correctly aggregates across mutations within one constellation.

3. **Diffusion learns marginals not joints**: ConstellationTransformer trained 
   with per-position cross-entropy → learns marginal frequencies. Consistency 
   loss added but collapses to trivial solution. Correct fix: block masking or 
   contrastive objective.

4. **Independence baseline dominates co-occurrence**: In Omicron-dominated 
   population, freq_i × freq_j ≈ real co-occurrence for 95% of pairs. 
   Residual structure is real but small signal.

5. **Length head unstable**: Poisson NLL with log(count) output still produces 
   -25 to -27 loss values (optimal is ~-70). Currently disabled. Generated 
   sequences have mean 1 mutation instead of ~32.

6. **Frontier H=1 coverage 55%**: Only 55% of new constellations are one 
   mutation away from current population. Model cannot predict the other 45% 
   even in principle.

7. **Generative metrics NaN**: CooccurrenceRegressionHead produces NaN 
   (diverged during training). All pairwise_coo_r metrics from generative 
   model are NaN.

---

## Config Parameters (colab_2022_test.yaml)

```yaml
model:
  d_model: 128        # hidden dimension throughout
  n_heads: 4          # transformer attention heads
  n_layers: 3         # transformer layers
  diffusion_T: 50     # diffusion steps
  phi_hidden: 256     # DeepSets inner MLP hidden size
  deepsets_batch_size: 512

trajectory:
  window_size: 3      # W — GRU reads last W+1 months
  mode: "gru"         # alternative: "velocity"
  gru_hidden: 128

horizon:
  max_h: 6            # maximum prediction horizon
  denoising_weight: 1.0
  length_weight: 0.0  # DISABLED
  
cooc_head:
  cooc_weight: 0.0    # DISABLED
  cooc_rank: 16       # low-rank factorization rank

consistency:
  weight: 5.0         # constellation consistency loss weight
  alpha: 3.0          # deviation upweighting

freq_head:
  freq_weight: 1.0
  intermediate_weight: 0.5  # weight for intermediate step supervision

train:
  train_months: []    # auto: all before test_month
  test_month: "2023-01"
  n_epochs: 30
  lr: 0.001
  batch_size: 256
```

---

## Suggested Improvements for Next Tool

Based on experiments:

1. **Replace diffusion training objective**: Use contrastive loss — real 
   sequences score higher than independence-sampled sequences with same 
   marginals. Forces model to learn joints.

2. **Block masking**: Instead of independent per-position reversion, mask 
   entire co-occurring groups simultaneously. Groups = positions with 
   co-occurrence > 0.8 in training data.

3. **Discriminative model on H=2 frontier**: Current discriminative model 
   only scores H=1 candidates. Extend to H=2 to cover the 87% of new 
   constellations within Hamming-2.

4. **Add fitness features**: Deep mutational scanning data (Bloom lab), 
   immune escape scores (Escape Calculator), ACE2 binding affinity. These 
   capture what actually determines which new constellation establishes.

5. **Simplify trajectory encoder**: Replace DeepSets population encoder with 
   direct posfreq matrix input. Proven to work equally well, much simpler.

6. **Ranking loss instead of classification**: Pairwise ranking loss 
   (margin ranking or ListNet) on frontier candidates. Easier to learn than 
   binary classification with 100:1 imbalance.

7. **Fix length head**: Use log-normal distribution instead of Poisson. 
   Mutation counts have heavier tail than Poisson assumes.
