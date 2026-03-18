# Logit-Shift Steering Validation Results

This note reflects an offline re-analysis of the existing sweep files in this
directory. No NDIF traces were re-run. The changes are limited to analysis and
plotting logic applied to:

- `layer_sweep.jsonl`
- `main_sweep_layer60.jsonl`
- `random_sweep_layer60.jsonl`

## What Was Fixed

1. **Specificity now uses steerability slopes, not alpha-averaged raw propensities.**
   The original cross-concept matrix averaged propensities over
   `[-8, -4, -2, 0, 2, 4, 8]`. Because the sweep is symmetric around zero, that
   can cancel directional effects and leave mostly token-set baseline
   sensitivity. The corrected specificity matrix now uses the same quantity as
   the main evaluation: slope of propensity vs `alpha`.

2. **The old stripe pattern is still preserved, but relabeled correctly.**
   New files `figures/mean_propensity_matrix_*` show the alpha-averaged matrix as
   a descriptive artifact. They should not be interpreted as the main
   specificity test.

3. **The entropy-clean analysis now removes all flagged rows.**
   Previously, only the first 50 flagged samples were excluded because the
   analysis reused a preview list rather than the full set of entropy outliers.

4. **Token-set quality is now reported at the analyzed layer (`60`).**
   The earlier report used a representative layer (`40`), which understated the
   overlap between projection and cosine token sets for the actual experiment.

5. **A small-alpha robustness summary was added.**
   This uses only `alpha in {-2, 0, 2}` to check whether the effect survives
   outside the most saturated part of the sweep.

6. **Baseline-token bias is now diagnosed explicitly.**
   The current saved sweep still uses the legacy low-index baseline token set,
   but the analysis now measures where that baseline sits inside the full-vocab
   logit distribution. A text-like, neutral-logit-quantile sampler is the
   recommended next code change instead of low tokenizer ids.

## Main Findings

### 1. The vectors are strongly steerable in logit space

- `50/50` concepts are significant after BH correction for both injection modes.
- All-positions summary:
  - mean steerability: `1.669`
  - min / max: `0.780` / `2.660`
  - concepts with any nonpositive prompt slope: `0`
- Last-token summary:
  - mean steerability: `1.710`
  - min / max: `0.822` / `2.628`
  - concepts with any nonpositive prompt slope: `0`
- Top concepts are stable across injections: `memories`, `mirrors`, `bags`,
  `sugar`, `blood` / `satellites`.
- Lowest-steerability concepts are also stable: `phones`, `information`,
  `masquerades`, `youths`, `oceans`.

### 2. The effect survives in a narrower, less saturated window

Using only `alpha in {-2, 0, 2}`:

- All-positions:
  - mean steerability: `3.886`
  - `50/50` significant
  - concepts with any nonpositive prompt slope: `0`
- Last-token:
  - mean steerability: `3.911`
  - `50/50` significant
  - concepts with any nonpositive prompt slope: `1`

This shows the directional effect is not created solely by the extreme
`|alpha| = 4, 8` endpoints.

### 3. Layer 60 is best for direct readout control

- Layer sweep still peaks at `60` for both injection modes.
- This should be interpreted narrowly: injecting closer to the output produces
  larger final-logit effects.
- It should **not** be interpreted as “layer 60 is where the concept lives” or
  “layer 60 is necessarily optimal for Experiment 1 / introspection.”

### 4. All-positions and last-token are nearly interchangeable here

- Pearson correlation between per-concept steerability under the two injection
  modes: `r = 0.978`.
- At this late layer, direct perturbation of the final position dominates.

### 5. Real vectors are far above the stored random control

- `50/50` concepts exceed the random 95th percentile under both injection
  conditions.
- This rules out a pure norm-only perturbation story.
- Caveat: the control is asymmetric. Real vectors are evaluated on token sets
  built from themselves, while random vectors are only evaluated on the
  concept-defined token sets. That makes this comparison supportive, but not a
  complete semantic control.

### 6. Cross-concept specificity is strong when computed correctly

Corrected slope-based specificity:

- All-positions:
  - mean diagonal: `1.669`
  - mean off-diagonal: `-0.010`
  - paired t-test: `p = 2.07e-35`
- Last-token:
  - mean diagonal: `1.710`
  - mean off-diagonal: `-0.007`
  - paired t-test: `p = 5.96e-36`

This means the raw sweep **does** contain strong cross-concept diagonal
dominance when specificity is defined in the same way as the main metric:
change in target propensity as `alpha` varies.

The previous conclusion that “no logit-level specificity survives correction” was
an artifact of averaging over symmetric positive and negative alphas.

### 7. The old vertical stripes are real, but they mean something else

- The new `figures/mean_propensity_matrix_raw_*` and
  `figures/mean_propensity_matrix_column_corrected_*` still show the column /
  token-set sensitivity structure.
- Those plots describe which token sets sit high or low on average under broad
  perturbation.
- They are **not** the right plots for testing directional specificity.

### 8. Prompt set looks usable, but not perfect

- Baseline prompt logit distributions are not degenerate:
  - pairwise cosine similarity mean: `0.785`
  - min / max: `0.560` / `0.940`
- `2` concept token sets are anomalous at `alpha = 0` by the saved z-score rule.

### 9. Abstract vs concrete remains inconclusive

- The split is `12` abstract vs `38` concrete, not roughly balanced.
- All-positions Mann-Whitney `p = 0.0997`
- Last-token Mann-Whitney `p = 0.0812`
- Treat this as a weak trend only.

### 10. Entropy remains an important caveat

- The entropy correlation remains negative, so higher steerability is not simply
  “more distribution spreading.”
- But the sweep is still aggressive:
  - all-positions: `78.5%` of steered rows exceed the `3x` entropy ratio rule
  - last-token: `77.0%`
- Because many baseline entropies are tiny, this ratio-based flag is itself
  harsh. The full-range slope should be read as a global monotonicity summary,
  not a local derivative.

### 11. The legacy baseline token set is biased low

For the saved run, the baseline set is numerically far below the center of the
vocabulary logit distribution:

- mean baseline logit: `-0.280`
- mean full-vocab logit: `2.184`
- mean full-vocab median logit: `2.552`
- mean percentile of the baseline mean inside the vocab distribution: `14.6%`

This means the legacy baseline contributes a strong positive constant offset to
absolute propensity values.

Implication:

- **Steerability slopes remain meaningful**, because the same baseline is used at
  every `alpha`, so the constant offset cancels.
- **Cross-concept slope-based specificity also remains meaningful** for the same
  reason.
- **Absolute propensity magnitudes** and especially alpha-averaged
  `mean_propensity_matrix_*` plots are harder to interpret and should be treated
  more cautiously.

## Interpretation

### What this experiment now supports

- These vectors are **real late-layer control directions** in logit space.
- They are not interchangeable with isotropic random directions.
- Their effects are prompt-stable and highly consistent across both injection
  modes.
- Cross-concept diagonal structure is present in the raw sweep once specificity
  is computed correctly.

### What it still does **not** prove cleanly

- It does not independently prove concept-specific semantics, because the main
  token sets are defined from the same vector being evaluated.
- The primary readout is therefore partly circular:
  `T_c` consists of tokens that `v_c` already scores highly under the
  unembedding map.
- The current experiment validates output control more strongly than it
  validates semantic interpretation.
- The current saved run also uses a baseline token set that is too low-logit,
  so avoid overinterpreting absolute propensity levels across concepts.

## Useful Caveat From The Token-Set Proxy

The updated `figures/cosine_token_propensity.png` gives a linear, no-new-traces
proxy using cosine-defined token sets that are independent of the injected
vector:

- mean projection/cosine token overlap at layer 60: `0.311`
- projection vs cosine expected propensity Spearman `rho = 0.568`
- same-sign in `45/50` concepts
- median cosine/projection ratio: `0.642`

This suggests some signal likely survives under an independent token-set
definition, but weaker than under the self-defined projection token sets.

## Updated Figure Guide

- `figures/specificity_matrix_raw_*.png`
  - corrected slope-based specificity matrix
- `figures/specificity_matrix_normalized_*.png`
  - row-normalized slope-based specificity
- `figures/specificity_matrix_column_corrected_*.png`
  - column-corrected slope-based specificity
- `figures/mean_propensity_matrix_raw_*.png`
  - alpha-averaged mean propensity matrix; descriptive only
- `figures/mean_propensity_matrix_column_corrected_*.png`
  - column-corrected version of the descriptive mean-propensity matrix
- `figures/entropy_steerability_correlation.png`
  - entropy confound check
- `figures/cosine_token_propensity.png`
  - independent-token-set proxy using only saved vectors and `unembed.pt`

## Recommended Reading Of The Experiment

The cleanest summary is:

> The vectors are clearly steerable, clearly non-random, and clearly impose
> prompt-stable directional structure on the logits. But because the main token
> readout is partly self-defined, this experiment should be read as a validation
> of **output control directions**, not as a stand-alone proof of
> concept-specific semantic representation.
