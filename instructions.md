# Discover a Generative Materials Geometry with Blind Functional Utility with No Hyperparameter Tuning

## Scientific task

Using the supplied JARVIS `dft_3d` catalog, learn **one structural model of material state** that simultaneously:

1. induces a material geometry substantially different from ordinary physicochemical proximity;
2. retrieves blind functional analogs from regions that ordinary non-quantized distance regards as far apart;
3. generates distinct complete material-state vectors from an empty state ie no properties specified which must be validated in an objective sense to be physically probabale (not just marginals but the entire vector of values of the physical properties); and
4. conditionally reconstructs hidden material properties from partial observations.

The same frozen fitted model must provide both geometry and generation. The model family and implementation stack are unrestricted.

The defining scientific constraint is **no hyperparameter tuning**. 

---

## Input and required command

Run from the repository root.

Input:

```text
./jarvis.tgz
```

containing:

```text
jarvis_dft3d.pkl
```

All graded outputs must be written under:

```text
./analysis/
```

The complete workflow must run from a fresh checkout using:

```bash
python analysis/run_all.py --data jarvis.tgz
```

---

## Authoritative dataframe and fixed outer split

Define row identity exactly as:

```python
df = df.drop_duplicates(subset="jid", keep="first").reset_index(drop=True)
```

`jid` is an identifier only. It may not be used as a model feature, embedding input, distance component, seed source, lookup key, or pair-specific feature.

Use exactly:

```python
rng = np.random.default_rng(1)
train_pos = rng.choice(len(df), size=50000, replace=False)
train_mask = np.zeros(len(df), dtype=bool)
train_mask[train_pos] = True
train_jids = df.loc[train_pos, "jid"].to_numpy()
test_jids = df.loc[~train_mask, "jid"].to_numpy()
```

This gives:

```text
outer training: 50,000 materials
outer held out: 43,902 materials
```

No outer-held-out row may influence model fitting or model selection.

---

## Fixed inner development split

Use exactly:

```python
inner_rng = np.random.default_rng(271828)
inner_dev_idx = inner_rng.choice(len(train_pos), size=10000, replace=False)
inner_dev_mask = np.zeros(len(train_pos), dtype=bool)
inner_dev_mask[inner_dev_idx] = True
fit_pos = train_pos[~inner_dev_mask]
inner_dev_pos = train_pos[inner_dev_mask]
```

Thus:

```text
inner fit:         40,000
inner development: 10,000
```

---

## Precisely bounded no-hyperparameter-tuning rule

A submission must use **one prespecified structural-model configuration**.

The only quantity that may be selected by comparing alternatives is the **quantization/discretization resolution**, if quantization is used. Its finite candidate set and endpoint-independent selection rule must be declared before comparison and may use only the fixed inner fit/development split.

Allowed:

- parameters and structure learned internally by the one chosen algorithm;
- deterministic preprocessing fixed before comparative evaluation;
- one fixed value for architecture, regularization, geometry, representation, sampler, capacity, thresholds, bandwidths, etc., provided competing values are not empirically compared;
- implementation/unit debugging that does not compare scientific performance across alternative configurations;
- comparative selection only among the declared quantization resolutions.

Not allowed:

- comparing alternative model families, geometry families, representations, latent dimensions/ranks, regularizers, feature-weighting schemes, component/tree/ensemble counts, sampler settings, thresholds, bandwidths, or analogous settings and selecting among them by empirical performance.

`analysis/selection_trace.csv` must contain exactly one non-quantization structural configuration. Multiple rows are allowed only when they differ solely in quantization/discretization resolution.

The verifier grades only observable submitted code/artifacts/selection records. It does not infer private experimentation or grade a reasoning transcript.

---

## Allowed material-state information

The structural model may use only these measured-property columns when present:

```text
crys
dimensionality
nat
density
formation_energy_peratom
ehull
exfoliation_energy
optb88vdw_bandgap
mbj_bandgap
hse_gap
spillage
magmom_oszicar
magmom_outcar
epsx
epsy
epsz
mepsx
mepsy
mepsz
avg_elec_mass
avg_hole_mass
bulk_modulus_kv
shear_modulus_gv
poisson
dfpt_piezo_max_eij
dfpt_piezo_max_dij
dfpt_piezo_max_dielectric
dfpt_piezo_max_dielectric_electronic
dfpt_piezo_max_dielectric_ionic
max_ir_mode
min_ir_mode
Tc_supercon
```

Chemistry may additionally be represented only by these deterministic summaries of `formula`:

```text
chem_n_elements
chem_composition_entropy
chem_mean_atomic_number
chem_atomic_number_range
chem_mean_electronegativity
chem_electronegativity_range
chem_mean_atomic_radius
chem_atomic_radius_range
chem_mean_group
chem_group_range
chem_mean_period
chem_period_range
```

Means are atomic-fraction weighted; ranges are max-minus-min over elements present; composition entropy is

$$
H_{comp}=-\sum_e x_e\log x_e.
$$

Raw `formula` may not be used as an identity token, learned embedding, element one-hot vector, or lookup key.

`spg_number` may be used only as a categorical comparator variable or to construct deterministic target-independent crystallographic descriptors; it may not be treated as an ordinary continuous geometry coordinate.

Missing-like categorical values (`""`, `"na"`, `"n/a"`, `"nan"`, `"none"`, `"null"`, `"--"`, `"missing"`) are missing states. Numerical missing values may not be silently replaced by physical zero.

---

## Blind functional endpoints

These eight variables are completely forbidden during model construction and selection:

```text
n-Seebeck
p-Seebeck
n-powerfact
p-powerfact
ncond
pcond
nkappa
pkappa
```

They are used only during the final held-out retrieval evaluation after the fitted structural model is frozen.

---

## Mechanical single-model requirement

The workflow must produce one learned artifact bundle `analysis/model_artifacts/*`, denoted $M$.

Conceptually the submission implements:

```text
fit(training data, allowed inner-development information) -> M
distance(M, a, b) -> d_G(a,b)
sample(M, observed coordinates, mask, seed) -> material-state sample
```

Both `distance` and `sample` must consume the same frozen learned artifact bundle. Deterministic read-only transformations of that bundle are allowed. No second learned model, separately fitted parameters, or post-hoc learned mapping specialized for only geometry or only generation may be fitted after $M$ is frozen.

Write `analysis/model_usage.json` identifying the artifact bundle and the code paths used by geometry and generation. Write `analysis/model_manifest.json` recording model family, fixed settings, fitting seed(s), and selected quantization resolution if applicable.

---

## Final-evaluation boundary

The final outer evaluation begins when the submitted evaluation stage first reads an outer-held-out material or blind-endpoint value.

Before that point, the fitted artifact bundle, fixed non-quantization settings, any permitted quantization choice, and evaluation code must be fixed. A syntax/runtime error may be repaired before this point. Once final evaluation begins, the scientific configuration is not revised and reevaluated.

No branch-history audit, append-only history, or hidden-development reconstruction is required. The final grader runs the submitted workflow once with fixed evaluation seeds.

---

## Submitted geometry and conventional comparators

The frozen model must induce one finite symmetric dissimilarity $d_G(a,b)\ge0$ satisfying

$$
d_G(a,a)=0
$$

within `1e-12`, and

$$
|d_G(a,b)-d_G(b,a)|\le10^{-10}.
$$

Construct standardized non-quantized Euclidean distance $d_E$ from eligible numerical construction variables and the 12 chemistry summaries. Estimate coordinate SDs from the 50,000 training rows only:

$$
d_E(a,b)=\sqrt{\frac{1}{|O_{ab}|}\sum_{j\in O_{ab}}\left(\frac{x_{aj}-x_{bj}}{\sigma_j}\right)^2}.
$$

Require at least 10 jointly observed numerical coordinates.

Also compute a mixed Gower-style comparator $d_M$ including `crys`, `dimensionality`, and `spg_number` as categorical variables. Mixed-distance and rank-correlation results are diagnostics, not strong-effect gates.

For all evaluation nearest-neighbor, `argmin`, and matched-control ties, break ties by smaller deduplicated dataframe position.

---

## Fixed large geometry/retrieval cohort

During final evaluation:

```python
blind_cols = [
    "n-Seebeck", "p-Seebeck",
    "n-powerfact", "p-powerfact",
    "ncond", "pcond",
    "nkappa", "pkappa",
]

blind_numeric = df[blind_cols].apply(pd.to_numeric, errors="coerce")
eligible_pos = np.flatnonzero((~train_mask) & blind_numeric.notna().all(axis=1).to_numpy())
eval_rng = np.random.default_rng(20260915)
eval_pos = eval_rng.choice(eligible_pos, size=3000, replace=False)
```

Use exactly these 3,000 materials, giving

$$
\binom{3000}{2}=4,498,500
$$

unordered pairs.

---

# TEST 1 — geometry novelty

For each anchor $a$, compute its 20 nearest neighbors under $d_G,d_E,d_M$. Define

$$
J^E_{20}(a)=\frac{|N^G_{20}(a)\cap N^E_{20}(a)|}{20}.
$$

Report the corresponding mixed overlap and global/anchor-wise Spearman correlations as diagnostics.

Use 3,000 anchor-bootstrap replicates with seed `101002`.

## Strong Gate 1

Require

$$
\boxed{CI^{upper}_{95}(\bar J^E_{20})<0.33}.
$$

Interpretation: at the upper confidence bound, no more than about one-third of learned neighbors may coincide with ordinary Euclidean neighbors, so at least about two-thirds are reorganized. Raw-feature perturbation supports this as a strong effect: dropping 10%, 20%, and 50% of raw coordinates preserves mean J20 of about 0.84, 0.74, and 0.44 respectively; the 5th percentile across 50%-feature removals is about 0.294.

---

# TEST 2 — blind distant-analog retrieval

Standardize the eight blind endpoints using means and SDs estimated from the 50,000 training rows. Define

$$
d_F(a,b)=\sqrt{\frac18\sum_{k=1}^{8}(\widetilde Y_{ak}-\widetilde Y_{bk})^2}.
$$

Let

$$
e_{0.90}=Q_{0.90}(d_E)
$$

over eligible evaluation pairs. For each anchor define

$$
C_a=\{b:d_E(a,b)\ge e_{0.90}\}
$$

and retrieve

$$
b_G(a)=\arg\min_{b\in C_a}d_G(a,b).
$$

Select 20 distinct controls from $C_a$, excluding $b_G(a)$, whose $d_E$ values are closest to the retrieved pair's $d_E$.

Define

$$
R_{retrieval}=\frac{\operatorname{mean}_a d_F(a,b_G(a))}{\operatorname{mean}_a\operatorname{mean}_{c\in C_a^{match}}d_F(a,c)}
$$

and

$$
W=\frac1A\sum_a I\left[d_F(a,b_G(a))<\operatorname{mean}_{c\in C_a^{match}}d_F(a,c)\right].
$$

Use 3,000 anchor-bootstrap replicates with seed `101004`.

### Raw-distance matching validity

Define

$$
G_{match}=\frac{\operatorname{mean}_a\frac1{20}\sum_{c\in C_a^{match}}|d_E(a,c)-d_E(a,b_G(a))|}{e_{0.90}}.
$$

Use 3,000 anchor-bootstrap replicates with seed `101005`. Require both

$$
G_{match}\le0.10
$$

and

$$
CI^{upper}_{95}(G_{match})<0.10.
$$

### Strong Gate 2

Require

$$
\boxed{CI^{upper}_{95}(R_{retrieval})<0.70}.
$$

Raw-distance-matched random retrieval gives $R_{null}\approx0.995$ with SD about 0.009 and 1st percentile about 0.976; none of 3,000 null experiments reached 0.70. The gate requires at least a 30% functional-distance reduction.

### Strong Gate 3

Require

$$
\boxed{CI^{lower}_{95}(W)>0.80}.
$$

The corresponding raw-null win rate is about 0.585 with 99th percentile about 0.604; none of 3,000 null experiments reached 0.80. The gate therefore requires improvement for at least about four out of five anchors with 95% confidence.

---

## Generative evaluation panel

Using the 50,000 training materials only, include a numerical construction coordinate iff:

```text
finite coverage >= 0.50
distinct observed values >= 3
```

Include `crys` and `dimensionality` iff nonmissing coverage is at least `0.50`. Apply the same numerical rule to the 12 chemistry summaries.

Write `analysis/generation_panel.json` before final evaluation. On the supplied dataset the reference protocol yields a 24-coordinate panel.

---

## Independent-marginal baseline

Construct

$$
P_{IND}(X)=\prod_j\widehat P_{train}(X_j)
$$

from the 50,000 training materials. For conditional completion, observed coordinates remain fixed and hidden coordinates are sampled independently from training marginals. Use seed `314159`.

---

# TEST 3 — unconditional generation

Generate exactly 10,000 samples from $P_M(X)$ with no observed material coordinates, using seed `161803`.

Write `analysis/unconditional_samples.csv.gz`.

Validity requires exactly:

```text
valid complete rows >= 9500 / 10000
unique generated rows >= 80%
exact training-row matches <= 10%
```

Use numerical rounding to `1e-8` for uniqueness and exact-row comparison. Marginal fidelity is reported but descriptive.

---

# TEST 3B — cross-structure retention

For every generation-panel coordinate $j$, predict $X_j$ from all other coordinates.

For numerical targets use exactly:

```text
ExtraTreesRegressor
n_estimators = 100
min_samples_leaf = 3
max_features = "sqrt"
random_state = 99173
```

For categorical targets use exactly:

```text
ExtraTreesClassifier
n_estimators = 100
min_samples_leaf = 3
max_features = "sqrt"
random_state = 99173
```

Fit preprocessing statistics and encoders from the real 50,000-row training set. Train verifier versions on 10,000 rows each from:

```text
REAL: real training materials
GEN:  submitted generated materials
IND:  independent-marginal materials
```

Evaluate all three on the same real outer-held-out rows eligible for that target.

For numerical targets:

$$
L_j=MAE_j/IQR_j^{train}.
$$

For categorical targets:

$$
L_j=1-accuracy_j.
$$

A target is informative iff

$$
L_j^{REAL}\le0.95L_j^{IND}.
$$

For each informative target define

$$
g_j=\frac{L_j^{IND}-L_j^{GEN}}{L_j^{IND}-L_j^{REAL}},
$$

without clipping, and

$$
G=\frac1{|J|}\sum_{j\in J}g_j.
$$

Bootstrap outer-held-out verifier rows 2,000 times with seed `424242`, without refitting verifier models.

## Strong Gate 4

Require

$$
\boxed{CI^{lower}_{95}(G)>0.70}.
$$

Calibration: a real joint resample gives $G\approx1.00$; replacing 30% and 50% of rows with independent-marginal samples gives approximately 0.88 and 0.80. The 0.70 floor is therefore conservative relative to severe deliberate destruction of joint structure.

---

# TEST 4 — partial-state completion

Use exactly **8,000** outer-held-out materials having at least 80% of the generation panel originally observed, sampled without replacement using seed `20260916`.

For each row and each masking level, let `obs` be the list of originally observed panel coordinates and hide exactly

```python
k = int(round(hidden_fraction * len(obs)))
```

coordinates selected from `np.random.default_rng(27182818).permutation(len(obs))` in the sequential row/level order of the evaluation loop.

Evaluate:

```text
20% hidden
50% hidden
80% hidden
```

For every row and masking level generate exactly 10 conditional completions. Supplied observed coordinates must remain unchanged.

For a hidden numerical coordinate, use the median of the 10 generated values. For a hidden categorical coordinate, use the mode, breaking ties lexicographically.

For numerical coordinate $j$:

$$
\ell_j=\frac{|\hat x_j-x_j|}{IQR_j^{train}}.
$$

If training IQR is zero, use the positive training range; if that is also zero, use 1.0.

For categorical coordinate $j$:

$$
\ell_j=I(\hat x_j\ne x_j).
$$

Average first within coordinate across rows and then equally across coordinates. Let $L_M(q)$ and $L_{IND}(q)$ be submitted-model and independent-marginal loss. Define

$$
R_C(q)=\frac{L_M(q)}{L_{IND}(q)}.
$$

Use 3,000 row-bootstrap replicates with seed `577215`. Use the same bootstrap row-index arrays for model and baseline.

Report all three masking levels and coordinate-wise win fractions.

## Strong Gate 5

Require

$$
\boxed{CI^{upper}_{95}(R_C(0.20))<0.50}.
$$

Why 20%: on a 24-coordinate panel this hides about five properties, approximately the 99th-percentile natural missingness level in the blind-eligible cohort (5/24 = 20.8%).

Why 0.50: this requires at least a twofold improvement over independent-marginal completion. Raw-space nearest-neighbor completion yields ratios around 0.276, 0.284, and 0.329 for 1-, 5-, and 20-NN respectively, showing that the raw material-state distribution contains sufficient real conditional redundancy for a sub-0.50 effect to be feasible. These kNN values are calibration only and are not additional gates.

The 50%- and 80%-hidden results are required diagnostics, not strong gates.

---

# Five strong discovery gates

All five must pass simultaneously:

$$
\boxed{
\begin{aligned}
CI^{upper}_{95}(\bar J^E_{20}) &< 0.33,\\
CI^{upper}_{95}(R_{retrieval}) &< 0.70,\\
CI^{lower}_{95}(W) &> 0.80,\\
CI^{lower}_{95}(G) &> 0.70,\\
CI^{upper}_{95}(R_C(0.20)) &< 0.50.
\end{aligned}
}
$$

These thresholds are fixed before new solver evaluation and must not be altered in response to solver performance.

---

# Required outputs

Write `analysis/answers.json` containing at least:

```text
analysis_status
primary_discovery
geometry_novel
retrieval_useful
blank_generation_valid
cross_structure_preserved
conditional_completion_useful
single_model_requirement_pass
no_model_tuning_pass
raw_control_matching_pass

pair_count

mean_J20_euclidean
mean_J20_euclidean_ci95_lower
mean_J20_euclidean_ci95_upper
mean_J20_mixed
mean_J20_mixed_ci95_lower
mean_J20_mixed_ci95_upper
global_rho_geometry_euclidean
global_rho_geometry_mixed

n_evaluable_anchors
retrieval_ratio
retrieval_ratio_ci95_lower
retrieval_ratio_ci95_upper
retrieval_win_rate
retrieval_win_rate_ci95_lower
retrieval_win_rate_ci95_upper
raw_control_matching_gap_ratio
raw_control_matching_gap_ratio_ci95_lower
raw_control_matching_gap_ratio_ci95_upper
raw_far_threshold_q90

generation_panel_size
unconditional_requested_count
unconditional_valid_count
unconditional_unique_fraction
unconditional_exact_training_match_fraction

n_cross_structure_targets
cross_structure_retention_G
cross_structure_retention_G_ci95_lower
cross_structure_retention_G_ci95_upper
cross_structure_target_win_fraction

completion_ratio_hidden20
completion_ratio_hidden20_ci95_lower
completion_ratio_hidden20_ci95_upper
completion_ratio_hidden50
completion_ratio_hidden50_ci95_lower
completion_ratio_hidden50_ci95_upper
completion_ratio_hidden80
completion_ratio_hidden80_ci95_lower
completion_ratio_hidden80_ci95_upper
completion_coordinate_win_fraction_hidden20
completion_coordinate_win_fraction_hidden50
completion_coordinate_win_fraction_hidden80
```

Also write:

```text
analysis/retrieval_by_anchor.csv
analysis/cross_structure_by_target.csv
analysis/completion_by_coordinate.csv
analysis/unconditional_marginal_fidelity.csv
analysis/unconditional_samples.csv.gz
analysis/generation_panel.json
analysis/model_manifest.json
analysis/model_usage.json
analysis/selection_trace.csv
analysis/model_artifacts/*
```

---

# Failure states

Use exactly one final `analysis_status`.

`IMPLEMENTATION_FAILURE` is reserved for mechanically observable invalidity: wrong fixed split; blind endpoints used in fitting/selection code; submitted comparative non-quantization search; geometry and generation using separately fitted learned models; missing required artifacts; invalid/asymmetric distance; or altered supplied coordinates during completion.

`GEOMETRY_FAILURE`: implementation valid but Strong Gate 1 fails.

`RETRIEVAL_FAILURE`: geometry passes but Strong Gate 2, Strong Gate 3, or raw-distance matching validity fails.

`GENERATION_FAILURE`: geometry/retrieval pass but unconditional-generation validity, Strong Gate 4, or Strong Gate 5 fails.

`DISCOVERY_SUCCESS`: all mechanical validity checks, generation validity, raw-control matching, and all five strong gates pass.

Set `primary_discovery = YES` only for `DISCOVERY_SUCCESS`; otherwise `NO`.

---

# Reproducibility

A fresh run must recreate the graded outputs from:

```text
jarvis.tgz
submitted source code
declared package dependencies
the submitted/fitted model artifact created by the workflow
```

Evaluation randomness must use exactly the seeds specified above. If model fitting itself is stochastic, its seed must be declared in `model_manifest.json` and fixed for the evaluated realization.

The grader does not require reproduction of the reference model's numerical values. It evaluates the submitted model against the fixed gates and validity conditions above.
