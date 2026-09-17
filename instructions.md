# Discover a Generative Materials Geometry with Blind Functional Utility — No Hyperparameter Tuning

## Scientific task

Using the supplied JARVIS `dft_3d` catalog, learn **one structural model of material state** that simultaneously:

1. induces a material geometry substantially different from ordinary physicochemical proximity;
2. retrieves blind functional analogs from regions that ordinary non-quantized distance regards as far apart;
3. generates complete material-state vectors from an empty state;
4. conditionally reconstructs hidden material properties from partial observations.

The geometry and generator must arise from the **same frozen fitted structural model**.

The model family is unrestricted, but this is a **no-hyperparameter-tuning challenge**. The only allowed comparative selection is a prespecified finite choice of quantization/discretization resolution, if quantization is used.

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

`jid` is an identifier only and may not be used as a feature, embedding input, distance component, random-seed source, lookup key, or pair-specific feature.

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

No outer-held-out row may influence preprocessing, feature construction, model choice, model settings, geometry, generation, quantization selection, or any later methodological change.

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

### Hard no-hyperparameter-tuning rule

A submission must use **one prespecified structural-model configuration**.

The only quantity that may be selected by comparing alternatives is the **quantization/discretization resolution**, if quantization is used. Its finite candidate set and endpoint-independent selection rule must be declared before comparison and may use only the fixed inner fit/development split.

No model, architecture, geometry, representation, regularization, complexity, feature weighting, sampler setting, threshold, bandwidth, latent dimension, component count, tree/ensemble size, or analogous setting may be selected by comparing alternatives on any data.

A solver may choose one fixed value for such quantities. Parameters learned internally by that one fixed algorithm are allowed. If quantization is not used, **no hyperparameter tuning of any kind is allowed**.

Any forbidden tuning gives:

```text
analysis_status = IMPLEMENTATION_FAILURE
no_model_tuning_pass = NO
```

`analysis/selection_trace.csv` must contain exactly one structural configuration, except that multiple rows are allowed only when they differ solely in quantization/discretization resolution.

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

Means are atomic-fraction weighted; ranges are max-minus-min over elements present; and

\[
H_{comp}=-\sum_e x_e\log x_e.
\]

Raw `formula` may not be used as an identity token, embedding, element one-hot vector, or lookup key.

`spg_number` may be used to construct deterministic target-independent crystallographic/group-theoretic descriptors or as a categorical comparator variable, but not as an ordinary continuous geometry coordinate.

Missing-like categorical values (`""`, `"na"`, `"n/a"`, `"nan"`, `"none"`, `"null"`, `"--"`, `"missing"`) are missing states. Numerical missing values may not be silently replaced by physical zero.

---

## Blind functional endpoints

These eight variables are completely forbidden during model construction and methodological selection:

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

They may first be read only after the structural model, geometry, quantization rule, sampler, evaluation code, and all fixed settings have been frozen and hashed.

---

## Single-model requirement

Define one fitted model \(M\) that provides both

\[
d_G(a,b;M)
\]

and

\[
X\sim P_M(X\mid X_O=x_O),
\]

including unconditional generation \(X\sim P_M(X)\).

A geometry model combined with an independently fitted imputation/generation model is an implementation failure.

Write `analysis/model_usage.json` listing every learned artifact and whether it is used by `geometry`, `generation`, or `both`.

---

## Protocol lock and one-shot outer evaluation

Before reading any outer-held-out row or blind endpoint, create:

```text
analysis/selection_trace.csv
analysis/model_manifest.json
analysis/model_usage.json
analysis/model_lock.json
analysis/protocol_lock.json
analysis/model_artifacts/*
```

`protocol_lock.json` must contain SHA-256 hashes for all submitted source files, locks, manifests, selection records, and fitted model artifacts, and must record:

```text
model_family
geometry_family
fixed_model_settings
quantization_candidates
quantization_selection_rule
selected_quantization_resolution
all evaluation seeds
```

Once outer evaluation begins, the model and protocol are immutable. Only one scientific outer evaluation is allowed.

Write:

```text
analysis/outer_evaluation_history.jsonl
analysis/outer_evaluation_receipt.json
```

with at least:

```text
outer_evaluation_count = 1
protocol_lock_sha256
model_lock_sha256
answers_sha256
```

A syntax/runtime bug may be fixed before outer data are read. Methodological changes after outer evaluation begins are prohibited.

---

## Submitted geometry and conventional comparators

The frozen model must induce one finite symmetric dissimilarity \(d_G(a,b)\ge0\) satisfying

\[
d_G(a,a)=0
\]

within `1e-12` and

\[
|d_G(a,b)-d_G(b,a)|\le10^{-10}.
\]

Construct standardized non-quantized Euclidean distance \(d_E\) from eligible numerical construction variables and the 12 chemistry summaries. Estimate standard deviations from the 50,000 training rows only:

\[
d_E(a,b)=\sqrt{\frac{1}{|O_{ab}|}\sum_{j\in O_{ab}}\left(\frac{x_{aj}-x_{bj}}{\sigma_j}\right)^2}.
\]

Require at least 10 jointly observed numerical coordinates.

Also compute a mixed Gower-style comparator \(d_M\) including `crys`, `dimensionality`, and `spg_number` as categorical variables. Mixed-distance and rank-correlation results are required diagnostics but are not strong-effect gates.

---

## Fixed large geometry/retrieval cohort

After protocol locking:

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

Use exactly these 3,000 materials:

\[
\binom{3000}{2}=4,498,500
\]

unordered pairs.

---

# TEST 1 — geometry novelty

For each anchor \(a\), compute its 20 nearest neighbors under \(d_G,d_E,d_M\). Define

\[
J^E_{20}(a)=\frac{|N^G_{20}(a)\cap N^E_{20}(a)|}{20}.
\]

Also report the corresponding mixed overlap and global/anchor-wise Spearman correlations.

Use 3,000 anchor-bootstrap replicates with seed `101002`.

## Strong Gate 1 — nontrivial geometric reorganization

Require

\[
\boxed{CI^{upper}_{95}(\bar J^E_{20})<0.30}.
\]

**Why 0.30?** In the raw material space, dropping 10% of coordinates preserves about 84% of 20-neighbor structure, dropping 20% preserves about 74%, and even dropping 50% preserves about 44% on average. Across random 50%-coordinate perturbations, the 5th percentile of mean J20 is approximately `0.294`. Thus `<0.30` requires a reorganization stronger than approximately 95% of geometries produced even after discarding half of the measured raw-property representation.

Set `geometry_novel = YES` iff this gate and geometry validity checks pass.

---

# TEST 2 — blind distant-analog retrieval

Standardize the eight blind transport variables using means and SDs estimated from the 50,000 training rows. Define

\[
d_F(a,b)=\sqrt{\frac18\sum_{k=1}^{8}(\widetilde Y_{ak}-\widetilde Y_{bk})^2}.
\]

Let

\[
e_{0.90}=Q_{0.90}(d_E)
\]

over eligible evaluation pairs. For each anchor define

\[
C_a=\{b:d_E(a,b)\ge e_{0.90}\},
\]

and retrieve

\[
b_G(a)=\arg\min_{b\in C_a}d_G(a,b).
\]

Select 20 distinct controls from \(C_a\), excluding \(b_G(a)\), whose \(d_E\) values are closest to that of the retrieved pair.

Define

\[
R_{retrieval}=\frac{\operatorname{mean}_a d_F(a,b_G(a))}{\operatorname{mean}_a\operatorname{mean}_{c\in C_a^{match}}d_F(a,c)}
\]

and

\[
W=\frac1A\sum_a I\left[d_F(a,b_G(a))<\operatorname{mean}_{c\in C_a^{match}}d_F(a,c)\right].
\]

Use 3,000 anchor-bootstrap replicates with seed `101004`.

### Raw-distance matching validity

Define

\[
G_{match}=\frac{\operatorname{mean}_a\frac1{20}\sum_{c\in C_a^{match}}|d_E(a,c)-d_E(a,b_G(a))|}{e_{0.90}}.
\]

Use 3,000 anchor-bootstrap replicates with seed `101005`. Require both

\[
G_{match}\le0.10
\]

and

\[
CI^{upper}_{95}(G_{match})<0.10.
\]

This is a validity control, not one of the five strong-effect gates.

## Strong Gate 2 — retrieval effect size

Require

\[
\boxed{CI^{upper}_{95}(R_{retrieval})<0.70}.
\]

**Why 0.70?** Random retrieval among raw-distance-matched far materials gives \(R_{null}\approx0.995\), SD about `0.009`, and a 1st percentile around `0.976`; none of 3,000 raw-null experiments reached `0.70`. The gate therefore requires at least a 30% reduction in blind functional distance in a regime where ordinary random variation produces ratios near 1.

## Strong Gate 3 — retrieval consistency

Require

\[
\boxed{CI^{lower}_{95}(W)>0.80}.
\]

**Why 0.80?** Raw-distance-matched random retrieval gives \(W_{null}\approx0.585\), with a 99th percentile around `0.604`; none of 3,000 null experiments reached `0.80`. The gate therefore requires improvement for at least four out of five materials with 95% confidence.

Set `retrieval_useful = YES` iff both strong retrieval gates and raw-distance matching validity pass.

---

## Generative evaluation panel

Using the 50,000 training materials only, include a numerical construction coordinate iff:

```text
finite coverage >= 0.50
distinct observed values >= 3
```

Include `crys` and `dimensionality` iff nonmissing coverage is at least `0.50`. Apply the same numerical rule to the 12 chemistry summaries.

Write the fixed panel to:

```text
analysis/generation_panel.json
```

before outer evaluation. On the supplied dataset the reference protocol yields a 24-coordinate panel.

---

## Independent-marginal baseline

Construct

\[
P_{IND}(X)=\prod_j\widehat P_{train}(X_j)
\]

from the 50,000 training materials. For conditional completion, observed coordinates remain fixed and hidden coordinates are sampled independently from their training marginals. Use seed `314159`.

---

# TEST 3 — unconditional generation

Generate exactly `10,000` samples from \(P_M(X)\) with no material coordinates supplied, using seed `161803`.

Write:

```text
analysis/unconditional_samples.csv.gz
```

Validity requires:

```text
valid complete rows >= 9500 / 10000
unique generated rows >= 80%
exact training-row matches <= 10%
```

Use numerical rounding to `1e-8` for exact-row comparison. Marginal fidelity must be reported but is descriptive.

Set `blank_generation_valid = YES` iff all three conditions pass.

---

# TEST 3B — cross-structure retention

For every generation-panel coordinate \(j\), predict \(X_j\) from all other coordinates.

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

Fit preprocessing statistics and encoders from the real 50,000-row training set. Train three verifier versions on 10,000 rows each:

1. real training materials;
2. submitted generated materials;
3. independent-marginal materials.

Evaluate all three on the same real outer-held-out rows.

For numerical targets use \(L_j=MAE_j/IQR_j^{train}\). For categorical targets use \(L_j=1-accuracy_j\).

A target is informative iff

\[
L_j^{REAL}\le0.95L_j^{IND}.
\]

For each informative target define

\[
g_j=\frac{L_j^{IND}-L_j^{GEN}}{L_j^{IND}-L_j^{REAL}},
\]

without clipping, and

\[
G=\frac1{|J|}\sum_{j\in J}g_j.
\]

Bootstrap outer-held-out verifier rows 2,000 times with seed `424242`, without refitting verifier models.

## Strong Gate 4 — cross-structure retention

Require

\[
\boxed{CI^{lower}_{95}(G)>0.70}.
\]

**Why 0.70?** \(G=0\) corresponds to independent-marginal structure and \(G\approx1\) to real-data-level recoverable structure. Raw-data degradation gives approximately `G=1.00` for a real joint resample, `0.88` after replacing 30% of rows by independent-marginal samples, and `0.80` after replacing 50%. A 0.70 lower-confidence floor is therefore conservative relative to severe deliberate destruction of the empirical joint structure while still requiring preservation of a substantial majority of recoverable cross-variable information.

Set `cross_structure_preserved = YES` iff this gate passes.

---

# TEST 4 — arbitrary partial-state completion

Use exactly **8,000** outer-held-out materials having at least 80% of the generation panel originally observed, sampled with seed `20260916`.

Create independent reproducible masks with seed `27182818` at:

```text
20% hidden
50% hidden
80% hidden
```

For each masked row and masking level generate exactly 10 conditional completions. Supplied observed coordinates must remain unchanged.

For hidden numerical coordinates, use the median of the 10 generated values as the point completion. For hidden categorical coordinates, use the modal state, breaking ties lexicographically.

For numerical coordinate \(j\):

\[
\ell_j=\frac{|\hat x_j-x_j|}{IQR_j^{train}}.
\]

For categorical coordinate \(j\):

\[
\ell_j=I(\hat x_j\ne x_j).
\]

Average first within coordinate across rows and then equally across coordinates. Let \(L_M(q)\) be submitted-model loss and \(L_{IND}(q)\) independent-marginal loss. Define

\[
R_C(q)=\frac{L_M(q)}{L_{IND}(q)}.
\]

Bootstrap completion rows 3,000 times with seed `577215`.

Report all three masking levels and coordinate-wise win fractions.

## Strong Gate 5 — high-information conditional completion

Require

\[
\boxed{CI^{upper}_{95}(R_C(0.20))<0.50}.
\]

**Why 20% hidden?** The reference generation panel contains 24 coordinates, so 20% masking hides approximately five properties. In the raw outer-test population, the 99th percentile of naturally missing panel coordinates is four of 24 (`16.7%`); among the blind-eligible 3,000-material cohort, the 99th percentile is five of 24 (`20.8%`). Thus 20% masking represents an empirically severe, approximately 99th-percentile partial-observation regime rather than an arbitrary masking fraction.

**Why 0.50?** \(R_C<0.50\) requires at least a 50% reduction in reconstruction loss relative to independent-marginal completion: a minimum twofold improvement under an empirically high-missingness regime.

The 50%- and 80%-hidden results are required diagnostics but are not strong-effect gates.

Set `conditional_completion_useful = YES` iff the 20%-hidden gate passes.

---

# Five strong discovery gates

A scientific submission must satisfy all five simultaneously:

\[
\boxed{
\begin{aligned}
CI^{upper}_{95}(\bar J^E_{20}) &< 0.30,\\
CI^{upper}_{95}(R_{retrieval}) &< 0.70,\\
CI^{lower}_{95}(W) &> 0.80,\\
CI^{lower}_{95}(G) &> 0.70,\\
CI^{upper}_{95}(R_C(0.20)) &< 0.50.
\end{aligned}
}
\]

These are **strong-effect**, not merely statistical-significance, requirements. The first three are calibrated against raw-data perturbation/null experiments; the cross-structure floor is conservative relative to deliberate destruction of the observed joint distribution; and the completion test combines a twofold improvement requirement with an approximately 99th-percentile empirical missingness regime.

These gates are fixed before any new solver evaluation and may not be changed in response to solver performance.

---

# Required primary outputs

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
single_shot_outer_evaluation_pass
no_model_tuning_pass
raw_control_matching_pass

evaluation_count
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
mean_abs_control_raw_distance_gap
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
analysis/model_lock.json
analysis/protocol_lock.json
analysis/selection_trace.csv
analysis/outer_evaluation_history.jsonl
analysis/outer_evaluation_receipt.json
```

---

# Failure states

Use exactly one final `analysis_status`.

## IMPLEMENTATION_FAILURE

Use for any protocol violation, including wrong split, forbidden tuning, outer/blind leakage, post-outer methodological revision, more than one scientific outer evaluation, independently fitted geometry/generation models, lock/hash mismatch, invalid/asymmetric distance, altered observed completion values, real-row-seeded unconditional generation, or missing/unreproducible required artifacts.

## GEOMETRY_FAILURE

Use if implementation checks pass but Strong Gate 1 fails.

## RETRIEVAL_FAILURE

Use if geometry passes but either Strong Gate 2, Strong Gate 3, or raw-distance matching validity fails.

## GENERATION_FAILURE

Use if geometry and retrieval pass but any of the following fail:

```text
blank_generation_valid
Strong Gate 4
Strong Gate 5
single_model_requirement_pass
```

## DISCOVERY_SUCCESS

Set

```text
analysis_status = DISCOVERY_SUCCESS
primary_discovery = YES
```

iff all implementation checks, generation validity, raw-control matching, and all five strong discovery gates pass.

Otherwise set:

```text
primary_discovery = NO
```

---

# Reproducibility

A fresh run must recreate every graded output from:

```text
jarvis.tgz
submitted source code
declared package dependencies
```

No precomputed pair labels, external material database, hidden manually curated table, or unpublished artifact may be required. Evaluation randomness must use exactly the seeds specified above. Model-fitting randomness is allowed but must be documented in `model_manifest.json`, and the evaluated fitted realization must be frozen before outer evaluation.
