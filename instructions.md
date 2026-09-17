# Discover a generative materials geometry with blind functional utility and No Tuning of hyperparameters

## Scientific task

Using the supplied JARVIS `dft_3d` materials catalog, learn a **single structural model of material state** that simultaneously:

1. induces a material-to-material geometry substantially different from ordinary physicochemical proximity;
2. retrieves functionally analogous materials from regions that ordinary non-quantized distance regards as far apart;
3. generates complete material-state vectors from no observed material coordinates;
4. conditionally completes held-out material states when arbitrary fractions of their coordinates are hidden.

The model family is not specified. Any scientifically admissible probabilistic, latent-variable, energy-based, autoregressive, conditional, graphical, kernel, ensemble, neural, or other construction may be used. Quantization or discretization is permitted but not required.

The geometry and generator must arise from the **same frozen fitted structural model**. A geometry-only model combined after the fact with an independently fitted generative or imputation model does not solve the problem.

## Hard no-model-tuning rule

This problem tests whether a **single prespecified model configuration** can discover the required structure without model tuning.

A submission may tune **only the quantization/discretization resolution**, if quantization is used. For example, a solver may compare a prespecified finite set such as `q in {5, 10}` using only the fixed inner fit/development split and an endpoint-independent objective. The quantization candidate set and selection rule must be fixed before the comparison is run.

No other model, geometry, architecture, complexity, regularization, or sampler hyperparameter may be selected by comparing alternatives on any data. In particular, the following are forbidden:

- searching over latent dimension, rank, number of mixture components, number of clusters, embedding dimension, network width/depth, tree count, ensemble size, or analogous capacity parameters;
- searching over regularization strengths, priors, penalties, learning rates, temperatures, bandwidths, kernel parameters, thresholds, significance levels, subset-selection settings, or analogous model hyperparameters;
- comparing multiple model families or multiple geometry families and selecting among them using fit, inner-development, outer-held-out, or blind-endpoint performance;
- tuning sampler sweeps, burn-in, proposal parameters, stopping tolerances, convergence thresholds, or other generation/completion settings by comparative performance;
- trying multiple feature-weighting schemes, representations, preprocessing families, or learned distance constructions and selecting among them by performance.

A solver may choose one fixed value for any such quantity, but that value must be part of the single submitted configuration and may not be selected by evaluating competing values. Ordinary fitted parameters learned inside that one fixed algorithm are allowed. Likewise, data-driven structure inferred internally by the fixed algorithm is allowed; for example, learned conditional subsets, fitted mixture weights, fitted regression coefficients, or learned tree structure are model parameters, not tuning, provided no external hyperparameter/model search is performed.

If the model does not use quantization/discretization, **no hyperparameter tuning of any kind is allowed**.

This rule is a primary gate. Write `no_model_tuning_pass = YES` only if the submitted workflow satisfies it. Any non-quantization tuning makes the submission an `IMPLEMENTATION_FAILURE` and prevents `primary_discovery = YES` regardless of all scientific test scores.

The eight external transport variables defined below are used only for blind scientific validation. Their values may not influence model construction, feature selection, architecture selection, quantization selection, model-family selection, geometry definition, generative-model selection, sampler settings, stopping rules, or any subsequent methodological revision.

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

The complete workflow must be executable from a fresh checkout using:

```bash
python analysis/run_all.py --data jarvis.tgz
```

---

## Authoritative dataframe

Load the dataframe and define row identity exactly as follows:

```python
df = df.drop_duplicates(subset="jid", keep="first").reset_index(drop=True)
```

`jid` is an identifier only and may not be used as a model feature, embedding input, distance component, random-seed source, generative conditioning variable, lookup key, or pair-specific feature.

---

## Fixed outer split

Use exactly:

```python
rng = np.random.default_rng(1)
train_pos = rng.choice(len(df), size=50000, replace=False)
train_mask = np.zeros(len(df), dtype=bool)
train_mask[train_pos] = True
train_jids = df.loc[train_pos, "jid"].to_numpy()
test_jids = df.loc[~train_mask, "jid"].to_numpy()
```

The 50,000 selected rows are the complete outer training set. Every other row is outer held-out data.

No outer held-out row may influence preprocessing, representation learning, feature selection, quantization choice, architecture choice, model configuration, model weights, geometry definition, generator definition, sampler settings, stopping rules, or any later methodological change.

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
inner fit:         40,000 materials
inner development: 10,000 materials
```

The inner split may be used to tune **only quantization/discretization resolution**. It may also be used for non-selective diagnostics that do not alter any model, geometry, representation, preprocessing, or sampler choice. It may not be used to select any other hyperparameter, architecture, model family, geometry family, feature-weighting scheme, or sampler setting.

If quantization resolution is selected on the inner split, the selected resolution may then be refit/recomputed using all 50,000 outer-training materials. The one fixed structural model must then be fitted on the permitted training data and frozen before any outer evaluation begins.

---

# Allowed material-state information

## Measured material variables

The structural model may use only the following JARVIS measured-property columns, when present:

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

## Chemistry representation

Chemistry may be represented only by these 12 deterministic summaries derived from `formula`:

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

Means must be atomic-fraction weighted. Ranges are maximum minus minimum among elements present. Composition entropy is

\[
H_{\rm comp}=-\sum_e x_e\log x_e.
\]

Raw `formula` may not be used as an identity token, learned string representation, element one-hot vector, embedding, or lookup key.

## Symmetry

`spg_number` may be used to construct deterministic target-independent crystallographic or group-theoretic descriptors. Raw `spg_number` may not be treated as an ordinary continuous numerical coordinate in the learned geometry. Any derived symmetry representation must be documented.

---

# Missing values

Missing-like categorical values must be interpreted as missing. At minimum, after whitespace stripping and lowercase conversion:

```text
""
"na"
"n/a"
"nan"
"none"
"null"
"--"
"missing"
```

are missing states and may not become legitimate physical categories. Numerical missing values may not be silently replaced by physical zero. Any imputation or missing-state mechanism must be fitted only on outer-training data.

---

# Blind external functional variables

These eight variables are completely forbidden during structural-model construction and methodological selection:

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

They may not be used for training, quantization selection, auxiliary-target selection, representation construction, geometry weighting, model selection, generative-model selection, development diagnostics, or any post-hoc revision.

Their values may first be read only after the final structural model, geometry definition, quantization rule, evaluation code, and all fixed settings have been frozen and their hashes recorded.

---

# Endpoint-agnostic construction

Knowledge of the names of the eight validation variables may not be used to hand-select construction variables because they are believed to be mechanistically related to transport.

Deterministic preprocessing, feature construction, feature filtering, state-space construction, geometry definition, and sampler settings must be specified as part of the one fixed submitted configuration. Data-derived fitted quantities inside that fixed configuration are allowed, but alternative non-quantization configurations may not be compared and selected.

If quantization/discretization resolution is tuned, its objective must be endpoint-independent and use only the fixed inner fit/development data. Permissible objectives include reconstruction quality, held-out likelihood, entropy-based criteria, or another prespecified endpoint-independent quantity.

---

# No-model-tuning gate

Before outer evaluation, write enough information to mechanically verify that no forbidden tuning occurred.

`analysis/selection_trace.csv` must contain exactly one structural-model configuration, except that multiple rows are allowed when and only when they differ solely in quantization/discretization resolution. If multiple quantization resolutions are compared, all other model, geometry, representation, and sampler settings must be identical across those rows.

At minimum, record:

```text
candidate_id
scientific_definition
fit_data_description
development_data_description
quantization_resolution
fixed_model_settings
development_objective
development_result
selected
notes
```

`analysis/model_manifest.json` must distinguish:

```text
fitted_parameters
fixed_model_settings
quantization_resolution
quantization_selection_rule
quantization_candidates
```

If no quantization tuning is performed, `quantization_candidates` must contain at most the single submitted resolution or be empty/null for a non-quantized model.

Set:

```text
no_model_tuning_pass = YES
```

iff all of the following hold:

1. only one model family and one geometry family are implemented/evaluated;
2. all non-quantization hyperparameters/settings are single fixed values rather than candidate grids;
3. any candidate comparison differs only in quantization/discretization resolution;
4. no non-quantization setting is changed in response to fit, inner-development, outer-held-out, or blind-endpoint results;
5. branch history and protocol artifacts are consistent with these constraints.

Otherwise set `no_model_tuning_pass = NO` and `analysis_status = IMPLEMENTATION_FAILURE`.

---

# Single structural-model requirement

The submission must define one fitted structural model \(M\). From this same frozen fitted object it must provide both

\[
d_G(a,b;M)
\]

and

\[
X\sim P_M(X\mid X_O=x_O),
\]

including the unconditional case \(O=\varnothing\):

\[
X\sim P_M(X).
\]

It is not sufficient to fit one learned model for geometry and another independently learned model for generation. Deterministic transformations with no fitted parameters are allowed.

Write:

```text
analysis/model_usage.json
```

listing every learned artifact and whether it is used by `geometry`, `generation`, or `both`. Any learned artifact used only on one side is an implementation failure unless it is a deterministic transformation with no fitted parameters.

---

# Pre-outer protocol lock and one-shot outer evaluation

This is a strict **single-shot outer evaluation** problem.

Before any outer-held-out row or any blind transport value is read, the workflow must create:

```text
analysis/selection_trace.csv
analysis/model_manifest.json
analysis/model_usage.json
analysis/model_lock.json
analysis/protocol_lock.json
analysis/model_artifacts/*
```

`protocol_lock.json` must contain SHA-256 hashes for:

```text
analysis/run_all.py
all analysis/*.py source files used by the run
selection_trace.csv
model_manifest.json
model_usage.json
model_lock.json
every fitted model artifact
```

It must also explicitly record, before outer evaluation:

```text
model_family
geometry_family
fixed_model_settings
quantization_candidates
quantization_selection_rule
selected_quantization_resolution
all evaluation seeds
```

No non-quantization model/geometry/sampler candidate grid is permitted.

The outer evaluation is terminal. Once any outer-held-out row or any blind transport value has been read, no model family, geometry family, quantization rule, fixed model setting, feature rule, weighting rule, preprocessing rule, sampler setting, model artifact, or locked source file may be changed.

In particular, it is an **IMPLEMENTATION_FAILURE** to:

- compare alternative non-quantization hyperparameters or model configurations on the inner-development set;
- tune latent rank, dimension, number of components/clusters, architecture size, regularization, tree/ensemble size, geometry family, or sampler settings;
- inspect a failed outer retrieval result and then try a new geometry;
- alter a quantization candidate set or quantization selection objective after observing outer/blind performance;
- switch model, representation, whitening/projection, or geometry family after observing performance;
- alter a novelty margin because the outer evaluation was close to a gate;
- rerun a materially revised model after seeing any blind result;
- delete an earlier failed outer result and present a later run as the only evaluation.

If the frozen model fails an outer gate, the submitted result must remain a failure.

After outer evaluation starts, write an append-only:

```text
analysis/outer_evaluation_history.jsonl
```

containing exactly one record for the scientific submission, and:

```text
analysis/outer_evaluation_receipt.json
```

containing at least:

```text
outer_evaluation_count = 1
protocol_lock_sha256
model_lock_sha256
answers_sha256
```

The final branch history is part of grading. If a commit containing outer-evaluation outputs is followed by a commit that changes any locked source, model artifact, selection rule, quantization candidate set, or fixed setting and then performs another outer evaluation, the submission is an implementation failure even if the final files alone appear internally consistent.

A crash or syntax/runtime failure before any outer-held-out row or blind endpoint is read may be fixed. Once blind evaluation data are read, methodological changes are prohibited.

---

# Submitted geometry

The frozen structural model must induce one symmetric dissimilarity \(d_G(a,b)\ge0\). It need not satisfy the triangle inequality.

It must satisfy

\[
d_G(a,a)=0
\]

within `1e-12`, and

\[
|d_G(a,b)-d_G(b,a)|\le10^{-10}.
\]

It must return finite values for every eligible evaluation pair and may not use `jid`, row index, pair-specific lookup tables, outer validation ranks, or blind endpoints.

---

# Conventional reference geometry

Construct standardized non-quantized distance \(d_E\) using all eligible numerical construction variables and all 12 chemistry summaries. Estimate standard deviations only from the 50,000 outer-training rows.

For pair \((a,b)\), let \(O_{ab}\) denote coordinates observed in both materials. Define

\[
d_E(a,b)=\sqrt{\frac{1}{|O_{ab}|}\sum_{j\in O_{ab}}\left(\frac{x_{aj}-x_{bj}}{\sigma_j}\right)^2}.
\]

Pairs with fewer than 10 jointly observed conventional numerical variables are not eligible for analyses using \(d_E\).

A mixed-type comparator \(d_M\) must additionally include

```text
crys
dimensionality
spg_number treated categorically
```

using Gower-style normalized numerical differences and 0/1 categorical mismatches.

---

# Fixed geometry/transport evaluation cohort

Only after the protocol lock and model lock are written and verified:

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
eval_pos = eval_rng.choice(eligible_pos, size=1500, replace=False)
```

Exactly these 1,500 materials must be used. There are

\[
\binom{1500}{2}=1,124,250
\]

unordered evaluation pairs.

---

# TEST 1 — geometry novelty

For each anchor \(a\), compute its 20 nearest neighbors under \(d_G,d_E,d_M\). Define

\[
J^E_{20}(a)=\frac{|N^G_{20}(a)\cap N^E_{20}(a)|}{20},
\]

\[
J^M_{20}(a)=\frac{|N^G_{20}(a)\cap N^M_{20}(a)|}{20}.
\]

Use 3,000 anchor-bootstrap replicates with seed `101002`.

Neighborhood novelty passes iff

\[
CI^{upper}_{95}(\bar J^E_{20})<0.50
\]

and

\[
CI^{upper}_{95}(\bar J^M_{20})<0.50.
\]

Thus, at the upper 95% confidence bound, a majority of the 20 nearest neighbors must differ from the corresponding conventional-geometry neighborhood.

Also compute pairwise Spearman correlations \(\rho(d_G,d_E)\) and \(\rho(d_G,d_M)\). The rank guard passes iff the absolute global correlations and the upper 95% CI of mean absolute anchor-wise correlations are all below `0.75`.

Set `geometry_novel = YES` iff both conditions pass.

---

# TEST 2 — blind distant-analog retrieval

For each blind transport variable, estimate its training mean and standard deviation using only the 50,000 outer-training rows. Define standardized transport distance

\[
d_F(a,b)=\sqrt{\frac18\sum_{k=1}^{8}(\widetilde Y_{ak}-\widetilde Y_{bk})^2}.
\]

Let

\[
e_{0.90}=Q_{0.90}(d_E)
\]

over eligible evaluation pairs.

For anchor \(a\), define the conventionally far set

\[
C_a=\{b:d_E(a,b)\ge e_{0.90}\}.
\]

The geometry-retrieved partner is

\[
b_G(a)=\arg\min_{b\in C_a}d_G(a,b).
\]

For each anchor, choose 20 distinct controls in \(C_a\), excluding \(b_G(a)\), whose \(d_E(a,c)\) values are closest to \(d_E(a,b_G(a))\).

Define

\[
R_{\rm retrieval}=\frac{\operatorname{mean}_a d_F(a,b_G(a))}{\operatorname{mean}_a\operatorname{mean}_{c\in C_a^{match}}d_F(a,c)}.
\]

Define anchor win rate

\[
W=\frac1A\sum_a I\left[d_F(a,b_G(a))<\operatorname{mean}_{c\in C_a^{match}}d_F(a,c)\right].
\]

Use 3,000 anchor-bootstrap replicates with seed `101004`.

## Raw-control matching-quality gate

The functional comparison is only interpretable if retrieved pairs and controls are genuinely matched on ordinary raw separation.

For each anchor define

\[
g_a=\frac1{20}\sum_{c\in C_a^{match}}\left|d_E(a,c)-d_E(a,b_G(a))\right|.
\]

Define the dimensionless global matching-gap ratio

\[
G_{match}=\frac{\operatorname{mean}_a g_a}{e_{0.90}}.
\]

Use 3,000 anchor-bootstrap replicates with seed `101005`.

The matching-quality gate passes iff

\[
G_{match}\le0.10
\]

and

\[
CI^{upper}_{95}(G_{match})<0.10.
\]

The 10% tolerance is relative to the fixed 90th-percentile raw-distance scale itself; it prevents apparent functional gains from being attributed to controls that are materially farther or nearer in ordinary physicochemical space than the retrieved pair.

Retrieval utility passes iff **all** of the following hold:

\[
R_{\rm retrieval}\le0.95,
\]

\[
CI^{upper}_{95}(R_{\rm retrieval})<1,
\]

\[
W\ge0.60,
\]

\[
CI^{lower}_{95}(W)>0.55,
\]

and the raw-control matching-quality gate passes.

Set `retrieval_useful = YES` iff all five requirements pass.

---

# Generative evaluation panel

The generative tests do not use the eight transport endpoints.

Construct a fixed generation panel using only allowed construction variables. Include a numerical construction variable iff, among the 50,000 training rows:

```text
finite observed coverage >= 0.50
number of distinct observed values >= 3
```

Include `crys` and `dimensionality` iff nonmissing training coverage is at least `0.50`. Include each of the 12 chemistry summaries under the same numerical rule.

Write the resulting fixed set to:

```text
analysis/generation_panel.json
```

before outer generation evaluation.

---

# Independent-marginal baseline

Construct a fixed negative-control generator using the 50,000 training materials. For each coordinate independently, sample from its empirical training marginal:

\[
P_{IND}(X)=\prod_j\widehat P_{train}(X_j).
\]

For conditional completion, observed coordinates remain fixed and hidden coordinates are independently sampled from their training marginals. Use seed `314159`.

---

# TEST 3 — generation from a completely empty state

Generate exactly `10000` independent material-state samples from \(P_M(X)\) with no material coordinates supplied, using seed `161803`.

Write:

```text
analysis/unconditional_samples.csv.gz
```

All coordinates in the generation panel must be populated.

The blank-generation validity gate passes iff:

```text
valid rows >= 9500 / 10000
unique generated rows >= 80%
exact training-row matches <= 10%
```

Exact row comparison uses numerical rounding to `1e-8`.

---

# TEST 3A — marginal fidelity

For each numerical generation variable \(j\), report

\[
D_j^{num}=\frac{W_1(X_j^{generated},X_j^{heldout})}{IQR_j^{train}}.
\]

If training IQR is zero, use training range. For categorical variables report total-variation distance

\[
D_j^{cat}=\frac12\sum_c|p_j^{generated}(c)-p_j^{heldout}(c)|.
\]

This test is descriptive and is not by itself a primary success gate.

---

# TEST 3B — cross-structure retention

For each generation-panel coordinate \(j\), predict \(X_j\) from all other panel coordinates using fixed verifier models.

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

Fit preprocessing statistics and encoders from the real 50,000-row training set.

Train three verifier versions on 10,000 rows each:

1. real training materials;
2. submitted generated materials;
3. independent-marginal baseline materials.

Evaluate all three on the same outer-held-out real materials.

For numerical targets use \(L_j=MAE_j/IQR_j^{train}\). For categorical targets use \(L_j=1-accuracy_j\).

A target is informative iff

\[
L_j^{REAL}\le0.95L_j^{IND}.
\]

For each informative target define

\[
g_j=\frac{L_j^{IND}-L_j^{GEN}}{L_j^{IND}-L_j^{REAL}}.
\]

Do not clip \(g_j\). Define

\[
G=\frac1{|J|}\sum_{j\in J}g_j.
\]

Bootstrap outer-held-out evaluation rows 2,000 times with seed `424242`, without refitting verifier models.

The cross-structure gate passes iff

\[
CI^{lower}_{95}(G)>0
\]

and at least 60% of informative targets satisfy

\[
L_j^{GEN}<L_j^{IND}.
\]

Set `cross_structure_preserved = YES` iff both conditions pass.

---

# TEST 4 — arbitrary partial-state completion

Use exactly 2,000 outer-held-out rows having at least 80% of the generation panel originally observed, sampled with seed `20260916`.

Create independent reproducible masks with seed `27182818` at:

```text
20% hidden
50% hidden
80% hidden
```

For each masked row and masking level, generate 10 conditional completions. Observed coordinates must remain fixed.

For hidden numerical coordinates, use the median of the 10 generated values as the point completion. For hidden categorical coordinates, use the modal state, breaking ties lexicographically.

For numerical coordinate \(j\):

\[
\ell_j=\frac{|\hat x_j-x_j|}{IQR_j^{train}}.
\]

For categorical coordinate \(j\):

\[
\ell_j=I(\hat x_j\ne x_j).
\]

Average first within coordinate across rows, then equally across coordinates. Let \(L_M(q)\) be model loss and \(L_{IND}(q)\) independent-marginal loss. Define

\[
R_C(q)=\frac{L_M(q)}{L_{IND}(q)}.
\]

Bootstrap completion rows 3,000 times with seed `577215`.

The 20%-hidden result is diagnostic. The completion gate passes iff both

\[
CI^{upper}_{95}(R_C(0.50))<1
\]

and

\[
CI^{upper}_{95}(R_C(0.80))<1,
\]

and at both masking levels at least 60% of individually evaluable coordinates have lower submitted-model loss than independent-marginal loss.

Set `conditional_completion_useful = YES` iff all requirements pass.

---

# Required primary outputs

Write:

```text
analysis/answers.json
```

with at least:

```text
analysis_status
geometry_novel
retrieval_useful
blank_generation_valid
cross_structure_preserved
conditional_completion_useful
single_model_requirement_pass
single_shot_outer_evaluation_pass
no_model_tuning_pass
raw_control_matching_pass
primary_discovery

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

Use if any of the following occurs:

- wrong outer or inner split;
- outer-held-out data used before model freezing;
- blind transport values used in model construction or methodological selection;
- `no_model_tuning_pass = NO`;
- any non-quantization hyperparameter/model/geometry/sampler tuning is performed;
- alternative model families, architecture sizes, latent ranks/dimensions, component counts, regularization values, geometry families, or sampler settings are compared and selected;
- model or geometry revised after any outer/blind evaluation;
- more than one scientific outer evaluation is performed;
- a failed outer result is discarded and followed by a revised model evaluation;
- geometry and generator use independently fitted learned models;
- protocol-lock or model-lock hash mismatch;
- branch history shows post-outer changes to locked source/model/selection files;
- missing required artifact;
- invalid/asymmetric distance;
- supplied observed values altered during conditional completion;
- unconditional generation seeded from real material rows;
- required outputs cannot be mechanically reproduced.

## GEOMETRY_FAILURE

Use if implementation checks pass but `geometry_novel = NO`.

## RETRIEVAL_FAILURE

Use if geometry passes but either `retrieval_useful = NO` or `raw_control_matching_pass = NO`.

## GENERATION_FAILURE

Use if geometry and retrieval pass but any of the following are `NO`:

```text
blank_generation_valid
cross_structure_preserved
conditional_completion_useful
single_model_requirement_pass
```

## DISCOVERY_SUCCESS

Set

```text
analysis_status = DISCOVERY_SUCCESS
primary_discovery = YES
```

iff all of the following are `YES`:

```text
geometry_novel
retrieval_useful
raw_control_matching_pass
blank_generation_valid
cross_structure_preserved
conditional_completion_useful
single_model_requirement_pass
single_shot_outer_evaluation_pass
no_model_tuning_pass
```

Otherwise set `primary_discovery = NO`.

---

# Scientific interpretation

The final report must distinguish four scientific claims:

1. the learned geometry differs from conventional material similarity;
2. it retrieves blind external functional analogs under tightly raw-distance-matched controls;
3. the underlying structural model generates coherent multivariate material states from no observations;
4. the same model reconstructs hidden material state from arbitrary partial observations.

Passing geometry/retrieval alone is insufficient. Passing generation alone is insufficient. The scientific result requires all four capabilities from one frozen structural model, with no non-quantization model tuning, under a one-shot blind outer evaluation.

---

# Reproducibility

A fresh run must recreate every graded output from:

```text
jarvis.tgz
submitted source code
declared package dependencies
```

No precomputed pair labels, external material database, hidden manually curated table, or unpublished artifact may be required. Evaluation randomness must use exactly the seeds specified above. Model-fitting randomness is allowed but must be documented in `model_manifest.json` and the evaluated fitted realization must be frozen before outer evaluation.
