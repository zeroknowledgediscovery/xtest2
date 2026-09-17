"""Fixed, prespecified configuration constants for the whole workflow.

Every value here is chosen once, before any outer-held-out row or blind
transport value is read, and is never changed in response to a result. This
module is hashed into protocol_lock.json.
"""

MEASURED_COLUMNS = [
    "crys",
    "dimensionality",
    "nat",
    "density",
    "formation_energy_peratom",
    "ehull",
    "exfoliation_energy",
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "hse_gap",
    "spillage",
    "magmom_oszicar",
    "magmom_outcar",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "avg_elec_mass",
    "avg_hole_mass",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "poisson",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_dielectric_electronic",
    "dfpt_piezo_max_dielectric_ionic",
    "max_ir_mode",
    "min_ir_mode",
    "Tc_supercon",
]

# Numeric-valued measured columns (everything above except the two categoricals).
MEASURED_NUMERIC_COLUMNS = [c for c in MEASURED_COLUMNS if c not in ("crys", "dimensionality")]

CATEGORICAL_COLUMNS = ["crys", "dimensionality"]

CHEM_COLUMNS = [
    "chem_n_elements",
    "chem_composition_entropy",
    "chem_mean_atomic_number",
    "chem_atomic_number_range",
    "chem_mean_electronegativity",
    "chem_electronegativity_range",
    "chem_mean_atomic_radius",
    "chem_atomic_radius_range",
    "chem_mean_group",
    "chem_group_range",
    "chem_mean_period",
    "chem_period_range",
]

BLIND_COLS = [
    "n-Seebeck", "p-Seebeck",
    "n-powerfact", "p-powerfact",
    "ncond", "pcond",
    "nkappa", "pkappa",
]

MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "--", "missing"}

# ---- fixed seeds mandated by instructions.md ----
OUTER_SPLIT_SEED = 1
OUTER_TRAIN_SIZE = 50000
INNER_DEV_SEED = 271828
INNER_DEV_SIZE = 10000

EVAL_COHORT_SEED = 20260915
EVAL_COHORT_SIZE = 1500

GEOMETRY_NOVELTY_BOOTSTRAP_SEED = 101002
RETRIEVAL_BOOTSTRAP_SEED = 101004
MATCHING_GAP_BOOTSTRAP_SEED = 101005

INDEPENDENT_MARGINAL_SEED = 314159
UNCONDITIONAL_SAMPLE_SEED = 161803
UNCONDITIONAL_SAMPLE_COUNT = 10000

CROSS_STRUCTURE_VERIFIER_SEED = 99173
CROSS_STRUCTURE_BOOTSTRAP_SEED = 424242

COMPLETION_ROW_SEED = 20260916
COMPLETION_ROW_COUNT = 2000
COMPLETION_MASK_SEED = 27182818
COMPLETION_BOOTSTRAP_SEED = 577215

N_NEIGHBORS = 20
N_ANCHOR_BOOTSTRAP = 3000
N_CONTROLS = 20

# ---- single fixed structural-model configuration (no tuning) ----
# Semiparametric ("extended rank likelihood") Gaussian copula over the
# generation-panel coordinates, fit by a plug-in EM algorithm for the
# missing-data multivariate Gaussian correlation matrix. No quantization is
# used, so no hyperparameter of any kind is tuned (per instructions.md,
# "If the model does not use quantization/discretization, no hyperparameter
# tuning of any kind is allowed").
MODEL_FAMILY = "semiparametric_gaussian_copula_em"
GEOMETRY_FAMILY = "copula_precision_mahalanobis_full_completion"
EM_ITERATIONS = 30
COVARIANCE_RIDGE = 1e-6
GENERATION_MIN_COVERAGE = 0.50
GENERATION_MIN_DISTINCT = 3
