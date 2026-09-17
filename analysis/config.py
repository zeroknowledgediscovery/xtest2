"""Single fixed configuration for the submitted structural model.

Every value here is a prespecified constant, fixed before any outer-held-out
row or blind transport value is read. The only quantity selected by
comparison on the inner fit/development split is QUANTIZATION_CANDIDATES ->
one resolution; everything else is a single fixed value used throughout.
"""

MEASURED_NUMERIC_COLUMNS = [
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

MEASURED_CATEGORICAL_COLUMNS = ["crys", "dimensionality"]

SYMMETRY_CATEGORICAL_COLUMNS = ["spg_centrosymmetric"]

MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "--", "missing"}

MISSING_STATE = "__MISSING__"

# ---- Model-family fixed settings (single submitted configuration) --------
MODEL_FAMILY = "chow_liu_tree_categorical_bn"
GEOMETRY_FAMILY = "fisher_score_euclidean"

LAPLACE_ALPHA = 1.0          # fixed CPT/marginal smoothing constant
FISHER_VAR_EPS = 1e-6        # fixed epsilon for Fisher-information normalization

QUANTIZATION_CANDIDATES = [5, 10]   # the only tunable quantity
QUANTIZATION_SELECTION_RULE = (
    "Fit the full Chow-Liu tree model on the 40000-row inner fit split for "
    "each candidate q in {5,10} (identical model settings otherwise). "
    "Evaluate mean held-out log-likelihood per material on the 10000-row "
    "inner development split. Select the q with higher mean held-out "
    "log-likelihood; ties broken in favor of the smaller q. This objective "
    "does not use any blind transport variable or any outer-held-out row."
)

RANDOM_SEED_QUANT_TIEBREAK = None  # deterministic rule, no randomness

# Generation / evaluation seeds specified by the task
SEED_UNCONDITIONAL_GENERATION = 161803
SEED_INDEPENDENT_MARGINAL = 314159
SEED_ANCHOR_BOOTSTRAP_TEST1 = 101002
SEED_ANCHOR_BOOTSTRAP_TEST2_RETRIEVAL = 101004
SEED_ANCHOR_BOOTSTRAP_TEST2_MATCHING = 101005
SEED_EVAL_COHORT = 20260915
SEED_VERIFIER_BOOTSTRAP = 424242
SEED_COMPLETION_ROW_SAMPLE = 20260916
SEED_COMPLETION_MASKS = 27182818
SEED_COMPLETION_BOOTSTRAP = 577215
VERIFIER_RANDOM_STATE = 99173

N_ANCHOR_BOOTSTRAP = 3000
N_VERIFIER_BOOTSTRAP = 2000
N_COMPLETION_BOOTSTRAP = 3000

K_NEIGHBORS = 20

BLIND_COLUMNS = [
    "n-Seebeck", "p-Seebeck",
    "n-powerfact", "p-powerfact",
    "ncond", "pcond",
    "nkappa", "pkappa",
]
