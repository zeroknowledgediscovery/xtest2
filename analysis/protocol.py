"""Writers for the pre-outer protocol-lock artifacts required by
instructions.md. Every function here is called before any outer-held-out
row or blind transport value is read."""
import csv
import json
import os

from . import config
from .lock_utils import sha256_file, list_source_files


def write_selection_trace(path):
    """Exactly one structural-model configuration: no quantization is used
    by this submission, so no quantization-resolution comparison rows exist
    (permitted degenerate case of the multi-row exception)."""
    fields = [
        "candidate_id", "scientific_definition", "fit_data_description",
        "development_data_description", "quantization_resolution",
        "fixed_model_settings", "development_objective", "development_result",
        "selected", "notes",
    ]
    row = {
        "candidate_id": "M1",
        "scientific_definition": (
            "Semiparametric Gaussian copula over the generation-panel "
            "coordinates (empirical-CDF / fixed-alphabetical-order marginals "
            "to standard-normal latent scores; latent correlation matrix "
            "Sigma fit by plug-in EM for the missing-data multivariate "
            "normal). Geometry = whitened Mahalanobis distance on each "
            "material's fully-completed latent vector using Sigma^-1. "
            "Generation = unconditional/conditional sampling from N(0,Sigma) "
            "followed by the inverse marginal transform."
        ),
        "fit_data_description": "50,000-row fixed outer training set (all rows; inner fit/dev split not needed since no quantization is tuned)",
        "development_data_description": "not used (no quantization/discretization in this model; no other hyperparameter is eligible for inner-development comparison)",
        "quantization_resolution": "",
        "fixed_model_settings": json.dumps({
            "em_iterations": config.EM_ITERATIONS,
            "covariance_ridge": config.COVARIANCE_RIDGE,
            "categorical_ordering_rule": "fixed alphabetical order of observed category labels",
        }),
        "development_objective": "not applicable",
        "development_result": "not applicable",
        "selected": "YES",
        "notes": "Single prespecified configuration; no candidate grid of any kind was implemented or evaluated.",
    }
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)


def write_model_manifest(path, panel):
    manifest = {
        "fitted_parameters": {
            "Sigma": "p x p latent Gaussian correlation matrix (p = %d), fit by plug-in EM for the missing-data multivariate normal on the 50,000-row outer training set." % (len(panel["numeric_columns"]) + len(panel["categorical_columns"])),
            "per_column_marginals": "Empirical training CDF (numeric columns) or empirical category frequencies under a fixed alphabetical category order (categorical columns), fit on the same 50,000 training rows.",
        },
        "fixed_model_settings": {
            "model_family": config.MODEL_FAMILY,
            "geometry_family": config.GEOMETRY_FAMILY,
            "em_iterations": config.EM_ITERATIONS,
            "covariance_ridge": config.COVARIANCE_RIDGE,
            "generation_panel_numeric_columns": panel["numeric_columns"],
            "generation_panel_categorical_columns": panel["categorical_columns"],
            "generation_min_coverage": config.GENERATION_MIN_COVERAGE,
            "generation_min_distinct": config.GENERATION_MIN_DISTINCT,
            "cross_structure_real_verifier_subsample": "train_pos[:10000]: the first 10,000 rows of the fixed 50,000-row outer training set, in the order produced by the mandated outer-split RNG (np.random.default_rng(1)). A fixed, prespecified, non-comparative choice.",
            "cross_structure_verifier_evaluation_rows": "the full outer-held-out set (all rows with train_mask == False), restricted per-target to rows with that target observed.",
        },
        "quantization_resolution": None,
        "quantization_selection_rule": "not applicable: this model uses no quantization/discretization of any coordinate.",
        "quantization_candidates": [],
        "model_fitting_randomness": "None. EM is a deterministic fixed-point iteration given Sigma initialized at the identity matrix; the only stochastic parts of the workflow are outer/blind evaluation sampling and generative sampling, both under the fixed seeds listed in protocol_lock.json.",
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def write_model_usage(path):
    usage = {
        "Sigma_correlation_matrix": {"used_by": "both", "fitted": True,
                                      "notes": "Precision Sigma^-1 defines d_G; Sigma defines the unconditional/conditional sampling distributions used for generation and for the geometry-side latent completion."},
        "per_column_empirical_marginals": {"used_by": "both", "fitted": True,
                                            "notes": "Forward transform (raw->latent) is used to build every latent vector consumed by geometry; inverse transform (latent->raw) is used to realize every generated/completed sample."},
        "deterministic_alphabetical_category_ordering": {"used_by": "both", "fitted": False,
                                                            "notes": "Deterministic transformation with no fitted parameters."},
        "chemistry_summary_features": {"used_by": "both", "fitted": False,
                                        "notes": "Deterministic transformation of `formula` with no fitted parameters; enter the same latent model as any other numeric coordinate."},
    }
    with open(path, "w") as f:
        json.dump(usage, f, indent=2)


def write_model_lock(path, model_dir, panel):
    artifact_hashes = {}
    for fname in ("sigma.npy", "meta.json", "marginals.json"):
        p = os.path.join(model_dir, fname)
        if os.path.isfile(p):
            artifact_hashes[fname] = sha256_file(p)
    lock = {
        "model_family": config.MODEL_FAMILY,
        "geometry_family": config.GEOMETRY_FAMILY,
        "fixed_model_settings": {
            "em_iterations": config.EM_ITERATIONS,
            "covariance_ridge": config.COVARIANCE_RIDGE,
        },
        "generation_panel": panel,
        "model_artifact_sha256": artifact_hashes,
    }
    with open(path, "w") as f:
        json.dump(lock, f, indent=2)
    return sha256_file(path)


def write_protocol_lock(path, analysis_dir, extra_paths, seeds):
    hashes = {}
    src_files = list_source_files(analysis_dir)
    for p in src_files:
        hashes[os.path.relpath(p, analysis_dir)] = sha256_file(p)
    for p in extra_paths:
        if os.path.isfile(p):
            hashes[os.path.relpath(p, analysis_dir)] = sha256_file(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    fp = os.path.join(root, fn)
                    hashes[os.path.relpath(fp, analysis_dir)] = sha256_file(fp)

    lock = {
        "model_family": config.MODEL_FAMILY,
        "geometry_family": config.GEOMETRY_FAMILY,
        "fixed_model_settings": {
            "em_iterations": config.EM_ITERATIONS,
            "covariance_ridge": config.COVARIANCE_RIDGE,
        },
        "quantization_candidates": [],
        "quantization_selection_rule": "not applicable: no quantization used.",
        "selected_quantization_resolution": None,
        "all_evaluation_seeds": seeds,
        "sha256": hashes,
    }
    with open(path, "w") as f:
        json.dump(lock, f, indent=2)
    return sha256_file(path)
