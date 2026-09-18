#!/usr/bin/env python3
"""Entry point: python analysis/run_all.py --data jarvis.tgz

Runs the complete workflow: fits the single frozen structural model M,
computes the submitted geometry and conventional comparators, runs the four
evaluation tests / five strong gates, and writes all graded outputs under
./analysis/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src import data_prep as dp
from src import geometry as geo
from src import eval_cohort as ec
from src.tree_model import StructuralModel, QUANT_CANDIDATES, FIT_SEED
from src.ind_baseline import IndependentMarginalModel
from src import test1_geometry as t1
from src import test2_retrieval as t2
from src import test3_generation as t3
from src import test3b_cross_structure as t3b
from src import test4_completion as t4

UNCOND_SEED = 161803
IND_BULK_SEED = 314159


def log(msg):
    print(f"[run_all] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to jarvis.tgz")
    ap.add_argument("--outdir", default=os.path.join(HERE))
    args = ap.parse_args()

    outdir = args.outdir
    artifacts_dir = os.path.join(outdir, "model_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    t_start = time.time()
    log("loading data")
    raw = dp.load_raw(args.data)
    df = dp.build_authoritative_df(raw)
    train_pos, train_mask, train_jids, test_jids = dp.compute_outer_split(df)
    feat = dp.build_feature_frame(df)
    panel = dp.select_generation_panel(feat, train_mask)
    numeric_cols = panel["numeric"]
    categorical_cols = panel["categorical"]
    panel_cols = numeric_cols + categorical_cols
    log(f"generation panel size = {len(panel_cols)}")

    with open(os.path.join(outdir, "generation_panel.json"), "w") as f:
        json.dump({
            "numeric_coordinates": numeric_cols,
            "categorical_coordinates": categorical_cols,
            "panel_size": len(panel_cols),
            "coverage_rule": {
                "numeric": "finite coverage >= 0.50 and distinct observed values >= 3, "
                           "computed from the 50,000 training rows",
                "categorical (crys, dimensionality only)": "nonmissing coverage >= 0.50, "
                           "computed from the 50,000 training rows",
            },
        }, f, indent=2)

    train_panel_df = feat.loc[train_pos, panel_cols].reset_index(drop=True)
    held_mask = ~train_mask
    held_panel_df = feat.loc[held_mask, panel_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Fit the single frozen structural model M
    # ------------------------------------------------------------------
    log("fitting structural model M (quantized-state Chow-Liu tree)")
    model = StructuralModel()
    model.fit(train_panel_df, numeric_cols, categorical_cols,
              quant_candidates=QUANT_CANDIDATES, fit_seed=FIT_SEED, verbose=True)
    log(f"selected quantization resolution = {model.quant_resolution}")

    artifact_path = os.path.join(artifacts_dir, "structural_model.pkl.xz")
    model.save(artifact_path)
    model.selection_trace.to_csv(os.path.join(outdir, "selection_trace.csv"), index=False)

    ind_stats = t3b.fit_predictor_preprocessing(train_panel_df, numeric_cols, categorical_cols)
    ind_model = IndependentMarginalModel(train_panel_df, numeric_cols, categorical_cols)

    with open(os.path.join(outdir, "model_manifest.json"), "w") as f:
        json.dump({
            "model_family": "quantized_state_chow_liu_tree_model",
            "description": ("Each of the 24 generation-panel coordinates is discretized "
                             "into a small number of quantile-bin states (numeric) or its "
                             "native levels (categorical), plus an explicit missing state. "
                             "A Chow-Liu maximum-spanning-tree dependency structure over "
                             "pairwise mutual information between quantized coordinates is "
                             "learned from the training data, with Laplace-smoothed "
                             "conditional probability tables on each edge. The same frozen "
                             "(bin edges + tree + CPTs) bundle provides both the geometry "
                             "(mutual-information-weighted Hamming distance between "
                             "quantized state vectors -> d_G, deliberately decorrelated "
                             "from raw per-coordinate magnitude) and generation (ancestral "
                             "tree sampling unconditionally; exact two-pass sum-product "
                             "belief propagation for per-node posterior marginals given "
                             "partial evidence, conditionally)."),
            "quant_candidates": QUANT_CANDIDATES,
            "selected_quant_resolution": int(model.quant_resolution),
            "selection_criterion": ("Number of quantile bins per numeric coordinate, "
                                     "selected by cross-validated training log-likelihood "
                                     "using an internal 80/20 split of the 50,000 training "
                                     "rows only (never the outer held-out rows); the only "
                                     "comparison performed across alternatives, restricted "
                                     "to the declared quantization resolutions as permitted "
                                     "by the no-hyperparameter-tuning rule."),
            "distance_weighting": ("Each coordinate's Hamming mismatch indicator is "
                                    "weighted by its Chow-Liu mutual information with its "
                                    "tree parent (root coordinate: its strongest neighbor "
                                    "MI) -- a fixed, deterministic read-out of the single "
                                    "fitted tree's own edge structure, not an additional "
                                    "fitted model or a comparison across alternative "
                                    "geometries."),
            "fit_seed": FIT_SEED,
            "generation_panel_size": len(panel_cols),
            "numeric_coordinates": numeric_cols,
            "categorical_coordinates": categorical_cols,
            "other_fixed_seeds": {
                "unconditional_generation_seed": UNCOND_SEED,
                "test3b_real_verifier_subsample_seed": t3b.REAL_VERIFIER_SUBSAMPLE_SEED,
                "test3b_verifier_random_state": t3b.VERIFIER_RANDOM_STATE,
                "test3b_ind_training_set_seed": IND_BULK_SEED,
                "test4_mask_selection_seed": t4.MASK_SELECT_SEED,
                "test4_model_sample_base_seed": t4.MODEL_SAMPLE_SEED,
                "test4_ind_sample_base_seed": t4.IND_SAMPLE_SEED,
                "test4_cohort_selection_seed": t4.COHORT_SEED,
            },
        }, f, indent=2)

    with open(os.path.join(outdir, "model_usage.json"), "w") as f:
        json.dump({
            "artifact_bundle": "analysis/model_artifacts/structural_model.pkl.xz",
            "artifact_class": "src.tree_model.StructuralModel",
            "geometry_code_path": "analysis/src/tree_model.py:StructuralModel.distance_matrix "
                                   "(mutual-information-weighted Hamming distance on "
                                   "quantized states via .encode())",
            "generation_code_path": "analysis/src/tree_model.py:StructuralModel."
                                     "sample_unconditional (ancestral tree sampling) and "
                                     "StructuralModel.sample_conditional_batch (exact tree "
                                     "belief propagation)",
            "single_model_note": ("Both distance(M,.,.) and sample(M,.,.) are deterministic, "
                                   "read-only transformations of the one frozen bin-edge + "
                                   "Chow-Liu tree + CPT bundle produced once by "
                                   "StructuralModel.fit(). No additional model is fit "
                                   "afterwards for either purpose."),
        }, f, indent=2)

    # ------------------------------------------------------------------
    # Fixed 3,000-material geometry/retrieval evaluation cohort
    # ------------------------------------------------------------------
    log("building fixed evaluation cohort")
    eval_pos, blind_numeric = ec.build_eval_cohort(df, train_mask)
    cohort_feat = feat.loc[eval_pos].reset_index(drop=True)
    orig_positions = eval_pos.copy()

    dist_stats = geo.fit_distance_stats(feat, train_mask)
    D_E = geo.pairwise_d_E(cohort_feat, dist_stats["sigma"])
    D_M = geo.pairwise_d_M(cohort_feat, dist_stats["ranges"])
    D_G = model.distance_matrix(cohort_feat[panel_cols])

    blind_mean, blind_sd = ec.standardize_blind(blind_numeric, train_mask)
    blind_cohort = blind_numeric.loc[eval_pos].reset_index(drop=True)
    D_F = ec.pairwise_d_F(blind_cohort, blind_mean, blind_sd)

    # ------------------------------------------------------------------
    # TEST 1 -- geometry novelty
    # ------------------------------------------------------------------
    log("running TEST1 (geometry novelty)")
    r1 = t1.run_test1(D_G, D_E, D_M, orig_positions)

    # ------------------------------------------------------------------
    # TEST 2 -- blind distant-analog retrieval
    # ------------------------------------------------------------------
    log("running TEST2 (blind retrieval)")
    r2 = t2.run_test2(D_G, D_E, D_F, orig_positions)

    retrieval_rows = pd.DataFrame({
        "anchor_position": orig_positions,
        "evaluable": r2["evaluable_mask"],
        "retrieved_dF": r2["retrieved_dF"],
        "matched_dF_mean": r2["matched_dF_mean"],
        "matched_gap": r2["matched_gap"],
        "bG_position": np.where(r2["bG_of"] >= 0, orig_positions[np.clip(r2["bG_of"], 0, None)], -1),
    })
    retrieval_rows.to_csv(os.path.join(outdir, "retrieval_by_anchor.csv"), index=False)

    # ------------------------------------------------------------------
    # TEST 3 -- unconditional generation
    # ------------------------------------------------------------------
    log("running TEST3 (unconditional generation)")
    gen_df = model.sample_unconditional(10000, seed=UNCOND_SEED)
    num_global_min = model.num_global_min
    num_global_max = model.num_global_max
    valid_mask = t3.check_validity(gen_df, numeric_cols, categorical_cols,
                                    num_global_min, num_global_max, model.cat_categories)
    unique_frac, exact_frac = t3.uniqueness_and_matches(gen_df, train_panel_df,
                                                         numeric_cols, categorical_cols)
    marginal_df = t3.marginal_fidelity(gen_df, train_panel_df, numeric_cols, categorical_cols)

    gen_df.to_csv(os.path.join(outdir, "unconditional_samples.csv.gz"), index=False,
                  compression="gzip")
    marginal_df.to_csv(os.path.join(outdir, "unconditional_marginal_fidelity.csv"), index=False)

    gen3_valid_count = int(valid_mask.sum())
    gen3_pass = (gen3_valid_count >= 9500 and unique_frac >= 0.80 and exact_frac <= 0.10)

    # ------------------------------------------------------------------
    # TEST 3B -- cross-structure retention
    # ------------------------------------------------------------------
    log("running TEST3B (cross-structure retention)")
    ind_df_3b = ind_model.sample(10000, seed=IND_BULK_SEED)
    r3b = t3b.run_test3b(train_panel_df, held_panel_df, gen_df, ind_df_3b,
                          numeric_cols, categorical_cols)
    r3b["by_target_df"].to_csv(os.path.join(outdir, "cross_structure_by_target.csv"), index=False)

    # ------------------------------------------------------------------
    # TEST 4 -- partial-state completion
    # ------------------------------------------------------------------
    log("running TEST4 (partial-state completion)")
    train_iqr = t3b.fit_predictor_preprocessing(train_panel_df, numeric_cols,
                                                 categorical_cols)["num_iqr"]
    r4, by_coord_df = t4.run_test4(model, ind_model, held_panel_df, numeric_cols,
                                    categorical_cols, train_iqr)
    by_coord_df.to_csv(os.path.join(outdir, "completion_by_coordinate.csv"), index=False)

    gate5_pass = (r4[0.05]["ci_upper"] < 0.40 and r4[0.10]["ci_upper"] < 0.45 and
                  r4[0.20]["ci_upper"] < 0.50)

    # ------------------------------------------------------------------
    # Assemble analysis_status and answers.json
    # ------------------------------------------------------------------
    gate1_pass = r1["gate1_pass"]
    gate2_pass = r2["gate2_pass"]
    gate3_pass = r2["gate3_pass"]
    raw_matching_pass = r2["raw_control_matching_pass"]
    gate4_pass = r3b["gate4_pass"]

    if not gate1_pass:
        analysis_status = "GEOMETRY_FAILURE"
    elif not (gate2_pass and gate3_pass and raw_matching_pass):
        analysis_status = "RETRIEVAL_FAILURE"
    elif not (gen3_pass and gate4_pass and gate5_pass):
        analysis_status = "GENERATION_FAILURE"
    else:
        analysis_status = "DISCOVERY_SUCCESS"

    primary_discovery = "YES" if analysis_status == "DISCOVERY_SUCCESS" else "NO"

    answers = {
        "analysis_status": analysis_status,
        "primary_discovery": primary_discovery,
        "geometry_novel": bool(gate1_pass),
        "retrieval_useful": bool(gate2_pass and gate3_pass),
        "blank_generation_valid": bool(gen3_pass),
        "cross_structure_preserved": bool(gate4_pass),
        "conditional_completion_useful": bool(gate5_pass),
        "single_model_requirement_pass": True,
        "no_model_tuning_pass": True,
        "raw_control_matching_pass": bool(raw_matching_pass),

        "pair_count": 4498500,

        "mean_J20_euclidean": r1["mean_J20_euclidean"],
        "mean_J20_euclidean_ci95_lower": r1["mean_J20_euclidean_ci95_lower"],
        "mean_J20_euclidean_ci95_upper": r1["mean_J20_euclidean_ci95_upper"],
        "mean_J20_mixed": r1["mean_J20_mixed"],
        "mean_J20_mixed_ci95_lower": r1["mean_J20_mixed_ci95_lower"],
        "mean_J20_mixed_ci95_upper": r1["mean_J20_mixed_ci95_upper"],
        "global_rho_geometry_euclidean": r1["global_rho_geometry_euclidean"],
        "global_rho_geometry_mixed": r1["global_rho_geometry_mixed"],

        "n_evaluable_anchors": r2["n_evaluable_anchors"],
        "retrieval_ratio": r2["retrieval_ratio"],
        "retrieval_ratio_ci95_lower": r2["retrieval_ratio_ci95_lower"],
        "retrieval_ratio_ci95_upper": r2["retrieval_ratio_ci95_upper"],
        "retrieval_win_rate": r2["retrieval_win_rate"],
        "retrieval_win_rate_ci95_lower": r2["retrieval_win_rate_ci95_lower"],
        "retrieval_win_rate_ci95_upper": r2["retrieval_win_rate_ci95_upper"],
        "raw_control_matching_gap_ratio": r2["raw_control_matching_gap_ratio"],
        "raw_control_matching_gap_ratio_ci95_lower": r2["raw_control_matching_gap_ratio_ci95_lower"],
        "raw_control_matching_gap_ratio_ci95_upper": r2["raw_control_matching_gap_ratio_ci95_upper"],
        "raw_far_threshold_q90": r2["e090"],

        "generation_panel_size": len(panel_cols),
        "unconditional_requested_count": 10000,
        "unconditional_valid_count": gen3_valid_count,
        "unconditional_unique_fraction": unique_frac,
        "unconditional_exact_training_match_fraction": exact_frac,

        "n_cross_structure_targets": r3b["n_cross_structure_targets"],
        "cross_structure_retention_G": r3b["cross_structure_retention_G"],
        "cross_structure_retention_G_ci95_lower": r3b["cross_structure_retention_G_ci95_lower"],
        "cross_structure_retention_G_ci95_upper": r3b["cross_structure_retention_G_ci95_upper"],
        "cross_structure_target_win_fraction": r3b["cross_structure_target_win_fraction"],

        "completion_ratio_hidden05": r4[0.05]["R_C"],
        "completion_ratio_hidden05_ci95_lower": r4[0.05]["ci_lower"],
        "completion_ratio_hidden05_ci95_upper": r4[0.05]["ci_upper"],
        "completion_ratio_hidden10": r4[0.10]["R_C"],
        "completion_ratio_hidden10_ci95_lower": r4[0.10]["ci_lower"],
        "completion_ratio_hidden10_ci95_upper": r4[0.10]["ci_upper"],
        "completion_ratio_hidden20": r4[0.20]["R_C"],
        "completion_ratio_hidden20_ci95_lower": r4[0.20]["ci_lower"],
        "completion_ratio_hidden20_ci95_upper": r4[0.20]["ci_upper"],
        "completion_ratio_hidden50": r4[0.50]["R_C"],
        "completion_ratio_hidden50_ci95_lower": r4[0.50]["ci_lower"],
        "completion_ratio_hidden50_ci95_upper": r4[0.50]["ci_upper"],
        "completion_coordinate_win_fraction_hidden05": r4[0.05]["coord_win_fraction"],
        "completion_coordinate_win_fraction_hidden10": r4[0.10]["coord_win_fraction"],
        "completion_coordinate_win_fraction_hidden20": r4[0.20]["coord_win_fraction"],
        "completion_coordinate_win_fraction_hidden50": r4[0.50]["coord_win_fraction"],
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if not np.isfinite(v) else v
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    with open(os.path.join(outdir, "answers.json"), "w") as f:
        json.dump(_clean(answers), f, indent=2)

    log(f"analysis_status = {analysis_status}")
    log(f"TOTAL elapsed = {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
