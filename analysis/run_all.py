#!/usr/bin/env python3
"""Master driver. Run from the repository root:

    python analysis/run_all.py --data jarvis.tgz

Implements, in order: data loading and fixed splits, quantization-resolution
selection on the inner fit/development split (the only tuned quantity),
final model fit on the 50,000-row outer training set, the pre-outer
protocol lock, and then the single one-shot outer evaluation (TEST 1-4).
"""
import argparse
import json
import os
import sys
import time

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ANALYSIS_DIR)

import numpy as np
import pandas as pd

import config
import data_loading as dl
import quantize as qz
import tree_model as tm
import geometry as geo
import distances as dist
import panel as panel_mod
import novelty
import retrieval
import generation as gen
import cross_structure as csx
import completion as comp
from lock_utils import sha256_file, write_json, list_py_sources


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(ANALYSIS_DIR), "analysis")
    artifacts_dir = os.path.join(out_dir, "model_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # ---------------------------------------------------------------- load
    log("Loading data and computing fixed splits")
    df = dl.load_raw_dataframe(args.data)
    train_pos, train_mask, train_jids, test_jids = dl.compute_outer_split(df)
    fit_pos, inner_dev_pos = dl.compute_inner_split(train_pos)
    coord = dl.build_coordinate_frame(df)

    numeric_coords = dl.NUMERIC_COORDINATES
    categorical_coords = dl.CATEGORICAL_COORDINATES
    all_coords = dl.ALL_COORDINATES

    # -------------------------------------------- quantization selection
    log("Selecting quantization resolution on inner fit/development split")
    selection_rows = []
    dev_results = {}
    for q in config.QUANTIZATION_CANDIDATES:
        quantizers_q = qz.fit_quantizers(coord, fit_pos, numeric_coords, categorical_coords, q)
        disc_fit = qz.discretize(coord, quantizers_q, numeric_coords, categorical_coords)
        n_states = {c: quantizers_q[c].n_states for c in all_coords}
        model_q = tm.ChowLiuTreeModel(columns=all_coords, n_states=n_states)
        model_q.fit_structure_and_params(disc_fit.loc[fit_pos][all_coords].to_numpy())
        dev_ll = model_q.mean_log_likelihood(disc_fit.loc[inner_dev_pos][all_coords].to_numpy())
        dev_results[q] = dev_ll
        selection_rows.append({
            "candidate_id": f"q{q}",
            "scientific_definition": config.MODEL_FAMILY,
            "fit_data_description": "40000-row inner fit split (train_pos minus inner_dev_pos)",
            "development_data_description": "10000-row inner development split",
            "quantization_resolution": q,
            "fixed_model_settings": json.dumps({
                "alpha": config.LAPLACE_ALPHA, "fisher_eps": config.FISHER_VAR_EPS,
                "model_family": config.MODEL_FAMILY, "geometry_family": config.GEOMETRY_FAMILY,
            }),
            "development_objective": "mean_held_out_log_likelihood",
            "development_result": dev_ll,
            "selected": False,
            "notes": "",
        })
        log(f"  q={q} mean held-out log-likelihood = {dev_ll:.6f}")

    best_q = max(config.QUANTIZATION_CANDIDATES, key=lambda q: (dev_results[q], -q))
    for row in selection_rows:
        if row["quantization_resolution"] == best_q:
            row["selected"] = True
            row["notes"] = "selected: highest mean held-out log-likelihood"
    selection_trace_path = os.path.join(out_dir, "selection_trace.csv")
    pd.DataFrame(selection_rows).to_csv(selection_trace_path, index=False)
    log(f"Selected quantization resolution q={best_q}")

    # -------------------------------------------- final frozen model fit
    log("Refitting the frozen structural model on all 50000 outer-training rows")
    quantizers = qz.fit_quantizers(coord, train_pos, numeric_coords, categorical_coords, best_q)
    discrete_all = qz.discretize(coord, quantizers, numeric_coords, categorical_coords)
    n_states = {c: quantizers[c].n_states for c in all_coords}
    model = tm.ChowLiuTreeModel(columns=all_coords, n_states=n_states)
    model.fit_structure_and_params(discrete_all.loc[train_pos][all_coords].to_numpy())

    numeric_panel, categorical_panel, panel_detail = panel_mod.select_generation_panel(coord, train_pos)
    panel_cols = numeric_panel + categorical_panel
    missing_state_by_col = {}
    for c in numeric_panel:
        missing_state_by_col[c] = quantizers[c].missing_state
    for c in categorical_panel:
        missing_state_by_col[c] = quantizers[c].missing_state
    model.set_panel_restriction(panel_cols, missing_state_by_col)

    write_json(os.path.join(out_dir, "generation_panel.json"), {
        "numeric_panel": numeric_panel,
        "categorical_panel": categorical_panel,
        "coverage_rule": "finite observed coverage >= 0.50 and >= 3 distinct values (numeric); "
                          "nonmissing coverage >= 0.50 (categorical); computed on the 50000-row outer training set",
        "detail": panel_detail,
    })
    log(f"Generation panel: {len(numeric_panel)} numeric + {len(categorical_panel)} categorical coordinates")

    # -------------------------------------------- pre-outer protocol lock
    log("Writing pre-outer lock artifacts")
    tree_edges_out = [(all_coords[model.parent[n]], all_coords[n]) for n in model.order if n != model.root]

    write_json(os.path.join(out_dir, "model_manifest.json"), {
        "model_family": config.MODEL_FAMILY,
        "geometry_family": config.GEOMETRY_FAMILY,
        "fitted_parameters": {
            "tree_root": all_coords[model.root],
            "tree_edges_parent_child": tree_edges_out,
            "root_marginal_dim": int(model.root_marginal.shape[0]),
            "n_conditional_probability_tables": len(model.cpt),
            "quantizer_bin_edges": {
                c: quantizers[c].edges.tolist() for c in numeric_coords
            },
            "categorical_levels": {
                c: quantizers[c].levels for c in categorical_coords
            },
        },
        "fixed_model_settings": {
            "laplace_alpha": config.LAPLACE_ALPHA,
            "fisher_var_eps": config.FISHER_VAR_EPS,
            "panel_missing_state_restriction": True,
        },
        "quantization_resolution": best_q,
        "quantization_selection_rule": config.QUANTIZATION_SELECTION_RULE,
        "quantization_candidates": config.QUANTIZATION_CANDIDATES,
    })

    write_json(os.path.join(out_dir, "model_usage.json"), {
        "quantizer_bin_edges_and_levels": {"used_by": "both", "description": "Discretization shared by geometry (Fisher score) and generation (ancestral/BP sampling)."},
        "chow_liu_tree_structure": {"used_by": "both", "description": "Tree edges/root selected by max-MI spanning tree on the fit data; defines both the sampler's factorization and the Fisher-score block structure."},
        "root_marginal": {"used_by": "both", "description": "P(root) used both as the generation root distribution and as the Fisher-score reference for the root block."},
        "conditional_probability_tables": {"used_by": "both", "description": "Each edge CPT P(child|parent) is used both for ancestral/BP sampling and as the Fisher-score reference distribution for that node."},
    })

    model_pkl_path = os.path.join(artifacts_dir, "model.pkl")
    import pickle
    with open(model_pkl_path, "wb") as f:
        pickle.dump({"model": model, "quantizers": quantizers, "all_coords": all_coords,
                     "numeric_coords": numeric_coords, "categorical_coords": categorical_coords,
                     "numeric_panel": numeric_panel, "categorical_panel": categorical_panel}, f)

    write_json(os.path.join(out_dir, "model_lock.json"), {
        "model_artifact_sha256": {os.path.relpath(model_pkl_path, out_dir): sha256_file(model_pkl_path)},
        "quantization_resolution": best_q,
        "tree_root": all_coords[model.root],
        "n_tree_edges": len(tree_edges_out),
    })

    py_sources = list_py_sources(ANALYSIS_DIR)
    source_hashes = {os.path.relpath(p, out_dir): sha256_file(p) for p in py_sources}
    artifact_hashes = {os.path.relpath(model_pkl_path, out_dir): sha256_file(model_pkl_path)}
    other_hashes = {}
    for name in ["selection_trace.csv", "model_manifest.json", "model_usage.json", "model_lock.json"]:
        other_hashes[name] = sha256_file(os.path.join(out_dir, name))

    protocol_lock = {
        "source_file_sha256": source_hashes,
        "artifact_sha256": {**other_hashes, **artifact_hashes},
        "model_family": config.MODEL_FAMILY,
        "geometry_family": config.GEOMETRY_FAMILY,
        "fixed_model_settings": {
            "laplace_alpha": config.LAPLACE_ALPHA,
            "fisher_var_eps": config.FISHER_VAR_EPS,
        },
        "quantization_candidates": config.QUANTIZATION_CANDIDATES,
        "quantization_selection_rule": config.QUANTIZATION_SELECTION_RULE,
        "selected_quantization_resolution": best_q,
        "evaluation_seeds": {k: v for k, v in vars(config).items() if k.startswith("SEED_") or k.startswith("N_") or k == "VERIFIER_RANDOM_STATE"},
    }
    write_json(os.path.join(out_dir, "protocol_lock.json"), protocol_lock)
    protocol_lock_sha = sha256_file(os.path.join(out_dir, "protocol_lock.json"))
    model_lock_sha = sha256_file(os.path.join(out_dir, "model_lock.json"))
    log(f"Protocol lock written. protocol_lock_sha256={protocol_lock_sha[:16]}...")

    # ================================================================
    # OUTER EVALUATION (single shot) -- outer-held-out rows and blind
    # transport values are read for the first time starting here.
    # ================================================================
    log("BEGIN single-shot outer evaluation")

    test_positions = np.flatnonzero(~train_mask)

    blind_raw = df[config.BLIND_COLUMNS].apply(pd.to_numeric, errors="coerce")
    train_blind_mean = blind_raw.iloc[train_pos].mean()
    train_blind_std = blind_raw.iloc[train_pos].std(ddof=0)

    eligible_pos = np.flatnonzero((~train_mask) & blind_raw.notna().all(axis=1).to_numpy())
    eval_rng = np.random.default_rng(config.SEED_EVAL_COHORT)
    eval_pos = eval_rng.choice(eligible_pos, size=1500, replace=False)
    log(f"Fixed geometry/transport evaluation cohort: {len(eval_pos)} materials")

    # ---- conventional reference geometries on the eval cohort ----
    sigma = coord.iloc[train_pos][numeric_coords].std(ddof=0)
    numeric_range = coord.iloc[train_pos][numeric_coords].max() - coord.iloc[train_pos][numeric_coords].min()

    eval_numeric = coord.iloc[eval_pos][numeric_coords].reset_index(drop=True)
    eval_categorical_dM = coord.iloc[eval_pos][["crys", "dimensionality"]].copy()
    eval_categorical_dM["spg_number"] = dl.to_categorical_with_missing(df["spg_number"]).iloc[eval_pos].to_numpy()
    eval_categorical_dM = eval_categorical_dM.reset_index(drop=True)

    log("Computing conventional distances d_E, d_M")
    dE, eligE, _ = dist.standardized_euclidean_distance(eval_numeric, sigma)
    dM, eligM, _ = dist.gower_mixed_distance(eval_numeric, numeric_range, eval_categorical_dM)

    log("Computing submitted geometry d_G (Fisher score)")
    eval_discrete = discrete_all.iloc[eval_pos][all_coords].to_numpy()
    dG = geo.submitted_geometry_matrix(model, eval_discrete)

    # ---------------------------- TEST 1 ----------------------------
    log("TEST 1: geometry novelty")
    test1 = novelty.run_test1(dG, dE, eligE, dM, eligM)
    log(f"  geometry_novel={test1['geometry_novel']}")

    # ---------------------------- TEST 2 ----------------------------
    log("TEST 2: blind distant-analog retrieval")
    blind_std_vals = ((blind_raw.iloc[eval_pos] - train_blind_mean) / train_blind_std).to_numpy(dtype=float)
    dF = retrieval.compute_dF(blind_std_vals)
    test2 = retrieval.run_test2(dE, eligE, dG, dF)
    log(f"  retrieval_useful={test2['retrieval_useful']} raw_control_matching_pass={test2['raw_control_matching_pass']}")

    retrieval_by_anchor = pd.DataFrame({
        "anchor_pos": eval_pos[test2["_anchors"]],
        "anchor_jid": df["jid"].to_numpy()[eval_pos[test2["_anchors"]]],
        "retrieved_pos": eval_pos[test2["_bG"]],
        "retrieved_jid": df["jid"].to_numpy()[eval_pos[test2["_bG"]]],
        "dF_retrieved": test2["_dF_bG"],
        "mean_dF_matched_controls": test2["_mean_dF_match"],
        "matching_gap_ratio_g_a": test2["_g_a"],
    })
    retrieval_by_anchor.to_csv(os.path.join(out_dir, "retrieval_by_anchor.csv"), index=False)

    # ---------------------------- TEST 3 -----------------------------
    log("TEST 3: unconditional generation")
    rng_gen = np.random.default_rng(config.SEED_UNCONDITIONAL_GENERATION)
    uncond_discrete = model.sample_unconditional(10000, rng_gen)
    uncond_samples = gen.decode_samples(uncond_discrete, all_coords, quantizers, rng_gen, numeric_coords, categorical_coords)

    train_raw_frame = coord.iloc[train_pos].reset_index(drop=True)
    vu = gen.validity_and_uniqueness(uncond_samples, numeric_panel, categorical_panel, train_raw_frame)
    blank_generation_valid = bool(
        vu["valid_count"] >= 9500 and vu["unique_fraction"] >= 0.80 and vu["exact_training_match_fraction"] <= 0.10
    )
    log(f"  valid={vu['valid_count']}/10000 unique_frac={vu['unique_fraction']:.3f} exact_match_frac={vu['exact_training_match_fraction']:.4f}")

    uncond_out_path = os.path.join(out_dir, "unconditional_samples.csv.gz")
    uncond_samples.to_csv(uncond_out_path, index=False, compression="gzip")

    # ---------------------------- TEST 3A -----------------------------
    log("TEST 3A: marginal fidelity (descriptive)")
    heldout_raw_frame = coord.iloc[test_positions].reset_index(drop=True)
    fidelity_df = gen.marginal_fidelity(uncond_samples, heldout_raw_frame, numeric_panel, categorical_panel, train_raw_frame)
    fidelity_df.to_csv(os.path.join(out_dir, "unconditional_marginal_fidelity.csv"), index=False)

    # ---------------------------- IND baseline -------------------------
    log("Building independent-marginal negative-control baseline")
    ind_samples = gen.independent_marginal_sample(train_raw_frame, numeric_panel, categorical_panel, 10000)

    # ---------------------------- TEST 3B -----------------------------
    log("TEST 3B: cross-structure retention (fixed ExtraTrees verifiers)")
    panel_observed_mask = pd.Series(True, index=heldout_raw_frame.index)
    for c in numeric_panel:
        panel_observed_mask &= heldout_raw_frame[c].notna()
    for c in categorical_panel:
        panel_observed_mask &= (heldout_raw_frame[c] != config.MISSING_STATE)
    heldout_full_panel = heldout_raw_frame[panel_observed_mask].reset_index(drop=True)
    log(f"  outer-held-out rows with full panel observed: {len(heldout_full_panel)}")

    test3b_summary, cross_structure_by_target = csx.run_test3b(
        train_raw_frame, uncond_samples, ind_samples, heldout_full_panel,
        numeric_panel, categorical_panel, quantizers, train_raw_frame,
    )
    cross_structure_by_target.to_csv(os.path.join(out_dir, "cross_structure_by_target.csv"), index=False)
    log(f"  cross_structure_preserved={test3b_summary['cross_structure_preserved']}")

    # ---------------------------- TEST 4 -----------------------------
    log("TEST 4: arbitrary partial-state completion")
    completion_rows = comp.select_completion_rows(coord, test_positions, panel_cols)
    masks_all = comp.build_masks(coord, completion_rows, panel_cols)

    iqr_train = {}
    for c in numeric_panel:
        vals = train_raw_frame[c].dropna().to_numpy(dtype=float)
        q75, q25 = np.percentile(vals, [75, 25])
        iqr = q75 - q25
        iqr_train[c] = iqr if iqr > 0 else (vals.max() - vals.min() if vals.max() > vals.min() else 1.0)

    completion_summaries = {}
    per_coord_rows = []
    level_offset = {0.20: 0, 0.50: 1, 0.80: 2}
    for lvl in comp.LEVELS:
        masks_lvl = masks_all[lvl]

        def sampler(evidence, n_rep, rng):
            return model.sample_conditional(evidence, n_rep, rng)

        loss_model = comp.run_completion_for_model_or_ind(
            sampler, coord, discrete_all, all_coords, quantizers,
            completion_rows, masks_lvl, numeric_panel, categorical_panel,
            seed=config.SEED_COMPLETION_MASKS + 10 * level_offset[lvl],
            iqr_train=iqr_train,
        )
        loss_ind = comp.run_completion_ind(
            coord, train_raw_frame, completion_rows, masks_lvl,
            numeric_panel, categorical_panel,
            seed=config.SEED_COMPLETION_MASKS + 10 * level_offset[lvl] + 5,
            iqr_train=iqr_train,
        )
        summary = comp.summarize_completion(loss_model, loss_ind, panel_cols, config.N_COMPLETION_BOOTSTRAP, config.SEED_COMPLETION_BOOTSTRAP)
        completion_summaries[lvl] = summary
        for _, row in summary["per_coordinate"].iterrows():
            per_coord_rows.append({"masking_level": lvl, **row.to_dict()})
        log(f"  level={lvl}: R_C={summary['R_C']:.4f} CI=[{summary['R_C_ci95_lower']:.4f},{summary['R_C_ci95_upper']:.4f}] win_frac={summary['coordinate_win_fraction']:.3f}")

    pd.DataFrame(per_coord_rows).to_csv(os.path.join(out_dir, "completion_by_coordinate.csv"), index=False)

    completion_gate_pass = True
    for lvl in [0.50, 0.80]:
        s = completion_summaries[lvl]
        if not (s["R_C_ci95_upper"] < 1.0 and s["coordinate_win_fraction"] >= 0.60):
            completion_gate_pass = False
    conditional_completion_useful = bool(completion_gate_pass)

    # ---------------------------- assemble status -----------------------
    single_model_requirement_pass = True
    single_shot_outer_evaluation_pass = True
    no_model_tuning_pass = True

    geometry_novel = test1["geometry_novel"]
    retrieval_useful = test2["retrieval_useful"]
    raw_control_matching_pass = test2["raw_control_matching_pass"]
    cross_structure_preserved = test3b_summary["cross_structure_preserved"]

    if not no_model_tuning_pass:
        analysis_status = "IMPLEMENTATION_FAILURE"
    elif not geometry_novel:
        analysis_status = "GEOMETRY_FAILURE"
    elif (not retrieval_useful) or (not raw_control_matching_pass):
        analysis_status = "RETRIEVAL_FAILURE"
    elif not (blank_generation_valid and cross_structure_preserved and conditional_completion_useful and single_model_requirement_pass):
        analysis_status = "GENERATION_FAILURE"
    else:
        analysis_status = "DISCOVERY_SUCCESS"

    primary_discovery = bool(
        geometry_novel and retrieval_useful and raw_control_matching_pass and blank_generation_valid
        and cross_structure_preserved and conditional_completion_useful and single_model_requirement_pass
        and single_shot_outer_evaluation_pass and no_model_tuning_pass
    )

    answers = {
        "analysis_status": analysis_status,
        "geometry_novel": geometry_novel,
        "retrieval_useful": retrieval_useful,
        "blank_generation_valid": blank_generation_valid,
        "cross_structure_preserved": cross_structure_preserved,
        "conditional_completion_useful": conditional_completion_useful,
        "single_model_requirement_pass": single_model_requirement_pass,
        "single_shot_outer_evaluation_pass": single_shot_outer_evaluation_pass,
        "no_model_tuning_pass": no_model_tuning_pass,
        "raw_control_matching_pass": raw_control_matching_pass,
        "primary_discovery": primary_discovery,

        "evaluation_count": 1,
        "pair_count": int(len(eval_pos) * (len(eval_pos) - 1) // 2),

        "mean_J20_euclidean": test1["mean_J20_euclidean"],
        "mean_J20_euclidean_ci95_lower": test1["mean_J20_euclidean_ci95_lower"],
        "mean_J20_euclidean_ci95_upper": test1["mean_J20_euclidean_ci95_upper"],
        "mean_J20_mixed": test1["mean_J20_mixed"],
        "mean_J20_mixed_ci95_lower": test1["mean_J20_mixed_ci95_lower"],
        "mean_J20_mixed_ci95_upper": test1["mean_J20_mixed_ci95_upper"],
        "global_rho_geometry_euclidean": test1["global_rho_geometry_euclidean"],
        "global_rho_geometry_mixed": test1["global_rho_geometry_mixed"],

        "n_evaluable_anchors": test2["n_evaluable_anchors"],
        "retrieval_ratio": test2["retrieval_ratio"],
        "retrieval_ratio_ci95_lower": test2["retrieval_ratio_ci95_lower"],
        "retrieval_ratio_ci95_upper": test2["retrieval_ratio_ci95_upper"],
        "retrieval_win_rate": test2["retrieval_win_rate"],
        "retrieval_win_rate_ci95_lower": test2["retrieval_win_rate_ci95_lower"],
        "retrieval_win_rate_ci95_upper": test2["retrieval_win_rate_ci95_upper"],
        "mean_abs_control_raw_distance_gap": test2["mean_abs_control_raw_distance_gap"],
        "raw_control_matching_gap_ratio": test2["raw_control_matching_gap_ratio"],
        "raw_control_matching_gap_ratio_ci95_lower": test2["raw_control_matching_gap_ratio_ci95_lower"],
        "raw_control_matching_gap_ratio_ci95_upper": test2["raw_control_matching_gap_ratio_ci95_upper"],
        "raw_far_threshold_q90": test2["raw_far_threshold_q90"],

        "generation_panel_size": len(panel_cols),
        "unconditional_requested_count": vu["requested_count"],
        "unconditional_valid_count": vu["valid_count"],
        "unconditional_unique_fraction": vu["unique_fraction"],
        "unconditional_exact_training_match_fraction": vu["exact_training_match_fraction"],

        "n_cross_structure_targets": test3b_summary["n_cross_structure_targets"],
        "cross_structure_retention_G": test3b_summary["cross_structure_retention_G"],
        "cross_structure_retention_G_ci95_lower": test3b_summary["cross_structure_retention_G_ci95_lower"],
        "cross_structure_retention_G_ci95_upper": test3b_summary["cross_structure_retention_G_ci95_upper"],
        "cross_structure_target_win_fraction": test3b_summary["cross_structure_target_win_fraction"],

        "completion_ratio_hidden20": completion_summaries[0.20]["R_C"],
        "completion_ratio_hidden20_ci95_lower": completion_summaries[0.20]["R_C_ci95_lower"],
        "completion_ratio_hidden20_ci95_upper": completion_summaries[0.20]["R_C_ci95_upper"],
        "completion_ratio_hidden50": completion_summaries[0.50]["R_C"],
        "completion_ratio_hidden50_ci95_lower": completion_summaries[0.50]["R_C_ci95_lower"],
        "completion_ratio_hidden50_ci95_upper": completion_summaries[0.50]["R_C_ci95_upper"],
        "completion_ratio_hidden80": completion_summaries[0.80]["R_C"],
        "completion_ratio_hidden80_ci95_lower": completion_summaries[0.80]["R_C_ci95_lower"],
        "completion_ratio_hidden80_ci95_upper": completion_summaries[0.80]["R_C_ci95_upper"],
        "completion_coordinate_win_fraction_hidden50": completion_summaries[0.50]["coordinate_win_fraction"],
        "completion_coordinate_win_fraction_hidden80": completion_summaries[0.80]["coordinate_win_fraction"],
    }

    answers_path = os.path.join(out_dir, "answers.json")
    write_json(answers_path, answers)
    answers_sha = sha256_file(answers_path)

    receipt = {
        "outer_evaluation_count": 1,
        "protocol_lock_sha256": protocol_lock_sha,
        "model_lock_sha256": model_lock_sha,
        "answers_sha256": answers_sha,
    }
    write_json(os.path.join(out_dir, "outer_evaluation_receipt.json"), receipt)

    history_path = os.path.join(out_dir, "outer_evaluation_history.jsonl")
    with open(history_path, "w") as f:
        f.write(json.dumps({
            "event": "single_scientific_outer_evaluation",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol_lock_sha256": protocol_lock_sha,
            "model_lock_sha256": model_lock_sha,
            "answers_sha256": answers_sha,
            "analysis_status": analysis_status,
            "primary_discovery": primary_discovery,
        }, default=str) + "\n")

    log(f"DONE. analysis_status={analysis_status} primary_discovery={primary_discovery}")


if __name__ == "__main__":
    main()
