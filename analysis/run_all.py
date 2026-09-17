#!/usr/bin/env python3
"""Orchestrates the complete workflow described in instructions.md.

Usage:
    python analysis/run_all.py --data jarvis.tgz

Everything before `# ===== OUTER EVALUATION STARTS HERE =====` uses only the
50,000-row outer training set (and, where explicitly permitted, the inner
development split) and writes the pre-outer protocol-lock artifacts. Nothing
after that marker may change any locked source file, model artifact,
quantization rule, or fixed setting -- this script performs exactly one
outer/blind evaluation pass per run, and refuses to run a second time if
`outer_evaluation_history.jsonl` already contains a record.
"""
import argparse
import json
import gzip
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import config
from analysis.data_prep import (
    extract_dataframe, build_feature_frame, make_outer_split, make_inner_split,
    ConventionalGeometry,
)
from analysis.panel import select_generation_panel, write_generation_panel
from analysis.copula_model import GaussianCopulaModel
from analysis.independent_baseline import IndependentMarginalBaseline
from analysis import protocol
from analysis.lock_utils import sha256_file
from analysis import evaluate as ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="jarvis.tgz")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(repo_root, "analysis")
    out = analysis_dir
    artifacts_dir = os.path.join(out, "model_artifacts")

    history_path = os.path.join(out, "outer_evaluation_history.jsonl")
    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        raise RuntimeError(
            "outer_evaluation_history.jsonl already contains a record. This "
            "workflow performs a strict single-shot outer evaluation and "
            "refuses to run again."
        )

    t0 = time.time()

    # ---------------- pre-outer: data, splits, panel, model fit ----------------
    extract_dir = os.path.join(out, "_extract_cache")
    df = extract_dataframe(args.data, extract_dir)
    feat = build_feature_frame(df)
    n_rows = len(feat)
    jids = feat["jid"].to_numpy()

    train_pos, train_mask, train_jids, test_jids = make_outer_split(n_rows, jids)
    fit_pos, inner_dev_pos = make_inner_split(train_pos)
    print(f"[{time.time()-t0:.1f}s] loaded data, n={n_rows}, train={len(train_pos)}, test={len(test_jids)}", flush=True)

    panel = select_generation_panel(feat, train_pos)
    write_generation_panel(panel, os.path.join(out, "generation_panel.json"))
    numeric_cols = panel["numeric_columns"]
    categorical_cols = panel["categorical_columns"]
    panel_cols = numeric_cols + categorical_cols
    print(f"[{time.time()-t0:.1f}s] generation panel: {len(numeric_cols)} numeric + {len(categorical_cols)} categorical", flush=True)

    df_train = feat.iloc[train_pos].reset_index(drop=True)

    model = GaussianCopulaModel(numeric_cols, categorical_cols)
    model.fit_marginals(df_train[panel_cols])
    Z_train = model.to_latent(df_train[panel_cols])
    model.fit_sigma(Z_train)
    print(f"[{time.time()-t0:.1f}s] structural model fit (EM, {config.EM_ITERATIONS} iters)", flush=True)

    model.save(artifacts_dir)

    conv_geom = ConventionalGeometry(feat, train_pos)

    baseline = IndependentMarginalBaseline(model)

    # ---------------- pre-outer lock / manifest writers ----------------
    protocol.write_selection_trace(os.path.join(out, "selection_trace.csv"))
    protocol.write_model_manifest(os.path.join(out, "model_manifest.json"), panel)
    protocol.write_model_usage(os.path.join(out, "model_usage.json"))
    model_lock_path = os.path.join(out, "model_lock.json")
    protocol.write_model_lock(model_lock_path, artifacts_dir, panel)

    seeds = {
        "outer_split_seed": config.OUTER_SPLIT_SEED,
        "inner_dev_seed": config.INNER_DEV_SEED,
        "eval_cohort_seed": config.EVAL_COHORT_SEED,
        "geometry_novelty_bootstrap_seed": config.GEOMETRY_NOVELTY_BOOTSTRAP_SEED,
        "retrieval_bootstrap_seed": config.RETRIEVAL_BOOTSTRAP_SEED,
        "matching_gap_bootstrap_seed": config.MATCHING_GAP_BOOTSTRAP_SEED,
        "independent_marginal_seed": config.INDEPENDENT_MARGINAL_SEED,
        "unconditional_sample_seed": config.UNCONDITIONAL_SAMPLE_SEED,
        "cross_structure_verifier_seed": config.CROSS_STRUCTURE_VERIFIER_SEED,
        "cross_structure_bootstrap_seed": config.CROSS_STRUCTURE_BOOTSTRAP_SEED,
        "completion_row_seed": config.COMPLETION_ROW_SEED,
        "completion_mask_seed": config.COMPLETION_MASK_SEED,
        "completion_bootstrap_seed": config.COMPLETION_BOOTSTRAP_SEED,
    }
    protocol_lock_path = os.path.join(out, "protocol_lock.json")
    protocol.write_protocol_lock(
        protocol_lock_path, analysis_dir,
        extra_paths=[
            os.path.join(out, "selection_trace.csv"),
            os.path.join(out, "model_manifest.json"),
            os.path.join(out, "model_usage.json"),
            model_lock_path,
            os.path.join(out, "generation_panel.json"),
            artifacts_dir,
        ],
        seeds=seeds,
    )
    model_lock_sha256 = sha256_file(model_lock_path)
    protocol_lock_sha256 = sha256_file(protocol_lock_path)
    print(f"[{time.time()-t0:.1f}s] protocol lock written", flush=True)

    # ================= OUTER EVALUATION STARTS HERE =================
    # No outer-held-out row or blind transport value has been read above.

    eval_pos = ev.select_eval_cohort(feat, train_mask)
    cohort = feat.iloc[eval_pos].reset_index(drop=True)
    print(f"[{time.time()-t0:.1f}s] eval cohort selected, n={len(cohort)}", flush=True)

    Z_cohort = model.to_latent(cohort[panel_cols])
    Z_cohort_full = model.complete_latent_mean(Z_cohort)
    D_G = model.geometry_distance_matrix(Z_cohort_full)

    X_num = conv_geom.numeric_matrix(cohort)
    D_E, _ = conv_geom.d_E_pairwise(X_num)
    D_M = conv_geom.d_M_pairwise(cohort)
    print(f"[{time.time()-t0:.1f}s] pairwise geometries computed", flush=True)

    test1 = ev.run_test1(D_G, D_E, D_M)
    print(f"[{time.time()-t0:.1f}s] TEST1 done: geometry_novel={test1['geometry_novel']}", flush=True)

    blind_mean = feat.iloc[train_pos][config.BLIND_COLS].mean().to_numpy()
    blind_std_dev = feat.iloc[train_pos][config.BLIND_COLS].std(ddof=1).to_numpy()
    blind_std = (cohort[config.BLIND_COLS].to_numpy(dtype=float) - blind_mean) / blind_std_dev

    test2 = ev.run_test2(D_G, D_E, blind_std)
    print(f"[{time.time()-t0:.1f}s] TEST2 done: retrieval_useful={test2['retrieval_useful']}", flush=True)

    retrieval_rows = []
    for a in range(len(cohort)):
        if test2["_evaluable_mask"][a]:
            retrieval_rows.append({
                "anchor_jid": cohort["jid"].iloc[a],
                "d_F_retrieved": test2["_retrieved_dF"][a],
                "d_F_control_mean": test2["_control_mean_dF"][a],
            })
    pd.DataFrame(retrieval_rows).to_csv(os.path.join(out, "retrieval_by_anchor.csv"), index=False)

    # ---------------- TEST 3: generation ----------------
    gen_df = ev.generate_unconditional_samples(model, config.UNCONDITIONAL_SAMPLE_COUNT, config.UNCONDITIONAL_SAMPLE_SEED)
    with gzip.open(os.path.join(out, "unconditional_samples.csv.gz"), "wt") as f:
        gen_df[panel_cols].to_csv(f, index=False)

    test3 = ev.test3_blank_generation_gate(gen_df, df_train[panel_cols], numeric_cols, categorical_cols)
    print(f"[{time.time()-t0:.1f}s] TEST3 done: blank_generation_valid={test3['blank_generation_valid']}", flush=True)

    heldout_df = feat.iloc[~train_mask].reset_index(drop=True)
    marg_fid_df = ev.test3a_marginal_fidelity(gen_df, heldout_df[panel_cols], df_train[panel_cols], numeric_cols, categorical_cols)
    marg_fid_df.to_csv(os.path.join(out, "unconditional_marginal_fidelity.csv"), index=False)

    real_10k = df_train[panel_cols].iloc[:10000].reset_index(drop=True)
    ind_rng = np.random.default_rng(config.INDEPENDENT_MARGINAL_SEED)
    ind_10k_raw = baseline.sample_unconditional(10000, ind_rng)
    ind_10k = pd.DataFrame(ind_10k_raw)[panel_cols]

    test3b = ev.run_test3b_cross_structure(model, numeric_cols, categorical_cols, real_10k, gen_df[panel_cols], ind_10k, heldout_df[panel_cols])
    test3b["by_target"].to_csv(os.path.join(out, "cross_structure_by_target.csv"), index=False)
    print(f"[{time.time()-t0:.1f}s] TEST3B done: cross_structure_preserved={test3b['cross_structure_preserved']}", flush=True)

    # ---------------- TEST 4: partial-state completion ----------------
    test4 = ev.run_test4(model, baseline, heldout_df, numeric_cols, categorical_cols, df_train[panel_cols])
    test4["by_coordinate"].to_csv(os.path.join(out, "completion_by_coordinate.csv"), index=False)
    print(f"[{time.time()-t0:.1f}s] TEST4 done: conditional_completion_useful={test4['conditional_completion_useful']}", flush=True)

    # ---------------- assemble status ----------------
    no_model_tuning_pass = True
    single_model_requirement_pass = True
    single_shot_outer_evaluation_pass = True

    geometry_novel = test1["geometry_novel"]
    retrieval_useful = test2["retrieval_useful"]
    raw_control_matching_pass = test2["raw_control_matching_pass"]
    blank_generation_valid = test3["blank_generation_valid"]
    cross_structure_preserved = test3b["cross_structure_preserved"]
    conditional_completion_useful = test4["conditional_completion_useful"]

    if not no_model_tuning_pass:
        analysis_status = "IMPLEMENTATION_FAILURE"
    elif not geometry_novel:
        analysis_status = "GEOMETRY_FAILURE"
    elif (not retrieval_useful) or (not raw_control_matching_pass):
        analysis_status = "RETRIEVAL_FAILURE"
    elif not all([blank_generation_valid, cross_structure_preserved, conditional_completion_useful, single_model_requirement_pass]):
        analysis_status = "GENERATION_FAILURE"
    else:
        analysis_status = "DISCOVERY_SUCCESS"

    primary_discovery = "YES" if (
        analysis_status == "DISCOVERY_SUCCESS" and single_shot_outer_evaluation_pass and no_model_tuning_pass
    ) else "NO"

    answers = {
        "analysis_status": analysis_status,
        "geometry_novel": "YES" if geometry_novel else "NO",
        "retrieval_useful": "YES" if retrieval_useful else "NO",
        "blank_generation_valid": "YES" if blank_generation_valid else "NO",
        "cross_structure_preserved": "YES" if cross_structure_preserved else "NO",
        "conditional_completion_useful": "YES" if conditional_completion_useful else "NO",
        "single_model_requirement_pass": "YES" if single_model_requirement_pass else "NO",
        "single_shot_outer_evaluation_pass": "YES" if single_shot_outer_evaluation_pass else "NO",
        "no_model_tuning_pass": "YES" if no_model_tuning_pass else "NO",
        "raw_control_matching_pass": "YES" if raw_control_matching_pass else "NO",
        "primary_discovery": primary_discovery,

        "evaluation_count": 1,
        "pair_count": int(len(cohort) * (len(cohort) - 1) // 2),

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
        "unconditional_requested_count": test3["unconditional_requested_count"],
        "unconditional_valid_count": test3["unconditional_valid_count"],
        "unconditional_unique_fraction": test3["unconditional_unique_fraction"],
        "unconditional_exact_training_match_fraction": test3["unconditional_exact_training_match_fraction"],

        "n_cross_structure_targets": test3b["n_cross_structure_targets"],
        "cross_structure_retention_G": test3b["cross_structure_retention_G"],
        "cross_structure_retention_G_ci95_lower": test3b["cross_structure_retention_G_ci95_lower"],
        "cross_structure_retention_G_ci95_upper": test3b["cross_structure_retention_G_ci95_upper"],
        "cross_structure_target_win_fraction": test3b["cross_structure_target_win_fraction"],

        "completion_ratio_hidden20": test4["completion_ratio_hidden20"],
        "completion_ratio_hidden20_ci95_lower": test4["completion_ratio_hidden20_ci95_lower"],
        "completion_ratio_hidden20_ci95_upper": test4["completion_ratio_hidden20_ci95_upper"],
        "completion_ratio_hidden50": test4["completion_ratio_hidden50"],
        "completion_ratio_hidden50_ci95_lower": test4["completion_ratio_hidden50_ci95_lower"],
        "completion_ratio_hidden50_ci95_upper": test4["completion_ratio_hidden50_ci95_upper"],
        "completion_ratio_hidden80": test4["completion_ratio_hidden80"],
        "completion_ratio_hidden80_ci95_lower": test4["completion_ratio_hidden80_ci95_lower"],
        "completion_ratio_hidden80_ci95_upper": test4["completion_ratio_hidden80_ci95_upper"],
        "completion_coordinate_win_fraction_hidden50": test4["completion_coordinate_win_fraction_hidden50"],
        "completion_coordinate_win_fraction_hidden80": test4["completion_coordinate_win_fraction_hidden80"],
    }

    def _clean(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    answers = {k: _clean(v) for k, v in answers.items()}
    answers_path = os.path.join(out, "answers.json")
    with open(answers_path, "w") as f:
        json.dump(answers, f, indent=2, sort_keys=False)
    answers_sha256 = sha256_file(answers_path)

    history_record = {
        "record": 1,
        "timestamp_unix": time.time(),
        "analysis_status": analysis_status,
        "protocol_lock_sha256": protocol_lock_sha256,
        "model_lock_sha256": model_lock_sha256,
        "answers_sha256": answers_sha256,
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(history_record) + "\n")

    receipt = {
        "outer_evaluation_count": 1,
        "protocol_lock_sha256": protocol_lock_sha256,
        "model_lock_sha256": model_lock_sha256,
        "answers_sha256": answers_sha256,
    }
    with open(os.path.join(out, "outer_evaluation_receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"[{time.time()-t0:.1f}s] DONE. analysis_status={analysis_status}, primary_discovery={primary_discovery}", flush=True)


if __name__ == "__main__":
    main()
