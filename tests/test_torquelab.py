import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from torquelab import (
    BOUNDS,
    EXPANDED_FEATURES,
    LBFT_TO_NM,
    LB_TO_KG,
    SEED,
    TARGET,
    _metrics,
    conformal_quantile,
    evaluate_models,
    explain_rf_prediction,
    group_conformal_calibration,
    input_support,
    grouped_cross_validation,
    leaderboard,
    model_error_diagnostics,
    out_of_distribution_features,
    predict_vehicle,
    prediction_frame,
    prediction_interval,
    predict_with_conformal_interval,
    prepare_dataset,
    run_analysis,
    safe_explain_rf_prediction,
    tune_vehicle,
)


def test_source_filter_keeps_only_gasoline_and_exact_car_and_driver(make_source_db, valid_source_row):
    rows = [
        valid_source_row,
        {**valid_source_row, "car_id": "diesel", "propulsion": "Diesel"},
        {**valid_source_row, "car_id": "hybrid", "propulsion": "Electric/Petrol"},
        {**valid_source_row, "car_id": "mixed", "sources": "Car & Driver, EPA"},
    ]
    data, summary = prepare_dataset(make_source_db(rows))
    assert data["car_id"].tolist() == ["valid-1"]
    assert summary["exact_source_gasoline_pool"] == 1
    assert summary["not_exact_car_and_driver_source_or_not_gasoline"] == 3


def test_incomplete_and_implausible_rows_are_excluded(make_source_db, valid_source_row):
    rows = [
        valid_source_row,
        {**valid_source_row, "car_id": "no-weight", "curb_weight_lb": None},
        {**valid_source_row, "car_id": "bad-torque-unit", "torque": "271 Nm"},
        {**valid_source_row, "car_id": "oversized-hp", "power_hp": 2501},
        {**valid_source_row, "car_id": "bad-drive", "drivetrain": "UNKNOWN"},
        {**valid_source_row, "car_id": "too-fast", "acceleration": 0.4},
    ]
    data, summary = prepare_dataset(make_source_db(rows))
    assert data["car_id"].tolist() == ["valid-1"]
    assert summary["incomplete_required_fields"] == 2
    assert summary["implausible_or_malformed_values"] == 3


def test_unit_conversions_and_engineered_ratios(make_source_db, valid_source_row):
    data, _ = prepare_dataset(make_source_db([valid_source_row]))
    row = data.iloc[0]
    expected_weight = 3000 * LB_TO_KG
    expected_torque = 200 * LBFT_TO_NM
    assert row["weight_kg"] == pytest.approx(expected_weight)
    assert row["torque_nm"] == pytest.approx(expected_torque)
    assert row["power_hp"] == pytest.approx(300)
    assert row["power_hp_per_1000kg"] == pytest.approx(300 / (expected_weight / 1000))
    assert row["torque_nm_per_1000kg"] == pytest.approx(expected_torque / (expected_weight / 1000))


def test_prediction_feature_builder_rejects_nonphysical_inputs():
    row = prediction_frame(300, 271.16, 1360.78, "rwd")
    assert row.loc[0, "drivetrain"] == "RWD"
    assert row.loc[0, "power_hp_per_1000kg"] > 0
    with pytest.raises(ValueError, match="physical feature bounds"):
        prediction_frame(300, 271, 100, "RWD")
    with pytest.raises(ValueError, match="Unsupported drivetrain"):
        prediction_frame(300, 271, 1360, "2WD")


def test_grouped_holdout_has_no_shared_nameplate_groups(synthetic_training_data):
    data = synthetic_training_data
    result = evaluate_models(data)
    split = result["split"]
    train_index, test_index = next(GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=SEED
    ).split(data, data[TARGET], groups=data["group_key"]))
    train_groups = set(data.iloc[train_index]["group_key"])
    test_groups = set(data.iloc[test_index]["group_key"])
    assert train_groups.isdisjoint(test_groups)
    assert split["overlapping_groups"] == 0
    assert split["train_rows"] + split["test_rows"] == len(data)
    # The scaler is fitted on training rows only; holdout values cannot leak
    # into preprocessing statistics.
    scaler = result["models"]["Horsepower baseline"].named_steps["preprocess"].named_transformers_["numeric"]
    assert scaler.mean_[0] == pytest.approx(data.iloc[train_index]["power_hp"].mean())
    bounds = result["input_support"]["numeric"]["power_hp"]
    assert bounds == pytest.approx([
        data.iloc[train_index]["power_hp"].quantile(0.05),
        data.iloc[train_index]["power_hp"].quantile(0.95),
    ])


def test_metrics_match_manual_values():
    scores = _metrics(np.array([3.0, 4.0]), np.array([2.0, 6.0]))
    assert scores["MAE (s)"] == pytest.approx(1.5)
    assert scores["RMSE (s)"] == pytest.approx(np.sqrt(2.5))
    assert scores["R²"] == pytest.approx(-9.0)


def test_model_serialization_and_inference_are_finite_and_plausible(synthetic_training_data, tmp_path):
    result = evaluate_models(synthetic_training_data)
    artifact = tmp_path / "models.joblib"
    joblib.dump(result["models"], artifact)
    loaded = joblib.load(artifact)
    sample = synthetic_training_data.iloc[0]
    estimates = predict_vehicle(
        loaded,
        sample["power_hp"], sample["torque_nm"], sample["weight_kg"], sample["drivetrain"],
    )
    assert set(estimates) == {"Horsepower baseline", "Expanded linear regression", "Random forest"}
    assert all(np.isfinite(value) and BOUNDS[TARGET][0] <= value <= BOUNDS[TARGET][1]
               for value in estimates.values())
    assert set(EXPANDED_FEATURES).issubset(prediction_frame(300, 300, 1400, "AWD").columns)


def test_out_of_distribution_inputs_are_flagged(synthetic_training_data):
    row = prediction_frame(2500, 300, 1400, "AWD")
    outside = out_of_distribution_features(input_support(synthetic_training_data), row)
    assert "power_hp" in outside


def test_tuner_converts_imperial_weight_recalculates_features_and_predicts_both_specs(synthetic_training_data):
    fitted = evaluate_models(synthetic_training_data)
    vehicle = synthetic_training_data.iloc[0]
    changed_hp = float(vehicle["power_hp"] + 25)
    changed_weight_lb = float(vehicle["weight_kg"] / LB_TO_KG - 100)
    result = tune_vehicle(fitted["models"], fitted["input_support"], vehicle, changed_hp, changed_weight_lb)
    modified_weight_kg = changed_weight_lb * LB_TO_KG
    assert result["modified_row"].iloc[0]["weight_kg"] == pytest.approx(modified_weight_kg)
    assert result["modified_row"].iloc[0]["power_hp_per_1000kg"] == pytest.approx(changed_hp / (modified_weight_kg / 1000))
    direct_original = fitted["models"]["Random forest"].predict(result["original_row"][EXPANDED_FEATURES])[0]
    direct_modified = fitted["models"]["Random forest"].predict(result["modified_row"][EXPANDED_FEATURES])[0]
    assert result["original_prediction_s"] == pytest.approx(direct_original)
    assert result["modified_prediction_s"] == pytest.approx(direct_modified)
    assert result["change_s"] == pytest.approx(direct_modified - direct_original)
    outside = tune_vehicle(fitted["models"], fitted["input_support"], vehicle, 2500, 10000)
    assert "power_hp" in outside["outside_features"]
    assert "weight_kg" in outside["outside_features"]


def test_leaderboard_ranking_and_filters(synthetic_training_data):
    data = synthetic_training_data
    fastest = leaderboard(data, "Fastest measured 0–60 mph", "Make0", "FWD")
    assert not fastest.empty
    assert fastest["manufacturer"].eq("Make0").all()
    assert fastest["drivetrain"].eq("FWD").all()
    assert fastest[TARGET].is_monotonic_increasing
    horsepower = leaderboard(data, "Highest horsepower")
    assert horsepower["power_hp"].is_monotonic_decreasing
    ratio = leaderboard(data, "Best horsepower-to-weight")
    ratios = ratio["power_hp"] / (ratio["weight_kg"] / 1000)
    assert ratios.is_monotonic_decreasing


def test_heldout_diagnostic_residuals_and_subgroup_mae():
    predictions = pd.DataFrame({
        "car_id": ["a", "b", "c", "d"], "manufacturer": ["M"] * 4,
        "model": ["A", "B", "C", "D"], "year": ["2020"] * 4,
        "drivetrain": ["FWD", "FWD", "RWD", "AWD"],
        "power_hp": [120, 200, 300, 500], TARGET: [4, 5, 6, 7],
        "Random forest": [5, 4, 8, 6],
    })
    result = model_error_diagnostics(predictions)
    assert result["residuals"].tolist() == [1, -1, 2, -1]
    fwd_mae = result["by_drivetrain"].set_index("drivetrain").loc["FWD", "MAE (s)"]
    assert fwd_mae == pytest.approx(1)
    hp_mae = result["by_horsepower"].set_index("horsepower_range").loc["≤150 hp", "MAE (s)"]
    assert hp_mae == pytest.approx(1)
    assert len(result["largest_errors"]) == 4


def test_five_fold_grouped_cv_uses_training_partition_and_preserves_holdout(synthetic_training_data):
    folds, result = grouped_cross_validation(synthetic_training_data)
    meta = result["metadata"]
    assert len(folds) == 15
    assert meta["training_rows"] + meta["holdout_rows"] == len(synthetic_training_data)
    assert meta["holdout_rows"] > 0
    assert meta["train_holdout_group_overlap"] == 0
    assert meta["cv_holdout_id_overlap"] == 0
    assert meta["cv_validation_rows"] == meta["training_rows"]
    assert meta["fold_group_overlaps"] == [0] * 5
    assert set(folds["Model"]) == {"Horsepower baseline", "Expanded linear regression", "Random forest"}


def test_finite_sample_conformal_quantile_and_interval_bounds():
    q, rank = conformal_quantile([1, 2, 3, 4, 5, 6, 7, 8, 9], .90)
    assert rank == 9
    assert q == 9
    assert conformal_quantile([1, 2, 3, 4, 5, 6, 7, 8], .90) == (float("inf"), 9)
    assert prediction_interval(1.2, 2.0) == (0.0, 3.2)
    with pytest.raises(ValueError, match="between zero and one"):
        conformal_quantile([1, 2], 1.0)


def test_group_conformal_fit_calibration_and_holdout_are_disjoint_and_reproducible(synthetic_training_data):
    # Increase the number of independent nameplate groups so a finite 90% group quantile is available.
    frames = []
    for replicate in range(5):
        copy = synthetic_training_data.copy()
        copy["car_id"] = copy["car_id"].map(lambda value: f"r{replicate}-{value}")
        copy["group_key"] = copy["group_key"].map(lambda value: f"r{replicate}-{value}")
        frames.append(copy)
    data = pd.concat(frames, ignore_index=True)
    first = group_conformal_calibration(data, .90)
    second = group_conformal_calibration(data, .90)
    partitions = first["partition_ids"]
    assert partitions["fit"].isdisjoint(partitions["calibration"])
    assert partitions["fit"].isdisjoint(partitions["holdout"])
    assert partitions["calibration"].isdisjoint(partitions["holdout"])
    assert partitions["fit_groups"].isdisjoint(partitions["calibration_groups"])
    assert partitions["calibration_groups"].isdisjoint(partitions["holdout_groups"])
    assert first["metadata"]["quantile_radius_s"] == pytest.approx(second["metadata"]["quantile_radius_s"])
    assert first["metadata"]["holdout_empirical_coverage"] == pytest.approx(second["metadata"]["holdout_empirical_coverage"])
    assert first["metadata"]["calibration_residual_count"] == len(first["absolute_residuals"])
    intervals = first["holdout_intervals"]
    assert (intervals["lower_s"] >= 0).all()
    assert (intervals["lower_s"] <= intervals["upper_s"]).all()
    assert not (partitions["calibration"] & partitions["holdout"]).intersection(set(intervals["car_id"]))


def test_calibrated_prediction_uses_same_si_features_and_conformal_residuals(synthetic_training_data):
    fitted = evaluate_models(synthetic_training_data)
    model = fitted["models"]["Random forest"]
    imperial_hp, imperial_torque, imperial_weight = 300, 200, 3000
    row = prediction_frame(imperial_hp, imperial_torque * LBFT_TO_NM,
                           imperial_weight * LB_TO_KG, "RWD")
    output = predict_with_conformal_interval(model, row, [0.5, 0.7, 0.9, 1.2], .75)
    expected = model.predict(row[EXPANDED_FEATURES])[0]
    assert output["point_s"] == pytest.approx(expected)
    assert output["lower_s"] <= output["point_s"] <= output["upper_s"]
    assert output["radius_s"] == pytest.approx(1.2)


def test_shap_reconciles_pipeline_prediction_and_transformed_feature_names(synthetic_training_data):
    fitted = evaluate_models(synthetic_training_data)
    model = fitted["models"]["Random forest"]
    background = synthetic_training_data[EXPANDED_FEATURES].iloc[:20]
    custom = prediction_frame(300, 200 * LBFT_TO_NM, 3000 * LB_TO_KG, "RWD")
    explanation = explain_rf_prediction(model, custom, background)
    assert len(explanation["feature_names"]) == len(explanation["contributions_s"])
    assert any(name.startswith("Drivetrain ") for name in explanation["feature_names"])
    assert explanation["reconstructed_s"] == pytest.approx(explanation["prediction_s"], abs=2e-4)
    selected_vehicle = synthetic_training_data.iloc[[0]][EXPANDED_FEATURES]
    real_explanation = explain_rf_prediction(model, selected_vehicle, background)
    assert real_explanation["reconstructed_s"] == pytest.approx(real_explanation["prediction_s"], abs=2e-4)


def test_shap_errors_raise_cleanly_for_ui_graceful_handling(synthetic_training_data):
    fitted = evaluate_models(synthetic_training_data)

    class BrokenExplainer:
        def shap_values(self, _values):
            raise RuntimeError("optional SHAP failure")

    row = synthetic_training_data[EXPANDED_FEATURES].iloc[[0]]
    explanation, error = safe_explain_rf_prediction(
        fitted["models"]["Random forest"], row,
        synthetic_training_data[EXPANDED_FEATURES], BrokenExplainer(),
    )
    assert explanation is None
    assert "optional SHAP failure" in error


def test_nameplate_family_groups_known_variant_forms():
    from torquelab import _family_key

    assert _family_key("Ford", "Mustang GT") == _family_key("Ford", "Mustang EcoBoost")
    assert _family_key("BMW", "330i") == _family_key("BMW", "3 Series 330i")
    assert _family_key("BMW", "M3") != _family_key("BMW", "3 Series")


def test_training_regenerates_all_artifacts(make_source_db, valid_source_row, tmp_path):
    rows = []
    for group in range(24):
        for variant in range(2):
            rows.append({
                **valid_source_row,
                "car_id": f"{group}-{variant}",
                "manufacturer": f"Make {group % 4}",
                "model": f"Nameplate{group} Trim{variant}",
                "power_hp": 120 + group * 8 + variant * 3,
                "torque": f"{180 + group * 4 + variant * 5} lb-ft",
                "curb_weight_lb": 2400 + group * 30 + variant * 80,
                "drivetrain": ["FWD", "RWD", "AWD", "4WD"][group % 4],
                "acceleration": 10 - (120 + group * 8 + variant * 3) / 180 + (2400 + group * 30 + variant * 80) / 1600,
                "year": str(2000 + group),
            })
    output = tmp_path / "artifacts"
    result = run_analysis(make_source_db(rows), output)
    expected = set(result["manifest"]["generated_artifacts"])
    assert expected == {path.name for path in output.iterdir()}
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["preparation"]["clean_rows"] == 48
    assert manifest["split"]["overlapping_groups"] == 0
    serialized_models = joblib.load(output / "models.joblib")
    sample = result["data"].iloc[0]
    estimates = predict_vehicle(
        serialized_models,
        sample["power_hp"], sample["torque_nm"], sample["weight_kg"], sample["drivetrain"],
    )
    assert all(np.isfinite(value) and 1 <= value <= 30 for value in estimates.values())
