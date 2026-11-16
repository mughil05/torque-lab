"""Reproducible data preparation and grouped-holdout modeling for TorqueLab."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "sources/car-performance-comparison/public/data/car-performance-data.db"
ARTIFACTS = ROOT / "artifacts"
TARGET = "zero_to_sixty_s"
NUMERIC_FEATURES = [
    "power_hp",
    "torque_nm",
    "weight_kg",
    "power_hp_per_1000kg",
    "torque_nm_per_1000kg",
]
CATEGORICAL_FEATURES = ["drivetrain"]
EXPANDED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SEED = 42

# Broad physical sanity bounds applied to complete source rows. Values are
# validated before unit conversion; invalid observations are excluded whole.
BOUNDS = {
    "power_hp": (50.0, 2500.0),
    "torque_lb_ft": (0.000001, 3000.0),
    "curb_weight_lb": (1000.0, 10000.0),
    "zero_to_sixty_s": (1.0, 30.0),
}
VALID_DRIVETRAINS = {"FWD", "RWD", "AWD", "4WD"}
LB_TO_KG = 0.45359237
LBFT_TO_NM = 1.3558179483


def _number(value) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _torque_lb_ft(value) -> float | None:
    """Accept numeric lb-ft spellings only; reject unknown or mixed units."""
    if value is None:
        return None
    raw = str(value).strip().lower().replace("−", "-")
    if not re.search(r"\blb\s*[- ]?\s*ft\b", raw):
        return None
    return _number(raw)


def _family_key(manufacturer: str, model: str) -> str:
    """Group nameplate variants so related trims cannot cross the split.

    Most models use their first normalized token as the nameplate. Known
    multi-token nameplates retain the second token, including Tesla Model
    variants, BMW numbered Series, Mercedes classes and common SUV names.
    """
    tokens = re.findall(r"[a-z0-9]+", str(model).lower())
    if not tokens:
        return f"{manufacturer.lower()}::unknown"
    make = manufacturer.strip().lower()
    joined = " ".join(tokens[:2])
    base = tokens[0]
    if make == "bmw":
        # 320i / 330e / M340i are trims within the corresponding numbered
        # Series. M3 and M4 remain their own nameplates.
        bmw_series = re.fullmatch(r"m?([1-8])\d{2}(?:i|d|e|x)?", tokens[0])
        if bmw_series:
            base = f"{bmw_series.group(1)} series"
    two_word_prefixes = {
        "model", "3 series", "5 series", "6 series", "7 series", "8 series",
        "1 series", "2 series", "c class", "e class", "s class", "a class",
        "g class", "range rover", "land cruiser", "grand cherokee", "f series",
    }
    if joined in two_word_prefixes:
        base = joined
    elif make == "mercedes-benz" and tokens[0] in {"a", "b", "c", "e", "s", "g"}:
        base = f"{tokens[0]} class"
    elif make == "tesla" and tokens[0] in {"model", "s", "3", "x", "y"}:
        base = f"model {tokens[1]}" if tokens[0] == "model" and len(tokens) > 1 else f"model {tokens[0]}"
    return f"{make}::{base}"


def prepare_dataset(database: Path = DATABASE) -> tuple[pd.DataFrame, dict]:
    """Select source-verified gasoline rows, validate, and engineer SI features."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        raw = pd.DataFrame([dict(row) for row in connection.execute("SELECT * FROM cars")])

    total = len(raw)
    exact_source = raw["sources"].fillna("").str.strip().eq("Car & Driver")
    gasoline = raw["propulsion"].fillna("").str.strip().str.casefold().eq("petrol")
    source_pool = raw.loc[exact_source & gasoline].copy()
    missing = {
        "horsepower": source_pool["power_hp"].isna(),
        "torque": source_pool["torque"].map(_torque_lb_ft).isna(),
        "curb weight": source_pool["curb_weight_lb"].isna(),
        "drivetrain": source_pool["drivetrain"].fillna("").str.strip().eq(""),
        "genuine 0-60 target": source_pool["0_60_mph_sec"].isna(),
    }

    rows = []
    exclusion_counts = {
        "not_exact_car_and_driver_source_or_not_gasoline": total - len(source_pool),
        "incomplete_required_fields": 0,
        "implausible_or_malformed_values": 0,
        "duplicate_records": 0,
    }
    seen_ids = set()
    for _, row in source_pool.iterrows():
        car_id = str(row.get("car_id") or "")
        if car_id in seen_ids:
            exclusion_counts["duplicate_records"] += 1
            continue
        seen_ids.add(car_id)

        torque = _torque_lb_ft(row.get("torque"))
        hp = _number(row.get("power_hp"))
        weight_lb = _number(row.get("curb_weight_lb"))
        target = _number(row.get("0_60_mph_sec"))
        drivetrain = str(row.get("drivetrain") or "").strip().upper()
        if hp is None or torque is None or weight_lb is None or target is None or not drivetrain:
            exclusion_counts["incomplete_required_fields"] += 1
            continue
        valid = (
            BOUNDS["power_hp"][0] <= hp <= BOUNDS["power_hp"][1]
            and BOUNDS["torque_lb_ft"][0] < torque <= BOUNDS["torque_lb_ft"][1]
            and BOUNDS["curb_weight_lb"][0] <= weight_lb <= BOUNDS["curb_weight_lb"][1]
            and BOUNDS["zero_to_sixty_s"][0] <= target <= BOUNDS["zero_to_sixty_s"][1]
            and drivetrain in VALID_DRIVETRAINS
        )
        if not valid:
            exclusion_counts["implausible_or_malformed_values"] += 1
            continue

        weight_kg = weight_lb * LB_TO_KG
        torque_nm = torque * LBFT_TO_NM
        model = str(row["model"]).strip()
        manufacturer = str(row["manufacturer"]).strip()
        year = str(row.get("year") or "").strip()
        rows.append({
            "car_id": car_id,
            "manufacturer": manufacturer,
            "model": model,
            # Dataset A has a model string, not a separate trim field. Retain
            # that exact source label as trim; do not infer a trim designation.
            "trim": model,
            "year": year,
            "engine": str(row.get("engine") or "").strip(),
            "power_hp": hp,
            "torque_nm": torque_nm,
            "weight_kg": weight_kg,
            "drivetrain": drivetrain,
            TARGET: target,
            "power_hp_per_1000kg": hp / (weight_kg / 1000.0),
            "torque_nm_per_1000kg": torque_nm / (weight_kg / 1000.0),
            "group_key": _family_key(manufacturer, model),
            "source": "Car & Driver",
            "source_target_field": "zero_to_60_mph",
        })

    prepared = pd.DataFrame(rows)
    exclusions = {
        **exclusion_counts,
        "total_source_rows": total,
        "exact_source_gasoline_pool": len(source_pool),
        "clean_rows": len(prepared),
        "unique_nameplate_groups": int(prepared["group_key"].nunique()) if len(prepared) else 0,
        "missing_by_required_field_in_source_pool": {key: int(mask.sum()) for key, mask in missing.items()},
        "bounds": BOUNDS,
        "unit_conversions": {
            "curb weight": "lb × 0.45359237 = kg",
            "torque": "lb-ft × 1.3558179483 = N·m",
            "power": "horsepower (hp), retained as published",
        },
        "source_rule": "propulsion exactly Petrol AND source list exactly Car & Driver",
        "target_rule": "0_60_mph_sec sourced from Car & Driver parser's dedicated zero_to_60_mph field; never 0-62 mph or 0-100 km/h",
        "trim_rule": "trim is a copy of the source model label; no inferred trim parsing",
        "group_rule": "manufacturer plus normalized nameplate family prefix; grouped split keeps each family in one partition",
    }
    return prepared, exclusions


def _linear_pipeline(features: list[str]) -> Pipeline:
    numeric = [name for name in features if name != "drivetrain"]
    categorical = [name for name in features if name == "drivetrain"]
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", preprocess), ("regressor", LinearRegression())])


def _random_forest_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="drop")
    return Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=1.0,
            random_state=SEED, n_jobs=1,
        )),
    ])


def _metrics(y_true, prediction) -> dict:
    return {
        "MAE (s)": float(mean_absolute_error(y_true, prediction)),
        "RMSE (s)": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "R²": float(r2_score(y_true, prediction)),
    }


def original_training_partition(data: pd.DataFrame) -> pd.DataFrame:
    """Return only the training rows from the immutable seed-42 grouped split."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, _ = next(splitter.split(data, data[TARGET], groups=data["group_key"]))
    return data.iloc[train_idx].copy()


def prediction_frame(power_hp: float, torque_nm: float, weight_kg: float, drivetrain: str) -> pd.DataFrame:
    """Build the same deterministic features used by training for one input."""
    values = [power_hp, torque_nm, weight_kg]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Horsepower, torque, and weight must be finite numbers")
    if not (50 <= power_hp <= 2500 and 0 < torque_nm <= BOUNDS["torque_lb_ft"][1] * LBFT_TO_NM
            and BOUNDS["curb_weight_lb"][0] * LB_TO_KG <= weight_kg <= BOUNDS["curb_weight_lb"][1] * LB_TO_KG):
        raise ValueError("Input is outside the documented physical feature bounds")
    drive = str(drivetrain).strip().upper()
    if drive not in VALID_DRIVETRAINS:
        raise ValueError(f"Unsupported drivetrain: {drivetrain}")
    tonnes = weight_kg / 1000.0
    return pd.DataFrame([{
        "power_hp": float(power_hp),
        "torque_nm": float(torque_nm),
        "weight_kg": float(weight_kg),
        "power_hp_per_1000kg": float(power_hp / tonnes),
        "torque_nm_per_1000kg": float(torque_nm / tonnes),
        "drivetrain": drive,
    }])


def predict_vehicle(models: dict, power_hp: float, torque_nm: float,
                    weight_kg: float, drivetrain: str) -> dict[str, float]:
    """Return finite, physically bounded predictions from fitted pipelines."""
    row = prediction_frame(power_hp, torque_nm, weight_kg, drivetrain)
    output = {}
    for name, model in models.items():
        features = ["power_hp"] if name == "Horsepower baseline" else EXPANDED_FEATURES
        value = float(model.predict(row[features])[0])
        if not np.isfinite(value) or not BOUNDS["zero_to_sixty_s"][0] <= value <= BOUNDS["zero_to_sixty_s"][1]:
            raise ValueError(f"{name} produced a 0-60 estimate outside 1–30 seconds; input is unsupported")
        output[name] = value
    return output


def input_support(data: pd.DataFrame) -> dict:
    """Summarize input support bounds; call with training rows only."""
    return {
        "numeric": {
            feature: [float(data[feature].quantile(0.05)), float(data[feature].quantile(0.95))]
            for feature in NUMERIC_FEATURES
        },
        "drivetrain": sorted(data["drivetrain"].unique().tolist()),
    }


def out_of_distribution_features(support: dict, inputs: pd.DataFrame) -> list[str]:
    """Name inputs outside the training-only central 90% support bounds."""
    row = inputs.iloc[0]
    outside = [
        feature for feature, (low, high) in support["numeric"].items()
        if row[feature] < low or row[feature] > high
    ]
    if row["drivetrain"] not in set(support["drivetrain"]):
        outside.append("drivetrain")
    return outside


def conformal_quantile(residuals, coverage: float = 0.90) -> tuple[float, int]:
    """Finite-sample split-conformal order statistic for absolute residuals."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    values = np.asarray(residuals, dtype=float).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration residuals must be a nonempty finite sample")
    values = np.abs(values)
    rank = int(np.ceil((len(values) + 1) * coverage))
    if rank > len(values):
        return float("inf"), rank
    return float(np.partition(values, rank - 1)[rank - 1]), rank


def prediction_interval(point: float, radius: float) -> tuple[float, float]:
    """Return a physically nonnegative symmetric interval around one prediction."""
    if not np.isfinite(point) or radius < 0 or np.isnan(radius):
        raise ValueError("Prediction and interval radius must be valid")
    return max(0.0, float(point) - float(radius)), float(point) + float(radius)


def predict_with_conformal_interval(model, row: pd.DataFrame, residuals,
                                    coverage: float = 0.90) -> dict:
    """Predict with the distinct calibration RF and its matching residual quantile."""
    if not set(EXPANDED_FEATURES).issubset(row.columns):
        raise ValueError("Prediction row is missing model features")
    point = float(model.predict(row[EXPANDED_FEATURES])[0])
    if not np.isfinite(point):
        raise ValueError("Calibration model returned a non-finite estimate")
    radius, rank = conformal_quantile(residuals, coverage)
    lower, upper = prediction_interval(point, radius)
    return {"point_s": point, "lower_s": lower, "upper_s": upper,
            "radius_s": radius, "coverage": coverage, "quantile_rank": rank}


def group_conformal_calibration(data: pd.DataFrame, coverage: float = 0.90) -> dict:
    """Fit a separate RF and calibrate it using group-disjoint rows from original training only.

    The final grouped holdout is never used for model fitting or quantile choice.
    After fixing the calibration quantile, it is used once to estimate empirical
    coverage and mean interval width for this separate calibration model.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    training_idx, holdout_idx = next(outer.split(data, data[TARGET], groups=data["group_key"]))
    training = data.iloc[training_idx].copy()
    holdout = data.iloc[holdout_idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + 1)
    fit_idx, calibration_idx = next(
        inner.split(training, training[TARGET], groups=training["group_key"])
    )
    fit = training.iloc[fit_idx].copy()
    calibration = training.iloc[calibration_idx].copy()
    fit_groups, calibration_groups = set(fit["group_key"]), set(calibration["group_key"])
    holdout_groups = set(holdout["group_key"])
    assert fit_groups.isdisjoint(calibration_groups)
    assert fit_groups.isdisjoint(holdout_groups)
    assert calibration_groups.isdisjoint(holdout_groups)

    model = _random_forest_pipeline()
    model.fit(fit[EXPANDED_FEATURES], fit[TARGET])
    calibration_prediction = model.predict(calibration[EXPANDED_FEATURES])
    calibration_residual_rows = calibration[["car_id", "group_key"]].copy()
    calibration_residual_rows["absolute_residual_s"] = np.abs(
        calibration[TARGET].to_numpy() - calibration_prediction
    )
    # One maximum score per held-out nameplate gives group-level calibration,
    # avoiding treating correlated trims as independent calibration examples.
    absolute_residuals = calibration_residual_rows.groupby("group_key")["absolute_residual_s"].max()
    radius, rank = conformal_quantile(absolute_residuals.to_numpy(), coverage)

    # Holdout is touched only after model and interval width are fixed.
    holdout_prediction = model.predict(holdout[EXPANDED_FEATURES])
    bounds = [prediction_interval(value, radius) for value in holdout_prediction]
    intervals = holdout[["car_id", "manufacturer", "model", "year", "drivetrain", TARGET]].copy()
    intervals["calibration_rf_prediction_s"] = holdout_prediction
    intervals["lower_s"] = [pair[0] for pair in bounds]
    intervals["upper_s"] = [pair[1] for pair in bounds]
    covered = intervals[TARGET].between(intervals["lower_s"], intervals["upper_s"], inclusive="both")
    intervals["covered"] = covered
    intervals["_group"] = intervals["car_id"].map(holdout.set_index("car_id")["group_key"])
    holdout_groups_covered = intervals.groupby("_group", observed=True)["covered"].all()
    group_hash = lambda values: hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()
    metadata = {
        "method": "split conformal absolute residuals with GroupShuffleSplit partitions",
        "coverage": coverage,
        "quantile_radius_s": radius,
        "finite_sample_rank_1_based": rank,
        "calibration_residual_count": len(absolute_residuals),
        "calibration_residual_unit": "maximum absolute residual per calibration nameplate group",
        "fit_rows": len(fit), "calibration_rows": len(calibration),
        "fit_groups": len(fit_groups), "calibration_groups": len(calibration_groups),
        "original_training_rows": len(training), "final_holdout_rows": len(holdout),
        "final_holdout_groups": len(holdout_groups),
        "fit_calibration_group_overlap": len(fit_groups & calibration_groups),
        "fit_holdout_group_overlap": len(fit_groups & holdout_groups),
        "calibration_holdout_group_overlap": len(calibration_groups & holdout_groups),
        "holdout_empirical_coverage": float(covered.mean()),
        "holdout_all_vehicles_covered_group_fraction": float(holdout_groups_covered.mean()),
        "holdout_mean_interval_width_s": float((intervals["upper_s"] - intervals["lower_s"]).mean()),
        "holdout_evaluation_rows": len(intervals),
        "fit_ids_sha256": group_hash(fit["car_id"]),
        "calibration_ids_sha256": group_hash(calibration["car_id"]),
        "holdout_ids_sha256": group_hash(holdout["car_id"]),
        "assumptions": "Approximate exchangeability of group-sampled calibration and future examples; within-nameplate dependence can weaken nominal marginal coverage.",
    }
    return {
        "model": model, "metadata": metadata,
        "absolute_residuals": pd.DataFrame({
            "group_key": absolute_residuals.index.astype(str),
            "absolute_residual_s": absolute_residuals.to_numpy(),
        }),
        "holdout_intervals": intervals,
        "partition_ids": {
            "fit": set(fit["car_id"].astype(str)),
            "calibration": set(calibration["car_id"].astype(str)),
            "holdout": set(holdout["car_id"].astype(str)),
            "fit_groups": fit_groups, "calibration_groups": calibration_groups,
            "holdout_groups": holdout_groups,
        },
    }


def save_group_conformal_artifacts(data: pd.DataFrame, output_dir: Path = ARTIFACTS,
                                  coverage: float = 0.90) -> dict:
    """Persist only the independent calibration model and its local diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = group_conformal_calibration(data, coverage)
    joblib.dump(result["model"], output_dir / "calibration_rf.joblib", compress=3)
    result["absolute_residuals"].to_csv(output_dir / "calibration_residuals.csv", index=False)
    result["holdout_intervals"].to_csv(output_dir / "calibrated_holdout_intervals.csv", index=False)
    (output_dir / "calibration_manifest.json").write_text(json.dumps(result["metadata"], indent=2) + "\n")
    return result


def create_shap_explainer(model, background: pd.DataFrame):
    """Build TreeSHAP for the forest inside the existing preprocessing pipeline."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"shap(?:\..*)?")
        import shap

    preprocess = model.named_steps["preprocess"]
    forest = model.named_steps["regressor"]
    transformed_background = preprocess.transform(background[EXPANDED_FEATURES])
    return shap.TreeExplainer(
        forest, data=transformed_background,
        feature_perturbation="interventional", model_output="raw",
    )


def explain_rf_prediction(model, row: pd.DataFrame, background: pd.DataFrame,
                          explainer=None) -> dict:
    """Explain the pipeline's RF output in seconds with transformed feature names."""
    preprocess = model.named_steps["preprocess"]
    transformed = preprocess.transform(row[EXPANDED_FEATURES])
    explainer = explainer or create_shap_explainer(model, background)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"shap(?:\..*)?")
        values = np.asarray(explainer.shap_values(transformed), dtype=float)
    if values.ndim == 3:
        values = values[0]
    if values.ndim == 2:
        values = values[0]
    contribution = values.reshape(-1)
    names = list(preprocess.get_feature_names_out())
    if len(names) != len(contribution):
        raise ValueError("SHAP feature names do not match transformed model inputs")
    base_values = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    baseline = float(base_values[0])
    prediction = float(model.predict(row[EXPANDED_FEATURES])[0])
    reconstructed = baseline + float(contribution.sum())
    # TreeSHAP's interventional explainer can accumulate small float32
    # rounding error through the one-hot transformed forest input.
    if not np.isclose(reconstructed, prediction, rtol=2e-5, atol=2e-4):
        raise ValueError(f"SHAP contributions do not reconcile: {reconstructed:.6f} vs {prediction:.6f} s")
    display_names = [name.replace("numeric__", "").replace("categorical__drivetrain_", "Drivetrain ") for name in names]
    return {
        "baseline_s": baseline, "prediction_s": prediction,
        "feature_names": display_names, "contributions_s": contribution,
        "reconstructed_s": reconstructed,
    }


def safe_explain_rf_prediction(model, row: pd.DataFrame, background: pd.DataFrame,
                               explainer=None) -> tuple[dict | None, str | None]:
    """Return an optional explanation failure as data so prediction UI can continue."""
    try:
        return explain_rf_prediction(model, row, background, explainer), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def tune_vehicle(models: dict, support: dict, vehicle: pd.Series | dict,
                 power_hp: float, weight_lb: float) -> dict:
    """Compare a saved Random Forest prediction for a real row and a hypothetical edit."""
    original = dict(vehicle)
    original_row = prediction_frame(
        float(original["power_hp"]), float(original["torque_nm"]),
        float(original["weight_kg"]), str(original["drivetrain"]),
    )
    modified_weight_kg = float(weight_lb) * LB_TO_KG
    modified_row = prediction_frame(
        float(power_hp), float(original["torque_nm"]), modified_weight_kg,
        str(original["drivetrain"]),
    )
    model = models["Random forest"]
    original_prediction = float(model.predict(original_row[EXPANDED_FEATURES])[0])
    modified_prediction = float(model.predict(modified_row[EXPANDED_FEATURES])[0])
    if not all(np.isfinite(value) for value in (original_prediction, modified_prediction)):
        raise ValueError("Random Forest returned a non-finite estimate")
    return {
        "original_row": original_row,
        "modified_row": modified_row,
        "original_prediction_s": original_prediction,
        "modified_prediction_s": modified_prediction,
        "change_s": modified_prediction - original_prediction,
        "original_power_hp_per_1000kg": float(original_row.iloc[0]["power_hp_per_1000kg"]),
        "modified_power_hp_per_1000kg": float(modified_row.iloc[0]["power_hp_per_1000kg"]),
        "outside_features": out_of_distribution_features(support, modified_row),
    }


def leaderboard(data: pd.DataFrame, ranking: str, manufacturer: str = "All manufacturers",
                drivetrain: str = "All drivetrains") -> pd.DataFrame:
    """Return a stable, identity-deduplicated leaderboard from observed rows."""
    frame = data.copy()
    if manufacturer != "All manufacturers":
        frame = frame[frame["manufacturer"] == manufacturer]
    if drivetrain != "All drivetrains":
        frame = frame[frame["drivetrain"] == drivetrain]
    frame["power_to_weight_hp_per_1000kg"] = frame["power_hp"] / (frame["weight_kg"] / 1000.0)
    sort_columns = {
        "Fastest measured 0–60 mph": [TARGET, "manufacturer", "model", "year", "car_id"],
        "Highest horsepower": ["power_hp", "manufacturer", "model", "year", "car_id"],
        "Best horsepower-to-weight": ["power_to_weight_hp_per_1000kg", "manufacturer", "model", "year", "car_id"],
    }
    if ranking not in sort_columns:
        raise ValueError(f"Unsupported leaderboard ranking: {ranking}")
    ascending = [ranking == "Fastest measured 0–60 mph"] + [True] * 4
    frame = frame.sort_values(sort_columns[ranking], ascending=ascending, kind="mergesort")
    frame = frame.drop_duplicates(subset=["manufacturer", "model", "year"], keep="first")
    return frame


def model_error_diagnostics(predictions: pd.DataFrame, model: str = "Random forest") -> dict:
    """Calculate exploratory diagnostics from held-out prediction records only."""
    frame = predictions.copy()
    frame["predicted_s"] = frame[model].astype(float)
    frame["actual_s"] = frame[TARGET].astype(float)
    frame["residual_s"] = frame["predicted_s"] - frame["actual_s"]
    frame["absolute_error_s"] = frame["residual_s"].abs()
    frame["horsepower_range"] = pd.cut(
        frame["power_hp"], bins=[-np.inf, 150, 250, 400, np.inf],
        labels=["≤150 hp", "151–250 hp", "251–400 hp", ">400 hp"],
    )
    by_drivetrain = frame.groupby("drivetrain", observed=True)["absolute_error_s"].agg(
        ["mean", "count"]
    ).reset_index().rename(columns={"mean": "MAE (s)", "count": "Vehicles"})
    by_horsepower = frame.groupby("horsepower_range", observed=True)["absolute_error_s"].agg(
        ["mean", "count"]
    ).reset_index().rename(columns={"mean": "MAE (s)", "count": "Vehicles"})
    largest = frame.sort_values(
        ["absolute_error_s", "manufacturer", "model", "year", "car_id"],
        ascending=[False, True, True, True, True], kind="mergesort",
    ).head(10)
    identity = ["car_id", "manufacturer", "model", "year", "drivetrain"]
    return {
        "predictions": frame,
        "by_drivetrain": by_drivetrain,
        "by_horsepower": by_horsepower,
        "largest_errors": largest[identity + ["actual_s", "predicted_s", "absolute_error_s"]],
        "residuals": frame["residual_s"],
    }


def grouped_cross_validation(data: pd.DataFrame, n_splits: int = 5) -> tuple[pd.DataFrame, dict]:
    """Evaluate unchanged model pipelines with grouped folds of the holdout's training partition."""
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_index, holdout_index = next(split.split(data, data[TARGET], groups=data["group_key"]))
    train = data.iloc[train_index].copy()
    holdout = data.iloc[holdout_index]
    assert set(train["group_key"]).isdisjoint(set(holdout["group_key"]))
    splitter = GroupKFold(n_splits=n_splits)
    specs = [
        ("Horsepower baseline", ["power_hp"], _linear_pipeline(["power_hp"])),
        ("Expanded linear regression", EXPANDED_FEATURES, _linear_pipeline(EXPANDED_FEATURES)),
        ("Random forest", EXPANDED_FEATURES, _random_forest_pipeline()),
    ]
    rows = []
    validation_ids = set()
    fold_group_overlaps = []
    for fold, (fit_idx, validation_idx) in enumerate(
        splitter.split(train, train[TARGET], groups=train["group_key"]), start=1
    ):
        fit, validation = train.iloc[fit_idx], train.iloc[validation_idx]
        overlap = len(set(fit["group_key"]) & set(validation["group_key"]))
        fold_group_overlaps.append(overlap)
        assert overlap == 0
        validation_ids.update(validation["car_id"].astype(str))
        for name, features, pipeline in specs:
            pipeline.fit(fit[features], fit[TARGET])
            prediction = pipeline.predict(validation[features])
            rows.append({"Fold": fold, "Model": name, "Validation rows": len(validation),
                         "Validation groups": validation["group_key"].nunique(),
                         **_metrics(validation[TARGET], prediction)})
    fold_metrics = pd.DataFrame(rows)
    summary_rows = []
    for name, group in fold_metrics.groupby("Model", sort=False):
        summary_rows.append({"Model": name, "MAE mean (s)": group["MAE (s)"].mean(),
                             "MAE SD (s)": group["MAE (s)"].std(ddof=1),
                             "RMSE mean (s)": group["RMSE (s)"].mean(),
                             "RMSE SD (s)": group["RMSE (s)"].std(ddof=1),
                             "R² mean": group["R²"].mean(), "R² SD": group["R²"].std(ddof=1)})
    metadata = {
        "n_splits": n_splits, "training_rows": len(train), "holdout_rows": len(holdout),
        "training_groups": int(train["group_key"].nunique()),
        "holdout_groups": int(holdout["group_key"].nunique()),
        "train_holdout_group_overlap": len(set(train["group_key"]) & set(holdout["group_key"])),
        "cv_validation_rows": len(validation_ids),
        "cv_holdout_id_overlap": len(validation_ids & set(holdout["car_id"].astype(str))),
        "fold_group_overlaps": fold_group_overlaps,
        "cv_train_ids": sorted(validation_ids),
        "holdout_ids": sorted(holdout["car_id"].astype(str).tolist()),
    }
    return fold_metrics, {"summary": pd.DataFrame(summary_rows), "metadata": metadata}


def evaluate_models(data: pd.DataFrame) -> dict:
    """Train the requested three models and diagnostics on one grouped holdout."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_index, test_index = next(splitter.split(data, data[TARGET], groups=data["group_key"]))
    train = data.iloc[train_index].copy()
    test = data.iloc[test_index].copy()
    assert set(train["group_key"]).isdisjoint(set(test["group_key"]))
    y_train, y_test = train[TARGET], test[TARGET]

    specs = [
        ("Horsepower baseline", ["power_hp"], _linear_pipeline(["power_hp"])),
        ("Expanded linear regression", EXPANDED_FEATURES, _linear_pipeline(EXPANDED_FEATURES)),
        ("Random forest", EXPANDED_FEATURES, _random_forest_pipeline()),
    ]
    fitted = {}
    metric_rows = []
    prediction_frame = test[["car_id", "manufacturer", "model", "trim", "year", "drivetrain", TARGET, "group_key"]].copy()
    for name, features, pipeline in specs:
        pipeline.fit(train[features], y_train)
        pred = pipeline.predict(test[features])
        fitted[name] = (pipeline, features)
        prediction_frame[name] = pred
        metric_rows.append({"Model": name, **_metrics(y_test, pred)})
    metrics = pd.DataFrame(metric_rows)

    # Test whether each feature family contributes to the full linear model.
    # Ratios are removed with the source dimensions on which they depend.
    ablation_groups = {
        "Horsepower": {"power_hp", "power_hp_per_1000kg"},
        "Torque": {"torque_nm", "torque_nm_per_1000kg"},
        "Weight": {"weight_kg", "power_hp_per_1000kg", "torque_nm_per_1000kg"},
        "Drivetrain": {"drivetrain"},
    }
    full_features = EXPANDED_FEATURES
    full_pred = prediction_frame["Expanded linear regression"].to_numpy()
    full_metrics = _metrics(y_test, full_pred)
    ablation_rows = []
    for group_name, excluded in ablation_groups.items():
        retained = [f for f in full_features if f not in excluded]
        reduced = _linear_pipeline(retained)
        reduced.fit(train[retained], y_train)
        reduced_metrics = _metrics(y_test, reduced.predict(test[retained]))
        ablation_rows.append({
            "Removed feature family": group_name,
            "MAE without family (s)": reduced_metrics["MAE (s)"],
            "MAE increase vs full (s)": reduced_metrics["MAE (s)"] - full_metrics["MAE (s)"],
            "R² without family": reduced_metrics["R²"],
            "R² change vs full": full_metrics["R²"] - reduced_metrics["R²"],
        })
    ablation = pd.DataFrame(ablation_rows)

    forest = fitted["Random forest"][0]
    importance = permutation_importance(
        forest, test[EXPANDED_FEATURES], y_test,
        scoring="neg_mean_absolute_error", n_repeats=12, random_state=SEED, n_jobs=1,
    )
    importance_frame = pd.DataFrame({
        "Feature": EXPANDED_FEATURES,
        "MAE increase when shuffled (s)": importance.importances_mean,
        "Standard deviation (s)": importance.importances_std,
    }).sort_values("MAE increase when shuffled (s)", ascending=False)

    split_info = {
        "random_seed": SEED,
        "splitter": "GroupShuffleSplit",
        "test_size_target": 0.20,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_groups": int(train["group_key"].nunique()),
        "test_groups": int(test["group_key"].nunique()),
        "overlapping_groups": len(set(train["group_key"]) & set(test["group_key"])),
        "target_test_mean_s": float(y_test.mean()),
        "target_test_sd_s": float(y_test.std(ddof=1)),
    }
    return {
        "metrics": metrics,
        "ablation": ablation,
        "importance": importance_frame,
        "predictions": prediction_frame,
        "models": {name: item[0] for name, item in fitted.items()},
        "split": split_info,
        "input_support": input_support(train),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_cross_validation(data: pd.DataFrame, output_dir: Path = ARTIFACTS) -> dict:
    """Write grouped CV outputs independently of the fixed holdout artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    folds, result = grouped_cross_validation(data)
    folds.to_csv(output_dir / "cross_validation_folds.csv", index=False)
    result["summary"].to_csv(output_dir / "cross_validation_summary.csv", index=False)
    (output_dir / "cross_validation_manifest.json").write_text(
        json.dumps(result["metadata"], indent=2) + "\n"
    )
    return result


def run_analysis(database: Path = DATABASE, output_dir: Path = ARTIFACTS) -> dict:
    data, exclusions = prepare_dataset(database)
    results = evaluate_models(data)
    cv_folds, cv_result = grouped_cross_validation(data)
    calibration_result = save_group_conformal_artifacts(data, output_dir, coverage=0.90)
    output_dir.mkdir(parents=True, exist_ok=True)
    # These outputs contain derived vehicle data and are deliberately local;
    # Dataset A's compiled-data redistribution rights have not been cleared.
    data.to_csv(output_dir / "clean_dataset.csv", index=False)
    results["metrics"].to_csv(output_dir / "model_metrics.csv", index=False)
    results["ablation"].to_csv(output_dir / "feature_ablation.csv", index=False)
    results["importance"].to_csv(output_dir / "feature_importance.csv", index=False)
    results["predictions"].to_csv(output_dir / "holdout_predictions.csv", index=False)
    manifest = {
        "runtime": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
        },
        "dataset_path": str(database.relative_to(ROOT)) if database.is_relative_to(ROOT) else str(database),
        "dataset_sha256": _sha256(database),
        "preparation": exclusions,
        "split": results["split"],
        "training_input_support": results["input_support"],
        "models": results["metrics"].to_dict(orient="records"),
        "generated_artifacts": [
            "clean_dataset.csv", "model_metrics.csv", "feature_ablation.csv",
            "feature_importance.csv", "holdout_predictions.csv", "models.joblib",
            "run_manifest.json", "cross_validation_folds.csv",
            "cross_validation_summary.csv", "cross_validation_manifest.json",
            "calibration_rf.joblib", "calibration_residuals.csv",
            "calibrated_holdout_intervals.csv", "calibration_manifest.json",
        ],
        "redistribution": "Derived data are local and excluded from Git; Dataset A's compiled-data rights require review before publication.",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    joblib.dump(results["models"], output_dir / "models.joblib", compress=3)
    cv_folds.to_csv(output_dir / "cross_validation_folds.csv", index=False)
    cv_result["summary"].to_csv(output_dir / "cross_validation_summary.csv", index=False)
    (output_dir / "cross_validation_manifest.json").write_text(json.dumps(cv_result["metadata"], indent=2) + "\n")
    results["data"] = data
    results["exclusions"] = exclusions
    results["manifest"] = manifest
    results["cv_folds"] = cv_folds
    results["cv_summary"] = cv_result["summary"]
    results["cv_manifest"] = cv_result["metadata"]
    results["calibration_model"] = calibration_result["model"]
    results["calibration_residuals"] = calibration_result["absolute_residuals"]["absolute_residual_s"].to_numpy()
    results["calibration_manifest"] = calibration_result["metadata"]
    return results


if __name__ == "__main__":
    result = run_analysis()
    print(json.dumps(result["manifest"], indent=2, default=str))
    print("\nHoldout model metrics")
    print(result["metrics"].to_string(index=False))
    print("\nLinear feature ablation")
    print(result["ablation"].to_string(index=False))
    print("\nRandom forest permutation importance")
    print(result["importance"].to_string(index=False))
