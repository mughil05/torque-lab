"""TorqueLab: gasoline 0–60 mph analytics dashboard."""
from __future__ import annotations

import pandas as pd
import altair as alt
import hashlib
import joblib
import sklearn
import streamlit as st

from torquelab import (
    ARTIFACTS, DATABASE, LB_TO_KG, TARGET, leaderboard,
    model_error_diagnostics, out_of_distribution_features, prediction_frame,
    create_shap_explainer,
    original_training_partition, predict_vehicle, predict_with_conformal_interval,
    run_analysis, safe_explain_rf_prediction, save_cross_validation,
    save_group_conformal_artifacts, tune_vehicle,
)

st.set_page_config(
    page_title="TorqueLab | Performance Analytics",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

def apply_theme():
    """Apply the shared TorqueLab dark motorsport design system."""
    st.markdown("""
<style>
  :root { --ink:#f2f4f7; --muted:#9aa3af; --panel:#15181d; --line:#30353d; --accent:#e34242; }
  html, body, [class*="css"] { font-family: 'Avenir Next', 'Segoe UI', sans-serif; }
  .stApp { background:#0d0f12; color:var(--ink); }
  [data-testid="stSidebar"] { background:#111317; border-right:1px solid var(--line); }
  .block-container { max-width:1440px; padding-top:1.5rem; padding-bottom:3rem; }
  h1,h2,h3 { letter-spacing:-.035em; }
  h1 { font-weight:800 !important; }
  .eyebrow { color:var(--accent); font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700; }
  .hero { border:1px solid #383d45; border-left:4px solid var(--accent); background:#15181d; border-radius:14px; padding:1.6rem 1.8rem; margin:.5rem 0 1.35rem; }
  .hero p { color:var(--muted); margin:.4rem 0 0; max-width:780px; }
  .metric-card { border:1px solid var(--line); background:var(--panel); border-radius:12px; padding:1rem 1.1rem; min-height:112px; }
  .metric-label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.09em; }
  .metric-value { color:var(--ink); font-size:1.65rem; font-weight:800; margin-top:.25rem; }
  .metric-caption { color:#7f8da0; font-size:.75rem; margin-top:.2rem; }
  .metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; margin-bottom:1.25rem; }
  .metric-card { min-width:0; overflow-wrap:anywhere; }
  @media (max-width: 900px) { .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
  @media (max-width: 480px) { .metric-grid { gap:.55rem; } .metric-card { padding:.8rem; min-height:100px; } .metric-value { font-size:1.35rem; } .metric-label,.metric-caption { font-size:.68rem; } }
  .section-note { color:var(--muted); font-size:.92rem; }
  div[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:1rem 1.1rem; border-radius:12px; }
  div[data-testid="stMetricLabel"] { color:var(--muted); }
  div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
  .stButton>button, .stFormSubmitButton>button { border-radius:9px; border:1px solid #5a3034; background:#351c20; color:#f7f3f3; font-weight:700; }
  .stButton>button:hover, .stFormSubmitButton>button:hover { border-color:var(--accent); color:var(--accent); }
  hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)


apply_theme()


class ArtifactCompatibilityError(RuntimeError):
    """Saved estimators cannot be loaded by this project's runtime."""


def cached_shap_explainer(model_key, background_key, _model, _background):
    return create_shap_explainer(_model, _background)


cached_shap_explainer = st.cache_resource(cached_shap_explainer)


def shap_chart(explanation: dict, label: str):
    """Render compact signed SHAP contributions, with time units and baseline."""
    st.markdown(f"#### Why the model estimated {label}")
    st.caption(f"Baseline {explanation['baseline_s']:.2f} s → prediction {explanation['prediction_s']:.2f} s. Contributions are seconds; positive pushes slower and negative pushes faster.")
    st.markdown('<p class="section-note">Each bar shows how one vehicle feature shifts the estimate from the baseline: positive contributions increase predicted 0–60 time, while negative contributions decrease it.</p>', unsafe_allow_html=True)
    def readable_feature(name: str) -> str:
        labels = {
            "power_hp_per_1000kg": "Power-to-weight ratio",
            "torque_nm_per_1000kg": "Torque-to-weight ratio",
            "power_hp": "Horsepower",
            "torque_nm": "Torque",
            "weight_kg": "Curb weight",
        }
        if name.startswith("Drivetrain "):
            return name
        return labels.get(name, name.replace("_", " ").title())
    chart_data = pd.DataFrame({
        "Feature": [readable_feature(name) for name in explanation["feature_names"]],
        "Contribution (s)": explanation["contributions_s"],
    }).assign(_magnitude=lambda frame: frame["Contribution (s)"].abs())
    chart_data = chart_data.sort_values("_magnitude", ascending=False, kind="mergesort").reset_index(drop=True)
    current = explanation["baseline_s"]
    starts, ends = [], []
    for value in chart_data["Contribution (s)"]:
        starts.append(current)
        current += value
        ends.append(current)
    chart_data["Start (s)"], chart_data["End (s)"] = starts, ends
    chart_data["Order"] = range(len(chart_data))
    chart_data["Effect"] = chart_data["Contribution (s)"].map(lambda value: "Slower" if value > 0 else "Faster")
    chart = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=3, height=16).encode(
        x=alt.X("Start (s):Q", title="Cumulative model output (s)", scale=alt.Scale(zero=False)),
        x2=alt.X2("End (s):Q"),
        y=alt.Y("Feature:N", sort=chart_data["Feature"].tolist(), title=None,
                axis=alt.Axis(labelLimit=220, labelFontSize=11, labelPadding=8)),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Faster", "Slower"], range=["#53b7a5", "#e34242"]), legend=None),
        tooltip=["Feature:N", alt.Tooltip("Contribution (s):Q", format="+.3f"),
                 alt.Tooltip("Start (s):Q", format=".3f"), alt.Tooltip("End (s):Q", format=".3f"), "Effect:N"],
    ).properties(height=max(260, 42 * len(chart_data)))
    references = pd.DataFrame({
        "value": [explanation["baseline_s"], explanation["prediction_s"]],
        "label": ["Background baseline", "Model prediction"],
    })
    rules = alt.Chart(references).mark_rule(strokeDash=[4, 3]).encode(
        x=alt.X("value:Q"), color=alt.Color("label:N", scale=alt.Scale(range=["#aab2bd", "#e34242"]), title=None),
        tooltip=["label:N", alt.Tooltip("value:Q", format=".3f")],
    )
    st.altair_chart((chart + rules).properties(height=max(300, 42 * len(chart_data))), use_container_width=True)
    st.caption("SHAP describes model behavior, not physical causation. Correlated specifications and engineered ratios can make individual attributions sensitive to background assumptions.")


def show_shap(row: pd.DataFrame, results: dict, context: str):
    """Show cached TreeSHAP output when available without blocking predictions."""
    try:
        explainer = cached_shap_explainer(
            results["shap_model_key"], results["shap_background_key"],
            results["models"]["Random forest"], results["shap_background"],
        )
        explanation, error = safe_explain_rf_prediction(
            results["models"]["Random forest"], row, results["shap_background"], explainer
        )
        if error:
            st.warning(f"SHAP explanation is unavailable for this input ({error}). Prediction output remains valid.")
        else:
            shap_chart(explanation, context)
    except Exception as exc:
        st.warning(f"SHAP explanation is unavailable for this input ({type(exc).__name__}). Prediction output remains valid.")


def navigate_to(destination: str, tuner_label: str | None = None,
                compare_labels: tuple[str, str] | None = None):
    """Update workspace state in a button callback before widgets are created."""
    if tuner_label is not None:
        st.session_state["tuner_vehicle"] = tuner_label
    if compare_labels is not None:
        st.session_state["compare_vehicle_one"], st.session_state["compare_vehicle_two"] = compare_labels
    st.session_state["workspace"] = destination


@st.cache_resource(show_spinner="Preparing verified records and evaluating models…")
def load_analysis():
    artifact_files = {
        "models": ARTIFACTS / "models.joblib",
        "metrics": ARTIFACTS / "model_metrics.csv",
        "ablation": ARTIFACTS / "feature_ablation.csv",
        "importance": ARTIFACTS / "feature_importance.csv",
        "predictions": ARTIFACTS / "holdout_predictions.csv",
        "data": ARTIFACTS / "clean_dataset.csv",
        "manifest": ARTIFACTS / "run_manifest.json",
    }
    if all(path.exists() for path in artifact_files.values()):
        import json

        manifest = json.loads(artifact_files["manifest"].read_text())
        runtime = manifest.get("runtime", {})
        saved_sklearn = runtime.get("scikit_learn")
        if saved_sklearn and saved_sklearn != sklearn.__version__:
            raise ArtifactCompatibilityError(
                f"These model artifacts were created with scikit-learn {saved_sklearn}, "
                f"but this app is running scikit-learn {sklearn.__version__}."
            )
        predictions = pd.read_csv(artifact_files["predictions"])
        data = pd.read_csv(artifact_files["data"])
        if "power_hp" not in predictions:
            extra = data[["car_id", "power_hp", "torque_nm", "weight_kg"]]
            predictions = predictions.merge(extra, on="car_id", how="left", validate="one_to_one")
        cv_paths = [ARTIFACTS / name for name in (
            "cross_validation_folds.csv", "cross_validation_summary.csv",
            "cross_validation_manifest.json",
        )]
        if not all(path.exists() for path in cv_paths):
            save_cross_validation(data)
        calibration_paths = [ARTIFACTS / name for name in (
            "calibration_rf.joblib", "calibration_residuals.csv",
            "calibrated_holdout_intervals.csv", "calibration_manifest.json",
        )]
        if not all(path.exists() for path in calibration_paths):
            save_group_conformal_artifacts(data)
        training = original_training_partition(data)
        background = training.sort_values("car_id", kind="mergesort").sample(
            n=min(100, len(training)), random_state=42
        )[ ["power_hp", "torque_nm", "weight_kg", "power_hp_per_1000kg",
            "torque_nm_per_1000kg", "drivetrain"] ].reset_index(drop=True)
        background_hash = hashlib.sha256(
            pd.util.hash_pandas_object(background, index=True).values.tobytes()
        ).hexdigest()
        try:
            models = joblib.load(artifact_files["models"])
            calibration_model = joblib.load(ARTIFACTS / "calibration_rf.joblib")
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as exc:
            raise ArtifactCompatibilityError(
                f"Could not deserialize a saved model ({type(exc).__name__}: {exc}). "
                "The artifacts were likely created with a different scikit-learn version."
            ) from exc
        return {
            "models": models,
            "metrics": pd.read_csv(artifact_files["metrics"]),
            "ablation": pd.read_csv(artifact_files["ablation"]),
            "importance": pd.read_csv(artifact_files["importance"]),
            "predictions": predictions,
            "data": data,
            "cv_summary": pd.read_csv(ARTIFACTS / "cross_validation_summary.csv"),
            "cv_folds": pd.read_csv(ARTIFACTS / "cross_validation_folds.csv"),
            "cv_manifest": json.loads((ARTIFACTS / "cross_validation_manifest.json").read_text()),
            "manifest": manifest,
            "split": manifest["split"],
            "input_support": manifest["training_input_support"],
            "calibration_model": calibration_model,
            "calibration_residuals": pd.read_csv(ARTIFACTS / "calibration_residuals.csv")["absolute_residual_s"].to_numpy(),
            "calibration_manifest": json.loads((ARTIFACTS / "calibration_manifest.json").read_text()),
            "shap_background": background,
            "shap_model_key": str((ARTIFACTS / "models.joblib").stat().st_mtime_ns),
            "shap_background_key": background_hash,
        }
    fresh = run_analysis()
    training = original_training_partition(fresh["data"])
    background = training.sort_values("car_id", kind="mergesort").sample(
        n=min(100, len(training)), random_state=42
    )[["power_hp", "torque_nm", "weight_kg", "power_hp_per_1000kg",
       "torque_nm_per_1000kg", "drivetrain"]].reset_index(drop=True)
    fresh["shap_background"] = background
    fresh["shap_background_key"] = hashlib.sha256(
        pd.util.hash_pandas_object(background, index=True).values.tobytes()
    ).hexdigest()
    fresh["shap_model_key"] = str((ARTIFACTS / "models.joblib").stat().st_mtime_ns)
    return fresh


with st.sidebar:
    st.markdown('<div class="eyebrow">TORQUELAB / ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown("### Control room")
    view = st.radio("Workspace", [
        "Home",
        "Performance Predictor", "Performance Tuner", "Explore Data", "Compare Cars",
        "Leaderboards", "Model Evaluation & Diagnostics", "Method & Data",
    ], label_visibility="collapsed", key="workspace")
    st.divider()
    st.caption("Target")
    st.markdown("**0–60 mph · seconds**")
    st.caption("Gasoline only · Car and Driver source only")
    if st.button("Rebuild analysis", use_container_width=True):
        load_analysis.clear()
        st.rerun()
    st.caption("Fixed seed · grouped holdout · no imputation")
    st.caption("Saved models are reused; grouped cross-validation is written separately.")

if not DATABASE.exists():
    st.error("TorqueLab's source database is missing. Place the Car Performance Database at the path below, then rerun the app.")
    st.code("sources/car-performance-comparison/public/data/car-performance-data.db")
    st.info("Setup: install requirements with `python3 -m pip install -r requirements.txt`, then run `python3 torquelab.py` and `streamlit run app.py`.")
    st.stop()

if not (ARTIFACTS / "models.joblib").exists():
    st.info("Saved model artifacts are missing. TorqueLab will regenerate them from the local source database now. To generate them manually, run `python3 torquelab.py`.")

try:
    results = load_analysis()
except ArtifactCompatibilityError as exc:
    st.error("TorqueLab's saved models are incompatible with this Python environment.")
    st.write(str(exc))
    st.markdown(
        "Activate the project environment, regenerate artifacts with the existing training "
        "pipeline, then start Streamlit from that same environment."
    )
    st.code("source .venv/bin/activate\npython torquelab.py\nstreamlit run app.py")
    st.stop()
data = results["data"]
metrics = results["metrics"]
ablation = results["ablation"]
importance = results["importance"]
predictions = results["predictions"]
manifest = results["manifest"]
split = results["split"]
support = results["input_support"]
summary = manifest["preparation"]

if view != "Home":
    st.markdown('<div class="eyebrow">VEHICLE PERFORMANCE / 0–60 MPH</div>', unsafe_allow_html=True)
    st.title("TorqueLab")

if view == "Home":
    st.markdown('<div class="eyebrow">TORQUELAB / PERFORMANCE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("TorqueLab")
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">REAL VEHICLES · GROUPED VALIDATION · MODEL INSIGHT</div>
      <h1 style="margin:.55rem 0">What makes a car fast?</h1>
      <p>TorqueLab uses real vehicle specifications and machine learning to analyze and predict 0–60 mph acceleration. Explore measured performance, compare vehicles, and inspect what the models learn.</p>
    </div>
    """, unsafe_allow_html=True)
    model_metadata = pd.DataFrame(manifest["models"]).set_index("Model")
    baseline_metric = model_metadata.loc["Horsepower baseline"]
    best_metric = model_metadata.loc["Random forest"]
    improvement = 100 * (baseline_metric["MAE (s)"] - best_metric["MAE (s)"]) / baseline_metric["MAE (s)"]
    home_metrics = [
        ("Best model · holdout MAE", f"{best_metric['MAE (s)']:.3f} s", "Random Forest"),
        ("Held-out R²", f"{best_metric['R²']:.3f}", "281 grouped holdout vehicles"),
        ("Clean vehicles", f"{summary['clean_rows']:,}", "Source-verified gasoline records"),
        ("MAE improvement", f"{improvement:.1f}%", "Versus horsepower-only baseline"),
    ]
    st.markdown('<div class="metric-grid">' + ''.join(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-caption">{note}</div></div>'
        for label, value, note in home_metrics
    ) + '</div>', unsafe_allow_html=True)

    st.markdown("## Featured vehicles")
    st.caption("A few real records from the cleaned dataset. Specifications and measured times are shown as sourced.")
    feature_patterns = [
        ("Ford Mustang", "mustang"), ("Chevrolet Corvette", "corvette"),
        ("Porsche 911", r"\b911\b"), ("BMW M models", r"\bM[34]\b|M[34] Series"),
    ]
    featured = []
    ordered_data = data.sort_values([TARGET, "manufacturer", "model", "year", "car_id"], kind="mergesort")
    for label, pattern in feature_patterns:
        match = ordered_data[ordered_data["model"].astype(str).str.contains(pattern, case=False, regex=True, na=False)]
        if "BMW" in label:
            match = match[match["manufacturer"].astype(str).str.contains("BMW", case=False, na=False)]
        if not match.empty:
            featured.append((label, match.iloc[0]))
    if featured:
        cols = st.columns(min(4, len(featured)))
        for index, (family, vehicle) in enumerate(featured):
            label = f"{vehicle['manufacturer']} {vehicle['model']} · {vehicle['year']} · {vehicle['car_id']}"
            with cols[index % len(cols)]:
                with st.container(border=True):
                    st.markdown(f"#### {family}")
                    st.markdown(f"**{vehicle['manufacturer']} {vehicle['model']}** · {vehicle['year']}")
                    st.caption(f"{vehicle['power_hp']:.0f} hp · {vehicle['drivetrain']} · measured {vehicle[TARGET]:.2f} s")
                    st.button("Tune this vehicle", key=f"home_tune_{vehicle['car_id']}", use_container_width=True,
                              on_click=navigate_to, args=("Performance Tuner", label))
                    options = data.apply(lambda row: f"{row['manufacturer']} {row['model']} · {row['year']} · {row['car_id']}", axis=1)
                    other = next((option for option in options if option != label), label)
                    st.button("Compare this vehicle", key=f"home_compare_{vehicle['car_id']}", use_container_width=True,
                              on_click=navigate_to, args=("Compare Cars", None, (label, other)))
    else:
        st.info("Featured examples are not available in the current cleaned cohort. Explore the complete source-verified data instead.")

    st.markdown("## Go to a workspace")
    nav = st.columns(4)
    for col, label, destination in zip(nav, ["Predict a 0–60 time", "Tune a vehicle", "Explore records", "Compare cars"],
                                      ["Performance Predictor", "Performance Tuner", "Explore Data", "Compare Cars"]):
        col.button(label, key=f"home_nav_{destination}", use_container_width=True,
                   on_click=navigate_to, args=(destination,))

elif view == "Model Evaluation & Diagnostics":
    eval_metrics = [
        ("Clean records", f"{len(data):,}", "Verified gasoline test rows"),
        ("Train / holdout", f"{split['train_rows']:,} / {split['test_rows']:,}", "Rows · grouped by nameplate"),
        ("Families held out", f"{split['test_groups']:,}", "No train/test group overlap"),
        ("Median measured 0–60", f"{data[TARGET].median():.2f} s", "Observed clean-cohort target"),
    ]
    st.markdown('<div class="metric-grid">' + ''.join(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-caption">{caption}</div></div>'
        for label, value, caption in eval_metrics
    ) + '</div>', unsafe_allow_html=True)

    st.markdown("## Holdout model comparison")
    st.markdown('<p class="section-note">All models use the same fixed grouped split. Lower MAE and RMSE are better; higher R² is better.</p>', unsafe_allow_html=True)
    left, right = st.columns([1.2, .8], gap="large")
    with left:
        display_metrics = metrics.copy()
        for column in ["MAE (s)", "RMSE (s)", "R²"]:
            display_metrics[column] = display_metrics[column].map(lambda x: f"{x:.3f}")
        st.dataframe(display_metrics, use_container_width=True, hide_index=True)
        st.caption(f"Holdout target mean {split['target_test_mean_s']:.2f}s · SD {split['target_test_sd_s']:.2f}s · seed {split['random_seed']}")
    with right:
        st.markdown("#### Prediction error by model")
        error_df = predictions[["zero_to_sixty_s"]].copy()
        for model in metrics["Model"]:
            error_df[model] = (predictions[model] - predictions["zero_to_sixty_s"]).abs()
        error_plot = error_df.drop(columns="zero_to_sixty_s").mean().rename("Mean absolute error (s)").rename_axis("Model").reset_index()
        error_chart = alt.Chart(error_plot).mark_bar(color="#f4a340").encode(
            x=alt.X("Mean absolute error (s):Q", title="Mean absolute error (s)"),
            y=alt.Y("Model:N", sort="-x", title=None),
            tooltip=["Model:N", alt.Tooltip("Mean absolute error (s):Q", format=".3f")],
        ).properties(height=180)
        st.altair_chart(error_chart, use_container_width=True)

    st.markdown("## Does each input family help?")
    st.markdown('<p class="section-note">Leave-one-family-out test: positive changes mean removing that family raised MAE or lowered R² versus the full linear model.</p>', unsafe_allow_html=True)
    a, b = st.columns([1.1, .9], gap="large")
    with a:
        st.dataframe(ablation.style.format({
            "MAE without family (s)": "{:.3f}",
            "MAE increase vs full (s)": "{:+.3f}",
            "R² without family": "{:.3f}",
            "R² change vs full": "{:+.3f}",
        }), use_container_width=True, hide_index=True)
    with b:
        chart = ablation.set_index("Removed feature family")[["MAE increase vs full (s)"]]
        ablation_plot = chart.reset_index()
        ablation_chart = alt.Chart(ablation_plot).mark_bar(color="#8db7e2").encode(
            x=alt.X("MAE increase vs full (s):Q", title="MAE increase vs full (s)"),
            y=alt.Y("Removed feature family:N", sort="-x", title=None),
            tooltip=["Removed feature family:N", alt.Tooltip("MAE increase vs full (s):Q", format="+.3f")],
        ).properties(height=180)
        st.altair_chart(ablation_chart, use_container_width=True)
    st.markdown("#### Random forest permutation importance")
    st.markdown('<p class="section-note">Holdout MAE increase after shuffling one input column. Correlated features can share importance.</p>', unsafe_allow_html=True)
    importance_plot = importance.sort_values("MAE increase when shuffled (s)").copy()
    importance_chart = alt.Chart(importance_plot).mark_bar(color="#f4a340").encode(
        x=alt.X("MAE increase when shuffled (s):Q", title="MAE increase when shuffled (s)"),
        y=alt.Y("Feature:N", sort="-x", title=None),
        tooltip=["Feature:N", alt.Tooltip("MAE increase when shuffled (s):Q", format="+.3f")],
    ).properties(height=260)
    st.altair_chart(importance_chart, use_container_width=True)

    st.markdown("## Actual vs predicted")
    chosen_model = st.selectbox("Model", metrics["Model"].tolist(), key="overview_model")
    plot_data = predictions[["zero_to_sixty_s", chosen_model]].rename(columns={"zero_to_sixty_s": "Measured 0–60 (s)", chosen_model: "Predicted 0–60 (s)"})
    st.scatter_chart(plot_data, x="Measured 0–60 (s)", y="Predicted 0–60 (s)", color="#f4a340", height=330)
    st.caption("Points near the diagonal are closer to the measured result. Values shown here are holdout predictions.")

    st.markdown("## Model Error Explorer")
    st.caption("Exploratory diagnostics calculated only from the fixed held-out predictions.")
    diagnostics = model_error_diagnostics(predictions)
    diag_left, diag_right = st.columns(2)
    with diag_left:
        st.markdown("#### Predicted vs actual · Random Forest")
        error_frame = diagnostics["predictions"]
        point = alt.Chart(error_frame).mark_circle(size=55, opacity=.65, color="#4da3ff").encode(
            x=alt.X("actual_s:Q", title="Actual 0–60 (s)"),
            y=alt.Y("predicted_s:Q", title="Predicted 0–60 (s)"),
            tooltip=["manufacturer:N", "model:N", "year:N", alt.Tooltip("actual_s:Q", format=".2f"), alt.Tooltip("predicted_s:Q", format=".2f")],
        )
        lo = min(error_frame["actual_s"].min(), error_frame["predicted_s"].min())
        hi = max(error_frame["actual_s"].max(), error_frame["predicted_s"].max())
        diagonal = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(color="#f4a340", strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
        st.altair_chart((point + diagonal).properties(height=300), use_container_width=True)
    with diag_right:
        st.markdown("#### Residual distribution")
        residual_chart = alt.Chart(diagnostics["predictions"]).mark_bar(color="#c084fc").encode(
            x=alt.X("residual_s:Q", bin=alt.Bin(maxbins=24), title="Prediction − actual (s)"),
            y=alt.Y("count():Q", title="Held-out vehicles"),
            tooltip=[alt.Tooltip("count():Q", title="Vehicles")],
        ).properties(height=300)
        st.altair_chart(residual_chart, use_container_width=True)
    group_left, group_right = st.columns(2)
    with group_left:
        st.markdown("#### MAE by drivetrain")
        st.dataframe(diagnostics["by_drivetrain"].round(3), use_container_width=True, hide_index=True)
    with group_right:
        st.markdown("#### MAE by horsepower range")
        st.dataframe(diagnostics["by_horsepower"].round(3), use_container_width=True, hide_index=True)
    st.markdown("#### Ten largest absolute errors")
    largest = diagnostics["largest_errors"].rename(columns={
        "actual_s": "Actual 0–60 (s)", "predicted_s": "Predicted 0–60 (s)",
        "absolute_error_s": "Absolute error (s)",
    })
    st.dataframe(largest.round(3), use_container_width=True, hide_index=True)
    st.markdown("### Five-fold grouped cross-validation · training partition only")
    st.caption("Each fold refits preprocessing independently. The 281-row final holdout is excluded from CV and remains the final evaluation.")
    st.dataframe(results["cv_summary"].round(3), use_container_width=True, hide_index=True)
    with st.expander("Fold-level CV results"):
        st.dataframe(results["cv_folds"].round(3), use_container_width=True, hide_index=True)
    st.markdown("### Group-calibrated interval check")
    calibration_meta = results["calibration_manifest"]
    interval_cols = st.columns(3)
    interval_cols[0].metric("Nominal coverage", f"{calibration_meta['coverage']:.0%}")
    interval_cols[1].metric("Empirical holdout coverage", f"{calibration_meta['holdout_empirical_coverage']:.1%}",
                            help="Measured once on the unchanged final holdout, after calibration choices were fixed.")
    interval_cols[2].metric("Mean interval width", f"{calibration_meta['holdout_mean_interval_width_s']:.2f} s")
    st.caption("Coverage is evaluated for the separate calibration Random Forest. Its interval must not be interpreted as an interval around the original production Random Forest estimate.")

elif view == "Performance Tuner":
    st.markdown("## Performance Tuner")
    st.markdown('<p class="section-note">Choose an observed vehicle, then explore hypothetical horsepower and curb weight changes.</p>', unsafe_allow_html=True)
    choices = data.sort_values(["manufacturer", "model", "year", "car_id"], kind="mergesort").copy()
    choices["vehicle_label"] = choices.apply(lambda row: f"{row['manufacturer']} {row['model']} · {row['year']} · {row['car_id']}", axis=1)
    selected_label = st.selectbox("Real vehicle", choices["vehicle_label"].tolist(),
                                 index=choices["vehicle_label"].tolist().index(st.session_state["tuner_vehicle"]) if st.session_state.get("tuner_vehicle") in set(choices["vehicle_label"]) else 0,
                                 key="tuner_vehicle")
    vehicle = choices.loc[choices["vehicle_label"] == selected_label].iloc[0]
    original_weight_lb = float(vehicle["weight_kg"] / LB_TO_KG)
    values = st.columns(5)
    for col, label, value in zip(values,
        ["Horsepower", "Torque", "Curb weight", "Drivetrain", "Measured 0–60"],
        [f"{vehicle['power_hp']:,.0f} hp", f"{vehicle['torque_nm'] / 1.3558179483:,.0f} lb-ft",
         f"{original_weight_lb:,.0f} lb", vehicle["drivetrain"], f"{vehicle[TARGET]:.2f} s"]):
        col.metric(label, value)
    tune_left, tune_right = st.columns(2)
    with tune_left:
        adjusted_hp = st.slider("Hypothetical horsepower (hp)", 50, 2500, int(round(vehicle["power_hp"])), step=10)
    with tune_right:
        adjusted_weight_lb = st.slider("Hypothetical curb weight (lb)", 1000, 10000, int(round(original_weight_lb / 10) * 10), step=50)
    try:
        tuned = tune_vehicle(results["models"], support, vehicle, adjusted_hp, adjusted_weight_lb)
        left_card, right_card = st.columns(2, gap="large")
        with left_card:
            with st.container(border=True):
                st.markdown("### Original vehicle")
                st.metric("Random Forest prediction", f"{tuned['original_prediction_s']:.2f} s")
                st.write(f"{vehicle['power_hp']:.0f} hp · {vehicle['torque_nm'] / 1.3558179483:.0f} lb-ft · {original_weight_lb:.0f} lb · {vehicle['drivetrain']}")
                st.metric("Power-to-weight", f"{tuned['original_power_hp_per_1000kg']:.1f} hp / 1,000 kg")
        with right_card:
            with st.container(border=True):
                st.markdown("### Modified specifications")
                st.metric("Random Forest prediction", f"{tuned['modified_prediction_s']:.2f} s",
                          delta=f"{tuned['change_s']:+.2f} s vs original", delta_color="inverse")
                st.write(f"{adjusted_hp:.0f} hp · {vehicle['torque_nm'] / 1.3558179483:.0f} lb-ft · {adjusted_weight_lb:.0f} lb · {vehicle['drivetrain']}")
                ratio_change = tuned["modified_power_hp_per_1000kg"] - tuned["original_power_hp_per_1000kg"]
                st.metric("Power-to-weight", f"{tuned['modified_power_hp_per_1000kg']:.1f} hp / 1,000 kg", delta=f"{ratio_change:+.1f} hp / 1,000 kg")
        st.caption("Negative predicted time change indicates a shorter model estimate; positive indicates a regression. This is not a causal mechanical simulation.")
        if st.checkbox("Explain modified estimate (SHAP)", key="tuner_shap"):
            show_shap(tuned["modified_row"], results, "modified Random Forest estimate")
        if tuned["outside_features"]:
            st.warning("Modified specifications are outside the central 90% training range for: " + ", ".join(tuned["outside_features"]) + ". Estimates may be unreliable.")
    except ValueError as exc:
        st.error(str(exc))
    st.info("These are hypothetical model estimates based on changed input specifications. They are not validated mechanical modifications and do not account for safety, drivability, emissions, or component limits.")

elif view == "Explore Data":
    st.markdown("## Vehicle explorer")
    st.markdown('<p class="section-note">Inspect only the source-verified records used by the analysis. Trim labels are preserved as published model strings.</p>', unsafe_allow_html=True)
    makes = ["All manufacturers"] + sorted(data["manufacturer"].unique().tolist())
    selected_make = st.selectbox("Manufacturer", makes)
    filtered = data if selected_make == "All manufacturers" else data[data["manufacturer"] == selected_make]
    minimum, maximum = float(filtered[TARGET].min()), float(filtered[TARGET].max())
    low, high = st.slider("Measured 0–60 range (seconds)", min_value=float(int(minimum)), max_value=float(int(maximum + 1)), value=(float(int(minimum)), float(int(maximum + 1))), step=.1)
    filtered = filtered[filtered[TARGET].between(low, high)]
    st.caption(f"{len(filtered):,} records · {filtered['group_key'].nunique():,} nameplate groups")
    chart_frame = filtered[["manufacturer", "model", "year", "power_hp", TARGET, "drivetrain"]].rename(columns={"power_hp": "Horsepower (hp)", TARGET: "Actual 0–60 (s)"})
    drivetrain_colors = {"FWD": "#f4a340", "RWD": "#4da3ff", "AWD": "#65c98a", "4WD": "#c084fc"}
    scatter = alt.Chart(chart_frame).mark_circle(size=75, opacity=.78).encode(
        x=alt.X("Horsepower (hp):Q", title="Horsepower (hp)"),
        y=alt.Y("Actual 0–60 (s):Q", title="Actual 0–60 (s)"),
        color=alt.Color("drivetrain:N", scale=alt.Scale(domain=list(drivetrain_colors), range=list(drivetrain_colors.values())), title="Drivetrain"),
        tooltip=["manufacturer:N", "model:N", "year:N", alt.Tooltip("Horsepower (hp):Q", format=".0f"), alt.Tooltip("Actual 0–60 (s):Q", format=".2f"), "drivetrain:N"],
    ).properties(height=360)
    st.altair_chart(scatter, use_container_width=True)
    st.dataframe(filtered[["manufacturer", "model", "trim", "year", "power_hp", "torque_nm", "weight_kg", "drivetrain", TARGET]].sort_values(TARGET), use_container_width=True, hide_index=True)

elif view == "Compare Cars":
    st.markdown("## Compare Cars")
    st.caption("Select source-verified records from the cleaned cohort for a side-by-side comparison.")
    ordered = data.sort_values(["manufacturer", "model", "year", "car_id"], kind="mergesort").copy()
    ordered["vehicle_label"] = ordered.apply(lambda row: f"{row['manufacturer']} {row['model']} · {row['year']} · {row['car_id']}", axis=1)
    labels = ordered["vehicle_label"].tolist()
    select_a, select_b = st.columns(2)
    label_a = select_a.selectbox("Vehicle A", labels, index=0, key="compare_vehicle_one")
    initial_b = 1 if len(labels) > 1 else 0
    label_b = select_b.selectbox("Vehicle B", labels, index=initial_b, key="compare_vehicle_two")
    car_a = ordered.loc[ordered["vehicle_label"] == label_a].iloc[0]
    car_b = ordered.loc[ordered["vehicle_label"] == label_b].iloc[0]

    def render_compare_card(car, label):
        with st.container(border=True):
            st.markdown(f"### {car['manufacturer']} {car['model']}")
            st.caption(f"{car['year']} · {car['trim'] if pd.notna(car['trim']) else 'Trim not recorded'} · {car['drivetrain']}")
            row = prediction_frame(car["power_hp"], car["torque_nm"], car["weight_kg"], car["drivetrain"])
            prediction = float(results["models"]["Random forest"].predict(row[[
                "power_hp", "torque_nm", "weight_kg", "power_hp_per_1000kg",
                "torque_nm_per_1000kg", "drivetrain",
            ]])[0])
            cells = st.columns(2)
            cells[0].metric("Horsepower", f"{car['power_hp']:.0f} hp")
            cells[1].metric("Torque", f"{car['torque_nm'] / 1.3558179483:.0f} lb-ft")
            cells[0].metric("Curb weight", f"{car['weight_kg'] / LB_TO_KG:.0f} lb")
            cells[1].metric("Drivetrain", car["drivetrain"])
            cells[0].metric("Power-to-weight", f"{row.iloc[0]['power_hp_per_1000kg']:.1f} hp / 1,000 kg")
            cells[1].metric("Measured 0–60", f"{car[TARGET]:.2f} s")
            st.metric("Existing RF prediction", f"{prediction:.2f} s")
        if st.checkbox(f"Explain {label} prediction (SHAP)", key=f"compare_shap_{label}"):
            show_shap(row, results, f"{label} vehicle")
        return prediction, float(row.iloc[0]["power_hp_per_1000kg"])

    card_a, card_b = st.columns(2, gap="large")
    with card_a:
        pred_a, ratio_a = render_compare_card(car_a, "Vehicle A")
    with card_b:
        pred_b, ratio_b = render_compare_card(car_b, "Vehicle B")
    st.markdown("### Comparison")
    if car_a[TARGET] < car_b[TARGET]:
        st.write(f"**{car_a['model']}** has the faster measured 0–60 time by {car_b[TARGET] - car_a[TARGET]:.2f} s.")
    elif car_b[TARGET] < car_a[TARGET]:
        st.write(f"**{car_b['model']}** has the faster measured 0–60 time by {car_a[TARGET] - car_b[TARGET]:.2f} s.")
    else:
        st.write("Both records have the same measured 0–60 time.")
    if ratio_a != ratio_b:
        better = "Vehicle A" if ratio_a > ratio_b else "Vehicle B"
        st.write(f"{better} has the higher horsepower-to-weight ratio ({max(ratio_a, ratio_b):.1f} hp / 1,000 kg).")
    difference = st.columns(3)
    difference[0].metric("Horsepower · B − A", f"{car_b['power_hp'] - car_a['power_hp']:+.0f} hp")
    difference[1].metric("Torque · B − A", f"{(car_b['torque_nm'] - car_a['torque_nm']) / 1.3558179483:+.0f} lb-ft")
    difference[2].metric("Weight · B − A", f"{(car_b['weight_kg'] - car_a['weight_kg']) / LB_TO_KG:+.0f} lb")
    st.caption(f"Random Forest estimates: Vehicle A {pred_a:.2f} s · Vehicle B {pred_b:.2f} s. These are model predictions, not measurements.")

elif view == "Leaderboards":
    st.markdown("## Car Leaderboards")
    ranking = st.selectbox("Rank by", ["Fastest measured 0–60 mph", "Highest horsepower", "Best horsepower-to-weight"])
    filter_cols = st.columns(2)
    make = filter_cols[0].selectbox("Manufacturer filter", ["All manufacturers"] + sorted(data["manufacturer"].unique().tolist()))
    drive = filter_cols[1].selectbox("Drivetrain filter", ["All drivetrains"] + sorted(data["drivetrain"].unique().tolist()))
    ranked = leaderboard(data, ranking, make, drive).head(100).copy()
    ranked["Torque (lb-ft)"] = ranked["torque_nm"] / 1.3558179483
    ranked["Curb weight (lb)"] = ranked["weight_kg"] / LB_TO_KG
    ranked = ranked.rename(columns={"manufacturer": "Manufacturer", "model": "Model", "year": "Year", "drivetrain": "Drivetrain", "power_hp": "Horsepower (hp)", TARGET: "Measured 0–60 (s)"})
    st.dataframe(ranked[["Manufacturer", "Model", "Year", "Drivetrain", "Horsepower (hp)", "Torque (lb-ft)", "Curb weight (lb)", "Measured 0–60 (s)"]].round(1), use_container_width=True, hide_index=True)
    st.caption(f"Showing up to 100 distinct manufacturer/model/year records · {ranking.lower()}.")

elif view == "Performance Predictor":
    st.markdown("## Performance Predictor")
    st.markdown('<p class="section-note">Enter an illustrative vehicle configuration to estimate 0–60 mph acceleration. Defaults are rounded example inputs, not specifications for a particular real vehicle; inputs are not added to training data.</p>', unsafe_allow_html=True)
    coverage = st.selectbox("Prediction interval coverage", [0.80, 0.90, 0.95], index=1,
                            format_func=lambda value: f"{value:.0%}", key="prediction_coverage")
    with st.form("predict_vehicle"):
        c1, c2, c3 = st.columns(3)
        hp = c1.number_input("Horsepower (hp)", min_value=50.0, max_value=2500.0, value=300.0, step=10.0)
        torque_lb_ft = c2.number_input("Torque (lb-ft)", min_value=1.0, max_value=3000.0, value=310.0, step=10.0)
        weight_lb = c3.number_input("Curb weight (lb)", min_value=1000.0, max_value=10000.0, value=3850.0, step=25.0)
        drivetrain = st.selectbox("Drivetrain", ["FWD", "RWD", "AWD", "4WD"], index=2)
        submitted = st.form_submit_button("Estimate acceleration", use_container_width=True)
    if submitted:
        st.session_state["predictor_has_result"] = True
    if submitted or st.session_state.get("predictor_has_result", False):
        try:
            torque_nm = torque_lb_ft * 1.3558179483
            weight_kg = weight_lb * 0.45359237
            row = prediction_frame(hp, torque_nm, weight_kg, drivetrain)
            outside = out_of_distribution_features(support, row)
            if outside:
                st.warning("This input is outside the central 90% training range for: " + ", ".join(outside) + ". Estimates may be unreliable.")
            estimates = predict_vehicle(results["models"], hp, torque_nm, weight_kg, drivetrain)
            calibrated = predict_with_conformal_interval(
                results["calibration_model"], row, results["calibration_residuals"], coverage
            )
            st.markdown("### Acceleration estimate")
            prod, interval = st.columns([1, 1.25], gap="large")
            with prod:
                with st.container(border=True):
                    st.markdown("#### Existing production Random Forest")
                    st.metric("Predicted 0–60 mph", f"{estimates['Random forest']:.2f} s")
                    st.caption("Original model, trained on all 1,119 training records; holdout performance remains the reported production result.")
            with interval:
                with st.container(border=True):
                    st.markdown("#### Group-calibrated estimate")
                    st.metric("Separate calibration RF point", f"{calibrated['point_s']:.2f} s")
                    st.markdown(f"**{coverage:.0%} prediction interval:** {calibrated['lower_s']:.2f}–{calibrated['upper_s']:.2f} s")
                    st.caption("This interval is centered on its own separately fitted calibration RF, not the production RF. Nominal coverage is not guaranteed for an individual vehicle or subgroup.")
            st.markdown("### Vehicle specifications")
            specs = st.columns(6)
            for col, label, value in zip(specs,
                    ["Horsepower", "Torque", "Curb weight", "Drivetrain", "Power / weight", "Torque / weight"],
                    [f"{hp:,.0f} hp", f"{torque_lb_ft:,.0f} lb-ft", f"{weight_lb:,.0f} lb", drivetrain,
                     f"{row.iloc[0]['power_hp_per_1000kg']:.1f} hp / 1,000 kg", f"{row.iloc[0]['torque_nm_per_1000kg']:.1f} N·m / 1,000 kg"]):
                col.metric(label, value)
            st.markdown("### Model estimates")
            columns = st.columns(3)
            for col, name in zip(columns, ["Horsepower baseline", "Expanded linear regression", "Random forest"]):
                col.metric(name, f"{estimates[name]:.2f} s")
            st.caption("All point estimates are model outputs, not measured test results. Confidence and coverage vary by vehicle.")
            if st.checkbox("Explain production Random Forest estimate (SHAP)", key="predictor_shap"):
                show_shap(row, results, "production Random Forest estimate")
        except ValueError as exc:
            st.error(str(exc))
    st.markdown("### Training data ranges")
    ranges = pd.DataFrame([
        {"Measure": "Horsepower", "Minimum": f"{data['power_hp'].min():,.0f} hp", "Maximum": f"{data['power_hp'].max():,.0f} hp"},
        {"Measure": "Torque", "Minimum": f"{data['torque_nm'].min() / 1.3558179483:,.0f} lb-ft", "Maximum": f"{data['torque_nm'].max() / 1.3558179483:,.0f} lb-ft"},
        {"Measure": "Curb weight", "Minimum": f"{data['weight_kg'].min() / 0.45359237:,.0f} lb", "Maximum": f"{data['weight_kg'].max() / 0.45359237:,.0f} lb"},
    ])
    st.dataframe(ranges, use_container_width=True, hide_index=True)
    with st.expander("Full statistical summary (SI units)"):
        st.dataframe(data[["power_hp", "torque_nm", "weight_kg", TARGET]].describe().T, use_container_width=True)

else:
    st.markdown("## Method & data")
    st.caption("Technical preparation, evaluation, and data provenance details.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Target provenance")
        st.write("Rows are accepted only when `propulsion` is exactly `Petrol` and the complete source list is exactly `Car & Driver`. The Car and Driver ETL parser reads its dedicated `zero_to_60_mph` field. Rows from Autoevolution, Kaggle, mixed-source merges, and other sources are excluded because their nominal 0–60 values may be 0–62 mph, converted, or lack field-level provenance.")
        st.markdown("### Preparation")
        st.write("Required fields are kept complete; nothing is imputed. Torque is converted from lb-ft to N·m and curb weight from lb to kg. Drivetrain labels are normalized to FWD, RWD, AWD, and 4WD. Power-to-weight and torque-to-weight use per 1,000 kg units.")
        st.write("Broad checks: horsepower 50–2,500 hp; torque greater than 0 and at most 3,000 lb-ft; curb weight 1,000–10,000 lb; acceleration 1–30 seconds. Malformed or out-of-range rows are dropped whole.")
    with c2:
        st.markdown("### Split and evaluation")
        st.write(f"A single 80/20 `GroupShuffleSplit` uses seed {split['random_seed']}. The group key combines manufacturer with a normalized model-family prefix. All variants in a group stay in either training or holdout; current overlap: **{split['overlapping_groups']}**.")
        st.write("Standard scaling and one-hot encoding are fitted inside scikit-learn pipelines on training rows only. The Random Forest uses the same input set as expanded Linear Regression. The baseline uses horsepower only.")
        st.markdown("### Redistribution status")
        st.warning("The source repository credits third-party data providers, but a blanket license for the compiled vehicle data was not verified. Derived rows and prediction artifacts are written only to the ignored local `artifacts/` directory. Review upstream data terms before publishing or redistributing.")
    with st.expander("Run manifest"):
        st.json({"prepared_rows": len(data), "exclusions": summary, "grouped_split": split, "source_sha256": manifest["dataset_sha256"]})
    with st.expander("Included training records"):
        st.dataframe(data[["car_id", "manufacturer", "model", "trim", "year", "power_hp", "torque_nm", "weight_kg", "drivetrain", TARGET]], use_container_width=True, hide_index=True)

st.divider()
st.caption(f"TorqueLab · Car Performance Analytics  |  Source database fingerprint: `{manifest['dataset_sha256'][:16]}`  |  Generated artifacts are local and ignored by Git.")
