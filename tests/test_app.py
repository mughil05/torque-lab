from pathlib import Path
import csv
import hashlib
import json

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from torquelab import ARTIFACTS, LBFT_TO_NM, LB_TO_KG, predict_vehicle


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_views_and_prediction_form_render_against_source_data():
    model_bundle = ARTIFACTS / "models.joblib"
    original_model_sha = hashlib.sha256(model_bundle.read_bytes()).hexdigest()
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    assert model_bundle.exists()

    with (ARTIFACTS / "model_metrics.csv").open(newline="") as handle:
        holdout_metrics = {row["Model"]: row for row in csv.DictReader(handle)}
    assert float(holdout_metrics["Horsepower baseline"]["MAE (s)"]) == pytest.approx(1.148, abs=.001)
    assert float(holdout_metrics["Expanded linear regression"]["MAE (s)"]) == pytest.approx(.983, abs=.001)
    assert float(holdout_metrics["Random forest"]["MAE (s)"]) == pytest.approx(.572, abs=.001)
    cv = pd.read_csv(ARTIFACTS / "cross_validation_summary.csv").set_index("Model")
    assert cv.loc["Random forest", "MAE mean (s)"] == pytest.approx(.505, abs=.001)
    calibration = json.loads((ARTIFACTS / "calibration_manifest.json").read_text())
    assert calibration["fit_calibration_group_overlap"] == 0
    assert calibration["calibration_holdout_group_overlap"] == 0
    assert calibration["holdout_evaluation_rows"] == 281

    home_nav = next(button for button in app.button if button.label == "Predict a 0–60 time")
    home_nav.click().run()
    assert app.radio[0].value == "Performance Predictor"

    # Home and all seven workspaces render from saved, source-derived artifacts.
    pages = ["Home", "Performance Predictor", "Performance Tuner", "Explore Data", "Compare Cars",
             "Leaderboards", "Model Evaluation & Diagnostics", "Method & Data"]
    for page in pages:
        app.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {[error.message for error in app.exception]}"
        assert app.title[0].value == "TorqueLab"
        if page == "Explore Data":
            assert any("records ·" in caption.value for caption in app.caption)
            assert len(app.dataframe) >= 1
        if page == "Model Evaluation & Diagnostics":
            assert any("Five-fold grouped cross-validation" in item.value for item in app.markdown)
            assert len(app.dataframe) >= 3
        if page == "Performance Tuner":
            assert any(metric.label == "Measured 0–60" for metric in app.metric)
            assert any(slider.label == "Hypothetical horsepower (hp)" for slider in app.slider)
            app.slider[0].set_value(2500)
            app.slider[1].set_value(10000)
            app.run()
            assert any("outside the central 90% training range" in warning.value for warning in app.warning)

    app.radio[0].set_value("Performance Predictor").run()
    assert not app.exception
    submit = next(button for button in app.button if button.label == "Estimate acceleration")
    submit.click().run()
    assert not app.exception
    prediction_metrics = [metric for metric in app.metric if metric.label in {
        "Horsepower baseline", "Expanded linear regression", "Random forest"
    }]
    assert len(prediction_metrics) == 3
    assert all(metric.value.endswith(" s") for metric in prediction_metrics)
    assert any("Separate calibration RF point" == metric.label for metric in app.metric)
    assert any("prediction interval" in item.value.lower() for item in app.markdown)

    # App inputs are imperial while the feature builder/model remain SI.
    hp, torque_lb_ft, weight_lb = [item.value for item in app.number_input]
    models = joblib.load(model_bundle)
    expected = predict_vehicle(
        models, hp, torque_lb_ft * LBFT_TO_NM,
        weight_lb * LB_TO_KG, "AWD",
    )
    for metric in prediction_metrics:
        assert float(metric.value.removesuffix(" s")) == pytest.approx(expected[metric.label], abs=.01)
    shap_toggle = next(checkbox for checkbox in app.checkbox if checkbox.label == "Explain production Random Forest estimate (SHAP)")
    shap_toggle.check().run()
    assert not app.exception
    assert any("SHAP describes model behavior" in caption.value for caption in app.caption)

    # A physical but far out-of-distribution horsepower input triggers a
    # visible warning and fails closed if a model estimate leaves 1–30 seconds.
    app.number_input[0].set_value(2500.0)
    next(button for button in app.button if button.label == "Estimate acceleration").click().run()
    assert any("outside the central 90% training range" in warning.value for warning in app.warning)
    assert hashlib.sha256(model_bundle.read_bytes()).hexdigest() == original_model_sha
