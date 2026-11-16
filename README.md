# TorqueLab v1.2 — Performance Intelligence

TorqueLab is a Streamlit application for exploring real gasoline vehicle performance, predicting 0–60 mph acceleration, comparing cars, and inspecting model behavior. Version 1.2 adds group-aware prediction intervals and individual Random Forest explanations while preserving the original production models and grouped holdout.

> **At a glance:** Predict gasoline-car 0–60 mph performance, compare real vehicles, explore hypothetical tuning, and inspect SHAP explanations. The Random Forest achieved **0.572 s MAE** and **0.896 R²** on a grouped 281-row holdout; group-calibrated intervals are evaluated separately. The public repository contains **code only**: external automotive data must be obtained separately, as explained in [Local data and reproduction](#local-data-and-reproduction).


## What it does

- **Home:** model highlights sourced from generated evaluation metadata, recognizable real-record examples when available, and working workspace shortcuts.
- **Performance Predictor:** estimates 0–60 mph using horsepower, torque, curb weight, and drivetrain. Inputs use hp, lb-ft, and lb and convert to the model's existing internal units.
- **Performance Tuner:** compares the production Random Forest output for an observed vehicle and a hypothetical horsepower/weight scenario. It is not a mechanical simulation.
- **Explore Data, Compare Cars, Leaderboards:** inspect and compare the source-verified cohort without external image or data services.
- **Model Evaluation & Diagnostics:** fixed holdout metrics, held-out error diagnostics, five-fold grouped CV on training rows, and separate calibration-model interval evaluation.
- **SHAP:** optional per-prediction TreeSHAP contributions in seconds, with transformed drivetrain categories shown explicitly.

## Technology stack

Python 3.10+, Streamlit 1.39, pandas 2.2, NumPy 1.26, scikit-learn 1.5, SHAP 0.46, Altair, and pytest. Direct dependencies are pinned in `requirements.txt`.

## Project structure

- `app.py` — Streamlit dashboard and visualizations.
- `torquelab.py` — source preparation, feature engineering, evaluation, calibration, and inference.
- `audit_datasets.py` — read-only data provenance and completeness audit.
- `tests/` — synthetic-unit coverage and source-backed eight-workspace AppTest.
- `artifacts/` and `sources/` — local-only inputs and generated outputs; both are ignored by Git.

## Setup and launch

Use Python 3.10–3.12 from the project root. Run training and Streamlit from the same environment so serialized scikit-learn estimators are loaded by the version that created them:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

SHAP is pinned to `0.46.0`, compatible with the project's NumPy `1.26.4`. If SHAP explanation computation fails at runtime, the app reports that explanation as unavailable while keeping the prediction visible.

## Local data and reproduction

TorqueLab expects the upstream SQLite file at:

```text
sources/car-performance-comparison/public/data/car-performance-data.db
```

The upstream [Car Performance Database repository](https://github.com/geoffmyers/car-performance-comparison) includes this database. From the TorqueLab project root, clone the audited upstream revision to the required directory:

```bash
git clone https://github.com/geoffmyers/car-performance-comparison.git sources/car-performance-comparison
git -C sources/car-performance-comparison checkout 42013d81b4bde7bd508250f2491f096a829c3eb5
test -f sources/car-performance-comparison/public/data/car-performance-data.db
```

The checkout command pins the upstream revision used for the current results. If that revision or database is unavailable, stop: a newer database may yield different records and evaluation results. TorqueLab does not fetch data or call external services.

Once the database is present, reproduce the local artifacts and evaluations with:

```bash
.venv/bin/python torquelab.py
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run app.py
```

The app loads complete saved artifacts without refitting the production model. Its slider interactions do not train models. The explicit “Rebuild analysis” action runs the documented pipeline again. Artifacts and source files are local and ignored by Git.

## Data provenance and limits

The cleaned cohort contains 1,400 records retained only when propulsion is exactly Petrol and the complete source list is exactly Car & Driver. The Car and Driver parser maps a dedicated `zero_to_60_mph` field. Mixed-source rows, Autoevolution 0–62 mph values, and estimated Kaggle conversions are excluded. Required fields must be complete; no values are imputed. Torque and weight are converted internally to N·m and kg; horsepower remains hp.

The upstream database is maintained by Geoff Myers and credits Car and Driver, Wikipedia, fueleconomy.gov (US EPA), Kaggle, and Autoevolution as sources. TorqueLab retains only rows whose complete source list is exactly Car and Driver and uses the dedicated Car and Driver 0–60 mph field; it excludes mixed-source rows and records from the other publishers. TorqueLab is not affiliated with Car and Driver or the upstream project.

### Code and data licensing

TorqueLab's original application code is licensed under the MIT License; see [`LICENSE`](LICENSE). That license applies only to original TorqueLab code. It does not grant rights to the external database, upstream source repositories, third-party material, derived vehicle records, or trained models.

The upstream program is licensed GPL-3.0-or-later, but its documentation credits third-party publishers without establishing blanket redistribution rights for the compiled automotive records. Those data rights remain unresolved. Keep the upstream repository and database, derived datasets and vehicle records, and trained production or calibration models outside Git and do not redistribute them unless the relevant rights are cleared. See the upstream project's [data sources and license](https://github.com/geoffmyers/car-performance-comparison).

To obtain the upstream database for local use, follow the separate clone and checkout instructions above. The database remains outside the TorqueLab Git repository; do not add it as a submodule or copy it into a public release.

## Models and fixed evaluation

The three existing estimators are horsepower-only Linear Regression, expanded Linear Regression, and a 500-tree Random Forest (`min_samples_leaf=2`, `max_features=1.0`, seed 42). The final split remains the original 80/20 seed-42 `GroupShuffleSplit`: **1,119 training rows**, **281 holdout rows**, and zero shared nameplate groups. The holdout remains the final point-prediction evaluation; it is not used for tuning.

| Existing production model | Holdout MAE (s) | Holdout RMSE (s) | Holdout R² |
|---|---:|---:|---:|
| Horsepower-only Linear Regression | 1.148 | 1.477 | 0.632 |
| Expanded Linear Regression | 0.983 | 1.360 | 0.687 |
| Random Forest | 0.572 | 0.783 | 0.896 |

Five-fold `GroupKFold` runs only inside the original 1,119-row training partition. Preprocessing is refit independently within each fold and nameplate groups do not overlap between fold fit and validation sets.

| Model | CV MAE mean ± SD (s) | CV RMSE mean ± SD (s) | CV R² mean ± SD |
|---|---:|---:|---:|
| Horsepower-only Linear Regression | 1.078 ± 0.063 | 1.541 ± 0.057 | 0.559 ± 0.046 |
| Expanded Linear Regression | 0.967 ± 0.060 | 1.447 ± 0.117 | 0.609 ± 0.069 |
| Random Forest | 0.505 ± 0.011 | 0.716 ± 0.038 | 0.905 ± 0.005 |

Both CV and holdout results are exploratory for this source-limited cohort. The model-family grouping is a heuristic based on normalized manufacturer/model prefixes, not a source-provided canonical trim hierarchy.

## Prediction intervals

TorqueLab uses split conformal calibration with absolute errors and group-disjoint partitions taken exclusively from the original training partition. The original production Random Forest remains fit on all 1,119 training rows. A **separate** Random Forest with unchanged hyperparameters is fit on 888 rows, calibrated on 82 nameplate groups (231 vehicle rows), and evaluated once on the untouched 281-row final holdout. Calibration uses the finite-sample order statistic `ceil((n + 1) × coverage)` on each calibration group's maximum absolute residual. The initial nominal coverage is 90%; the UI also allows 80% and 95% quantiles from the same saved calibration scores.

At 90%, the group-score radius is **1.610 s**. On the fixed holdout, the separate calibration RF's intervals covered **94.7% of vehicle rows**; **87.4% of holdout nameplate groups** had every row covered. Mean interval width was **3.220 s**. These are empirical results on one holdout, not a guarantee for a particular vehicle, subgroup, or future population. The app displays the original production RF prediction separately from the calibration RF point and interval; it never attaches the latter's interval to the former's prediction.

The procedure assumes approximate exchangeability of calibration and future nameplate groups. Group dependence, unequal group sizes, source selection, and distribution shifts can weaken nominal coverage. Intervals are model uncertainty estimates, not measurement error bars or physical guarantees.

## SHAP explanations

TreeSHAP explains the actual saved production Random Forest. TorqueLab applies the fitted preprocessing pipeline, including one-hot drivetrain encoding, and explains the transformed forest inputs. A fixed 100-row background sample comes from the original training partition. The baseline plus feature contributions reconstructs the model output in seconds within a `2e-4` second numerical tolerance in tests.

SHAP describes model behavior, not physical causation. Horsepower, weight, torque, and their engineered ratios are correlated; individual attributions can change with feature dependence and the chosen background. Drivetrain dummy columns are shown separately and should not be treated as independent causal effects.

Generated local outputs include `clean_dataset.csv`, the existing model/holdout/CV files, `calibration_rf.joblib`, `calibration_residuals.csv`, `calibrated_holdout_intervals.csv`, and `calibration_manifest.json`. These remain ignored and must not be committed while data rights are unresolved.
