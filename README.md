# FraudShield AI

### Explainable fraud detection with an interactive threshold simulator

A Python ML application that scores transactions, explains predictions with SHAP, and shows how decision thresholds change fraud recall and false alarms. Built with **scikit-learn, XGBoost, Streamlit, and FastAPI**.

**[Open the live demo →](https://muneer-fraudshield.streamlit.app/)** · [Run locally](#run-locally) · [Evaluation](#evaluation) · [Prediction API](#prediction-api)

> Portfolio project using synthetic transactions from the Fraud Detection Handbook. The Streamlit dashboard is hosted; the FastAPI service is separately runnable locally.

## Explore the demo

| Page | What to try |
| --- | --- |
| **Overview** | Inspect frozen test metrics and the confusion matrix. |
| **Investigator** | Browse caught fraud, missed fraud, false alarms, and correctly accepted transactions. View the features that raised or lowered the score. |
| **Threshold lab** | Compare the original policy with a stricter preset, or move the cutoff to explore precision–recall trade-offs on validation data. |

No API key is needed. The dashboard loads the saved model directly; it does not depend on a running FastAPI server.

## What this project demonstrates

- **Model comparison:** Logistic Regression with scaling versus XGBoost, with class weighting for rare fraud.
- **Behavioral features:** Spending relative to a customer's previous average and transaction frequency over the preceding 24 hours.
- **Time-based evaluation:** Chronological training, validation, and test splits; model and threshold selection on validation data.
- **Decision trade-offs:** Threshold selection under a false-positive constraint, rather than relying on accuracy or a default 0.5 cutoff.
- **Explainability:** SHAP feature contributions for individual XGBoost predictions.
- **Model serving:** FastAPI, Pydantic input validation, automated tests, and a local HTTP latency benchmark.

## Evaluation

### Held-out test — original policy

XGBoost was selected by validation recall under a **1% false-positive-rate constraint**. The selected threshold was then evaluated on **134,251 held-out synthetic transactions**, including **1,210 fraud cases**.

| Metric | Result |
| --- | ---: |
| Threshold | 0.56126684 |
| Average Precision (AP) | **0.2973** |
| Precision | **24.29%** |
| Fraud recall | **32.73%** |
| False-positive rate | **0.93%** |
| F1 | **0.2789** |
| Fraud caught / missed | **396 / 814** |
| Legitimate transactions incorrectly flagged | **1,234** |

Source: [`artifacts/test_metrics.json`](artifacts/test_metrics.json). AP is reported explicitly; it is not the same calculation as trapezoidal PR-AUC.

### Validation — why the threshold matters

The stricter preset is an exploration on validation data, **not a second held-out test result**.

| XGBoost validation policy | Precision | Recall | False alarms | Fraud caught |
| --- | ---: | ---: | ---: | ---: |
| Original: FPR ≤ 1% | 22.43% | 31.00% | 1,224 | 354 |
| Stricter: FPR ≤ 0.1% | 70.79% | 27.58% | 130 | 315 |

On validation data, the stricter policy reduced false alarms by **89.4%**, while catching **39 fewer fraud cases**. The dashboard recomputes this preset from the saved model and validation data; it does not overwrite the original policy or test report.

Logistic Regression achieved higher validation AP (**0.2923**) than XGBoost (**0.2790**). XGBoost was ultimately selected because it caught more fraud under the chosen 1% FPR constraint. A model's ranking performance and its performance at a particular decision threshold answer different questions.

See [model comparison](artifacts/model_comparison.csv), [validation metrics](artifacts/validation_metrics.json), and [selected policy](artifacts/threshold_selection.json).

### Local API latency

| Measurement | Result |
| --- | ---: |
| Median HTTP round-trip (p50) | **4.53 ms** |
| p95 HTTP round-trip | **5.51 ms** |
| Measured requests / errors | **200 / 0** |

Conditions: 200 unique validation payloads, 20 warm-up requests, concurrency 1, reused connection, Windows 11, Python 3.12.10. Timing includes request encoding and response decoding and excludes startup. This is **local HTTP latency, not hosted latency or a production load test**. Exact CPU model and RAM were not captured in the report.

Source: [`artifacts/api_latency.json`](artifacts/api_latency.json).

## Data and features

The project uses **863,006 synthetic transactions across 90 days** from the [Fraud Detection Handbook dataset](https://github.com/Fraud-Detection-Handbook/simulated-data-raw).

| Split | Dates (2018) | Transactions | Fraud cases |
| --- | --- | ---: | ---: |
| Training | April 1 – June 1 | 594,735 | 4,409 |
| Validation | June 2 – June 15 | 134,020 | 1,142 |
| Test | June 16 – June 29 | 134,251 | 1,210 |

| Model input | Meaning |
| --- | --- |
| `TX_AMOUNT` | Transaction amount in dataset units |
| `TX_HOUR` | Hour of day, 0–23 |
| `IS_WEEKEND` | Weekend indicator, 0 or 1 |
| `AMOUNT_VS_AVG` | Amount divided by the customer's prior average; fallback 1.0 when no usable average exists |
| `TX_COUNT_24H` | Customer transaction count in the preceding 24 hours |

Customer and terminal IDs are display context, not model inputs. Fraud labels and fraud-scenario labels are not predictive features. Historical transaction amounts may carry across chronological split boundaries, as they would be available at scoring time. Scaling is fitted only on training data.

## Run locally

Use **Python 3.12** to match the development environment. Commands below are for Windows PowerShell.

```powershell
git clone https://github.com/muneer-ahamad/fraudshield-ai.git
cd fraudshield-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The repository includes the XGBoost model, validation split, and saved evaluation reports, so the dashboard runs without retraining or downloading the raw dataset.

## Prediction API

In the activated environment:

```powershell
python -m pip install -r requirements-api.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open [interactive API documentation](http://127.0.0.1:8000/docs).

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Confirm the model is loaded; return its name, threshold, and file hash |
| `POST /predict` | Return a score and `flag_for_review` or `not_flagged` decision |

Example request body:

```json
{
  "TX_AMOUNT": 250.0,
  "TX_HOUR": 14,
  "IS_WEEKEND": 0,
  "AMOUNT_VS_AVG": 5.0,
  "TX_COUNT_24H": 4
}
```

The API accepts **prepared features**, not raw customer histories. Callers must compute historical features consistently with `src/features.py`. This example is illustrative, not a labeled fraud example. Scores are uncalibrated, and the service loads the original saved threshold at startup.

To test and benchmark, keep the server running and use a second activated terminal:

```powershell
python -m pytest -q
python benchmark_api.py --samples 200
```

The benchmark writes `artifacts/api_latency.json`. See [`START_API.md`](START_API.md) for details.

## Reproduce training

Use a separate working copy if you want to preserve the shipped model and reports: these commands regenerate data, models, and validation outputs.

```powershell
python -m pip install -r requirements-training.txt -r requirements-api.txt
git clone https://github.com/Fraud-Detection-Handbook/simulated-data-raw data/raw/simulated-data-raw
python -m src.prepare_data --days 90
python -m src.train
python -m src.select_threshold
python -m src.inspect_tradeoffs
```

After fixing the model and policy, `python -m src.evaluate_test` evaluates the test split. It intentionally refuses to overwrite an existing `artifacts/test_metrics.json`: archive the shipped report separately in your experimental copy before a new evaluation. Do not repeatedly tune against the same test set. New policies developed after reviewing test results need a fresh holdout for unbiased confirmation.

## Code map

| File | Responsibility |
| --- | --- |
| `src/features.py` | Five numerical features and customer history |
| `src/prepare_data.py` | Data loading and chronological splitting |
| `src/train.py` | Model training, validation metrics, and model persistence |
| `src/select_threshold.py` | Policy selection under an FPR constraint |
| `src/inspect_tradeoffs.py` | Validation policy comparisons |
| `src/evaluate_test.py` | Frozen-policy test evaluation |
| `streamlit_app.py` | Dashboard, SHAP explanations, and threshold experiments |
| `api/main.py` | Validated prediction API |
| `benchmark_api.py` | Local HTTP benchmark |
| `tests/` | Feature and API contract checks |

## Limitations

- Synthetic patterns do not establish performance on real banking transactions.
- The original policy misses **814 of 1,210 test fraud cases**; low false-positive rate does not imply high fraud coverage.
- The stricter preset's results are validation-only, and FPR constraints do not guarantee future FPR.
- Class-weighted scores are not calibrated fraud probabilities. SHAP explains model behavior in log-odds units, not causal reasons for fraud.
- Feature history uses dataset ordering and timestamp conventions; production event ordering, delayed labels, monitoring, authentication, and live customer-history storage are outside this demo's scope.
- Only the Streamlit dashboard is hosted. FastAPI results are from a separate local service.

## Dataset credit

[Fraud Detection Handbook — simulated dataset documentation](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html) · [Original data repository](https://github.com/Fraud-Detection-Handbook/simulated-data-raw)
