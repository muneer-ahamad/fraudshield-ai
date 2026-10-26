# FraudShield AI — Stage 1: Data Preparation

This is the first stage of a beginner-friendly, interview-ready, Python fraud detection project. This is **not yet a trained model or dashboard**. It prepares the training data safely. We will add Logistic Regression, XGBoost, SHAP, a threshold simulator, FastAPI, and Streamlit in later stages.

## Requirements

- Windows PowerShell + Python 3.10–3.12 recommended
- Git installed and accessible from PowerShell

## Installation

Open this folder in VS Code, then run in its terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is not available but Python 3.10+ is installed, use `py -m venv .venv` or `python -m venv .venv`.

If the PowerShell script is blocked, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Get the source dataset

The [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html) provides a simulated payment dataset with transaction timestamps, customer IDs, terminal IDs, amounts, and fraud labels. Download using Git:

```powershell
git clone https://github.com/Fraud-Detection-Handbook/simulated-data-raw data/raw/simulated-data-raw
```

Ensure the daily `.pkl` files exist in `data/raw/simulated-data-raw/data/`. Avoid committing raw data to your own repository.

## Run Stage 1

```powershell
python -m pytest -q
python -m src.prepare_data --days 90
```

If your laptop struggles with memory, use `--days 30` for the first run. For a meaningful final evaluation use 90 days or more.

This saves:

```
data/processed/train.csv.gz
data/processed/val.csv.gz
data/processed/test.csv.gz
```

The program prints transaction counts and fraud rates per split. It does **not** train a model yet.

## How the features work

Five ML inputs:

| Feature | Meaning |
| --- | --- |
| TX_AMOUNT | Transaction amount |
| TX_HOUR | Hour in the day (0–23) |
| IS_WEEKEND | Saturday or Sunday? (0/1) |
| AMOUNT_VS_AVG | Transaction amount divided by customer's prior average (set to 1 if no history) |
| TX_COUNT_24H | Count of the customer's transactions in previous 24 hours |

`CUSTOMER_AVG_AMOUNT_PRIOR` is retained to show an understandable explanation, but is not part of the model input list. Fraud labels (`TX_FRAUD`) are the targets. Identifiers are available for display only and never used as ordered numeric model inputs. `TX_FRAUD_SCENARIO` is not loaded into model features, because it's an obvious target leakage risk.

Train/validation/test sets are split by whole days in chronological order, approximately 70% / 15% / 15%. Feature calculations may use prior transaction **amounts** across split boundaries (which would already be available at prediction time), but never use future transactions or any fraud labels.

## Learning checkpoints

1. Describe what each row in a DataFrame represents.
2. Explain what the target `TX_FRAUD` means.
3. Explain why we cannot simply use 99% accuracy for rare fraud.
4. Explain why a customer's history must exclude the transaction being predicted.
5. Explain why we split by time rather than randomly splitting future and past transactions.

## Original source

Fraud Detection Handbook: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html

Dataset repository: https://github.com/Fraud-Detection-Handbook/simulated-data-raw
