# FraudShield API: setup and explanation

Copy api/, tests/test_api.py, benchmark_api.py, and requirements-api.txt into
your existing project. Merge tests into the existing tests folder; keep its
other files. Do not replace src, artifacts, or the Streamlit app.

In PowerShell, from the project root with (.venv) active:

```powershell
python -m pip install -r requirements-api.txt
python -m pytest -q
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs. Expand POST /predict, select Try it out,
and submit this illustrative payload (not a labeled dataset example):

```json
{
  "TX_AMOUNT": 250.0,
  "TX_HOUR": 14,
  "IS_WEEKEND": 0,
  "AMOUNT_VS_AVG": 5.0,
  "TX_COUNT_24H": 4
}
```

The actual score depends on your saved model. The response includes score,
threshold, decision, model, and a SHA-256 identifier of the model file. The
threshold is loaded at full precision from threshold_selection.json; it is not
hardcoded. Restart the server after changing local artifacts.

GET /health returns ready only after startup loads and checks the model.
Missing artifacts stop startup rather than returning fake predictions.

The API accepts the five PREPARED features, not raw transactions. Callers must
compute history exactly as in src/features.py: previous average, prior 24-hour
count, and the same timestamp convention. Do not supply TX_FRAUD or customer
IDs. Pydantic validates shape/ranges; it cannot verify that supplied historical
features are truthful or were calculated correctly. Scores are uncalibrated.

The Streamlit app still loads the model directly. This API is a separate
interface to the same model; the dashboard does not depend on it.

## Measure latency

Leave Uvicorn running. Open a second PowerShell terminal in the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python benchmark_api.py --samples 200
```

The script uses validation features, 20 warm-up requests, one connection and
one request at a time. It saves artifacts/api_latency.json (overwritten on
rerun). Errors stop the run rather than being silently excluded.

p50 is the median response time; p95 means 95% of measured requests finished
within that time. This is local HTTP round-trip latency, including validation,
inference, serialization and client overhead. It excludes startup and is not a
production load test. Record CPU, RAM, server command, request count and that
both client/server were on the same machine before using numbers on a resume.
Do not use --reload while benchmarking. Stop Streamlit and other heavy tasks
for a controlled run and report those conditions.

## Concepts to explain in an interview

- FastAPI exposes Python model inference through HTTP.
- Pydantic checks required fields, numeric types, finite values and ranges.
- Lifespan loads the model once rather than once per request.
- predict_proba supplies a score; the frozen threshold supplies the decision.
- A normal def endpoint lets FastAPI run blocking inference in its thread pool.
- Pytest checks invalid inputs, missing artifacts, feature order and the exact
  threshold boundary. These contract tests use a fixture, not the real model.
- The app is a local demo without authentication or public hosting yet.

Only load joblib artifacts you created or trust. No user-provided model uploads
are accepted by this API.
