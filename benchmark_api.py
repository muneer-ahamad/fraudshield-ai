"""Measure sequential local HTTP round-trip latency, not production throughput."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main(samples):
    if samples < 20:
        raise ValueError("Use at least 20 requests")
    data = pd.read_csv(ROOT / "data/processed/val.csv.gz").sample(frac=1, random_state=42)
    payloads = [dict(TX_AMOUNT=float(r.TX_AMOUNT), TX_HOUR=int(r.TX_HOUR),
        IS_WEEKEND=int(r.IS_WEEKEND), AMOUNT_VS_AVG=float(r.AMOUNT_VS_AVG),
        TX_COUNT_24H=int(r.TX_COUNT_24H)) for r in data.head(samples).itertuples()]
    timings = []
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30, trust_env=False) as client:
        health = client.get("/health"); health.raise_for_status()
        metadata = health.json()
        for i in range(20):
            client.post("/predict", json=payloads[i % len(payloads)]).raise_for_status()
        for i in range(samples):
            start = perf_counter()
            response = client.post("/predict", json=payloads[i % len(payloads)])
            response.raise_for_status()
            result = response.json()
            if result['model_sha256'] != metadata['model_sha256'] or result['threshold'] != metadata['threshold']:
                raise RuntimeError("Model or policy changed during benchmark")
            timings.append((perf_counter() - start) * 1000)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requests": samples, "unique_payloads": len(payloads), "warmup_requests": 20,
        "concurrency": 1, "connection_reused": True, "errors": 0,
        "measurement": "Local HTTP round-trip including request encoding and response decoding; excludes startup",
        "p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)),
        "python": platform.python_version(), "os": platform.platform(),
        "processor": platform.processor(), "server": metadata,
        "note": "Not a production load test; record CPU/RAM and server command with any published result."}
    destination = ROOT / "artifacts/api_latency.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    main(parser.parse_args().samples)
