"""
Formal latency test for the Disaster Text Classifier /predict endpoint.

Sends 100 sequential POST requests and reports p50, p95, p99, mean, and error rate.
This is a single-threaded test measuring per-request latency, not throughput.

Usage:
  python scripts/latency_test.py
"""

from __future__ import annotations

import statistics
import sys
import time

import requests

PREDICT_URL = "https://disaster-text-service-19133435506.us-east1.run.app/predict"
PAYLOAD = {"text": "Forest fire near La Ronge Sask. Canada"}
NUM_REQUESTS = 100
TIMEOUT_SECONDS = 30


def main() -> None:
    times: list[float] = []
    errors = 0

    print(f"Sending {NUM_REQUESTS} POST requests to {PREDICT_URL} ...")
    print()

    for i in range(NUM_REQUESTS):
        start = time.perf_counter()
        try:
            response = requests.post(
                PREDICT_URL,
                json=PAYLOAD,
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            if "label" not in data or "score" not in data:
                raise ValueError("Response missing expected fields")
        except Exception as exc:
            errors += 1
            print(f"  Request {i + 1:03d}: FAILED ({exc})")
            continue
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1:03d} / {NUM_REQUESTS}")

    print()
    print("=" * 40)
    print("Latency Report")
    print("=" * 40)
    print(f"Total requests:  {NUM_REQUESTS}")
    print(f"Successful:      {len(times)}")
    print(f"Errors:          {errors}")
    print()

    if not times:
        print("No successful requests. Cannot compute statistics.")
        sys.exit(1)

    sorted_times = sorted(times)
    p50 = statistics.median(sorted_times)
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]
    mean = statistics.mean(sorted_times)

    print(f"Mean:            {mean:.3f}s")
    print(f"p50 (median):    {p50:.3f}s")
    print(f"p95:             {p95:.3f}s")
    print(f"p99:             {p99:.3f}s")
    print(f"Min:             {min(times):.3f}s")
    print(f"Max:             {max(times):.3f}s")
    print()

    if p95 < 2.0:
        print("Result: p95 is under the 2-second requirement.")
    else:
        print("Warning: p95 exceeds the 2-second requirement.")


if __name__ == "__main__":
    main()
