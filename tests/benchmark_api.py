"""End-to-end T1 latency measurement: 1000 sequential POST /compare.

Closes T1 acceptance bar (evaluation_criteria.md): "wall-clock p95 on
/compare request, both images already in HTTP body, single-core, no GPU,
N=1000 sequential requests".

Distinct from `tests/test_api.py` (which uses FastAPI TestClient in-process,
measures application logic only) and from `notebooks/profile.ipynb` (which
times the hasher's `fromFile` directly with no HTTP transport). This script
measures the full user-facing latency: HTTP transport + uvicorn parsing +
multipart form parsing + 2x hashing + JSON serialization.

Procedure:
  1. Start the API server in another terminal:
        uvicorn api.main:app --host 127.0.0.1 --port 8945
  2. Run this script:
        python tests/benchmark_api.py
  3. Compare output's p95 against T1 acceptance bar (< 100 ms / stretch < 20 ms).
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import numpy as np

URL = "http://127.0.0.1:8945/compare"
HEALTH_URL = "http://127.0.0.1:8945/health"
FIXTURES = Path(__file__).resolve().parent.parent / "meme_images"
IMAGE1 = FIXTURES / "0000_12268686.jpg"
IMAGE2 = FIXTURES / "0000_12270286.jpg"

N_WARMUP = 10
N_REQUESTS = 1000


def main() -> None:
    img1_bytes = IMAGE1.read_bytes()
    img2_bytes = IMAGE2.read_bytes()

    files = {
        "image1": ("a.jpg", img1_bytes, "image/jpeg"),
        "image2": ("b.jpg", img2_bytes, "image/jpeg"),
    }

    with httpx.Client(timeout=10.0) as client:
        # Verify server is reachable before benchmarking.
        r = client.get(HEALTH_URL)
        r.raise_for_status()
        print(f"Server up: {r.json()}")

        # Warmup: triggers singleton hasher init at first request, JIT
        # any caches in uvicorn/Pillow, lets the OS settle network state.
        print(f"Warming up ({N_WARMUP} requests)...")
        for _ in range(N_WARMUP):
            r = client.post(URL, files=files)
            r.raise_for_status()

        # Real measurement.
        print(f"Measuring {N_REQUESTS} sequential requests...")
        latencies_ms: list[float] = []
        bad_status_codes: list[int] = []
        t_start = time.perf_counter()
        for _ in range(N_REQUESTS):
            t0 = time.perf_counter()
            r = client.post(URL, files=files)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            if r.status_code != 200:
                bad_status_codes.append(r.status_code)
        wall_seconds = time.perf_counter() - t_start

    if bad_status_codes:
        raise RuntimeError(
            f"{len(bad_status_codes)} non-200 responses: {bad_status_codes[:5]}"
        )

    arr = np.array(latencies_ms)
    print()
    print(f"=== T1 latency via HTTP — N={N_REQUESTS} sequential ===")
    print(f"  mean      = {arr.mean():.2f} ms")
    print(f"  median    = {np.median(arr):.2f} ms")
    print(f"  p95       = {np.percentile(arr, 95):.2f} ms")
    print(f"  p99       = {np.percentile(arr, 99):.2f} ms")
    print(f"  max       = {arr.max():.2f} ms")
    print(f"  std       = {arr.std():.2f} ms")
    print(f"  min       = {arr.min():.2f} ms")
    print()
    print(f"  total wall-clock = {wall_seconds:.2f} s")
    print(f"  throughput       = {N_REQUESTS / wall_seconds:.1f} req/s (single client)")
    print()
    print(f"  pure compute (per profile.ipynb run2) = 5.09 ms")
    print(f"  HTTP overhead estimate                = {arr.mean() - 2 * 5.09:.2f} ms "
          f"(mean - 2*hash_compute)")


if __name__ == "__main__":
    main()
