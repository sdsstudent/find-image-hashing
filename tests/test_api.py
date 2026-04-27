"""HTTP-level tests for the FINd FastAPI service.

Closes acceptance criteria T6 (API spec compliance, happy path matches
task.txt p.38 curl example) and T8 (input validation, 0 unhandled
exceptions on five categorical bad inputs).

Five bad-input categories (one test each):
  1. Missing field        → FastAPI auto 422
  2. Wrong content-type   → 400 (our validator)
  3. Oversized upload     → 413
  4. Corrupted JPEG       → 400
  5. Degenerate dimensions (synthetic 0-byte / parse failure path)

Plus three happy-path tests:
  - /compare on a real fixture (matches task.txt JSON shape)
  - /health returns ok + hasher name
  - /version returns the three expected keys
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REAL_IMAGE = FIXTURES_DIR / "01_typical.jpg"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def real_image_bytes() -> bytes:
    return REAL_IMAGE.read_bytes()


# ---------------------------------------------------------------------------
# T6: happy path — must match task.txt p.38 contract
# ---------------------------------------------------------------------------

def test_compare_happy_path_same_image(client, real_image_bytes):
    """Hashing the same image twice must give identical hashes and distance 0.

    Verifies:
      - 200 OK
      - Required fields (image1_hash, image2_hash, distance) per task.txt p.38
      - Hashes are 64-char hex
      - Extension fields (confidence, image{1,2}_meta, threshold_recommendation)
        present and well-formed
    """
    response = client.post(
        "/compare",
        files={
            "image1": ("a.jpg", real_image_bytes, "image/jpeg"),
            "image2": ("b.jpg", real_image_bytes, "image/jpeg"),
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()

    # Required fields per brief
    assert "image1_hash" in data and "image2_hash" in data and "distance" in data
    assert len(data["image1_hash"]) == 64
    assert len(data["image2_hash"]) == 64
    assert data["image1_hash"] == data["image2_hash"]
    assert data["distance"] == 0

    # Extension fields (D7)
    assert data["confidence"] == "high"  # distance 0 < 50 → high
    assert "image1_meta" in data and "image2_meta" in data
    assert data["image1_meta"]["size_bytes"] == len(real_image_bytes)
    assert "threshold_recommendation" in data
    assert set(data["threshold_recommendation"].keys()) == {
        "high_precision", "balanced", "high_recall",
    }


def test_compare_happy_path_different_images(client, real_image_bytes):
    """Different image bytes (one corrupted artifact via re-encoding) should
    still hash successfully and return a non-negative distance."""
    # Re-encode at lower quality to perturb pixels slightly
    img = Image.open(io.BytesIO(real_image_bytes)).copy()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=10)
    perturbed = buf.getvalue()

    response = client.post(
        "/compare",
        files={
            "image1": ("a.jpg", real_image_bytes, "image/jpeg"),
            "image2": ("b.jpg", perturbed, "image/jpeg"),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["distance"] <= 256
    assert data["confidence"] in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# T8: five categorical bad inputs — 0 unhandled exceptions
# ---------------------------------------------------------------------------

def test_compare_missing_image1_field(client, real_image_bytes):
    """Category 1: required field missing → FastAPI auto 422."""
    response = client.post(
        "/compare",
        files={"image2": ("b.jpg", real_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


def test_compare_wrong_content_type(client, real_image_bytes):
    """Category 2: text file masquerading as image → our validator returns 400."""
    response = client.post(
        "/compare",
        files={
            "image1": ("hello.txt", b"hello world", "text/plain"),
            "image2": ("b.jpg", real_image_bytes, "image/jpeg"),
        },
    )
    assert response.status_code == 400
    assert "not an image" in response.json()["detail"]


def test_compare_oversized_upload(client, real_image_bytes):
    """Category 3: payload exceeds 10 MB limit → 413 Payload Too Large."""
    oversized = b"\x00" * (11 * 1024 * 1024)  # 11 MB
    response = client.post(
        "/compare",
        files={
            "image1": ("big.jpg", oversized, "image/jpeg"),
            "image2": ("b.jpg", real_image_bytes, "image/jpeg"),
        },
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_compare_corrupted_jpeg(client, real_image_bytes):
    """Category 4: bytes that look like JPEG but cannot be decoded → 400."""
    corrupted = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # JPEG SOI + garbage
    response = client.post(
        "/compare",
        files={
            "image1": ("bad.jpg", corrupted, "image/jpeg"),
            "image2": ("b.jpg", real_image_bytes, "image/jpeg"),
        },
    )
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower()


def test_compare_empty_upload(client, real_image_bytes):
    """Category 5 (degenerate input variant): zero-byte upload → 400 via
    parse-failure path. PIL cannot decode 0 bytes; raises in validate_image."""
    response = client.post(
        "/compare",
        files={
            "image1": ("empty.jpg", b"", "image/jpeg"),
            "image2": ("b.jpg", real_image_bytes, "image/jpeg"),
        },
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# /health and /version smoke tests
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    """/health returns 200 OK with status and loaded hasher name."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["hasher"] == "FINDHasherOptimized"


def test_version_endpoint(client):
    """/version returns three expected keys; values default to 'unknown' when
    not running inside a Docker build."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"find_version", "git_sha", "build_date"}
    assert data["find_version"] == "0.1.0"


def test_docs_endpoint_renders(client):
    """FastAPI auto-generated Swagger UI is accessible at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_module_main_entry_point_imports():
    """`python -m api` entry point imports cleanly and exposes `main()`.

    Doesn't actually call `main()` (would block forever in `uvicorn.run`).
    Verifies the entry point file isn't broken — catches rename / typo
    regressions that would surface only at deploy time otherwise.
    """
    from api import __main__
    assert callable(__main__.main)
