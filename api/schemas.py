"""Pydantic models for the FINd hashing API.

The schema is split into two layers:

1. **Required** (per task.txt p.38) — `image1_hash`, `image2_hash`, `distance`.
   These exactly match the brief's curl example and must remain stable.

2. **Extensions** (consultant value-add) — `confidence`, `image{1,2}_meta`,
   `threshold_recommendation`. Optional fields that surface useful
   information for downstream consumers without breaking the required
   contract; clients that only need the three core fields can ignore them.

See text.txt sections D7 (extensions rationale) and D10 (confidence
threshold derivation) for the full design discussion.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageMeta(BaseModel):
    """Per-image metadata returned alongside its hash.

    Useful for downstream pipelines that want to filter / index without
    re-fetching the original image bytes.
    """

    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    size_bytes: int = Field(..., description="Original upload size in bytes")


class CompareResponse(BaseModel):
    """Response payload for POST /compare.

    The first three fields are the required contract from the assignment
    brief (task.txt p.38) and match the curl example exactly. Extension
    fields below are optional value-add for FIN's consumers — see D7 in
    text.txt for rationale.
    """

    # === REQUIRED (task.txt p.38 contract) ===
    image1_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="64-character hex FINd hash of image1",
    )
    image2_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="64-character hex FINd hash of image2",
    )
    distance: int = Field(
        ..., ge=0, le=256,
        description="Hamming distance between the two hashes (0-256 bits)",
    )

    # === EXTENSIONS (consultant value-add, see D7/D10 in text.txt) ===
    confidence: str = Field(
        ...,
        description="Bucketed confidence label derived from distance via "
                    "ROC-calibrated thresholds: <75 high, 75-110 medium, >=110 low",
    )
    image1_meta: ImageMeta
    image2_meta: ImageMeta
    threshold_recommendation: dict[str, int] = Field(
        default_factory=lambda: {
            "high_precision": 75,
            "balanced": 90,
            "high_recall": 110,
        },
        description="Suggested distance cutoffs for different operating "
                    "points (high precision: FPR≈0.1% / TPR≈96%, balanced: "
                    "FPR≈0% / TPR≈94%, high recall: FPR≈5% / TPR≈98%). "
                    "Calibrated on meme_images; clients should re-validate "
                    "for other domains.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Optional non-fatal warnings about the request. Currently "
                    "populated when an input image has near-zero luma variance "
                    "(uniform / single-colour image), in which case the hash "
                    "is determined by floating-point noise rather than visual "
                    "content and should not be relied on for similarity "
                    "judgements. Empty list = no warnings.",
    )


class HealthResponse(BaseModel):
    """Response payload for GET /health.

    Body is informational only — Docker HEALTHCHECK and Kubernetes
    liveness probes only read the HTTP status code, not the body. The
    `hasher` field exists as a debugging aid (operator can curl /health
    and immediately see which implementation is loaded).
    """

    status: str
    hasher: str


class VersionResponse(BaseModel):
    """Response payload for GET /version.

    Build metadata injected at Docker build time via --build-arg; falls
    back to `unknown` when run outside a build (e.g. local `uvicorn`).
    """

    find_version: str
    git_sha: str
    build_date: str


class ErrorResponse(BaseModel):
    """Standard FastAPI HTTPException response shape (used implicitly
    by FastAPI when raising HTTPException). Documented here for OpenAPI."""

    detail: str
