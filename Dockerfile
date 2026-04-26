## Single-stage production image for the FINd hashing API.
## See private_summative/text.txt section D11 for the full design rationale
## (single-stage vs multi-stage, base image choice, non-root user, etc).

FROM python:3.12-slim

## curl is required for the HEALTHCHECK below; --no-install-recommends and
## the apt-lists cleanup keep the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

## Install runtime-only deps in a separate layer from application code so
## that pure-code edits don't invalidate the (slow) pip-install layer.
## We use requirements-api.txt (subset of requirements.txt) — excludes
## jupyter / nbconvert / matplotlib / pandas / pytest which are dev-only
## and would inflate the image by ~500 MB.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

## Copy only what the API needs at runtime — see .dockerignore for the
## inverse list (dataset, notebooks, fixtures, summaries are excluded).
COPY FINd.py FINd_fixed.py FINd_optimized.py matrix.py ./
COPY api/ ./api/

## Build-time metadata exposed via GET /version (overridable):
##   docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
##                --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) ...
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
ENV GIT_SHA=$GIT_SHA
ENV BUILD_DATE=$BUILD_DATE

## Run as a non-root user — Docker security best practice; if the
## container is ever compromised the attacker has limited privileges.
RUN useradd --create-home --shell /bin/bash apiuser \
    && chown -R apiuser:apiuser /app
USER apiuser

EXPOSE 8945

## Liveness probe consumed by `docker run` and Kubernetes; failure here
## triggers automatic container restart in orchestrated environments.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8945/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8945"]
