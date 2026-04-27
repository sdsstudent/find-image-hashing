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

## Install the library + its API extras via pyproject.toml.
## Using `pip install .` (not editable) so the wheel is built once and
## the source dir doesn't need to stay in the image. Layered cache:
## copy pyproject.toml first so dependency resolution is cached when
## only application code changes.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY api/ ./api/
RUN pip install --no-cache-dir ".[api]"

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
