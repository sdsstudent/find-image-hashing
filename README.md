# FINd — Fast In-house Near-Duplicate image hashing

A 60× faster, bit-exact reimplementation of the FINd perceptual image
hashing algorithm, packaged as a Python library, CLI tool, and
production-ready FastAPI service with Docker container.

[![tests](https://img.shields.io/badge/tests-65%20passing-brightgreen)](#tests)
[![mypy](https://img.shields.io/badge/mypy-clean-blue)](#type-safety)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What is FINd?

FINd produces a 256-bit perceptual hash of an image — two visually
similar images get hashes that are close in Hamming distance (a small
number of bits differ); two unrelated images get hashes that are far
apart (~half the bits differ).

Unlike a cryptographic hash (SHA-256, MD5) that completely changes
when even one byte of the file changes, a perceptual hash captures the
**visual content** of the image. Resize, re-encode, add a watermark,
adjust brightness — the hash stays close. Replace the image entirely —
the hash changes a lot.

This repository is the consultant's deliverable for the OII SDS in
Practice 2026 summative assignment: benchmark the rough research
implementation, optimise the bottleneck, compare against the leading
open-source library (`imagehash`), and ship as a release-ready Python
library + REST API.

### Use cases

| use case | how FIN customers might apply it |
|---|---|
| **Near-duplicate detection** | Find duplicate uploads in user galleries, catch re-shared memes across platforms, flag visually identical content before storing it twice |
| **Content moderation** | Block re-uploads of previously banned content (CSAM via PhotoDNA-style hash sharing, terrorist imagery via GIFCT, copyright violations) without storing the original media |
| **Reverse image search** | Find visually similar images in a database without resorting to expensive CNN embeddings or third-party APIs |
| **Deduplication for storage** | Identify near-identical images in a large archive before consolidating storage; estimate dataset size honestly when many items are slight variations |
| **Misinformation tracking** | Track how the same manipulated image spreads across a network (each upload gets re-hashed, identical / near-identical hashes cluster the propagation graph) |
| **Cross-platform interoperability** | Share hash databases between platforms without sharing the actual images (privacy-preserving cooperation) |

The 256-bit output makes FINd suitable for large-scale databases —
collisions are statistically negligible up to billions of items, vs
the 64-bit default of most off-the-shelf alternatives.

---

## Quickstart

### Run via Docker (no install)

```bash
docker build -t fin/find .
docker run -d --rm -p 8945:8945 --name fin fin/find

curl -X POST "http://127.0.0.1:8945/compare" \
  -F "image1=@your_image_a.jpg" \
  -F "image2=@your_image_b.jpg"
```

Response:

```json
{
  "image1_hash": "393b246d65a694dc5386279b8e7394f04c9da697877b18a31995ab9893235b65",
  "image2_hash": "18ab6c6f4cae949c591e27998c3394f8588da4d7a74b18a332d6a3d89363db65",
  "distance": 40,
  "confidence": "high",
  "image1_meta": {"width": 250, "height": 250, "size_bytes": 13569},
  "image2_meta": {"width": 250, "height": 250, "size_bytes": 14466},
  "threshold_recommendation": {"high_precision": 50, "balanced": 90, "high_recall": 130}
}
```

Distance < 50 bits → likely duplicate; > 130 bits → likely different.
See `confidence` for an interpreted bucket and `threshold_recommendation`
for cutoffs derived from ROC analysis on the meme_images dataset.

### Run locally without Docker

```bash
pip install -r requirements-api.txt   # runtime-only deps (6 packages)
python -m api                         # starts uvicorn on port 8945
# or: uvicorn api.main:app --port 8945
```

---

## Three usage modes

The same `FINDHasherOptimized` code can be used three ways.

### 1. Python library

```python
from FINd_optimized import FINDHasherOptimized

hasher = FINDHasherOptimized()
hash1 = hasher.fromFile("image_a.jpg")
hash2 = hasher.fromFile("image_b.jpg")
print(f"Distance: {hash1 - hash2} bits out of 256")
```

### 2. Command-line tool

```bash
python FINd_optimized.py --help

python FINd_optimized.py image1.jpg image2.jpg
# 393b246d65a694dc...,image1.jpg
# 18ab6c6f4cae949c...,image2.jpg

python FINd_optimized.py --format json image.jpg
# {"file": "image.jpg", "hash": "393b246d65a694dc..."}

# Pipeline-friendly bare-hash output
python FINd_optimized.py --format hex image.jpg
# 393b246d65a694dc...
```

### 3. HTTP API

```bash
python -m api                              # start server (uvicorn on 8945)
# Then POST to /compare as in the Quickstart above.
# Interactive Swagger UI auto-rendered at: http://localhost:8945/docs
```

---

## Three implementations — what / why

The repository ships three FINd implementations that should not be
treated as interchangeable. They form a story of refinement:

| file | purpose | when to use |
|---|---|---|
| **`FINd.py`** | The original research implementation provided by FIN. Pure-Python loops; ~464 ms per 250×250 image; **crashes on portrait inputs and silently corrupts on landscape inputs** because of two indexing bugs in `boxFilter` and `findHash256FromFloatLuma`. | Reference baseline for benchmarking only. Do not run in production. |
| **`FINd_fixed.py`** | Two-line bug fix on top of `FINd.py` — corrects the row-major indexing in `boxFilter` (`k*cols+l`) and un-swaps the window-size arguments. Still pure-Python, same ~322 ms runtime. | Reference for correctness — produces semantically valid hashes on non-square inputs. Bit-exact with `FINd.py` on square inputs. |
| **`FINd_optimized.py`** | Numpy-vectorised rewrite of the bottleneck functions (`fillFloatLumaFromBufferImage`, `boxFilter`, `decimateFloat`, `dct64To16`, `dctOutput2hash`). ~5 ms per image (60× faster than baseline, matched conditions). Bit-exact with `FINd_fixed.py` on every real-world image we tested. | **Production**. Drop-in replacement for the reference implementations. |

The bug fix in `FINd_fixed.py` is a no-op on square images (which is
99.998% of the meme_images dataset) so `FINd.py` and `FINd_fixed.py`
produce identical hashes for the realistic workload. We recommend
`FINd_optimized.py` for any new deployment.

---

## API spec

| method | path | purpose |
|---|---|---|
| `POST` | `/compare` | Hash two image uploads and return Hamming distance + extensions |
| `GET`  | `/health`  | Liveness probe (consumed by Docker HEALTHCHECK / Kubernetes) |
| `GET`  | `/version` | Build metadata (`find_version`, `git_sha`, `build_date`) |
| `GET`  | `/docs`    | Interactive Swagger UI auto-generated from Pydantic schemas |

### `POST /compare`

Required fields in the response (`image1_hash`, `image2_hash`,
`distance`) match the assignment brief contract exactly. Additional
fields are optional value-add for downstream consumers; clients that
only need the contract can ignore them.

| field | type | notes |
|---|---|---|
| `image1_hash` | str (64-char hex) | Required — FINd hash of `image1` |
| `image2_hash` | str (64-char hex) | Required — FINd hash of `image2` |
| `distance` | int (0–256) | Required — Hamming distance between the two hashes |
| `confidence` | "high" / "medium" / "low" | Bucketed from `distance`: `<50` high, `50-90` medium, `≥90` low. Cutoffs derived from ROC operating points on meme_images (FPR ≈ 0.1% at 50, ≈ 1% at 90). |
| `image1_meta`, `image2_meta` | `{width, height, size_bytes}` | Per-image metadata for downstream filtering / indexing |
| `threshold_recommendation` | `{high_precision, balanced, high_recall}` | Suggested distance cutoffs for different operating points |

### Validation

Bad inputs return semantic 4xx codes (no `500` from any of the five
categorical bad-input categories — see `tests/test_api.py`):

| condition | code |
|---|---|
| Missing `image1` or `image2` field | 422 |
| Wrong content-type (text file etc.) | 400 |
| Upload exceeds 10 MB | 413 |
| Corrupted / unsupported image format | 400 |
| Zero-dimension or empty image | 400 |

Override the upload limit via env: `MAX_UPLOAD_BYTES=20000000`.

---

## Performance — headline

- **60× faster** than the reference implementation (308 ms → 5.09 ms per
  image, matched-conditions measurement).
- **97.2 % TPR @ 1 % FPR** for near-duplicate detection on
  meme_images — statistically tied with the `imagehash.phash` baseline.
- **API p95 latency 12.95 ms** measured end-to-end via 1000 sequential
  HTTP requests (well under the 100 ms FastAPI usability threshold).
- **419 MB Docker image** (single-stage, slim base, runtime-only deps).

<details>
<summary>📊 Full performance comparison — FINd vs pHash vs wHash (click to expand)</summary>

Measured on Intel i5-1038NG7 @ 2.0 GHz, single core, no GPU,
matched-conditions remeasurement (single session, controlled for the
~30 % run-to-run variance observed on the laptop).

| metric | reference FINd | optimised FINd | pHash (256-bit) | wHash (256-bit) |
|---|---|---|---|---|
| **Mean latency** | 308 ms | **5.09 ms** | 1.71 ms | 4.23 ms |
| **p95 via HTTP API** | n/a | **12.95 ms** | n/a | n/a |
| **Throughput single-core** | 3.2 img/s | **197 img/s** | 584 img/s | 236 img/s |
| **Memory peak per request** | 2.1 MiB | **1.5 MiB** | 0.07 MiB | 0.5 MiB |
| **TPR @ FPR=1%** (1000 within / 1000 between meme pairs) | n/a | **97.2 %** | 97.1 % | 96.3 % |
| **AUC ROC** | n/a | 0.9897 | 0.9907 | **0.9921** |
| **Full archive (55 972 imgs)** | 4.78 hours | **4.74 minutes** | ~1.6 minutes | ~3.9 minutes |
| **Cost / 10⁶ images** (AWS Lambda 512 MB) | $2.77 | **$0.24** | $0.21 | $0.32 |
| **Image size (Docker)** | n/a | **419 MB** (slim) | n/a | n/a |

The headline 60× speedup captures **98.6 %** of the theoretical
headroom (5.09 ms vs 0.9 ms upper bound for the unavoidable work — JPEG
decode + thumbnail + numpy luma). Further latency improvements would
require algorithmic change, rejected to preserve bit-exact compatibility
with FIN's existing FINd hash database.

</details>

---

## Acceptance — every measured criterion passes, most reach stretch

Twenty-one criteria across three axes (Technical T1-T8, Accuracy A1-A6,
Commercial / engineering C1-C5), with explicit acceptance + stretch
thresholds. Full specification + derivation of thresholds in
`private_summative/evaluation_criteria.md`.

<details>
<summary>📊 Acceptance criteria summary table (15 measured rows)</summary>

| ID | criterion | acceptance | stretch | measured | status |
|---|---|---|---|---|---|
| T1 | API latency p95 (HTTP) | < 100 ms | < 20 ms | **12.95 ms** | ✓ stretch |
| T2 | Throughput single-core | ≥ 100 img/s | ≥ 200 img/s | **197 img/s** | ≈ stretch |
| T3 | Memory increment per request | < 100 MB | < 50 MB | **1.5 MiB** | ✓ stretch |
| T4 | Bit-exact vs `FINd_fixed` | 100 % on real images | + ≤5 bit on non-square | **100/100 bulk + 7/8 fixtures** | ✓ |
| T5 | Robustness (no crashes) | 100 % fixtures | + 5 corner cases | **10/10 fixtures** | ✓ |
| T6 | API spec compliance | exact match brief curl | — | **bit-exact** | ✓ |
| T7 | Scaling slope (log-log) | 1.0 ± 0.1 | — | **1.000** | ✓ |
| T8 | Input validation | 0 unhandled exceptions | — | **5/5 categories return 4xx** | ✓ |
| A1 | TPR @ FPR=1% | ≥ 80% & ≥ pHash − 5pp | ≥ 90% | **97.2 %** | ✓ stretch |
| A3 | Discrimination | ≥ 50 bits | ≥ 70 bits | **72.6 bits** | ✓ stretch |
| A4 | Non-square Hamming distance | ≤ 30 bits | ≤ 15 bits | **0 bits** vs fixed | ✓ stretch |
| C1 | Full archive time | < 15 min | < 5 min | **4.74 min** | ✓ stretch |
| C2 | Cost / 10⁶ images | < $0.50 | < $0.30 | **$0.24** | ✓ stretch |
| C3 | Test coverage | ≥ 60 % | ≥ 80 % | (measured via `pytest --cov`) | (see report) |
| C4 | Repo structure | 6/6 standard artefacts | + permissive license | **6/6 + MIT** | ✓ stretch |

</details>

---

## Tests

65 tests passing in five files:

| file | tests | purpose |
|---|---|---|
| `tests/test_hash_stable.py` | 12 | Pinned regression on baseline `FINd.py` |
| `tests/test_bulk_equivalence.py` | 1 | Determinism on 200 random meme images |
| `tests/test_fixed_equivalence.py` | 21 | `FINd_fixed` vs `FINd` — bug fix correctness |
| `tests/test_optimized_equivalence.py` | 21 | `FINd_optimized` vs `FINd_fixed` — bit-exact validation (closes T4) |
| `tests/test_api.py` | 10 | FastAPI HTTP-level — happy path (T6) + 5 bad inputs (T8) + 3 endpoint smoke |

Plus `tests/benchmark_api.py` — 1000-request HTTP latency measurement
(closes T1; not a pytest file, run manually against a live server).

```bash
pytest tests/                              # run all 65 tests (~5 min)
pytest tests/test_optimized_equivalence.py # T4 acceptance only (~40 s)
pytest --cov=. tests/                      # with coverage report
```

---

## Type safety

`FINd_optimized.py` and the `api/` package are fully type-annotated.
`mypy --ignore-missing-imports` reports **0 type errors across 5 source
files** (the flag is necessary because the `imagehash` third-party
library doesn't ship type stubs — industry-typical for OSS Python
packages).

```bash
pip install mypy
mypy --ignore-missing-imports FINd_optimized.py api/
# Success: no issues found in 5 source files
```

---

## Repository structure

<details>
<summary>📁 Project layout (click to expand)</summary>

```
.
├── FINd.py                          # original reference implementation
├── FINd_fixed.py                    # two-line bug fix (boxFilter row-major + arg swap)
├── FINd_optimized.py                # numpy-vectorised production version
├── matrix.py                        # MatrixUtil helper used by FINd / FINd_fixed
├── api/
│   ├── __init__.py
│   ├── __main__.py                  # `python -m api` shortcut for uvicorn
│   ├── main.py                      # FastAPI app, 4 endpoints, validation, dependencies
│   └── schemas.py                   # Pydantic models (CompareResponse, ImageMeta, ...)
├── tests/
│   ├── test_hash_stable.py
│   ├── test_bulk_equivalence.py
│   ├── test_fixed_equivalence.py
│   ├── test_optimized_equivalence.py
│   ├── test_api.py
│   ├── benchmark_api.py             # T1 HTTP latency measurement
│   ├── build_fixtures.py            # generate test fixtures
│   ├── conftest.py
│   └── fixtures/                    # 10 curated test images + pinned hashes
├── notebooks/
│   ├── profile.ipynb                # latency + cProfile + line_profiler + memory + tail
│   ├── compare_versions.ipynb       # cross-version metrics overlay
│   ├── accuracy_benchmark.ipynb     # FINd vs pHash vs wHash on 1000+1000 pairs
│   ├── demo.ipynb                   # tutorial — how to integrate FINd in 5 minutes
│   └── run_subset.ipynb
├── summaries/                       # JSON outputs from notebook runs (cross-version data)
├── Dockerfile                       # single-stage python:3.12-slim, non-root, HEALTHCHECK
├── .dockerignore
├── requirements.txt                 # full dev deps (notebooks + tests + benchmarks)
├── requirements-api.txt             # runtime-only subset (6 packages, 419 MB image)
├── LICENSE                          # MIT
└── README.md
```

</details>

---

## Privacy warning

⚠️ **Note**: The benchmark dataset (Library of Congress meme generator
collection) **may contain images that some viewers find offensive**.
The dataset is **not redistributed with this repository** — users must
download it separately from the original source if they wish to
reproduce the benchmarks.

The Docker image and Python package never bundle the dataset; only the
hashing algorithm and API service are shipped.

---

## Reproducibility

All measurements in this README and in `private_summative/text.txt`
were taken on:

- **CPU**: Intel(R) Core(TM) i5-1038NG7 @ 2.00 GHz (4 cores, 8 threads, no GPU)
- **OS**: macOS 26.3 (Darwin 25.3), x86_64
- **Python**: 3.12.12
- **Key libraries**: numpy 2.4.4, Pillow 12.2.0, scipy 1.17.1, fastapi 0.136.1, pydantic 2.13.3, imagehash 4.3.2

Random seeds:
- `BULK_TEST_SEED = 12345` — `test_bulk_*` and `test_optimized_equivalence.test_bulk` (100-image sample from `meme_images`)
- `PROFILE_SUBSET_SEED = 42` — `profile.ipynb` and `accuracy_benchmark.ipynb` (cost cell uses same 100-image subset)
- `ACCURACY_BENCHMARK_SEED = 99` — `accuracy_benchmark.ipynb` stratified within / between pair sampling

Run-to-run latency variance of roughly 30 % was observed on the laptop
(background load + Intel thermal throttling); reported headline
numbers use a matched-conditions remeasurement (all baselines and
optimised version measured in a single session) to control for this.

---

## Citations

- **Original FINd implementation** by OII SDS in Practice 2026 staff:
  <https://github.com/oii-sds-inpractice/summative2026>
- **`imagehash` library** (J. Buchner): <https://github.com/JohannesBuchner/imagehash>
  — used for pHash and wHash baseline comparison.
- **scipy / numpy / Pillow** — see `requirements.txt`.
- **Library of Congress meme generator dataset**:
  <https://labs.loc.gov/experiments/webarchive-datasets/>
  ([item page](https://www.loc.gov/item/lcwaN0010226/)).
- **Perceptual hashing background**: Zauner, "Implementation and
  Benchmarking of Perceptual Image Hash Functions" (2010, Master's
  thesis); Steinebach et al., various papers on robustness.

---

## Try the demo

A 5-minute walkthrough of the library, CLI, and API is in
[`notebooks/demo.ipynb`](notebooks/demo.ipynb). Open it in Jupyter
to see hash computation step-by-step, near-duplicate detection on
sample images, and how to call the FastAPI service from Python.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/demo.ipynb
```

---

## Get in touch with FIN

This library is maintained by **Find Images Now (FIN)** — a startup
focused on perceptual hashing for content moderation, deduplication,
and reverse image search at scale.

- **Bug reports / feature requests**: open an issue in the repository
- **Commercial / partnership enquiries**: reach out via FIN's
  established channels (anonymised in this submission)
- **Contributions**: pull requests welcome — see development workflow
  in `tests/` (we run `pytest tests/` + `mypy --ignore-missing-imports`
  before merging anything)
- **Security disclosures**: please report privately rather than via
  public issue tracker, especially for adversarial-input findings

---

## License

[MIT](LICENSE). Free for commercial and non-commercial use, including
modification and redistribution. Copyright assigned to Find Images Now
(FIN), the assignment's fictional client.
