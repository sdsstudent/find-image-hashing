# FINd image hashing

A bit-exact reimplementation of the FINd perceptual image hashing
algorithm. The package ships three implementations (`reference`,
`fixed`, `optimized`); use **`FINDHasherOptimized`** in production —
the other two are kept for benchmarking and as the bug-fix audit trail.

[![tests](https://img.shields.io/badge/tests-66%20passing-brightgreen)](#tests)
[![mypy](https://img.shields.io/badge/mypy-clean-blue)](#type-safety)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What is FINd?

FINd produces a 256-bit perceptual hash of an image: two visually
similar images get hashes that are close in Hamming distance (a small
number of bits differ); two unrelated images get hashes that are far
apart (~half the bits differ).

### Use cases

- **Near-duplicate detection** — catch re-shared content across platforms or galleries.
- **Content moderation** — block known-bad re-uploads without storing the original media.
- **Reverse image search** — find similar images without CNN embeddings or third-party APIs.
- **Cross-platform interoperability** — share hash databases without sharing the images themselves.

---
## Try the demo

A 5-minute walkthrough of the library, CLI, and API is in
[`notebooks/demo.ipynb`](notebooks/demo.ipynb). Open it in Jupyter
to see hash computation step-by-step, near-duplicate detection on
sample images, and how to call the FastAPI service from Python.

```bash
pip install -e ".[dev]"
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
- **Contributions**: pull requests welcome — we run `pytest tests/` +
  `mypy --ignore-missing-imports` before merging anything
- **Security disclosures**: please report privately rather than via
  public issue tracker, especially for adversarial-input findings

---

## Quickstart

### Run via Docker

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
  "threshold_recommendation": {"high_precision": 75, "balanced": 90, "high_recall": 110}
}
```

Distance < 75 bits → likely duplicate; > 110 bits → likely different.
See `confidence` for an interpreted bucket and `threshold_recommendation`
for cutoffs derived from ROC analysis on the meme_images dataset.

### Run locally without Docker

```bash
pip install ".[api]"   # library + API runtime deps
python -m api          # starts uvicorn on port 8945
# or: uvicorn api.main:app --port 8945
```

For full development setup (tests, notebooks, profiling tools):

```bash
pip install -e ".[api,dev]"
```

---

## Three usage modes

The same `FINDHasherOptimized` code can be used three ways.

### 1. Python library

```python
from find_image_hashing import FINDHasherOptimized

hasher = FINDHasherOptimized()
hash1 = hasher.fromFile("image_a.jpg")
hash2 = hasher.fromFile("image_b.jpg")
print(f"Distance: {hash1 - hash2} bits out of 256")
```

### 2. Command-line tool

```bash
find-hash --help

find-hash image1.jpg image2.jpg
# 393b246d65a694dc...,image1.jpg
# 18ab6c6f4cae949c...,image2.jpg

find-hash --format json image.jpg
# {"file": "image.jpg", "hash": "393b246d65a694dc..."}

# Pipeline-friendly bare-hash output
find-hash --format hex image.jpg
# 393b246d65a694dc...
```

### 3. HTTP API

```bash
python -m api                              # start server (uvicorn on 8945)
# Then POST to /compare as in the Quickstart above.
# Interactive Swagger UI auto-rendered at: http://localhost:8945/docs
```

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
| `confidence` | "high" / "medium" / "low" | Bucketed from `distance`: `<75` high, `75-110` medium, `≥110` low. Cutoffs calibrated from measured ROC on meme_images (75 ≈ FPR 0.1% / TPR 96%; 110 ≈ FPR 5% / TPR 98%). |
| `image1_meta`, `image2_meta` | `{width, height, size_bytes}` | Per-image metadata for downstream filtering / indexing |
| `threshold_recommendation` | `{high_precision, balanced, high_recall}` | Suggested distance cutoffs for different operating points |

---



## Privacy warning

⚠️ **Note**: The benchmark dataset (Library of Congress meme generator
collection) **may contain images that some viewers find offensive**.
The dataset is **not redistributed with this repository**; users must
download it separately from the original source if they wish to
reproduce the benchmarks.

The Docker image and Python package never bundle the dataset; only the
hashing algorithm and API service are shipped.

---

## Citations

- **Original FINd implementation**:
  <https://github.com/oii-sds-inpractice/summative2026>
- **`imagehash` library** (J. Buchner): <https://github.com/JohannesBuchner/imagehash>
- **Library of Congress meme generator dataset**:
  <https://labs.loc.gov/experiments/webarchive-datasets/>
  ([item page](https://www.loc.gov/item/lcwaN0010226/)).
- **Perceptual hashing background**: Zauner, "Implementation and
  Benchmarking of Perceptual Image Hash Functions" (2010, Master's
  thesis); Steinebach et al., various papers on robustness.

---

## License

[MIT](LICENSE). Free for commercial and non-commercial use, including
modification and redistribution. Copyright assigned to Find Images Now
(FIN), the assignment's fictional client.
