"""Find Image Hashing — fast in-house near-duplicate image hashing.

Three implementations of the FINd 256-bit perceptual hash, plus the
matrix helper they share:

* `find_image_hashing.reference`  — `FINDHasher`, the original research code
* `find_image_hashing.fixed`      — `FINDHasherFixed`, with two index bug fixes
* `find_image_hashing.optimized`  — `FINDHasherOptimized`, numpy-vectorised
* `find_image_hashing.matrix`     — `MatrixUtil` (used by reference + fixed)

The three classes share the same public surface (`fromFile`, `fromImage`)
so they're drop-in interchangeable. Import them directly from the
package for short usage:

    from find_image_hashing import FINDHasherOptimized
    h = FINDHasherOptimized()
    print(h.fromFile("image.jpg"))

or from the submodule for explicit attribution:

    from find_image_hashing.optimized import FINDHasherOptimized

See README and the `notebooks/demo.ipynb` tutorial for end-to-end examples.
"""

from .reference import FINDHasher
from .fixed import FINDHasherFixed
from .optimized import FINDHasherOptimized
from .matrix import MatrixUtil

__all__ = [
    "FINDHasher",
    "FINDHasherFixed",
    "FINDHasherOptimized",
    "MatrixUtil",
]

__version__ = "0.1.0"
