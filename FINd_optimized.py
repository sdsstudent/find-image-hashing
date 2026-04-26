#!/usr/bin/env python
"""Numpy-vectorised FINd hasher.

Drop-in replacement for FINDHasherFixed: same public surface (`fromFile`,
`fromImage`), same 256-bit hash output, bit-exact on every real-world
image we tested (100/100 random meme_images + 7/8 pinned fixtures).

OVERVIEW OF CHANGES vs FINDHasherFixed (see also: text.txt section
OPTIMIZATION DECISIONS for the full trade-off rationale):

  fillFloatLumaFromBufferImage  REWRITTEN: nested for-loop with
                                PIL.Image.getpixel() per pixel  →
                                np.asarray(img.convert("RGB")) @ BT.601 coeffs.
                                Was 35% of baseline runtime; now ~1 ms/img.

  boxFilter                     REWRITTEN: nested for-loop computing
                                window means in O(N * W^2) Python  →
                                summed-area table (integral image) in O(N) numpy.
                                Was 50% of baseline runtime; now ~1-2 ms/img.
                                Mean-centred to avoid catastrophic cancellation
                                on near-constant inputs.

  decimateFloat                 REWRITTEN: double for-loop over 64*64
                                output positions  →  numpy fancy indexing.
                                Trivial speedup but matches algorithmic style.

  dct64To16                     REWRITTEN: triple for-loop matrix multiply  →
                                D @ A @ D.T (BLAS).
                                Was <1% of baseline; rewritten for code
                                cleanliness, not speed.

  dctOutput2hash                REWRITTEN: Torben median-select via Python
                                bisection + double for-loop threshold  →
                                np.partition + boolean masking.

UNCHANGED: thumbnail((512, 512)) — already C-level via PIL.
UNCHANGED: computeBoxFilterWindowSize — already trivial pure-Python.
UNCHANGED: BT.601 coefficients — same float values as the reference
           (we deliberately did not switch to PIL's faster integer-coefficient
           convert("L") because it would shift the hash by 1-3 bits).

ACCEPTANCE: see tests/test_optimized_equivalence.py — 21 tests, all pass.
"""

import math

import numpy as np
from PIL import Image

from imagehash import ImageHash


class FINDHasherOptimized:

    # BT.601 luma coefficients — kept identical to the reference so that
    # luma values are bit-exact (necessary for T4 no-regression).
    LUMA_FROM_R_COEFF = float(0.299)
    LUMA_FROM_G_COEFF = float(0.587)
    LUMA_FROM_B_COEFF = float(0.114)

    # Window divisor matches the 64×64 decimation grid: each output pixel
    # corresponds to one window of width = ceil(input_dim / 64).
    FIND_WINDOW_SIZE_DIVISOR = 64

    def __init__(self):
        # Precompute the 16×64 DCT projection matrix and BT.601 coefficient
        # vector once per hasher; both are immutable and shared across calls.
        self.DCT_matrix = self._compute_dct_matrix()
        self._luma_coeffs = np.array(
            [self.LUMA_FROM_R_COEFF, self.LUMA_FROM_G_COEFF, self.LUMA_FROM_B_COEFF],
            dtype=np.float64,
        )

    @classmethod
    def _compute_dct_matrix(cls):
        # Identical formula to FINDHasher.compute_dct_matrix; we materialise
        # the result as numpy float64 for use in matrix multiplication later.
        d = np.empty((16, 64), dtype=np.float64)
        for i in range(16):
            for j in range(64):
                d[i, j] = math.cos((math.pi / 2 / 64.0) * (i + 1) * (2 * j + 1))
        return d

    @classmethod
    def computeBoxFilterWindowSize(cls, dimension):
        # Round-up division, matching the reference (kept verbatim).
        return int(
            (dimension + cls.FIND_WINDOW_SIZE_DIVISOR - 1)
            / cls.FIND_WINDOW_SIZE_DIVISOR
        )

    def fromFile(self, filepath):
        img = Image.open(filepath)
        return self.fromImage(img)

    def fromImage(self, img):
        # Same pipeline shape as the reference: copy → thumbnail → luma →
        # box-filter → decimate → DCT → threshold-into-hash. Each stage is
        # vectorised; the orchestration here is unchanged.
        img = img.copy()
        img.thumbnail((512, 512))

        luma = self._luma_from_image(img)
        numRows, numCols = luma.shape

        windowSizeAlongRows = self.computeBoxFilterWindowSize(numRows)
        windowSizeAlongCols = self.computeBoxFilterWindowSize(numCols)
        blurred = self._box_filter(
            luma, windowSizeAlongRows, windowSizeAlongCols
        )

        decimated = self._decimate(blurred)
        dct_out = self._dct64_to_16(decimated)
        return self._hash_from_dct(dct_out)

    def _luma_from_image(self, img):
        # REWRITE NOTE: replaces fillFloatLumaFromBufferImage's per-pixel
        # PIL.Image.getpixel() loop. Profiling showed getpixel cost
        # ~10.7 µs/pixel (line_profiler cell 13: 71.9% of function time on
        # one PIL call). One numpy matmul does it at ~1 ns/pixel.
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
        return rgb @ self._luma_coeffs

    @staticmethod
    def _box_filter(arr, rowWin, colWin):
        # REWRITE NOTE: replaces the reference boxFilter (nested for-loops
        # of total cost O(N * W^2)) with a summed-area table of cost O(N).
        # Boundary handling matches the reference exactly: the window is
        # clipped at image bounds and the mean is taken over the actual
        # (clipped) area — uniform_filter / cv2.boxFilter use different
        # boundary conventions and would shift the hash.
        #
        # Mean-centring trick (numerical correctness): the naive integral
        # image accumulates values up to rows*cols*max_pixel (~8e6 for a
        # 250×250 all-128 image) and then takes differences of nearly equal
        # quantities — catastrophic cancellation leaves ~1e-11 noise. By
        # subtracting the mean first the cumulative sums stay bounded; the
        # mean is added back analytically per window, exact for constant
        # inputs.
        halfRowWin = (rowWin + 2) // 2
        halfColWin = (colWin + 2) // 2
        rows, cols = arr.shape

        mean = arr.mean()
        centered = arr - mean

        # S[i, j] = sum of centered[0:i, 0:j]; the extra zero row/col lets
        # us write window_sum = S[xmax, ymax] - S[xmin, ymax] - S[xmax, ymin] + S[xmin, ymin]
        # without special-casing the upper-left edges.
        S = np.zeros((rows + 1, cols + 1), dtype=np.float64)
        np.cumsum(np.cumsum(centered, axis=0), axis=1, out=S[1:, 1:])

        # Per-row and per-column window bounds, clipped at 0 and dim — the
        # reference computes these inside the inner loop with max/min calls.
        i = np.arange(rows)
        j = np.arange(cols)
        xmin = np.maximum(0, i - halfRowWin)
        xmax = np.minimum(rows, i + halfRowWin)
        ymin = np.maximum(0, j - halfColWin)
        ymax = np.minimum(cols, j + halfColWin)

        # Broadcast (rows, 1) and (1, cols) into (rows, cols) to compute
        # all window sums at once via fancy indexing into S.
        window_sum_centered = (
            S[xmax[:, None], ymax[None, :]]
            - S[xmin[:, None], ymax[None, :]]
            - S[xmax[:, None], ymin[None, :]]
            + S[xmin[:, None], ymin[None, :]]
        )
        area = (xmax - xmin)[:, None] * (ymax - ymin)[None, :]
        return window_sum_centered / area + mean

    @staticmethod
    def _decimate(arr):
        # REWRITE NOTE: replaces decimateFloat's nested for-loop. Same
        # indexing formula int(((i + 0.5) * dim) / 64), now applied as
        # integer numpy indices via fancy indexing. <1% of baseline time,
        # rewritten for code style consistency.
        rows, cols = arr.shape
        i_idx = ((np.arange(64) + 0.5) * rows / 64).astype(np.int64)
        j_idx = ((np.arange(64) + 0.5) * cols / 64).astype(np.int64)
        return arr[np.ix_(i_idx, j_idx)]

    def _dct64_to_16(self, A):
        # REWRITE NOTE: replaces dct64To16's triple for-loop (O(16*16*64*64)
        # scalar operations) with two BLAS matmuls. <1% of baseline, but
        # the rewrite keeps the optimized class free of Python loops.
        #
        # CAVEAT: BLAS reorders the inner-product summation, so on inputs
        # whose true DCT is mathematically zero (e.g. a perfectly
        # homogeneous image) the residual roundoff-noise pattern differs
        # from the reference. This is the only documented case where the
        # optimized hash diverges from the fixed hash — see
        # test_optimized_noise_dominated_input.
        return self.DCT_matrix @ A @ self.DCT_matrix.T

    @staticmethod
    def _hash_from_dct(dct_out):
        # REWRITE NOTE: replaces dctOutput2hash. Two changes:
        #   1. MatrixUtil.torben → np.partition. Torben's median-select
        #      returns the lower median (sorted[127]) for a 256-element
        #      array; np.partition with k=127 gives the same value.
        #   2. Per-cell `if dctOutput[i, j] > median: hash[15-i, 15-j] = 1`
        #      → numpy boolean mask + axis-flip slicing.
        flat = dct_out.ravel()
        median = np.partition(flat, 127)[127]

        # Reference indexes hash[15-i, 15-j], i.e. flips both axes of the
        # comparison mask. ::-1 along both axes is the vectorised form.
        mask = (dct_out > median)[::-1, ::-1]
        return ImageHash(mask.ravel().astype(int))


if __name__ == "__main__":
    import sys
    h = FINDHasherOptimized()
    for filename in sys.argv[1:]:
        print(f"{h.fromFile(filename)},{filename}")
