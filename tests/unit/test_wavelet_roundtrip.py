from __future__ import annotations

import numpy as np

from src.core.compression import WaveletCompressor


def test_wavelet_roundtrip_basic() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((256, 4)).astype(np.float32)
    comp = WaveletCompressor(wavelet="db2", level=2, top_k_ratio=0.25)
    blob = comp.compress(data)
    rec = comp.decompress(blob)
    assert rec.shape == data.shape
    # lossy, but correlation should be strong
    corr = np.corrcoef(data.ravel(), rec.ravel())[0, 1]
    assert corr > 0.8


def test_wavelet_roundtrip_empty() -> None:
    comp = WaveletCompressor()
    out = comp.decompress(b"")
    assert out.size == 0
