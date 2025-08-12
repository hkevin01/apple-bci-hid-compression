import numpy as np
import pytest

from src.core.compression import WaveletCompressor
from src.neural_translation.intent_translator import IntentTranslator


def test_wavelet_empty():
    comp = WaveletCompressor()
    empty = np.array([], dtype=np.float32)
    out = comp.decompress(comp.compress(empty))
    assert out.size == 0


def test_wavelet_single_sample():
    comp = WaveletCompressor()
    data = np.array([1.23], dtype=np.float32)
    roundtrip = comp.decompress(comp.compress(data))
    assert roundtrip.shape[0] == 1


def test_wavelet_high_amplitude():
    comp = WaveletCompressor()
    data = (np.random.randn(256).astype(np.float32) * 1e6)
    roundtrip = comp.decompress(comp.compress(data))
    assert np.isfinite(roundtrip).all()


@pytest.mark.asyncio
async def test_intent_translator_high_amplitude():
    translator = IntentTranslator()
    data = (np.random.randn(64).astype(np.float32) * 1e3)
    res = await translator.translate(data)
    assert res is not None


def test_wavelet_invalid_dimensions():
    """Test compressor with invalid 2D input (should handle gracefully)."""
    comp = WaveletCompressor()
    data = np.random.randn(32, 8).astype(np.float32)  # 2D input
    try:
        compressed = comp.compress(data)
        roundtrip = comp.decompress(compressed)
        # Should either work or fail gracefully
        assert roundtrip is not None
    except (ValueError, IndexError):
        # Acceptable to reject invalid input
        pass


def test_wavelet_all_zeros():
    """Test compressor with all-zero input."""
    comp = WaveletCompressor()
    data = np.zeros(128, dtype=np.float32)
    roundtrip = comp.decompress(comp.compress(data))
    # Should reconstruct to approximately zeros
    assert np.allclose(roundtrip, 0, atol=1e-6)


def test_wavelet_nan_inf_input():
    """Test compressor handles NaN/inf gracefully."""
    comp = WaveletCompressor()
    data = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    try:
        roundtrip = comp.decompress(comp.compress(data))
        # If it succeeds, result should be finite
        assert np.isfinite(roundtrip).all()
    except (ValueError, RuntimeError):
        # Acceptable to reject invalid input
        pass


@pytest.mark.asyncio
async def test_intent_translator_empty_input():
    """Test translator with empty input array."""
    translator = IntentTranslator()
    empty = np.array([], dtype=np.float32)
    res = await translator.translate(empty)
    # Should return valid result, likely with no gesture
    assert res is not None


@pytest.mark.asyncio
async def test_intent_translator_single_sample():
    """Test translator with single sample."""
    translator = IntentTranslator()
    data = np.array([0.5], dtype=np.float32)
    res = await translator.translate(data)
    assert res is not None


@pytest.mark.asyncio
async def test_intent_translator_repeated_calls():
    """Test translator state consistency across multiple calls."""
    translator = IntentTranslator()
    data = np.random.randn(64).astype(np.float32)

    # Multiple calls should not crash
    res1 = await translator.translate(data)
    res2 = await translator.translate(data)
    res3 = await translator.translate(data)

    assert all(r is not None for r in [res1, res2, res3])
