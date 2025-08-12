"""Core compression algorithms for BCI data.

Provides a wavelet-based compressor (PyWavelets if available) with fallback
to a simple hierarchical average/split transform when pywt is absent.
"""

from __future__ import annotations

from typing import List, Tuple, cast

import numpy as np

try:  # Optional dependency
    import pywt  # type: ignore
except ImportError:  # pragma: no cover
    pywt = None


class CompressionAlgorithm:
    """Base class for compression algorithms."""

    def compress(self, data: np.ndarray) -> bytes:
        raise NotImplementedError

    def decompress(self, data: bytes) -> np.ndarray:
        raise NotImplementedError


class WaveletCompressor(CompressionAlgorithm):
    """Wavelet-based neural signal compression.

            Strategy:
                - Per-channel wavelet (or fallback) decomposition.
                - Keep approximation + top-K detail coefficients by magnitude.
                - Store metadata header and sparse coefficients.
                - Lossy sparsification aimed at low latency and reduced size.
            """

    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 3,
        top_k_ratio: float = 0.2,
    ):
        self.wavelet = wavelet
        self.level = level
        self.top_k_ratio = top_k_ratio

    def _dwt_channel(self, channel: np.ndarray) -> List[np.ndarray]:
        if pywt:
            coeffs = pywt.wavedec(channel, self.wavelet, level=self.level)
            return cast(List[np.ndarray], coeffs)
        # Fallback: simple hierarchical averaging (very rough)
        dwt_coeffs: List[np.ndarray] = []
        current = channel.copy()
        for _ in range(self.level):
            if len(current) < 2:
                break
            even = current[::2]
            odd = current[1::2]
            approx = (even + odd) / 2
            detail = (even - odd) / 2
            dwt_coeffs.append(detail)
            current = approx
        dwt_coeffs.append(current)  # final approximation last
        return dwt_coeffs[::-1]

    def _idwt_channel(
        self, coeffs: List[np.ndarray], original_len: int
    ) -> np.ndarray:
        if pywt:
            rec = pywt.waverec(coeffs, self.wavelet)
            return cast(np.ndarray, rec)[:original_len]
        # Fallback inverse of simplistic transform
        coeffs_copy = coeffs.copy()
        approx = coeffs_copy[0]
        details = coeffs_copy[1:]
        for detail in details:
            # Reconstruct pairs
            up_len = len(detail) * 2
            rec = np.zeros(up_len)
            even = approx + detail
            odd = approx - detail
            rec[::2] = even
            rec[1::2] = odd
            approx = rec
        return approx[:original_len]

    def compress(self, data: np.ndarray) -> bytes:
        if data.ndim != 2:
            raise ValueError("Expected 2D array (frames x channels)")
        frames, channels = data.shape
        coeffs_all: List[List[np.ndarray]] = []
        sparse_coeffs: List[np.ndarray] = []
        meta: List[Tuple[int, int]] = []  # (index into flat array, length)
        # Decompose each channel
        for c in range(channels):
            coeffs = self._dwt_channel(data[:, c])
            coeffs_all.append(coeffs)
            # Flatten details then select top-K (excluding first approx)
            approx = coeffs[0]
            details = coeffs[1:]
            flat_details = (
                np.concatenate([d.ravel() for d in details])
                if details
                else np.array([])
            )
            k = (
                max(1, int(len(flat_details) * self.top_k_ratio))
                if len(flat_details)
                else 0
            )
            if k > 0:
                idx = np.argpartition(np.abs(flat_details), -k)[-k:]
                mask = np.zeros_like(flat_details, dtype=bool)
                mask[idx] = True
                sparse = np.zeros_like(flat_details)
                sparse[mask] = flat_details[mask]
            else:
                sparse = flat_details
            # Packed layout: len(approx), approx..., len(flat), sparse...
            chan_vec = np.concatenate([
                np.array([len(approx)], dtype=np.int32).view(np.float32),
                approx.astype(np.float32),
                np.array([len(sparse)], dtype=np.int32).view(np.float32),
                sparse.astype(np.float32)
            ])
            meta.append((len(sparse_coeffs), len(chan_vec)))
            sparse_coeffs.append(chan_vec)
        flat = (
            np.concatenate(sparse_coeffs)
            if sparse_coeffs
            else np.array([], dtype=np.float32)
        )
        header = np.array([
            frames, channels, self.level,
            int(self.top_k_ratio * 1e4)  # store ratio scaled
        ], dtype=np.int32).tobytes()
        return header + flat.tobytes()

    def decompress(self, data: bytes) -> np.ndarray:
        if len(data) < 16:
            return np.array([])
        header = np.frombuffer(data[:16], dtype=np.int32)
        frames, channels, _level, _topk = header
        # Payload floats
        payload = np.frombuffer(data[16:], dtype=np.float32)
        # Reconstruct channel by channel
        out = np.zeros((frames, channels), dtype=np.float32)
        cursor = 0
        for c in range(channels):
            if cursor >= len(payload):
                break
            approx_len = int(payload[cursor].view(np.int32))
            cursor += 1
            approx = payload[cursor:cursor + approx_len]
            cursor += approx_len
            details_len = int(payload[cursor].view(np.int32))
            cursor += 1
            details_sparse = payload[cursor:cursor + details_len]
            cursor += details_len
            if self.level > 0 and details_len > 0:
                per_level = max(1, details_len // self.level)
                coeff_list: List[np.ndarray] = [approx]
                for li in range(self.level):
                    start = li * per_level
                    end = (
                        (li + 1) * per_level
                        if li < self.level - 1
                        else details_len
                    )
                    coeff_list.append(details_sparse[start:end])
            else:
                coeff_list = [approx]
            rec = self._idwt_channel(coeff_list, frames)
            out[:, c] = rec[:frames]
        return out
