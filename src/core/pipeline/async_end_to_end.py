"""Async streaming end-to-end pipeline utilities."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from src.core.compression import WaveletCompressor
from src.hid_interface import HIDEvent, select_backend
from src.neural_translation.intent_translator import AsyncIntentSession


async def async_process_stream(
    frames: AsyncIterator[np.ndarray],
    compressor: WaveletCompressor | None = None,
) -> AsyncIterator[HIDEvent]:
    """Consume frames and yield HID events when produced."""
    compressor = compressor or WaveletCompressor(level=2, top_k_ratio=0.2)
    session = AsyncIntentSession()
    backend = select_backend()
    async for frame in frames:
        if frame.size == 0:
            continue
        comp = compressor.compress(frame)
        _ = compressor.decompress(comp)
        event = await session.push_and_translate(frame)
        if event:
            backend.send(event)
            yield event


class ListFrameStream:
    """Helper adapter to stream a list of frames asynchronously for testing."""

    def __init__(self, frames: list[np.ndarray], delay: float = 0.0) -> None:
        self.frames = frames
        self.delay = delay
        self._iter: list[np.ndarray] | None = None

    def __aiter__(self) -> ListFrameStream:
        self._iter = list(self.frames)
        return self

    async def __anext__(self) -> np.ndarray:
        try:
            assert self._iter is not None
            if not self._iter:
                raise StopIteration
            nxt = self._iter.pop(0)
        except StopIteration as exc:  # pragma: no cover - control flow
            raise StopAsyncIteration from exc
        if self.delay:
            await asyncio.sleep(self.delay)
        return nxt
