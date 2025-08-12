"""Async streaming end-to-end pipeline utilities."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional
import numpy as np

from ..compression import WaveletCompressor
from ...neural_translation.intent_translator import AsyncIntentSession
from ...hid_interface import select_backend, HIDEvent


async def async_process_stream(frames: AsyncIterator[np.ndarray], compressor: Optional[WaveletCompressor] = None) -> AsyncIterator[HIDEvent]:
    """Consume an async iterator of frames and yield HID events when produced."""
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
    def __init__(self, frames: list[np.ndarray], delay: float = 0.0):
        self.frames = frames
        self.delay = delay

    def __aiter__(self):
        self._iter = iter(self.frames)
        return self

    async def __anext__(self):
        try:
            nxt = next(self._iter)
        except StopIteration:
            raise StopAsyncIteration
        if self.delay:
            await asyncio.sleep(self.delay)
        return nxt
