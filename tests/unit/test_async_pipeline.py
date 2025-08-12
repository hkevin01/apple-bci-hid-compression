import asyncio
import numpy as np
import pytest
from src.core.pipeline.async_end_to_end import async_process_stream, ListFrameStream

@pytest.mark.asyncio
async def test_async_stream_basic():
    frames = [np.random.randn(64).astype(np.float32) for _ in range(5)]
    stream = ListFrameStream(frames)
    events = []
    async for ev in async_process_stream(stream):
        events.append(ev)
    # Not guaranteed to have events; ensure no error and list type
    assert isinstance(events, list)

@pytest.mark.asyncio
async def test_async_stream_empty_and_nonempty():
    frames = [np.array([], dtype=np.float32), np.random.randn(64).astype(np.float32)]
    stream = ListFrameStream(frames)
    async for _ in async_process_stream(stream):
        pass  # just ensure consumption without error

@pytest.mark.asyncio
async def test_async_stream_large_frame():
    frames = [np.random.randn(4096).astype(np.float32)]
    stream = ListFrameStream(frames)
    async for _ in async_process_stream(stream):
        pass
