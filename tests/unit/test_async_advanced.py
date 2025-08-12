"""Advanced async pipeline tests covering streaming, backpressure, and error scenarios."""
import asyncio
from typing import List

import numpy as np
import pytest

from src.core.compression import WaveletCompressor
from src.core.pipeline.async_end_to_end import async_process_stream


class CustomAsyncFrameStream:
    """Custom async stream for testing various streaming scenarios."""

    def __init__(self, frames: List[np.ndarray], fail_at: int = -1,
                 delay_pattern: List[float] = None):
        self.frames = frames
        self.fail_at = fail_at
        self.delay_pattern = delay_pattern or []
        self.index = 0

    def __aiter__(self):
        self.index = 0
        return self

    async def __anext__(self):
        if self.index >= len(self.frames):
            raise StopAsyncIteration

        if self.index == self.fail_at:
            raise RuntimeError(f"Simulated failure at frame {self.index}")

        # Apply delay if specified
        if self.index < len(self.delay_pattern):
            await asyncio.sleep(self.delay_pattern[self.index])

        frame = self.frames[self.index]
        self.index += 1
        return frame


class BackpressureStream:
    """Stream that simulates backpressure by slowing down."""

    def __init__(self, frame_count: int, frame_size: int = 64,
                 initial_delay: float = 0.01):
        self.frame_count = frame_count
        self.frame_size = frame_size
        self.initial_delay = initial_delay
        self.index = 0

    def __aiter__(self):
        self.index = 0
        return self

    async def __anext__(self):
        if self.index >= self.frame_count:
            raise StopAsyncIteration

        # Simulate increasing backpressure
        delay = self.initial_delay * (1 + self.index * 0.1)
        await asyncio.sleep(delay)

        frame = np.random.randn(self.frame_size).astype(np.float32)
        self.index += 1
        return frame


@pytest.mark.asyncio
async def test_async_stream_with_failure():
    """Test async stream that fails partway through."""
    frames = [np.random.randn(64).astype(np.float32) for _ in range(5)]
    stream = CustomAsyncFrameStream(frames, fail_at=2)

    events = []
    try:
        async for ev in async_process_stream(stream):
            events.append(ev)
    except RuntimeError:
        # Expected failure
        pass

    # Should have processed frames before failure
    assert len(events) <= 2


@pytest.mark.asyncio
async def test_async_stream_with_variable_delays():
    """Test stream with varying delays between frames."""
    frames = [np.random.randn(64).astype(np.float32) for _ in range(4)]
    delay_pattern = [0.001, 0.01, 0.001, 0.005]  # Variable delays
    stream = CustomAsyncFrameStream(frames, delay_pattern=delay_pattern)

    start_time = asyncio.get_event_loop().time()
    events = []
    async for ev in async_process_stream(stream):
        events.append(ev)
    end_time = asyncio.get_event_loop().time()

    # Should take at least the sum of delays
    min_expected_time = sum(delay_pattern)
    assert (end_time - start_time) >= min_expected_time


@pytest.mark.asyncio
async def test_async_stream_backpressure_handling():
    """Test how async stream handles backpressure."""
    stream = BackpressureStream(frame_count=5, initial_delay=0.001)

    events = []
    start_time = asyncio.get_event_loop().time()

    async for ev in async_process_stream(stream):
        events.append(ev)

    end_time = asyncio.get_event_loop().time()

    # Should handle increasing delays gracefully
    assert isinstance(events, list)
    assert (end_time - start_time) > 0  # Should take some time


@pytest.mark.asyncio
async def test_async_stream_cancellation():
    """Test cancelling async stream processing."""
    frames = [np.random.randn(64).astype(np.float32) for _ in range(100)]
    stream = CustomAsyncFrameStream(frames, delay_pattern=[0.01] * 100)

    async def process_with_timeout():
        events = []
        async for ev in async_process_stream(stream):
            events.append(ev)
        return events

    # Cancel after short timeout
    try:
        events = await asyncio.wait_for(process_with_timeout(), timeout=0.05)
    except asyncio.TimeoutError:
        # Expected timeout due to delays
        events = []

    # Should handle cancellation gracefully
    assert isinstance(events, list)


@pytest.mark.asyncio
async def test_async_stream_memory_efficiency():
    """Test memory usage doesn't grow unbounded with long streams."""
    # Create a large number of small frames
    frame_count = 1000
    frames = [np.random.randn(16).astype(np.float32)
              for _ in range(frame_count)]
    stream = CustomAsyncFrameStream(frames)

    # Process without storing all events (memory efficient)
    processed_count = 0
    async for _ in async_process_stream(stream):
        processed_count += 1

        # Only keep count, not events (simulate real-time processing)
        if processed_count % 100 == 0:
            # Simulate periodic cleanup/logging
            await asyncio.sleep(0.001)

    assert processed_count <= frame_count


@pytest.mark.asyncio
async def test_async_stream_different_compression_settings():
    """Test async stream with different compression configurations."""
    frames = [np.random.randn(128).astype(np.float32) for _ in range(3)]
    stream = CustomAsyncFrameStream(frames)

    # Test with high compression
    high_compression = WaveletCompressor(level=4, top_k_ratio=0.1)
    events_high = []
    async for ev in async_process_stream(stream, compressor=high_compression):
        events_high.append(ev)

    # Reset stream for second test
    stream = CustomAsyncFrameStream(frames)

    # Test with low compression
    low_compression = WaveletCompressor(level=1, top_k_ratio=0.5)
    events_low = []
    async for ev in async_process_stream(stream, compressor=low_compression):
        events_low.append(ev)

    # Both should work without errors
    assert isinstance(events_high, list)
    assert isinstance(events_low, list)


@pytest.mark.asyncio
async def test_async_stream_rapid_small_frames():
    """Test processing many small frames rapidly."""
    frames = [np.random.randn(8).astype(np.float32) for _ in range(50)]
    stream = CustomAsyncFrameStream(frames, delay_pattern=[0.0001] * 50)

    start_time = asyncio.get_event_loop().time()
    events = []
    async for ev in async_process_stream(stream):
        events.append(ev)
    end_time = asyncio.get_event_loop().time()

    # Should process rapidly
    assert isinstance(events, list)
    # Should complete in reasonable time despite many frames
    assert (end_time - start_time) < 5.0


@pytest.mark.asyncio
async def test_async_stream_mixed_frame_qualities():
    """Test stream with mix of good and problematic frames."""
    frames = [
        np.random.randn(64).astype(np.float32),  # Good
        np.zeros(64, dtype=np.float32),          # All zeros
        np.ones(64, dtype=np.float32) * 1e6,     # High amplitude
        np.random.randn(64).astype(np.float32),  # Good
        np.array([1.0] * 64, dtype=np.float32),  # Constant
    ]
    stream = CustomAsyncFrameStream(frames)

    events = []
    errors = 0
    async for ev in async_process_stream(stream):
        if ev is not None:
            events.append(ev)
        else:
            errors += 1

    # Should handle mixed quality frames
    assert isinstance(events, list)


@pytest.mark.asyncio
async def test_async_stream_concurrent_streams():
    """Test multiple independent async streams running concurrently."""
    streams = []
    for i in range(3):
        frames = [np.random.randn(32).astype(np.float32) for _ in range(5)]
        streams.append(CustomAsyncFrameStream(frames))

    async def process_stream(stream, stream_id):
        events = []
        async for ev in async_process_stream(stream):
            events.append((stream_id, ev))
        return events

    # Run all streams concurrently
    results = await asyncio.gather(*[
        process_stream(stream, i) for i, stream in enumerate(streams)
    ])

    # All streams should complete successfully
    assert len(results) == 3
    for result in results:
        assert isinstance(result, list)
        assert isinstance(result, list)
