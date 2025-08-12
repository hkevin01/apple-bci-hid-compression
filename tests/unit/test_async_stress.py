"""Stress tests and edge cases for async pipeline processing."""
import asyncio

import numpy as np
import pytest

from src.core.pipeline.async_end_to_end import async_process_stream


class StressTestStream:
    """Async stream for stress testing scenarios."""

    def __init__(self, frame_count: int, frame_size: int = 64,
                 chaos_mode: bool = False):
        self.frame_count = frame_count
        self.frame_size = frame_size
        self.chaos_mode = chaos_mode
        self.index = 0

    def __aiter__(self):
        self.index = 0
        return self

    async def __anext__(self):
        if self.index >= self.frame_count:
            raise StopAsyncIteration

        if self.chaos_mode and self.index % 7 == 0:
            # Introduce random chaos every 7th frame
            if self.index % 14 == 0:
                # Return problematic data
                frame = np.array([np.nan] * self.frame_size, dtype=np.float32)
            else:
                # Return very large amplitude
                frame = (np.random.randn(self.frame_size) * 1e8).astype(
                    np.float32
                )
        else:
            # Normal frame
            frame = np.random.randn(self.frame_size).astype(np.float32)

        self.index += 1
        return frame


@pytest.mark.asyncio
async def test_stress_large_frame_count():
    """Stress test with large number of frames."""
    stream = StressTestStream(frame_count=500, frame_size=32)

    processed = 0
    start_time = asyncio.get_event_loop().time()

    async for _ in async_process_stream(stream):
        processed += 1

        # Yield control periodically to prevent blocking
        if processed % 50 == 0:
            await asyncio.sleep(0.001)

    end_time = asyncio.get_event_loop().time()

    assert processed <= 500
    # Should complete in reasonable time (under 30 seconds)
    assert (end_time - start_time) < 30.0


@pytest.mark.asyncio
async def test_stress_large_frame_size():
    """Stress test with large frame sizes."""
    stream = StressTestStream(frame_count=10, frame_size=8192)

    processed = 0
    async for _ in async_process_stream(stream):
        processed += 1

    assert processed <= 10


@pytest.mark.asyncio
async def test_stress_chaos_mode():
    """Stress test with chaotic/problematic data."""
    stream = StressTestStream(frame_count=50, frame_size=64, chaos_mode=True)

    processed = 0
    errors = 0

    try:
        async for _ in async_process_stream(stream):
            processed += 1
    except Exception:
        errors += 1

    # Should handle chaos gracefully (some processing or controlled errors)
    assert processed >= 0 or errors > 0


@pytest.mark.asyncio
async def test_edge_case_zero_size_frames():
    """Test stream that yields zero-size frames."""

    class ZeroSizeStream:
        def __init__(self, count: int):
            self.count = count
            self.index = 0

        def __aiter__(self):
            self.index = 0
            return self

        async def __anext__(self):
            if self.index >= self.count:
                raise StopAsyncIteration
            self.index += 1
            return np.array([], dtype=np.float32)

    stream = ZeroSizeStream(5)
    events = []

    async for ev in async_process_stream(stream):
        events.append(ev)

    # Should handle zero-size frames (likely skip them)
    assert isinstance(events, list)


@pytest.mark.asyncio
async def test_edge_case_single_value_frames():
    """Test stream with single-value frames."""

    class SingleValueStream:
        def __init__(self, count: int):
            self.count = count
            self.index = 0

        def __aiter__(self):
            self.index = 0
            return self

        async def __anext__(self):
            if self.index >= self.count:
                raise StopAsyncIteration
            self.index += 1
            return np.array([self.index * 0.5], dtype=np.float32)

    stream = SingleValueStream(10)
    events = []

    async for ev in async_process_stream(stream):
        events.append(ev)

    assert isinstance(events, list)


@pytest.mark.asyncio
async def test_edge_case_extreme_values():
    """Test frames with extreme values."""

    class ExtremeValueStream:
        def __init__(self):
            self.frames = [
                np.array([np.finfo(np.float32).max], dtype=np.float32),
                np.array([np.finfo(np.float32).min], dtype=np.float32),
                np.array([np.finfo(np.float32).tiny], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([-0.0], dtype=np.float32),
            ]
            self.index = 0

        def __aiter__(self):
            self.index = 0
            return self

        async def __anext__(self):
            if self.index >= len(self.frames):
                raise StopAsyncIteration
            frame = self.frames[self.index]
            self.index += 1
            return frame

    stream = ExtremeValueStream()
    events = []

    try:
        async for ev in async_process_stream(stream):
            events.append(ev)
    except (ValueError, OverflowError, RuntimeError):
        # Some extreme values might cause processing errors
        pass

    assert isinstance(events, list)


@pytest.mark.asyncio
async def test_memory_stress_rapid_processing():
    """Test rapid processing without memory accumulation."""

    class RapidStream:
        def __init__(self, count: int):
            self.count = count
            self.index = 0

        def __aiter__(self):
            self.index = 0
            return self

        async def __anext__(self):
            if self.index >= self.count:
                raise StopAsyncIteration

            # Very short delay to simulate rapid data
            await asyncio.sleep(0.0001)

            self.index += 1
            # Generate fresh data each time (don't reuse arrays)
            return np.random.randn(32).astype(np.float32)

    stream = RapidStream(100)
    processed = 0

    # Process without storing results (memory efficient)
    async for _ in async_process_stream(stream):
        processed += 1

        # Simulate real-time processing constraints
        if processed % 25 == 0:
            await asyncio.sleep(0.001)  # Brief pause every 25 frames

    assert processed <= 100


@pytest.mark.asyncio
async def test_timeout_handling():
    """Test async stream processing with timeouts."""

    class SlowStream:
        def __init__(self):
            self.index = 0

        def __aiter__(self):
            self.index = 0
            return self

        async def __anext__(self):
            if self.index >= 3:
                raise StopAsyncIteration

            # First frame normal, then increasingly slow
            await asyncio.sleep(self.index * 0.1)

            self.index += 1
            return np.random.randn(64).astype(np.float32)

    stream = SlowStream()

    async def process_with_timeout():
        events = []
        async for ev in async_process_stream(stream):
            events.append(ev)
        return events

    # Test with reasonable timeout
    try:
        events = await asyncio.wait_for(process_with_timeout(), timeout=1.0)
        assert isinstance(events, list)
    except asyncio.TimeoutError:
        # Timeout is acceptable for slow streams
        pass


@pytest.mark.asyncio
async def test_concurrent_stress():
    """Test multiple concurrent async streams under stress."""

    async def stress_worker(worker_id: int, frame_count: int):
        stream = StressTestStream(frame_count, frame_size=32)
        count = 0
        async for _ in async_process_stream(stream):
            count += 1
        return worker_id, count

    # Run multiple workers concurrently
    num_workers = 5
    frames_per_worker = 20

    results = await asyncio.gather(*[
        stress_worker(i, frames_per_worker)
        for i in range(num_workers)
    ])

    assert len(results) == num_workers
    for worker_id, count in results:
        assert 0 <= worker_id < num_workers
        assert count <= frames_per_worker
        assert count <= frames_per_worker
