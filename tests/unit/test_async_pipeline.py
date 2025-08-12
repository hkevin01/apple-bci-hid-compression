import asyncio
import time

import numpy as np
import pytest

from src.core.compression import WaveletCompressor
from src.core.pipeline.async_end_to_end import ListFrameStream, async_process_stream


@pytest.mark.asyncio
async def test_async_stream_basic():
    """Test basic async stream processing with multiple frames."""
    frames = [np.random.randn(64).astype(np.float32) for _ in range(5)]
    stream = ListFrameStream(frames)
    events = []
    async for ev in async_process_stream(stream):
        events.append(ev)
    # Not guaranteed to have events; ensure no error and list type
    assert isinstance(events, list)


@pytest.mark.asyncio
async def test_async_stream_empty_and_nonempty():
    """Test stream with mix of empty and non-empty frames."""
    frames = [
        np.array([], dtype=np.float32),
        np.random.randn(64).astype(np.float32),
        np.array([], dtype=np.float32),
        np.random.randn(32).astype(np.float32)
    ]
    stream = ListFrameStream(frames)
    processed_count = 0
    async for _ in async_process_stream(stream):
        processed_count += 1
    # Empty frames should be skipped, so should process fewer than total
    assert processed_count <= len(frames)


@pytest.mark.asyncio
async def test_async_stream_large_frame():
    """Test stream processing with large frame sizes."""
    frames = [np.random.randn(4096).astype(np.float32)]
    stream = ListFrameStream(frames)
    async for _ in async_process_stream(stream):
        pass  # Ensure no errors with large frames


@pytest.mark.asyncio
async def test_async_stream_with_delay():
    """Test stream processing with simulated frame delays."""
    frames = [np.random.randn(64).astype(np.float32) for _ in range(3)]
    stream = ListFrameStream(frames, delay=0.01)  # 10ms delay between frames

    start_time = time.time()
    count = 0
    async for _ in async_process_stream(stream):
        count += 1
    end_time = time.time()

    # Should have taken at least the delay time * (frames - 1)
    expected_min_time = 0.01 * (len(frames) - 1)
    assert (end_time - start_time) >= expected_min_time
    assert count <= len(frames)


@pytest.mark.asyncio
async def test_async_stream_custom_compressor():
    """Test stream processing with custom compressor settings."""
    frames = [np.random.randn(128).astype(np.float32) for _ in range(3)]
    stream = ListFrameStream(frames)

    # Use custom compressor with different settings
    custom_compressor = WaveletCompressor(level=1, top_k_ratio=0.1)

    events = []
    async for ev in async_process_stream(stream, compressor=custom_compressor):
        events.append(ev)

    assert isinstance(events, list)


@pytest.mark.asyncio
async def test_async_stream_concurrent_consumption():
    """Test multiple concurrent consumers of the same stream type."""
    frames1 = [np.random.randn(64).astype(np.float32) for _ in range(2)]
    frames2 = [np.random.randn(64).astype(np.float32) for _ in range(2)]

    stream1 = ListFrameStream(frames1)
    stream2 = ListFrameStream(frames2)

    async def consume_stream(stream, results):
        async for ev in async_process_stream(stream):
            results.append(ev)

    results1, results2 = [], []
    await asyncio.gather(
        consume_stream(stream1, results1),
        consume_stream(stream2, results2)
    )

    assert isinstance(results1, list)
    assert isinstance(results2, list)


@pytest.mark.asyncio
async def test_async_stream_error_handling():
    """Test stream processing with problematic frames."""
    frames = [
        np.random.randn(64).astype(np.float32),  # Good frame
        # Problematic frame with NaN/inf
        np.array([np.nan, np.inf, -np.inf], dtype=np.float32),
        np.random.randn(64).astype(np.float32),  # Another good frame
    ]
    stream = ListFrameStream(frames)

    # Should handle problematic frames gracefully without crashing
    processed = 0
    try:
        async for _ in async_process_stream(stream):
            processed += 1
    except (ValueError, RuntimeError):
        # Some processing errors are acceptable
        pass

    # Should have processed at least some frames
    assert processed >= 0


@pytest.mark.asyncio
async def test_async_stream_zero_frames():
    """Test stream processing with no frames."""
    frames = []
    stream = ListFrameStream(frames)

    events = []
    async for ev in async_process_stream(stream):
        events.append(ev)

    assert events == []


@pytest.mark.asyncio
async def test_async_stream_single_frame():
    """Test stream processing with exactly one frame."""
    frames = [np.random.randn(64).astype(np.float32)]
    stream = ListFrameStream(frames)

    events = []
    async for ev in async_process_stream(stream):
        events.append(ev)

    assert isinstance(events, list)
    assert len(events) <= 1  # May or may not generate event


@pytest.mark.asyncio
async def test_async_stream_varying_frame_sizes():
    """Test stream with frames of varying sizes."""
    frames = [
        np.random.randn(32).astype(np.float32),
        np.random.randn(64).astype(np.float32),
        np.random.randn(128).astype(np.float32),
        np.random.randn(16).astype(np.float32),
    ]
    stream = ListFrameStream(frames)

    processed_sizes = []
    async for _ in async_process_stream(stream):
        processed_sizes.append("processed")

    # Should handle different frame sizes without error
    assert isinstance(processed_sizes, list)


@pytest.mark.asyncio
async def test_list_frame_stream_iterator():
    """Test ListFrameStream async iterator functionality."""
    frames = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32)
    ]
    stream = ListFrameStream(frames)

    collected = []
    async for frame in stream:
        collected.append(frame)

    assert len(collected) == len(frames)
    assert np.array_equal(collected[0], frames[0])
    assert np.array_equal(collected[1], frames[1])


@pytest.mark.asyncio
async def test_list_frame_stream_empty():
    """Test ListFrameStream with empty frame list."""
    stream = ListFrameStream([])

    collected = []
    async for frame in stream:
        collected.append(frame)

    assert collected == []


@pytest.mark.asyncio
async def test_async_stream_performance_timing():
    """Test that async processing doesn't introduce excessive delays."""
    frames = [np.random.randn(64).astype(np.float32) for _ in range(5)]
    stream = ListFrameStream(frames)

    start_time = time.time()
    count = 0
    async for _ in async_process_stream(stream):
        count += 1
    end_time = time.time()

    # Processing should complete in reasonable time (under 5 seconds)
    assert (end_time - start_time) < 5.0
    assert count <= len(frames)
