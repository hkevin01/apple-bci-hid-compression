"""Tests for the data pipeline implementation."""

import asyncio

import numpy as np
import pytest

from src.core.pipeline.data_pipeline import (DataPipeline, DataStreamConfig,
                                             RealTimeDataProcessor)
from src.core.processing import NeuralData


@pytest.fixture
def config():
    """Create a test configuration."""
    return DataStreamConfig(
        chunk_size=100,
        sample_rate=500.0,
        buffer_size=1000,
        channels=['C1', 'C2', 'C3']
    )


@pytest.fixture
def pipeline(config):
    """Create a test pipeline."""
    return DataPipeline(config)


async def test_pipeline_basic_operation(pipeline):
    """Test basic pipeline operations."""
    # Start the pipeline
    await pipeline.start()
    assert pipeline._running is True

    # Create test data
    test_data = NeuralData(
        timestamp=0.0,
        channels=['C1', 'C2', 'C3'],
        samples=np.random.random((100, 3))
    )

    # Push data
    await pipeline.push_data(test_data)

    # Get data
    async for data in pipeline.get_data():
        assert isinstance(data, NeuralData)
        assert data.channels == test_data.channels
        assert data.samples.shape == test_data.samples.shape
        break

    # Stop the pipeline
    await pipeline.stop()
    assert pipeline._running is False


async def test_pipeline_processor(pipeline):
    """Test pipeline with data processor."""
    def test_processor(data: NeuralData) -> NeuralData:
        # Simple processor that doubles the values
        data.samples *= 2
        return data

    pipeline.add_processor(test_processor)
    await pipeline.start()

    # Create test data
    test_data = NeuralData(
        timestamp=0.0,
        channels=['C1', 'C2', 'C3'],
        samples=np.ones((100, 3))
    )

    # Push data
    await pipeline.push_data(test_data)

    # Get data
    async for data in pipeline.get_data():
        assert np.all(data.samples == 2.0)
        break

    await pipeline.stop()


async def test_realtime_processor(pipeline):
    """Test real-time data processing."""
    processed_data = []

    def callback(data: NeuralData):
        processed_data.append(data)

    processor = RealTimeDataProcessor(pipeline)
    await processor.start_processing(callback)

    # Create and push test data
    test_data = NeuralData(
        timestamp=0.0,
        channels=['C1', 'C2', 'C3'],
        samples=np.random.random((100, 3))
    )

    await pipeline.push_data(test_data)
    await asyncio.sleep(0.1)  # Give time for processing

    assert len(processed_data) > 0
    assert isinstance(processed_data[0], NeuralData)

    await processor.stop_processing()
