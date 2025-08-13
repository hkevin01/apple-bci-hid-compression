"""Data pipeline implementation for direct BCI data streaming."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from src.core.processing import NeuralData

# local type aliases to keep signatures short and within line length
Processor = Callable[[NeuralData], NeuralData]
Consumer = Callable[[NeuralData], None]


@dataclass
class DataStreamConfig:
    """Configuration for the data streaming pipeline."""

    chunk_size: int = 512  # Number of samples per chunk
    sample_rate: float = 1000.0  # Hz
    buffer_size: int = 8192  # Samples to buffer
    channels: list[str] | None = None  # Channel names

    def __post_init__(self) -> None:
        if self.channels is None:
            self.channels = []


class DataPipeline:
    """Direct device-to-host data pipeline implementation."""
    config: DataStreamConfig
    _processors: list[Processor]
    _running: bool
    _buffer: asyncio.Queue

    def __init__(self, config: DataStreamConfig):
        self.config = config
        self._processors = []
        self._running = False
        self._buffer = asyncio.Queue(maxsize=self.config.buffer_size)

    def add_processor(self, processor: Processor) -> None:
        """Add a data processor to the pipeline."""
        self._processors.append(processor)

    async def process_chunk(self, data: NeuralData) -> NeuralData:
        """Process a chunk of neural data through the pipeline."""
        result = data
        for processor in self._processors:
            result = processor(result)
        return result

    async def start(self) -> None:
        """Start the data pipeline."""
        self._running = True

    async def stop(self) -> None:
        """Stop the data pipeline."""
        self._running = False

    async def push_data(self, data: NeuralData) -> None:
        """Push new data into the pipeline."""
        if not self._running:
            raise RuntimeError("Pipeline is not running")

        try:
            await self._buffer.put(data)
        except asyncio.QueueFull:
            # If buffer is full, remove oldest data
            _ = await self._buffer.get()
            await self._buffer.put(data)

    async def get_data(self) -> AsyncIterator[NeuralData]:
        """Get processed data from the pipeline."""
        while self._running:
            if not self._buffer.empty():
                data = await self._buffer.get()
                processed_data = await self.process_chunk(data)
                yield processed_data
            else:
                await asyncio.sleep(0.001)  # avoid busy wait


class RealTimeDataProcessor:
    """Real-time data processing implementation."""
    pipeline: DataPipeline
    _consumer_task: asyncio.Task | None

    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline
        self._consumer_task = None

    async def start_processing(self, callback: Consumer) -> None:
        """Start processing data in real-time."""
        await self.pipeline.start()

        async def consumer() -> None:
            async for data in self.pipeline.get_data():
                callback(data)

        self._consumer_task = asyncio.create_task(consumer())

    async def stop_processing(self) -> None:
        """Stop processing data."""
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        await self.pipeline.stop()
