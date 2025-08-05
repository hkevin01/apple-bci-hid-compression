"""Real-time performance optimization implementations."""

import asyncio
import heapq
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from queue import PriorityQueue, Queue
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np


class Priority(Enum):
    """Task priority levels."""
    CRITICAL = 1    # Real-time HID events
    HIGH = 2        # User interaction
    NORMAL = 3      # Background processing
    LOW = 4         # Maintenance tasks


@dataclass
class Task:
    """Task representation for processing."""
    id: str
    priority: Priority
    function: Callable
    args: tuple
    kwargs: dict
    created_at: float

    def __lt__(self, other):
        """Comparison for priority queue."""
        return self.priority.value < other.priority.value


class PerformanceOptimizer(Protocol):
    """Protocol for performance optimization strategies."""

    def optimize(self, data: Any) -> Any:
        """Optimize the processing of data."""
        ...

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        ...


class ParallelProcessor:
    """Parallel processing pipeline implementation."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.metrics = {
            'tasks_completed': 0,
            'average_latency': 0.0,
            'throughput': 0.0,
            'cpu_utilization': 0.0
        }
        self._start_time = time.time()

    def optimize(self, data_chunks: List[np.ndarray]) -> List[np.ndarray]:
        """Process data chunks in parallel."""
        if not data_chunks:
            return []

        start_time = time.time()

        # Submit all chunks for parallel processing
        futures = []
        for i, chunk in enumerate(data_chunks):
            future = self.executor.submit(self._process_chunk, chunk, i)
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=1.0)  # 1 second timeout
                results.append(result)
            except Exception as e:
                # Log error and use empty result
                print(f"Processing error: {e}")
                results.append(np.array([]))

        # Update metrics
        end_time = time.time()
        latency = end_time - start_time
        self._update_metrics(len(data_chunks), latency)

        return results

    def _process_chunk(self, chunk: np.ndarray, chunk_id: int) -> np.ndarray:
        """Process a single data chunk."""
        # Simulate processing with compression
        from scipy.fft import fft

        # Apply windowing to reduce artifacts
        window = np.hanning(len(chunk))
        windowed = chunk * window

        # Apply FFT for compression
        compressed = fft(windowed)

        # Keep only significant coefficients (compression)
        threshold = np.max(np.abs(compressed)) * 0.1
        compressed[np.abs(compressed) < threshold] = 0

        return compressed

    def _update_metrics(self, task_count: int, latency: float):
        """Update performance metrics."""
        self.metrics['tasks_completed'] += task_count

        # Update average latency (exponential moving average)
        alpha = 0.1
        self.metrics['average_latency'] = (
            alpha * latency +
            (1 - alpha) * self.metrics['average_latency']
        )

        # Calculate throughput
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self.metrics['throughput'] = self.metrics['tasks_completed'] / elapsed

    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.metrics.copy()

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class EventDrivenProcessor:
    """Event-driven architecture implementation."""

    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_queue = asyncio.Queue()
        self.running = False
        self.metrics = {
            'events_processed': 0,
            'event_latency': 0.0,
            'queue_size': 0
        }

    def register_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def emit_event(self, event_type: str, data: Any):
        """Emit an event."""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        await self.event_queue.put(event)

    async def start(self):
        """Start the event processing loop."""
        self.running = True
        while self.running:
            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=0.1
                )
                await self._process_event(event)
            except asyncio.TimeoutError:
                # Update queue size metric
                self.metrics['queue_size'] = self.event_queue.qsize()
                continue

    async def _process_event(self, event: Dict[str, Any]):
        """Process a single event."""
        start_time = time.time()

        event_type = event['type']
        handlers = self.event_handlers.get(event_type, [])

        # Execute all handlers for this event type
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event['data'])
                else:
                    # Run sync handler in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler, event['data'])
            except Exception as e:
                print(f"Handler error for {event_type}: {e}")

        # Update metrics
        latency = time.time() - start_time
        self.metrics['events_processed'] += 1

        # Exponential moving average for latency
        alpha = 0.1
        self.metrics['event_latency'] = (
            alpha * latency +
            (1 - alpha) * self.metrics['event_latency']
        )

    def stop(self):
        """Stop the event processing loop."""
        self.running = False

    def optimize(self, data: Any) -> Any:
        """Optimize using event-driven processing."""
        # Convert data processing to events
        asyncio.create_task(self.emit_event('data_received', data))
        return data

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return self.metrics.copy()


class HybridProcessor:
    """Hybrid approach with prioritized processing."""

    def __init__(self, max_workers: int = 4):
        self.priority_queue = PriorityQueue()
        self.parallel_processor = ParallelProcessor(max_workers)
        self.event_processor = EventDrivenProcessor()
        self.task_counter = 0
        self.metrics = {
            'high_priority_latency': 0.0,
            'normal_priority_latency': 0.0,
            'queue_depth': 0,
            'context_switches': 0
        }

        # Setup event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup event handlers for different priority levels."""
        self.event_processor.register_handler(
            'critical_data',
            self._handle_critical_data
        )
        self.event_processor.register_handler(
            'normal_data',
            self._handle_normal_data
        )

    async def _handle_critical_data(self, data: Any):
        """Handle critical priority data with minimal latency."""
        start_time = time.time()

        # Direct processing for critical data
        result = self._fast_process(data)

        # Update metrics
        latency = time.time() - start_time
        alpha = 0.1
        self.metrics['high_priority_latency'] = (
            alpha * latency +
            (1 - alpha) * self.metrics['high_priority_latency']
        )

        return result

    async def _handle_normal_data(self, data: Any):
        """Handle normal priority data with parallel processing."""
        start_time = time.time()

        # Use parallel processing for normal data
        if isinstance(data, list):
            result = self.parallel_processor.optimize(data)
        else:
            result = self._standard_process(data)

        # Update metrics
        latency = time.time() - start_time
        alpha = 0.1
        self.metrics['normal_priority_latency'] = (
            alpha * latency +
            (1 - alpha) * self.metrics['normal_priority_latency']
        )

        return result

    def _fast_process(self, data: Any) -> Any:
        """Fast processing for critical data."""
        # Minimal processing with maximum speed
        if isinstance(data, np.ndarray):
            # Simple downsampling for speed
            return data[::2]  # Decimate by 2
        return data

    def _standard_process(self, data: Any) -> Any:
        """Standard processing for normal priority data."""
        if isinstance(data, np.ndarray):
            # Apply standard compression
            from scipy.fft import fft
            return fft(data)
        return data

    def submit_task(self, function: Callable, priority: Priority,
                   *args, **kwargs) -> str:
        """Submit a task with specified priority."""
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1

        task = Task(
            id=task_id,
            priority=priority,
            function=function,
            args=args,
            kwargs=kwargs,
            created_at=time.time()
        )

        self.priority_queue.put(task)
        self.metrics['queue_depth'] = self.priority_queue.qsize()

        return task_id

    async def process_queue(self):
        """Process tasks from the priority queue."""
        while True:
            try:
                # Get highest priority task
                task = self.priority_queue.get(timeout=0.1)

                # Route based on priority
                if task.priority == Priority.CRITICAL:
                    await self.event_processor.emit_event('critical_data', task)
                else:
                    await self.event_processor.emit_event('normal_data', task)

                self.metrics['queue_depth'] = self.priority_queue.qsize()
                self.metrics['context_switches'] += 1

            except:
                # No tasks available, continue
                await asyncio.sleep(0.001)

    def optimize(self, data: Any, priority: Priority = Priority.NORMAL) -> Any:
        """Optimize processing based on priority."""
        if priority == Priority.CRITICAL:
            # Direct fast processing
            return self._fast_process(data)
        else:
            # Queue for parallel processing
            if isinstance(data, list):
                return self.parallel_processor.optimize(data)
            else:
                return self._standard_process(data)

    def get_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        metrics = self.metrics.copy()
        metrics.update(self.parallel_processor.get_metrics())
        metrics.update(self.event_processor.get_metrics())
        return metrics

    async def start(self):
        """Start the hybrid processor."""
        # Start event processor
        event_task = asyncio.create_task(self.event_processor.start())
        queue_task = asyncio.create_task(self.process_queue())

        await asyncio.gather(event_task, queue_task)

    def shutdown(self):
        """Shutdown all processors."""
        self.event_processor.stop()
        self.parallel_processor.shutdown()


class PerformanceManager:
    """Manages different performance optimization strategies."""

    def __init__(self):
        self.optimizers = {
            'parallel': ParallelProcessor(),
            'event_driven': EventDrivenProcessor(),
            'hybrid': HybridProcessor()
        }
        self.current_optimizer = 'hybrid'  # Default to hybrid

    def set_optimizer(self, optimizer_name: str):
        """Set the active performance optimizer."""
        if optimizer_name in self.optimizers:
            self.current_optimizer = optimizer_name
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def optimize(self, data: Any, **kwargs) -> Any:
        """Optimize using the current optimizer."""
        optimizer = self.optimizers[self.current_optimizer]
        return optimizer.optimize(data, **kwargs)

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from all optimizers."""
        metrics = {}
        for name, optimizer in self.optimizers.items():
            metrics[name] = optimizer.get_metrics()
        return metrics

    def benchmark_optimizers(self, test_data: Any) -> Dict[str, float]:
        """Benchmark all optimizers with test data."""
        results = {}

        for name, optimizer in self.optimizers.items():
            start_time = time.time()
            try:
                _ = optimizer.optimize(test_data)
                end_time = time.time()
                results[name] = end_time - start_time
            except Exception as e:
                print(f"Benchmark error for {name}: {e}")
                results[name] = float('inf')

        return results
