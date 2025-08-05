"""Automated benchmarking suite for Phase 4 performance testing."""

import asyncio
import gc
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from accessibility.accessibility_features import AccessibilityManager
from core.acceleration.hardware_acceleration import HardwareAccelerator
from core.compression import CompressionQuality, WaveletCompressor
from core.performance.realtime_optimization import RealtimeOptimizer
from core.pipeline.data_pipeline import DataPipeline
from core.processing import NeuralData, NeuralProcessor
from interfaces.device_communication import MultiProtocolCommunicator
from mapping.input_mapping import MultiModalInputMapper
from recognition.gesture_recognition import HybridGestureRecognizer, NeuralSignal


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    test_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: Optional[float] = None
    latency: Optional[float] = None
    accuracy: Optional[float] = None
    error_rate: Optional[float] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemInfo:
    """System information for benchmark context."""
    platform: str
    cpu_count: int
    total_memory: int
    python_version: str
    numpy_version: str
    available_memory: int
    cpu_frequency: float


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.system_info = self._get_system_info()
        self.test_data_cache: Dict[str, Any] = {}

        # Initialize components for testing
        self.neural_processor = NeuralProcessor()
        self.compressor = WaveletCompressor()
        self.data_pipeline = DataPipeline()
        self.hardware_accelerator = HardwareAccelerator()
        self.realtime_optimizer = RealtimeOptimizer()
        self.device_communicator = MultiProtocolCommunicator()
        self.gesture_recognizer = HybridGestureRecognizer()
        self.input_mapper = MultiModalInputMapper()
        self.accessibility_manager = AccessibilityManager()

        print(f"🔧 Benchmark initialized on {self.system_info.platform}")
        print(f"   CPU: {self.system_info.cpu_count} cores @ {self.system_info.cpu_frequency:.1f} MHz")
        print(f"   Memory: {self.system_info.total_memory / (1024**3):.1f} GB")

    def _get_system_info(self) -> SystemInfo:
        """Get system information."""
        return SystemInfo(
            platform=platform.system() + " " + platform.release(),
            cpu_count=os.cpu_count(),
            total_memory=psutil.virtual_memory().total,
            python_version=sys.version.split()[0],
            numpy_version=np.__version__,
            available_memory=psutil.virtual_memory().available,
            cpu_frequency=psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
        )

    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("\n🚀 Starting Comprehensive Performance Benchmarks")
        print("=" * 60)

        # Core component benchmarks
        await self.benchmark_neural_processing()
        await self.benchmark_compression()
        await self.benchmark_data_pipeline()
        await self.benchmark_hardware_acceleration()
        await self.benchmark_realtime_optimization()

        # Interface component benchmarks
        await self.benchmark_device_communication()
        await self.benchmark_gesture_recognition()
        await self.benchmark_input_mapping()
        await self.benchmark_accessibility_features()

        # Integration benchmarks
        await self.benchmark_end_to_end_latency()
        await self.benchmark_concurrent_processing()
        await self.benchmark_memory_efficiency()
        await self.benchmark_scalability()

        # Generate comprehensive report
        return self.generate_performance_report()

    async def benchmark_neural_processing(self):
        """Benchmark neural signal processing performance."""
        print("\n🧠 Benchmarking Neural Processing...")

        # Test different signal sizes
        signal_sizes = [100, 500, 1000, 2500, 5000]

        for size in signal_sizes:
            # Generate test data
            neural_data = self._generate_neural_data(size, 8)  # 8 channels

            # Benchmark processing
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss

            for _ in range(100):  # 100 iterations for averaging
                processed = await self.neural_processor.process_data(neural_data)

            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss

            execution_time = (end_time - start_time) / 100  # Average per iteration
            memory_delta = end_memory - start_memory
            throughput = size / execution_time  # Samples per second

            result = BenchmarkResult(
                test_name=f"neural_processing_{size}_samples",
                execution_time=execution_time,
                memory_usage=memory_delta,
                cpu_usage=psutil.cpu_percent(interval=0.1),
                throughput=throughput,
                custom_metrics={
                    'samples_per_second': throughput,
                    'signal_size': size,
                    'channels': 8
                }
            )

            self.results.append(result)
            print(f"  ✓ {size} samples: {execution_time*1000:.2f}ms, {throughput:.0f} samples/sec")

    async def benchmark_compression(self):
        """Benchmark compression algorithm performance."""
        print("\n📦 Benchmarking Compression...")

        # Test different compression qualities and data sizes
        data_sizes = [1000, 5000, 10000, 25000]
        qualities = [CompressionQuality.LOW, CompressionQuality.MEDIUM, CompressionQuality.HIGH]

        for size in data_sizes:
            for quality in qualities:
                # Generate test data
                neural_data = self._generate_neural_data(size, 8)

                # Benchmark compression
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss

                compressed_data = None
                for _ in range(50):  # 50 iterations
                    compressed_data = await self.compressor.compress(neural_data, quality)

                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss

                # Benchmark decompression
                decomp_start = time.perf_counter()
                decompressed_data = await self.compressor.decompress(compressed_data)
                decomp_end = time.perf_counter()

                # Calculate metrics
                execution_time = (end_time - start_time) / 50
                decomp_time = decomp_end - decomp_start
                memory_delta = end_memory - start_memory

                original_size = neural_data.data.nbytes
                compressed_size = len(compressed_data.compressed_data)
                compression_ratio = original_size / compressed_size

                # Calculate reconstruction error
                mse = np.mean((neural_data.data - decompressed_data.data) ** 2)

                result = BenchmarkResult(
                    test_name=f"compression_{quality.value}_{size}_samples",
                    execution_time=execution_time,
                    memory_usage=memory_delta,
                    cpu_usage=psutil.cpu_percent(interval=0.1),
                    custom_metrics={
                        'compression_ratio': compression_ratio,
                        'compression_time': execution_time,
                        'decompression_time': decomp_time,
                        'reconstruction_mse': mse,
                        'quality': quality.value,
                        'original_size_bytes': original_size,
                        'compressed_size_bytes': compressed_size
                    }
                )

                self.results.append(result)
                print(f"  ✓ {quality.value} {size} samples: {compression_ratio:.1f}x ratio, {execution_time*1000:.2f}ms")

    async def benchmark_data_pipeline(self):
        """Benchmark data pipeline performance."""
        print("\n⚡ Benchmarking Data Pipeline...")

        # Test different batch sizes and processing loads
        batch_sizes = [10, 50, 100, 200]

        for batch_size in batch_sizes:
            # Generate batch of neural data
            batch_data = [self._generate_neural_data(1000, 8) for _ in range(batch_size)]

            # Benchmark pipeline processing
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss

            processed_count = 0
            for data in batch_data:
                await self.data_pipeline.process_data(data)
                processed_count += 1

            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss

            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            throughput = processed_count / execution_time

            result = BenchmarkResult(
                test_name=f"data_pipeline_batch_{batch_size}",
                execution_time=execution_time,
                memory_usage=memory_delta,
                cpu_usage=psutil.cpu_percent(interval=0.1),
                throughput=throughput,
                custom_metrics={
                    'batch_size': batch_size,
                    'items_per_second': throughput,
                    'avg_time_per_item': execution_time / processed_count
                }
            )

            self.results.append(result)
            print(f"  ✓ Batch {batch_size}: {throughput:.1f} items/sec, {execution_time*1000:.2f}ms total")

    async def benchmark_hardware_acceleration(self):
        """Benchmark hardware acceleration performance."""
        print("\n🚀 Benchmarking Hardware Acceleration...")

        # Test different acceleration backends
        backends = ['metal', 'coreml', 'simd', 'cpu']
        data_sizes = [1000, 5000, 10000]

        for backend in backends:
            for size in data_sizes:
                try:
                    # Configure accelerator backend
                    await self.hardware_accelerator.configure_backend(backend)

                    # Generate test data
                    input_data = np.random.randn(size, 8).astype(np.float32)

                    # Benchmark acceleration
                    start_time = time.perf_counter()
                    start_memory = psutil.Process().memory_info().rss

                    for _ in range(20):  # 20 iterations
                        result_data = await self.hardware_accelerator.accelerate_computation(
                            input_data, operation='signal_processing'
                        )

                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss

                    execution_time = (end_time - start_time) / 20
                    memory_delta = end_memory - start_memory
                    throughput = size / execution_time

                    result = BenchmarkResult(
                        test_name=f"acceleration_{backend}_{size}_samples",
                        execution_time=execution_time,
                        memory_usage=memory_delta,
                        cpu_usage=psutil.cpu_percent(interval=0.1),
                        throughput=throughput,
                        custom_metrics={
                            'backend': backend,
                            'data_size': size,
                            'acceleration_factor': 1.0  # Would compare with CPU baseline
                        }
                    )

                    self.results.append(result)
                    print(f"  ✓ {backend} {size} samples: {execution_time*1000:.2f}ms, {throughput:.0f} samples/sec")

                except Exception as e:
                    print(f"  ✗ {backend} {size} samples: Failed ({e})")

    async def benchmark_realtime_optimization(self):
        """Benchmark real-time optimization performance."""
        print("\n⚡ Benchmarking Real-time Optimization...")

        # Test different optimization strategies
        strategies = ['parallel', 'event_driven', 'hybrid']
        workloads = [10, 50, 100, 200]  # Number of concurrent operations

        for strategy in strategies:
            for workload in workloads:
                try:
                    # Configure optimization strategy
                    await self.realtime_optimizer.set_strategy(strategy)

                    # Generate concurrent workload
                    tasks = []
                    start_time = time.perf_counter()
                    start_memory = psutil.Process().memory_info().rss

                    for i in range(workload):
                        neural_data = self._generate_neural_data(500, 8)
                        task = self.realtime_optimizer.optimize_processing(neural_data)
                        tasks.append(task)

                    # Wait for all tasks to complete
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss

                    # Calculate metrics
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    successful_results = [r for r in results if not isinstance(r, Exception)]
                    success_rate = len(successful_results) / len(results)
                    throughput = len(successful_results) / execution_time

                    result = BenchmarkResult(
                        test_name=f"realtime_optimization_{strategy}_{workload}_ops",
                        execution_time=execution_time,
                        memory_usage=memory_delta,
                        cpu_usage=psutil.cpu_percent(interval=0.1),
                        throughput=throughput,
                        accuracy=success_rate,
                        custom_metrics={
                            'strategy': strategy,
                            'workload': workload,
                            'success_rate': success_rate,
                            'ops_per_second': throughput
                        }
                    )

                    self.results.append(result)
                    print(f"  ✓ {strategy} {workload} ops: {throughput:.1f} ops/sec, {success_rate*100:.1f}% success")

                except Exception as e:
                    print(f"  ✗ {strategy} {workload} ops: Failed ({e})")

    async def benchmark_device_communication(self):
        """Benchmark device communication performance."""
        print("\n📡 Benchmarking Device Communication...")

        # Test different message sizes and protocols
        message_sizes = [64, 256, 1024, 4096]  # bytes
        protocols = ['bluetooth_le', 'usb']

        for protocol in protocols:
            for msg_size in message_sizes:
                try:
                    # Simulate device discovery and connection
                    devices = await self.device_communicator.discover_all_devices()

                    # Generate test message
                    test_message = os.urandom(msg_size)

                    # Benchmark message sending
                    start_time = time.perf_counter()

                    send_count = 0
                    for _ in range(100):  # 100 messages
                        # Mock sending to first available device of this protocol
                        mock_device_id = f"test_{protocol}_device"
                        success = await self.device_communicator.send_data(mock_device_id, test_message)
                        if success:
                            send_count += 1

                    end_time = time.perf_counter()

                    execution_time = end_time - start_time
                    throughput = send_count / execution_time
                    bandwidth = (send_count * msg_size) / execution_time  # bytes per second
                    latency = execution_time / send_count if send_count > 0 else float('inf')

                    result = BenchmarkResult(
                        test_name=f"device_communication_{protocol}_{msg_size}_bytes",
                        execution_time=execution_time,
                        memory_usage=psutil.Process().memory_info().rss,
                        cpu_usage=psutil.cpu_percent(interval=0.1),
                        throughput=throughput,
                        latency=latency,
                        custom_metrics={
                            'protocol': protocol,
                            'message_size': msg_size,
                            'bandwidth_bps': bandwidth,
                            'messages_sent': send_count
                        }
                    )

                    self.results.append(result)
                    print(f"  ✓ {protocol} {msg_size}B: {throughput:.1f} msg/sec, {bandwidth/1024:.1f} KB/s")

                except Exception as e:
                    print(f"  ✗ {protocol} {msg_size}B: Failed ({e})")

    async def benchmark_gesture_recognition(self):
        """Benchmark gesture recognition performance."""
        print("\n🤖 Benchmarking Gesture Recognition...")

        # Test different signal patterns and recognition methods
        signal_lengths = [250, 500, 1000, 2000]  # samples
        recognition_methods = ['ml', 'rule_based', 'hybrid']

        for method in recognition_methods:
            for length in signal_lengths:
                try:
                    # Generate test signals with known gestures
                    test_signals = []
                    for _ in range(50):  # 50 test signals
                        signal = self._generate_gesture_signal(length, gesture_type='click')
                        test_signals.append(signal)

                    # Benchmark recognition
                    start_time = time.perf_counter()
                    start_memory = psutil.Process().memory_info().rss

                    recognized_count = 0
                    correct_recognitions = 0

                    for signal in test_signals:
                        gesture = await self.gesture_recognizer.process_signal(signal)
                        if gesture:
                            recognized_count += 1
                            # Check if correct gesture type was recognized
                            if hasattr(gesture, 'gesture_type') and 'click' in str(gesture.gesture_type).lower():
                                correct_recognitions += 1

                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss

                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    throughput = len(test_signals) / execution_time
                    accuracy = correct_recognitions / len(test_signals) if test_signals else 0
                    recognition_rate = recognized_count / len(test_signals) if test_signals else 0

                    result = BenchmarkResult(
                        test_name=f"gesture_recognition_{method}_{length}_samples",
                        execution_time=execution_time,
                        memory_usage=memory_delta,
                        cpu_usage=psutil.cpu_percent(interval=0.1),
                        throughput=throughput,
                        accuracy=accuracy,
                        custom_metrics={
                            'method': method,
                            'signal_length': length,
                            'recognition_rate': recognition_rate,
                            'signals_per_second': throughput,
                            'correct_recognitions': correct_recognitions,
                            'total_recognitions': recognized_count
                        }
                    )

                    self.results.append(result)
                    print(f"  ✓ {method} {length} samples: {throughput:.1f} signals/sec, {accuracy*100:.1f}% accuracy")

                except Exception as e:
                    print(f"  ✗ {method} {length} samples: Failed ({e})")

    async def benchmark_input_mapping(self):
        """Benchmark input mapping performance."""
        print("\n🎯 Benchmarking Input Mapping...")

        # Test different mapping types and gesture loads
        mapping_types = ['fixed', 'configurable', 'context_aware']
        gesture_loads = [10, 50, 100, 500]  # gestures per second

        for mapping_type in mapping_types:
            for load in gesture_loads:
                try:
                    # Generate test gestures
                    test_gestures = []
                    for _ in range(load):
                        gesture = self._generate_test_gesture()
                        test_gestures.append(gesture)

                    # Benchmark mapping
                    start_time = time.perf_counter()
                    start_memory = psutil.Process().memory_info().rss

                    mapped_count = 0
                    for gesture in test_gestures:
                        input_action = self.input_mapper.map_gesture(gesture)
                        if input_action:
                            mapped_count += 1

                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss

                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    throughput = len(test_gestures) / execution_time
                    mapping_rate = mapped_count / len(test_gestures) if test_gestures else 0

                    result = BenchmarkResult(
                        test_name=f"input_mapping_{mapping_type}_{load}_gestures",
                        execution_time=execution_time,
                        memory_usage=memory_delta,
                        cpu_usage=psutil.cpu_percent(interval=0.1),
                        throughput=throughput,
                        accuracy=mapping_rate,
                        custom_metrics={
                            'mapping_type': mapping_type,
                            'gesture_load': load,
                            'mapping_rate': mapping_rate,
                            'gestures_per_second': throughput,
                            'mapped_gestures': mapped_count
                        }
                    )

                    self.results.append(result)
                    print(f"  ✓ {mapping_type} {load} gestures: {throughput:.1f} gestures/sec, {mapping_rate*100:.1f}% mapped")

                except Exception as e:
                    print(f"  ✗ {mapping_type} {load} gestures: Failed ({e})")

    async def benchmark_accessibility_features(self):
        """Benchmark accessibility features performance."""
        print("\n♿ Benchmarking Accessibility Features...")

        # Test different accessibility actions and loads
        features = ['voice_over', 'switch_control', 'custom_protocol']
        action_loads = [5, 20, 50, 100]  # actions per test

        for feature in features:
            for load in action_loads:
                try:
                    # Enable feature
                    from accessibility.accessibility_features import (
                        AccessibilityFeature,
                    )
                    feature_enum = getattr(AccessibilityFeature, feature.upper(), None)
                    if feature_enum:
                        self.accessibility_manager.enable_feature(feature_enum)

                    # Generate test gestures for accessibility
                    test_gestures = []
                    for _ in range(load):
                        gesture = self._generate_test_gesture()
                        test_gestures.append(gesture)

                    # Benchmark accessibility processing
                    start_time = time.perf_counter()
                    start_memory = psutil.Process().memory_info().rss

                    processed_count = 0
                    for gesture in test_gestures:
                        results = await self.accessibility_manager.process_gesture(gesture)
                        if results:
                            processed_count += len(results)

                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss

                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    throughput = len(test_gestures) / execution_time
                    processing_rate = processed_count / len(test_gestures) if test_gestures else 0

                    result = BenchmarkResult(
                        test_name=f"accessibility_{feature}_{load}_actions",
                        execution_time=execution_time,
                        memory_usage=memory_delta,
                        cpu_usage=psutil.cpu_percent(interval=0.1),
                        throughput=throughput,
                        accuracy=processing_rate,
                        custom_metrics={
                            'feature': feature,
                            'action_load': load,
                            'processing_rate': processing_rate,
                            'actions_per_second': throughput,
                            'processed_actions': processed_count
                        }
                    )

                    self.results.append(result)
                    print(f"  ✓ {feature} {load} actions: {throughput:.1f} actions/sec, {processing_rate:.1f} avg results")

                except Exception as e:
                    print(f"  ✗ {feature} {load} actions: Failed ({e})")

    async def benchmark_end_to_end_latency(self):
        """Benchmark end-to-end latency of complete pipeline."""
        print("\n🔄 Benchmarking End-to-End Latency...")

        # Test different pipeline configurations
        configurations = [
            {'compression': 'low', 'acceleration': 'cpu'},
            {'compression': 'medium', 'acceleration': 'metal'},
            {'compression': 'high', 'acceleration': 'coreml'},
        ]

        for config in configurations:
            try:
                # Generate test neural signal
                neural_signal = self._generate_neural_data(1000, 8)

                latencies = []
                successful_runs = 0

                # Run multiple iterations to get average latency
                for _ in range(100):
                    start_time = time.perf_counter()

                    try:
                        # Complete pipeline: signal -> gesture -> action -> output
                        # 1. Process neural signal
                        processed_data = await self.neural_processor.process_data(neural_signal)

                        # 2. Compress if needed
                        if config['compression'] != 'none':
                            quality = getattr(CompressionQuality, config['compression'].upper())
                            compressed_data = await self.compressor.compress(processed_data, quality)
                            decompressed_data = await self.compressor.decompress(compressed_data)
                        else:
                            decompressed_data = processed_data

                        # 3. Convert to gesture signal format
                        gesture_signal = NeuralSignal(
                            channels=decompressed_data.data.T,
                            timestamp=time.time(),
                            sample_rate=250.0
                        )

                        # 4. Recognize gesture
                        gesture = await self.gesture_recognizer.process_signal(gesture_signal)

                        # 5. Map to input action
                        if gesture:
                            input_action = self.input_mapper.map_gesture(gesture)

                        end_time = time.perf_counter()
                        latency = (end_time - start_time) * 1000  # Convert to milliseconds
                        latencies.append(latency)
                        successful_runs += 1

                    except Exception:
                        # Skip failed runs
                        continue

                if latencies:
                    avg_latency = np.mean(latencies)
                    p95_latency = np.percentile(latencies, 95)
                    p99_latency = np.percentile(latencies, 99)
                    success_rate = successful_runs / 100

                    result = BenchmarkResult(
                        test_name=f"end_to_end_latency_{config['compression']}_{config['acceleration']}",
                        execution_time=avg_latency / 1000,  # Convert back to seconds
                        memory_usage=psutil.Process().memory_info().rss,
                        cpu_usage=psutil.cpu_percent(interval=0.1),
                        latency=avg_latency,
                        accuracy=success_rate,
                        custom_metrics={
                            'avg_latency_ms': avg_latency,
                            'p95_latency_ms': p95_latency,
                            'p99_latency_ms': p99_latency,
                            'configuration': config,
                            'success_rate': success_rate
                        }
                    )

                    self.results.append(result)
                    print(f"  ✓ {config}: {avg_latency:.1f}ms avg, {p95_latency:.1f}ms p95, {success_rate*100:.1f}% success")

            except Exception as e:
                print(f"  ✗ {config}: Failed ({e})")

    async def benchmark_concurrent_processing(self):
        """Benchmark concurrent processing capabilities."""
        print("\n⚡ Benchmarking Concurrent Processing...")

        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50]

        for concurrency in concurrency_levels:
            try:
                # Create concurrent processing tasks
                tasks = []
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss

                for i in range(concurrency):
                    neural_data = self._generate_neural_data(1000, 8)
                    task = self._process_data_async(neural_data)
                    tasks.append(task)

                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss

                # Calculate metrics
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                successful_results = [r for r in results if not isinstance(r, Exception)]
                success_rate = len(successful_results) / len(results)
                throughput = len(successful_results) / execution_time

                result = BenchmarkResult(
                    test_name=f"concurrent_processing_{concurrency}_tasks",
                    execution_time=execution_time,
                    memory_usage=memory_delta,
                    cpu_usage=psutil.cpu_percent(interval=0.1),
                    throughput=throughput,
                    accuracy=success_rate,
                    custom_metrics={
                        'concurrency_level': concurrency,
                        'tasks_per_second': throughput,
                        'success_rate': success_rate,
                        'memory_per_task': memory_delta / concurrency if concurrency > 0 else 0
                    }
                )

                self.results.append(result)
                print(f"  ✓ {concurrency} concurrent: {throughput:.1f} tasks/sec, {success_rate*100:.1f}% success")

            except Exception as e:
                print(f"  ✗ {concurrency} concurrent: Failed ({e})")

    async def benchmark_memory_efficiency(self):
        """Benchmark memory efficiency and garbage collection."""
        print("\n💾 Benchmarking Memory Efficiency...")

        # Test memory usage under different loads
        data_sizes = [1000, 5000, 10000, 25000]

        for size in data_sizes:
            try:
                # Start memory tracking
                tracemalloc.start()
                gc.collect()  # Clean up before test

                start_memory = psutil.Process().memory_info().rss

                # Allocate and process data
                data_objects = []
                for i in range(100):  # Create 100 objects
                    neural_data = self._generate_neural_data(size, 8)
                    data_objects.append(neural_data)

                    # Process every 10th object to simulate real usage
                    if i % 10 == 0:
                        await self.neural_processor.process_data(neural_data)

                peak_memory = psutil.Process().memory_info().rss

                # Clean up and measure
                data_objects.clear()
                gc.collect()

                end_memory = psutil.Process().memory_info().rss
                current, peak_traced = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # Calculate metrics
                memory_used = peak_memory - start_memory
                memory_retained = end_memory - start_memory
                memory_efficiency = 1.0 - (memory_retained / memory_used) if memory_used > 0 else 1.0

                result = BenchmarkResult(
                    test_name=f"memory_efficiency_{size}_samples",
                    execution_time=0,  # Not time-based test
                    memory_usage=memory_used,
                    cpu_usage=psutil.cpu_percent(interval=0.1),
                    accuracy=memory_efficiency,
                    custom_metrics={
                        'data_size': size,
                        'peak_memory_bytes': memory_used,
                        'retained_memory_bytes': memory_retained,
                        'memory_efficiency': memory_efficiency,
                        'traced_peak_bytes': peak_traced,
                        'objects_created': 100
                    }
                )

                self.results.append(result)
                print(f"  ✓ {size} samples: {memory_used/1024/1024:.1f}MB peak, {memory_efficiency*100:.1f}% efficiency")

            except Exception as e:
                print(f"  ✗ {size} samples: Failed ({e})")

    async def benchmark_scalability(self):
        """Benchmark system scalability under increasing load."""
        print("\n📈 Benchmarking Scalability...")

        # Test increasing loads to find performance limits
        load_multipliers = [1, 2, 5, 10, 20]
        base_load = 10  # Base number of operations

        for multiplier in load_multipliers:
            current_load = base_load * multiplier

            try:
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss

                # Create increasing load
                tasks = []
                for i in range(current_load):
                    neural_data = self._generate_neural_data(1000, 8)

                    # Create a complete processing task
                    task = self._complete_processing_task(neural_data)
                    tasks.append(task)

                # Execute all tasks
                results = await asyncio.gather(*tasks, return_exceptions=True)

                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss

                # Calculate scalability metrics
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                successful_results = [r for r in results if not isinstance(r, Exception)]
                success_rate = len(successful_results) / len(results)
                throughput = len(successful_results) / execution_time

                # Calculate scalability coefficient (throughput per unit load)
                scalability_coeff = throughput / current_load

                result = BenchmarkResult(
                    test_name=f"scalability_load_{current_load}",
                    execution_time=execution_time,
                    memory_usage=memory_delta,
                    cpu_usage=psutil.cpu_percent(interval=0.1),
                    throughput=throughput,
                    accuracy=success_rate,
                    custom_metrics={
                        'load_level': current_load,
                        'load_multiplier': multiplier,
                        'scalability_coefficient': scalability_coeff,
                        'memory_per_operation': memory_delta / current_load if current_load > 0 else 0,
                        'time_per_operation': execution_time / current_load if current_load > 0 else 0
                    }
                )

                self.results.append(result)
                print(f"  ✓ Load {current_load}: {throughput:.1f} ops/sec, {success_rate*100:.1f}% success, {scalability_coeff:.3f} scalability")

            except Exception as e:
                print(f"  ✗ Load {current_load}: Failed ({e})")

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print("\n📊 Generating Performance Report...")

        # Categorize results
        categories = {}
        for result in self.results:
            category = result.test_name.split('_')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        # Generate summary statistics
        report = {
            'system_info': self.system_info.__dict__,
            'test_summary': {
                'total_tests': len(self.results),
                'categories': list(categories.keys()),
                'test_duration': sum(r.execution_time for r in self.results),
                'total_memory_used': sum(r.memory_usage for r in self.results if r.memory_usage > 0)
            },
            'category_summaries': {},
            'performance_insights': {},
            'recommendations': []
        }

        # Generate category summaries
        for category, results in categories.items():
            category_summary = {
                'test_count': len(results),
                'avg_execution_time': np.mean([r.execution_time for r in results]),
                'avg_memory_usage': np.mean([r.memory_usage for r in results if r.memory_usage > 0]),
                'avg_throughput': np.mean([r.throughput for r in results if r.throughput is not None]),
                'avg_accuracy': np.mean([r.accuracy for r in results if r.accuracy is not None]),
                'best_performance': None,
                'worst_performance': None
            }

            # Find best and worst performing tests
            if results:
                # Best = highest throughput or lowest latency
                if any(r.throughput for r in results):
                    best_result = max(results, key=lambda x: x.throughput or 0)
                    category_summary['best_performance'] = best_result.test_name

                if any(r.latency for r in results):
                    worst_result = max(results, key=lambda x: x.latency or 0)
                    category_summary['worst_performance'] = worst_result.test_name

            report['category_summaries'][category] = category_summary

        # Generate performance insights
        insights = self._generate_performance_insights(categories)
        report['performance_insights'] = insights

        # Generate recommendations
        recommendations = self._generate_recommendations(categories)
        report['recommendations'] = recommendations

        return report

    def _generate_performance_insights(self, categories: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Generate performance insights from benchmark results."""
        insights = {}

        # Latency analysis
        if 'end' in categories:  # end-to-end tests
            latencies = [r.latency for r in categories['end'] if r.latency is not None]
            if latencies:
                insights['latency'] = {
                    'min_ms': min(latencies),
                    'max_ms': max(latencies),
                    'avg_ms': np.mean(latencies),
                    'p95_ms': np.percentile(latencies, 95),
                    'target_met': np.mean(latencies) < 100  # 100ms target
                }

        # Throughput analysis
        throughputs = [r.throughput for r in self.results if r.throughput is not None]
        if throughputs:
            insights['throughput'] = {
                'min_ops_sec': min(throughputs),
                'max_ops_sec': max(throughputs),
                'avg_ops_sec': np.mean(throughputs),
                'total_capacity': sum(throughputs)
            }

        # Memory efficiency
        memory_usages = [r.memory_usage for r in self.results if r.memory_usage > 0]
        if memory_usages:
            insights['memory'] = {
                'total_used_mb': sum(memory_usages) / 1024 / 1024,
                'avg_per_test_mb': np.mean(memory_usages) / 1024 / 1024,
                'peak_usage_mb': max(memory_usages) / 1024 / 1024,
                'efficiency_good': max(memory_usages) < self.system_info.total_memory * 0.1  # <10% of total
            }

        # Accuracy analysis
        accuracies = [r.accuracy for r in self.results if r.accuracy is not None]
        if accuracies:
            insights['accuracy'] = {
                'min_accuracy': min(accuracies),
                'max_accuracy': max(accuracies),
                'avg_accuracy': np.mean(accuracies),
                'target_met': np.mean(accuracies) > 0.8  # 80% target
            }

        return insights

    def _generate_recommendations(self, categories: Dict[str, List[BenchmarkResult]]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Check end-to-end latency
        if 'end' in categories:
            latencies = [r.latency for r in categories['end'] if r.latency is not None]
            if latencies and np.mean(latencies) > 100:
                recommendations.append(
                    f"Consider optimizing end-to-end latency (current avg: {np.mean(latencies):.1f}ms, target: <100ms)"
                )

        # Check memory usage
        memory_usages = [r.memory_usage for r in self.results if r.memory_usage > 0]
        if memory_usages:
            total_memory_mb = sum(memory_usages) / 1024 / 1024
            if total_memory_mb > 1000:  # >1GB
                recommendations.append(
                    f"High memory usage detected ({total_memory_mb:.1f}MB). Consider implementing memory pooling."
                )

        # Check concurrent performance
        if 'concurrent' in categories:
            concurrent_results = categories['concurrent']
            success_rates = [r.accuracy for r in concurrent_results if r.accuracy is not None]
            if success_rates and np.mean(success_rates) < 0.9:
                recommendations.append(
                    f"Concurrent processing success rate is low ({np.mean(success_rates)*100:.1f}%). Review error handling."
                )

        # Check scalability
        if 'scalability' in categories:
            scalability_results = categories['scalability']
            coeffs = [r.custom_metrics.get('scalability_coefficient', 0) for r in scalability_results]
            if coeffs and any(c < 0.01 for c in coeffs):  # Scalability degradation
                recommendations.append(
                    "Scalability degradation detected at high loads. Consider load balancing improvements."
                )

        # Hardware acceleration recommendations
        if 'acceleration' in categories:
            accel_results = categories['acceleration']
            cpu_results = [r for r in accel_results if 'cpu' in r.test_name]
            gpu_results = [r for r in accel_results if 'metal' in r.test_name or 'coreml' in r.test_name]

            if cpu_results and gpu_results:
                cpu_throughput = np.mean([r.throughput for r in cpu_results if r.throughput])
                gpu_throughput = np.mean([r.throughput for r in gpu_results if r.throughput])

                if gpu_throughput > cpu_throughput * 2:
                    recommendations.append(
                        f"GPU acceleration shows {gpu_throughput/cpu_throughput:.1f}x improvement. Enable by default."
                    )

        if not recommendations:
            recommendations.append("Performance is within acceptable ranges. No immediate optimizations needed.")

        return recommendations

    # Helper methods for generating test data
    def _generate_neural_data(self, samples: int, channels: int) -> NeuralData:
        """Generate realistic neural data for testing."""
        # Use proper random generator
        rng = np.random.default_rng()

        # Create realistic neural signal patterns
        data = rng.normal(0, 1, (samples, channels)).astype(np.float32)

        # Add some realistic frequency components
        t = np.linspace(0, samples / 250.0, samples)  # 250 Hz sample rate
        for ch in range(channels):
            # Alpha waves (8-12 Hz)
            data[:, ch] += 0.1 * np.sin(2 * np.pi * 10 * t)
            # Beta waves (13-30 Hz)
            data[:, ch] += 0.05 * np.sin(2 * np.pi * 20 * t)

        return NeuralData(
            data=data,
            timestamp=time.time(),
            sample_rate=250.0,
            channels=channels,
            metadata={'test': True, 'samples': samples}
        )

    def _generate_gesture_signal(self, length: int, gesture_type: str = 'click') -> NeuralSignal:
        """Generate neural signal with specific gesture pattern."""
        rng = np.random.default_rng()

        # Base signal
        channels = rng.normal(0, 0.1, (length, 8)).astype(np.float32)

        # Add gesture-specific patterns
        if gesture_type == 'click':
            # Add a spike in motor cortex channels (2, 3, 4)
            spike_start = length // 3
            spike_end = spike_start + length // 10

            for ch in [2, 3, 4]:
                spike = np.exp(-((np.arange(spike_end - spike_start) - (spike_end - spike_start)//2) ** 2) / 10)
                channels[spike_start:spike_end, ch] += spike * 0.5

        return NeuralSignal(
            channels=channels,
            timestamp=time.time(),
            sample_rate=250.0,
            metadata={'gesture_type': gesture_type}
        )

    def _generate_test_gesture(self):
        """Generate test gesture event."""
        from recognition.gesture_recognition import GestureEvent, GestureType

        gesture_types = [GestureType.CLICK, GestureType.HOLD]
        if hasattr(GestureType, 'SWIPE_LEFT'):
            gesture_types.extend([GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT])

        rng = np.random.default_rng()
        gesture_type = rng.choice(gesture_types)

        return GestureEvent(
            gesture_type=gesture_type,
            confidence=rng.uniform(0.5, 1.0),
            timestamp=time.time(),
            duration=rng.uniform(0.1, 2.0),
            parameters={'test': True}
        )

    async def _process_data_async(self, neural_data: NeuralData):
        """Async data processing for concurrent testing."""
        processed = await self.neural_processor.process_data(neural_data)
        compressed = await self.compressor.compress(processed, CompressionQuality.MEDIUM)
        decompressed = await self.compressor.decompress(compressed)
        return decompressed

    async def _complete_processing_task(self, neural_data: NeuralData):
        """Complete processing task for scalability testing."""
        # Process data
        processed = await self.neural_processor.process_data(neural_data)

        # Compress
        compressed = await self.compressor.compress(processed, CompressionQuality.MEDIUM)

        # Convert to gesture signal
        gesture_signal = NeuralSignal(
            channels=processed.data.T,
            timestamp=time.time(),
            sample_rate=250.0
        )

        # Recognize gesture
        gesture = await self.gesture_recognizer.process_signal(gesture_signal)

        # Map to action
        if gesture:
            action = self.input_mapper.map_gesture(gesture)
            return action

        return None

    def save_results(self, filename: str = "performance_benchmark_results.json"):
        """Save benchmark results to file."""
        report = self.generate_performance_report()

        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = {
                'test_name': result.test_name,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'cpu_usage': result.cpu_usage,
                'throughput': result.throughput,
                'latency': result.latency,
                'accuracy': result.accuracy,
                'error_rate': result.error_rate,
                'custom_metrics': result.custom_metrics
            }
            serializable_results.append(result_dict)

        report['detailed_results'] = serializable_results

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📁 Results saved to {filename}")


async def main():
    """Main benchmark execution function."""
    print("🚀 Apple BCI-HID Compression Bridge - Performance Benchmark Suite")
    print("=" * 70)

    benchmark = PerformanceBenchmark()

    try:
        # Run all benchmarks
        report = await benchmark.run_all_benchmarks()

        # Print summary
        print("\n" + "=" * 70)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 70)

        print(f"\nSystem: {report['system_info']['platform']}")
        print(f"CPU: {report['system_info']['cpu_count']} cores")
        print(f"Memory: {report['system_info']['total_memory'] / (1024**3):.1f} GB")

        print(f"\nTotal Tests: {report['test_summary']['total_tests']}")
        print(f"Categories: {', '.join(report['test_summary']['categories'])}")
        print(f"Total Duration: {report['test_summary']['test_duration']:.2f} seconds")

        # Print category summaries
        print("\n📈 Category Performance:")
        for category, summary in report['category_summaries'].items():
            print(f"  {category.title()}:")
            print(f"    Tests: {summary['test_count']}")
            print(f"    Avg Execution Time: {summary['avg_execution_time']*1000:.2f}ms")
            if summary['avg_throughput'] > 0:
                print(f"    Avg Throughput: {summary['avg_throughput']:.1f} ops/sec")
            if summary['avg_accuracy'] is not None and summary['avg_accuracy'] > 0:
                print(f"    Avg Accuracy: {summary['avg_accuracy']*100:.1f}%")

        # Print insights
        if 'performance_insights' in report:
            insights = report['performance_insights']
            print("\n🎯 Performance Insights:")

            if 'latency' in insights:
                lat = insights['latency']
                print(f"  Latency: {lat['avg_ms']:.1f}ms avg, {lat['p95_ms']:.1f}ms p95")
                print(f"           Target (<100ms): {'✅ MET' if lat['target_met'] else '❌ NOT MET'}")

            if 'throughput' in insights:
                thr = insights['throughput']
                print(f"  Throughput: {thr['avg_ops_sec']:.1f} ops/sec avg, {thr['max_ops_sec']:.1f} ops/sec peak")

            if 'memory' in insights:
                mem = insights['memory']
                print(f"  Memory: {mem['avg_per_test_mb']:.1f}MB avg, {mem['peak_usage_mb']:.1f}MB peak")
                print(f"          Efficiency: {'✅ GOOD' if mem['efficiency_good'] else '⚠️ HIGH USAGE'}")

        # Print recommendations
        if 'recommendations' in report:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")

        # Save results
        benchmark.save_results()

        print("\n✅ Benchmark completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
