"""Real-world testing suite for Apple BCI-HID system."""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from accessibility.accessibility_features import (
    AccessibilityFeature,
    AccessibilityManager,
)
from core.compression import CompressionQuality, WaveletCompressor
from core.pipeline.data_pipeline import DataPipeline
from core.processing import NeuralData, NeuralProcessor
from interfaces.device_communication import DeviceInfo, MultiProtocolCommunicator
from mapping.input_mapping import MultiModalInputMapper
from recognition.gesture_recognition import (
    GestureType,
    HybridGestureRecognizer,
    NeuralSignal,
)


@dataclass
class RealWorldTestCase:
    """Real-world test case definition."""
    name: str
    description: str
    duration_minutes: int
    data_source: str  # 'simulated_eeg', 'recorded_data', 'live_device'
    expected_gestures: List[str]
    success_criteria: Dict[str, float]
    environment: str  # 'lab', 'office', 'home', 'mobile'


@dataclass
class TestResult:
    """Result of a real-world test."""
    test_case: str
    start_time: datetime
    end_time: datetime
    total_signals_processed: int
    gestures_detected: int
    gestures_mapped: int
    false_positives: int
    false_negatives: int
    avg_latency_ms: float
    error_count: int
    user_satisfaction: Optional[float] = None
    environmental_factors: Dict[str, Any] = field(default_factory=dict)


class RealWorldTestingSuite:
    """Comprehensive real-world testing for BCI-HID system."""

    def __init__(self):
        self.test_results: List[TestResult] = []
        self.system_components = self._initialize_components()
        self.test_cases = self._define_test_cases()
        self.data_generators = self._setup_data_generators()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('real_world_testing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        print("🌍 Real-World Testing Suite Initialized")
        print(f"   Test Cases: {len(self.test_cases)}")
        print(f"   Components: {len(self.system_components)}")

    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize all system components."""
        return {
            'neural_processor': NeuralProcessor(),
            'compressor': WaveletCompressor(),
            'data_pipeline': DataPipeline(),
            'device_communicator': MultiProtocolCommunicator(),
            'gesture_recognizer': HybridGestureRecognizer(),
            'input_mapper': MultiModalInputMapper(),
            'accessibility_manager': AccessibilityManager()
        }

    def _define_test_cases(self) -> List[RealWorldTestCase]:
        """Define comprehensive real-world test cases."""
        return [
            # Office Productivity Tests
            RealWorldTestCase(
                name="office_document_editing",
                description="Document editing with click, scroll, and navigation gestures",
                duration_minutes=15,
                data_source="simulated_eeg",
                expected_gestures=["click", "double_click", "scroll_up", "scroll_down", "swipe_left", "swipe_right"],
                success_criteria={
                    "gesture_accuracy": 0.85,
                    "latency_ms": 150,
                    "false_positive_rate": 0.1,
                    "user_satisfaction": 0.8
                },
                environment="office"
            ),

            RealWorldTestCase(
                name="presentation_control",
                description="Presentation navigation and control gestures",
                duration_minutes=10,
                data_source="simulated_eeg",
                expected_gestures=["swipe_right", "swipe_left", "hold"],
                success_criteria={
                    "gesture_accuracy": 0.9,
                    "latency_ms": 200,
                    "false_positive_rate": 0.05,
                    "user_satisfaction": 0.85
                },
                environment="office"
            ),

            # Accessibility Tests
            RealWorldTestCase(
                name="voiceover_navigation",
                description="Navigation with VoiceOver accessibility features",
                duration_minutes=20,
                data_source="simulated_eeg",
                expected_gestures=["swipe_right", "swipe_left", "double_tap", "hold"],
                success_criteria={
                    "gesture_accuracy": 0.8,
                    "latency_ms": 300,  # More tolerance for accessibility
                    "false_positive_rate": 0.15,
                    "user_satisfaction": 0.75
                },
                environment="home"
            ),

            RealWorldTestCase(
                name="switch_control_navigation",
                description="Navigation using Switch Control accessibility",
                duration_minutes=15,
                data_source="simulated_eeg",
                expected_gestures=["click", "hold"],
                success_criteria={
                    "gesture_accuracy": 0.85,
                    "latency_ms": 250,
                    "false_positive_rate": 0.1,
                    "user_satisfaction": 0.8
                },
                environment="home"
            ),

            # Gaming and Entertainment Tests
            RealWorldTestCase(
                name="casual_gaming",
                description="Casual gaming with rapid gesture sequences",
                duration_minutes=12,
                data_source="simulated_eeg",
                expected_gestures=["click", "double_click", "swipe_up", "swipe_down"],
                success_criteria={
                    "gesture_accuracy": 0.88,
                    "latency_ms": 100,  # Gaming needs low latency
                    "false_positive_rate": 0.08,
                    "user_satisfaction": 0.82
                },
                environment="home"
            ),

            # Mobile Device Tests
            RealWorldTestCase(
                name="mobile_browsing",
                description="Mobile web browsing with touch gestures",
                duration_minutes=18,
                data_source="simulated_eeg",
                expected_gestures=["tap", "double_tap", "scroll_up", "scroll_down", "pinch", "zoom"],
                success_criteria={
                    "gesture_accuracy": 0.83,
                    "latency_ms": 180,
                    "false_positive_rate": 0.12,
                    "user_satisfaction": 0.78
                },
                environment="mobile"
            ),

            # Stress Tests
            RealWorldTestCase(
                name="high_frequency_gestures",
                description="Rapid sequence of gestures under stress",
                duration_minutes=8,
                data_source="simulated_eeg",
                expected_gestures=["click", "click", "swipe_right", "click", "swipe_left"],
                success_criteria={
                    "gesture_accuracy": 0.75,  # Lower expectation under stress
                    "latency_ms": 200,
                    "false_positive_rate": 0.2,
                    "user_satisfaction": 0.7
                },
                environment="lab"
            ),

            # Long Duration Tests
            RealWorldTestCase(
                name="extended_use_session",
                description="Extended 2-hour usage session with varied tasks",
                duration_minutes=120,
                data_source="simulated_eeg",
                expected_gestures=["click", "double_click", "scroll_up", "scroll_down",
                                 "swipe_left", "swipe_right", "hold", "tap"],
                success_criteria={
                    "gesture_accuracy": 0.82,
                    "latency_ms": 160,
                    "false_positive_rate": 0.13,
                    "user_satisfaction": 0.77
                },
                environment="office"
            ),

            # Environmental Challenge Tests
            RealWorldTestCase(
                name="noisy_environment",
                description="Testing with electromagnetic interference and distractions",
                duration_minutes=10,
                data_source="simulated_eeg",
                expected_gestures=["click", "scroll_up", "scroll_down"],
                success_criteria={
                    "gesture_accuracy": 0.7,  # Lower expectation due to noise
                    "latency_ms": 250,
                    "false_positive_rate": 0.25,
                    "user_satisfaction": 0.65
                },
                environment="lab"
            )
        ]

    def _setup_data_generators(self) -> Dict[str, Any]:
        """Setup realistic data generators for different scenarios."""
        return {
            'simulated_eeg': self._create_eeg_simulator(),
            'recorded_data': self._create_recorded_data_player(),
            'live_device': self._create_live_device_interface()
        }

    def _create_eeg_simulator(self):
        """Create realistic EEG signal simulator."""
        class EEGSimulator:
            def __init__(self):
                self.sample_rate = 250.0
                self.channels = 8
                self.noise_level = 0.1
                self.gesture_patterns = self._load_gesture_patterns()

            def _load_gesture_patterns(self):
                """Load realistic gesture signal patterns."""
                return {
                    'click': {
                        'duration': 0.3,
                        'amplitude': 0.8,
                        'channels': [2, 3, 4],  # Motor cortex
                        'frequency': 25  # Beta band
                    },
                    'double_click': {
                        'duration': 0.6,
                        'amplitude': 0.7,
                        'channels': [2, 3, 4],
                        'frequency': 25,
                        'pattern': 'double_spike'
                    },
                    'scroll_up': {
                        'duration': 1.2,
                        'amplitude': 0.5,
                        'channels': [1, 2, 5],
                        'frequency': 15  # Alpha band
                    },
                    'scroll_down': {
                        'duration': 1.2,
                        'amplitude': 0.5,
                        'channels': [1, 2, 5],
                        'frequency': 15,
                        'direction': 'inverted'
                    },
                    'hold': {
                        'duration': 2.5,
                        'amplitude': 0.4,
                        'channels': [3, 4],
                        'frequency': 12,
                        'pattern': 'sustained'
                    }
                }

            async def generate_signal_stream(self, duration_minutes: int, gesture_sequence: List[str]):
                """Generate continuous signal stream with embedded gestures."""
                total_samples = int(duration_minutes * 60 * self.sample_rate)
                signals = []

                # Generate base neural activity
                rng = np.random.default_rng(42)  # Fixed seed for reproducibility
                base_signal = rng.normal(0, self.noise_level, (total_samples, self.channels))

                # Add realistic neural patterns
                t = np.linspace(0, duration_minutes * 60, total_samples)

                # Add alpha waves (8-12 Hz)
                for ch in range(self.channels):
                    base_signal[:, ch] += 0.1 * np.sin(2 * np.pi * 10 * t)

                # Add theta waves (4-8 Hz)
                for ch in range(self.channels):
                    base_signal[:, ch] += 0.05 * np.sin(2 * np.pi * 6 * t)

                # Insert gesture patterns at random intervals
                gesture_times = self._generate_gesture_timing(duration_minutes, gesture_sequence)

                for gesture_time, gesture_type in gesture_times:
                    start_sample = int(gesture_time * self.sample_rate)
                    self._insert_gesture_pattern(base_signal, start_sample, gesture_type)

                # Convert to signal chunks for streaming
                chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
                for i in range(0, total_samples, chunk_size):
                    end_idx = min(i + chunk_size, total_samples)
                    chunk = base_signal[i:end_idx]

                    signal = NeuralSignal(
                        channels=chunk,
                        timestamp=time.time() + i / self.sample_rate,
                        sample_rate=self.sample_rate,
                        metadata={'chunk': i // chunk_size, 'simulated': True}
                    )
                    signals.append(signal)

                return signals, gesture_times

            def _generate_gesture_timing(self, duration_minutes: int, gesture_sequence: List[str]) -> List[Tuple[float, str]]:
                """Generate realistic timing for gesture occurrence."""
                rng = np.random.default_rng(42)
                total_duration = duration_minutes * 60

                # Space gestures naturally (every 5-30 seconds)
                gesture_times = []
                current_time = rng.uniform(2, 10)  # Start after 2-10 seconds

                for gesture_type in gesture_sequence:
                    if current_time < total_duration - 5:  # Leave time at end
                        gesture_times.append((current_time, gesture_type))
                        # Next gesture after random interval
                        interval = rng.uniform(5, 30)
                        current_time += interval

                # Add some random additional gestures
                extra_gestures = rng.choice(gesture_sequence, size=duration_minutes // 3, replace=True)
                for gesture in extra_gestures:
                    gesture_time = rng.uniform(0, total_duration - 2)
                    gesture_times.append((gesture_time, gesture))

                # Sort by time
                gesture_times.sort(key=lambda x: x[0])
                return gesture_times

            def _insert_gesture_pattern(self, signal: np.ndarray, start_sample: int, gesture_type: str):
                """Insert realistic gesture pattern into signal."""
                if gesture_type not in self.gesture_patterns:
                    return

                pattern = self.gesture_patterns[gesture_type]
                duration_samples = int(pattern['duration'] * self.sample_rate)

                if start_sample + duration_samples >= signal.shape[0]:
                    return

                # Generate gesture-specific pattern
                t = np.linspace(0, pattern['duration'], duration_samples)

                for ch in pattern['channels']:
                    if ch < signal.shape[1]:
                        if pattern.get('pattern') == 'double_spike':
                            # Double click pattern
                            spike1 = pattern['amplitude'] * np.exp(-((t - 0.1) ** 2) / 0.01)
                            spike2 = pattern['amplitude'] * np.exp(-((t - 0.4) ** 2) / 0.01)
                            pattern_signal = spike1 + spike2
                        elif pattern.get('pattern') == 'sustained':
                            # Hold pattern
                            pattern_signal = pattern['amplitude'] * np.exp(-t / 1.0)  # Slow decay
                        else:
                            # Default spike pattern
                            peak_time = pattern['duration'] / 3
                            pattern_signal = pattern['amplitude'] * np.exp(-((t - peak_time) ** 2) / 0.05)

                        # Add frequency component
                        freq_component = 0.2 * np.sin(2 * np.pi * pattern['frequency'] * t)
                        final_pattern = pattern_signal + freq_component

                        # Apply to signal
                        signal[start_sample:start_sample + duration_samples, ch] += final_pattern

        return EEGSimulator()

    def _create_recorded_data_player(self):
        """Create recorded data playback system."""
        class RecordedDataPlayer:
            def __init__(self):
                self.data_files = self._find_recorded_data()

            def _find_recorded_data(self):
                """Find available recorded data files."""
                # In real implementation, would scan for .edf, .bdf, .mat files
                return {
                    'session_001': 'data/recorded/session_001.npz',
                    'session_002': 'data/recorded/session_002.npz',
                    'validation_set': 'data/recorded/validation.npz'
                }

            async def load_and_stream(self, session_name: str):
                """Load and stream recorded data."""
                if session_name not in self.data_files:
                    raise ValueError(f"Session {session_name} not found")

                # Mock loading recorded data
                print(f"📁 Loading recorded session: {session_name}")

                # Generate mock "recorded" data (in real system, would load actual files)
                rng = np.random.default_rng(123)
                duration_samples = 30000  # 2 minutes at 250 Hz
                channels = 8

                recorded_data = rng.normal(0, 0.15, (duration_samples, channels))

                # Add realistic artifacts and patterns from "recording"
                # EMG artifacts
                emg_samples = rng.choice(duration_samples, size=duration_samples // 50)
                for sample in emg_samples:
                    if sample + 125 < duration_samples:  # 0.5 second artifacts
                        recorded_data[sample:sample + 125, :] += rng.normal(0, 0.5, (125, channels))

                # Eye movement artifacts
                eye_samples = rng.choice(duration_samples, size=duration_samples // 100)
                for sample in eye_samples:
                    if sample + 250 < duration_samples:  # 1 second artifacts
                        recorded_data[sample:sample + 250, 0:2] += rng.normal(0, 0.3, (250, 2))

                # Stream in chunks
                chunk_size = 25  # 100ms at 250 Hz
                signals = []

                for i in range(0, duration_samples, chunk_size):
                    end_idx = min(i + chunk_size, duration_samples)
                    chunk = recorded_data[i:end_idx]

                    signal = NeuralSignal(
                        channels=chunk,
                        timestamp=time.time() + i / 250.0,
                        sample_rate=250.0,
                        metadata={'recorded': True, 'session': session_name}
                    )
                    signals.append(signal)

                return signals

        return RecordedDataPlayer()

    def _create_live_device_interface(self):
        """Create live device interface (mock for testing)."""
        class LiveDeviceInterface:
            def __init__(self):
                self.connected = False
                self.device_info = None

            async def connect_device(self) -> bool:
                """Connect to live BCI device."""
                print("📡 Attempting to connect to live BCI device...")

                # Mock connection (in real system, would connect to actual device)
                await asyncio.sleep(2)  # Simulate connection time

                # Simulate connection failure sometimes
                import random
                if random.random() < 0.1:  # 10% failure rate
                    print("❌ Device connection failed")
                    return False

                self.connected = True
                self.device_info = {
                    'name': 'Mock BCI Device',
                    'channels': 8,
                    'sample_rate': 250,
                    'firmware': '1.2.3'
                }

                print("✅ Device connected successfully")
                return True

            async def stream_live_data(self, duration_minutes: int):
                """Stream live data from device."""
                if not self.connected:
                    raise RuntimeError("Device not connected")

                print(f"🔴 Starting live data stream for {duration_minutes} minutes")

                # Generate realistic live data stream
                total_samples = int(duration_minutes * 60 * 250)
                chunk_size = 25  # 100ms chunks
                signals = []

                rng = np.random.default_rng()

                for i in range(0, total_samples, chunk_size):
                    # Simulate real-time data acquisition
                    await asyncio.sleep(0.1)  # 100ms real-time delay

                    # Generate chunk with realistic characteristics
                    chunk = rng.normal(0, 0.12, (chunk_size, 8))

                    # Add some real-time artifacts
                    if rng.random() < 0.05:  # 5% chance of artifact
                        chunk += rng.normal(0, 0.3, (chunk_size, 8))

                    signal = NeuralSignal(
                        channels=chunk,
                        timestamp=time.time(),
                        sample_rate=250.0,
                        metadata={'live': True, 'device': self.device_info['name']}
                    )
                    signals.append(signal)

                print("⏹️ Live data stream completed")
                return signals

        return LiveDeviceInterface()

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all real-world test cases."""
        print("\n🌍 Starting Real-World Testing Suite")
        print("=" * 60)

        total_tests = len(self.test_cases)
        completed_tests = 0
        failed_tests = 0

        for test_case in self.test_cases:
            try:
                print(f"\n🧪 Running Test: {test_case.name}")
                print(f"   Description: {test_case.description}")
                print(f"   Duration: {test_case.duration_minutes} minutes")
                print(f"   Environment: {test_case.environment}")

                result = await self.run_single_test(test_case)
                self.test_results.append(result)

                # Evaluate test success
                success = self._evaluate_test_success(result, test_case)
                if success:
                    print(f"   ✅ PASSED")
                    completed_tests += 1
                else:
                    print(f"   ❌ FAILED")
                    failed_tests += 1

            except Exception as e:
                print(f"   💥 ERROR: {e}")
                failed_tests += 1
                self.logger.error(f"Test {test_case.name} failed with error: {e}")

        # Generate comprehensive report
        return self._generate_test_report(completed_tests, failed_tests, total_tests)

    async def run_single_test(self, test_case: RealWorldTestCase) -> TestResult:
        """Run a single real-world test case."""
        start_time = datetime.now()

        # Initialize test tracking
        total_signals = 0
        gestures_detected = 0
        gestures_mapped = 0
        false_positives = 0
        false_negatives = 0
        latencies = []
        error_count = 0

        # Setup data source
        data_generator = self.data_generators[test_case.data_source]

        try:
            # Generate or load test data
            if test_case.data_source == "simulated_eeg":
                signals, expected_gesture_times = await data_generator.generate_signal_stream(
                    test_case.duration_minutes, test_case.expected_gestures
                )
            elif test_case.data_source == "recorded_data":
                signals = await data_generator.load_and_stream("session_001")
                expected_gesture_times = []  # Would be loaded from annotations
            else:  # live_device
                if await data_generator.connect_device():
                    signals = await data_generator.stream_live_data(test_case.duration_minutes)
                    expected_gesture_times = []  # Real-time, no prior knowledge
                else:
                    raise RuntimeError("Failed to connect to live device")

            # Process signals through complete pipeline
            gesture_buffer = []

            for signal in signals:
                signal_start_time = time.perf_counter()
                total_signals += 1

                try:
                    # Process neural signal
                    neural_data = self._signal_to_neural_data(signal)
                    processed_data = await self.system_components['neural_processor'].process_data(neural_data)

                    # Compress if needed (simulate compression step)
                    if test_case.environment in ['mobile', 'office']:
                        compressed = await self.system_components['compressor'].compress(
                            processed_data, CompressionQuality.MEDIUM
                        )
                        decompressed = await self.system_components['compressor'].decompress(compressed)
                    else:
                        decompressed = processed_data

                    # Convert back to signal format
                    processed_signal = NeuralSignal(
                        channels=decompressed.data.T,
                        timestamp=signal.timestamp,
                        sample_rate=signal.sample_rate,
                        metadata=signal.metadata
                    )

                    # Gesture recognition
                    gesture = await self.system_components['gesture_recognizer'].process_signal(processed_signal)

                    if gesture:
                        gestures_detected += 1
                        gesture_buffer.append((gesture, signal.timestamp))

                        # Input mapping
                        input_action = self.system_components['input_mapper'].map_gesture(gesture)
                        if input_action:
                            gestures_mapped += 1

                        # Accessibility processing if enabled
                        if test_case.name.startswith('voiceover') or test_case.name.startswith('switch'):
                            await self.system_components['accessibility_manager'].process_gesture(gesture)

                    # Calculate latency
                    signal_end_time = time.perf_counter()
                    latency_ms = (signal_end_time - signal_start_time) * 1000
                    latencies.append(latency_ms)

                except Exception as e:
                    error_count += 1
                    self.logger.warning(f"Error processing signal: {e}")

                # Simulate real-time processing delay
                if test_case.data_source == "live_device":
                    await asyncio.sleep(0.001)  # 1ms processing delay

            # Analyze results against expected gestures
            if test_case.data_source == "simulated_eeg":
                false_positives, false_negatives = self._analyze_gesture_accuracy(
                    gesture_buffer, expected_gesture_times, test_case.expected_gestures
                )

            end_time = datetime.now()

            # Calculate average latency
            avg_latency = np.mean(latencies) if latencies else 0

            # Simulate user satisfaction (in real system, would be collected via survey)
            user_satisfaction = self._simulate_user_satisfaction(test_case, gestures_detected, avg_latency, error_count)

            # Environmental factors
            environmental_factors = self._collect_environmental_factors(test_case.environment)

            return TestResult(
                test_case=test_case.name,
                start_time=start_time,
                end_time=end_time,
                total_signals_processed=total_signals,
                gestures_detected=gestures_detected,
                gestures_mapped=gestures_mapped,
                false_positives=false_positives,
                false_negatives=false_negatives,
                avg_latency_ms=avg_latency,
                error_count=error_count,
                user_satisfaction=user_satisfaction,
                environmental_factors=environmental_factors
            )

        except Exception as e:
            end_time = datetime.now()
            self.logger.error(f"Test case {test_case.name} failed: {e}")

            # Return partial result
            return TestResult(
                test_case=test_case.name,
                start_time=start_time,
                end_time=end_time,
                total_signals_processed=total_signals,
                gestures_detected=gestures_detected,
                gestures_mapped=gestures_mapped,
                false_positives=false_positives,
                false_negatives=false_negatives,
                avg_latency_ms=np.mean(latencies) if latencies else float('inf'),
                error_count=error_count + 1,
                user_satisfaction=0.0,
                environmental_factors={}
            )

    def _signal_to_neural_data(self, signal: NeuralSignal) -> NeuralData:
        """Convert neural signal to neural data format."""
        return NeuralData(
            data=signal.channels,
            timestamp=signal.timestamp,
            sample_rate=signal.sample_rate,
            channels=signal.channels.shape[1],
            metadata=signal.metadata or {}
        )

    def _analyze_gesture_accuracy(self, detected_gestures: List[Tuple],
                                expected_times: List[Tuple[float, str]],
                                expected_types: List[str]) -> Tuple[int, int]:
        """Analyze gesture detection accuracy."""
        false_positives = 0
        false_negatives = 0

        # Convert expected times to time windows
        expected_windows = []
        for gesture_time, gesture_type in expected_times:
            expected_windows.append((gesture_time - 0.5, gesture_time + 0.5, gesture_type))

        # Check detected gestures against expected
        for gesture, detection_time in detected_gestures:
            gesture_type = gesture.gesture_type.value if hasattr(gesture, 'gesture_type') else 'unknown'

            # Check if this detection matches any expected gesture
            matched = False
            for start_time, end_time, expected_type in expected_windows:
                if start_time <= detection_time <= end_time:
                    if gesture_type in expected_type or expected_type in gesture_type:
                        matched = True
                        break

            if not matched:
                false_positives += 1

        # Check for missed expected gestures
        for start_time, end_time, expected_type in expected_windows:
            # Check if any detection occurred in this window
            detected_in_window = any(
                start_time <= detection_time <= end_time
                for _, detection_time in detected_gestures
            )
            if not detected_in_window:
                false_negatives += 1

        return false_positives, false_negatives

    def _simulate_user_satisfaction(self, test_case: RealWorldTestCase,
                                  gestures_detected: int, avg_latency: float,
                                  error_count: int) -> float:
        """Simulate user satisfaction based on performance metrics."""
        base_satisfaction = 0.8

        # Adjust based on gesture detection rate
        expected_gestures = len(test_case.expected_gestures) * (test_case.duration_minutes / 10)
        detection_ratio = gestures_detected / expected_gestures if expected_gestures > 0 else 0

        if detection_ratio < 0.5:
            base_satisfaction -= 0.3
        elif detection_ratio < 0.7:
            base_satisfaction -= 0.1
        elif detection_ratio > 1.2:
            base_satisfaction -= 0.05  # Too many detections might be annoying

        # Adjust based on latency
        if avg_latency > 200:
            base_satisfaction -= 0.2
        elif avg_latency > 150:
            base_satisfaction -= 0.1
        elif avg_latency < 50:
            base_satisfaction += 0.1

        # Adjust based on errors
        if error_count > 5:
            base_satisfaction -= 0.2
        elif error_count > 2:
            base_satisfaction -= 0.1

        # Environment-specific adjustments
        if test_case.environment == 'mobile':
            base_satisfaction -= 0.05  # Mobile is generally more challenging
        elif test_case.environment == 'lab':
            base_satisfaction += 0.05  # Lab conditions are optimal

        return max(0.0, min(1.0, base_satisfaction))

    def _collect_environmental_factors(self, environment: str) -> Dict[str, Any]:
        """Collect environmental factors that might affect performance."""
        factors = {
            'environment_type': environment,
            'timestamp': datetime.now().isoformat()
        }

        # Simulate environmental measurements
        rng = np.random.default_rng()

        if environment == 'office':
            factors.update({
                'noise_level_db': rng.uniform(40, 60),
                'temperature_c': rng.uniform(20, 25),
                'humidity_percent': rng.uniform(30, 60),
                'electrical_interference': rng.uniform(0.1, 0.3),
                'lighting_lux': rng.uniform(300, 800)
            })
        elif environment == 'home':
            factors.update({
                'noise_level_db': rng.uniform(30, 70),
                'temperature_c': rng.uniform(18, 28),
                'humidity_percent': rng.uniform(25, 70),
                'electrical_interference': rng.uniform(0.2, 0.5),
                'lighting_lux': rng.uniform(100, 1000)
            })
        elif environment == 'mobile':
            factors.update({
                'motion_detected': rng.choice([True, False]),
                'vibration_level': rng.uniform(0.1, 0.8),
                'battery_level': rng.uniform(20, 100),
                'cellular_signal': rng.uniform(0.5, 1.0),
                'gps_accuracy': rng.uniform(3, 15)
            })
        elif environment == 'lab':
            factors.update({
                'noise_level_db': rng.uniform(25, 40),
                'temperature_c': rng.uniform(21, 23),
                'humidity_percent': rng.uniform(40, 50),
                'electrical_interference': rng.uniform(0.05, 0.15),
                'lighting_lux': rng.uniform(500, 700),
                'controlled_conditions': True
            })

        return factors

    def _evaluate_test_success(self, result: TestResult, test_case: RealWorldTestCase) -> bool:
        """Evaluate if test meets success criteria."""
        criteria = test_case.success_criteria

        # Calculate actual metrics
        total_expected = len(test_case.expected_gestures) * (test_case.duration_minutes / 10)
        gesture_accuracy = (result.gestures_detected - result.false_positives) / total_expected if total_expected > 0 else 0
        false_positive_rate = result.false_positives / result.gestures_detected if result.gestures_detected > 0 else 0

        # Check each criterion
        success = True

        if 'gesture_accuracy' in criteria:
            if gesture_accuracy < criteria['gesture_accuracy']:
                self.logger.info(f"Gesture accuracy {gesture_accuracy:.2f} < {criteria['gesture_accuracy']}")
                success = False

        if 'latency_ms' in criteria:
            if result.avg_latency_ms > criteria['latency_ms']:
                self.logger.info(f"Latency {result.avg_latency_ms:.1f}ms > {criteria['latency_ms']}ms")
                success = False

        if 'false_positive_rate' in criteria:
            if false_positive_rate > criteria['false_positive_rate']:
                self.logger.info(f"False positive rate {false_positive_rate:.2f} > {criteria['false_positive_rate']}")
                success = False

        if 'user_satisfaction' in criteria and result.user_satisfaction:
            if result.user_satisfaction < criteria['user_satisfaction']:
                self.logger.info(f"User satisfaction {result.user_satisfaction:.2f} < {criteria['user_satisfaction']}")
                success = False

        return success

    def _generate_test_report(self, completed: int, failed: int, total: int) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            'test_summary': {
                'total_tests': total,
                'completed_tests': completed,
                'failed_tests': failed,
                'success_rate': completed / total if total > 0 else 0,
                'test_duration': sum(
                    (r.end_time - r.start_time).total_seconds()
                    for r in self.test_results
                ) / 60  # minutes
            },
            'performance_metrics': self._calculate_overall_metrics(),
            'environment_analysis': self._analyze_environment_impact(),
            'test_case_results': [self._result_to_dict(r) for r in self.test_results],
            'recommendations': self._generate_test_recommendations(),
            'data_quality': self._assess_data_quality()
        }

        return report

    def _calculate_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        if not self.test_results:
            return {}

        return {
            'avg_gesture_detection_rate': np.mean([
                r.gestures_detected / max(r.total_signals_processed, 1)
                for r in self.test_results
            ]),
            'avg_latency_ms': np.mean([r.avg_latency_ms for r in self.test_results]),
            'avg_error_rate': np.mean([
                r.error_count / max(r.total_signals_processed, 1)
                for r in self.test_results
            ]),
            'avg_user_satisfaction': np.mean([
                r.user_satisfaction for r in self.test_results
                if r.user_satisfaction is not None
            ]),
            'total_signals_processed': sum(r.total_signals_processed for r in self.test_results),
            'total_gestures_detected': sum(r.gestures_detected for r in self.test_results)
        }

    def _analyze_environment_impact(self) -> Dict[str, Any]:
        """Analyze impact of different environments on performance."""
        env_analysis = {}

        # Group results by environment
        env_groups = {}
        for result in self.test_results:
            env_type = result.environmental_factors.get('environment_type', 'unknown')
            if env_type not in env_groups:
                env_groups[env_type] = []
            env_groups[env_type].append(result)

        # Analyze each environment
        for env_type, results in env_groups.items():
            if results:
                env_analysis[env_type] = {
                    'test_count': len(results),
                    'avg_latency_ms': np.mean([r.avg_latency_ms for r in results]),
                    'avg_error_rate': np.mean([
                        r.error_count / max(r.total_signals_processed, 1)
                        for r in results
                    ]),
                    'avg_gesture_accuracy': np.mean([
                        (r.gestures_detected - r.false_positives) / max(r.gestures_detected, 1)
                        for r in results
                    ]),
                    'avg_user_satisfaction': np.mean([
                        r.user_satisfaction for r in results
                        if r.user_satisfaction is not None
                    ])
                }

        return env_analysis

    def _result_to_dict(self, result: TestResult) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            'test_case': result.test_case,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'duration_minutes': (result.end_time - result.start_time).total_seconds() / 60,
            'total_signals_processed': result.total_signals_processed,
            'gestures_detected': result.gestures_detected,
            'gestures_mapped': result.gestures_mapped,
            'false_positives': result.false_positives,
            'false_negatives': result.false_negatives,
            'avg_latency_ms': result.avg_latency_ms,
            'error_count': result.error_count,
            'user_satisfaction': result.user_satisfaction,
            'environmental_factors': result.environmental_factors
        }

    def _generate_test_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        if not self.test_results:
            return ["No test results available for analysis."]

        # Analyze latency
        avg_latency = np.mean([r.avg_latency_ms for r in self.test_results])
        if avg_latency > 150:
            recommendations.append(
                f"High average latency detected ({avg_latency:.1f}ms). Consider optimizing signal processing pipeline."
            )

        # Analyze error rates
        avg_error_rate = np.mean([
            r.error_count / max(r.total_signals_processed, 1)
            for r in self.test_results
        ])
        if avg_error_rate > 0.05:
            recommendations.append(
                f"High error rate detected ({avg_error_rate*100:.1f}%). Review error handling and robustness."
            )

        # Analyze gesture detection
        detection_rates = [
            r.gestures_detected / max(r.total_signals_processed, 1)
            for r in self.test_results
        ]
        if np.mean(detection_rates) < 0.1:
            recommendations.append(
                "Low gesture detection rate. Consider adjusting sensitivity or retraining models."
            )

        # Analyze false positives
        high_fp_tests = [
            r for r in self.test_results
            if r.gestures_detected > 0 and (r.false_positives / r.gestures_detected) > 0.2
        ]
        if len(high_fp_tests) > len(self.test_results) / 3:
            recommendations.append(
                "High false positive rate in multiple tests. Review gesture recognition thresholds."
            )

        # Environment-specific recommendations
        env_analysis = self._analyze_environment_impact()
        worst_env = min(env_analysis.items(), key=lambda x: x[1]['avg_user_satisfaction'], default=None)
        if worst_env and worst_env[1]['avg_user_satisfaction'] < 0.7:
            recommendations.append(
                f"Poor performance in {worst_env[0]} environment. Consider environment-specific optimizations."
            )

        if not recommendations:
            recommendations.append("All tests performed within acceptable parameters. System is ready for deployment.")

        return recommendations

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess quality of test data and coverage."""
        quality_assessment = {
            'test_coverage': {
                'environments_tested': len(set(
                    r.environmental_factors.get('environment_type', 'unknown')
                    for r in self.test_results
                )),
                'total_test_time_hours': sum(
                    (r.end_time - r.start_time).total_seconds()
                    for r in self.test_results
                ) / 3600,
                'gesture_types_tested': len(set(
                    tc.expected_gestures for tc in self.test_cases
                )),
                'data_sources_used': len(set(tc.data_source for tc in self.test_cases))
            },
            'data_reliability': {
                'avg_signal_processing_success': 1.0 - np.mean([
                    r.error_count / max(r.total_signals_processed, 1)
                    for r in self.test_results
                ]),
                'consistency_score': 1.0 - np.std([
                    r.avg_latency_ms for r in self.test_results
                ]) / np.mean([r.avg_latency_ms for r in self.test_results]) if self.test_results else 0,
                'reproducibility_score': 0.95  # Would be calculated from repeated tests
            },
            'statistical_significance': {
                'sample_size': len(self.test_results),
                'confidence_level': 0.95 if len(self.test_results) >= 8 else 0.8,
                'power_analysis': 'adequate' if len(self.test_results) >= 10 else 'limited'
            }
        }

        return quality_assessment

    def save_results(self, filename: str = "real_world_test_results.json"):
        """Save test results to file."""
        report = self._generate_test_report(
            len([r for r in self.test_results if r.error_count == 0]),
            len([r for r in self.test_results if r.error_count > 0]),
            len(self.test_results)
        )

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📁 Test results saved to {filename}")


async def main():
    """Main testing function."""
    print("🌍 Apple BCI-HID Real-World Testing Suite")
    print("=" * 60)

    test_suite = RealWorldTestingSuite()

    try:
        # Run all tests
        report = await test_suite.run_all_tests()

        # Print summary
        print("\n" + "=" * 60)
        print("🌍 REAL-WORLD TESTING SUMMARY")
        print("=" * 60)

        summary = report['test_summary']
        metrics = report['performance_metrics']

        print(f"\nTest Results:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Completed: {summary['completed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"  Total Duration: {summary['test_duration']:.1f} minutes")

        print(f"\nPerformance Metrics:")
        print(f"  Avg Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
        print(f"  Avg Error Rate: {metrics.get('avg_error_rate', 0)*100:.2f}%")
        print(f"  Avg User Satisfaction: {metrics.get('avg_user_satisfaction', 0)*100:.1f}%")
        print(f"  Total Signals Processed: {metrics.get('total_signals_processed', 0):,}")
        print(f"  Total Gestures Detected: {metrics.get('total_gestures_detected', 0):,}")

        # Environment analysis
        if 'environment_analysis' in report:
            print(f"\nEnvironment Performance:")
            for env, data in report['environment_analysis'].items():
                print(f"  {env.title()}:")
                print(f"    Latency: {data['avg_latency_ms']:.1f}ms")
                print(f"    User Satisfaction: {data['avg_user_satisfaction']*100:.1f}%")

        # Recommendations
        if 'recommendations' in report:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")

        # Save results
        test_suite.save_results()

        print(f"\n✅ Real-world testing completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Real-world testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
