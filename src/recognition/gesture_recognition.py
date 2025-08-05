"""Gesture recognition system implementations."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np


class GestureType(Enum):
    """Types of gestures that can be recognized."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH = "pinch"
    ZOOM = "zoom"
    ROTATE = "rotate"
    HOLD = "hold"
    TAP = "tap"
    PRESS = "press"
    RELEASE = "release"
    MOVE = "move"
    CUSTOM = "custom"


class GestureState(Enum):
    """States of gesture recognition."""
    IDLE = "idle"
    DETECTING = "detecting"
    RECOGNIZED = "recognized"
    REJECTED = "rejected"


@dataclass
class NeuralSignal:
    """Neural signal data structure."""
    channels: np.ndarray
    timestamp: float
    sample_rate: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GestureEvent:
    """Gesture recognition event."""
    gesture_type: GestureType
    confidence: float
    timestamp: float
    duration: float
    parameters: Dict[str, Any] = None
    raw_signals: Optional[List[NeuralSignal]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class GesturePattern:
    """Gesture pattern definition."""
    name: str
    gesture_type: GestureType
    features: Dict[str, Any]
    thresholds: Dict[str, float]
    duration_range: Tuple[float, float]  # min, max duration in seconds
    confidence_threshold: float = 0.7


class GestureRecognizer(Protocol):
    """Protocol for gesture recognition implementations."""

    async def process_signal(self, signal: NeuralSignal) -> Optional[GestureEvent]:
        """Process a neural signal and return detected gesture."""
        ...

    def add_pattern(self, pattern: GesturePattern) -> bool:
        """Add a new gesture pattern."""
        ...

    def remove_pattern(self, pattern_name: str) -> bool:
        """Remove a gesture pattern."""
        ...


class MLBasedGestureRecognizer:
    """Machine learning-based gesture recognition."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.patterns: Dict[str, GesturePattern] = {}
        self.signal_buffer: List[NeuralSignal] = []
        self.buffer_size = 100  # Number of signals to keep in buffer
        self.feature_extractor = FeatureExtractor()
        self.gesture_classifier = GestureClassifier()
        self.state = GestureState.IDLE

        # Initialize with default patterns
        self._load_default_patterns()

    def _load_default_patterns(self):
        """Load default gesture patterns."""
        # Click pattern - quick spike in motor cortex
        click_pattern = GesturePattern(
            name="click",
            gesture_type=GestureType.CLICK,
            features={
                "peak_amplitude": {"min": 0.3, "max": 1.0},
                "duration": {"min": 0.1, "max": 0.5},
                "frequency_bands": {
                    "beta": {"min": 0.2, "max": 0.8},  # 13-30 Hz
                    "gamma": {"min": 0.1, "max": 0.6}  # 30-100 Hz
                }
            },
            thresholds={
                "amplitude_threshold": 0.4,
                "duration_threshold": 0.3,
                "frequency_threshold": 0.3
            },
            duration_range=(0.1, 0.5),
            confidence_threshold=0.7
        )
        self.patterns["click"] = click_pattern

        # Scroll patterns
        scroll_up_pattern = GesturePattern(
            name="scroll_up",
            gesture_type=GestureType.SCROLL_UP,
            features={
                "direction_vector": {"x": 0, "y": 1},
                "sustained_activity": {"min": 0.5, "max": 2.0},
                "channel_activation": [2, 3, 4]  # Specific motor cortex areas
            },
            thresholds={
                "direction_confidence": 0.6,
                "activity_threshold": 0.3
            },
            duration_range=(0.5, 2.0),
            confidence_threshold=0.6
        )
        self.patterns["scroll_up"] = scroll_up_pattern

        # Hold pattern - sustained activity
        hold_pattern = GesturePattern(
            name="hold",
            gesture_type=GestureType.HOLD,
            features={
                "sustained_amplitude": {"min": 0.2, "max": 0.8},
                "stability": {"variance_threshold": 0.1},
                "duration": {"min": 1.0, "max": 10.0}
            },
            thresholds={
                "stability_threshold": 0.8,
                "amplitude_consistency": 0.7
            },
            duration_range=(1.0, 10.0),
            confidence_threshold=0.75
        )
        self.patterns["hold"] = hold_pattern

    async def process_signal(self, signal: NeuralSignal) -> Optional[GestureEvent]:
        """Process neural signal using ML-based recognition."""
        # Add signal to buffer
        self.signal_buffer.append(signal)
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

        # Need minimum signals for recognition
        if len(self.signal_buffer) < 10:
            return None

        # Extract features from recent signals
        features = self.feature_extractor.extract_features(self.signal_buffer[-20:])

        # Classify gesture
        gesture_result = self.gesture_classifier.classify(features, self.patterns)

        if gesture_result:
            confidence = gesture_result['confidence']
            gesture_type = gesture_result['gesture_type']

            # Create gesture event
            gesture_event = GestureEvent(
                gesture_type=gesture_type,
                confidence=confidence,
                timestamp=signal.timestamp,
                duration=gesture_result.get('duration', 0.0),
                parameters=gesture_result.get('parameters', {}),
                raw_signals=self.signal_buffer[-20:].copy()
            )

            print(f"ML: Recognized {gesture_type.value} with confidence {confidence:.2f}")
            return gesture_event

        return None

    def add_pattern(self, pattern: GesturePattern) -> bool:
        """Add new gesture pattern."""
        try:
            self.patterns[pattern.name] = pattern
            print(f"Added gesture pattern: {pattern.name}")
            return True
        except Exception as e:
            print(f"Failed to add pattern {pattern.name}: {e}")
            return False

    def remove_pattern(self, pattern_name: str) -> bool:
        """Remove gesture pattern."""
        if pattern_name in self.patterns:
            del self.patterns[pattern_name]
            print(f"Removed gesture pattern: {pattern_name}")
            return True
        return False

    def train_pattern(self, pattern_name: str, training_signals: List[NeuralSignal]):
        """Train a gesture pattern from example signals."""
        if not training_signals:
            return False

        # Extract features from training data
        all_features = []
        for signals in [training_signals[i:i+20] for i in range(0, len(training_signals), 20)]:
            if len(signals) >= 10:
                features = self.feature_extractor.extract_features(signals)
                all_features.append(features)

        if not all_features:
            return False

        # Calculate pattern statistics
        pattern_stats = self._calculate_pattern_statistics(all_features)

        # Create new pattern
        new_pattern = GesturePattern(
            name=pattern_name,
            gesture_type=GestureType.CUSTOM,
            features=pattern_stats['features'],
            thresholds=pattern_stats['thresholds'],
            duration_range=pattern_stats['duration_range'],
            confidence_threshold=0.7
        )

        self.patterns[pattern_name] = new_pattern
        print(f"Trained new pattern: {pattern_name}")
        return True

    def _calculate_pattern_statistics(self, feature_sets: List[Dict]) -> Dict:
        """Calculate statistics from training features."""
        # Simple statistical analysis
        stats = {
            'features': {},
            'thresholds': {},
            'duration_range': (0.1, 2.0)
        }

        # Extract common features
        if feature_sets:
            first_features = feature_sets[0]
            for key in first_features:
                if isinstance(first_features[key], (int, float)):
                    values = [fs.get(key, 0) for fs in feature_sets]
                    stats['features'][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                    stats['thresholds'][key] = np.mean(values) * 0.8

        return stats


class RuleBasedGestureRecognizer:
    """Rule-based gesture recognition using predefined rules."""

    def __init__(self):
        self.patterns: Dict[str, GesturePattern] = {}
        self.signal_buffer: List[NeuralSignal] = []
        self.buffer_size = 50
        self.state_machine = GestureStateMachine()

        # Load default rule-based patterns
        self._load_rule_patterns()

    def _load_rule_patterns(self):
        """Load rule-based gesture patterns."""
        # Simple amplitude-based click
        click_pattern = GesturePattern(
            name="rule_click",
            gesture_type=GestureType.CLICK,
            features={
                "amplitude_spike": True,
                "quick_return": True,
                "channel_focus": [2, 3]  # Motor cortex
            },
            thresholds={
                "spike_threshold": 0.5,
                "duration_max": 0.4
            },
            duration_range=(0.05, 0.4),
            confidence_threshold=0.6
        )
        self.patterns["rule_click"] = click_pattern

        # Sustained hold
        hold_pattern = GesturePattern(
            name="rule_hold",
            gesture_type=GestureType.HOLD,
            features={
                "sustained_level": True,
                "minimal_variance": True
            },
            thresholds={
                "level_threshold": 0.3,
                "variance_threshold": 0.1,
                "min_duration": 1.0
            },
            duration_range=(1.0, 5.0),
            confidence_threshold=0.7
        )
        self.patterns["rule_hold"] = hold_pattern

    async def process_signal(self, signal: NeuralSignal) -> Optional[GestureEvent]:
        """Process signal using rule-based recognition."""
        # Add to buffer
        self.signal_buffer.append(signal)
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

        # Apply rules to recent signals
        recent_signals = self.signal_buffer[-10:]

        for pattern_name, pattern in self.patterns.items():
            if self._apply_rules(recent_signals, pattern):
                # Calculate confidence based on rule satisfaction
                confidence = self._calculate_rule_confidence(recent_signals, pattern)

                if confidence >= pattern.confidence_threshold:
                    gesture_event = GestureEvent(
                        gesture_type=pattern.gesture_type,
                        confidence=confidence,
                        timestamp=signal.timestamp,
                        duration=self._estimate_duration(recent_signals, pattern),
                        parameters={'pattern_name': pattern_name},
                        raw_signals=recent_signals.copy()
                    )

                    print(f"Rule: Recognized {pattern.gesture_type.value} with confidence {confidence:.2f}")
                    return gesture_event

        return None

    def _apply_rules(self, signals: List[NeuralSignal], pattern: GesturePattern) -> bool:
        """Apply rule-based logic to signals."""
        if not signals:
            return False

        features = pattern.features
        thresholds = pattern.thresholds

        # Rule for click: amplitude spike followed by quick return
        if pattern.gesture_type == GestureType.CLICK:
            if len(signals) < 5:
                return False

            # Check for amplitude spike
            amplitudes = [np.max(s.channels) for s in signals]
            max_amp = max(amplitudes)

            if max_amp < thresholds.get('spike_threshold', 0.5):
                return False

            # Check for quick return to baseline
            spike_idx = amplitudes.index(max_amp)
            if spike_idx < len(amplitudes) - 2:
                post_spike = amplitudes[spike_idx + 1:]
                if all(amp < max_amp * 0.5 for amp in post_spike):
                    return True

        # Rule for hold: sustained amplitude with low variance
        elif pattern.gesture_type == GestureType.HOLD:
            if len(signals) < 10:
                return False

            amplitudes = [np.mean(s.channels) for s in signals]
            mean_amp = np.mean(amplitudes)
            variance = np.var(amplitudes)

            level_threshold = thresholds.get('level_threshold', 0.3)
            variance_threshold = thresholds.get('variance_threshold', 0.1)

            if mean_amp >= level_threshold and variance <= variance_threshold:
                return True

        return False

    def _calculate_rule_confidence(self, signals: List[NeuralSignal],
                                 pattern: GesturePattern) -> float:
        """Calculate confidence for rule-based recognition."""
        if not signals:
            return 0.0

        # Simple confidence calculation based on signal strength
        amplitudes = [np.max(s.channels) for s in signals]

        if pattern.gesture_type == GestureType.CLICK:
            max_amp = max(amplitudes)
            confidence = min(max_amp / 0.8, 1.0)  # Normalize to 0-1
            return confidence

        elif pattern.gesture_type == GestureType.HOLD:
            mean_amp = np.mean(amplitudes)
            variance = np.var(amplitudes)
            # Higher amplitude and lower variance = higher confidence
            confidence = mean_amp * (1.0 - min(variance, 1.0))
            return min(confidence, 1.0)

        return 0.5  # Default confidence

    def _estimate_duration(self, signals: List[NeuralSignal],
                          pattern: GesturePattern) -> float:
        """Estimate gesture duration."""
        if len(signals) < 2:
            return 0.0

        return signals[-1].timestamp - signals[0].timestamp

    def add_pattern(self, pattern: GesturePattern) -> bool:
        """Add rule-based pattern."""
        self.patterns[pattern.name] = pattern
        return True

    def remove_pattern(self, pattern_name: str) -> bool:
        """Remove rule-based pattern."""
        if pattern_name in self.patterns:
            del self.patterns[pattern_name]
            return True
        return False


class HybridGestureRecognizer:
    """Hybrid approach combining ML and rule-based recognition."""

    def __init__(self, ml_weight: float = 0.7, rule_weight: float = 0.3):
        self.ml_recognizer = MLBasedGestureRecognizer()
        self.rule_recognizer = RuleBasedGestureRecognizer()
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.gesture_fusion = GestureFusion()

        # Gesture history for temporal analysis
        self.gesture_history: List[GestureEvent] = []
        self.history_size = 20

    async def process_signal(self, signal: NeuralSignal) -> Optional[GestureEvent]:
        """Process signal using hybrid approach."""
        # Get results from both recognizers
        ml_result = await self.ml_recognizer.process_signal(signal)
        rule_result = await self.rule_recognizer.process_signal(signal)

        # Fusion of results
        if ml_result and rule_result:
            # Both detected gestures
            fused_result = self.gesture_fusion.fuse_gestures(
                ml_result, rule_result, self.ml_weight, self.rule_weight
            )
            if fused_result:
                self._add_to_history(fused_result)
                return fused_result

        elif ml_result:
            # Only ML detected
            if ml_result.confidence >= 0.6:
                self._add_to_history(ml_result)
                return ml_result

        elif rule_result:
            # Only rules detected
            if rule_result.confidence >= 0.5:
                self._add_to_history(rule_result)
                return rule_result

        return None

    def _add_to_history(self, gesture: GestureEvent):
        """Add gesture to history."""
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)

    def add_pattern(self, pattern: GesturePattern) -> bool:
        """Add pattern to both recognizers."""
        ml_success = self.ml_recognizer.add_pattern(pattern)
        rule_success = self.rule_recognizer.add_pattern(pattern)
        return ml_success or rule_success

    def remove_pattern(self, pattern_name: str) -> bool:
        """Remove pattern from both recognizers."""
        ml_success = self.ml_recognizer.remove_pattern(pattern_name)
        rule_success = self.rule_recognizer.remove_pattern(pattern_name)
        return ml_success or rule_success

    def get_gesture_statistics(self) -> Dict[str, Any]:
        """Get recognition statistics."""
        if not self.gesture_history:
            return {}

        gesture_counts = {}
        confidence_sums = {}

        for gesture in self.gesture_history:
            gesture_type = gesture.gesture_type.value
            gesture_counts[gesture_type] = gesture_counts.get(gesture_type, 0) + 1
            confidence_sums[gesture_type] = confidence_sums.get(gesture_type, 0) + gesture.confidence

        stats = {
            'total_gestures': len(self.gesture_history),
            'gesture_counts': gesture_counts,
            'average_confidence': {
                gesture_type: confidence_sums[gesture_type] / gesture_counts[gesture_type]
                for gesture_type in gesture_counts
            },
            'recent_activity': len([g for g in self.gesture_history
                                  if time.time() - g.timestamp < 60])  # Last minute
        }

        return stats


class FeatureExtractor:
    """Extract features from neural signals for ML processing."""

    def extract_features(self, signals: List[NeuralSignal]) -> Dict[str, Any]:
        """Extract comprehensive features from signals."""
        if not signals:
            return {}

        # Convert signals to numpy array
        signal_data = np.array([s.channels for s in signals])
        timestamps = np.array([s.timestamp for s in signals])

        features = {}

        # Time-domain features
        features.update(self._extract_time_features(signal_data))

        # Frequency-domain features
        features.update(self._extract_frequency_features(signal_data, signals[0].sample_rate))

        # Spatial features (across channels)
        features.update(self._extract_spatial_features(signal_data))

        # Temporal features
        features.update(self._extract_temporal_features(signal_data, timestamps))

        return features

    def _extract_time_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract time-domain features."""
        features = {}

        # Basic statistics
        features['mean_amplitude'] = float(np.mean(data))
        features['max_amplitude'] = float(np.max(data))
        features['min_amplitude'] = float(np.min(data))
        features['std_amplitude'] = float(np.std(data))
        features['variance'] = float(np.var(data))

        # Peak-related features
        features['peak_count'] = int(self._count_peaks(data))
        features['peak_prominence'] = float(self._calculate_peak_prominence(data))

        # Energy features
        features['signal_energy'] = float(np.sum(data ** 2))
        features['rms'] = float(np.sqrt(np.mean(data ** 2)))

        return features

    def _extract_frequency_features(self, data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        """Extract frequency-domain features."""
        features = {}

        # FFT analysis
        fft_data = np.fft.fft(data, axis=0)
        freqs = np.fft.fftfreq(len(data), 1/sample_rate)
        power_spectrum = np.abs(fft_data) ** 2

        # Band power features
        features['delta_power'] = float(self._band_power(power_spectrum, freqs, 0.5, 4))
        features['theta_power'] = float(self._band_power(power_spectrum, freqs, 4, 8))
        features['alpha_power'] = float(self._band_power(power_spectrum, freqs, 8, 13))
        features['beta_power'] = float(self._band_power(power_spectrum, freqs, 13, 30))
        features['gamma_power'] = float(self._band_power(power_spectrum, freqs, 30, 100))

        # Spectral features
        features['dominant_frequency'] = float(freqs[np.argmax(np.mean(power_spectrum, axis=1))])
        features['spectral_centroid'] = float(self._spectral_centroid(power_spectrum, freqs))

        return features

    def _extract_spatial_features(self, data: np.ndarray) -> Dict[str, float]:
        """Extract spatial features across channels."""
        features = {}

        if data.shape[1] > 1:  # Multiple channels
            # Channel correlations
            corr_matrix = np.corrcoef(data.T)
            features['mean_correlation'] = float(np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            features['max_correlation'] = float(np.max(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))

            # Channel-wise statistics
            channel_means = np.mean(data, axis=0)
            features['channel_asymmetry'] = float(np.std(channel_means))
            features['dominant_channel'] = int(np.argmax(np.var(data, axis=0)))

        return features

    def _extract_temporal_features(self, data: np.ndarray, timestamps: np.ndarray) -> Dict[str, float]:
        """Extract temporal features."""
        features = {}

        # Duration
        features['duration'] = float(timestamps[-1] - timestamps[0])

        # Rate of change
        if len(data) > 1:
            diff_data = np.diff(data, axis=0)
            features['mean_rate_of_change'] = float(np.mean(np.abs(diff_data)))
            features['max_rate_of_change'] = float(np.max(np.abs(diff_data)))

        return features

    def _count_peaks(self, data: np.ndarray) -> int:
        """Count peaks in signal."""
        # Simple peak detection
        mean_data = np.mean(data, axis=1) if len(data.shape) > 1 else data
        threshold = np.mean(mean_data) + np.std(mean_data)
        peaks = 0

        for i in range(1, len(mean_data) - 1):
            if (mean_data[i] > mean_data[i-1] and
                mean_data[i] > mean_data[i+1] and
                mean_data[i] > threshold):
                peaks += 1

        return peaks

    def _calculate_peak_prominence(self, data: np.ndarray) -> float:
        """Calculate average peak prominence."""
        mean_data = np.mean(data, axis=1) if len(data.shape) > 1 else data
        baseline = np.mean(mean_data)
        max_peak = np.max(mean_data)
        return max_peak - baseline

    def _band_power(self, power_spectrum: np.ndarray, freqs: np.ndarray,
                   low_freq: float, high_freq: float) -> float:
        """Calculate power in frequency band."""
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        return np.sum(power_spectrum[band_mask])

    def _spectral_centroid(self, power_spectrum: np.ndarray, freqs: np.ndarray) -> float:
        """Calculate spectral centroid."""
        power_sum = np.sum(power_spectrum, axis=1)
        weighted_freqs = freqs * power_sum
        return np.sum(weighted_freqs) / np.sum(power_sum)


class GestureClassifier:
    """Simple gesture classifier for ML-based recognition."""

    def classify(self, features: Dict[str, Any], patterns: Dict[str, GesturePattern]) -> Optional[Dict]:
        """Classify gesture based on features."""
        best_match = None
        best_confidence = 0.0

        for pattern_name, pattern in patterns.items():
            confidence = self._calculate_pattern_match(features, pattern)

            if confidence > best_confidence and confidence >= pattern.confidence_threshold:
                best_confidence = confidence
                best_match = {
                    'gesture_type': pattern.gesture_type,
                    'confidence': confidence,
                    'pattern_name': pattern_name,
                    'duration': features.get('duration', 0.0),
                    'parameters': {'features': features}
                }

        return best_match

    def _calculate_pattern_match(self, features: Dict[str, Any],
                               pattern: GesturePattern) -> float:
        """Calculate how well features match a pattern."""
        if not features:
            return 0.0

        match_scores = []
        pattern_features = pattern.features

        # Check amplitude-based features
        if 'peak_amplitude' in pattern_features:
            amp_range = pattern_features['peak_amplitude']
            feature_amp = features.get('max_amplitude', 0)
            if amp_range['min'] <= feature_amp <= amp_range['max']:
                match_scores.append(0.8)
            else:
                match_scores.append(0.2)

        # Check duration
        if 'duration' in pattern_features:
            dur_range = pattern_features['duration']
            feature_dur = features.get('duration', 0)
            if dur_range['min'] <= feature_dur <= dur_range['max']:
                match_scores.append(0.9)
            else:
                match_scores.append(0.1)

        # Check frequency bands
        if 'frequency_bands' in pattern_features:
            freq_bands = pattern_features['frequency_bands']
            freq_score = 0.0
            freq_count = 0

            for band, band_range in freq_bands.items():
                feature_key = f"{band}_power"
                if feature_key in features:
                    power = features[feature_key]
                    normalized_power = min(power / 1000.0, 1.0)  # Normalize
                    if band_range['min'] <= normalized_power <= band_range['max']:
                        freq_score += 1.0
                    freq_count += 1

            if freq_count > 0:
                match_scores.append(freq_score / freq_count)

        # Return average match score
        return np.mean(match_scores) if match_scores else 0.0


class GestureFusion:
    """Fusion logic for combining multiple gesture recognition results."""

    def fuse_gestures(self, ml_result: GestureEvent, rule_result: GestureEvent,
                     ml_weight: float, rule_weight: float) -> Optional[GestureEvent]:
        """Fuse ML and rule-based results."""
        # If gestures match, combine confidences
        if ml_result.gesture_type == rule_result.gesture_type:
            combined_confidence = (
                ml_result.confidence * ml_weight +
                rule_result.confidence * rule_weight
            )

            return GestureEvent(
                gesture_type=ml_result.gesture_type,
                confidence=combined_confidence,
                timestamp=max(ml_result.timestamp, rule_result.timestamp),
                duration=max(ml_result.duration, rule_result.duration),
                parameters={
                    'ml_confidence': ml_result.confidence,
                    'rule_confidence': rule_result.confidence,
                    'fusion_method': 'weighted_average'
                },
                raw_signals=ml_result.raw_signals or rule_result.raw_signals
            )

        # If gestures don't match, choose higher confidence
        else:
            if ml_result.confidence > rule_result.confidence:
                return ml_result
            else:
                return rule_result


class GestureStateMachine:
    """State machine for gesture recognition states."""

    def __init__(self):
        self.current_state = GestureState.IDLE
        self.state_start_time = time.time()
        self.state_history: List[Tuple[GestureState, float]] = []

    def transition_to(self, new_state: GestureState):
        """Transition to new state."""
        current_time = time.time()
        self.state_history.append((self.current_state, current_time - self.state_start_time))

        self.current_state = new_state
        self.state_start_time = current_time

        # Keep only recent history
        if len(self.state_history) > 50:
            self.state_history.pop(0)

    def get_current_state(self) -> GestureState:
        """Get current state."""
        return self.current_state

    def time_in_current_state(self) -> float:
        """Get time spent in current state."""
        return time.time() - self.state_start_time
