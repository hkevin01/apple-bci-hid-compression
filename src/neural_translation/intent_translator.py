"""Intent translation integrating gesture recognition and input mapping."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.hid_interface import HIDEvent, create_click, create_mouse_move
from src.mapping.input_mapping import FixedInputMapper, InputAction, InputType
from src.recognition.gesture_recognition import HybridGestureRecognizer, NeuralSignal


@dataclass
class TranslationResult:
    gesture: str | None
    action: InputAction | None
    hid_event: HIDEvent | None


class IntentTranslator:
    sample_rate: float
    window_size: int
    _recognizer: HybridGestureRecognizer
    _mapper: FixedInputMapper
    _buffer: deque[np.ndarray]

    def __init__(self, sample_rate: float = 1000.0, window_size: int = 50):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self._recognizer = HybridGestureRecognizer()
        self._mapper = FixedInputMapper()
        # rolling buffer for potential temporal features
        self._buffer = deque(maxlen=window_size)

    def _to_signal(self, samples: np.ndarray) -> NeuralSignal:
        return NeuralSignal(
            channels=samples.astype(np.float32),
            timestamp=time.time(),
            sample_rate=self.sample_rate,
        )

    async def translate(self, samples: np.ndarray) -> TranslationResult:
        self._buffer.append(samples)
        signal = self._to_signal(samples)
        gesture_event = await self._recognizer.process_signal(signal)
        if not gesture_event:
            return TranslationResult(None, None, None)
        action = self._mapper.map_gesture(gesture_event)
        hid_event: HIDEvent | None = None
        if action:
            hid_event = self._action_to_hid(action)
        return TranslationResult(
            gesture_event.gesture_type.value, action, hid_event
        )

    def _action_to_hid(self, action: InputAction) -> HIDEvent | None:
        if action.action_type == InputType.MOUSE_CLICK:
            return create_click()
        if action.action_type == InputType.MOUSE_SCROLL:
            dy = action.parameters.get('amount', 1)
            return create_mouse_move(0, float(dy))
        if action.action_type == InputType.MOUSE_MOVE:
            params = action.parameters
            return create_mouse_move(
                params.get('dx', 0.0), params.get('dy', 0.0)
            )
        return None


class AsyncIntentSession:
    def __init__(self, translator: IntentTranslator | None = None):
        self.translator = translator or IntentTranslator()

    async def push_and_translate(
        self, frame: np.ndarray
    ) -> HIDEvent | None:
        res = await self.translator.translate(frame)
        return res.hid_event
