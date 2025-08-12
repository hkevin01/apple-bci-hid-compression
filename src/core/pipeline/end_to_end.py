"""Synchronous end-to-end neural -> compression -> intent -> HID pipeline."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...hid_interface import HIDEvent, select_backend
from ...neural_translation.intent_translator import IntentTranslator
from ..compression import WaveletCompressor


def process_neural_input(samples: np.ndarray, compressor: Optional[WaveletCompressor] = None) -> Optional[HIDEvent]:
    """Process one frame of neural samples and emit a HID event if any.

    Steps:
      1. Compress + decompress (simulating transmission constraints)
      2. Run gesture recognition + mapping
      3. Produce HID event
    """
    if samples.size == 0:
        return None
    compressor = compressor or WaveletCompressor(level=2, top_k_ratio=0.2)
    compressed = compressor.compress(samples)
    _ = compressor.decompress(compressed)  # currently not used downstream
    translator = IntentTranslator()
    # Translator is async; run a minimal loop
    import asyncio
    async def _translate():
        return await translator.translate(samples)
    res = asyncio.run(_translate())
    backend = select_backend()
    if res.hid_event:
        backend.send(res.hid_event)
    return res.hid_event
    return res.hid_event
