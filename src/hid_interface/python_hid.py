"""Python HID backend abstraction (mock + mac adapter)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from src.interfaces.hid_protocol import HIDEvent as _SysHIDEvent
    from src.interfaces.hid_protocol import HIDEventType as _SysHIDEventType
    from src.interfaces.hid_protocol import HIDProtocolManager as _HPM
    HPMType = _HPM
    SysEventType = _SysHIDEvent
    SysEventEnum = _SysHIDEventType
else:  # runtime optional import
    try:  # Attempt to import system protocol manager
        from src.interfaces.hid_protocol import HIDEvent as SysEventType  # type: ignore
        from src.interfaces.hid_protocol import (
            HIDEventType as SysEventEnum,  # type: ignore
        )
        from src.interfaces.hid_protocol import (
            HIDProtocolManager as HPMType,  # type: ignore
        )
    except ImportError:  # pragma: no cover
        HPMType = None  # type: ignore
        SysEventType = None  # type: ignore
        SysEventEnum = None  # type: ignore


@dataclass
class HIDEvent:
    event_type: str
    timestamp: float
    data: dict[str, Any]


class HIDBackend(Protocol):
    def send(self, event: HIDEvent) -> bool: ...


class MockHIDBackend:
    def __init__(self) -> None:
        self.events: list[HIDEvent] = []

    def send(self, event: HIDEvent) -> bool:  # pragma: no cover simple
        self.events.append(event)
        return True


class MacHIDBackend:
    def __init__(self) -> None:
        # Real mac backend not available in this environment; placeholder
        self.manager = None

    def send(self, event: HIDEvent) -> bool:  # pragma: no cover platform
        _ = event  # unused placeholder
        return False  # intentionally not implemented


def select_backend() -> HIDBackend:
    kind = os.getenv("BCI_HID_BACKEND", "mock").lower()
    if kind == "mac":
        return MacHIDBackend()
    return MockHIDBackend()


def create_mouse_move(dx: float, dy: float) -> HIDEvent:
    return HIDEvent(
        event_type="mouse_move",
        timestamp=time(),
        data={"x": dx, "y": dy},
    )


def create_click(button: str = "left", pressed: bool = True) -> HIDEvent:
    return HIDEvent(
        event_type="mouse_click",
        timestamp=time(),
        data={"button": button, "pressed": pressed},
    )
