"""Device communication layer implementations."""

import asyncio
import struct
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union


class ConnectionType(Enum):
    """Types of device connections."""
    BLUETOOTH_LE = "bluetooth_le"
    USB = "usb"
    WIFI = "wifi"
    SERIAL = "serial"


class DeviceState(Enum):
    """Device connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    PAIRED = "paired"
    ERROR = "error"


@dataclass
class DeviceInfo:
    """Device information structure."""
    device_id: str
    name: str
    connection_type: ConnectionType
    vendor_id: Optional[int] = None
    product_id: Optional[int] = None
    serial_number: Optional[str] = None
    firmware_version: Optional[str] = None
    capabilities: List[str] = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


@dataclass
class ConnectionStatus:
    """Connection status information."""
    state: DeviceState
    signal_strength: Optional[float] = None
    latency: Optional[float] = None
    bandwidth: Optional[float] = None
    error_rate: Optional[float] = None


class DeviceCommunicator(Protocol):
    """Protocol for device communication."""

    async def connect(self, device_info: DeviceInfo) -> bool:
        """Connect to a device."""
        ...

    async def disconnect(self, device_id: str) -> bool:
        """Disconnect from a device."""
        ...

    async def send_data(self, device_id: str, data: bytes) -> bool:
        """Send data to a device."""
        ...

    async def receive_data(self, device_id: str) -> Optional[bytes]:
        """Receive data from a device."""
        ...


class BluetoothLECommunicator:
    """Bluetooth LE protocol implementation."""

    def __init__(self):
        self.connected_devices: Dict[str, DeviceInfo] = {}
        self.connection_status: Dict[str, ConnectionStatus] = {}
        self.data_callbacks: Dict[str, Callable] = {}
        self.scan_results: List[DeviceInfo] = []

    async def scan_devices(self, timeout: float = 10.0) -> List[DeviceInfo]:
        """Scan for available Bluetooth LE devices."""
        print(f"Scanning for BLE devices for {timeout} seconds...")

        # Simulate device discovery
        await asyncio.sleep(2.0)  # Simulate scan time

        # Mock discovered devices
        mock_devices = [
            DeviceInfo(
                device_id="ble_bci_001",
                name="BCI Neural Interface",
                connection_type=ConnectionType.BLUETOOTH_LE,
                vendor_id=0x1234,
                product_id=0x0001,
                capabilities=["neural_data", "gesture_recognition"]
            ),
            DeviceInfo(
                device_id="ble_eeg_002",
                name="EEG Headset Pro",
                connection_type=ConnectionType.BLUETOOTH_LE,
                vendor_id=0x5678,
                product_id=0x0002,
                capabilities=["eeg_data", "real_time_stream"]
            )
        ]

        self.scan_results = mock_devices
        print(f"Found {len(mock_devices)} BLE devices")
        return mock_devices

    async def connect(self, device_info: DeviceInfo) -> bool:
        """Connect to a Bluetooth LE device."""
        device_id = device_info.device_id

        print(f"Connecting to BLE device: {device_info.name}")

        # Update status to connecting
        self.connection_status[device_id] = ConnectionStatus(
            state=DeviceState.CONNECTING
        )

        try:
            # Simulate connection process
            await asyncio.sleep(1.0)  # Connection time

            # Check if device is available (simulate)
            if device_info in self.scan_results or device_id.startswith("ble_"):
                self.connected_devices[device_id] = device_info
                self.connection_status[device_id] = ConnectionStatus(
                    state=DeviceState.CONNECTED,
                    signal_strength=0.8,
                    latency=15.0,  # ms
                    bandwidth=1000.0  # bps
                )

                # Start data receiving loop
                asyncio.create_task(self._receive_loop(device_id))

                print(f"Successfully connected to {device_info.name}")
                return True
            else:
                raise ConnectionError("Device not available")

        except Exception as e:
            print(f"Failed to connect to {device_info.name}: {e}")
            self.connection_status[device_id] = ConnectionStatus(
                state=DeviceState.ERROR
            )
            return False

    async def disconnect(self, device_id: str) -> bool:
        """Disconnect from a Bluetooth LE device."""
        if device_id not in self.connected_devices:
            return False

        try:
            print(f"Disconnecting from BLE device: {device_id}")

            # Update status
            self.connection_status[device_id] = ConnectionStatus(
                state=DeviceState.DISCONNECTED
            )

            # Remove from connected devices
            del self.connected_devices[device_id]

            print(f"Disconnected from {device_id}")
            return True

        except Exception as e:
            print(f"Error disconnecting from {device_id}: {e}")
            return False

    async def send_data(self, device_id: str, data: bytes) -> bool:
        """Send data to a Bluetooth LE device."""
        if device_id not in self.connected_devices:
            return False

        try:
            # Simulate BLE data transmission
            print(f"BLE: Sending {len(data)} bytes to {device_id}")

            # Add BLE packet overhead simulation
            await asyncio.sleep(0.001)  # 1ms transmission delay

            # Update bandwidth metrics
            status = self.connection_status[device_id]
            if status.bandwidth:
                # Simple bandwidth calculation
                transmission_time = len(data) / status.bandwidth
                await asyncio.sleep(transmission_time)

            return True

        except Exception as e:
            print(f"BLE send error: {e}")
            return False

    async def receive_data(self, device_id: str) -> Optional[bytes]:
        """Receive data from a Bluetooth LE device."""
        if device_id not in self.connected_devices:
            return None

        try:
            # Simulate receiving data
            await asyncio.sleep(0.01)  # Small delay

            # Generate mock neural data
            mock_data = self._generate_mock_neural_data()
            return mock_data

        except Exception as e:
            print(f"BLE receive error: {e}")
            return None

    def _generate_mock_neural_data(self) -> bytes:
        """Generate mock neural data for testing."""
        # Simple packet structure: [header:4][channels:1][samples:N*4]
        header = struct.pack('<I', 0xDEADBEEF)  # Magic number
        channels = struct.pack('<B', 8)  # 8 channels

        # Generate 32 samples per channel (4 bytes each)
        samples = []
        for _ in range(8 * 32):
            # Random-ish neural data
            sample = int((time.time() * 1000) % 65536)
            samples.append(struct.pack('<I', sample))

        return header + channels + b''.join(samples)

    async def _receive_loop(self, device_id: str):
        """Background loop for receiving data."""
        while device_id in self.connected_devices:
            try:
                data = await self.receive_data(device_id)
                if data and device_id in self.data_callbacks:
                    callback = self.data_callbacks[device_id]
                    await callback(device_id, data)

                await asyncio.sleep(0.01)  # 100Hz reception rate

            except Exception as e:
                print(f"Receive loop error for {device_id}: {e}")
                break

    def set_data_callback(self, device_id: str, callback: Callable):
        """Set callback for received data."""
        self.data_callbacks[device_id] = callback

    def get_connection_status(self, device_id: str) -> Optional[ConnectionStatus]:
        """Get connection status for a device."""
        return self.connection_status.get(device_id)


class USBCommunicator:
    """USB protocol implementation."""

    def __init__(self):
        self.connected_devices: Dict[str, DeviceInfo] = {}
        self.connection_status: Dict[str, ConnectionStatus] = {}
        self.usb_endpoints: Dict[str, Dict] = {}

    async def enumerate_devices(self) -> List[DeviceInfo]:
        """Enumerate available USB devices."""
        print("Enumerating USB devices...")

        # Mock USB device enumeration
        mock_devices = [
            DeviceInfo(
                device_id="usb_bci_hid_001",
                name="USB BCI HID Device",
                connection_type=ConnectionType.USB,
                vendor_id=0x1234,
                product_id=0x5678,
                serial_number="SN001234",
                capabilities=["hid_reports", "bulk_transfer"]
            ),
            DeviceInfo(
                device_id="usb_neural_interface_002",
                name="Neural Interface USB",
                connection_type=ConnectionType.USB,
                vendor_id=0xABCD,
                product_id=0xEF01,
                serial_number="SN005678",
                capabilities=["interrupt_transfer", "iso_transfer"]
            )
        ]

        print(f"Found {len(mock_devices)} USB devices")
        return mock_devices

    async def connect(self, device_info: DeviceInfo) -> bool:
        """Connect to a USB device."""
        device_id = device_info.device_id

        print(f"Connecting to USB device: {device_info.name}")

        try:
            # Simulate USB device claiming
            await asyncio.sleep(0.5)  # USB enumeration time

            # Setup endpoints
            self.usb_endpoints[device_id] = {
                'in_endpoint': 0x81,    # Bulk IN
                'out_endpoint': 0x02,   # Bulk OUT
                'int_endpoint': 0x83    # Interrupt IN
            }

            self.connected_devices[device_id] = device_info
            self.connection_status[device_id] = ConnectionStatus(
                state=DeviceState.CONNECTED,
                latency=2.0,  # USB is faster than BLE
                bandwidth=12000000.0  # 12 Mbps for USB Full Speed
            )

            print(f"Successfully connected to USB device: {device_info.name}")
            return True

        except Exception as e:
            print(f"Failed to connect to USB device: {e}")
            self.connection_status[device_id] = ConnectionStatus(
                state=DeviceState.ERROR
            )
            return False

    async def disconnect(self, device_id: str) -> bool:
        """Disconnect from a USB device."""
        if device_id not in self.connected_devices:
            return False

        try:
            print(f"Disconnecting from USB device: {device_id}")

            # Release USB device
            if device_id in self.usb_endpoints:
                del self.usb_endpoints[device_id]

            del self.connected_devices[device_id]
            self.connection_status[device_id] = ConnectionStatus(
                state=DeviceState.DISCONNECTED
            )

            return True

        except Exception as e:
            print(f"USB disconnect error: {e}")
            return False

    async def send_data(self, device_id: str, data: bytes) -> bool:
        """Send data to USB device."""
        if device_id not in self.connected_devices:
            return False

        try:
            endpoint = self.usb_endpoints[device_id]['out_endpoint']
            print(f"USB: Sending {len(data)} bytes to endpoint {endpoint:02x}")

            # Simulate USB bulk transfer
            status = self.connection_status[device_id]
            if status.bandwidth:
                transfer_time = len(data) / status.bandwidth
                await asyncio.sleep(transfer_time)

            return True

        except Exception as e:
            print(f"USB send error: {e}")
            return False

    async def receive_data(self, device_id: str) -> Optional[bytes]:
        """Receive data from USB device."""
        if device_id not in self.connected_devices:
            return None

        try:
            endpoint = self.usb_endpoints[device_id]['in_endpoint']

            # Simulate USB bulk transfer
            await asyncio.sleep(0.001)  # USB transfer delay

            # Generate mock HID report
            mock_report = self._generate_hid_report()
            return mock_report

        except Exception as e:
            print(f"USB receive error: {e}")
            return None

    def _generate_hid_report(self) -> bytes:
        """Generate mock HID report."""
        # HID report structure: [report_id:1][data:N]
        report_id = 0x01

        # Mock neural data as HID report
        timestamp = int(time.time() * 1000) & 0xFFFFFFFF
        x_data = int((time.time() * 100) % 2048)
        y_data = int((time.time() * 150) % 2048)

        report_data = struct.pack('<BIII',
                                report_id,
                                timestamp,
                                x_data,
                                y_data)

        return report_data

    async def send_control_request(self, device_id: str,
                                 request_type: int, request: int,
                                 value: int, index: int,
                                 data: bytes = b'') -> Optional[bytes]:
        """Send USB control request."""
        if device_id not in self.connected_devices:
            return None

        try:
            print(f"USB Control: type={request_type:02x} req={request:02x} " +
                  f"val={value:04x} idx={index:04x}")

            # Simulate control transfer
            await asyncio.sleep(0.005)  # Control transfer delay

            # Mock response
            if request == 0x06:  # GET_DESCRIPTOR
                return b'\x12\x01\x00\x02\x00\x00\x00\x40'  # Mock descriptor

            return b''

        except Exception as e:
            print(f"USB control request error: {e}")
            return None


class MultiProtocolCommunicator:
    """Multi-protocol support implementation."""

    def __init__(self):
        self.communicators = {
            ConnectionType.BLUETOOTH_LE: BluetoothLECommunicator(),
            ConnectionType.USB: USBCommunicator()
        }
        self.device_registry: Dict[str, DeviceInfo] = {}
        self.connection_map: Dict[str, ConnectionType] = {}

    async def discover_all_devices(self) -> Dict[ConnectionType, List[DeviceInfo]]:
        """Discover devices across all protocols."""
        all_devices = {}

        # Scan Bluetooth LE
        ble_comm = self.communicators[ConnectionType.BLUETOOTH_LE]
        ble_devices = await ble_comm.scan_devices()
        all_devices[ConnectionType.BLUETOOTH_LE] = ble_devices

        # Enumerate USB
        usb_comm = self.communicators[ConnectionType.USB]
        usb_devices = await usb_comm.enumerate_devices()
        all_devices[ConnectionType.USB] = usb_devices

        # Update device registry
        for device_list in all_devices.values():
            for device in device_list:
                self.device_registry[device.device_id] = device

        return all_devices

    async def connect(self, device_info: DeviceInfo) -> bool:
        """Connect to device using appropriate protocol."""
        connection_type = device_info.connection_type

        if connection_type not in self.communicators:
            print(f"Unsupported connection type: {connection_type}")
            return False

        communicator = self.communicators[connection_type]
        success = await communicator.connect(device_info)

        if success:
            self.connection_map[device_info.device_id] = connection_type

        return success

    async def disconnect(self, device_id: str) -> bool:
        """Disconnect from device."""
        if device_id not in self.connection_map:
            return False

        connection_type = self.connection_map[device_id]
        communicator = self.communicators[connection_type]

        success = await communicator.disconnect(device_id)

        if success:
            del self.connection_map[device_id]

        return success

    async def send_data(self, device_id: str, data: bytes) -> bool:
        """Send data to device using its protocol."""
        if device_id not in self.connection_map:
            return False

        connection_type = self.connection_map[device_id]
        communicator = self.communicators[connection_type]

        return await communicator.send_data(device_id, data)

    async def receive_data(self, device_id: str) -> Optional[bytes]:
        """Receive data from device using its protocol."""
        if device_id not in self.connection_map:
            return None

        connection_type = self.connection_map[device_id]
        communicator = self.communicators[connection_type]

        return await communicator.receive_data(device_id)

    def get_connected_devices(self) -> List[DeviceInfo]:
        """Get list of connected devices."""
        connected = []
        for device_id in self.connection_map:
            if device_id in self.device_registry:
                connected.append(self.device_registry[device_id])
        return connected

    def get_connection_status(self, device_id: str) -> Optional[ConnectionStatus]:
        """Get connection status for device."""
        if device_id not in self.connection_map:
            return None

        connection_type = self.connection_map[device_id]
        communicator = self.communicators[connection_type]

        if hasattr(communicator, 'get_connection_status'):
            return communicator.get_connection_status(device_id)

        return None

    async def broadcast_data(self, data: bytes,
                           filter_type: Optional[ConnectionType] = None) -> int:
        """Broadcast data to all connected devices."""
        sent_count = 0

        for device_id, connection_type in self.connection_map.items():
            if filter_type and connection_type != filter_type:
                continue

            success = await self.send_data(device_id, data)
            if success:
                sent_count += 1

        return sent_count

    def get_protocol_statistics(self) -> Dict[str, Any]:
        """Get statistics for all protocols."""
        stats = {
            'total_devices': len(self.device_registry),
            'connected_devices': len(self.connection_map),
            'by_protocol': {}
        }

        # Count by protocol
        for connection_type in self.connection_map.values():
            protocol_name = connection_type.value
            stats['by_protocol'][protocol_name] = (
                stats['by_protocol'].get(protocol_name, 0) + 1
            )

        return stats
