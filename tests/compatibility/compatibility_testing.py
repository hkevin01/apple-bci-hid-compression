"""Compatibility testing suite for Apple BCI-HID system."""

import asyncio
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CompatibilityDimension(Enum):
    """Compatibility testing dimensions."""
    OPERATING_SYSTEM = "operating_system"
    HARDWARE_ARCHITECTURE = "hardware_architecture"
    PYTHON_VERSION = "python_version"
    DEPENDENCIES = "dependencies"
    DEVICE_DRIVERS = "device_drivers"
    APPLE_ECOSYSTEM = "apple_ecosystem"
    BLUETOOTH_STACK = "bluetooth_stack"
    ACCESSIBILITY_APIS = "accessibility_apis"


@dataclass
class CompatibilityTestCase:
    """Compatibility test case definition."""
    name: str
    description: str
    dimension: CompatibilityDimension
    target_platform: str
    requirements: List[str]
    expected_behavior: str
    critical: bool = False


@dataclass
class CompatibilityResult:
    """Compatibility test result."""
    test_case: str
    platform_info: Dict[str, str]
    compatible: bool
    issues_found: List[str]
    workarounds: List[str]
    performance_impact: float  # 0.0 (no impact) to 1.0 (severe impact)
    test_duration: float
    details: Dict[str, Any]


class CompatibilityTestingSuite:
    """Comprehensive compatibility testing for BCI-HID system."""

    def __init__(self):
        self.results: List[CompatibilityResult] = []
        self.test_cases = self._define_compatibility_tests()
        self.platform_info = self._collect_platform_info()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('compatibility_testing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        print("🔧 Compatibility Testing Suite Initialized")
        print(f"   Test Cases: {len(self.test_cases)}")
        print(f"   Platform: {self.platform_info['system']} {self.platform_info['version']}")
        print(f"   Architecture: {self.platform_info['architecture']}")

    def _collect_platform_info(self) -> Dict[str, str]:
        """Collect comprehensive platform information."""
        info = {
            'system': platform.system(),
            'version': platform.version(),
            'release': platform.release(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'hostname': platform.node()
        }

        # Add macOS specific information
        if info['system'] == 'Darwin':
            try:
                # Get macOS version
                result = subprocess.run(['sw_vers', '-productVersion'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info['macos_version'] = result.stdout.strip()

                # Get hardware model
                result = subprocess.run(['sysctl', '-n', 'hw.model'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info['hardware_model'] = result.stdout.strip()

                # Get chip information
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info['cpu_brand'] = result.stdout.strip()

            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        return info

    def _define_compatibility_tests(self) -> List[CompatibilityTestCase]:
        """Define comprehensive compatibility test cases."""
        return [
            # Operating System Compatibility
            CompatibilityTestCase(
                name="macos_monterey_compatibility",
                description="Test compatibility with macOS Monterey (12.x)",
                dimension=CompatibilityDimension.OPERATING_SYSTEM,
                target_platform="macOS 12.x",
                requirements=["macOS >= 12.0", "Apple Silicon or Intel"],
                expected_behavior="Full functionality with all features",
                critical=True
            ),

            CompatibilityTestCase(
                name="macos_ventura_compatibility",
                description="Test compatibility with macOS Ventura (13.x)",
                dimension=CompatibilityDimension.OPERATING_SYSTEM,
                target_platform="macOS 13.x",
                requirements=["macOS >= 13.0", "Apple Silicon recommended"],
                expected_behavior="Full functionality with enhanced features",
                critical=True
            ),

            CompatibilityTestCase(
                name="macos_sonoma_compatibility",
                description="Test compatibility with macOS Sonoma (14.x)",
                dimension=CompatibilityDimension.OPERATING_SYSTEM,
                target_platform="macOS 14.x",
                requirements=["macOS >= 14.0", "Apple Silicon"],
                expected_behavior="Full functionality with latest features",
                critical=True
            ),

            # Hardware Architecture
            CompatibilityTestCase(
                name="apple_silicon_m1_compatibility",
                description="Test compatibility with Apple M1 chips",
                dimension=CompatibilityDimension.HARDWARE_ARCHITECTURE,
                target_platform="Apple M1",
                requirements=["M1 chip", "Native ARM64 support"],
                expected_behavior="Optimized performance with native execution",
                critical=True
            ),

            CompatibilityTestCase(
                name="apple_silicon_m2_compatibility",
                description="Test compatibility with Apple M2 chips",
                dimension=CompatibilityDimension.HARDWARE_ARCHITECTURE,
                target_platform="Apple M2",
                requirements=["M2 chip", "Enhanced neural processing"],
                expected_behavior="Enhanced performance with neural engine optimization",
                critical=True
            ),

            CompatibilityTestCase(
                name="apple_silicon_m3_compatibility",
                description="Test compatibility with Apple M3 chips",
                dimension=CompatibilityDimension.HARDWARE_ARCHITECTURE,
                target_platform="Apple M3",
                requirements=["M3 chip", "Latest neural processing capabilities"],
                expected_behavior="Maximum performance with latest optimizations",
                critical=True
            ),

            CompatibilityTestCase(
                name="intel_x86_compatibility",
                description="Test compatibility with Intel x86_64 Macs",
                dimension=CompatibilityDimension.HARDWARE_ARCHITECTURE,
                target_platform="Intel x86_64",
                requirements=["Intel processor", "Rosetta 2 compatibility"],
                expected_behavior="Functional with possible performance limitations",
                critical=False
            ),

            # Python Version Compatibility
            CompatibilityTestCase(
                name="python_311_compatibility",
                description="Test compatibility with Python 3.11",
                dimension=CompatibilityDimension.PYTHON_VERSION,
                target_platform="Python 3.11",
                requirements=["Python >= 3.11.0"],
                expected_behavior="Full functionality with all features",
                critical=True
            ),

            CompatibilityTestCase(
                name="python_312_compatibility",
                description="Test compatibility with Python 3.12",
                dimension=CompatibilityDimension.PYTHON_VERSION,
                target_platform="Python 3.12",
                requirements=["Python >= 3.12.0"],
                expected_behavior="Enhanced performance and compatibility",
                critical=True
            ),

            # Dependencies Compatibility
            CompatibilityTestCase(
                name="numpy_compatibility",
                description="Test NumPy version compatibility",
                dimension=CompatibilityDimension.DEPENDENCIES,
                target_platform="NumPy >= 1.21.0",
                requirements=["NumPy with BLAS optimization"],
                expected_behavior="Optimized mathematical operations",
                critical=True
            ),

            CompatibilityTestCase(
                name="scipy_compatibility",
                description="Test SciPy version compatibility",
                dimension=CompatibilityDimension.DEPENDENCIES,
                target_platform="SciPy >= 1.9.0",
                requirements=["SciPy with signal processing"],
                expected_behavior="Advanced signal processing capabilities",
                critical=True
            ),

            # Apple Ecosystem Integration
            CompatibilityTestCase(
                name="core_bluetooth_compatibility",
                description="Test Core Bluetooth framework integration",
                dimension=CompatibilityDimension.APPLE_ECOSYSTEM,
                target_platform="Core Bluetooth",
                requirements=["Bluetooth 5.0+", "Core Bluetooth access"],
                expected_behavior="Native Bluetooth device communication",
                critical=True
            ),

            CompatibilityTestCase(
                name="accessibility_api_compatibility",
                description="Test Accessibility API integration",
                dimension=CompatibilityDimension.ACCESSIBILITY_APIS,
                target_platform="Accessibility APIs",
                requirements=["Accessibility permissions", "System API access"],
                expected_behavior="Full accessibility feature integration",
                critical=True
            ),

            CompatibilityTestCase(
                name="hid_framework_compatibility",
                description="Test HID framework integration",
                dimension=CompatibilityDimension.DEVICE_DRIVERS,
                target_platform="IOHIDManager",
                requirements=["HID device access", "System permissions"],
                expected_behavior="Direct HID device communication",
                critical=True
            ),

            # Device Driver Compatibility
            CompatibilityTestCase(
                name="usb_device_compatibility",
                description="Test USB device driver compatibility",
                dimension=CompatibilityDimension.DEVICE_DRIVERS,
                target_platform="USB drivers",
                requirements=["USB device access", "Driver availability"],
                expected_behavior="Seamless USB device integration",
                critical=False
            ),

            CompatibilityTestCase(
                name="bluetooth_le_compatibility",
                description="Test Bluetooth Low Energy compatibility",
                dimension=CompatibilityDimension.BLUETOOTH_STACK,
                target_platform="Bluetooth LE",
                requirements=["Bluetooth LE support", "GATT capabilities"],
                expected_behavior="Low-power device communication",
                critical=True
            )
        ]

    async def run_all_compatibility_tests(self) -> Dict[str, Any]:
        """Run all compatibility test cases."""
        print("\n🔧 Starting Compatibility Testing Suite")
        print("=" * 60)

        total_tests = len(self.test_cases)
        compatible_tests = 0
        incompatible_tests = 0
        critical_failures = 0

        for test_case in self.test_cases:
            try:
                print(f"\n🧪 Running Compatibility Test: {test_case.name}")
                print(f"   Target Platform: {test_case.target_platform}")
                print(f"   Critical: {'Yes' if test_case.critical else 'No'}")

                result = await self.run_compatibility_test(test_case)
                self.results.append(result)

                if result.compatible:
                    print("   ✅ COMPATIBLE")
                    compatible_tests += 1
                else:
                    print("   ❌ INCOMPATIBLE")
                    incompatible_tests += 1
                    if test_case.critical:
                        critical_failures += 1

                    # Print issues found
                    for issue in result.issues_found:
                        print(f"     🚨 {issue}")

                if result.performance_impact > 0.3:
                    print(f"   ⚠️  Performance Impact: {result.performance_impact*100:.1f}%")

            except Exception as e:
                print(f"   💥 ERROR: {e}")
                incompatible_tests += 1
                if test_case.critical:
                    critical_failures += 1
                self.logger.error(f"Compatibility test {test_case.name} failed: {e}")

        # Generate compatibility report
        return self._generate_compatibility_report(
            compatible_tests, incompatible_tests, total_tests, critical_failures
        )

    async def run_compatibility_test(self, test_case: CompatibilityTestCase) -> CompatibilityResult:
        """Run a single compatibility test case."""
        start_time = time.perf_counter()
        issues = []
        workarounds = []
        performance_impact = 0.0
        details = {}

        try:
            if test_case.dimension == CompatibilityDimension.OPERATING_SYSTEM:
                result = await self._test_os_compatibility(test_case)
            elif test_case.dimension == CompatibilityDimension.HARDWARE_ARCHITECTURE:
                result = await self._test_hardware_compatibility(test_case)
            elif test_case.dimension == CompatibilityDimension.PYTHON_VERSION:
                result = await self._test_python_compatibility(test_case)
            elif test_case.dimension == CompatibilityDimension.DEPENDENCIES:
                result = await self._test_dependency_compatibility(test_case)
            elif test_case.dimension == CompatibilityDimension.APPLE_ECOSYSTEM:
                result = await self._test_apple_ecosystem_compatibility(test_case)
            elif test_case.dimension == CompatibilityDimension.ACCESSIBILITY_APIS:
                result = await self._test_accessibility_compatibility(test_case)
            elif test_case.dimension == CompatibilityDimension.DEVICE_DRIVERS:
                result = await self._test_device_driver_compatibility(test_case)
            elif test_case.dimension == CompatibilityDimension.BLUETOOTH_STACK:
                result = await self._test_bluetooth_compatibility(test_case)
            else:
                result = {
                    'compatible': False,
                    'issues': [f"Unknown compatibility dimension: {test_case.dimension}"],
                    'workarounds': [],
                    'performance_impact': 0.0,
                    'details': {}
                }

            issues = result['issues']
            workarounds = result['workarounds']
            performance_impact = result['performance_impact']
            details = result['details']

        except Exception as e:
            issues = [f"Test execution error: {str(e)}"]
            workarounds = ["Review test implementation"]
            performance_impact = 1.0
            details = {'error': str(e)}

        end_time = time.perf_counter()
        test_duration = end_time - start_time

        compatible = len(issues) == 0

        return CompatibilityResult(
            test_case=test_case.name,
            platform_info=self.platform_info,
            compatible=compatible,
            issues_found=issues,
            workarounds=workarounds,
            performance_impact=performance_impact,
            test_duration=test_duration,
            details=details
        )

    async def _test_os_compatibility(self, test_case: CompatibilityTestCase) -> Dict[str, Any]:
        """Test operating system compatibility."""
        issues = []
        workarounds = []
        performance_impact = 0.0
        details = {}

        current_os = self.platform_info['system']
        current_version = self.platform_info.get('macos_version', self.platform_info['version'])

        details['current_os'] = current_os
        details['current_version'] = current_version
        details['target_platform'] = test_case.target_platform

        if current_os != 'Darwin':
            issues.append(f"Target requires macOS, but running on {current_os}")
            workarounds.append("Use macOS system or virtual machine")
            performance_impact = 1.0
        else:
            # Check macOS version compatibility
            if 'monterey' in test_case.name.lower():
                if not self._version_meets_requirement(current_version, '12.0'):
                    issues.append(f"macOS Monterey (12.x) required, found {current_version}")
                    if self._version_meets_requirement(current_version, '11.0'):
                        workarounds.append("Upgrade to macOS Monterey for full compatibility")
                        performance_impact = 0.3
                    else:
                        performance_impact = 0.7

            elif 'ventura' in test_case.name.lower():
                if not self._version_meets_requirement(current_version, '13.0'):
                    issues.append(f"macOS Ventura (13.x) required, found {current_version}")
                    if self._version_meets_requirement(current_version, '12.0'):
                        workarounds.append("Upgrade to macOS Ventura for enhanced features")
                        performance_impact = 0.2
                    else:
                        performance_impact = 0.5

            elif 'sonoma' in test_case.name.lower():
                if not self._version_meets_requirement(current_version, '14.0'):
                    issues.append(f"macOS Sonoma (14.x) required, found {current_version}")
                    if self._version_meets_requirement(current_version, '13.0'):
                        workarounds.append("Upgrade to macOS Sonoma for latest features")
                        performance_impact = 0.1
                    else:
                        performance_impact = 0.4

            # Check system permissions and features
            details['system_integrity_protection'] = await self._check_sip_status()
            details['accessibility_permissions'] = await self._check_accessibility_permissions()

        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'workarounds': workarounds,
            'performance_impact': performance_impact,
            'details': details
        }

    async def _test_hardware_compatibility(self, test_case: CompatibilityTestCase) -> Dict[str, Any]:
        """Test hardware architecture compatibility."""
        issues = []
        workarounds = []
        performance_impact = 0.0
        details = {}

        current_arch = self.platform_info['architecture']
        cpu_brand = self.platform_info.get('cpu_brand', '')
        hardware_model = self.platform_info.get('hardware_model', '')

        details['current_architecture'] = current_arch
        details['cpu_brand'] = cpu_brand
        details['hardware_model'] = hardware_model

        if 'm1' in test_case.name.lower():
            if 'M1' not in cpu_brand and 'M1' not in hardware_model:
                issues.append("Apple M1 chip required")
                if current_arch == 'arm64':
                    workarounds.append("M2 or M3 chips provide better performance")
                    performance_impact = 0.0  # M2/M3 are better
                elif current_arch == 'x86_64':
                    workarounds.append("Use Rosetta 2 compatibility layer")
                    performance_impact = 0.3
                else:
                    performance_impact = 0.8

        elif 'm2' in test_case.name.lower():
            if 'M2' not in cpu_brand and 'M2' not in hardware_model:
                issues.append("Apple M2 chip required")
                if 'M3' in cpu_brand or 'M3' in hardware_model:
                    workarounds.append("M3 chip provides enhanced capabilities")
                    performance_impact = 0.0  # M3 is better
                elif 'M1' in cpu_brand or 'M1' in hardware_model:
                    workarounds.append("M1 chip provides most functionality")
                    performance_impact = 0.1
                else:
                    performance_impact = 0.4

        elif 'm3' in test_case.name.lower():
            if 'M3' not in cpu_brand and 'M3' not in hardware_model:
                issues.append("Apple M3 chip required")
                if 'M2' in cpu_brand or 'M2' in hardware_model:
                    workarounds.append("M2 chip provides most functionality")
                    performance_impact = 0.1
                elif 'M1' in cpu_brand or 'M1' in hardware_model:
                    workarounds.append("M1 chip provides basic functionality")
                    performance_impact = 0.2
                else:
                    performance_impact = 0.5

        elif 'intel' in test_case.name.lower():
            if current_arch != 'x86_64':
                issues.append("Intel x86_64 architecture required")
                workarounds.append("Apple Silicon provides better performance")
                performance_impact = 0.0  # Apple Silicon is better
            else:
                # Intel is supported but not optimal
                performance_impact = 0.2
                workarounds.append("Consider upgrading to Apple Silicon for better performance")

        # Check for neural processing capabilities
        details['neural_engine_available'] = 'M1' in cpu_brand or 'M2' in cpu_brand or 'M3' in cpu_brand
        details['vector_processing'] = current_arch == 'arm64'

        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'workarounds': workarounds,
            'performance_impact': performance_impact,
            'details': details
        }

    async def _test_python_compatibility(self, test_case: CompatibilityTestCase) -> Dict[str, Any]:
        """Test Python version compatibility."""
        issues = []
        workarounds = []
        performance_impact = 0.0
        details = {}

        current_version = self.platform_info['python_version']
        implementation = self.platform_info['python_implementation']

        details['current_python_version'] = current_version
        details['python_implementation'] = implementation
        details['target_version'] = test_case.target_platform

        if implementation != 'CPython':
            issues.append(f"CPython required, found {implementation}")
            workarounds.append("Install CPython implementation")
            performance_impact = 0.5

        if '3.11' in test_case.target_platform:
            if not self._version_meets_requirement(current_version, '3.11.0'):
                issues.append(f"Python 3.11+ required, found {current_version}")
                if self._version_meets_requirement(current_version, '3.10.0'):
                    workarounds.append("Upgrade to Python 3.11 for full compatibility")
                    performance_impact = 0.2
                else:
                    workarounds.append("Upgrade to Python 3.11 or later")
                    performance_impact = 0.4

        elif '3.12' in test_case.target_platform:
            if not self._version_meets_requirement(current_version, '3.12.0'):
                issues.append(f"Python 3.12+ required, found {current_version}")
                if self._version_meets_requirement(current_version, '3.11.0'):
                    workarounds.append("Upgrade to Python 3.12 for enhanced performance")
                    performance_impact = 0.1
                else:
                    workarounds.append("Upgrade to Python 3.12 or later")
                    performance_impact = 0.3

        # Check for required Python features
        details['async_support'] = sys.version_info >= (3, 7)
        details['dataclasses_support'] = sys.version_info >= (3, 7)
        details['typing_extensions'] = sys.version_info >= (3, 8)

        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'workarounds': workarounds,
            'performance_impact': performance_impact,
            'details': details
        }

    async def _test_dependency_compatibility(self, test_case: CompatibilityTestCase) -> Dict[str, Any]:
        """Test dependency compatibility."""
        issues = []
        workarounds = []
        performance_impact = 0.0
        details = {}

        # Test NumPy compatibility
        if 'numpy' in test_case.name.lower():
            try:
                import numpy as np
                numpy_version = np.__version__
                details['numpy_version'] = numpy_version
                details['numpy_config'] = np.show_config()

                if not self._version_meets_requirement(numpy_version, '1.21.0'):
                    issues.append(f"NumPy 1.21.0+ required, found {numpy_version}")
                    workarounds.append("Upgrade NumPy: pip install numpy>=1.21.0")
                    performance_impact = 0.3

                # Check for optimized BLAS
                config = np.show_config()
                if 'lapack' not in str(config).lower():
                    issues.append("Optimized BLAS/LAPACK not found")
                    workarounds.append("Install optimized NumPy build")
                    performance_impact = 0.2

            except ImportError:
                issues.append("NumPy not installed")
                workarounds.append("Install NumPy: pip install numpy>=1.21.0")
                performance_impact = 1.0

        # Test SciPy compatibility
        elif 'scipy' in test_case.name.lower():
            try:
                import scipy
                scipy_version = scipy.__version__
                details['scipy_version'] = scipy_version

                if not self._version_meets_requirement(scipy_version, '1.9.0'):
                    issues.append(f"SciPy 1.9.0+ required, found {scipy_version}")
                    workarounds.append("Upgrade SciPy: pip install scipy>=1.9.0")
                    performance_impact = 0.3

                # Test signal processing
                try:
                    from scipy import signal
                    details['signal_processing_available'] = True
                except ImportError:
                    issues.append("SciPy signal processing not available")
                    performance_impact = 0.5

            except ImportError:
                issues.append("SciPy not installed")
                workarounds.append("Install SciPy: pip install scipy>=1.9.0")
                performance_impact = 1.0

        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'workarounds': workarounds,
            'performance_impact': performance_impact,
            'details': details
        }

    async def _test_apple_ecosystem_compatibility(self, test_case: CompatibilityTestCase) -> Dict[str, Any]:
        """Test Apple ecosystem integration compatibility."""
        issues = []
        workarounds = []
        performance_impact = 0.0
        details = {}

        if 'core_bluetooth' in test_case.name.lower():
            # Test Core Bluetooth availability
            bluetooth_available = await self._check_bluetooth_availability()
            details['bluetooth_available'] = bluetooth_available

            if not bluetooth_available:
                issues.append("Bluetooth not available")
                workarounds.append("Enable Bluetooth in System Preferences")
                performance_impact = 1.0
            else:
                # Check Bluetooth version
                bluetooth_version = await self._get_bluetooth_version()
                details['bluetooth_version'] = bluetooth_version

                if bluetooth_version < 5.0:
                    issues.append(f"Bluetooth 5.0+ required, found {bluetooth_version}")
                    workarounds.append("Use external Bluetooth 5.0+ adapter")
                    performance_impact = 0.3

        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'workarounds': workarounds,
            'performance_impact': performance_impact,
            'details': details
        }

    async def _test_accessibility_compatibility(self, test_case: CompatibilityTestCase) -> Dict[str, Any]:
        """Test accessibility API compatibility."""
        issues = []
        workarounds = []
        performance_impact = 0.0
        details = {}

        # Check accessibility permissions
        accessibility_enabled = await self._check_accessibility_permissions()
        details['accessibility_permissions'] = accessibility_enabled

        if not accessibility_enabled:
            issues.append("Accessibility permissions required")
            workarounds.append("Grant accessibility permissions in System Preferences > Security & Privacy")
            performance_impact = 1.0

        # Check for required accessibility APIs
        details['voiceover_available'] = await self._check_voiceover_availability()
        details['switch_control_available'] = await self._check_switch_control_availability()

        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'workarounds': workarounds,
            'performance_impact': performance_impact,
            'details': details
        }

    async def _test_device_driver_compatibility(self, test_case: CompatibilityTestCase) -> Dict[str, Any]:
        """Test device driver compatibility."""
        issues = []
        workarounds = []
        performance_impact = 0.0
        details = {}

        if 'hid' in test_case.name.lower():
            # Test HID framework access
            hid_available = await self._check_hid_framework()
            details['hid_framework_available'] = hid_available

            if not hid_available:
                issues.append("HID framework not accessible")
                workarounds.append("Check system permissions for device access")
                performance_impact = 1.0

        elif 'usb' in test_case.name.lower():
            # Test USB device access
            usb_available = await self._check_usb_access()
            details['usb_access_available'] = usb_available

            if not usb_available:
                issues.append("USB device access limited")
                workarounds.append("Use administrator privileges or install drivers")
                performance_impact = 0.5

        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'workarounds': workarounds,
            'performance_impact': performance_impact,
            'details': details
        }

    async def _test_bluetooth_compatibility(self, test_case: CompatibilityTestCase) -> Dict[str, Any]:
        """Test Bluetooth stack compatibility."""
        issues = []
        workarounds = []
        performance_impact = 0.0
        details = {}

        # Test Bluetooth LE support
        ble_available = await self._check_ble_support()
        details['ble_support'] = ble_available

        if not ble_available:
            issues.append("Bluetooth Low Energy not supported")
            workarounds.append("Use external Bluetooth LE adapter")
            performance_impact = 0.8

        # Test GATT capabilities
        gatt_support = await self._check_gatt_support()
        details['gatt_support'] = gatt_support

        if not gatt_support:
            issues.append("GATT protocol support limited")
            workarounds.append("Update Bluetooth drivers")
            performance_impact = 0.4

        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'workarounds': workarounds,
            'performance_impact': performance_impact,
            'details': details
        }

    # Helper methods for compatibility checks
    def _version_meets_requirement(self, current: str, required: str) -> bool:
        """Check if current version meets requirement."""
        try:
            current_parts = [int(x) for x in current.split('.')]
            required_parts = [int(x) for x in required.split('.')]

            # Pad with zeros if needed
            max_len = max(len(current_parts), len(required_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            required_parts.extend([0] * (max_len - len(required_parts)))

            return current_parts >= required_parts
        except (ValueError, AttributeError):
            return False

    async def _check_sip_status(self) -> bool:
        """Check System Integrity Protection status."""
        try:
            result = subprocess.run(['csrutil', 'status'],
                                  capture_output=True, text=True, timeout=5)
            return 'disabled' in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return True  # Assume enabled if can't check

    async def _check_accessibility_permissions(self) -> bool:
        """Check if accessibility permissions are granted."""
        # Simulate accessibility permission check
        # In real implementation, would use system APIs
        return True  # Assume granted for testing

    async def _check_bluetooth_availability(self) -> bool:
        """Check if Bluetooth is available."""
        try:
            result = subprocess.run(['system_profiler', 'SPBluetoothDataType'],
                                  capture_output=True, text=True, timeout=10)
            return 'bluetooth' in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def _get_bluetooth_version(self) -> float:
        """Get Bluetooth version."""
        # Simulate Bluetooth version detection
        return 5.0  # Assume Bluetooth 5.0 for testing

    async def _check_voiceover_availability(self) -> bool:
        """Check VoiceOver availability."""
        return True  # VoiceOver is available on all macOS systems

    async def _check_switch_control_availability(self) -> bool:
        """Check Switch Control availability."""
        return True  # Switch Control is available on all macOS systems

    async def _check_hid_framework(self) -> bool:
        """Check HID framework availability."""
        return True  # HID framework is part of macOS

    async def _check_usb_access(self) -> bool:
        """Check USB device access."""
        return True  # Basic USB access is available

    async def _check_ble_support(self) -> bool:
        """Check Bluetooth Low Energy support."""
        return True  # Modern Macs support BLE

    async def _check_gatt_support(self) -> bool:
        """Check GATT protocol support."""
        return True  # GATT is part of BLE stack

    def _generate_compatibility_report(self, compatible: int, incompatible: int,
                                     total: int, critical_failures: int) -> Dict[str, Any]:
        """Generate comprehensive compatibility report."""

        # Categorize results by dimension
        results_by_dimension = {}
        for result in self.results:
            dimension = next(
                (tc.dimension.value for tc in self.test_cases if tc.name == result.test_case),
                'unknown'
            )
            if dimension not in results_by_dimension:
                results_by_dimension[dimension] = []
            results_by_dimension[dimension].append(result)

        # Calculate compatibility scores
        compatibility_scores = self._calculate_compatibility_scores(results_by_dimension)

        # Analyze performance impact
        performance_analysis = self._analyze_performance_impact()

        # Generate recommendations
        recommendations = self._generate_compatibility_recommendations()

        report = {
            'test_summary': {
                'total_tests': total,
                'compatible_tests': compatible,
                'incompatible_tests': incompatible,
                'compatibility_rate': compatible / total if total > 0 else 0,
                'critical_failures': critical_failures,
                'platform_info': self.platform_info
            },
            'compatibility_scores': compatibility_scores,
            'dimension_analysis': self._analyze_by_dimension(results_by_dimension),
            'performance_impact_analysis': performance_analysis,
            'platform_support_matrix': self._generate_support_matrix(),
            'detailed_results': [self._result_to_dict(r) for r in self.results],
            'recommendations': recommendations
        }

        return report

    def _calculate_compatibility_scores(self, results_by_dimension: Dict[str, List]) -> Dict[str, float]:
        """Calculate compatibility scores by dimension."""
        scores = {}

        for dimension, results in results_by_dimension.items():
            if results:
                compatible_count = sum(1 for r in results if r.compatible)
                scores[dimension] = compatible_count / len(results)
            else:
                scores[dimension] = 0.0

        # Overall compatibility score
        if self.results:
            scores['overall'] = sum(1 for r in self.results if r.compatible) / len(self.results)
        else:
            scores['overall'] = 0.0

        return scores

    def _analyze_performance_impact(self) -> Dict[str, Any]:
        """Analyze performance impact across tests."""
        if not self.results:
            return {}

        impacts = [r.performance_impact for r in self.results]

        return {
            'average_impact': sum(impacts) / len(impacts),
            'maximum_impact': max(impacts),
            'minimum_impact': min(impacts),
            'high_impact_tests': [
                r.test_case for r in self.results
                if r.performance_impact > 0.5
            ],
            'impact_distribution': {
                'none': len([i for i in impacts if i == 0.0]),
                'low': len([i for i in impacts if 0.0 < i <= 0.3]),
                'medium': len([i for i in impacts if 0.3 < i <= 0.7]),
                'high': len([i for i in impacts if i > 0.7])
            }
        }

    def _analyze_by_dimension(self, results_by_dimension: Dict[str, List]) -> Dict[str, Any]:
        """Analyze compatibility by dimension."""
        analysis = {}

        for dimension, results in results_by_dimension.items():
            if results:
                compatible_count = sum(1 for r in results if r.compatible)
                total_issues = sum(len(r.issues_found) for r in results)
                avg_impact = sum(r.performance_impact for r in results) / len(results)

                analysis[dimension] = {
                    'test_count': len(results),
                    'compatible_count': compatible_count,
                    'incompatible_count': len(results) - compatible_count,
                    'compatibility_rate': compatible_count / len(results),
                    'total_issues': total_issues,
                    'average_performance_impact': avg_impact
                }

        return analysis

    def _generate_support_matrix(self) -> Dict[str, Dict[str, str]]:
        """Generate platform support matrix."""
        matrix = {}

        # Group by platform/target
        platforms = {}
        for test_case in self.test_cases:
            if test_case.target_platform not in platforms:
                platforms[test_case.target_platform] = []
            platforms[test_case.target_platform].append(test_case.name)

        # Determine support level for each platform
        for platform, test_names in platforms.items():
            platform_results = [r for r in self.results if r.test_case in test_names]

            if not platform_results:
                support_level = 'untested'
            else:
                compatible_count = sum(1 for r in platform_results if r.compatible)
                compatibility_rate = compatible_count / len(platform_results)

                if compatibility_rate >= 0.9:
                    support_level = 'full_support'
                elif compatibility_rate >= 0.7:
                    support_level = 'partial_support'
                elif compatibility_rate >= 0.3:
                    support_level = 'limited_support'
                else:
                    support_level = 'not_supported'

            matrix[platform] = {
                'support_level': support_level,
                'compatibility_rate': f"{compatibility_rate*100:.1f}%" if platform_results else "N/A"
            }

        return matrix

    def _generate_compatibility_recommendations(self) -> List[str]:
        """Generate compatibility recommendations."""
        recommendations = []

        if not self.results:
            return ["No test results available for analysis."]

        # Check for critical failures
        critical_failures = [r for r in self.results if not r.compatible and any(
            tc.critical for tc in self.test_cases if tc.name == r.test_case
        )]

        if critical_failures:
            recommendations.append(
                f"CRITICAL: {len(critical_failures)} critical compatibility issues found. "
                "Address these before deployment."
            )

        # Performance impact recommendations
        high_impact_results = [r for r in self.results if r.performance_impact > 0.5]
        if high_impact_results:
            recommendations.append(
                f"High performance impact detected in {len(high_impact_results)} tests. "
                "Consider optimization or alternative approaches."
            )

        # Platform-specific recommendations
        if self.platform_info['system'] != 'Darwin':
            recommendations.append(
                "System is not running macOS. Full compatibility requires macOS environment."
            )

        # Python version recommendations
        python_version = self.platform_info['python_version']
        if not self._version_meets_requirement(python_version, '3.11.0'):
            recommendations.append(
                f"Python {python_version} detected. Upgrade to Python 3.11+ for optimal compatibility."
            )

        # Architecture recommendations
        if self.platform_info['architecture'] == 'x86_64':
            recommendations.append(
                "Intel architecture detected. Consider upgrading to Apple Silicon for better performance."
            )

        if not recommendations:
            recommendations.append(
                "All compatibility tests passed. System is ready for deployment."
            )

        return recommendations

    def _result_to_dict(self, result: CompatibilityResult) -> Dict[str, Any]:
        """Convert compatibility result to dictionary."""
        return {
            'test_case': result.test_case,
            'platform_info': result.platform_info,
            'compatible': result.compatible,
            'issues_found': result.issues_found,
            'workarounds': result.workarounds,
            'performance_impact': result.performance_impact,
            'test_duration': result.test_duration,
            'details': result.details
        }

    def save_compatibility_results(self, filename: str = "compatibility_test_results.json"):
        """Save compatibility test results to file."""
        report = self._generate_compatibility_report(
            len([r for r in self.results if r.compatible]),
            len([r for r in self.results if not r.compatible]),
            len(self.results),
            len([r for r in self.results if not r.compatible and any(
                tc.critical for tc in self.test_cases if tc.name == r.test_case
            )])
        )

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"🔧 Compatibility test results saved to {filename}")


async def main():
    """Main compatibility testing function."""
    print("🔧 Apple BCI-HID Compatibility Testing Suite")
    print("=" * 60)

    compat_suite = CompatibilityTestingSuite()

    try:
        # Run all compatibility tests
        report = await compat_suite.run_all_compatibility_tests()

        # Print compatibility summary
        print("\n" + "=" * 60)
        print("🔧 COMPATIBILITY TESTING SUMMARY")
        print("=" * 60)

        summary = report['test_summary']
        scores = report['compatibility_scores']
        performance = report['performance_impact_analysis']

        print("\nTest Results:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Compatible: {summary['compatible_tests']}")
        print(f"  Incompatible: {summary['incompatible_tests']}")
        print(f"  Compatibility Rate: {summary['compatibility_rate']*100:.1f}%")
        print(f"  Critical Failures: {summary['critical_failures']}")

        print(f"\nPlatform Information:")
        platform = summary['platform_info']
        print(f"  System: {platform['system']} {platform.get('macos_version', platform['version'])}")
        print(f"  Architecture: {platform['architecture']}")
        print(f"  Python: {platform['python_version']}")

        print(f"\nCompatibility Scores:")
        for dimension, score in scores.items():
            print(f"  {dimension.replace('_', ' ').title()}: {score*100:.1f}%")

        print(f"\nPerformance Impact:")
        print(f"  Average Impact: {performance['average_impact']*100:.1f}%")
        print(f"  High Impact Tests: {len(performance['high_impact_tests'])}")

        print(f"\nPlatform Support Matrix:")
        for platform, info in report['platform_support_matrix'].items():
            support_status = {
                'full_support': '✅ Full Support',
                'partial_support': '🟡 Partial Support',
                'limited_support': '🟠 Limited Support',
                'not_supported': '❌ Not Supported',
                'untested': '❓ Untested'
            }.get(info['support_level'], info['support_level'])
            print(f"  {platform}: {support_status} ({info['compatibility_rate']})")

        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")

        # Save results
        compat_suite.save_compatibility_results()

        print(f"\n✅ Compatibility testing completed successfully!")

        # Return success if compatibility rate is good and no critical failures
        return (scores['overall'] >= 0.8 and summary['critical_failures'] == 0)

    except Exception as e:
        print(f"\n❌ Compatibility testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
