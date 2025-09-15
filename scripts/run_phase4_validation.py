#!/usr/bin/env python3
"""
Apple BCI-HID Compression Bridge - Phase 4 Validation Script

This script performs:
1. Dependency installation verification
2. Phase 4 comprehensive testing execution
3. Hardware validation setup and preparation
4. Results analysis and reporting
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
PHASE4_RUNNER = PROJECT_ROOT / "tests" / "phase4_runner.py"
RESULTS_DIR = PROJECT_ROOT / "test_results" / "phase4"

class Phase4ValidationRunner:
    """Comprehensive Phase 4 validation runner."""

    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()

        # Ensure results directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        print("🚀 Apple BCI-HID Compression Bridge - Phase 4 Validation")
        print("=" * 60)
        print(f"Project Root: {PROJECT_ROOT}")
        print(f"Results Directory: {RESULTS_DIR}")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    def step_1_install_dependencies(self) -> bool:
        """Step 1: Install and verify all dependencies."""
        print("📦 STEP 1: Installing Dependencies")
        print("-" * 40)

        try:
            # Check if requirements.txt exists
            if not REQUIREMENTS_FILE.exists():
                print(f"❌ Requirements file not found: {REQUIREMENTS_FILE}")
                return False

            print(f"✅ Found requirements file: {REQUIREMENTS_FILE}")

            # Run pip install
            print("⏳ Running pip install -r requirements.txt...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("✅ Dependencies installed successfully!")

                # Verify critical imports
                critical_packages = [
                    "numpy", "scipy", "PyWavelets", "sklearn",
                    "torch", "pandas", "matplotlib", "pytest"
                ]

                failed_imports = []
                for package in critical_packages:
                    try:
                        if package == "sklearn":
                            __import__("sklearn")
                        elif package == "PyWavelets":
                            __import__("pywt")
                        else:
                            __import__(package)
                        print(f"   ✅ {package}")
                    except ImportError as e:
                        print(f"   ❌ {package}: {e}")
                        failed_imports.append(package)

                if failed_imports:
                    print(f"❌ Failed to import: {', '.join(failed_imports)}")
                    return False

                self.results["dependencies"] = {
                    "status": "success",
                    "installed_packages": critical_packages,
                    "install_output": result.stdout
                }
                return True

            else:
                print("❌ Dependency installation failed!")
                print(f"Error: {result.stderr}")
                self.results["dependencies"] = {
                    "status": "failed",
                    "error": result.stderr
                }
                return False

        except subprocess.TimeoutExpired:
            print("❌ Dependency installation timed out!")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during dependency installation: {e}")
            return False

    def step_2_run_phase4_testing(self) -> bool:
        """Step 2: Execute comprehensive Phase 4 testing."""
        print("\n🧪 STEP 2: Phase 4 Comprehensive Testing")
        print("-" * 40)

        try:
            # Check if Phase 4 runner exists
            if not PHASE4_RUNNER.exists():
                print(f"❌ Phase 4 runner not found: {PHASE4_RUNNER}")
                return False

            print(f"✅ Found Phase 4 runner: {PHASE4_RUNNER}")

            # Change to project root for execution
            original_cwd = os.getcwd()
            os.chdir(PROJECT_ROOT)

            try:
                # Run Phase 4 testing
                print("⏳ Executing Phase 4 comprehensive testing suite...")
                start_time = time.time()

                result = subprocess.run([
                    sys.executable, "-m", "tests.phase4_runner"
                ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout

                execution_time = time.time() - start_time

                if result.returncode == 0:
                    print(f"✅ Phase 4 testing completed successfully! ({execution_time:.1f}s)")

                    # Look for generated results
                    result_files = list(RESULTS_DIR.glob("phase4_comprehensive_results_*.json"))
                    if result_files:
                        latest_result = max(result_files, key=lambda f: f.stat().st_mtime)
                        print(f"✅ Results saved to: {latest_result}")

                        # Load and display summary
                        try:
                            with open(latest_result) as f:
                                results_data = json.load(f)

                            summary = results_data.get('summary', {})
                            print("\n📊 PHASE 4 RESULTS SUMMARY:")
                            print(f"   Overall Score: {summary.get('overall_score', 0):.1%}")
                            print(f"   Suites Run: {summary.get('suites_run', 0)}")
                            print(f"   Suites Passed: {summary.get('suites_passed', 0)}")
                            print(f"   Duration: {summary.get('total_duration_minutes', 0):.1f} minutes")

                        except Exception as e:
                            print(f"⚠️  Could not parse results file: {e}")

                    self.results["phase4_testing"] = {
                        "status": "success",
                        "execution_time": execution_time,
                        "output": result.stdout,
                        "result_files": [str(f) for f in result_files]
                    }
                    return True

                else:
                    print("❌ Phase 4 testing failed!")
                    print(f"Return code: {result.returncode}")
                    print(f"Error output: {result.stderr}")

                    self.results["phase4_testing"] = {
                        "status": "failed",
                        "return_code": result.returncode,
                        "error": result.stderr,
                        "output": result.stdout
                    }
                    return False

            finally:
                os.chdir(original_cwd)

        except subprocess.TimeoutExpired:
            print("❌ Phase 4 testing timed out after 30 minutes!")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during Phase 4 testing: {e}")
            return False

    def step_3_hardware_validation_setup(self) -> bool:
        """Step 3: Set up hardware validation framework."""
        print("\n🔧 STEP 3: Hardware Validation Setup")
        print("-" * 40)

        try:
            # Create hardware validation configuration
            hardware_validation_config = {
                "validation_framework": {
                    "version": "1.0.0",
                    "created": datetime.now().isoformat(),
                    "description": "Hardware validation configuration for BCI devices"
                },
                "supported_devices": {
                    "eeg_devices": [
                        {
                            "name": "OpenBCI Cyton",
                            "type": "EEG",
                            "channels": 8,
                            "sample_rate": 250,
                            "interface": "USB/Bluetooth",
                            "validation_tests": ["signal_quality", "latency", "compression_ratio"]
                        },
                        {
                            "name": "Emotiv EPOC X",
                            "type": "EEG",
                            "channels": 14,
                            "sample_rate": 256,
                            "interface": "Bluetooth",
                            "validation_tests": ["signal_quality", "latency", "gesture_recognition"]
                        },
                        {
                            "name": "g.tec g.USBamp",
                            "type": "EEG/EMG",
                            "channels": 16,
                            "sample_rate": 1200,
                            "interface": "USB",
                            "validation_tests": ["signal_quality", "latency", "compression_ratio", "real_time_performance"]
                        }
                    ],
                    "emg_devices": [
                        {
                            "name": "Delsys Trigno",
                            "type": "EMG",
                            "channels": 16,
                            "sample_rate": 2000,
                            "interface": "Wireless",
                            "validation_tests": ["signal_quality", "latency", "gesture_recognition"]
                        }
                    ],
                    "fnirs_devices": [
                        {
                            "name": "NIRx NIRSport2",
                            "type": "fNIRS",
                            "channels": 64,
                            "sample_rate": 10.4,
                            "interface": "USB",
                            "validation_tests": ["signal_quality", "compression_effectiveness"]
                        }
                    ]
                },
                "validation_tests": {
                    "signal_quality": {
                        "description": "Validate signal integrity after compression/decompression",
                        "metrics": ["SNR", "correlation", "frequency_preservation"],
                        "thresholds": {"min_snr_db": 15, "min_correlation": 0.85}
                    },
                    "latency": {
                        "description": "Measure end-to-end processing latency",
                        "metrics": ["compression_latency", "decompression_latency", "total_latency"],
                        "thresholds": {"max_total_latency_ms": 50}
                    },
                    "compression_ratio": {
                        "description": "Evaluate compression effectiveness",
                        "metrics": ["compression_ratio", "file_size_reduction"],
                        "thresholds": {"min_compression_ratio": 2.0}
                    },
                    "gesture_recognition": {
                        "description": "Test gesture classification accuracy",
                        "metrics": ["accuracy", "precision", "recall", "f1_score"],
                        "thresholds": {"min_accuracy": 0.8}
                    },
                    "real_time_performance": {
                        "description": "Validate real-time processing capabilities",
                        "metrics": ["throughput_samples_per_sec", "buffer_utilization", "dropped_samples"],
                        "thresholds": {"min_throughput": 1000, "max_dropped_samples_percent": 1.0}
                    }
                },
                "test_protocols": {
                    "motor_imagery": {
                        "description": "Motor imagery gesture classification test",
                        "duration_seconds": 300,
                        "gestures": ["rest", "left_hand", "right_hand", "both_hands"],
                        "trials_per_gesture": 10
                    },
                    "continuous_monitoring": {
                        "description": "Long-term system stability test",
                        "duration_seconds": 3600,
                        "data_collection_continuous": True,
                        "performance_monitoring": True
                    },
                    "stress_test": {
                        "description": "High-load system stress test",
                        "duration_seconds": 600,
                        "simultaneous_channels": "max_supported",
                        "high_sample_rate": True
                    }
                }
            }

            # Save hardware validation configuration
            config_file = PROJECT_ROOT / "config" / "hardware_validation.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(config_file, 'w') as f:
                json.dump(hardware_validation_config, f, indent=2)

            print(f"✅ Hardware validation configuration created: {config_file}")

            # Create hardware validation script template
            validation_script = '''#!/usr/bin/env python3
"""
Hardware Validation Script for Apple BCI-HID Compression Bridge

Usage:
    python hardware_validation.py --device <device_name> --test <test_type>
    python hardware_validation.py --list-devices
    python hardware_validation.py --run-all-tests
"""

import argparse
import json
import sys
from pathlib import Path

def load_validation_config():
    """Load hardware validation configuration."""
    config_file = Path(__file__).parent.parent / "config" / "hardware_validation.json"
    with open(config_file, 'r') as f:
        return json.load(f)

def list_supported_devices():
    """List all supported hardware devices."""
    config = load_validation_config()
    print("🔧 SUPPORTED HARDWARE DEVICES:")
    print("=" * 40)

    for device_type, devices in config["supported_devices"].items():
        print(f"\\n{device_type.upper().replace('_', ' ')}:")
        for device in devices:
            print(f"  • {device['name']}")
            print(f"    Type: {device['type']}")
            print(f"    Channels: {device['channels']}")
            print(f"    Sample Rate: {device['sample_rate']} Hz")
            print(f"    Interface: {device['interface']}")

def run_device_validation(device_name: str, test_type: str):
    """Run validation tests for specific device."""
    config = load_validation_config()

    # Find device configuration
    device_config = None
    for device_type, devices in config["supported_devices"].items():
        for device in devices:
            if device["name"].lower() == device_name.lower():
                device_config = device
                break
        if device_config:
            break

    if not device_config:
        print(f"❌ Device '{device_name}' not found in supported devices.")
        return False

    print(f"🔧 VALIDATING: {device_config['name']}")
    print("-" * 40)
    print(f"Device Type: {device_config['type']}")
    print(f"Channels: {device_config['channels']}")
    print(f"Sample Rate: {device_config['sample_rate']} Hz")
    print()

    # TODO: Implement actual hardware validation logic
    print("⚠️  Hardware validation implementation pending.")
    print("   This will be implemented when physical hardware is available.")
    print("   Current focus: Software simulation and synthetic data testing.")

    return True

def main():
    """Main hardware validation entry point."""
    parser = argparse.ArgumentParser(description="Hardware Validation for BCI-HID System")
    parser.add_argument("--device", help="Device name to validate")
    parser.add_argument("--test", help="Test type to run")
    parser.add_argument("--list-devices", action="store_true", help="List supported devices")
    parser.add_argument("--run-all-tests", action="store_true", help="Run all validation tests")

    args = parser.parse_args()

    if args.list_devices:
        list_supported_devices()
    elif args.device:
        test_type = args.test or "signal_quality"
        run_device_validation(args.device, test_type)
    elif args.run_all_tests:
        print("🚀 Running all hardware validation tests...")
        print("⚠️  Implementation pending - requires physical hardware.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
'''

            # Save hardware validation script
            validation_script_file = PROJECT_ROOT / "scripts" / "hardware_validation.py"
            validation_script_file.parent.mkdir(parents=True, exist_ok=True)

            with open(validation_script_file, 'w') as f:
                f.write(validation_script)

            # Make script executable
            import stat
            validation_script_file.chmod(validation_script_file.stat().st_mode | stat.S_IEXEC)

            print(f"✅ Hardware validation script created: {validation_script_file}")

            print("\n📋 HARDWARE VALIDATION SUMMARY:")
            print(f"   Configuration: {config_file}")
            print(f"   Validation Script: {validation_script_file}")
            print(f"   Supported Device Types: {len(hardware_validation_config['supported_devices'])}")
            print(f"   Total Supported Devices: {sum(len(devices) for devices in hardware_validation_config['supported_devices'].values())}")
            print(f"   Validation Tests: {len(hardware_validation_config['validation_tests'])}")

            self.results["hardware_validation_setup"] = {
                "status": "success",
                "config_file": str(config_file),
                "validation_script": str(validation_script_file),
                "supported_devices": hardware_validation_config["supported_devices"]
            }

            return True

        except Exception as e:
            print(f"❌ Error during hardware validation setup: {e}")
            self.results["hardware_validation_setup"] = {
                "status": "failed",
                "error": str(e)
            }
            return False

    def generate_final_report(self):
        """Generate final validation report."""
        print("\n📊 FINAL VALIDATION REPORT")
        print("=" * 60)

        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() / 60

        # Count successful steps
        successful_steps = sum(1 for result in self.results.values()
                             if isinstance(result, dict) and result.get("status") == "success")
        total_steps = len(self.results)

        print(f"⏱️  Total Duration: {total_duration:.1f} minutes")
        print(f"✅ Successful Steps: {successful_steps}/{total_steps}")
        print(f"📈 Success Rate: {successful_steps/total_steps*100:.1f}%")

        print("\n📋 STEP SUMMARY:")
        for step_name, result in self.results.items():
            if isinstance(result, dict):
                status = result.get("status", "unknown")
                icon = "✅" if status == "success" else "❌"
                print(f"   {icon} {step_name.replace('_', ' ').title()}: {status}")

        # Save detailed results
        detailed_results = {
            "validation_run": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_minutes": total_duration,
                "success_rate": successful_steps/total_steps,
                "successful_steps": successful_steps,
                "total_steps": total_steps
            },
            "step_results": self.results
        }

        results_file = RESULTS_DIR / f"phase4_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {results_file}")

        # Next steps recommendations
        print("\n🎯 NEXT STEPS:")
        if successful_steps == total_steps:
            print("   ✅ All validation steps completed successfully!")
            print("   📊 Review Phase 4 test results for detailed performance metrics")
            print("   🔧 Ready for hardware validation when devices are available")
            print("   🚀 Consider moving to Phase 5: Deployment & Maintenance")
        else:
            print("   ⚠️  Some validation steps failed - review errors above")
            print("   🔧 Fix any dependency or configuration issues")
            print("   🔄 Re-run validation after resolving issues")

        return results_file

def main():
    """Main validation runner."""
    runner = Phase4ValidationRunner()

    try:
        # Step 1: Install dependencies
        step1_success = runner.step_1_install_dependencies()

        # Step 2: Run Phase 4 testing (only if Step 1 succeeded)
        step2_success = False
        if step1_success:
            step2_success = runner.step_2_run_phase4_testing()
        else:
            print("\n⚠️  Skipping Phase 4 testing due to dependency installation failure")

        # Step 3: Hardware validation setup (independent of previous steps)
        step3_success = runner.step_3_hardware_validation_setup()

        # Generate final report
        results_file = runner.generate_final_report()

        # Exit with appropriate code
        if step1_success and step2_success and step3_success:
            print("\n🎉 Phase 4 validation completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Phase 4 validation completed with issues - check report for details")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        runner.generate_final_report()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during validation: {e}")
        runner.generate_final_report()
        sys.exit(1)

if __name__ == "__main__":
    main()
