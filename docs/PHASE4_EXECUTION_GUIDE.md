# Phase 4 Validation Execution Guide

## ✅ Dependencies Fixed
The package name issue has been resolved:
- **Fixed**: `pywt>=1.5.0` → `PyWavelets>=1.5.0` in requirements.txt
- **Status**: Ready for installation

## 🚀 Phase 4 Execution Options

### Option 1: Direct Python Execution (Recommended)
```bash
cd /home/kevin/Projects/apple-bci-hid-compression
python3 scripts/run_phase4_tests.py
```

### Option 2: Manual Step-by-Step
```bash
# Install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Run Phase 4 tests
python3 -m tests.phase4_runner
```

### Option 3: Bash Script (if preferred)
```bash
chmod +x scripts/quick_phase4_validation.sh
./scripts/quick_phase4_validation.sh
```

## 📊 Expected Outputs

After execution, you should see:

1. **Dependency Installation**: All packages install successfully
2. **Phase 4 Test Results**: JSON files in `test_results/phase4/`
3. **Performance Metrics**: Latency, throughput, compression ratios
4. **Readiness Assessment**: Overall system readiness score

## 🔧 Hardware Validation Framework

Created comprehensive hardware validation setup:

### Supported Device Categories
- **EEG Devices**: OpenBCI Cyton, Emotiv EPOC X, g.tec g.USBamp
- **EMG Devices**: Delsys Trigno
- **fNIRS Devices**: NIRx NIRSport2

### Validation Tests
- Signal quality assessment
- End-to-end latency measurement
- Compression effectiveness
- Gesture recognition accuracy
- Real-time performance validation

### Hardware Validation Usage
```bash
# List supported devices
python3 scripts/hardware_validation.py --list-devices

# Run validation for specific device (when available)
python3 scripts/hardware_validation.py --device "OpenBCI Cyton" --test signal_quality
```

## 📋 Next Steps After Execution

1. **Review Results**: Check generated JSON reports for detailed metrics
2. **Analyze Performance**: Verify real-time capabilities (target: <50ms latency)
3. **Hardware Testing**: Use validation framework when BCI devices are available
4. **Move to Phase 5**: If results are satisfactory, proceed to deployment planning

## 🎯 Success Criteria

The system is ready for deployment if:
- ✅ All dependencies install without errors
- ✅ Phase 4 tests complete successfully
- ✅ End-to-end latency < 50ms
- ✅ Compression ratio > 2x
- ✅ Gesture recognition accuracy > 80%
- ✅ No critical security vulnerabilities

## 🔄 If Issues Occur

If any step fails:
1. Check the error output for specific issues
2. Verify Python 3.11+ is being used
3. Ensure all file paths are correct
4. Review the generated logs in `test_results/phase4/`
5. Consult the project plan for troubleshooting steps
