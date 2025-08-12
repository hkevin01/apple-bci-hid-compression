# Apple BCI-HID Compression Bridge

A comprehensive brain-computer interface (BCI) bridge that processes neural signals through compression algorithms and translates them into HID events for Apple devices. This project implements a complete pipeline from neural signal processing to gesture recognition and input mapping.

## ✅ Implemented Features

### Core Compression (Phase 2 - Complete)
- ✅ **WaveletCompressor**: Real wavelet-based lossy compression using PyWavelets
- ✅ **Fallback algorithms**: Custom hierarchical averaging when PyWavelets unavailable
- ✅ **Configurable compression**: Adjustable wavelet levels and top-K detail coefficient selection
- ✅ **Multi-channel support**: Per-channel compression with metadata headers

### Neural Translation (Phase 3 - Complete)
- ✅ **IntentTranslator**: Integrates gesture recognition with input mapping
- ✅ **HybridGestureRecognizer**: ML + rule-based gesture detection
- ✅ **FixedInputMapper**: Gesture-to-action mapping system
- ✅ **Async streaming support**: Real-time signal processing pipeline

### HID Interface (Phase 3 - Complete)
- ✅ **Multi-backend support**: Mock and Mac HID backends with environment switching
- ✅ **HID event creation**: Mouse movement, clicks, keyboard input abstractions
- ✅ **Protocol abstraction**: Clean interface for different HID implementations
- ✅ **Device communication**: Ready for IOKit and system-level integration

### End-to-End Pipelines
- ✅ **Synchronous pipeline**: `process_neural_input()` function for single-frame processing
- ✅ **Async streaming**: `async_process_stream()` for real-time frame consumption
- ✅ **Complete signal flow**: Neural samples → compression → gesture recognition → HID events

### Testing & Quality (Phase 4 - In Progress)
- ✅ **Comprehensive test suite**: Unit tests for all core components
- ✅ **Edge case testing**: Empty inputs, high amplitude, invalid dimensions, NaN/inf handling
- ✅ **Async pipeline tests**: Stream processing validation
- ✅ **Performance benchmarking**: Automated performance testing framework
- ✅ **Security testing**: Encryption, authentication, injection resistance
- ✅ **Compatibility testing**: Cross-platform and version matrix support

## Requirements

- Python 3.11+
- NumPy/SciPy for signal processing
- PyWavelets (optional, with fallback)
- pytest for testing
- macOS for full HID integration (mock backend available for other platforms)

## Installation

### Python Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/apple-bci-hid-compression.git
cd apple-bci-hid-compression

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **Core**: numpy, scipy
- **Optional**: pywt (PyWavelets for advanced compression)
- **Testing**: pytest, pytest-cov, hypothesis
- **Development**: black, mypy, ruff
- **Coverage**: coverage

## Quick Start

### Basic Neural Signal Processing

```python
import numpy as np
from src.core.compression import WaveletCompressor
from src.core.pipeline.end_to_end import process_neural_input

# Generate sample neural data (64 channels, random signals)
neural_samples = np.random.randn(64).astype(np.float32)

# Process through complete pipeline: compression → gesture recognition → HID
hid_event = process_neural_input(neural_samples)

if hid_event:
    print(f"Generated HID event: {hid_event.event_type}")
```

### Async Streaming Pipeline

```python
import asyncio
import numpy as np
from src.core.pipeline.async_end_to_end import async_process_stream, ListFrameStream

async def stream_example():
    # Create sample frames
    frames = [np.random.randn(64).astype(np.float32) for _ in range(10)]
    stream = ListFrameStream(frames)
    
    # Process stream and collect HID events
    events = []
    async for event in async_process_stream(stream):
        events.append(event)
        print(f"Received: {event.event_type}")
    
    return events

# Run the async example
events = asyncio.run(stream_example())
```

### Manual Component Usage

```python
from src.core.compression import WaveletCompressor
from src.neural_translation.intent_translator import IntentTranslator
from src.hid_interface import select_backend

# Initialize components
compressor = WaveletCompressor(level=3, top_k_ratio=0.2)
translator = IntentTranslator()
backend = select_backend()  # Uses BCI_HID_BACKEND env var

# Process data step by step
neural_data = np.random.randn(128).astype(np.float32)

# 1. Compress and decompress
compressed = compressor.compress(neural_data)
decompressed = compressor.decompress(compressed)

# 2. Translate to gesture/action
import asyncio
result = asyncio.run(translator.translate(neural_data))

# 3. Send HID event if generated
if result.hid_event:
    success = backend.send(result.hid_event)
```

## Architecture

The project implements a modular Python-based architecture with the following structure:

```
apple-bci-hid-compression/
├── src/
│   ├── core/
│   │   ├── compression.py           # WaveletCompressor + base algorithms
│   │   └── pipeline/
│   │       ├── end_to_end.py        # Synchronous processing pipeline
│   │       ├── async_end_to_end.py  # Async streaming pipeline
│   │       └── data_pipeline.py     # Base data streaming utilities
│   ├── neural_translation/
│   │   └── intent_translator.py     # Neural signal → gesture → action
│   ├── hid_interface/
│   │   ├── python_hid.py           # HID backend abstraction
│   │   └── __init__.py             # HID interface exports
│   ├── recognition/
│   │   └── gesture_recognition.py  # Hybrid ML + rule-based recognition
│   ├── mapping/
│   │   └── input_mapping.py        # Gesture → input action mapping
│   └── interfaces/
│       └── hid_protocol.py         # Low-level HID protocol definitions
├── tests/
│   ├── unit/                       # Unit tests for all components
│   ├── integration/                # Integration testing
│   ├── performance/                # Performance benchmarks
│   ├── security/                   # Security validation
│   └── phase4_runner.py           # Comprehensive test orchestrator
└── docs/
    ├── api_overview.md             # API documentation
    └── project_plan.md             # Implementation roadmap
```

### Signal Flow

```
Neural Samples (np.ndarray)
    ↓
[WaveletCompressor] → Compressed bytes → Decompressed samples
    ↓
[HybridGestureRecognizer] → Gesture events
    ↓
[FixedInputMapper] → Input actions
    ↓
[HID Backend] → HID events (mouse, keyboard, etc.)
```

### Key Components

- **WaveletCompressor**: Lossy compression using PyWavelets or fallback algorithms
- **IntentTranslator**: Async gesture recognition and action mapping
- **HID Interface**: Multi-backend support (Mock, Mac) with environment switching  
- **Pipeline**: Both sync (`process_neural_input`) and async (`async_process_stream`) processing

## Development & Testing

### Running Tests

```bash
# Run all unit tests
python -m pytest tests/unit/

# Run with coverage
python -m pytest tests/unit/ --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/unit/test_wavelet_compressor.py  # Compression tests
python -m pytest tests/unit/test_intent_translator.py   # Translation tests  
python -m pytest tests/unit/test_async_pipeline.py      # Async pipeline tests
python -m pytest tests/unit/test_edge_cases.py         # Edge case validation
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking  
mypy src/

# Linting
ruff check src/ tests/

# Run all quality checks
black --check src/ && mypy src/ && ruff check src/
```

### Environment Configuration

Set the HID backend type via environment variable:

```bash
# Use mock backend (default, works on all platforms)
export BCI_HID_BACKEND=mock

# Use Mac backend (macOS only, requires system permissions)
export BCI_HID_BACKEND=mac
```

### Performance Testing

```bash
# Run performance benchmarks
python tests/phase4_runner.py

# Individual performance tests  
python -m pytest tests/performance/ -v
```

### Project Status

Based on the project plan phases:

- ✅ **Phase 1**: Foundation & Infrastructure (Complete)
- ✅ **Phase 2**: Core Compression Implementation (Complete) 
- ✅ **Phase 3**: HID Interface Implementation (Complete)
- 🔄 **Phase 4**: Testing & Optimization (In Progress - 80% complete)
- ⏳ **Phase 5**: Deployment & Maintenance (Planned)

Current implementation includes all core functionality with comprehensive testing framework in place.

## Contributing

Please read [CONTRIBUTING.md](.github/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Brain-Computer Compression Toolkit contributors
- Apple's Accessibility and HID teams
- The BCI research community

## Research & Publications

This project is part of ongoing research in BCI accessibility technology. For academic citations and research collaboration, please contact the maintainers.

## Privacy & Security

- All neural data processing occurs on-device
- End-to-end encryption for data transmission
- Compliance with Apple's privacy guidelines
- GDPR-compliant data handling

## Support

- 📚 [Documentation](docs/)
- 💬 [Discussions](https://github.com/yourusername/apple-bci-hid-compression/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/apple-bci-hid-compression/issues)
- 📧 [Contact](mailto:your.email@example.com)
