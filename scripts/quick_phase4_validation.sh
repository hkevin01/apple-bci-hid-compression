#!/usr/bin/env bash
"""
Quick Phase 4 Validation Launcher
This script runs the comprehensive Phase 4 validation process.
"""

cd "$(dirname "$0")/.." || exit 1

echo "🚀 Apple BCI-HID Compression Bridge - Phase 4 Validation"
echo "==========================================================="
echo "Starting comprehensive validation process..."
echo ""

# Make sure we're in the right directory
echo "Project Directory: $(pwd)"
echo "Python Version: $(python3 --version)"
echo ""

# Step 1: Install dependencies and run Phase 4 validation
echo "📦 Installing dependencies and running Phase 4 tests..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
    echo ""

    # Run Phase 4 tests
    echo "🧪 Running Phase 4 comprehensive testing..."
    python3 -m tests.phase4_runner

    if [ $? -eq 0 ]; then
        echo "✅ Phase 4 testing completed successfully!"
    else
        echo "⚠️  Phase 4 testing had issues - check output above"
    fi
else
    echo "❌ Dependency installation failed!"
    exit 1
fi

echo ""
echo "📊 Phase 4 validation process completed!"
echo "Check test_results/phase4/ for detailed results."
