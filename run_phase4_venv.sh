#!/bin/bash
cd /home/kevin/Projects/apple-bci-hid-compression

echo "🔧 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing dependencies in virtual environment..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🧪 Running Phase 4 testing and analysis..."
python scripts/execute_phase4_with_analysis.py

echo "Phase 4 execution completed with exit code: $?"
