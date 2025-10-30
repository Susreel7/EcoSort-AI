#!/bin/bash

# TrashFormer Flask App Launcher
# Smart Waste Classification System

echo "🗑️  TrashFormer - Smart Waste Classifier"
echo "========================================"
echo "🌐 Flask Web Application"
echo "🎯 7-Class AI Waste Classification"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Set environment variables for clean output
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=-1

echo ""
echo "🚀 Starting Flask server..."
echo "📍 Local URL: http://127.0.0.1:5000"
echo "⏹️  Press Ctrl+C to stop"
echo "----------------------------------------"

# Run the Flask app
python app.py

