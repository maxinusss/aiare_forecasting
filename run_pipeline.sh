#!/bin/bash

# Pipeline runner script
# Sets up conda environment and runs the preprocessing pipeline

set -e  # Exit on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "AIARE Forecasting Pipeline"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda and try again"
    exit 1
fi

ENV_NAME="course_analysis"

# Check if environment exists
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "📦 Creating conda environment from environment.yml..."
    conda env create -f environment.yml
    echo "✅ Environment created successfully"
else
    echo "✅ Environment '$ENV_NAME' already exists"
fi

# Run preprocessing script
echo ""
echo "🔄 Running preprocessing pipeline..."
echo "=========================================="

conda run -n $ENV_NAME python eda/preprocess.py

echo ""
echo "=========================================="
echo "✅ Pipeline completed successfully!"
echo "Output files:"
echo "  - data/cleaned_data/course_enrollment.csv"
echo "  - data/cleaned_data/student_counts.csv"
echo "=========================================="
