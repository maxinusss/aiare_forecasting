.PHONY: help setup run clean

help:
	@echo "AIARE Forecasting Pipeline"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup   - Create the conda environment"
	@echo "  make run     - Run the full preprocessing pipeline"
	@echo "  make all     - Setup environment and run pipeline (recommended)"
	@echo "  make clean   - Remove output files from data/cleaned_data/"
	@echo "  make env-remove - Remove the conda environment"

all: setup run

setup:
	@echo "📦 Setting up conda environment..."
	@conda env create -f environment.yml --force-reinstall 2>/dev/null || conda env update -f environment.yml

run:
	@echo "🔄 Running preprocessing pipeline..."
	@conda run -n course_analysis python eda/preprocess.py
	@echo "✅ Pipeline completed!"
	@echo "Output files:"
	@echo "  - data/cleaned_data/course_enrollment.csv"
	@echo "  - data/cleaned_data/student_counts.csv"

clean:
	@echo "🗑️  Removing output files..."
	@rm -f data/cleaned_data/course_enrollment.csv
	@rm -f data/cleaned_data/student_counts.csv
	@echo "✅ Cleaned up output files"

env-remove:
	@echo "⚠️  Removing conda environment 'course_analysis'..."
	@conda env remove -n course_analysis
	@echo "✅ Environment removed"
