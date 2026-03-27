# AIARE Forecasting

## Quick Start

The easiest way to run the entire pipeline is to use one of the provided scripts:

### Option 1: Using Make (Recommended)
```bash
make all
```

This will automatically set up the conda environment and run the preprocessing pipeline.

Other make commands:
- `make setup` - Just create/update the conda environment
- `make run` - Run the preprocessing pipeline (assumes environment exists)
- `make clean` - Remove generated output files
- `make help` - Show all available commands

### Option 2: Using the Python Script
```bash
python run_pipeline.py
```

### Option 3: Using the Shell Script
```bash
bash run_pipeline.sh
```

## Manual Setup

If you prefer to manage the environment manually:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate course_analysis

# Run the preprocessing
python eda/preprocess.py
```

## Output Files

The preprocessing script generates:
- `data/cleaned_data/course_enrollment.csv` - Combined course enrollment data
- `data/cleaned_data/student_counts.csv` - Combined student count data
