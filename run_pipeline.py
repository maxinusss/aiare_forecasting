#!/usr/bin/env python3
"""
Pipeline runner for AIARE Forecasting project.
Sets up the conda environment and runs the preprocessing pipeline.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description="", check=True):
    """Run a shell command and return success status."""
    if description:
        print(f"\n{description}")
        print("=" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=False, text=True)
        if check and result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            sys.exit(1)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        sys.exit(1)


def check_conda():
    """Check if conda is available."""
    result = subprocess.run(
        "conda --version",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def env_exists(env_name):
    """Check if a conda environment exists."""
    result = subprocess.run(
        f"conda env list | grep -q '^{env_name} '",
        shell=True,
        capture_output=True
    )
    return result.returncode == 0


def main():
    print("=" * 50)
    print("AIARE Forecasting Pipeline")
    print("=" * 50)
    
    # Set project directory
    project_dir = Path(__file__).parent.absolute()
    os.chdir(project_dir)
    
    env_name = "course_analysis"
    
    # Check conda availability
    if not check_conda():
        print("❌ Error: conda is not installed or not in PATH")
        print("Please install Miniconda or Anaconda and try again")
        sys.exit(1)
    
    # Create environment if needed
    if not env_exists(env_name):
        print(f"\n📦 Creating conda environment from environment.yml...")
        run_command(
            "conda env create -f environment.yml",
            check=True
        )
        print("✅ Environment created successfully")
    else:
        print(f"✅ Environment '{env_name}' already exists")
    
    # Run preprocessing
    print("\n🔄 Running preprocessing pipeline...")
    print("=" * 50)
    
    run_command(
        f"conda run -n {env_name} python eda/preprocess.py",
        check=True
    )

    # Run figure generation after preprocessing
    run_command(
        f"conda run -n {env_name} python eda/create_figs.py",
        check=True
    )

    print("\n" + "=" * 50)
    print("✅ Pipeline completed successfully!")
    print("Output files:")
    print("  - data/cleaned_data/course_enrollment.csv")
    print("  - data/cleaned_data/student_counts.csv")
    print("  - data/cleaned_data/master_data.csv")
    print("  - plot files (e.g., price_trends.png, enrollment_trends.png)")
    print("=" * 50)


if __name__ == "__main__":
    main()
