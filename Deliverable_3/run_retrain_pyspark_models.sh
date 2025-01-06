#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the Conda environment and Python script
CONDA_ENV="kafka_stream_preprocessing"
PYTHON_SCRIPT="retrain_pyspark_models.py"

# Activate the Conda environment
echo "Activating Conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Check if the Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

# Run the Python script
echo "Running Python script: $PYTHON_SCRIPT"
python "$PYTHON_SCRIPT"

# Wait until the script execution is completed
echo "Waiting for the script to finish..."
wait

echo "Script execution completed."

# Deactivate the Conda environment
echo "Deactivating Conda environment: $CONDA_ENV"
conda deactivate
