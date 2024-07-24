#!/bin/bash
# Path to your Conda installation
CONDA_PATH=/Users/fradm98/opt/anaconda3/envs/
# The name of the Conda environment you want to activate
ENV_NAME=qs_qh
# The path to your Python script
PYTHON_SCRIPT_PATH=src/notebooks/Z2/cronjob.py

# Activate the Conda environment
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Run the Python script
python $PYTHON_SCRIPT_PATH

# Deactivate the Conda environment
conda deactivate
