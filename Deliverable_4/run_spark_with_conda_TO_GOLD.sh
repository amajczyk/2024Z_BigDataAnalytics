#!/bin/bash

# Load the Miniconda initialization script
source /home/hadoop/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment (replace `my_env` with your environment name)
conda activate cassandra_env_39

# Run the Spark job
/home/hadoop/miniconda3/bin/spark-submit --master local[4] /home/hadoop/Documents/2024Z_BigDataAnalytics/Deliverable_4/load_to_gold_layer.py >> /usr/local/spark/logs/spark_silver_to_gold.log 2>&1

# Deactivate the environment (optional)
conda deactivate
