#!/bin/bash

# Configuration
RETRY_DELAY=5  # seconds
LOG_DIR="error_logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Function to log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to ensure log directory exists
ensure_log_dir() {
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        log "Created log directory: $LOG_DIR"
    fi
}

# Function to run python script with infinite retry
run_with_error_handling() {
    local script_name=$1
    local script_args=$2
    local retrain_historical=$3  # New argument for retrain_historical
    local log_file="${LOG_DIR}/${script_name%.*}_${TIMESTAMP}.log"
    local attempt=1
    
    ensure_log_dir
    
    while true; do
        log "Starting $script_name with args: $script_args, retrain_historical: $retrain_historical (Attempt $attempt)"
        log "Full command: python \"$script_name\" $script_args $retrain_historical"
        log "Using Python: $(which python)"
        log "Using conda env: $CONDA_DEFAULT_ENV"
        
        # Try to run the script with output capturing
        if python "$script_name" $script_args $retrain_historical 2>&1 | tee -a "$log_file"; then
            log "Script $script_name completed successfully"
            return 0
        else
            local exit_code=$?
            log "Script $script_name failed (Exit Code: $exit_code)"
            log "Error log saved to: $log_file"
            
            # Log the last few lines of the error
            log "Last few lines of error:"
            tail -n 5 "$log_file"
            
            log "Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
            
            # Create new log file for next attempt
            attempt=$((attempt + 1))
            TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
            log_file="${LOG_DIR}/${script_name%.*}_${TIMESTAMP}.log"
        fi
    done
}

# Check if script name is provided
if [ $# -lt 1 ]; then
    log "Error: No script name provided"
    echo "Usage: $0 <script_name.py> <symbol> [retrain_historical]"
    exit 1
fi

# Extract script name and arguments
SCRIPT_NAME=$1
SYMBOL=$2
RETRAIN_HISTORICAL=${3:-false}  # Default to false if not provided

# Validate retrain_historical argument
if [[ "$RETRAIN_HISTORICAL" != "true" && "$RETRAIN_HISTORICAL" != "false" ]]; then
    log "Error: retrain_historical must be 'true' or 'false'"
    exit 1
fi

# Initialize conda
log "Initializing conda..."
CONDA_PATH="$HOME/anaconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_PATH" ]; then
    CONDA_PATH="$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if [ ! -f "$CONDA_PATH" ]; then
    log "Error: Could not find conda.sh"
    exit 1
fi

source "$CONDA_PATH"
if [ $? -ne 0 ]; then
    log "Failed to initialize conda"
    exit 1
fi

# Activate conda environment (change 'base' to your specific environment if needed)
log "Activating conda environment..."
conda activate base
if [ $? -ne 0 ]; then
    log "Failed to activate conda environment"
    exit 1
fi

# Log environment information
log "Python version: $(python --version 2>&1)"
log "Conda environment: $CONDA_DEFAULT_ENV"
log "Current directory: $(pwd)"
log "Script to run: $SCRIPT_NAME"
log "Symbol: $SYMBOL"
log "Retrain historical: $RETRAIN_HISTORICAL"

# Main execution
log "Starting error handling wrapper for $SCRIPT_NAME"
run_with_error_handling "$SCRIPT_NAME" "$SYMBOL" "$RETRAIN_HISTORICAL"
exit_code=$?

# Deactivate conda environment
log "Deactivating conda environment..."
conda deactivate

exit $exit_code