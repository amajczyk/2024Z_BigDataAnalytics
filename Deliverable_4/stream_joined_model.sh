#!/bin/bash

# Configuration
RETRY_DELAY=5  # seconds
LOG_DIR="error_logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
SCRIPT_NAME="stream_joined_model.py"  # The Python script for joined correlation
LOG_FILE="${LOG_DIR}/joined_correlation_${TIMESTAMP}.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Forward all output to both console and log file
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

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
    local script_log_file="${LOG_DIR}/${script_name%.*}_${TIMESTAMP}.log"
    local attempt=1
    
    ensure_log_dir
    
    while true; do
        log "Starting joined correlation script $script_name (Attempt $attempt)"
        log "Full command: python \"$script_name\""
        log "Using Python: $(which python)"
        log "Using conda env: $CONDA_DEFAULT_ENV"
        
        # Try to run the script with output capturing
        if python "$script_name" 2>&1 | tee -a "$script_log_file"; then
            log "Script $script_name completed successfully"
            return 0
        else
            local exit_code=$?
            log "Script $script_name failed (Exit Code: $exit_code)"
            log "Error log saved to: $script_log_file"
            
            # Log the last few lines of the error
            log "Last few lines of error:"
            tail -n 5 "$script_log_file"
            
            log "Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
            
            # Create new log file for next attempt
            attempt=$((attempt + 1))
            TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
            script_log_file="${LOG_DIR}/${script_name%.*}_${TIMESTAMP}.log"
        fi
    done
}

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

# Activate conda environment
log "Activating conda environment..."
conda activate cassandra_env_39
if [ $? -ne 0 ]; then
    log "Failed to activate conda environment"
    exit 1
fi

# Log environment information
log "Python version: $(python --version 2>&1)"
log "Conda environment: $CONDA_DEFAULT_ENV"
log "Current directory: $(pwd)"
log "Script to run: $SCRIPT_NAME"
log "Main log file: $LOG_FILE"

# Main execution
log "Starting error handling wrapper for $SCRIPT_NAME"
run_with_error_handling "$SCRIPT_NAME"
exit_code=$?

# Deactivate conda environment
log "Deactivating conda environment..."
conda deactivate

log "Script execution completed with exit code: $exit_code"

exit $exit_code