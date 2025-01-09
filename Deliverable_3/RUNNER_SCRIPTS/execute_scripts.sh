#!/bin/bash

# ================================
# Configuration Variables
# ================================

# Name of the Conda environment to activate
CONDA_ENV="stream_sources"  # <-- Replace with your Conda environment name

# Directory paths
CURRENT_DIR="$(pwd)"
LOG_DIR="$CURRENT_DIR/error_logs"

# Python scripts
SCRIPT1="$CURRENT_DIR/yfinance_stream.py"
SCRIPT2="$CURRENT_DIR/xtb_stream.py"

# Path to Conda initialization script
# Update this path if Conda is installed elsewhere
CONDA_INIT_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"

# ================================
# Function Definitions
# ================================

# Function to activate Conda environment
activate_conda() {
    if [ -f "$CONDA_INIT_SCRIPT" ]; then
        source "$CONDA_INIT_SCRIPT"
        conda activate "$CONDA_ENV"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to activate Conda environment '$CONDA_ENV'."
            exit 1
        fi
    else
        echo "Error: Conda initialization script not found at '$CONDA_INIT_SCRIPT'."
        exit 1
    fi
}

# Function to run a Python script inside a screen session with error handling and real-time logging
run_script_in_screen() {
    local session_name=$1
    local script_path=$2

    # Create a subdirectory for the session logs
    local session_log_dir="$LOG_DIR/$session_name"
    mkdir -p "$session_log_dir"

    # Function to start the screen session
    start_screen_session() {
        # Generate a timestamp for the log file
        local timestamp=$(date +"%Y%m%d_%H%M%S")

        # Define the log file path
        local log_file="$session_log_dir/${session_name}_$timestamp.log"

        # Command to run inside the screen session
        local cmd="
            echo '--- Script started at $(date +"%Y-%m-%d %H:%M:%S") ---' | tee -a \"$log_file\"
            while true; do
                python \"$script_path\" 2>&1 | tee -a \"$log_file\"
                EXIT_CODE=\$?
                if [ \$EXIT_CODE -ne 0 ]; then
                    echo '--- Script encountered an error and will restart at $(date +"%Y-%m-%d %H:%M:%S") ---' | tee -a \"$log_file\"
                    sleep 5  # Wait before restarting
                else
                    echo '--- Script exited successfully at $(date +"%Y-%m-%d %H:%M:%S") ---' | tee -a \"$log_file\"
                    break  # Exit the loop if script finishes successfully
                fi
            done
        "

        # Start the screen session in detached mode
        screen -dmS "$session_name" bash -c "$cmd"

        if [ $? -eq 0 ]; then
            echo "Started '$script_path' in screen session '$session_name'."
        else
            echo "Error: Failed to start screen session '$session_name'."
        fi
    }

    # Start the initial screen session
    start_screen_session

    # Monitor the screen session and restart on error
    while true; do
        sleep 10  # Polling interval (adjust as needed)

        # Check if the screen session is still running
        if ! screen -list | grep -q "\.${session_name}"; then
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] Screen session '$session_name' terminated. Restarting the script..."
            start_screen_session
        fi
    done
}

# ================================
# Main Script Execution
# ================================

# Ensure the error_log directory exists
mkdir -p "$LOG_DIR"

# Activate the Conda environment
activate_conda

# Run the first Python script in a screen session
run_script_in_screen "script1_screen" "$SCRIPT1" &

# Run the second Python script in a screen session
run_script_in_screen "script2_screen" "$SCRIPT2" &

# Wait for both background monitoring processes
wait
