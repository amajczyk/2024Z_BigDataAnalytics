#!/bin/bash

# ================================
# Enhanced Bash Script to Start NIFI, Hadoop, Kafka, and Screen Sessions
# ================================

# Exit immediately if a command exits with a non-zero status
set -e

# Function to log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a process is running using jps
is_process_running() {
    local process_name=$1
    if command -v jps >/dev/null 2>&1; then
        jps_output=$(jps)
        echo "$jps_output" | grep -q "$process_name"
        return $?
    else
        log "jps command not found. Please ensure JDK is installed and jps is in your PATH."
        return 1
    fi
}

# Function to start NIFI
start_nifi() {
    log "Checking if NiFi is already running..."
    if is_process_running "NiFi" || is_process_running "RunNiFi"; then
        log "NiFi is already running. Skipping start command."
    else
        log "Starting NiFi..."
        /opt/nifi/bin/nifi.sh start
        if [ $? -eq 0 ]; then
            log "NiFi started successfully."
        else
            log "Failed to start NiFi."
            exit 1
        fi
    fi
}

# Function to start Hadoop
start_hadoop() {
    log "Checking if Hadoop is already running..."
    # Define key Hadoop processes to check
    local hadoop_processes=("NameNode" "ResourceManager")
    local all_running=true

    for process in "${hadoop_processes[@]}"; do
        if ! is_process_running "$process"; then
            all_running=false
            break
        fi
    done

    if $all_running; then
        log "Hadoop services are already running. Skipping start-all.sh."
    else
        log "Starting Hadoop..."
        start-all.sh
        if [ $? -eq 0 ]; then
            log "Hadoop started successfully."
        else
            log "Failed to start Hadoop."
            exit 1
        fi
    fi
}

# Function to check if Kafka is running using jps
is_kafka_running() {
    if command -v jps >/dev/null 2>&1; then
        jps_output=$(jps)
        echo "$jps_output" | grep -q "Kafka"
        return $?
    else
        log "jps command not found. Please ensure JDK is installed and jps is in your PATH."
        return 1
    fi
}

# Function to wait for a specific port to be open
wait_for_port() {
    local host=$1
    local port=$2
    local timeout=$3
    local interval=2
    local elapsed=0

    log "Waiting for $host:$port to be available..."

    while ! nc -z "$host" "$port"; do
        sleep "$interval"
        elapsed=$((elapsed + interval))
        if [ "$elapsed" -ge "$timeout" ]; then
            log "Timeout waiting for $host:$port"
            return 1
        fi
    done

    log "$host:$port is now available."
    return 0
}

# Function to find and kill processes using a specific port
kill_process_on_port() {
    local port=$1
    log "Checking for processes using port $port..."

    # Using lsof to find processes. If lsof is not available, fallback to netstat or ss
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -t -i:"$port")
    elif command -v netstat >/dev/null 2>&1; then
        pids=$(netstat -tuln | grep ":$port " | awk '{print $7}' | cut -d'/' -f1)
    elif command -v ss >/dev/null 2>&1; then
        pids=$(ss -tuln | grep ":$port " | awk '{print $6}' | cut -d',' -f1 | cut -d'=' -f2)
    else
        log "Neither lsof, netstat, nor ss is available to check ports. Cannot proceed."
        return 1
    fi

    if [ -n "$pids" ]; then
        log "Processes using port $port: $pids"
        for pid in $pids; do
            log "Killing process with PID $pid using port $port..."
            kill -9 "$pid" && log "Process $pid killed successfully."
        done
        return 0
    else
        log "No processes are using port $port."
        return 1
    fi
}

# Function to start a screen session
start_screen_session() {
    local session_name=$1
    local directory=$2
    local script=$3

    # Check if the screen session already exists
    if screen -list | grep -q "\.${session_name}"; then
        log "Screen session '$session_name' already exists. Skipping..."
    else
        # Command to run inside the screen session
        CMD="cd '$directory' && ./\"$script\""

        # Start a new detached screen session
        screen -dmS "$session_name" bash -c "$CMD"

        if [ $? -eq 0 ]; then
            log "Started screen session '$session_name' in directory '$directory' running script '$script'."
        else
            log "Failed to start screen session '$session_name'."
        fi
    fi
}

# Function to start Zookeeper and Kafka if not running
start_kafka_services() {
    if is_kafka_running; then
        log "Kafka is already running. Skipping Kafka startup."
    else
        log "Kafka is not running. Starting Zookeeper and Kafka..."

        # Define Kafka directories and scripts
        KAFKA_DIR="/usr/local/kafka"
        ZOOKEEPER_SCRIPT="bin/zookeeper-server-start.sh"
        ZOOKEEPER_CONFIG="config/zookeeper.properties"
        KAFKA_SCRIPT="bin/kafka-server-start.sh"
        KAFKA_CONFIG="config/server.properties"

        # Start Zookeeper in a new screen session
        if screen -list | grep -q "\.zookeeper_screen"; then
            log "Zookeeper screen session already exists. Skipping..."
        else
            log "Starting Zookeeper..."
            screen -dmS "zookeeper_screen" bash -c "cd '$KAFKA_DIR' && ./\"$ZOOKEEPER_SCRIPT\" \"$ZOOKEEPER_CONFIG\""
            log "Zookeeper started in screen session 'zookeeper_screen'."
        fi

        # Wait until Zookeeper is ready (default port 2181)
        if wait_for_port "localhost" 2181 60; then
            log "Zookeeper is ready."
        else
            log "Zookeeper failed to start within the expected time."
            exit 1
        fi

        # Start Kafka in a new screen session with retry logic
        local max_retries=3
        local attempt=1
        local kafka_started=false

        while [ $attempt -le $max_retries ]; do
            log "Attempt $attempt to start Kafka..."

            if screen -list | grep -q "\.kafka_screen"; then
                log "Kafka screen session already exists. Skipping..."
            else
                log "Starting Kafka..."
                screen -dmS "kafka_screen" bash -c "cd '$KAFKA_DIR' && ./\"$KAFKA_SCRIPT\" \"$KAFKA_CONFIG\""
                log "Kafka started in screen session 'kafka_screen'."
            fi

            # Wait until Kafka is ready (default port 9092)
            if wait_for_port "localhost" 9092 10; then
                log "Kafka is ready."
                kafka_started=true
                break
            else
                log "Kafka failed to start within the expected time."

                # Check if the port is congested
                if kill_process_on_port 9092; then
                    log "Freed up port 9092. Retrying Kafka startup..."
                else
                    log "Port 9092 is not congested. Cannot proceed with Kafka restart."
                fi
            fi

            attempt=$((attempt + 1))
        done

        if [ "$kafka_started" = false ]; then
            log "Failed to start Kafka after $max_retries attempts."
            exit 1
        fi
    fi
}

start_cassandra() {
    # Check if Cassandra is already running
    if pgrep -x "cassandra" > /dev/null; then
        echo "Cassandra is already running."
        return 0
    fi

    # Initialize conda
    echo "Initializing conda..."
    CONDA_PATH="$HOME/anaconda3/etc/profile.d/conda.sh"
    if [ ! -f "$CONDA_PATH" ]; then
        CONDA_PATH="$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
    
    if [ ! -f "$CONDA_PATH" ]; then
        echo "Could not find conda.sh. Please ensure Anaconda/Miniconda is installed."
        return 1
    fi
    
    source "$CONDA_PATH"
    
    # Activate the Conda environment
    echo "Activating Conda environment..."
    conda activate cassandra_env_39 || {
        echo "Failed to activate Conda environment. Please ensure it exists."
        return 1
    }

    # Start Cassandra with modified logging location
    echo "Starting Cassandra..."
    cassandra -R &> "$LOCAL_LOG_DIR/cassandra_startup.log" &
    
    # Get the Cassandra PID
    CASSANDRA_PID=$!
    
    # Function to check if Cassandra is initialized
    check_cassandra_status() {
        nodetool status &> /dev/null
        return $?
    }

    # Wait for Cassandra to initialize (timeout after 2 minutes)
    echo "Waiting for Cassandra to initialize..."
    COUNTER=0
    while ! check_cassandra_status && [ $COUNTER -lt 120 ]; do
        sleep 1
        ((COUNTER++))
        
        # Show progress every 10 seconds
        if [ $((COUNTER % 10)) -eq 0 ]; then
            echo "Still waiting for Cassandra to initialize... ($COUNTER seconds)"
        fi
    done

    # Check if initialization was successful
    if check_cassandra_status; then
        echo "Cassandra successfully started and initialized"
        
        # Deactivate Python virtual environment
        echo "Deactivating Python virtual environment..."
        deactivate
        return 0
    else
        echo "Failed to initialize Cassandra within timeout period"
        
        # Kill Cassandra process if it failed to initialize
        kill $CASSANDRA_PID
        
        # Deactivate Python virtual environment
        echo "Deactivating Python virtual environment..."
        deactivate
        return 1
    fi
}

# Function to start a python screen session with environment activation
start_python_screen_session() {
    local session_name=$1
    local directory=$2
    local script=$3
    local args=$4

    # Copy start_script.sh to the target directory if it doesn't exist
    if [ ! -f "$directory/start_script.sh" ]; then
        cp "$(dirname "${BASH_SOURCE[0]}")/start_script.sh" "$directory/"
        chmod +x "$directory/start_script.sh"
    fi

    log "Debug: Starting session '$session_name' with script '$script' and args '$args'"

    # Check if the screen session already exists
    if screen -list | grep -q "\.${session_name}"; then
        log "Screen session '$session_name' already exists. Skipping..."
        return 0
    fi

    # Create a log directory if it doesn't exist
    mkdir -p "$directory/error_logs"
    local ERROR_LOG="$directory/error_logs/${session_name}_error.log"
    
    # Use absolute path for start_script.sh
    CMD="cd '$directory' && $directory/start_script.sh \"$script\" $args 2>&1 | tee -a \"$ERROR_LOG\""
    
    log "Debug: About to execute command: $CMD"

    # Start a new detached screen session with error redirection
    screen -L -Logfile "$ERROR_LOG" -dmS "$session_name" bash -c "$CMD"
    
    # Wait a moment and verify the screen session was created
    sleep 2
    if ! screen -list | grep -q "\.${session_name}"; then
        log "Error: Failed to create screen session '$session_name'. Check logs at: $ERROR_LOG"
        if [ -f "$ERROR_LOG" ]; then
            log "Last few lines of error log:"
            tail -n 5 "$ERROR_LOG"
        fi
        return 1
    fi

    log "Successfully started screen session '$session_name' in directory '$directory' running script '$script' with args: $args"
    log "Logs available at: $ERROR_LOG"
}

# Get the absolute path of the current script's directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define directories
BATCH_DIR="$CURRENT_DIR/batch_sources"
STREAM_DIR="$CURRENT_DIR/stream_sources"

# Define scripts
BATCH_SCRIPT="execute_script.sh"
STREAM_SCRIPT="execute_scripts.sh"

# Stream scripts spark models dir
STREAM_SPARK_MODELS_DIR="$CURRENT_DIR/spark_models_scripts"

# Stream spark models scripts
STREAM_ETHEREUM_MODEL="NEW_train_offset_3_10min_ETH.py"
STREAM_YLIFE_MODEL="NEW_train_offset_3_10min.py"

# Streamlit app
STREAMLIT_APP_DIR="$CURRENT_DIR/streamlit_app"
STREAMLIT_APP_SCRIPT="app_2.py"

# ================================
# Main Execution Flow
# ================================

# # Start NIFI
# start_nifi

# # Start Hadoop
# start_hadoop

# # Start Kafka services if necessary
# start_kafka_services

# # Check current number of partitions and modify only if not 7
# current_partitions=$(kafka-topics.sh --describe --topic model-topic --bootstrap-server localhost:9092 | grep 'PartitionCount' | awk '{for(i=1;i<=NF;i++) if($i=="PartitionCount:") print $(i+1)}')

# if [ "$current_partitions" != "7" ]; then
#     log "Current partitions: $current_partitions. Updating to 7 partitions..."�
#     kafka-topics.sh --alter --topic model-topic --partitions 7 --bootstrap-server localhost:9092
# else
#     log "Topic already has 7 partitions. Skipping modification."
# fi

# Start Cassandra 
# start_cassandra

# Start screen sessions for batch and stream sources
start_screen_session "batch_sources_screen" "$BATCH_DIR" "$BATCH_SCRIPT"
start_screen_session "stream_sources_screen" "$STREAM_DIR" "$STREAM_SCRIPT"

# # Start Ethereum model screen session
# log "Starting Ethereum model screen session..."
# start_python_screen_session "ethereum_model" "$STREAM_SPARK_MODELS_DIR" "$STREAM_ETHEREUM_MODEL" "ETHEREUM"

# # Start screens for other symbols
# for symbol in BP COP SHEL XOM; do
#     log "Starting model screen session for $symbol..."
#     start_python_screen_session "model_${symbol}" "$STREAM_SPARK_MODELS_DIR" "$STREAM_YLIFE_MODEL" "$symbol"
# done

# # Start Streamlit app screen session
# log "Starting Streamlit app screen session..."
# start_python_screen_session "streamlit_app" "$STREAMLIT_APP_DIR" "$STREAMLIT_APP_SCRIPT"

# log "Streamlit URLs:"
# echo "  Local URL: http://localhost:8501"
# echo "  Network URL: http://10.0.2.15:8501"
# echo "  External URL: http://89.64.68.185:8501"

# log "All screen sessions have been started successfully."

# log "All services and screen sessions have been initiated."
