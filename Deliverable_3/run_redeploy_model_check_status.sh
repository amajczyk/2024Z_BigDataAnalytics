#!/bin/bash

# Define the base URL and symbols with their respective ports
base_url="http://localhost"
declare -A symbols_ports=(
  [BP]=5000
  [COP]=5001
  [ETHEREUM]=5002
  [SHEL]=5003
  [XOM]=5004
)

# Deploy models for each symbol
for symbol in "${!symbols_ports[@]}"; do
  port="${symbols_ports[$symbol]}"
  deploy_url="$base_url:$port/$symbol/deploy"
  echo "Deploying model for symbol: $symbol at $deploy_url"

  # Send a POST request to deploy the model
  response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$deploy_url")

  if [ "$response" -eq 200 ]; then
    echo "Successfully deployed model for $symbol."
  else
    echo "Failed to deploy model for $symbol. HTTP Status: $response."
  fi

done

# Check the status of deployed models for each symbol
for symbol in "${!symbols_ports[@]}"; do
  port="${symbols_ports[$symbol]}"
  status_url="$base_url:$port/$symbol/status"
  echo "Checking status for symbol: $symbol at $status_url"

  # Send a GET request to check the status
  status_response=$(curl -s "$status_url")

  echo "Status for $symbol:"
  echo "$status_response"

done