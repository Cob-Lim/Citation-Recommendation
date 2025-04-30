#!/bin/bash
# run_llm_rerankers_jobs.sh

JOB_FILE="/home/ubuntu/Two-Stage Retrieval/two_stage_experiments.txt"
BASE_LOG_DIR="/home/ubuntu/Two-Stage Retrieval/Logs"

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines or comments
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi

    # Extract parameters from the current line
    read -r number_of_docs top_k <<< "$line"
    
    # Define the log directory based on number_of_docs (directory already exists)
    log_file="${BASE_LOG_DIR}/Reranked-${number_of_docs}_docs-top_${top_k}-eval.log"
    
    # Log the job start time and parameters
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting job with: docs=${number_of_docs}, top_k=${top_k}" | tee -a "$log_file"

    # Run the Python script with unbuffered output and log both stdout and stderr
    python -u deciding_best_setup.py --number_of_docs "$number_of_docs" --top_k "$top_k" >> "$log_file" 2>&1
    
    # Log the completion of the job
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Job completed. Proceeding to next job..." | tee -a "$log_file"
done < "$JOB_FILE"