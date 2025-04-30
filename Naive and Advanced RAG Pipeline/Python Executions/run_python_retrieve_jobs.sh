#!/bin/bash
# run_python_jobs.sh

JOB_FILE="/home/ubuntu/RAG Pipeline/Python Executions/All the Jobs/python_retrieve_jobs.txt"
BASE_LOG_DIR="/home/ubuntu/RAG Pipeline/Logs/March 27 Retrieval"

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines or comments
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi

    # Extract parameters from the current line
    read -r num_of_chunks num_of_overlaps num_of_docs <<< "$line"
    
    # Define the log file based on num_of_docs 
    log_file="${BASE_LOG_DIR}/${num_of_chunks}-${num_of_overlaps}_${num_of_docs}_retrieved.log"
    
    # Log the job start time and parameters
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting job with: chunks=${num_of_chunks}, overlaps=${num_of_overlaps}, docs=${num_of_docs}" | tee -a "$log_file"
    
    # Run the Python script with unbuffered output and log both stdout and stderr
    python -u final_rag_retrieve.py --num_of_chunks "$num_of_chunks" --num_of_overlaps "$num_of_overlaps" --num_of_docs "$num_of_docs" >> "$log_file" 2>&1
    
    # Log the completion of the job
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Job completed. Proceeding to next job..." | tee -a "$log_file"
done < "$JOB_FILE"