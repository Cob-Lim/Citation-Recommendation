#!/bin/bash
# run_llm_rerankers_jobs.sh

JOB_FILE="/home/ubuntu/LLM Rerankers/run_llm_rerankers_jobs.txt"
BASE_LOG_DIR="/home/ubuntu/LLM Rerankers/Logs"

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines or comments
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi

    # Extract parameters from the current line
    read -r target_num_of_chunks target_num_of_overlaps original_num_of_chunks original_num_of_overlaps number_of_docs <<< "$line"
    
    # Define the log directory based on number_of_docs (directory already exists)
    log_file="${BASE_LOG_DIR}/Gemini/Gemini-reranked_orig_chunks-${number_of_docs}.log"
    
    # Log the job start time and parameters
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting job with: target chunks=${target_num_of_chunks}, target overlaps=${target_num_of_overlaps}, original chunks=${original_num_of_chunks}, original overlaps=${original_num_of_overlaps}, docs=${number_of_docs}" | tee -a "$log_file"

    # Run the Python script with unbuffered output and log both stdout and stderr
    python -u Gemini_reranker.py --target_num_of_chunks "$target_num_of_chunks" --target_num_of_overlaps "$target_num_of_overlaps" --original_num_of_chunks "$original_num_of_chunks" --original_num_of_overlaps "$original_num_of_overlaps" --number_of_docs "$number_of_docs" >> "$log_file" 2>&1
    
    # Log the completion of the job
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Job completed. Proceeding to next job..." | tee -a "$log_file"
done < "$JOB_FILE"