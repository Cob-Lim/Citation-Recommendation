#!/bin/bash
# run_python_jobs.sh

JOB_FILE="/home/ubuntu/RAG Pipeline/Python Executions/All the Jobs/python_eval_yes-rr_jobs.txt"
BASE_LOG_DIR="/home/ubuntu/RAG Pipeline/Logs/March 27 Eval"

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines or comments
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi

    # Extract parameters from the current line
    read -r num_of_chunks num_of_overlaps number_of_docs top_k rerank reranker_type <<< "$line"
    
    # Define the log directory based on number_of_docs (directory already exists)
    log_file="${BASE_LOG_DIR}/${num_of_chunks}-${num_of_overlaps}-${number_of_docs}_${rerank}-rr_flag_eval.log"
    
    # Log the job start time and parameters
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting job with: chunks=${num_of_chunks}, overlaps=${num_of_overlaps}, docs=${number_of_docs}, top_k=${top_k}, rerank=${rerank}, reranker_type=${reranker_type}" | tee -a "$log_file"

    # Run the Python script with unbuffered output and log both stdout and stderr
    python -u final_rag_flag_eval.py --num_of_chunks "$num_of_chunks" --num_of_overlaps "$num_of_overlaps" --number_of_docs "$number_of_docs" --top_k "$top_k" --rerank "$rerank" --reranker_type "$reranker_type" >> "$log_file" 2>&1
    
    # Log the completion of the job
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Job completed. Proceeding to next job..." | tee -a "$log_file"
done < "$JOB_FILE"