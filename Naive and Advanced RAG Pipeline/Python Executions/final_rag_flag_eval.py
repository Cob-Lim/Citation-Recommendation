
import argparse
import time
import json
import torch
import pandas as pd
import numpy as np
from FlagEmbedding import LayerWiseFlagLLMReranker


def load_json(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"JSON file loaded from {file_path}")
        return json.load(f)


def load_queries_from_csv(csv_path: str):
    """Loads queries and citation targets from a CSV file using pandas."""

    df = pd.read_csv(csv_path, encoding="utf-8")

    queries = df["masked_cit_context"].tolist()
        
    citation_targets = df["masked_token_target"].tolist()

    return queries, citation_targets

def main(num_of_chunks, num_of_overlaps, number_of_docs, top_k, rerank, reranker_type):

    total_start_time = time.time()

    print(f"Running with {num_of_chunks} chunks, {num_of_overlaps} overlaps, {number_of_docs} docs, top {top_k} documents, with rerank? {rerank}-{reranker_type}.")

    # 1. Setup

    # Chunking parameters
    num_of_chunks = num_of_chunks
    num_of_overlaps = num_of_overlaps

    print(f"Number of Chunks: {num_of_chunks}")
    print(f"Number of Overlaps: {num_of_overlaps}")

    # Number of documents to retrieve
    number_of_docs = number_of_docs

    # Varying top k documents to get from rerank
    top_k = top_k

    print(f"Number of Docs: {number_of_docs}")
          
    model_name = "BAAI/bge-reranker-v2-minicpm-layerwise"

    device = "cuda"

    # Load reranker
    print("Loading reranker model...")
    reranker = LayerWiseFlagLLMReranker(model_name, use_fp16=True, device=device, trust_remote_code=True)
    print("Reranker model loaded.")
    print("Reranker model:", model_name)


    # 2. Loading Data
    queries, target_citations = load_queries_from_csv("/home/ubuntu/[March 27, 2025] Final Dataset/sampled_final_cleaned_acl_global_context_dataset_eval.csv")
    
    citations_file_path = f"/home/ubuntu/RAG Pipeline/March 27 Citations/{num_of_chunks}-{num_of_overlaps}-{number_of_docs}-retrieved_citations.json"
    titles_file_path = f"/home/ubuntu/RAG Pipeline/March 27 Titles/{num_of_chunks}-{num_of_overlaps}-{number_of_docs}-retrieved_titles.json"

    retrieved_citations = load_json(citations_file_path)
    retrieved_titles = load_json(titles_file_path)

    print(f"Number of queries: {len(queries)}")
    print("Number of target citations:", len(target_citations))
    print(f"Number of retrieved citations: {len(retrieved_citations)}")
    print(f"Number of titles: {len(retrieved_titles)}")

    # 3. Prepare input and compute scores
    batch_size = 100
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for i in range(len(queries)):

        reranker_input = []
        query_scores = []

        query_start_time = time.time()

        print("Query", i+1)

        for j in range(len(retrieved_titles[i])):

            reranker_input.append([queries[i], retrieved_titles[i][j]])

        for k in range(0, len(reranker_input), batch_size):

            reranker_input_batch = reranker_input[k:k+batch_size]
            scores = reranker.compute_score(reranker_input_batch, cutoff_layers=[28])
            query_scores.extend(scores)

        # Get indices that would sort the list in descending order
        sorted_indices = [i for i, _ in sorted(enumerate(query_scores), key=lambda x: x[1], reverse=True)]

        # Extract text content efficiently
        citation_items = retrieved_citations[i]

        # Keep only the highest-ranked document for each unique citation_item
        unique_citation_items = [] # List to track unique citations

        for idx in sorted_indices:
            citation = citation_items[idx]

            if citation and citation not in unique_citation_items:
                unique_citation_items.append(citation) # Track this citation
                
            if len(unique_citation_items) == top_k:  # Stop once we have enough unique results
                break
        
        # Match citations
        matched_citations = []
        matched_position = []

        citation_target = target_citations[i].strip()
        citation_set = {citation_target}  # Use set for O(1) lookup

        for index, citation in enumerate(unique_citation_items):
            if citation.strip() in citation_set:
                matched_citations.append(citation)
                matched_position.append(index)
                break

        # Compute metrics for this query
        if matched_position:
            pos = matched_position[0]
            recall_scores.append(1)
            mrr_scores.append(1 / (pos + 1))
            ndcg_scores.append(1 / np.log2(pos + 2))
        else:
            recall_scores.append(0)
            mrr_scores.append(0)
            ndcg_scores.append(0)

        # Optionally, print time statistics
        print(f"Time taken for query: {time.time() - query_start_time:.2f} seconds")
        print(f"Total time since start: {time.time() - total_start_time:.2f} seconds")

        if (i + 1) % 5 == 0:

            print(f"Processed {i+1} queries.")

            # Compute and print running metrics
            running_avg_recall = np.mean(recall_scores)
            running_avg_mrr = np.mean(mrr_scores)
            running_avg_ndcg = np.mean(ndcg_scores)
            
            print(f"Running Metrics: Average Recall: {running_avg_recall:.4f} | Average MRR: {running_avg_mrr:.4f} | Average NDCG: {running_avg_ndcg:.4f}")

        # Free up memory
        torch.cuda.empty_cache()

    # Aggregate metrics over all queries
    count = len(queries)
    overall_metrics = {
        "Recall": np.mean(recall_scores) if count > 0 else 0,
        "MRR": np.mean(mrr_scores) if count > 0 else 0,
        "NDCG": np.mean(ndcg_scores) if count > 0 else 0,
        "count": count
    }
    
    print("Overall Metrics:")
    print(overall_metrics)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final RAG Pipeline evaluation.")
    parser.add_argument("--num_of_chunks", type=int, default=1024,
                        help="Number of chunks (default: 1024)")
    parser.add_argument("--num_of_overlaps", type=int, default=256,
                        help="Number of overlaps (default: 256)")
    parser.add_argument("--number_of_docs", type=int, default=100,
                        help="Number of documents (default: 100)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top k documents (default: 10)")
    parser.add_argument("--rerank", type=str2bool, default=True, 
                        help="Whether to apply reranking.")
    parser.add_argument("--reranker_type", type=str, default="flag", 
                        help="Type of reranker to use (flag).")
    args = parser.parse_args()
    
    main(args.num_of_chunks, args.num_of_overlaps, args.number_of_docs, args.top_k, args.rerank, args.reranker_type)
