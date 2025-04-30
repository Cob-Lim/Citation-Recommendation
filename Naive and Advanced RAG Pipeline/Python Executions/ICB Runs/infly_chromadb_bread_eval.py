
# Install requirements.txt first!

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)         # Should return 12.1
print(torch.backends.cudnn.version())  # Should print cuDNN version

import os

# Completely disable GPU for TensorFlow
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Ensure no GPU is used
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

import tensorflow as tf

# Check if TensorFlow still detects any GPUs
print(tf.config.list_physical_devices("GPU"))  # Should return []


import json
import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from mxbai_rerank import MxbaiRerankV2
from FlagEmbedding import LightWeightFlagLLMReranker


import argparse

### Check GPU
def check_gpu():
    """Verify if a GPU is present and return device type."""
    print("GPU load successful.")
    return "cuda" if torch.cuda.is_available() else "cpu"

### Load Texts
def load_json(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"JSON file loaded from {file_path}")
        return json.load(f)


def rerank_results(query, retrieved_citation_items, retrieved_chunks, reranker_type, reranker_model, top_k):
    """Use cross-encoder to rerank retrieved texts, ensuring unique citation_items after reranking."""

    # Return early if there are no documents
    if not retrieved_chunks:
        return []

    # Rerank in batches to avoid OOM errors
    if len(retrieved_chunks) > 100:
        batch_size = 25
    else:
        batch_size = len(retrieved_chunks)

    results = []

    if reranker_type == "mxbai_v2":
        # Code for mxbai v2 rerankers
        try:
            for i in range(0, len(retrieved_chunks), batch_size):
                chunk_batch = retrieved_chunks[i:i + batch_size]
                with torch.no_grad():
                    batch_results = reranker_model.rank(query, chunk_batch)

                # 🔁 Adjust result.index before extending
                for result in batch_results:
                    result.index += i  # shift index to global position

                results.extend(batch_results)

            # Sort results by score in descending order
            results.sort(key=lambda x: x.score, reverse=True)
            sorted_indices = [result.index for result in results]

        except torch.cuda.OutOfMemoryError:
            print("OOM during reranking. Returning original chunks.")
            sorted_indices = [i for i in range(len(retrieved_chunks))]

    elif reranker_type == "mxbai_v1":
        # Code for mxbai v1 rerankers
        try:
            for i in range(0, len(retrieved_chunks), batch_size):
                chunk_batch = retrieved_chunks[i:i + batch_size]
                with torch.no_grad():
                    batch_results = reranker_model.rank(query, chunk_batch)

                # 🔁 Adjust result.index before extending
                for result in batch_results:
                    result["corpus_id"] += i  # shift index to global position
                results.extend(batch_results)

            # Sort results by score in descending order
            results.sort(key=lambda x: x["score"], reverse=True)
            sorted_indices = [result["corpus_id"] for result in results]

        except torch.cuda.OutOfMemoryError:
            print("OOM during reranking. Returning original chunks.")
            sorted_indices = [i for i in range(len(retrieved_chunks))]

    # Extract text content efficiently
    citation_items = retrieved_citation_items

    # Keep only the highest-ranked document for each unique citation_item
    unique_citation_docs = [] # List to track unique citations

    for idx in sorted_indices:
        citation = citation_items[idx]

        if citation and citation not in unique_citation_docs:
            unique_citation_docs.append(citation) # Track this citation
            
        if len(unique_citation_docs) == top_k:  # Stop once we have enough unique results
            break

    return unique_citation_docs


def match_citations(query, retrieved_citation_items, retrieved_chunks, reranker_model,  rerank, reranker_type, citation_target, top_k=10):
    """
    Matches citations.
    
    Args:
        query (str): The input query.
        retrieved_citation_items: list of retrieved citations (list of str)
        retrieved_chunks: list of retrieved chunks (list of str)
        reranker_model: the reranker model to be used
        rerank: True if you want to rerank, False otherwise
        citation_target (str): The citation to match.
        top_k (int): Number of top retrieved documents.

    Returns:
        List of documents.
    """

    if rerank == True:
        unique_citation_items = rerank_results(query, retrieved_citation_items, retrieved_chunks, reranker_type, reranker_model, top_k)
    else:

        # Keep only the highest-ranked document for each unique citation_item
        unique_citation_items = [] # List to track unique citations

        for citation in retrieved_citation_items:

            if citation and citation not in unique_citation_items:
                unique_citation_items.append(citation) # Track this citation
                
            if len(unique_citation_items) == top_k:  # Stop once we have enough unique results
                break

    # Match citations
    matched_citations = []
    matched_position = []

    citation_target = citation_target.strip()
    citation_set = {citation_target}  # Use set for O(1) lookup

    for index, citation in enumerate(unique_citation_items):
        if citation.strip() in citation_set:
            matched_citations.append(citation)
            matched_position.append(index)
            break

    return matched_citations, matched_position


### Evaluation Component
def load_queries_from_csv(csv_path: str, just_context: bool):
    """Loads queries and citation targets from a CSV file using pandas."""

    df = pd.read_csv(csv_path, encoding="utf-8")

    if just_context:
        queries = df["masked_cit_context"].tolist()
    else:
        # Fix invalid escape sequences in citing_abstract
        df["citing_abstract"] = df["citing_abstract"].str.replace(r"\/", "/", regex=True)

        # Construct the full query using the new format:
        # masked_cit_context + " </s> " + citing_title + " </s> " + citing_abstract
        queries = (df["masked_cit_context"] + " </s> " + df["citing_title"] + " </s> " + df["citing_abstract"]).tolist()
        
    citation_targets = df["masked_token_target"].tolist()

    return queries, citation_targets


def compute_all_metrics_batches(csv_path, just_context, chunks_file_path, citations_file_path, rerank, reranker_type, reranker_model, top_k, batch_size=100):
    """
    Evaluates citation retrieval over all queries using batching.
    Returns a dict with average metrics and the total number of queries processed.

    Args:
        csv_path (str): Path to the CSV file.
        just_context (bool): Whether to use just context.
        collection: The Chromadb containing the embeddings and metadata.
        embedding_model: The embedding model to use for encoding the queries.
        reranker_model: The reranker model used after initial retrieval.
        number_of_docs (int): Number of documents to retrieve in the first pass.
        top_k (int): Number of top retrieved documents to consider for matching.
        batch_size (int): Number of queries to process per batch.
    
    Returns:
        dict: {
            "Recall": float,
            "MRR": float,
            "NDCG": float,
            "count": int
        }
    """
    # Load queries and target citations
    all_queries, all_targets = load_queries_from_csv(csv_path, just_context)
    print(f"Total Number of Queries to be Evaluated: {len(all_queries)}")

    # Load the chunks and citations
    retrieved_chunks = load_json(chunks_file_path)
    retrieved_citations = load_json(citations_file_path)

    recall_scores = []
    mrr_scores = []
    ndcg_scores = []
    
    total_start_time = time.time()

    # Process queries in batches
    for batch_start in range(0, len(all_queries), batch_size):

        batch_queries = all_queries[batch_start:batch_start + batch_size]
        batch_targets = all_targets[batch_start:batch_start + batch_size]
        batch_chunks = retrieved_chunks[batch_start:batch_start + batch_size]
        batch_citations = retrieved_citations[batch_start:batch_start + batch_size]

        for query, citation_target, chunk, citation in zip(batch_queries, batch_targets, batch_chunks, batch_citations):
            matched_citations, matched_position = match_citations(
                query=query,
                retrieved_citation_items=citation, 
                retrieved_chunks=chunk,
                reranker_model=reranker_model,
                reranker_type=reranker_type,
                rerank=rerank,
                citation_target=citation_target,
                top_k=top_k
            )

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

        # Optionally, print batch statistics
        batch_elapsed = time.time() - total_start_time
        print(f"Processed batch ending at query {min(batch_start + batch_size, len(all_queries))} out of {len(all_queries)}, Total time since start: {batch_elapsed:.2f} seconds")

        # Compute and print running metrics
        running_avg_recall = np.mean(recall_scores)
        running_avg_mrr = np.mean(mrr_scores)
        running_avg_ndcg = np.mean(ndcg_scores)
        
        print(f"Running Metrics: Average Recall: {running_avg_recall:.4f} | Average MRR: {running_avg_mrr:.4f} | Average NDCG: {running_avg_ndcg:.4f}")

        # Free up memory
        torch.cuda.empty_cache()

    # Aggregate metrics over all queries
    count = len(all_queries)
    overall_metrics = {
        "Recall": np.mean(recall_scores) if count > 0 else 0,
        "MRR": np.mean(mrr_scores) if count > 0 else 0,
        "NDCG": np.mean(ndcg_scores) if count > 0 else 0,
        "count": count
    }
    
    print("Overall Metrics:")
    print(overall_metrics)
    
    return overall_metrics



### To Run the Code
def main(num_of_chunks, num_of_overlaps, number_of_docs, top_k, rerank, reranker_type):

    print(f"Running with {num_of_chunks} chunks, {num_of_overlaps} overlaps, {number_of_docs} docs, top {top_k} documents, with rerank? {rerank}.")

    device = check_gpu()
    print(f"GPU: {device}")

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

    chunks_file_path = f"/home/ubuntu/RAG Pipeline/March 24 Chunks/{num_of_chunks}-{num_of_overlaps}-{number_of_docs}-retrieved_chunks.json"
    citations_file_path = f"/home/ubuntu/RAG Pipeline/March 24 Citations/{num_of_chunks}-{num_of_overlaps}-{number_of_docs}-retrieved_citations.json"
    print("Chunks and Citations file path loaded.")

    # Will rerank or not
    rerank = rerank

    # Define reranker model
    if reranker_type == "flag":
        reranker_model_name = "BAAI/bge-reranker-v2.5-gemma2-lightweight"
        reranker_model = LightWeightFlagLLMReranker(reranker_model_name, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    
    if reranker_type == "mxbai_v1":
        reranker_model_name = "mixedbread-ai/mxbai-rerank-large-v1"
        reranker_model = CrossEncoder(reranker_model_name, device=device)
    
    if reranker_type == "mxbai_v2":
        reranker_model_name = "mixedbread-ai/mxbai-rerank-large-v2" # change to mxbai-rerank-large-v2 if you want
        reranker_model = MxbaiRerankV2(reranker_model_name, device=device)

    print(f"Reranker model loaded: {reranker_model_name}")

    # Evaluate Reranker
    eval_csv_path = "/home/ubuntu/[March 24, 2025] Final Dataset/full_text_sample_final_cleaned_acl_global_context_dataset_eval.csv" # Update with your actual file path

    # Just context or not
    just_context = True

    # Batch size
    batch_size = 5

    print("Other parameters loaded successfully!")

    print(f"Evaluating for {just_context}_Query_Only-{rerank}_Rerank-{num_of_chunks}-{num_of_overlaps}-Infly-ChromaDB-Bread-{number_of_docs}-top_{top_k}")

    overall_metrics = compute_all_metrics_batches(
        csv_path=eval_csv_path,  # Path to your CSV file
        just_context=just_context, # Just context or not
        chunks_file_path=chunks_file_path, # Chunks file path
        citations_file_path=citations_file_path, # Citations file path
        rerank=rerank, # True if will rerank and False if not
        reranker_type=reranker_type, # Type of reranker to use
        reranker_model=reranker_model,   # Replace with your reranker model
        top_k=top_k, # Top k documents to retain
        batch_size=batch_size # How many to evaluate in a batch
    )

    print(f"Done Evaluating for {just_context}_Query Only-{rerank}_Rerank-{num_of_chunks}-{num_of_overlaps}-Infly-ChromaDB-Bread-{number_of_docs}-top_{top_k}")


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
    parser = argparse.ArgumentParser(description="Infly-ChromaDB-Bread (ICB) evaluation.")
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
    parser.add_argument("--reranker_type", type=str, default="mxbai", 
                        help="Type of reranker to use (flag or mxbai).")
    args = parser.parse_args()
    
    main(args.num_of_chunks, args.num_of_overlaps, args.number_of_docs, args.top_k, args.rerank, args.reranker_type)