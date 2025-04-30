
import argparse
import time
import json
import torch
import pandas as pd
import os
from FlagEmbedding import LayerWiseFlagLLMReranker
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

def main(num_of_chunks, num_of_overlaps, target_num_of_chunks, target_num_of_overlaps, number_of_docs, top_k, rerank, reranker_type):

    total_start_time = time.time()

    print(f"Getting Representative Chunks of parameters {target_num_of_chunks} chunks and {target_num_of_overlaps} overlaps from {num_of_chunks} chunks and {num_of_overlaps} overlaps.")
    print(f"Running with {number_of_docs} docs, top {top_k} documents, with rerank? {rerank}, using {reranker_type} reranker.")

    # 1. Setup
    # Chunking parameters
    num_of_chunks = num_of_chunks
    num_of_overlaps = num_of_overlaps

    print(f"Number of Chunks: {num_of_chunks}")
    print(f"Number of Overlaps: {num_of_overlaps}")

    target_num_of_chunks = target_num_of_chunks
    target_num_of_overlaps = target_num_of_overlaps

    print(f"Target Number of Chunks: {target_num_of_chunks}")
    print(f"Target Number of Overlaps: {target_num_of_overlaps}")

    # Number of documents to retrieve
    number_of_docs = number_of_docs

    # Varying top k documents to get from rerank
    top_k = top_k

    print(f"Number of Docs: {number_of_docs}")
    
    print("Initializing text splitter...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=target_num_of_chunks, chunk_overlap=target_num_of_overlaps)

    model_name = "BAAI/bge-reranker-v2-minicpm-layerwise"

    device = "cuda"

    # Load reranker
    print("Loading reranker model...")
    reranker = LayerWiseFlagLLMReranker(model_name, use_fp16=True, device=device, trust_remote_code=True)
    print("Reranker model loaded.")
    print("Reranker model:", model_name)


    # 2. Loading Data
    queries, target_citations = load_queries_from_csv("/home/ubuntu/[March 27, 2025] Final Dataset/sampled_final_cleaned_acl_global_context_dataset_eval.csv")
    
    chunks_file_path = f"/home/ubuntu/RAG Pipeline/March 27 Chunks/{num_of_chunks}-{num_of_overlaps}-{number_of_docs}-retrieved_chunks.json"

    retrieved_chunks = load_json(chunks_file_path)

    print(f"Number of queries: {len(queries)}")
    print("Number of target citations:", len(target_citations))
    print(f"Number of retrieved chunks: {len(retrieved_chunks)}")

    # 3. Prepare input and compute representative chunk for each chunk

    complete_representative_chunks = []

    for i in range(len(queries)):

        representative_chunks_for_query = []

        query_start_time = time.time()

        print("Query", i+1)

        for j in range(len(retrieved_chunks[i])):

            reranker_input = []
            chunks_placeholder = []

            chunks = text_splitter.split_text(retrieved_chunks[i][j])  

            for chunk in chunks:
                reranker_input.append([queries[i], chunk])
                chunks_placeholder.append(chunk)

            scores = reranker.compute_score(reranker_input, cutoff_layers=[28])

            # Get indices that would sort the list in descending order
            sorted_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

            # Index of the representative chunk
            chosen_index = sorted_indices[0]
            
            # Get only the highest-ranked chunk for one document
            representative_chunks_for_query.append(chunks_placeholder[chosen_index])

        # Adds the representative chunk of each retrieved document for the query
        complete_representative_chunks.append(representative_chunks_for_query)

        # Optionally, print time statistics
        print(f"Time taken for query: {time.time() - query_start_time:.2f} seconds")
        print(f"Total time since start: {time.time() - total_start_time:.2f} seconds")

        # Free up memory
        torch.cuda.empty_cache()

    rep_chunks_file_name = f"{target_num_of_chunks}-{target_num_of_overlaps}-from-{num_of_chunks}-{num_of_overlaps}-{number_of_docs}-representative_chunks.json"
    rep_chunks_file_path = "/home/ubuntu/RAG Pipeline/March 27 Rep Chunks"
    rep_chunks_final_path = os.path.join(rep_chunks_file_path, rep_chunks_file_name)

    with open(rep_chunks_final_path, "w") as file:
        json.dump(complete_representative_chunks, file, indent=4)

    print("Representative chunks saved as a .json file.")

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
                        help="Number of overlaps (default: 256)"),
    parser.add_argument("--target_num_of_chunks", type=int, default=256,
                        help="Number of chunks for representative chunk (default: 256)")
    parser.add_argument("--target_num_of_overlaps", type=int, default=64,
                        help="Number of overlaps for representative chunk (default: 64)")
    parser.add_argument("--number_of_docs", type=int, default=100,
                        help="Number of documents (default: 100)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top k documents (default: 10)")
    parser.add_argument("--rerank", type=str2bool, default=True, 
                        help="Whether to apply reranking.")
    parser.add_argument("--reranker_type", type=str, default="flag", 
                        help="Type of reranker to use (flag).")
    args = parser.parse_args()
    
    main(args.num_of_chunks, args.num_of_overlaps, args.target_num_of_chunks, args.target_num_of_overlaps, args.number_of_docs, args.top_k, args.rerank, args.reranker_type)
