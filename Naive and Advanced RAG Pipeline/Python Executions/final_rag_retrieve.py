
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
import chromadb
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import argparse

### Check GPU
def check_gpu():
    """Verify if a GPU is present and return device type."""
    print("GPU load successful.")
    return "cuda" if torch.cuda.is_available() else "cpu"


### Retrieve similar items from ChromaDB
def retrieve_from_chroma(collection, query_embedding, number_of_docs=100):
    """
    Retrieve similar chunks from an already loaded ChromaDB collection.
    
    Args:
        collection: The already loaded ChromaDB collection.
        query_embedding: The query vector (NumPy array).
        number_of_docs: Number of similar chunks to return.
    
    Returns:
        List of retrieved chunks and their citation items.
    """
    # Convert query embedding to list format (required by ChromaDB)
    query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],  # Query must be a list
        n_results=number_of_docs
    )

    # Extract metadata list (metadata is inside a list)
    metadata_list = results["metadatas"][0]  # Get the first (and only) list

    # Extract citation items and chunked texts
    retrieved_citation_item = [entry.get("citation_item", "Unknown") for entry in metadata_list]
    retrieved_chunks = [entry.get("chunked_text", "") for entry in metadata_list]
    retrieved_titles = [entry.get("title_on_paper", "") for entry in metadata_list]

    return retrieved_citation_item, retrieved_chunks, retrieved_titles


### Load queries from csv for comparison
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



### Code to Run
def main(num_of_chunks, num_of_overlaps, num_of_docs):
    device = check_gpu()
    print(f"GPU: {device}")

    # Define embedding model
    embedding_model_name = "infly/inf-retriever-v1-1.5b"
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    print(f"Embedding model loaded: {embedding_model_name}")

    # Chunking parameters
    num_of_chunks = num_of_chunks
    num_of_overlaps = num_of_overlaps

    print(f"Number of Chunks: {num_of_chunks}")
    print(f"Number of Overlaps: {num_of_overlaps}")

    # Path to the Chroma DB
    chroma_path = f"../ChromaDB Storage"

    # Load the existing ChromaDB database
    client = chromadb.PersistentClient(path=chroma_path)

    collection_name = f"final_{num_of_chunks}-{num_of_overlaps}_vectorstore"

    # Load the collection
    loaded_collection = client.get_collection(name=collection_name)

    # Check stored data
    print("Collection loaded:", collection_name)
    print("Number of Data:", loaded_collection.count())

    # Number of documents to retrieve
    number_of_docs = num_of_docs
    
    print("Number of documents:", number_of_docs)

    # Evaluate Reranker
    eval_csv_path = "/home/ubuntu/[March 27, 2025] Final Dataset/sampled_final_cleaned_acl_global_context_dataset_eval.csv" # Update with your actual file path

    # Just context or not
    just_context = True

    queries, citation_targets = load_queries_from_csv(csv_path=eval_csv_path, just_context=just_context)

    print(f"Total number of queries: {len(queries)}")

    retrieved_chunks = []
    retrieved_citations = []
    retrieved_titles = []

    # Start time of the process
    start_time = time.time()

    for i, query in enumerate(queries, start=1):

        # Convert query to embedding
        query_embedding = embedding_model.encode(query, convert_to_numpy=True)
        citation_items, chunks, titles = retrieve_from_chroma(loaded_collection, query_embedding, number_of_docs=number_of_docs)

        # Appends the retrieved items to their respective list
        retrieved_chunks.append(chunks)
        retrieved_citations.append(citation_items)
        retrieved_titles.append(titles)

        # Print a status update every 100 queries
        if i % 100 == 0 or i == len(queries):

            check_time = time.time()
            elapsed_time = check_time - start_time

            print(f"Processing is ongoing/complete: {i} out of {len(queries)} queries processed.")
            print(f"Time elapsed since starting: {elapsed_time} seconds.")

    chunks_file_name = f"{num_of_chunks}-{num_of_overlaps}-{number_of_docs}-retrieved_chunks.json"
    chunks_file_path = "/home/ubuntu/RAG Pipeline/March 27 Chunks"
    chunks_final_path = os.path.join(chunks_file_path, chunks_file_name)

    with open(chunks_final_path, "w") as file:
        json.dump(retrieved_chunks, file, indent=4)

    print("Retrieved chunks saved as a .json file.")

    citations_file_name = f"{num_of_chunks}-{num_of_overlaps}-{number_of_docs}-retrieved_citations.json"
    citations_file_path = "/home/ubuntu/RAG Pipeline/March 27 Citations"
    citations_final_path = os.path.join(citations_file_path, citations_file_name)

    with open(citations_final_path, "w") as file:
        json.dump(retrieved_citations, file, indent=4)

    print("Retrieved citations saved as a .json file.")

    titles_file_name = f"{num_of_chunks}-{num_of_overlaps}-{number_of_docs}-retrieved_titles.json"
    titles_file_path = "/home/ubuntu/RAG Pipeline/March 27 Titles"
    titles_final_path = os.path.join(titles_file_path, titles_file_name)

    with open(titles_final_path, "w") as file:
        json.dump(retrieved_titles, file, indent=4)

    print("Retrieved titles saved as a .json file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final RAG retrieval.")
    parser.add_argument("--num_of_chunks", type=int, default=1024,
                        help="Number of chunks (default: 1024)")
    parser.add_argument("--num_of_overlaps", type=int, default=256,
                        help="Number of overlaps (default: 256)")
    parser.add_argument("--num_of_docs", type=int, default=100,
                        help="Number of documents (default: 100)")
    args = parser.parse_args()
    
    main(args.num_of_chunks, args.num_of_overlaps, args.num_of_docs)