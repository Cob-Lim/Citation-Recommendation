import json
import torch
import faiss
import time
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

### Check for GPU
def check_gpu():
    """Verify if a GPU is present and return device type."""
    print("GPU load successful.")
    return "cuda" if torch.cuda.is_available() else "cpu"

### Load Texts
def load_json(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"JSON file loaded") 
        return json.load(f)
              
### Chunk Texts, Convert to Emebdding, and Store into FAISS Vector Database
def create_vectorstore(text_data, embedding_model, num_of_chunks, num_of_overlaps, batch_size=50):
    """Manually compute embeddings, save FAISS index, and store metadata in a Pickle file."""

    start_time = time.time() # Start time

    # Define file paths
    faiss_pickle_path = f"{num_of_chunks}-chunks_{num_of_overlaps}-overlaps_faiss_store.pkl"

    print("Initializing text splitter...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=num_of_chunks, chunk_overlap=num_of_overlaps)

    # Initialize FAISS index
    embedding_dim = embedding_model.encode("test", convert_to_numpy=True).shape[0]
    faiss_index = faiss.IndexFlatL2(embedding_dim)

    metadata_list = []

    for i in range(0, len(text_data), batch_size):

        batch_start_time = time.time()  # Start time for batch processing

        batch = text_data[i:i+batch_size]
        
        batch_embeddings = []
        batch_metadata = []

        for entry in batch:
            chunks = text_splitter.split_text(entry['full_text'])  
            for chunk in chunks:
                metadata = {
                    "acl_id": entry['acl_id'],
                    "author": entry['author'],
                    "title_on_paper": entry['title_on_paper'],
                    "year_on_paper": entry['year_on_paper'],
                    "citation_item": entry['citation_item'],
                    "full_text": entry['full_text']
                }
                batch_metadata.append(metadata)

                # Generate embedding
                embedding = embedding_model.encode(chunk, convert_to_numpy=True)
                batch_embeddings.append(embedding)

        # Append batch metadata to global list
        metadata_list.extend(batch_metadata)

        # Add batch to FAISS
        if batch_embeddings:
            batch_number = i // batch_size + 1
            batch_embeddings_np = np.vstack(batch_embeddings)
            faiss_index.add(batch_embeddings_np)

            batch_end_time = time.time()  # End time for batch processing
            batch_time_taken = batch_end_time - batch_start_time

            print(f"Added batch {batch_number} to FAISS index. Time taken: {batch_time_taken:.2f} seconds.")
            print(f"Total of {batch_number * batch_size} documents out of {len(text_data)} documents.")
    
    total_time = time.time() - start_time  # Calculate total time
    print(f"Total documents for FAISS: {len(metadata_list)}")
    print(f"Total execution time: {total_time:.2f} seconds.")  # Print total execution time
    
    # Save everything into a pickle file
    data_to_store = {
        "vectorstore": faiss_index,
        "metadata_list": metadata_list
    }
    
    with open(faiss_pickle_path, "wb") as f:
        pickle.dump(data_to_store, f)
    
    print(f"FAISS index and metadata saved as {faiss_pickle_path}.")
    
    return faiss_index, metadata_list

### Custom Retriever (since I manually calculated embeddings)
def retrieve_from_faiss(index, metadata_store, query_embedding, number_of_docs=100):
    """
    Retrieve the top 100 closest documents from FAISS index.
    
    Args:
        index: FAISS index (faiss.IndexFlatL2)
        metadata_store: List of metadata corresponding to the FAISS index
        query_embedding: The query vector (numpy array)
        number_of_docs: Number of documents to return
    
    Returns:
        List of retrieved documents and their distances.
    """
    # Ensure the query embedding has the correct shape (1, embedding_dim)
    query_vector = query_embedding.reshape(1, -1).astype('float32')

    # Search the FAISS index
    distances, indices = index.search(query_vector, number_of_docs)

    # Convert indices array to NumPy and filter invalid indices
    valid_indices = indices[0]
    valid_indices = valid_indices[valid_indices >= 0]  # NumPy boolean masking

    # Retrieve metadata using NumPy take() for efficiency
    retrieved_docs = np.take(metadata_store, valid_indices).tolist()

    return retrieved_docs

### Reranker Component
def rerank_results(query, retrieved_docs, reranker_model, top_k):
    """Use cross-encoder to rerank retrieved texts, ensuring unique citation_items after reranking."""

    # Return early if there are no documents
    if not retrieved_docs:
        return []  
    
    # Extract text content efficiently
    retrieved_texts = [doc.get("full_text", "") for doc in retrieved_docs]
    citation_items = [doc.get("citation_item", None) for doc in retrieved_docs]

    # Create query-text pairs for reranking
    pairs = [(query, text) for text in retrieved_texts]

    # Get scores for each pair
    scores = reranker_model.predict(pairs)

    # Use NumPy argsort for fast sorting (descending order)
    sorted_indices = np.argsort(scores)[::-1]

    # Keep only the highest-ranked document for each unique citation_item
    unique_citation_docs = [] # List to track unique citations
    reranked_results = []
    
    for idx in sorted_indices:
        doc = retrieved_docs[idx]
        citation = citation_items[idx]

        if citation and citation not in unique_citation_docs:
            unique_citation_docs.append(citation) # Track this citation
            reranked_results.append(doc)
            
        if len(reranked_results) == top_k:  # Stop once we have enough unique results
            break
    
    return reranked_results

def retrieve_and_match_citations(embedding_model, reranker_model, query, citation_target, index, metadata_store, number_of_docs=100, top_k=10):
    """
    Retrieves and matches citations using the specified matching type.
    
    Args:
        embedding_model: The embedding model used.
        reranker_model: The reranker model used.
        query (str): The input query.
        citation_target (str): The citation to match.
        index: FAISS index.
        metadata_store: Metadata store.
        matching_type (str): Type of matching ('exact', 'fuzzy', or 'levenshtein').
        top_k (int): Number of top retrieved documents.

    Returns:
        List of documents.
    """

    # Convert query to embedding
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)

    # Retrieve and rerank documents
    retrieved_docs = retrieve_from_faiss(index, metadata_store, query_embedding, number_of_docs)
    reranked_docs = rerank_results(query, retrieved_docs, reranker_model, top_k)

    # Extract "citation_item" field efficiently
    list_of_citations = [doc.get("citation_item", "") for doc in reranked_docs]

    # Match citations
    matched_citations = []
    matched_position = []

    citation_target = citation_target.strip()
    citation_set = {citation_target}  # Use set for O(1) lookup

    for index, citation in enumerate(list_of_citations):
        if citation.strip() in citation_set:
            matched_citations.append(citation)
            matched_position.append(index)
            break

    return matched_citations, matched_position

### Evaluate Reranker
def load_queries_from_csv(csv_path: str, just_context: bool):
    """Loads queries and citation targets from a CSV file using pandas."""

    df = pd.read_csv(csv_path, encoding="utf-8")

    if just_context:
        queries = df["masked_cit_context"]
    else:
        # Fix invalid escape sequences in citing_abstract
        df["citing_abstract"] = df["citing_abstract"].str.replace(r"\/", "/", regex=True)

        # Construct the full query using the new format:
        # masked_cit_context + " </s> " + citing_title + " </s> " + citing_abstract
        queries = (df["masked_cit_context"] + " </s> " + df["citing_title"] + " </s> " + df["citing_abstract"]).tolist()
        
    citation_targets = df["masked_token_target"].tolist()

    return queries, citation_targets

def chunk_list(lst, batch_size):
    """Yield successive batch-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def compute_batch_metrics(batch_queries, batch_targets, embedding_model, reranker_model, index,
                          metadata_store, number_of_docs, top_k):
    """
    Evaluates a batch of queries and computes average metrics and the count of evaluations.
    Returns a dict with the average metrics and the number of queries processed.
    """
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for i, (query, citation_target) in enumerate(zip(batch_queries, batch_targets)):

        # Retrieve documents using the citation retrieval system
        matched_citations, matched_position = retrieve_and_match_citations(
            embedding_model, reranker_model, query, citation_target,
            index, metadata_store, number_of_docs, top_k
        )
        
        # Compute metrics for the query
        if matched_position:
            pos = matched_position[0]
            recall_scores.append(1)
            mrr_scores.append(1 / (pos + 1))
            ndcg_scores.append(1 / np.log2(pos + 2))
        else:
            recall_scores.append(0)
            mrr_scores.append(0)
            ndcg_scores.append(0)
    
    count = len(batch_queries)
    batch_metrics = {
        "Recall": np.mean(recall_scores) if count > 0 else 0,
        "MRR": np.mean(mrr_scores) if count > 0 else 0,
        "NDCG": np.mean(ndcg_scores) if count > 0 else 0,
        "count": count
    }
    return batch_metrics

def compute_evaluation_metrics_in_batches(csv_path, just_context, embedding_model, reranker_model, index, metadata_store,
                                          number_of_docs=100, top_k=10, batch_size=100,
                                          start_index=0, output_dir="evaluation_results"):
    """
    Evaluates citation retrieval across multiple queries by dividing them into batches.
    Allows resuming from a specific start index and saves each batch's results separately.
    
    Args:
        csv_path (str): Path to the CSV file.
        embedding_model: The embedding model to use.
        reranker_model: The reranker model.
        index: The index to use for retrieval.
        metadata_store: The metadata storage.
        number_of_docs (int): Number of documents to retrieve.
        top_k (int): Number of top retrieved documents to consider.
        batch_size (int): Number of queries to process in each batch.
        start_index (int): The index from which to start/resume evaluation.
        output_dir (str): Directory where result files will be saved.
        
    Returns:
        A list of dictionaries, each containing details of the batch evaluation.
    """

    # Start tracking total time
    total_start_time = time.time()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV file to get all queries and citation targets
    queries, citation_targets = load_queries_from_csv(csv_path, just_context)
    
    # If resuming, slice the lists to start from start_index
    queries = queries[start_index:]
    citation_targets = citation_targets[start_index:]
    
    batch_results = []
    overall_offset = start_index  # Used to print the original query numbers
    
    # Process queries in batches
    for batch_num, (batch_queries, batch_targets) in enumerate(zip(chunk_list(queries, batch_size),
                                                                   chunk_list(citation_targets, batch_size))):
        
        batch_start_time = time.time()  # Start time for batch processing

        start_query = overall_offset + batch_num * batch_size
        end_query = start_query + len(batch_queries) - 1
        next_start_index = end_query + 1
        print(f"\nEvaluating batch {batch_num+1} (queries {start_query} to {end_query})...")
        
        batch_metrics = compute_batch_metrics(batch_queries, batch_targets, embedding_model, reranker_model, index,
                                              metadata_store, number_of_docs, top_k)
        
        batch_end_time = time.time()  # End time for batch processing
        batch_time_taken = batch_end_time - batch_start_time

        print(f"Batch {batch_num+1} metrics: {batch_metrics}")
        print(f"Batch {batch_num+1} completed in {batch_time_taken:.2f} seconds.")
        
        # Save details about the batch evaluation
        batch_info = {
            "batch_number": batch_num + 1,
            "start_index": start_query,
            "end_index": end_query,
            "next_start_index": next_start_index,
            "metrics": batch_metrics,
            "batch_time_seconds": batch_time_taken
        }
        batch_results.append(batch_info)
        
        # Save each batch's results to a separate JSON file
        batch_filename = os.path.join(output_dir, f"evaluation_results_batch_{start_query}_{end_query}.json")
        with open(batch_filename, "w") as f:
            json.dump(batch_info, f, indent=4)
        print(f"Saved batch {batch_num+1} results to {batch_filename}")

    # Calculate total execution time
    total_time_taken = time.time() - total_start_time
    print(f"\nTotal evaluation process completed in {total_time_taken:.2f} seconds.")
    
    return batch_results

def compute_weighted_average_metrics(folder_path):
    metrics_sum = {"Recall": 0, "MRR": 0, "NDCG": 0}
    total_count = 0

    # Loop through JSON files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            # Read the JSON file
            with open(file_path, "r") as file:
                data = json.load(file)

                # Extract metrics and count
                count = data["metrics"]["count"]
                total_count += count
                
                for metric in metrics_sum.keys():
                    metrics_sum[metric] += data["metrics"][metric] * count

    # Compute weighted averages
    weighted_averages = {metric: metrics_sum[metric] / total_count for metric in metrics_sum}

    # Print the final metrics
    print(f"Final Weighted Recall: {weighted_averages['Recall']:.6f}")
    print(f"Final Weighted MRR: {weighted_averages['MRR']:.6f}")
    print(f"Final Weighted NDCG: {weighted_averages['NDCG']:.6f}")
    print(f"Final Count: {total_count}")

    # Save the results to a JSON file
    output_data = {**weighted_averages, "Total Count": total_count}
    output_path = os.path.join(folder_path, "final_weighted_metrics.json")
    with open(output_path, "w") as out_file:
        json.dump(output_data, out_file, indent=4)

    print(f"Results saved to {output_path}")
    return output_data

### Code to Run

# Load GPU
device = check_gpu()
print(f"GPU: {device}")

# Load data
full_text_data = load_json("/home/ubuntu/CiteRec RAG/global_train_eval_context_list_for_rag.json")

# Define embedding model
embedding_model_name = "BAAI/bge-large-en"  # Supports 1024 tokens
embedding_model = SentenceTransformer(embedding_model_name, device="cuda" if torch.cuda.is_available() else "cpu")

# Chunking parameters
num_of_chunks = 200
num_of_overlaps = 50

print(f"Number of chunks: {num_of_chunks}")
print(f"Number of overlaps: {num_of_overlaps}")

# Batch size
batch_size = 50

# Create FAISS vector store
vectorstore = create_vectorstore(
    text_data=full_text_data, 
    embedding_model=embedding_model, 
    num_of_chunks=num_of_chunks, 
    num_of_overlaps=num_of_overlaps, 
    batch_size=batch_size
    )

# Define index and metadata store
index = vectorstore[0]
metadata_store = vectorstore[1]

# Define reranker model
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_model = CrossEncoder(reranker_model_name, device=device)
print(f"Reranker model loaded: {reranker_model_name}")

# Number of documents to retrieve
number_of_docs = 100

# Varying top k documents to get from rerank
top_k = 10

# Batch size for evaluation
total_queries = 12591 # Do not change at all costs
num_batches = 9 # Do not change once processing starts, need this to correctly resume evaluation
batch_size = total_queries // num_batches # 12591 is the number of queries in the dataset

# Evaluate Reranker
eval_csv_path = "/home/ubuntu/CiteRec RAG/cleaned_acl_global_context_dataset_eval.csv" # Update with your actual file path

# Just context or not
just_context = True

# Batch Evaluation Output Folder
batch_output_dir = f"../{just_context} Context {number_of_docs} Documents Retrieved Top {top_k}_Reranked/{num_of_chunks} Chunks {num_of_overlaps} Overlaps Batch Evaluation Results"

# Create the directory if it does not exist
os.makedirs(batch_output_dir, exist_ok=True)

# Start index for evaluation
batches_done = len([f for f in os.listdir(batch_output_dir) if os.path.isfile(os.path.join(batch_output_dir, f))]) # Or, manually change this to resume from a specific batch
start_index = batch_size * batches_done

print("Other parameters loaded successfully!")

batch_results = compute_evaluation_metrics_in_batches(
    csv_path=eval_csv_path,  # Path to your CSV file
    just_context=just_context, # Just context or not
    embedding_model=embedding_model,  # Replace with your embedding model
    reranker_model=reranker_model,   # Replace with your reranker model
    index=index,            # Replace with your index
    metadata_store=metadata_store,   # Replace with your metadata store
    number_of_docs=number_of_docs,
    top_k=top_k,
    batch_size=batch_size,          # Adjust batch size as needed
    start_index=start_index,         # e.g., resume from the 0th query
    output_dir=batch_output_dir  # Ensure this directory exists or will be created
)

weighted_metrics = compute_weighted_average_metrics(batch_output_dir)