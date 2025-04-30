import json
import torch
import faiss
import os
import pickle
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

### Chunk Texts, Convert to Emebdding, and Store into FAISS Vector Database
def create_vectorstore(text_data, embedding_model, num_of_chunks, num_of_overlaps, batch_size=50):
    """Manually compute embeddings, save FAISS index, and store metadata in a Pickle file."""

    start_time = time.time() # Start time

    # Define file paths
    faiss_pickle_path = f"{num_of_chunks}-chunks_{num_of_overlaps}-overlaps_faiss_store.pkl"
    
    # Check if FAISS index already exists
    if os.path.exists(faiss_pickle_path):
        print("Loading existing FAISS index and metadata...")
        with open(faiss_pickle_path, "rb") as f:
            data = pickle.load(f)
        print("FAISS index and metadata loaded successfully!")
        return data["vectorstore"], data["metadata_list"]

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


### Code to Run
device = check_gpu()
print(f"GPU: {device}")

# Load data
full_text_data = load_json("/home/ubuntu/CiteRec RAG/global_train_eval_context_list_for_rag.json")

# Define embedding model
embedding_model_name = "BAAI/bge-large-en"  # Supports 1024 tokens
embedding_model = SentenceTransformer(embedding_model_name, device="cuda" if torch.cuda.is_available() else "cpu")

# Chunking parameters
num_of_chunks = 128
num_of_overlaps = 32

print(f"Number of Chunks: {num_of_chunks}")
print(f"Number of Overlaps: {num_of_overlaps}")

# Create FAISS vector store
vectorstore = create_vectorstore(
    text_data=full_text_data,
    embedding_model=embedding_model,
    num_of_chunks=num_of_chunks,
    num_of_overlaps=num_of_overlaps,
    batch_size=50
    )