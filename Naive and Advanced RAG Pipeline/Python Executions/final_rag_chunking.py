
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
import chromadb
import time
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


### Chunk Texts, Convert to Embedding, and Store into ChromaDB Vector Database
def create_chromadb(text_data, embedding_model, num_of_chunks, num_of_overlaps, chroma_path, batch_size=50):
    """Manually compute embeddings, save in ChromaDB, and store metadata."""

    start_time = time.time()  # Start time

    print("Initializing text splitter...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=num_of_chunks, chunk_overlap=num_of_overlaps)
    
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=chroma_path)

    collection_name = f"final_{num_of_chunks}-{num_of_overlaps}_vectorstore"
    print(f"Everything can be found in: {collection_name}")

    collection = client.get_or_create_collection(name=collection_name)
    
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
                    "citation_item": entry['masked_token_target'],
                    "chunked_text": chunk # Stores only the chunked text
                }
                batch_metadata.append(metadata)

                # Generate embedding
                embedding = embedding_model.encode(chunk, convert_to_numpy=True)
                batch_embeddings.append((embedding, metadata))
                
        # Add batch to ChromaDB
        if batch_embeddings:
            batch_number = i // batch_size + 1
            
            # Store embeddings and metadata in ChromaDB
            collection.add(
                embeddings=[emb[0].tolist() for emb in batch_embeddings],
                metadatas=[emb[1] for emb in batch_embeddings],
                ids=[str(i) for i in range(len(metadata_list), len(metadata_list) + len(batch_embeddings))]
            )
            
            # Append metadata to global list
            metadata_list.extend(batch_metadata)
            
            # Remove embedding from GPU memory to prevent OOM
            del batch_embeddings
            torch.cuda.empty_cache()
            
            batch_end_time = time.time()  # End time for batch processing
            batch_time_taken = batch_end_time - batch_start_time
            print(f"Added batch {batch_number} to ChromaDB. Time taken: {batch_time_taken:.2f} seconds.")
            print(f"Total of {batch_number * batch_size} documents out of {len(text_data)} documents.")
    
    total_time = time.time() - start_time  # Calculate total time
    print(f"Total documents stored in ChromaDB: {len(metadata_list)}")
    print(f"Total execution time: {total_time:.2f} seconds.")  # Print total execution time
    
    return collection



### To Run

device = check_gpu()
print(f"GPU: {device}")

# Load data
full_text_data = load_json("/home/ubuntu/[March 27, 2025] Final Dataset/full_text_sample_cleaned_global_eval_context_for_rag.json")

# Define embedding model
embedding_model_name = "infly/inf-retriever-v1-1.5b"
embedding_model = SentenceTransformer(embedding_model_name, device=device)
print(f"Embedding model loaded: {embedding_model_name}")

# Chunking parameters
num_of_chunks = 1024
num_of_overlaps = 256

print(f"Number of Chunks: {num_of_chunks}")
print(f"Number of Overlaps: {num_of_overlaps}")

# Path to the Chroma DB
chroma_path = "../ChromaDB Storage"

# Create ChromaDB vector store
collection = create_chromadb(
    text_data=full_text_data,
    embedding_model=embedding_model,
    num_of_chunks=num_of_chunks,
    num_of_overlaps=num_of_overlaps,
    chroma_path=chroma_path,
    batch_size=5
)
