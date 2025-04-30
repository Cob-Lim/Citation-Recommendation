
import json
import pandas as pd
import numpy as np
import torch
import time
from sentence_transformers import SentenceTransformer
from mxbai_rerank import MxbaiRerankV2
import argparse


print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)         # Should return 12.1
print(torch.backends.cudnn.version())  # Should print cuDNN version

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

# Apply reranking (Gemini or DeepSeek) to the retrieved citations
def apply_rerank_llm(reranked_vdb_documents_entry, retrieved_vdb_citations_entry, number_of_docs):
    """Apply reranking to the retrieved citations."""

    # Set up evaluation per query
    rerank_order = reranked_vdb_documents_entry
    original_citations = retrieved_vdb_citations_entry
    
    # Case 1: Rerank order is None (or empty)
    if rerank_order is None:
        print("Case 1")
        reranked_citations = original_citations  # Keep the original order

    # Case 2: Rerank order is longer than the number of documents
    elif len(rerank_order) > number_of_docs:
        print("Case 2")
        rerank_order = [index for index in rerank_order if index < number_of_docs] # Remove excess indices

        reranked_citations = [original_citations[i] for i in rerank_order] # Rerank the order of the original citations according to the reranked indices
    
    # Case 3: Rerank order contains the indices with value greather than the number_of_docs
    elif any(index >= number_of_docs for index in rerank_order):
        print("Case 3")

        unexpected_indices = []
        unexpected_values = []

        for i in range(len(rerank_order)): # Iterate through the entire list to find unexpected indices
            if rerank_order[i] >= number_of_docs:
                unexpected_indices.append(i)
                unexpected_values.append(rerank_order[i])

        set_order = set(rerank_order)
        range_of_docs = set(range(number_of_docs))
        missing_indices = list(range_of_docs - set_order)

        # Sort both lists of unexpected values and missing values
        sorted_unexpected_values = sorted(unexpected_values)
        sorted_missing_indices = sorted(missing_indices) # Ensure missing_indices is sorted

        # Create a mapping based on the sorted order
        mapping = {}
        for i in range(len(sorted_unexpected_values)):
            if i < len(sorted_missing_indices):
                mapping[sorted_unexpected_values[i]] = sorted_missing_indices[i]
            else:
                print("Warning: Not enough missing values to map all unexpected values.")
                break


        # Apply the mapping to change the unexpected_values with the original_values at the unexpected_indices
        mapped_rerank_order = list(rerank_order) # Create a copy to modify

        for i in range(len(unexpected_indices)):
            index_to_change = unexpected_indices[i]
            original_value = mapped_rerank_order[index_to_change]
            if original_value in mapping:
                mapped_rerank_order[index_to_change] = mapping[original_value]
            else:
                print(f"Warning: No mapping found for value {original_value} at index {index_to_change}.")

        reranked_citations = [original_citations[i] for i in mapped_rerank_order] # Rerank the order of the original citations according to the reranked indices
        
    # All good
    else:
        print("All good")
        reranked_citations = [original_citations[i] for i in rerank_order] # Rerank the order of the original citations according to the reranked indices
    
    return reranked_citations

# Apply reranking (Mxbai) to the retrieved citations
def apply_rerank_mxbai(query, retrieved_vdb_documents_entry, retrieved_vdb_citations_entry, reranker_model):
    """Apply reranking to the retrieved citations."""

    reranked_citations = []

    try:
        with torch.no_grad():
            results = reranker_model.rank(query, retrieved_vdb_documents_entry)

        # Sort results by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)
        sorted_indices = [result.index for result in results]

    except torch.cuda.OutOfMemoryError:
        print("OOM during reranking. Returning original chunks.")
        sorted_indices = [i for i in range(len(retrieved_vdb_documents_entry))]

    # Sort the original results based on the sorted indices
    for i in sorted_indices:
        reranked_citations.append(retrieved_vdb_citations_entry[i])

    return reranked_citations

# Add common entries from reranked citations to the final list
def add_common_entries(reranked_citations, google_citations):
    """Add common entries from reranked citations to the final list."""
    final_list = []

    for citation in google_citations:
        if citation in reranked_citations:
            final_list.append(citation)
            
    return final_list


# Ensure unique citations from both sources
def ensure_unique_citations(list_of_citations, common_citations, top_k):
    """Ensure that the list of citations is unique."""
    unique_citations = []

    for citation in list_of_citations:

        if citation not in common_citations and citation not in unique_citations:
            unique_citations.append(citation) # Track this citation
            
        if len(unique_citations) == top_k:  # Stop once we have enough unique results
            break

    return unique_citations


# Fill in the list until it reaches top k
def populate_final_citations(final_list, unique_vector_db_citations, unique_google_citations, top_k):
    """Populate the final list of citations ensuring a mix from both sources."""

    needed = top_k - len(final_list)
    vector_db_count = len(unique_vector_db_citations)
    google_count = len(unique_google_citations)

    vector_db_index = 0
    google_index = 0

    for _ in range(needed):

        can_add_vector_db = vector_db_index < vector_db_count
        can_add_google = google_index < google_count

        if can_add_vector_db and can_add_google:
            if len(final_list) % 2 == 0:  # Alternate: Even index -> Vector DB
                final_list.append(unique_vector_db_citations[vector_db_index])
                vector_db_index += 1
            else:  # Alternate: Odd index -> Google
                final_list.append(unique_google_citations[google_index])
                google_index += 1
        elif can_add_vector_db:
            final_list.append(unique_vector_db_citations[vector_db_index])
            vector_db_index += 1
        elif can_add_google:
            final_list.append(unique_google_citations[google_index])
            google_index += 1
        else:
            break  # No more citations available

    # Prioritize Google for odd remaining
    remaining = top_k - len(final_list)
    if remaining == 1 and google_index < google_count and (not final_list or final_list[-1] not in unique_google_citations):
        final_list.append(unique_google_citations[google_index])
    
    return final_list


# Run the evaluation
def eval_citations(target_citation, final_list_of_citations, recall_scores, mrr_scores, ndcg_scores):
        
    # Checking whether the target citation is in the unique citations
    citation_set = {target_citation}  # Use set for O(1) lookup

    found_match = False

    for index, citation in enumerate(final_list_of_citations):

        if citation.strip() in citation_set:
            recall_scores.append(1)
            mrr_scores.append(1 / (index + 1))
            ndcg_scores.append(1 / np.log2(index + 2))
            found_match = True
            break  # Exit the inner loop once a match is found

    if not found_match:
        recall_scores.append(0)
        mrr_scores.append(0)
        ndcg_scores.append(0)


def main(number_of_docs, top_k):
    """Main function to run the evaluation."""

    start_time = time.time()
    print("Starting evaluation...")

    # Check GPU
    device = check_gpu()
    print(f"Using device: {device}")

    # Define parameters
    number_of_docs = number_of_docs
    top_k = top_k

    # Load the scores
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    # Load google search results
    google_search_results_path = "/home/ubuntu/LLM and Search Experiments/Cleaned Text Responses/QTTTAtoC_cleaned-text-responses.json"

    google_search_results = load_json(google_search_results_path)
    google_search_citations = google_search_results["citations"]

    # Load citation targets
    eval_csv_path = "/home/ubuntu/[March 27, 2025] Final Dataset/sampled_final_cleaned_acl_global_context_dataset_eval.csv"

    eval_df = pd.read_csv(eval_csv_path)
    citation_contexts = eval_df["masked_cit_context"].tolist()
    citation_targets = eval_df['masked_token_target'].tolist()
    print("Citation contexts loaded successfully.")
    print("Citation targets loaded successfully.")

    if number_of_docs == 20 or number_of_docs == 35:
        reranked_vdb_documents_path = f"/home/ubuntu/LLM Rerankers/Reranked Responses/Gemini Structured Responses/Gemini_structured-reranked-{number_of_docs}-indices.json"
        retrieved_vdb_citations_path =  f"/home/ubuntu/RAG Pipeline/March 27 Citations/1024-256-{number_of_docs}-retrieved_citations.json"
    
    elif number_of_docs == 100:
        reranked_vdb_documents_path = f"/home/ubuntu/LLM Rerankers/Reranked Responses/DeepSeek Structured Responses/DeepSeek_structured-reranked-{number_of_docs}-indices.json"
        retrieved_vdb_citations_path =  f"/home/ubuntu/RAG Pipeline/March 27 Citations/1024-256-{number_of_docs}-retrieved_citations.json"
    else:
        reranked_vdb_documents_path = f"/home/ubuntu/RAG Pipeline/March 27 Chunks/1024-256-{number_of_docs}-retrieved_chunks.json"
        retrieved_vdb_citations_path =  f"/home/ubuntu/RAG Pipeline/March 27 Citations/1024-256-{number_of_docs}-retrieved_citations.json"

    # Load data
    reranked_vdb_documents = load_json(reranked_vdb_documents_path)
    retrieved_vdb_citations = load_json(retrieved_vdb_citations_path)

    # Load reranker model when needed
    if not(number_of_docs == 20 or number_of_docs == 35 or number_of_docs == 100):
        reranker_model_name = "mixedbread-ai/mxbai-rerank-base-v2" # change to mxbai-rerank-large-v2 if you want
        reranker_model = MxbaiRerankV2(reranker_model_name, device=device)
        print("Reranker model loaded successfully.")
    else:
        print("LLM Reranker being evaluated.")

    # Process each query
    for i in range(len(citation_targets)):

        print(f"Processing query {i + 1}...")

        reranked_vdb_documents_entry = reranked_vdb_documents[i]
        retrieved_vdb_citations_entry = retrieved_vdb_citations[i]

        citations_from_google = google_search_citations[i]

        query = citation_contexts[i]
        target_citation = citation_targets[i]

        # Apply the reranking
        if number_of_docs == 20 or number_of_docs == 35 or number_of_docs == 100:
            reranked_citations = apply_rerank_llm(reranked_vdb_documents_entry, retrieved_vdb_citations_entry, number_of_docs)
        else:
            reranked_citations = apply_rerank_mxbai(query, reranked_vdb_documents_entry, retrieved_vdb_citations_entry, reranker_model)

        # Add common entries to the final list
        temp_final_citations = add_common_entries(reranked_citations, citations_from_google)

        # Keep only the highest-ranked document for each unique citation in the vdb that are not in the common entires
        vdb_unique_citation_items = ensure_unique_citations(reranked_citations, temp_final_citations, top_k)

        # Keep only the highest-ranked document for each unique citation in the google search that are not in the common entries
        unique_citations_from_google = ensure_unique_citations(citations_from_google, temp_final_citations, top_k)
        
        # Complete the final list of citations
        final_list_of_citations = populate_final_citations(temp_final_citations, vdb_unique_citation_items, unique_citations_from_google, top_k)
        
        # Compare masked citation with the final list of citations
        eval_citations(target_citation, final_list_of_citations, recall_scores, mrr_scores, ndcg_scores)

        if (i + 1) % 15 == 0:
                
                print(f"Processed {i + 1} queries out of {len(citation_targets)}...")

                # Compute and print running metrics
                running_avg_recall = np.mean(recall_scores)
                running_avg_mrr = np.mean(mrr_scores)
                running_avg_ndcg = np.mean(ndcg_scores)
                
                print(f"Running Metrics: Average Recall: {running_avg_recall:.4f} | Average MRR: {running_avg_mrr:.4f} | Average NDCG: {running_avg_ndcg:.4f}")

        print(f"Finished processing query. Time elapsed since the start: {time.time() - start_time:.2f} seconds")

    # Aggregate metrics over all queries
    count = len(recall_scores)
    overall_metrics = {
        "Recall": np.mean(recall_scores) if count > 0 else 0,
        "MRR": np.mean(mrr_scores) if count > 0 else 0,
        "NDCG": np.mean(ndcg_scores) if count > 0 else 0,
        "count": count
    }
        
    print("Overall Metrics:")
    print(overall_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating LLM Rerankers")

    parser.add_argument("--number_of_docs", type=int, default=100,
                        help="Number of documents (default: 100)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top K citations to consider (default: 10)")
    
    args = parser.parse_args()
    
    main(args.number_of_docs, args.top_k)