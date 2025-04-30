
import json
import pandas as pd
import numpy as np
import argparse

def main(llm_type, number_of_docs, top_k):

    # Set parameters
    llm_type = llm_type
    number_of_docs = number_of_docs
    top_k = top_k

    # Load reranked indices, retrieved citations, and target citations
    reranked_indices_path = f"/home/ubuntu/LLM Rerankers/Reranked Responses/{llm_type} Structured Responses/{llm_type}_structured-reranked-{number_of_docs}-indices.json"
    retrieved_citations_path =  f"/home/ubuntu/RAG Pipeline/March 27 Citations/1024-256-{number_of_docs}-retrieved_citations.json"
    target_citations_path = "/home/ubuntu/[March 27, 2025] Final Dataset/sampled_final_cleaned_acl_global_context_dataset_eval.csv"

    with open(reranked_indices_path, 'r') as f:
        reranked_indices = json.load(f)
    print(f"Successfully loaded reranked indices with size {len(reranked_indices)}.")

    with open(retrieved_citations_path, 'r') as f:
        retrieved_citations = json.load(f)
    print(f"Successfully loaded retrieved citations with size {len(retrieved_citations)}.")

    evals_df = pd.read_csv(target_citations_path)
    target_citations = evals_df["masked_token_target"].tolist()
    print(f"Successfully loaded target citations with size {len(target_citations)}.")

    # For keeping track of metrics
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for i in range(len(target_citations)):

        print(f"Evaluating query {i+1}...")

        # Set up evaluation per query
        rerank_order = reranked_indices[i]
        original_citations = retrieved_citations[i]
        target_citation = target_citations[i].strip()

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

        # Keep only the highest-ranked document for each unique citation_item
        unique_citation_items = [] # List to track unique citations

        for citation in reranked_citations:

            if citation and citation not in unique_citation_items:
                unique_citation_items.append(citation) # Track this citation
                
            if len(unique_citation_items) == top_k:  # Stop once we have enough unique results
                break

        # Checking whether the target citation is in the unique citations
        citation_target = target_citation
        citation_set = {citation_target}  # Use set for O(1) lookup

        found_match = False

        for index, citation in enumerate(unique_citation_items):

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
        
        if (i + 1) % 15 == 0:

            # Compute and print running metrics
            running_avg_recall = np.mean(recall_scores)
            running_avg_mrr = np.mean(mrr_scores)
            running_avg_ndcg = np.mean(ndcg_scores)
            
            print(f"Running Metrics: Average Recall: {running_avg_recall:.4f} | Average MRR: {running_avg_mrr:.4f} | Average NDCG: {running_avg_ndcg:.4f}")

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

    parser.add_argument("--llm_type", type=str, default="DeepSeek",
                        help="LLM Used for Reranking (default: DeepSeek)")
    parser.add_argument("--number_of_docs", type=int, default=100,
                        help="Number of documents (default: 100)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top K citations to consider (default: 10)")
    
    args = parser.parse_args()
    
    main(args.llm_type, args.number_of_docs, args.top_k)
