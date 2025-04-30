import json
import pandas as pd
from langchain.prompts import PromptTemplate
from openai import OpenAI
import time
import argparse
import os
from pydantic import RootModel, ValidationError
from typing import List
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/ubuntu/LLM Rerankers/api_keys.env")

class RerankedIndices(RootModel[List[int]]):
    pass

def generate_reranked_indices(client, retrieved_chunks, query, number_of_docs, number_of_docs_minus_one):

    # Define template for the prompt
    q_template = (
        "Given this list of retrieved chunks (total: {number_of_docs}):\n"
        "{retrieved_chunks}\n\n"

        "And given the query:\n"
        "{query}\n\n"

        "Rerank the chunks above according to their relevance to the query. Rank them from most relevant to least relevant.\n"
        "After reranking, output the current order of their indices based on the original order. Format it as a Python list of integers. For example, [9, 7, 6, 5, 3, 1, 8, 2, 4, 0].\n"
        "There is no need to explain your answers. You can simply output the Python list of indices.\n\n"

        "Make sure that:\n"
        "- The list contains only integers between 0 and {number_of_docs_minus_one}.\n"
        "- The length of the list is exactly {number_of_docs}."
    )

    q_prompt = PromptTemplate(
        input_variables=["number_of_docs", "retrieved_chunks", "query", "number_of_docs_minus_one"],
        template=q_template
    )

    # Fills in the values for the prompt
    filled_prompt = q_prompt.format(
            number_of_docs=number_of_docs,
            retrieved_chunks=retrieved_chunks,
            query=query,
            number_of_docs_minus_one=number_of_docs_minus_one
        )

    # Generates a response for the prompt with error handling
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating the relevance of text passages to a given query. Your task is to rerank the provided list of chunks based on their relevance."},
                {"role": "user", "content": filled_prompt},
            ],
            stream=False,
            temperature=0.1
        )

        text_response = response.choices[0].message.content

        # Default structured response
        structured_response = None

        # Clean the text response (similar to your Gemini code)
        cleaned_response = text_response.strip()
        if cleaned_response.startswith("```") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()
            if cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[4:].strip()
            if cleaned_response.startswith("python"):
                cleaned_response = cleaned_response[6:].strip()

        # Attempt to parse the cleaned text response as JSON and validate with Pydantic
        try:
            json_output = json.loads(cleaned_response)
            structured_response = RerankedIndices.model_validate_json(json.dumps(json_output))
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Error parsing or validating JSON from DeepSeek: {e}")
            print(f"Failed to parse JSON. Raw response from DeepSeek: '{text_response}'")
            text_response = "[]"
            structured_response = None

        return text_response, structured_response

    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
        text_response = "[]"
        structured_response = None
        return text_response, structured_response
    

def main(target_num_of_chunks, target_num_of_overlaps, original_num_of_chunks, original_num_of_overlaps, number_of_docs):

    start_time = time.time()

    # Initialize the OpenAI client
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    # Set the parameters
    target_num_of_chunks = target_num_of_chunks
    target_num_of_overlaps = target_num_of_overlaps
    original_num_of_chunks = original_num_of_chunks
    original_num_of_overlaps = original_num_of_overlaps
    number_of_docs = number_of_docs
    number_of_docs_minus_one = number_of_docs - 1

    # Load the retrieved chunks from the specified file
    chunks_file_path = f"/home/ubuntu/RAG Pipeline/March 27 Rep Chunks/{target_num_of_chunks}-{target_num_of_overlaps}-from-{original_num_of_chunks}-{original_num_of_overlaps}-{number_of_docs}-representative_chunks.json"

    with open(chunks_file_path, 'r', encoding='utf-8') as f:
        all_retrieved_chunks = json.load(f)

    print(f"Number of Retrieved Chunks: {len(all_retrieved_chunks)}")
    print(f"Number of Documents per Retrieved Chunks: {len(all_retrieved_chunks[0])}")

    # Load the evaluation dataset
    eval_csv_path = "/home/ubuntu/[March 27, 2025] Final Dataset/sampled_final_cleaned_acl_global_context_dataset_eval.csv"
    df = pd.read_csv(eval_csv_path)

    # Extract the 'masked_cit_context' column and convert to list
    masked_contexts = df["masked_cit_context"].tolist()

    # Create a list to store the reranked indices
    text_sorted_indices = []
    structured_sorted_indices = []

    # Iterate through the masked contexts and their corresponding retrieved chunks
    for i in range(len(masked_contexts)):

        retrieved_chunks = all_retrieved_chunks[i]
        query = masked_contexts[i]

        text_reranked_indices, structured_reanked_indices = generate_reranked_indices(client, retrieved_chunks, query, number_of_docs, number_of_docs_minus_one)
        text_sorted_indices.append(text_reranked_indices)
        structured_sorted_indices.append(structured_reanked_indices)

        print(f"Done reranking query {i+1}. Elapsed time since start: {time.time() - start_time:.2f} seconds.")

        # Save every 15 responses
        if (i + 1) % 15 == 0:

            with open(f"DeepSeek_text-reranked-{number_of_docs}-indices.json", "w") as f:
                json.dump(text_sorted_indices, f, indent=4)
            print(f"Saved {i+1} reranked responses to DeepSeek_text-reranked-{number_of_docs}-indices.json")

            with open(f"DeepSeek_structured-reranked-{number_of_docs}-indices.json", "w") as f:
                # Extract the root value (the list of indices) before dumping
                serializable_indices = [item.root if item else None for item in structured_sorted_indices]
                json.dump(serializable_indices, f, indent=4)
            print(f"Saved {i+1} reranked responses to DeepSeek_structured-reranked-{number_of_docs}-indices.json")

        time.sleep(3)  # Sleep to ensure the model is ready for the next response (copium)
    
    # Last save
    with open(f"DeepSeek_text-reranked-{number_of_docs}-indices.json", "w") as f:
        json.dump(text_sorted_indices, f, indent=4)
    print(f"Saved all reranked responses to DeepSeek_text-reranked-{number_of_docs}-indices.json")

    with open(f"DeepSeek_structured-reranked-{number_of_docs}-indices.json", "w") as f:
        # Extract the root value (the list of indices) before dumping
        serializable_indices = [item.root if item else None for item in structured_sorted_indices]
        json.dump(serializable_indices, f, indent=4)
    print(f"Saved all reranked responses to DeepSeek_structured-reranked-{number_of_docs}-indices.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek Chat V3 Reranking (Not Fine-Tuned).")

    parser.add_argument("--target_num_of_chunks", type=int, default=256,
                        help="Number of chunks (default: 256)")
    parser.add_argument("--target_num_of_overlaps", type=int, default=64,
                        help="Number of overlaps (default: 64)")
    parser.add_argument("--original_num_of_chunks", type=int, default=1024,
                        help="Original number of chunks (default: 1024)")
    parser.add_argument("--original_num_of_overlaps", type=int, default=256,
                        help="Original number of overlaps (default: 256)")
    parser.add_argument("--number_of_docs", type=int, default=100,
                        help="Number of documents (default: 100)")
    
    args = parser.parse_args()
    
    main(args.target_num_of_chunks, args.target_num_of_overlaps, args.original_num_of_chunks, args.original_num_of_overlaps, args.number_of_docs)