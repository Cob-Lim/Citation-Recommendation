import os
from google import genai
from google.genai.types import GenerateContentConfig
import json
import pandas as pd
from langchain.prompts import PromptTemplate
from pydantic import RootModel, ValidationError
from typing import List
from dotenv import load_dotenv
import time
import argparse

load_dotenv(dotenv_path="/home/ubuntu/LLM Rerankers/api_keys.env")

class RerankedIndices(RootModel[List[int]]):
    pass

def generate(filled_prompt):

    # Initialize the Google Gemini client
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash-thinking-exp-01-21"
    contents = filled_prompt
    
    generate_content_config = GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=15000,
        response_mime_type="text/plain",
    )

    max_retries = 5
    initial_delay = 5
    
    for attempt in range(max_retries + 1):
        try:
            full_response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )

            # Default response
            text_response = "[]"
            structured_response = None  # Initialize structured_response

            # Check if the response is valid
            if full_response and full_response.candidates and full_response.candidates[0]:
                # Extracts the candidate response
                candidate = full_response.candidates[0]

                if candidate.content and candidate.content.parts:
                    # Extract text response
                    text_response = ""
                    for each in candidate.content.parts:
                        text_response += each.text

                    # Clean the text response: remove code block markers and strip whitespace
                    cleaned_response = text_response.strip()
                    if cleaned_response.startswith("```") and cleaned_response.endswith("```"):
                        # Remove the outer ``` if present
                        cleaned_response = cleaned_response[3:-3].strip()
                        # Remove language identifier if present (e.g., ```json)
                        if cleaned_response.startswith("json"):
                            cleaned_response = cleaned_response[4:].strip()
                        if cleaned_response.startswith("python"):
                            cleaned_response = cleaned_response[6:].strip()

                    # Attempt to parse the cleaned text response as JSON and validate with Pydantic
                    try:
                        json_output = json.loads(cleaned_response)
                        structured_response = RerankedIndices.model_validate_json(json.dumps(json_output))
                    except (json.JSONDecodeError, ValidationError) as e:
                        print(f"Error parsing or validating JSON: {e}")
                        print(f"Failed to parse JSON. Raw response: '{text_response}'")
                        text_response = "[]"
                        structured_response = None

            return text_response, structured_response

        except genai.errors.ServerError as e:
            if e.code == 503:
                print(f"Model overloaded (attempt {attempt + 1}/{max_retries + 1}). Retrying in {initial_delay * (2 ** attempt)} seconds...")
                if attempt < max_retries:
                    time.sleep(initial_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print("Max retries reached. Skipping this query.")
                    return "[]", None  # Return default values after max retries
            else:
                print(f"An unexpected ServerError occurred: {e}")
                return "[]", None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "[]", None

def main(target_num_of_chunks, target_num_of_overlaps, original_num_of_chunks, original_num_of_overlaps, number_of_docs):
    
    start_time = time.time()

    # Set the parameters
    target_num_of_chunks = target_num_of_chunks
    target_num_of_overlaps = target_num_of_overlaps
    original_num_of_chunks = original_num_of_chunks
    original_num_of_overlaps = original_num_of_overlaps
    number_of_docs = number_of_docs
    number_of_docs_minus_one = number_of_docs - 1

    # Load the retrieved chunks from the specified file
    chunks_file_path = f"/home/ubuntu/RAG Pipeline/March 27 Chunks/{original_num_of_chunks}-{original_num_of_overlaps}-{number_of_docs}-retrieved_chunks.json"

    with open(chunks_file_path, 'r', encoding='utf-8') as f:
        all_retrieved_chunks = json.load(f)

    print(f"Number of Retrieved Chunks: {len(all_retrieved_chunks)}")
    print(f"Number of Documents per Retrieved Chunks: {len(all_retrieved_chunks[0])}")

    # Load the evaluation dataset
    eval_csv_path = "/home/ubuntu/[March 27, 2025] Final Dataset/sampled_final_cleaned_acl_global_context_dataset_eval.csv"
    df = pd.read_csv(eval_csv_path)

    # Extract the 'masked_cit_context' column and convert to list
    masked_contexts = df["masked_cit_context"].tolist()

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

    # Create a list to store the reranked indices
    text_sorted_indices = []
    structured_sorted_indices = []

    # Iterate through the masked contexts and their corresponding retrieved chunks
    for i in range(len(masked_contexts)):

        retrieved_chunks = all_retrieved_chunks[i]
        query = masked_contexts[i]    

        filled_prompt = q_prompt.format(
                number_of_docs=number_of_docs,
                retrieved_chunks=retrieved_chunks,
                query=query,
                number_of_docs_minus_one=number_of_docs_minus_one
            )

        text_reranked_indices, structured_reranked_indices = generate(filled_prompt)
        text_sorted_indices.append(text_reranked_indices)
        structured_sorted_indices.append(structured_reranked_indices)
        
        print(f"Done reranking query {i+1}. Elapsed time since start: {time.time() - start_time:.2f} seconds.")

        # Save every 15 responses
        if (i + 1) % 15 == 0:

            with open(f"Gemini_text-reranked-{number_of_docs}-indices.json", "w") as f:
                json.dump(text_sorted_indices, f, indent=4)
            print(f"Saved {i+1} reranked responses to Gemini_text-reranked-{number_of_docs}-indices.json")

            with open(f"Gemini_structured-reranked-{number_of_docs}-indices.json", "w") as f:
                # Extract the root value (the list of indices) before dumping
                serializable_indices = [item.root if item else None for item in structured_sorted_indices]
                json.dump(serializable_indices, f, indent=4)
            print(f"Saved {i+1} reranked responses to Gemini_structured-reranked-{number_of_docs}-indices.json")

        time.sleep(3)  # Sleep to ensure the model is ready for the next response (copium)
    
    # Last save
    with open(f"Gemini_text-reranked-{number_of_docs}-indices.json", "w") as f:
        json.dump(text_sorted_indices, f, indent=4)
    print(f"Saved all reranked responses to Gemini_text-reranked-{number_of_docs}-indices.json")

    with open(f"Gemini_structured-reranked-{number_of_docs}-indices.json", "w") as f:
        # Extract the root value (the list of indices) before dumping
        serializable_indices = [item.root if item else None for item in structured_sorted_indices]
        json.dump(serializable_indices, f, indent=4)
    print(f"Saved all reranked responses to Gemini_structured-reranked-{number_of_docs}-indices.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Thnking Reranking.")

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