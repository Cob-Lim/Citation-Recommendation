from langchain.prompts import PromptTemplate
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
#from pydantic import BaseModel

start_time = time.time()

#class CitationsTitles(BaseModel):
#    citations: list[str]
#    titles: list[str]

q_template = (
    "Citation context:\n"
    "{citation_context}\n\n"
    "The query above came from a research paper in the ACL Anthology titled \"{citing_title}\".\n\n"
    "Suggest an appropriate citation for the citation that was masked in the text. Return 10 possible citations, ranked by relevance to the query, with the most relevant first. Make sure the citations are unique.\n"
    "Additionally, return the corresponding titles of each citation. There is no need to explain how you arrived at the answer. Return both of the answers as a Python list of strings.\n\n"
    "Please use Google Search to find the most relevant citations for this query. Make sure to search for each citation to verify its accuracy.\n"
    "If you do not know the answer, simply return an empty list of strings for both citations and titles.\n\n"
    "Format:\n"
    "- One author: Author, 2025\n"
    "- Two authors: Author and Coauthor, 2025\n"
    "- More than two authors: Author et al., 2025\n\n"
    "Return the result as a JSON object with two keys:\n"
    " - \"citations\": a list of 10 formatted citation strings\n"
    " - \"titles\": a list of 10 corresponding paper titles in the same order\n\n"
    "Example Answer:\n"
    "{{\n"
    "  \"citations\": [\n"
    "    \"Gao and Vogel, 2008\",\n"
    "    \"Hong et al., 2009\",\n"
    "    \"Felice et al., 2014\",\n"
    "    \"Kirschenbaum and Wintner, 2009\",\n"
    "    \"Schwenk et al., 2009\",\n"
    "    \"Ananthakrishnan et al., 2011\",\n"
    "    \"Nicolai et al., 2015\",\n"
    "    \"Neubig et al., 2011\",\n"
    "    \"Vilar et al., 2007\",\n"
    "    \"Lambert et al., 2011\"\n"
    "  ],\n"
    "  \"titles\": [\n"
    "    \"Fast and Adaptive Online Training of Feature-Rich Translation Models\",\n"
    "    \"Fast and Adaptive Online Training of Feature-Rich Translation Models\",\n"
    "    \"Edinburgh's Machine Translation Systems for European Language Pairs\",\n"
    "    \"Wider Context by Using Bilingual Language Models in Machine Translation\",\n"
    "    \"Minimum Translation Modeling with Recurrent Neural Networks\",\n"
    "    \"A Systematic Comparison of Phrase Table Pruning Techniques\",\n"
    "    \"Improved Models of Distortion Cost for Statistical Machine Translation\",\n"
    "    \"Dirt Cheap Web-Scale Parallel Text from the Common Crawl\",\n"
    "    \"On the Use of Comparable Corpora to Improve SMT Performance\",\n"
    "    \"The Cunei Machine Translation Platform for WMT '10\"\n"
    "  ]\n"
    "}}"
)

q_prompt = PromptTemplate(
    input_variables=["citation_context", "citing_title"],
    template=q_template
)


def generate(filled_prompt):
    client = genai.Client(
        api_key="" # Replace with your own Google Gemini API key
    )

    model = "gemini-2.0-flash"
    contents = filled_prompt
    
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )

    generate_content_config = GenerateContentConfig(
        temperature=0.1,
        tools=[google_search_tool],
        response_mime_type="text/plain"
    )

    # Gets the response of the model
    full_response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    # Default response
    text_response = "No response generated from the model."
    search_proof = "No response generated from the model."

    # Check if the response is valid
    if full_response and full_response.candidates and full_response.candidates[0]:

        # Extracts the candidate response
        candidate = full_response.candidates[0]

        if candidate.content and candidate.content.parts:
                # Extract text response
                text_response = ""
                for each in candidate.content.parts:
                    text_response += each.text

        # Extracts the search proof
        grounding = candidate.grounding_metadata

        if grounding and grounding.search_entry_point and grounding.search_entry_point.rendered_content:
            # The full HTML string
            rendered_content = grounding.search_entry_point.rendered_content

            # Parse it
            soup = BeautifulSoup(rendered_content, "html.parser")

            # Find the div containing all the chips (links)
            carousel_div = soup.find("div", class_="carousel")

            # Extract the div with class "carousel"
            search_proof = str(carousel_div)
        
    # Return the complete response text
    return text_response, search_proof


eval_csv_path = "/home/ubuntu/[March 27, 2025] Final Dataset/sampled_final_cleaned_acl_global_context_dataset_eval.csv"
df = pd.read_csv(eval_csv_path)

# Extract the 'masked_cit_context' column and convert to list
masked_contexts = df["masked_cit_context"].tolist()
citing_titles = df["citing_title"].tolist()

text_responses = []
search_proofs = []

print("Starting to generate responses...")

for i in range(len(masked_contexts)):
    filled_prompt = q_prompt.format(
        citation_context=masked_contexts[i],
        citing_title=citing_titles[i]
    )

    text_response, search_proof = generate(filled_prompt)

    print(f"Response {i+1} generated. Elapsed time since start: {time.time() - start_time:.2f} seconds.")

    text_responses.append(text_response)
    search_proofs.append(search_proof)

    # Save every 25 responses
    if (i + 1) % 25 == 0:

        with open("QCTtoC_text-responses.json", "w") as f:
            json.dump(text_responses, f, indent=4)
        print(f"Saved {i+1} text responses to QCTtoC_text-responses.json")

        with open("QCTtoC_search-proofs-2.json", "w") as f:
            json.dump(search_proofs, f, indent=4)
        print(f"Saved {i+1} search proofs to QCTtoC_search-proofs.json")

    time.sleep(6)  # Sleep to avoid rate limiting


# Final save (in case len(masked_contexts) is not a multiple of 25)
with open("QCTtoC_text-responses.json", "w") as f:
    json.dump(text_responses, f, indent=4)
print(f"Saved {i+1} text responses to QCTtoC_text-responses.json")

with open("QCTtoC_search-proofs.json", "w") as f:
    json.dump(search_proofs, f, indent=4)
print(f"Saved {i+1} search proofs to QCTtoC_search-proofs.json")