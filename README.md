# Citation Recommendation with CiteBART, RAG, and Agentic Retrieval

This project explores multiple **Citation Recommendation** strategies using both classic and modern **Retrieval-Augmented Generation (RAG)** techniques. It benchmarks **CiteBART (SOTA)** against variations of **RAG pipelines** (Naive, Advanced), an **Agentic Recommender using Gemini + Google Search**, and a **Hybrid Two-Stage Retrieval** system. Evaluation was conducted using ground truth citation contexts to assess relevance and performance.

---

## Pipeline Overview

### 1. Dataset Preparation
- Utilized the same dataset as CiteBART, including ACL-200, a curated benchmark derived from ACL Anthology papers.
- ACL-200 contains citation contexts paired with their corresponding references and is widely used to evaluate the performance of citation recommendation models.

### 2. Chunking Strategy
- Used LangChain’s RecursiveCharacterTextSplitter to experiment with various chunk sizes and overlap configurations.
- Found that 1024-token chunks with a 256-token overlap provided the best performance in terms of retrieval accuracy and context preservation.
- 
### 3. Embedding
- Used Infly’s INF-Retriever-v1-1.5B embedding model to generate dense vector representations of text chunks.
- Stored embeddings in vector databases such as ChromaDB for efficient similarity-based retrieval.
- 
### 4. Retrieval & Generation Approaches

#### CiteBART (SOTA)
- Fine-tuned transformer model trained specifically for citation generation.
- Input: citation context; Output: most probable citation string.
- Served as the baseline for comparison.
  
#### Naive RAG
- Retrieved top-k most similar chunks using vector similarity.
- Extracted citations directly from the retrieved chunks without passing them to an LLM; however, citation generation via LLM can optionally be added.

#### Advanced RAG
- Enhanced retrieval with reranking via **FlagEmbedding** or **MXBAI Rerankers**.
- Extracted citations from the top reranked chunks without LLM generation; LLM-based citation generation remains a configurable option.

#### Agentic Retrieval (Gemini + Google Search)
- Used **Gemini 2.0 Flash** to trigger structured web queries.
- Queried **Google Search** for relevant articles, filtered results, and used Gemini again to summarize and generate the citation.

#### Two-Stage Hybrid (Advanced RAG + Gemini)
- Stage 1: Retrieved results using Advanced RAG from internal documents.
- Stage 2: Retrieved results via Gemini 2.0 Flash + Google Search as external fallback.
- Both result sets were combined by merging similar citations, and the remaining entries were appended based on their original relevance scores.
- An LLM-based reranker can optionally be introduced in the future to refine final citation relevance ordering.
- 
### 5. Evaluation

- Used a benchmark dataset with **citation context + ground truth citation**.
- Evaluated with:
  - **Recall@k**
  - **Mean Reciprocal Rank (MRR)@k**
 
---

## Tech Stack

| Component              | Tool/Service                            |
|------------------------|-----------------------------------------|
| Dataset                | ACL-200 (from CiteBART)                 |
| Embeddings             | `INF-Retriever-v1-1.5B`                 |
| RAG Framework          | LangChain                               |
| Vector Store           | ChromaDB                                |
| Reranker               | FlagEmbedding, MXBAI Reranker           |
| LLM Reranker           | Gemini 2.0 Flash Thinking, DeepSeek V3  |
| Citation Model         | CiteBART (fine-tuned)                   |
| External Search Agent  | Gemini 2.0 Flash + Google Search        |
| Evaluation             | Recall, MRR                             |

---

## Key Results Summary

![Recall Across Architectures](Images/Recall%20Across%20Architectures.png)
![MRR Across Architectures](Images/MRR%20Across%20Architectures.png)

- The **Two-Stage Hybrid (Advanced RAG + Gemini)** pipeline achieved the highest performance across all methods, outperforming others in both Recall and Mean Reciprocal Rank (MRR).
---

## Disclaimer
- All experiments were run on an Amazon EC2 g5.xlarge instance, with the exception of external API calls, which were executed outside the instance.
