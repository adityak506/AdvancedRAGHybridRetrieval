# 📘 Simple & Clear Revision Notes: Advanced Hybrid RAG

> **Topic:** Advanced RAG — Hybrid Retrieval (Dense Semantic Search + Sparse Keyword Search)  
> **Goal:** Understand the whole project easily without getting lost in complicated math or heavy jargon.

---

## 🌟 1. What is This Project in 2 Minutes?

When building a question-answering AI (RAG = Retrieval-Augmented Generation), the AI needs to read your documents first, find the right paragraphs, and then write an answer based on those paragraphs.

Normally, people use **Vector Search (Embeddings)**. But vector search often fails when users ask for exact policy numbers, IDs, or specific words.

This project solves that by building a **Hybrid Search Engine**:
1. **Dense Search (FAISS):** Searches by **meaning/concept** (like a smart assistant).
2. **Sparse Search (BM25):** Searches by **exact keywords** (like `Ctrl + F`).
3. **Hybrid Search:** Combines both to get the most accurate, reliable answer.

It supports both **OpenAI** and **Google Gemini**, automatically manages embedding sizes, and runs all 3 search methods side-by-side so you can compare their answers.

---

## 💡 2. The Big "Why": Why Do We Need Hybrid Search?

Imagine you have an insurance document:

| Search Method | How it Works | When it Works Great | Where it Fails |
| :--- | :--- | :--- | :--- |
| **Dense Search (FAISS)** | Converts text into numbers (vectors) based on **meaning**. | Paraphrased questions (e.g., *"doctor consultation"* finds *"physician visit"*). | Exact codes, policy numbers (e.g., searching for `AUTO-987654` or `CLM-2025-1010` gets confused). |
| **Sparse Search (BM25)** | Matches **exact words** and calculates word importance. | Exact terms, policy names, serial numbers, error codes, phone numbers. | Synonyms (e.g., searching *"automobile"* will miss text that only says *"car"*). |
| **Hybrid Search (Both)** | Runs **both** searches and blends their scores. | **Best of both worlds!** Handles both concept questions and exact keyword queries. | Requires balancing weights and normalizing scores. |

---

## 🗺️ 3. How the Whole System Works (Step-by-Step)

Here is the exact journey from raw documents to the final answer:

```
Step 1: Load Files
[ policy_terms.txt & claim_procedure.txt ]
                  │
                  ▼
Step 2: Cut into Chunks (utils/loader.py)
[ 8 small chunks (500 characters each, with 100 character overlap) ]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
Step 3A: FAISS Index      Step 3B: BM25 Setup
(utils/retriever_faiss.py) (utils/retriever_bm25.py)
Converts chunks into      Counts words and builds
vector embeddings         an in-memory keyword list
(OpenAI or Gemini)        
        │                   │
        └─────────┬─────────┘
                  ▼
Step 4: User Asks a Question (in app.py)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
FAISS finds top chunks    BM25 finds top chunks
by meaning                by exact words
        │                   │
        └─────────┬─────────┘
                  ▼
Step 5: Blend Scores (utils/hybrid_retriever.py)
Normalize both scores to 0-1 scale.
Combined Score = (0.6 × FAISS) + (0.4 × BM25)
Pick the highest scoring chunks!
                  │
                  ▼
Step 6: Send to LLM
Put chunks in context:
"Answer only using the provided context. If not present, say you don't know."
                  │
                  ▼
Final Grounded Answer!
```

---

## 📂 4. Walkthrough of Every File in the Project

### ⚙️ [config.yaml](file:///d:/IITPatna/GenAI-DEV-Projects/Advanced-RAG-HybridRetrieval/config.yaml) — The Control Center
This is where you change settings without touching any Python code:
- **`llm.provider`:** Choose `"openai"` or `"gemini"`.
- **`llm.temperature`:** Set to `0.3` (low temperature = factual, creative mistakes minimized).
- **`embedding`:**
  - OpenAI uses `text-embedding-3-small` (creates 1,536 numbers per chunk).
  - Gemini uses `models/text-embedding-004` (creates 768 numbers per chunk).
- **`vectordb`:**
  - `./data/embeddings/faiss_openai` for OpenAI.
  - `./data/embeddings/faiss_gemini` for Gemini.
  - *Why separate folders?* You cannot mix 1,536-number vectors with 768-number vectors! Separate folders keep them safe.
- **`hybrid.weights`:** `[0.6, 0.4]` means 60% importance to meaning (FAISS) and 40% importance to exact words (BM25).
- **`bm25.k1` & `b`:** Parameters for keyword ranking (`k1=1.5`, `b=0.75`).

---

### 📄 [utils/loader.py](file:///d:/IITPatna/GenAI-DEV-Projects/Advanced-RAG-HybridRetrieval/utils/loader.py) — Reading & Cutting Documents
- **What it does:**
  1. Looks inside `data/raw/insurance_docs/`.
  2. Reads `.txt` files using `TextLoader` and `.pdf` files using `PyPDFLoader`.
  3. Uses `RecursiveCharacterTextSplitter` to cut text into chunks of **500 characters** with **100 characters overlap**.
- **Why Chunk Overlap (100 chars)?**  
  Imagine a sentence is cut in half at character 500. Without overlap, half the idea is in Chunk 1 and half is in Chunk 2. With 100 characters overlap, the boundary text appears in both chunks, so no information is lost!
- **Why Recursive Splitting?**  
  It tries to split by double line breaks (`\n\n` = paragraphs) first. If that's too big, it splits by single line breaks (`\n`), then spaces (` `), so words and sentences stay intact.

---

### 🧠 [utils/retriever_faiss.py](file:///d:/IITPatna/GenAI-DEV-Projects/Advanced-RAG-HybridRetrieval/utils/retriever_faiss.py) — The Meaning Searcher (Dense)
- **What it does:**
  - Connects to OpenAI or Gemini to convert text chunks into vector numbers.
  - Saves the numbers locally into two files:
    - `index.faiss`: The fast vector search database.
    - `index.pkl`: The dictionary storing the actual text corresponding to each vector.
- **Self-Healing Feature (`get_retriever`):**
  - If the index files already exist on disk, it simply loads them (super fast!).
  - If the files are missing or corrupted, it automatically rebuilds them from the raw document chunks so the app doesn't crash.
- **`allow_dangerous_deserialization=True`:**
  - LangChain requires this flag to load `.pkl` files safely from your local machine.

---

### 🔍 [utils/retriever_bm25.py](file:///d:/IITPatna/GenAI-DEV-Projects/Advanced-RAG-HybridRetrieval/utils/retriever_bm25.py) — The Keyword Searcher (Sparse)
- **What it does:**
  - Cleans and splits text into words (`simple_tokenize`): makes text lowercase and keeps only word characters.
  - Uses the `rank_bm25` library to build a keyword index in memory.
- **How BM25 Scores a Document (in plain English):**
  1. **Term Frequency (TF):** If your word appears multiple times, the score goes up. But after 3 or 4 times, extra mentions don't increase the score much (controlled by `k1=1.5`).
  2. **Inverse Document Frequency (IDF):** Rare words (like `Deductible`) get a high score. Common words (like `the`) get almost zero score.
  3. **Document Length Penalty:** Shorter, punchy paragraphs get a bonus over long, rambling documents (controlled by `b=0.75`).
- **Sorting with NumPy:**
  ```python
  top_k_idx = np.argsort(scores)[::-1][:k]
  ```
  `np.argsort` sorts low-to-high; `[::-1]` flips it to high-to-low; `[:k]` picks the top $k$ items.

---

### ⚖️ [utils/hybrid_retriever.py](file:///d:/IITPatna/GenAI-DEV-Projects/Advanced-RAG-HybridRetrieval/utils/hybrid_retriever.py) — The Mixer
- **The Problem:**  
  FAISS scores and BM25 scores are completely different scales (comparing apples to elephants).
- **The Solution: Score Normalization (`normalize_scores`):**
  - Scales all scores into a clean range between `0.0` and `1.0`:
    $$\text{Normalized Score} = \frac{\text{Score} - \text{Minimum}}{\text{Maximum} - \text{Minimum}}$$
  - **Zero-Division Guard:** If all documents score the same (e.g., all zero), `max - min` is 0. The code catches this and returns `1.0` so Python won't crash with a `ZeroDivisionError`.
- **The Blending Formula:**
  - Takes 60% from Dense and 40% from Sparse:
    $$\text{Final Score} = 0.6 \times \text{Dense Score} + 0.4 \times \text{Sparse Score}$$
  - Sorts all candidate chunks and returns the winners.

---

### 🚀 [app.py](file:///d:/IITPatna/GenAI-DEV-Projects/Advanced-RAG-HybridRetrieval/app.py) — The Main Application
- **What it does:**
  1. Reads `config.yaml` and `.env` API keys.
  2. Loads documents from `data/raw/insurance_docs`.
  3. Sets up FAISS, BM25, and Hybrid retrievers.
  4. Takes your question from the terminal.
  5. Runs **all 3 modes** and prints their answers side-by-side:
     - 🔹 `run_dense_only(...)`
     - 🔹 `run_sparse_only(...)`
     - 🔹 `run_hybrid(...)`
- **Context Construction (`make_context`):**
  - Joins chunks with `\n\n---\n\n`.
  - Cuts off at `3000` characters so you don't waste API tokens or slow down response time.
- **Anti-Hallucination Prompt:**
  - Tells the LLM: *"Answer only using the provided context. If the answer is not present, say you don't know."*
  - This prevents the LLM from making up false answers.

---

### 🧪 [test-nb.ipynb](file:///d:/IITPatna/GenAI-DEV-Projects/Advanced-RAG-HybridRetrieval/test-nb.ipynb) — The Notebook Playground
Contains 4 quick step-by-step tests:
1. **Test 1:** Checks API keys and configuration.
2. **Test 2:** Tests BM25 with query `"Essential Documents"`. (Finds the claim submission chunk with score `3.15`).
3. **Test 3:** Tests FAISS index creation and loading.
4. **Test 4:** Tests Hybrid search with query `"cashless hospitalization"` (Shows semantic retrieval working even when the exact keyword is not in the text!).

---

## 🆚 5. Real-World Query Examples: Who Wins?

| User Query | Who Wins? | Why? |
| :--- | :--- | :--- |
| *"What happens if my property is damaged?"* | **Dense (FAISS)** | User uses everyday words. FAISS understands this matches the concept of "Loss" and "Claim". |
| *"AUTO-987654"* or *"CLM-2025-1010"* | **Sparse (BM25)** | Exact policy and claim ID numbers. FAISS does not understand random numbers; BM25 matches them immediately. |
| *"What is an exclusion?"* | **Both (Tie)** | It is an exact word in the document, and also a key concept. |
| *"How do I file a claim with policy AUTO-987654?"* | **Hybrid (Big Winner!)** | Needs concept matching for *"how to file"* AND exact keyword matching for *"AUTO-987654"*. Pure Dense or pure Sparse would partially fail. Hybrid gets it 100% right! |

---

## 🎯 6. Top 10 Interview & Exam Questions (Simple Answers)

#### 1. What is Hybrid Search in RAG?
**Answer:** It is a search technique that combines **Dense Retrieval** (semantic meaning via embeddings) and **Sparse Retrieval** (keyword matching via BM25) to get high recall and high precision.

#### 2. Why is vector search alone not enough?
**Answer:** Vector search struggles with exact matches, such as product IDs, policy numbers, phone numbers, rare words, and specific codes. Keyword search (BM25) is needed to catch those exact terms.

#### 3. Why is BM25 alone not enough?
**Answer:** BM25 only checks if the exact words appear. If a user uses synonyms or asks a question in different words (e.g. *"house fire"* instead of *"property loss"*), BM25 finds nothing.

#### 4. Why do we need score normalization in hybrid retrieval?
**Answer:** FAISS similarity scores and BM25 scores use completely different number ranges. If we don't normalize both to a `0 to 1` scale, one retriever's scores will overpower the other.

#### 5. Why do we maintain separate folders for OpenAI and Gemini FAISS indexes?
**Answer:** OpenAI embeddings have **1,536 dimensions**, while Gemini embeddings have **768 dimensions**. Trying to load a 768-dimension vector into a 1,536-dimension index causes a dimension mismatch crash.

#### 6. What does `chunk_overlap=100` do?
**Answer:** It repeats the last 100 characters of each chunk at the start of the next chunk. This prevents sentences or ideas from being cut in half at chunk boundaries.

#### 7. What do the BM25 parameters $k_1$ and $b$ control?
**Answer:**
- **$k_1$ (1.5):** Term frequency saturation (how much repeated words boost the score before leveling off).
- **$b$ (0.75):** Document length penalty (prevents long documents from scoring high just because they have more words).

#### 8. What is `allow_dangerous_deserialization=True` in FAISS?
**Answer:** LangChain uses Python's `pickle` library to store document text. Since loading unknown pickle files can theoretically run malicious code, LangChain requires you to explicitly turn on this flag for trusted local files.

#### 9. What happens if a search word does not exist in any document for BM25?
**Answer:** BM25 returns a score of `0` for all documents. In hybrid retrieval, the dense retriever saves the day by finding chunks that match the general meaning.

#### 10. How do you prevent the LLM from making things up (hallucinating)?
**Answer:** We give it a strict system prompt: *"Answer only using the provided context. If the answer is not present, say you don't know."*
