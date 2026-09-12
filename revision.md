# 📚 Advanced RAG with Hybrid Retrieval: Comprehensive Master Revision Notes

> **Project:** Advanced RAG Mini Project — Hybrid Retrieval (Dense Semantic + Sparse Lexical Search)  
> **Tech Stack:** Python 3.11, LangChain (v0.3), FAISS, Rank-BM25, OpenAI API, Google Gemini API, NumPy, PyPDF  
> **Key Objective:** Overcoming the limitations of pure semantic search by combining **Dense Embeddings (FAISS)** and **Sparse Lexical Search (BM25)** with weighted score fusion and multi-provider embedding isolation.

---

## 📑 Table of Contents
1. [Core Concepts & Theoretical Foundations](#1-core-concepts--theoretical-foundations)
2. [High-Level Architecture & End-to-End Workflow](#2-high-level-architecture--end-to-end-workflow)
3. [Configuration & Multi-Provider Isolation (`config.yaml`)](#3-configuration--multi-provider-isolation-configyaml)
4. [Document Ingestion & Chunking (`utils/loader.py`)](#4-document-ingestion--chunking-utilsloaderpy)
5. [Dense Semantic Retriever with FAISS (`utils/retriever_faiss.py`)](#5-dense-semantic-retriever-with-faiss-utilsretriever_faisspy)
6. [Sparse Lexical Retriever with BM25 (`utils/retriever_bm25.py`)](#6-sparse-lexical-retriever-with-bm25-utilsretriever_bm25py)
7. [Hybrid Fusion Engine (`utils/hybrid_retriever.py`)](#7-hybrid-fusion-engine-utilshybrid_retrieverpy)
8. [End-to-End Execution Pipeline (`app.py`)](#8-end-to-end-execution-pipeline-apppy)
9. [Verification & Notebook Experiments (`test-nb.ipynb`)](#9-verification--notebook-experiments-test-nbinb)
10. [Dense vs. Sparse vs. Hybrid: The Definitive Comparison](#10-dense-vs-sparse-vs-hybrid-the-definitive-comparison)
11. [Code Nuances, Edge Cases & Industry Enhancements](#11-code-nuances-edge-cases--industry-enhancements)
12. [Interview & Revision Flashcards (Top 12 Questions)](#12-interview--revision-flashcards-top-12-questions)

---

## 1. Core Concepts & Theoretical Foundations

### 1.1 The Fundamental Flaw of "Naive RAG"
Traditional RAG pipelines rely exclusively on **Dense Semantic Embeddings** (e.g., OpenAI embeddings stored in FAISS, Chroma, or Pinecone). While dense retrieval excels at semantic understanding and paraphrasing, it fails in several real-world scenarios:
- **Exact Matches & Technical Identifiers:** Policy numbers (e.g., `AUTO-987654`), claim codes (`CLM-2025-1010`), phone numbers (`(800) 555-CLAIM`), error codes, and specific SKU numbers.
- **Acronyms and Domain Terms:** Industry-specific acronyms (`ACV`, `OOP`, `HIPAA`) that embedding models might not have encountered densely in training.
- **Out-of-Vocabulary (OOV) / Rare Words:** Words mapped to ambiguous or distant vectors in dense space.

Conversely, **Sparse Lexical Search** (e.g., BM25, TF-IDF) relies on exact token matching and term frequency:
- **Strength:** Precise matching of exact keywords, serial numbers, and entities.
- **Weakness:** Incapable of understanding synonyms or intent (e.g., searching *"physician consultation"* will miss documents mentioning only *"doctor visit"*).

### 1.2 The Solution: Hybrid Retrieval
Hybrid retrieval fuses dense vector similarity with sparse lexical matching. By combining both scoring spaces, the retrieval engine achieves:
1. **High Semantic Recall:** Captures the conceptual meaning and paraphrased user queries.
2. **High Lexical Precision:** Pinpoints specific keywords, exact terms, and codes.
3. **Resilience to Query Variability:** Whether the user enters a natural conversational question or keyword search shorthand, the system retrieves relevant chunks.

---

## 2. High-Level Architecture & End-to-End Workflow

```
[ Raw Documents ]
  ├── policy_terms.txt
  └── claim_procedure.txt
           │
           ▼
[ utils/loader.py ] ──► TextLoader / PyPDFLoader + RecursiveCharacterTextSplitter
           │           (chunk_size=500, chunk_overlap=100)
           ▼
     [ 8 Document Chunks ]
           │
     ┌─────┴────────────────────────────────┐
     ▼                                      ▼
[ Dense Stream: FAISS ]             [ Sparse Stream: BM25 ]
- OpenAI / Gemini Embeddings        - Tokenization: re.findall(\b\w+\b)
- Vector Index (.faiss + .pkl)      - Corpus Term Frequencies (k1=1.5, b=0.75)
- Cosine / L2 distance search       - Lexical inverted index matching
     │                                      │
     ▼                                      ▼
Dense Top-K Chunks                  Sparse Top-K Chunks
(Rank proxy: np.linspace)           (Raw BM25 scores)
     │                                      │
     └───────────────┬──────────────────────┘
                     ▼
       [ utils/hybrid_retriever.py ]
         1. Min-Max Score Normalization [0, 1]
         2. Weighted Linear Combination (0.6 * Dense + 0.4 * Sparse)
         3. Global Re-ranking (np.argsort descending)
                     │
                     ▼
         [ Top-K Combined Chunks ]
                     │
                     ▼
       [ Truncated Context (max_chars=3000) ]
                     │
                     ▼
       [ LLM: GPT-4o-mini / Gemini-2.5-Flash ]
                     │
                     ▼
       [ Grounded Final Answer ]
```

---

## 3. Configuration & Multi-Provider Isolation (`config.yaml`)

The project uses `config.yaml` as the centralized source of truth for models, hyperparameters, and vector store paths:

```yaml
llm:
  provider: "openai"            # options: openai | gemini
  model_openai: "gpt-4o-mini"
  model_gemini: "gemini-2.5-flash"
  temperature: 0.3
  max_tokens: 1000

embedding:
  openai_model: "text-embedding-3-small"      # Vector Dimension: 1536
  gemini_model: "models/text-embedding-004"   # Vector Dimension: 768

vectordb:
  faiss_openai: "./data/embeddings/faiss_openai"
  faiss_gemini: "./data/embeddings/faiss_gemini"

retrieval:
  top_k: 5

hybrid:
  k_semantic: 5        # Candidates fetched from FAISS
  k_sparse: 5          # Candidates fetched from BM25
  weights: [0.6, 0.4]  # [semantic_weight, sparse_weight]
  normalize: true      # Min-Max score normalization before linear fusion

bm25:
  k1: 1.5              # Term Frequency saturation parameter
  b: 0.75              # Document length normalization parameter
  stopwords: []        # Optional stopword filter list
```

### 🔑 Critical Architectural Design: Dimension Isolation
- **OpenAI `text-embedding-3-small`:** Output vector dimension = **1536**.
- **Gemini `models/text-embedding-004`:** Output vector dimension = **768**.
- **The Pitfall:** If a single vector store folder were reused across providers, switching providers would throw a fatal FAISS dimension mismatch error (`Index dimension mismatch: expected 1536, got 768`).
- **The Solution:** Two distinct directories (`data/embeddings/faiss_openai` and `data/embeddings/faiss_gemini`) ensure full environment stability.

---

## 4. Document Ingestion & Chunking (`utils/loader.py`)

### 4.1 Implementation Logic
The document loader automatically scans a given directory, picks appropriate loaders based on extension (`.txt` -> `TextLoader`, `.pdf` -> `PyPDFLoader`), and splits the raw content using `RecursiveCharacterTextSplitter`.

```python
def load_and_chunk_docs(folder_path, chunk_size=500, chunk_overlap=100):
    docs = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith('.txt'):
            loader = TextLoader(path)
        elif file.endswith('.pdf'):
            loader = PyPDFLoader(path)
        else:
            continue
        docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    return chunks
```

### 4.2 Key Ingestion Parameters Explained
- **`RecursiveCharacterTextSplitter`:** Recursively tries separators in the order `["\n\n", "\n", " ", ""]`. This preserves natural paragraph boundaries and sentence integrity rather than cutting arbitrary words in half.
- **`chunk_size=500`:** Keeps chunks tightly focused on 1–2 specific concepts or policy clauses. Smaller chunks reduce embedding noise.
- **`chunk_overlap=100` (20% overlap):** Prevents contextual boundary loss (e.g., when a claim procedure step begins at character 480 and finishes at character 550).
- **Result in this project:** 2 raw insurance files (`claim_procedure.txt` and `policy_terms.txt`) produce **8 clean chunks**.

---

## 5. Dense Semantic Retriever with FAISS (`utils/retriever_faiss.py`)

### 5.1 Roles and Responsibilities
1. Dynamically instantiates the correct embedding client (`OpenAIEmbeddings` or `GoogleGenerativeAIEmbeddings`) based on `config.yaml`.
2. Persists index files: `index.faiss` (binary vector index) and `index.pkl` (docstore mapping metadata and page content).
3. Provides self-healing index retrieval (`get_retriever`).

### 5.2 Self-Healing Index Flow (`get_retriever`)
```
                     [ get_retriever() called ]
                                 │
                 Do index.faiss & index.pkl exist?
                                 │
                   ┌─────────────┴─────────────┐
                  YES                          NO
                   │                           │
          Try load_retriever()        chunks_if_needed given?
                   │                           │
             ┌─────┴─────┐               ┌─────┴─────┐
          Success     Exception         YES          NO
             │           │               │           │
          Return      Rebuild      create_retriever() Raise Error
         Retriever  from chunks          │
                         │            Return
                    Return Retriever  Retriever
```

### 5.3 Safe Deserialization Note
In `FAISS.load_local()`, LangChain defaults to blocking pickle deserialization for security. The code sets:
```python
vectorstore = FAISS.load_local(
    index_path,
    embeddings,
    index_name=INDEX_NAME,
    allow_dangerous_deserialization=True,  # Required for loading trusted local pickle docstores
)
```

---

## 6. Sparse Lexical Retriever with BM25 (`utils/retriever_bm25.py`)

### 6.1 Mathematical Formulation of BM25Okapi
BM25 (Best Matching 25) calculates the relevance score of document $D$ for query $Q = \{q_1, q_2, \dots, q_n\}$:

$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $f(q_i, D)$: Term frequency of token $q_i$ in document $D$.
- $|D|$: Length of document $D$ in words.
- $\text{avgdl}$: Average document length across the entire corpus.
- $\text{IDF}(q_i)$: Inverse Document Frequency:
  $$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$
  (where $N$ is total documents, $n(q_i)$ is count of documents containing $q_i$).
- **$k_1$ Parameter (Config: `1.5`):** Controls term frequency saturation. As a term appears more frequently, its marginal score gain diminishes asymptotically.
- **$b$ Parameter (Config: `0.75`):** Controls length normalization penalty. Documents longer than average are penalized to prevent verbose documents from dominating results simply by having more total words.

### 6.2 Implementation Breakdown
1. **Tokenization (`simple_tokenize`):**
   ```python
   def simple_tokenize(text: str):
       text = text.lower()
       tokens = re.findall(r"\b\w+\b", text)  # strips punctuation, splits into words
       return tokens
   ```
2. **Corpus Preparation:** Converts all chunks into lists of tokens:
   ```python
   self.corpus = [simple_tokenize(doc.page_content) for doc in documents]
   self.bm25 = BM25Okapi(self.corpus, k1=1.5, b=0.75)
   ```
3. **Query Scoring & Reverse Sorting:**
   ```python
   scores = self.bm25.get_scores(query_tokens)
   top_k_idx = np.argsort(scores)[::-1][:k]  # [::-1] sorts descending
   ```

---

## 7. Hybrid Fusion Engine (`utils/hybrid_retriever.py`)

### 7.1 Score Incommensurability Problem
- FAISS distances/similarities (e.g., L2 distances or cosine similarities) operate in a completely different mathematical space from BM25 scores (which range from $0$ to $+\infty$, often peaking between $2.0$ and $15.0$).
- Directly summing raw FAISS scores with BM25 scores would cause BM25 to overpower dense similarity, or vice-versa.

### 7.2 Normalization Formula (`normalize_scores`)
Min-Max normalization rescales scores into the range $[0, 1]$:

$$S_{\text{norm}} = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$$

**Edge-Case Guard:** If all documents receive the exact same score (e.g. all 0), $S_{\max} - S_{\min} = 0$. The code safeguards against division-by-zero:
```python
if max_s - min_s == 0:
    return np.ones_like(scores)
return (scores - min_s) / (max_s - min_s)
```

### 7.3 Linear Weighted Score Combination
The hybrid retriever implements weighted linear combination:
```python
# dense_scores: normalized FAISS scores [k_semantic]
# sparse_scores: normalized BM25 scores [k_sparse]

combined_scores = (
    self.weights[0] * np.concatenate([dense_scores, np.zeros(len(sparse_scores))]) +
    self.weights[1] * np.concatenate([np.zeros(len(dense_scores)), sparse_scores])
)
```
- Configured weights: `weights: [0.6, 0.4]` (60% semantic weight, 40% lexical weight).
- If FAISS returns 5 docs and BM25 returns 5 docs, `combined_scores` has length 10.
- `np.argsort(combined_scores)[::-1][:top_k]` extracts the top-scoring candidate documents across both retrieval methods.

---

## 8. End-to-End Execution Pipeline (`app.py`)

### 8.1 Unified Document Format (`unify_docs`)
Different retrieval components return different signatures:
- FAISS `.invoke()` returns `list[Document]`
- BM25 `.get_top_k()` returns `list[tuple[Document, float]]`
- Hybrid `.retrieve()` returns `list[tuple[Document, float]]`

`unify_docs()` ensures downstream consumers always receive a standardized `list[Document]`:
```python
def unify_docs(docs_or_pairs):
    if not docs_or_pairs:
        return []
    if hasattr(docs_or_pairs[0], "page_content"):
        return docs_or_pairs
    return [d for d, _ in docs_or_pairs]
```

### 8.2 Context Window Management (`make_context`)
Concatenates retrieved chunk text using delimiter `\n\n---\n\n` and truncates at `max_chars=3000` to prevent context bloat, control API latency, and reduce token usage.

### 8.3 Anti-Hallucination Prompting (`answer_with_context`)
```python
prompt = [
    ("system", "Answer only using the provided context. If the answer is not present, say you don't know."),
    ("human", f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
]
```
This forces the LLM into strict extractive/extrapolative grounded generation, eliminating hallucinations when retrieved documents lack the answer.

### 8.4 Multi-Pipeline Comparison
`app.py` runs all three pipelines consecutively for the user's query:
1. `run_dense_only(...)`
2. `run_sparse_only(...)`
3. `run_hybrid(...)`

This provides direct, comparative observation of how retrieval modes alter the final generated answer.

---

## 9. Verification & Notebook Experiments (`test-nb.ipynb`)

The notebook contains 4 sequential integration smoke tests:
1. **Test 1: Environment & Config Check:** Validates `.env` keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`) and parses `config.yaml`.
2. **Test 2: BM25 Sparse Sanity Test:**
   - Query: `"Essential Documents"`
   - Result: Chunk containing *"II. Documentation Submission"* scored highest ($3.15$), other chunks scored $0.00$.
3. **Test 3: FAISS Dense Retriever Check:** Verifies creation/loading of `faiss_openai` index with 8 chunks.
4. **Test 4: Hybrid Search Integration:**
   - Query: `"cashless hospitalization"`
   - Demonstrates hybrid retrieval returning semantic chunks even when BM25 keyword score was $0.00$, proving semantic fallback capability.

---

## 10. Dense vs. Sparse vs. Hybrid: The Definitive Comparison

| Feature / Dimension | Dense Retrieval (FAISS) | Sparse Retrieval (BM25) | Hybrid Retrieval (Fusion) |
| :--- | :--- | :--- | :--- |
| **Primary Mechanism** | Dense vector cosine/L2 distance | Inverted index token frequency | Weighted union of normalized scores |
| **Best Used For** | Paraphrased queries, concept search, exploratory questions | Exact IDs, policy terms, product codes, rare words | Production RAG systems requiring high recall + precision |
| **Failure Case** | Exact codes, specific alphanumeric IDs, rare jargon | Synonyms, rephrased queries, typos, cross-lingual | Requires tuning weights & score normalization |
| **Computational Cost** | High (vector embedding API calls + vector indexing) | Low (CPU-bound in-memory inverted index) | Moderate (executes both, then merges) |
| **Out-of-Domain Robustness** | Moderate (depends on embedding pretraining) | High (pure lexical statistics of given corpus) | Highest overall robustness |
| **Index Size** | Large (e.g. 1536 floats per chunk) | Very compact (vocabulary index + doc lengths) | Requires storing both indices |

---

## 11. Code Nuances, Edge Cases & Industry Enhancements

### 11.1 The Pseudo-Score Approximation in FAISS
In `utils/hybrid_retriever.py`:
```python
dense_docs = self.faiss_retriever.invoke(query)
dense_scores = np.linspace(1, 0.1, len(dense_docs))
```
- **Why it exists:** LangChain's `retriever.invoke()` returns only `Document` objects without similarity scores.
- **Production Upgrade:** Use `vectorstore.similarity_search_with_score(query, k=...)` to retrieve the true distance/similarity metrics rather than synthetic linear decays.

### 11.2 Handling Duplicate Documents Across Streams
In the current implementation:
```python
combined_docs = dense_docs + sparse_docs
```
If chunk #2 is retrieved by **both** FAISS and BM25, it currently appears twice in `combined_docs` as distinct items.
- **Industry Best Practice — Reciprocal Rank Fusion (RRF):**
  Instead of normalizing disjoint score spaces, RRF computes a score based purely on rank positions:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k_{\text{rrf}} + \text{rank}_m(d)}$$
  (where $k_{\text{rrf}} \approx 60$). RRF naturally deduplicates documents and merges rank signals without requiring score normalization.

### 11.3 Two-Stage Retrieval (Reranking)
In modern enterprise architectures:
1. **Stage 1 (Bi-Encoder / Hybrid):** Retrieve top 20–50 candidate chunks fast using Hybrid FAISS + BM25.
2. **Stage 2 (Cross-Encoder / Reranker):** Pass candidates to a Cross-Encoder (e.g., `Cohere Rerank`, `bge-reranker-large`) that jointly evaluates `(query, document)` attention for precise ranking.

---

## 12. Interview & Revision Flashcards (Top 12 Questions)

#### Q1: Why can't we rely solely on dense embeddings in enterprise RAG?
**Answer:** Dense embeddings compress document semantics into fixed-size vectors, causing "lossy" compression for precise tokens. They often fail on exact alphanumeric queries, serial/claim numbers, technical acronyms, and rare vocabulary. Lexical search (BM25) is necessary to ensure exact-match precision.

#### Q2: What are the two key hyperparameters in BM25, and what do they control?
**Answer:**
1. $k_1$ (typically $1.2 - 2.0$, here $1.5$): Controls term frequency saturation.
2. $b$ (typically $0.75$, here $0.75$): Controls document length penalization. Higher $b$ penalizes long documents more severely.

#### Q3: Why is score normalization essential in weighted hybrid retrieval?
**Answer:** Dense vector similarities (e.g. cosine in $[0, 1]$ or L2 distance) and BM25 scores (unbounded positive floats) have different scales and distributions. Without normalization, the retriever with larger magnitude scores would dominate the ranking regardless of assigned weights.

#### Q4: What is the risk of dividing by `(max_s - min_s)` in Min-Max normalization, and how is it handled?
**Answer:** If all retrieved documents receive identical scores (e.g., all 0 or identical matches), `max_s - min_s == 0`, causing a `ZeroDivisionError`. The code checks `if max_s - min_s == 0:` and returns `np.ones_like(scores)` to maintain uniformity.

#### Q5: Why are OpenAI and Gemini FAISS indexes saved in separate directories?
**Answer:** OpenAI (`text-embedding-3-small`) outputs 1536-dimensional vectors, while Gemini (`models/text-embedding-004`) outputs 768-dimensional vectors. Storing them separately prevents dimension mismatch errors when toggling `provider` in `config.yaml`.

#### Q6: Why does `FAISS.load_local()` require `allow_dangerous_deserialization=True`?
**Answer:** LangChain uses Python's `pickle` module to serialize the document store (`index.pkl`). Because unpickling untrusted files can execute arbitrary code, LangChain requires explicit opt-in via this flag.

#### Q7: How does `RecursiveCharacterTextSplitter` differ from `CharacterTextSplitter`?
**Answer:** `CharacterTextSplitter` splits strictly on a single delimiter. `RecursiveCharacterTextSplitter` uses a list of delimiters (`["\n\n", "\n", " ", ""]`) hierarchically, splitting on larger semantic boundaries first and falling back to smaller ones only when chunks exceed `chunk_size`.

#### Q8: What does `np.argsort(scores)[::-1][:k]` do?
**Answer:**
1. `np.argsort(scores)` returns the indices that would sort the array in ascending order.
2. `[::-1]` slices the array backwards, reversing it to descending order (highest score first).
3. `[:k]` selects the top $k$ indices.

#### Q9: What happens when a user queries for a term not in the BM25 corpus?
**Answer:** BM25 assigns a score of 0 to all documents. In hybrid retrieval, the dense retriever compensates by locating conceptually related documents, preventing total retrieval failure.

#### Q10: What is the purpose of `chunk_overlap`?
**Answer:** It ensures that context spanning the boundary between adjacent chunks is not cut off, preventing incomplete thoughts or lost relationships across chunk splits.

#### Q11: What is Reciprocal Rank Fusion (RRF) and why is it often preferred over linear score fusion?
**Answer:** RRF scores documents based on their reciprocal rank positions rather than arbitrary raw scores: $RRF = \sum \frac{1}{60 + \text{rank}}$. It eliminates the need for score calibration/normalization across completely different scoring algorithms.

#### Q12: How does the system prompt protect against LLM hallucination?
**Answer:** By instructing the model: `"Answer only using the provided context. If the answer is not present, say you don't know."` This restricts the LLM to context-grounded reasoning and prevents speculative generation when context is missing.
