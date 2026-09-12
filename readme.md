# 📘 Advanced RAG Mini Project — Hybrid Search (Dense + Sparse Retrieval)

This mini-project demonstrates **Hybrid Retrieval**, one of the most powerful and practical techniques in modern RAG systems.
You will learn how to combine:

✅ **Dense semantic search (FAISS)**
✅ **Sparse lexical search (BM25)**
✅ **Hybrid weighted fusion (Dense + Sparse)**

The project supports **both OpenAI & Gemini** embeddings and automatically maintains separate FAISS indexes for each provider.

---

## 🚀 Project Features

### ✅ 1. Dual Embedding Provider Support

Choose your LLM/embedding provider via `config.yaml`:

* **OpenAI** → `text-embedding-3-small`
* **Gemini** → `models/text-embedding-004`

The system auto-creates separate indexes:

```
data/embeddings/faiss_openai/
data/embeddings/faiss_gemini/
```

so you never face dimension mismatch errors again.

---

### ✅ 2. FAISS Dense Retriever (Semantic)

* Captures meaning, not just keywords
* Excellent for paraphrased queries
* Provider-specific embedding dimensions are handled automatically

---

### ✅ 3. BM25 Sparse Retriever (Keyword)

Built using **rank-bm25**, delivering:

* Exact keyword matching
* High precision when user uses policy terms
* Fast, no external services required

---

### ✅ 4. Hybrid Retrieval (Dense + Sparse)

Uses:

* Z-score normalization
* Weighted linear fusion
* Tunable weights in `config.yaml`:

```yaml
hybrid:
  k_semantic: 5
  k_sparse: 5
  weights: [0.6, 0.4]   # 60% semantic, 40% sparse
  normalize: true
```

Hybrid search gives the **most stable retrieval performance** across all query types.

---

### ✅ 5. Consistent LLM Answering Pipeline

Each mode (Dense, Sparse, Hybrid) generates:

* Retrieved documents
* A context-block answer
* Source document preview

This allows students to compare different retrieval strategies.

---

## 📂 Project Structure

```
13-Advanced-RAG-Part2/
│
├── app.py                     # Main CLI application
├── config.yaml                # LLM provider + hybrid weights + FAISS paths
├── requirements.txt           # All libs (OpenAI, Gemini, LangChain, FAISS, BM25)
│
├── data/
│   ├── raw/
│   │   └── insurance_docs/    # Example insurance documents
│   └── embeddings/
│       ├── faiss_openai/      # Auto-created on first run
│       └── faiss_gemini/
│
└── utils/
    ├── loader.py              # Document loader + chunking
    ├── retriever_faiss.py     # Provider-aware FAISS builder/loader
    ├── retriever_bm25.py      # BM25 sparse retriever
    └── hybrid_retriever.py    # Dense + Sparse fusion
```

---

## ✅ Installation

Create virtual environment:

```bash
python -m venv myenv
source myenv/bin/activate   # or myenv\Scripts\activate on Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add API keys:

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

---

## ▶️ Run the App

```bash
python app.py
```

Example:

```
🔍 Enter your question: cashless hospitalization?
```

You’ll see:

### ✅ Dense (FAISS) answer

### ✅ Sparse (BM25) answer

### ✅ Hybrid (FAISS + BM25) answer

### ✅ Top document sources for each mode

---

## 🧪 Notebook Smoke Test

You can verify components programmatically using the included test snippet:

```python
from utils.loader import load_and_chunk_docs
from utils.retriever_faiss import get_retriever
from utils.retriever_bm25 import BM25Retriever
from utils.hybrid_retriever import HybridRetriever
import yaml

config = yaml.safe_load(open("config.yaml"))
chunks = load_and_chunk_docs("./data/raw/insurance_docs")

faiss = get_retriever(config, chunks_if_needed=chunks)
bm25 = BM25Retriever(chunks)
hybrid = HybridRetriever(chunks)

query = "cashless hospitalization"

dense_docs = faiss.invoke(query)
sparse_docs = [d for d,_ in bm25.get_top_k(query, k=5)]
hybrid_docs = [d for d,_ in hybrid.retrieve(query, top_k=5)]

print("DENSE:", dense_docs[0].page_content[:120])
print("SPARSE:", sparse_docs[0].page_content[:120])
print("HYBRID:", hybrid_docs[0].page_content[:120])
```

---

## 🎯 Learning Outcomes

By the end of the project you will understand:

✅ Why dense retrieval is not enough
✅ When sparse retrieval excels
✅ How to combine both for state-of-the-art recall
✅ How to manage multi-provider embeddings
✅ How to maintain multiple FAISS indexes safely
✅ How hybrid search stabilizes retrieval quality

---