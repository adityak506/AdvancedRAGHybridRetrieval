# app.py
import os
import yaml
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

from utils.loader import load_and_chunk_docs
from utils.retriever_faiss import get_retriever as get_faiss_retriever
from utils.retriever_bm25 import BM25Retriever
from utils.hybrid_retriever import HybridRetriever

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# ---------------------------
# Config & LLM helpers
# ---------------------------
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_llm(config: dict):
    provider = config["llm"]["provider"]
    temperature = config["llm"]["temperature"]

    if provider == "openai":
        return ChatOpenAI(model=config["llm"]["model_openai"], temperature=temperature)
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(model=config["llm"]["model_gemini"], temperature=temperature)
    else:
        raise ValueError("Provider must be one of: openai | gemini")


# ---------------------------
# Utilities
# ---------------------------
def unify_docs(docs_or_pairs):
    """
    Accepts either:
      - list[Document]
      - list[(Document, score)]
    Returns list[Document]
    """
    if not docs_or_pairs:
        return []
    if hasattr(docs_or_pairs[0], "page_content"):
        return docs_or_pairs
    return [d for d, _ in docs_or_pairs]


def make_context(docs, max_chars=3000):
    """
    Concatenate page_content into a single context string,
    truncated to max_chars for cost/speed while demoing.
    """
    text = "\n\n---\n\n".join(d.page_content.strip() for d in docs)
    return text[:max_chars]


def answer_with_context(llm, context: str, question: str) -> str:
    prompt = [
        ("system", "Answer only using the provided context. If the answer is not present, say you don't know."),
        ("human", f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    ]
    return llm.invoke(prompt).content.strip()


# ---------------------------
# Pipelines
# ---------------------------
#semantic retrieval only
def run_dense_only(query: str, faiss_retriever, llm, top_k=5):
    dense_docs = faiss_retriever.invoke(query)[:top_k]
    ctx = make_context(dense_docs)
    ans = answer_with_context(llm, ctx, query)
    return ans, dense_docs

#bm25 retrieval only (keyword based retrieval)
def run_sparse_only(query: str, bm25: BM25Retriever, llm, top_k=5):
    sparse_pairs = bm25.get_top_k(query, k=top_k)
    sparse_docs = unify_docs(sparse_pairs)
    ctx = make_context(sparse_docs)
    ans = answer_with_context(llm, ctx, query)
    return ans, sparse_docs

#hybrid retrieval (fusion of semantic and keyword based retrieval)
def run_hybrid(query: str, hybrid: HybridRetriever, llm, top_k=5):
    hybrid_pairs = hybrid.retrieve(query, top_k=top_k)
    hybrid_docs = unify_docs(hybrid_pairs)
    ctx = make_context(hybrid_docs)
    ans = answer_with_context(llm, ctx, query)
    return ans, hybrid_docs


# ---------------------------
# Main
# ---------------------------
def main():
    config = load_config()
    provider = config["llm"]["provider"].upper()
    print(f"\n🚀 Active Provider: {provider}")

    # 1) Load & chunk docs
    chunks = load_and_chunk_docs("./data/raw/insurance_docs")

    # 2) Build/load retrievers
    faiss_retriever = get_faiss_retriever(config, chunks_if_needed=chunks)
    bm25_retriever = BM25Retriever(chunks)                 # sparse
    hybrid_retriever = HybridRetriever(chunks)             # fusion

    # 3) LLM for answering
    llm = get_llm(config)

    # 4) Ask
    query = input("\n🔍 Enter your question: ").strip()
    top_k = max(config["retrieval"]["top_k"], 1)

    print("\n====================")
    print("✅ DENSE (FAISS) ONLY")
    print("====================")
    dense_ans, dense_docs = run_dense_only(query, faiss_retriever, llm, top_k=top_k)
    print(dense_ans)

    print("\n====================")
    print("✅ SPARSE (BM25) ONLY")
    print("====================")
    sparse_ans, sparse_docs = run_sparse_only(query, bm25_retriever, llm, top_k=top_k)
    print(sparse_ans)

    print("\n====================")
    print("✅ HYBRID (FAISS + BM25)")
    print("====================")
    hybrid_ans, hybrid_docs = run_hybrid(query, hybrid_retriever, llm, top_k=top_k)
    print(hybrid_ans)

    # # (Optional) Show which doc titles/metadata contributed
    # def brief(doc):
    #     title = doc.metadata.get("source", "") or doc.metadata.get("file_path", "") or ""
    #     snippet = doc.page_content[:80].replace("\n", " ")
    #     return f"{title} :: {snippet}..."

    # print("\n--- Top sources (Dense) ---")
    # for i, d in enumerate(dense_docs[:top_k], 1):
    #     print(f"{i}. {brief(d)}")

    # print("\n--- Top sources (Sparse) ---")
    # for i, d in enumerate(sparse_docs[:top_k], 1):
    #     print(f"{i}. {brief(d)}")

    # print("\n--- Top sources (Hybrid) ---")
    # for i, d in enumerate(hybrid_docs[:top_k], 1):
    #     print(f"{i}. {brief(d)}")


if __name__ == "__main__":
    main()