# utils/hybrid_retriever.py

import numpy as np #numpy library for numerical operations
import yaml

from utils.retriever_faiss import get_retriever as get_faiss_retriever
from utils.retriever_bm25 import BM25Retriever
from langchain_core.documents import Document


# ---------------------------
# Config Loader
# ---------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------------------------
# Score Normalization Utility
# ---------------------------
def normalize_scores(scores):
    """
    Normalizes a list/array of scores into the range [0, 1].
    Handles edge cases where all scores are equal.
    """

    # Convert input to a NumPy array of floats
    scores = np.array(scores, dtype=float)

    # Find the minimum and maximum values in the array
    min_s, max_s = scores.min(), scores.max()

    # Check if all values are the same (to avoid division by zero)
    if max_s - min_s == 0:
        return np.ones_like(scores) 
    # If all scores are the same, return an array of ones (or zeros) to indicate uniformity.

    # This maps the smallest value to 0, the largest to 1, and others proportionally in between.
    return (scores - min_s) / (max_s - min_s)

# ---------------------------
# Hybrid Retriever Class
# ---------------------------
class HybridRetriever:
    def __init__(self, chunks, config_path="config.yaml"):
        self.config = load_config(config_path)

        # Dense retriever (OpenAI or Gemini depending on config)
        self.faiss_retriever = get_faiss_retriever(self.config, chunks_if_needed=chunks)

        # Sparse retriever (BM25)
        self.bm25_retriever = BM25Retriever(chunks, config_path)

        # Settings
        self.k_semantic = self.config["hybrid"]["k_semantic"]
        self.k_sparse = self.config["hybrid"]["k_sparse"]
        self.weights = self.config["hybrid"]["weights"]  # [semantic_weight, sparse_weight]
        self.normalize = self.config["hybrid"]["normalize"]

    # ---------------------------
    # Main Hybrid Retrieval Logic
    # ---------------------------
    def retrieve(self, query, top_k=5):
        # --- 1. Dense FAISS Retrieval ---
        dense_docs = self.faiss_retriever.invoke(query) #based on the query, it retrieves the top-k
        #documents from the FAISS index using semantic similarity
        dense_scores = np.linspace(1, 0.1, len(dense_docs))  
        # FAISS doesn't return similarity scores so we assign a linear score from 1 to 0.1 for ranking purposes
        '''
        np.linspace(1, 0.1, len(dense_docs)) creates a list of numbers starting at 1 and ending at 0.1,
        with as many values as there are items in dense_docs.
        '''
         # --- 2. Sparse BM25 Retrieval ---
        sparse_results = self.bm25_retriever.get_top_k(query, k=self.k_sparse)
        sparse_docs = [doc for doc, score in sparse_results] 
        #extracting the Document objects from the sparse_results list
        
        sparse_scores = [score for doc, score in sparse_results]
        #extracting the BM25 scores from the sparse_results list
        
        # --- 3. Normalize scores using above function between 0 and 1 ---
        if self.normalize:
            dense_scores = normalize_scores(dense_scores)
            sparse_scores = normalize_scores(sparse_scores)

        # --- 4. Combine results of both retrievers documents---
        combined_docs = dense_docs + sparse_docs
        
         # This code combines the scores from two different retrieval methods (dense and sparse) into 
         # a single score for each document, using weighted addition.

        # If dense_scores = [0.8, 0.6] and sparse_scores = [0.7, 0.5], 
        # np.concatenate([dense_scores, np.zeros(len(sparse_scores))]) gives [0.8, 0.6, 0.0, 0.0].
        combined_scores = (
            self.weights[0] * np.concatenate([dense_scores, np.zeros(len(sparse_scores))]) +
            self.weights[1] * np.concatenate([np.zeros(len(dense_scores)), sparse_scores])
        )
        # 0.6 * [0.8, 0.6, 0.0, 0.0] + 0.4 * [0.0, 0.0, 0.7, 0.5] 
        # = [0.48, 0.36, 0.28, 0.20]
        
        # --- 5. Rank final results ---
        idx_sorted = np.argsort(combined_scores)[::-1][:top_k] 
        #to get the indices of the top-k scores in descending order

        ranked_results = []
        for idx in idx_sorted:
            ranked_results.append((combined_docs[idx], float(combined_scores[idx])))

        return ranked_results
    #ranked_results is a list of tuples where each tuple contains a Document object 
    #and its corresponding combined score.
