# utils/retriever_bm25.py
from rank_bm25 import BM25Okapi #importing the BM25Okapi class from the rank_bm25 library for BM25 retrieval
from langchain_core.documents import Document 
#importing the Document class from langchain_core for handling documents
import numpy as np #numpy library for numerical operations
import re #regular expression library for text processing
import yaml #yaml library for reading configuration files

import warnings
warnings.filterwarnings("ignore")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def simple_tokenize(text: str):
    """
    Basic tokenizer suitable for BM25. 
    You can enhance later (stopwords, lemmatization, etc.).
    """
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    '''
    This line returns a list of all words in text, ignoring punctuation and splitting on whitespace 
    or non-word characters. For example, "Hello, world!" → ["hello", "world"] (after lowercasing).
    '''
    return tokens

class BM25Retriever:
    def __init__(self, documents, config_path="config.yaml"):
        """
        documents: list[Document] — LangChain Document chunks
        """
        self.config = load_config(config_path)

        self.documents = documents
        self.corpus = [simple_tokenize(doc.page_content) for doc in documents]
        #obviously we are doing list comprehension to tokenize each document's content so need to go inside list
        
        # Initialize BM25 with corpus
        self.bm25 = BM25Okapi(
            self.corpus,
            k1=self.config["bm25"]["k1"],
            b=self.config["bm25"]["b"],
        )
        
    def get_top_k(self, query: str, k=None):
        """
        takes query and value of k, Returns top-k matching Document objects along with their scores 
        based on BM25 lexical (keyword)similarity.
        """
        if k is None:
            k = self.config["hybrid"]["k_sparse"]
            
        query_tokens = simple_tokenize(query) 
        #tokenizing the query string using the simple_tokenize function defined earlier
        
        scores = self.bm25.get_scores(query_tokens)
        #suppose if we have 50 documents then scores for each document will be calculated based on the query
        #tokens and stored in a list
        #print(f"BM25 Scores: {scores}")
        
        top_k_idx = np.argsort(scores)[::-1][:k]  #it is assigning a score to each document based on the query tokens
        #and then sorting the scores in descending order and getting the indices of the top k scores
        #[::-1] is used to reverse the order of the sorted indices so that we get the indices of 
        #the highest scores first.
        
        '''
        1. scores = np.array([0.2, 0.8, 0.5, 0.1])
        2. np.argsort(scores) → array([3, 0, 2, 1]) here 3 is the index of the lowest score, 0 is the index
        of the second lowest score, and so on.
        3. np.argsort(scores)[::-1] → array([1, 2, 0, 3]) #this reverses the order of the sorted indices
        so that we get the indices of the highest scores first.
        4. np.argsort(scores)[::-1][:2] → array([1, 2]) #this gets the indices of the top 2 scores (highest scores) 
        from the reversed sorted indices.
        '''
        
        docs = []
        for idx in top_k_idx:
            doc = self.documents[idx]
            docs.append((doc, float(scores[idx]))) 
            #it is appending a tuple of the document and its corresponding score to the docs list
        return docs