# utils/loader.py
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

import warnings
warnings.filterwarnings("ignore")

def load_and_chunk_docs(folder_path, chunk_size=500, chunk_overlap=100):
    docs = [] #this is a list that will hold all the documents loaded from the folder
    
    for file in os.listdir(folder_path): #this will iterate through all the files in the folder
        path = os.path.join(folder_path, file) #this will create the full path to the file
        
        if file.endswith('.txt'):
            loader = TextLoader(path)
        elif file.endswith('.pdf'):
            loader = PyPDFLoader(path)
        else:
            continue  # Skip unsupported file types
        
        docs.extend(loader.load()) #this will load the documents and add them to the docs list
    
    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    
    print(f"✅ Loaded {len(docs)} docs → {len(chunks)} chunks")
    return chunks