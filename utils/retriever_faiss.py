# utils/retriever_faiss.py
import os
import yaml
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

INDEX_NAME = "index"  # langchain FAISS default -> creates index.faiss + index.pkl


def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_embedding_model(provider: str, config: dict):
    if provider == "openai":
        return OpenAIEmbeddings(model=config["embedding"]["openai_model"])
    elif provider == "gemini":
        return GoogleGenerativeAIEmbeddings(model=config["embedding"]["gemini_model"])
    else:
        raise ValueError("Provider must be one of: openai | gemini")


def get_index_path(provider: str, config: dict) -> str:
    return (
        config["vectordb"]["faiss_openai"]
        if provider == "openai"
        else config["vectordb"]["faiss_gemini"]
    )


def _index_files_exist(index_dir: str, index_name: str = INDEX_NAME) -> bool:
    faiss_path = os.path.join(index_dir, f"{index_name}.faiss")
    pkl_path = os.path.join(index_dir, f"{index_name}.pkl")
    return os.path.isfile(faiss_path) and os.path.isfile(pkl_path)


def create_retriever(chunks, provider: str, config: dict):
    """Build a fresh FAISS index from chunks."""
    index_path = get_index_path(provider, config)
    embeddings = get_embedding_model(provider, config)

    os.makedirs(index_path, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path, index_name=INDEX_NAME)

    print(f"✅ Created FAISS index ({provider}) at: {index_path}")
    return vectorstore.as_retriever(search_kwargs={"k": config["retrieval"]["top_k"]})


def load_retriever(provider: str, config: dict):
    """Load an existing FAISS index."""
    index_path = get_index_path(provider, config)
    embeddings = get_embedding_model(provider, config)

    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ Loaded FAISS index ({provider}) from: {index_path}")
    return vectorstore.as_retriever(search_kwargs={"k": config["retrieval"]["top_k"]})


def get_retriever(config: dict, chunks_if_needed=None):
    """
    Robust loader:
      - If index files missing -> create (requires chunks)
      - If load fails -> rebuild (if chunks provided), else raise a clear error
    """
    provider = config["llm"]["provider"]
    index_path = get_index_path(provider, config)

    # Case 1: No index files -> build
    if not _index_files_exist(index_path, INDEX_NAME):
        print(f"⚠️ No FAISS index found for '{provider}' at {index_path}. Creating a new one...")
        if chunks_if_needed is None:
            raise RuntimeError(
                f"FAISS files not found at {index_path}. "
                f"Please pass chunks_if_needed to build the index."
            )
        return create_retriever(chunks_if_needed, provider, config)

    # Case 2: Try to load; if it fails, rebuild if possible
    try:
        return load_retriever(provider, config)
    except Exception as e:
        print(f"⚠️ Failed to load FAISS index at {index_path}: {e}")
        if chunks_if_needed is None:
            raise RuntimeError(
                "Existing FAISS index appears corrupted or incompatible, "
                "and no chunks were provided to rebuild it. "
                "Please delete the folder and rerun with chunks."
            )
        print("🔧 Rebuilding FAISS index from chunks...")
        return create_retriever(chunks_if_needed, provider, config)