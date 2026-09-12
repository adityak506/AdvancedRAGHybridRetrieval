import os
import pathlib

def create_hybrid_retrieval_structure():
    """Create the hybrid retrieval project structure with empty files"""
    
    # Define the complete folder structure
    structure = {
        "": [  # Root directory files
            "app.py",
            "config.yaml",
            "requirements.txt",
            "test-nb.ipynb",
            ".env",
            "readme.md"
        ],
        "data/raw/insurance_docs": [
            "policy_terms.txt",
            "claim_procedure.txt"
        ],
        "data/embeddings": [
            "faiss_openai/",
            "faiss_gemini/"
        ],
        "utils": [
            "loader.py",
            "retriever_faiss.py",
            "retriever_bm25.py",
            "hybrid_retriever.py"
        ]
    }
    
    print("Creating Hybrid Retrieval Project structure...")
    
    # Create all folders and files
    for folder, items in structure.items():
        # Create folder if it doesn't exist
        if folder:
            os.makedirs(folder, exist_ok=True)
            print(f"📁 Created folder: {folder}/")
        
        # Create files within the folder
        for item in items:
            file_path = os.path.join(folder, item) if folder else item
            
            # Handle directories (ending with /)
            if item.endswith('/'):
                os.makedirs(file_path, exist_ok=True)
                print(f"📁 Created folder: {file_path}")
            else:
                # Create empty file
                pathlib.Path(file_path).touch()
                print(f"📄 Created file: {file_path}")
    
    print("\n✅ Hybrid Retrieval Project structure created successfully!")
    print("📂 All files and folders are now ready.")

if __name__ == "__main__":
    create_hybrid_retrieval_structure()