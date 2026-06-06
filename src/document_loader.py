import os
from langchain_community.document_loaders import PyPDFLoader

def load_documents(directory_path):
    """Loads all PDF documents from a specified directory."""
    documents = []
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} not found. Creating it...")
        os.makedirs(directory_path)
        return documents

    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded {filename}")
            
    return documents
