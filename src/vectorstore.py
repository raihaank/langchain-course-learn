from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_or_load_vectorstore(chunks, persist_directory="./faiss_db"):
    """Creates or incrementally updates a FAISS vector store from document chunks."""
    embeddings = OpenAIEmbeddings()

    if chunks:
        new_vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        # If an existing index is present, merge into it instead of overwriting
        if os.path.exists(persist_directory):
            print("Existing index found. Merging new documents into it...")
            existing_vectorstore = FAISS.load_local(
                folder_path=persist_directory,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            existing_vectorstore.merge_from(new_vectorstore)
            existing_vectorstore.save_local(persist_directory)
            print("Merge complete. Index updated.")
            return existing_vectorstore
        else:
            print("Creating new vector store from chunks...")
            new_vectorstore.save_local(persist_directory)
            return new_vectorstore
    else:
        if os.path.exists(persist_directory):
            print("Loading existing vector store...")
            return FAISS.load_local(
                folder_path=persist_directory,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("No chunks provided and no existing DB found.")
            return None
