import os
import argparse
from src.config import load_config
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vectorstore import create_or_load_vectorstore
from src.rag_chain import create_rag_chain

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="RAG Project Main Interface")
    parser.add_argument("--index", action="store_true", help="Index PDFs in the data directory")
    parser.add_argument("--query", type=str, help="Ask a question")
    args = parser.parse_args()

    # Load configuration
    try:
        load_config()
    except ValueError as e:
        print(f"Error: {e}")
        return

    data_dir = "./data"
    persist_dir = "./faiss_db"

    # Step 1: Ingestion / Indexing
    if args.index:
        print("Starting ingestion process...")
        documents = load_documents(data_dir)
        if not documents:
            print("No documents found to process. Please add PDFs to the data folder.")
            return

        chunks = split_documents(documents)
        vectorstore = create_or_load_vectorstore(chunks, persist_directory=persist_dir)
        print("Ingestion complete.")

    # Step 2: Querying
    elif args.query:
        print("Setting up for querying...")
        vectorstore = create_or_load_vectorstore(None, persist_directory=persist_dir)
        if not vectorstore:
             print("No existing vector store found. Please run with --index first.")
             return
             
        rag_chain = create_rag_chain(vectorstore)
        print(f"\nQuestion: {args.query}")
        
        response = rag_chain.invoke(args.query)
        print(f"Answer: {response}\n")

    else:
        print("Starting interactive mode. Type 'exit' or 'quit' to stop.")
        print("Setting up for querying...")
        vectorstore = create_or_load_vectorstore(None, persist_directory=persist_dir)
        if not vectorstore:
             print("No existing vector store found. Please run with --index first.")
             return

        rag_chain = create_rag_chain(vectorstore)

        # Conversation memory: keeps track of Q&A pairs in the session
        chat_history = []

        while True:
            try:
                user_question = input("\nAsk a question: ")
                if user_question.lower() in ['exit', 'quit']:
                    print("Exiting...")
                    break

                if not user_question.strip():
                    continue

                # Fetch context and scores
                docs_with_scores = vectorstore.similarity_search_with_relevance_scores(user_question, k=4)

                print("\n--- Retrieved Context Accuracy ---")
                if not docs_with_scores:
                    print("No context found.")
                else:
                    for i, (doc, score) in enumerate(docs_with_scores):
                        print(f"Doc {i+1}: {score*100:.1f}%")
                print("----------------------------------\n")

                context = "\n\n".join(doc.page_content for doc, _ in docs_with_scores)

                # Format conversation history as a readable string
                history_str = ""
                for turn in chat_history:
                    history_str += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

                # Stream the answer
                print("Answer: ", end="", flush=True)
                full_answer = ""
                for chunk in rag_chain.stream({
                    "context": context,
                    "question": user_question,
                    "chat_history": history_str
                }):
                    print(chunk, end="", flush=True)
                    full_answer += chunk
                print("\n")

                # Source citations
                if docs_with_scores:
                    print("📄 Sources:")
                    seen = set()
                    for doc, score in docs_with_scores:
                        source = os.path.basename(doc.metadata.get("source", "Unknown"))
                        page = doc.metadata.get("page", "?")
                        key = (source, page)
                        if key not in seen:
                            seen.add(key)
                            print(f"   • {source} — Page {page + 1} ({score*100:.1f}% match)")
                    print()

                # Append to conversation history
                chat_history.append({"question": user_question, "answer": full_answer})

            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break


if __name__ == "__main__":
    main()
