# 📄 RAG Project — AI Document Q&A System

A **Retrieval-Augmented Generation (RAG)** application that lets you ask questions about your PDF documents using **GPT-4o** and **FAISS** vector search.

---

## ✨ Features

- 📥 **Ingest multiple PDFs** from a `data/` folder in one command
- ✂️ **Smart chunking** with overlap to preserve context at boundaries
- 🔢 **OpenAI Embeddings** for high-quality semantic search
- ⚡ **FAISS vector store** — fast, local, no cloud DB required
- ➕ **Incremental indexing** — add new PDFs without re-indexing old ones
- 💬 **Interactive Q&A** with ChatGPT-style streaming responses
- 🧠 **Conversation memory** — the AI remembers your earlier questions in the same session
- 📄 **Source citations** — every answer shows which PDF and page it came from
- 📊 **Accuracy scores** — displays relevance % for each retrieved chunk

---

## 🗂️ Project Structure

```
RAG_Project/
├── main.py                  # Entry point — indexing & interactive Q&A
├── requirements.txt         # Python dependencies
├── .env                     # Your API key (never commit this!)
├── .env.example             # Template for .env
├── data/                    # Drop your PDF files here (gitignored)
├── faiss_db/                # Auto-generated vector store (gitignored)
└── src/
    ├── config.py            # Loads environment variables
    ├── document_loader.py   # Reads PDFs from data/ folder
    ├── text_splitter.py     # Chunks documents for embedding
    ├── vectorstore.py       # FAISS index creation, loading & merging
    └── rag_chain.py         # GPT-4o prompt + streaming chain
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
```bash
# Copy the example and fill in your key
cp .env.example .env
```
Open `.env` and set:
```
OPENAI_API_KEY=sk-...your-key-here...
```

### 4. Add your PDFs
Create a `data/` folder and place your PDF files inside:
```
data/
  ├── my_document.pdf
  └── another_file.pdf
```

### 5. Index your PDFs
```bash
python main.py --index
```

### 6. Start asking questions
```bash
python main.py
```

---

## 💡 Usage

| Command | Description |
|---|---|
| `python main.py --index` | Index all PDFs in `./data/` (incremental) |
| `python main.py` | Start interactive Q&A session |
| `python main.py --query "Your question"` | One-shot query from terminal |

Type `exit` or `quit` inside the interactive session to stop.

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | FAISS (faiss-cpu) |
| Framework | LangChain |
| PDF Loader | PyPDFLoader |
