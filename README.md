# Offline RAG System using FAISS + Ollama

A privacy-first **offline Retrieval-Augmented Generation (RAG)** system that allows users to chat with private PDF documents **without sending any data to cloud APIs**.

This system extracts text from PDFs (including scanned images using OCR), converts them into embeddings, stores them locally using **FAISS**, and answers questions using a **local LLM via Ollama**.

Built for environments where **data privacy matters**, such as:

- Law firms
- Medical clinics
- Accounting offices
- Internal enterprise documentation
- Research organizations

---

# Problem Statement

Many organizations deal with hundreds of confidential PDF documents.

Employees often waste hours manually searching through:

- Legal case files
- Medical reports
- Internal policy documents
- Financial records
- Research papers

Cloud AI tools may not be suitable because sensitive documents cannot be uploaded externally.

This project solves that problem by enabling:

✅ **Completely offline document chat**  
✅ **Local inference using Ollama**  
✅ **Private vector search**  
✅ **No cloud dependency**  
✅ **Secure document retrieval**

---

# What We Built

We built an **Offline RAG Pipeline** where users can:

1. Drop PDF files into a folder
2. Build a searchable local knowledge base
3. Ask questions in natural language
4. Receive answers generated from document context

The entire pipeline runs **locally on the machine**.

---

# System Architecture

```text
                     ┌──────────────────┐
                     │   PDF Documents  │
                     └────────┬─────────┘
                              │
                              ▼
                   ┌────────────────────┐
                   │ Text Extraction    │
                   │ PyMuPDF + OCR      │
                   └────────┬───────────┘
                            │
                            ▼
                   ┌────────────────────┐
                   │ Text Chunking      │
                   │ Overlapping Chunks │
                   └────────┬───────────┘
                            │
                            ▼
                   ┌────────────────────┐
                   │ Embedding Model    │
                   │ MiniLM-L6-v2       │
                   └────────┬───────────┘
                            │
                            ▼
                   ┌────────────────────┐
                   │ FAISS Vector DB    │
                   │ Local Similarity   │
                   └────────┬───────────┘
                            │
                            ▼
                   ┌────────────────────┐
                   │ Ollama             │
                   │ llama3.2:3b        │
                   └────────┬───────────┘
                            │
                            ▼
                     Natural Language
                         Response
```

---

# Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend logic |
| PyMuPDF | PDF text extraction |
| Tesseract OCR | Extract text from scanned images |
| Sentence Transformers | Embedding generation |
| FAISS | Vector similarity search |
| Ollama | Local LLM inference |
| llama3.2:3b | Local language model |
| Docker | Deployment |

---

# Step-by-Step Pipeline

## 1. PDF Extraction

We first extract text from PDFs using **PyMuPDF**.

### Why PyMuPDF?

We selected PyMuPDF because:

- Fast
- Lightweight
- Accurate for structured PDFs
- Can extract embedded images

Example:

```python
page_text = page.get_text("text")
```

---

## 2. OCR for Scanned PDFs

Many PDFs contain:

- Flowcharts
- Images
- Scanned text
- Diagrams

Normal PDF extraction cannot read text inside images.

We use **Tesseract OCR** to extract text from embedded images.

Example:

```python
extracted_text = pytesseract.image_to_string(image)
```

### Why OCR?

Without OCR:

```text
Flowchart image → ignored
```

With OCR:

```text
Flowchart image → converted into searchable text
```

This improves retrieval quality significantly.

---

## 3. Chunking Strategy

Large PDFs cannot be embedded directly.

We split text into smaller chunks.

We used:

```text
Chunk Size = 500 characters
Overlap = 100 characters
```

### Why overlap?

Without overlap:

```text
Chunk 1:
"The patient should take"

Chunk 2:
"medicine twice daily"
```

Meaning gets broken.

With overlap:

```text
Chunk 1:
"The patient should take medicine"

Chunk 2:
"take medicine twice daily"
```

Context is preserved.

---

## 4. Embedding Generation

Each chunk is converted into vector embeddings.

Model used:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Why this model?

Because it is:

- Lightweight
- Fast
- CPU friendly
- Works on low RAM systems
- Good semantic understanding

Especially suitable for:

```text
8GB RAM laptops
```

---

## 5. Vector Search using FAISS

Embeddings are stored inside **FAISS**.

We use:

```python
faiss.IndexFlatL2
```

This allows similarity search.

When user asks a question:

1. Query gets embedded
2. FAISS searches nearest chunks
3. Relevant chunks are returned

---

# Why We Did NOT Use Fancy Vector Databases

Many modern RAG systems use:

- ChromaDB
- Pinecone
- Weaviate
- Qdrant

We intentionally avoided them.

## Why?

### 1. This Project is Offline First

Our goal was:

```text
Zero cloud dependency
```

Many vector databases are designed around:

```text
Server deployment
Cloud APIs
Persistent services
```

We wanted something lightweight.

---

### 2. FAISS is Faster for Local Systems

For local search:

```text
FAISS is extremely fast
```

It works directly in memory.

No extra service required.

No background database process.

Just:

```python
index.search()
```

---

### 3. Lower Resource Usage

Laptop Specs:

```text
8GB RAM
Linux partition
```

Running:

```text
ChromaDB + Ollama + OCR + Embedding model
```

could become heavy.

FAISS is lightweight and efficient.

---

### 4. Simpler Deployment

With FAISS:

```text
Single binary file
```

Example:

```text
faiss_index.bin
```

Easy to move.

Easy to deploy.

Easy to rebuild.

---

# Why Ollama?

We needed:

```text
Private local inference
```

Instead of:

```text
OpenAI API
Claude API
Gemini API
```

we used:

```text
Ollama
```

Benefits:

✅ Runs locally  
✅ No API cost  
✅ Privacy friendly  
✅ Works offline  
✅ Easy model switching

Example:

```bash
ollama pull llama3.2:3b
```

---

# Why llama3.2:3b?

Chosen because:

- Lightweight
- Good quality responses
- Runs on 8GB RAM laptops
- Fast inference
- Suitable for local deployment

Tradeoff:

```text
Smaller model
=
Slightly lower reasoning quality
```

But:

```text
Much faster + lower memory
```

for local systems.

---

# Project Structure

```text
Rag/
│── documents/
│── rag_build.py
│── rag_chat.py
│── requirements.txt
│── Dockerfile
│── docker-compose.yml
│── faiss_index.bin
│── chunks.pkl
│── README.md
│── .gitignore
```

### Folder Explanation

### `documents/`

Stores PDFs.

### `rag_build.py`

Builds knowledge base:

```text
PDF → OCR → Chunking → Embeddings → FAISS
```

### `rag_chat.py`

Chat interface for querying documents.

### `faiss_index.bin`

Stores vector embeddings.

### `chunks.pkl`

Stores chunk mappings.

---

# Installation

## Clone Repository

```bash
git clone YOUR_REPO_URL
cd Rag
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Install Ollama and pull model:

```bash
ollama pull llama3.2:3b
```

---

# Running the Project

## Step 1 — Add PDFs

Place PDFs inside:

```text
documents/
```

---

## Step 2 — Build Vector Database

```bash
python3 rag_build.py
```

This creates:

```text
faiss_index.bin
chunks.pkl
```

---

## Step 3 — Start Chat

```bash
python3 rag_chat.py
```

Ask questions:

```text
What is PM+?
Summarize chapter 3
What does WHO recommend?
```

---

# Docker Setup

## Build

```bash
docker compose build
```

---

## Run

```bash
docker compose run rag-app
```

---

# Example Query Flow

```text
User Question
        ↓
Embedding Generation
        ↓
FAISS Retrieval
        ↓
Top Relevant Chunks
        ↓
Prompt Construction
        ↓
Ollama (llama3.2:3b)
        ↓
Final Answer
```

---

# Limitations

Current limitations:

- CLI based interface
- No streaming responses
- Limited multi-document ranking
- OCR quality depends on image quality
- Small model reasoning limits

---

# Future Improvements

Planned upgrades:

- FastAPI backend
- React frontend
- Streaming responses
- Better reranking
- Hybrid search (keyword + semantic)
- Metadata filtering
- Multi-document indexing
- GPU acceleration
- User authentication

---

# Portfolio Impact

This project demonstrates:

### AI Engineering

- Embeddings
- RAG pipelines
- Prompt engineering
- Local inference

### Backend Engineering

- Data pipelines
- Vector search
- OCR processing
- Docker deployment

### Production Skills

- Offline architecture
- Privacy-first systems
- Local deployment
- Efficient resource management

---

# Author

**Yash Khandelwal**

Built as a privacy-first AI system for secure document intelligence.
