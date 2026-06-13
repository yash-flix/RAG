# 🚀 DocMind - Multi-Document RAG Assistant

DocMind is a Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents, build a searchable knowledge base, and ask natural language questions across all uploaded documents.

The application combines semantic search, reranking, conversational memory, and Large Language Models to generate accurate, source-grounded responses.

---

## 🌐 Live Demo

https://yav76anyd3xn4w6dcmrvb8.streamlit.app/

---

## 📂 GitHub Repository

https://github.com/yash-flix/DocMind

---

## 📌 Problem Statement

Large Language Models are powerful, but they cannot access private or custom documents unless that information is provided during inference.

For example:

* AWS Documentation
* Research Papers
* Company Policies
* Legal Documents
* Study Notes
* Medical Reports

A traditional LLM cannot reliably answer questions based on these documents because they were not part of its training data.

DocMind solves this problem using Retrieval-Augmented Generation (RAG).

Instead of relying only on the model's knowledge, the application retrieves relevant information from uploaded documents and uses that context to generate answers.

---

# 🏗️ Architecture

PDF Uploads
↓
Document Loading
↓
Text Chunking
↓
HuggingFace Embeddings
(all-MiniLM-L6-v2)
↓
Chroma Vector Database
↓
User Query
↓
Query Rewriting (LangGraph)
↓
Semantic Retrieval
↓
Reranking
↓
Groq Llama 3.3 70B
↓
Grounded Response + Sources

---

# ⚙️ Workflow

### 1. Upload PDFs

Users can upload one or more PDF documents.

### 2. Build Knowledge Base

The documents are:

* Loaded
* Split into chunks
* Embedded using HuggingFace Embeddings
* Stored inside ChromaDB

### 3. Ask Questions

Users can ask questions in natural language.

Example:

"What is Amazon S3?"

### 4. Query Rewriting

LangGraph rewrites ambiguous questions using conversation history to improve retrieval quality.

### 5. Retrieval

Relevant chunks are retrieved from the vector database using semantic similarity search.

### 6. Reranking

Retrieved chunks are reranked based on relevance.

Only the highest quality context is passed to the LLM.

### 7. Generation

Groq's Llama 3.3 70B model generates a final answer grounded in the retrieved documents.

---

# ✨ Features

* Multi-PDF Upload
* Conversational RAG
* Query Rewriting
* Semantic Search
* Context-Aware Retrieval
* Reranking Pipeline
* Source Referencing
* LangGraph Workflow
* Modern Streamlit UI
* Real-Time Document Question Answering

---

# 🛠️ Tech Stack

### Frontend

* Streamlit

### LLM

* Groq
* Llama 3.3 70B Versatile

### RAG Framework

* LangChain
* LangGraph

### Vector Database

* ChromaDB

### Embeddings

* HuggingFace Embeddings
* all-MiniLM-L6-v2

### Language

* Python

---

# 📁 Project Structure

```bash
DocMind/
│
├── app.py
│
├── rag/
│   ├── graph.py
│   ├── ingest.py
│   ├── llm.py
│   ├── prompts.py
│   ├── reranker.py
│   ├── state.py
│   └── embeddings.py
│
├── uploads/
├── chroma_db/
│
├── requirements.txt
├── .env
└── README.md
```

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yash-flix/DocMind.git
cd DocMind
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

### Run Application

```bash
streamlit run app.py
```

---

# 🎯 Use Cases

### Education

* Study assistants
* Course material search
* PDF note querying

### Enterprise

* Internal knowledge bases
* Employee onboarding
* Company policy assistants

### Research

* Research paper analysis
* Literature review assistants

### Legal

* Contract analysis
* Policy search systems

### Healthcare

* Medical document assistants
* Clinical knowledge retrieval

---

# 📈 Future Improvements

* Hybrid Search (BM25 + Vector Search)
* Persistent Cloud Storage
* User Authentication
* Citation Highlighting
* Multi-Modal Support
* Image Understanding
* Agentic Workflows
* AWS Bedrock Integration
* Advanced Evaluation Metrics

---

# 👨‍💻 Author

Yash Rane

Artificial Intelligence & Data Science Engineer

Passionate about AI, RAG Systems, LLM Applications, Backend Development, and Cloud Technologies.

LinkedIn:
https://www.linkedin.com/in/yash-rane

GitHub:
https://github.com/yash-flix

---

⭐ If you found this project interesting, consider giving it a star!
