# RAG Project Streamlit UI

This repository contains a Retrieval-Augmented Generation (RAG) pipeline and a Streamlit UI to ask questions against the loaded documents.

Run instructions:

1. Activate your virtualenv (PowerShell):

```powershell
.\.\venv\Scripts\Activate.ps1
```

2. Install Streamlit if you haven't already:

```powershell
python -m pip install streamlit
```

3. Run the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

Notes:
- The Streamlit app imports `rag_chain` and `retriever` from `app.py`. On first import the pipeline will initialize (PDF loading, embeddings, vectorstore building). That may take some time.
- If you prefer to separate initialization from the web process, refactor `app.py` to expose an initialization function which the Streamlit app can call explicitly.
