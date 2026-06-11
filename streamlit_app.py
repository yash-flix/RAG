import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("RAG Question Answering")

st.write("Ask questions about the AWS guide. The answer is produced using your existing RAG pipeline.")

# Import the pipeline initializer from app.py. This lets us control initialization.
try:
    from app import initialize_pipeline
except Exception as e:
    st.error(f"Failed to import initializer from app.py: {e}")
    raise

# Initialize pipeline (will build or load persisted vectorstore)
with st.spinner("Initializing RAG pipeline (may take a minute)..."):
    try:
        rag_chain, retriever, format_docs = initialize_pipeline()
    except Exception as e:
        st.error(f"Pipeline initialization failed: {e}")
        raise

question = st.text_input("Enter your question", value="What is Amazon S3?")
use_sources = st.checkbox("Show retrieved source documents", value=True)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant documents..."):
            try:
                docs = retriever.invoke(question)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                docs = []

        if use_sources and docs:
            st.subheader("Retrieved documents")
            for i, d in enumerate(docs[:5], start=1):
                meta = getattr(d, "metadata", {}) or {}
                source = meta.get("source") or meta.get("file_path") or meta.get("path") or meta.get("source_document", "")
                st.markdown(f"**Doc {i}** — source: {source}")
                st.code(d.page_content[:1000] + ("..." if len(d.page_content) > 1000 else ""))

        with st.spinner("Answering with RAG chain..."):
            try:
                answer = rag_chain.invoke(question)
            except Exception as e:
                st.error(f"RAG chain failed: {e}")
            else:
                st.subheader("Answer")
                st.write(answer)

        if use_sources and docs:
            st.subheader("Combined context used (preview)")
            try:
                context = format_docs(docs)
                st.text_area("Context", value=context[:10000], height=300)
            except Exception:
                pass

st.markdown("---")
st.markdown("Run this app with: `streamlit run streamlit_app.py`")
