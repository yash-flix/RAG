import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

parser = StrOutputParser()

prompt = PromptTemplate.from_template(
"""
Answer the question only using
the provided context.

Context:
{context}

Question:
{question}
"""
)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def initialize_pipeline(pdf_path="aws-guide.pdf", persist_directory="chroma_db", rebuild=False):
    """Initialize or load the RAG pipeline and return (rag_chain, retriever, format_docs).

    On first run this builds embeddings and a Chroma vectorstore and persists it to
    `persist_directory`. On subsequent runs it will load the persisted vectorstore
    unless `rebuild=True`.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    # Load documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore: load if persisted, otherwise build and persist
    if os.path.isdir(persist_directory) and not rebuild:
        vectorstore = Chroma(persist_directory=persist_directory, embedding=embedding_model)
    else:
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persist_directory)
        try:
            vectorstore.persist()
        except Exception:
            # Some Chroma builds auto-persist; ignore if not supported
            pass

    retriever = vectorstore.as_retriever()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )

    return rag_chain, retriever, format_docs


if __name__ == "__main__":
    # When run directly, build the pipeline and run a sample question.
    rc, retriever, fmt = initialize_pipeline()
    print("Pipeline initialized. Running sample query...")
    try:
        resp = rc.invoke("What is Amazon S3?")
        print(resp)
    except Exception as e:
        print("Sample run failed:", e)
