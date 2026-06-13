from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_chroma import Chroma

from rag.embeddings import embedding_model


def create_vectorstore(pdf_paths):

    documents = []

    for pdf_path in pdf_paths:

        loader = PyPDFLoader(pdf_path)

        documents.extend(
            loader.load()
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(
        documents
    )

    vectorstore = Chroma.from_documents(
        persist_directory="./chroma_db",    
        documents=chunks,
        embedding=embedding_model
    )

    return vectorstore