from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -----------------------------
# LLM + LangChain Imports
# -----------------------------

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# RAG Imports


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Step 1: Load PDF


loader = PyPDFLoader("aws-guide.pdf")

# Returns a list of Document objects
documents = loader.load()

print(f"Total Pages: {len(documents)}")


# Step 2: Chunk Documents


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Converts pages into smaller chunks
chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")


# Step 3: Create Embedding Model


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Optional: Generate embedding for one chunk
vector = embedding_model.embed_query(
    chunks[0].page_content
)

print(f"Embedding Dimensions: {len(vector)}")

# Step 4: Create Chroma Vector DB


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)


# Step 5: Create Retriever


retriever = vectorstore.as_retriever()

# Test Retrieval
docs = retriever.invoke(
    "What is Amazon S3?"
)

print("\nRetrieved Chunk:\n")
print(docs[0].page_content)


# Step 6: Format Retrieved Docs


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# Step 7: Create Prompt


prompt = PromptTemplate.from_template(
"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""
)


# Step 8: Initialize LLM


llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


# Step 9: Output Parser


parser = StrOutputParser()


# Step 10: Build RAG Chain


rag_chain = (
    {
        # Question goes to retriever
        "context": retriever | format_docs,

        # Original question preserved
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

# Step 11: Ask Question


response = rag_chain.invoke(
    "What is Amazon S3?"
)

print("\nFinal Answer:\n")
print(response)