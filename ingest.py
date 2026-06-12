from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Database
from langchain_chroma import Chroma



# Step 1: Load PDF
import os

documents = []

for file in os.listdir("docs"):

    if file.endswith(".pdf"):

        loader = PyPDFLoader(
            os.path.join("docs", file)
        )

        documents.extend(
            loader.load()
        )

print(f"Total Pages: {len(documents)}")



# Step 2: Split into Chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

# Step 3: Create Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Test Embedding
vector = embedding_model.embed_query(
    chunks[0].page_content
)

print(f"Embedding Dimensions: {len(vector)}")



# Step 4: Store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Vector DB Created Successfully!")