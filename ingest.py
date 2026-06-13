from dotenv import load_dotenv
load_dotenv()

import os
import shutil

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Database
from langchain_chroma import Chroma


# STEP 1: LOAD ALL PDFS
documents = []

for file in os.listdir("docs"):

    if file.endswith(".pdf"):

        pdf_path = os.path.join("docs", file)

        print(f"\nLoading: {pdf_path}")

        loader = PyPDFLoader(pdf_path)

        pdf_docs = loader.load()

        print(f"Pages Loaded: {len(pdf_docs)}")

        documents.extend(pdf_docs)

print("\n" + "=" * 50)
print(f"TOTAL PAGES LOADED: {len(documents)}")
print("=" * 50)

# Show metadata of loaded pages
print("\n===== LOADED PAGES =====")

for doc in documents:
    print(
        f"Source: {doc.metadata.get('source')} | "
        f"Page: {doc.metadata.get('page')}"
    )

print("========================\n")


# STEP 2: SPLIT INTO CHUNKS
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("\n" + "=" * 50)
print(f"TOTAL CHUNKS CREATED: {len(chunks)}")
print("=" * 50)

# Inspect first few chunks
print("\n=FIRST 20 CHUNKS ")

for i, chunk in enumerate(chunks[:20]):

    print(
        f"\nChunk {i+1}"
        f"\nSource: {chunk.metadata.get('source')}"
        f"\nPage: {chunk.metadata.get('page')}"
    )

print("============\n")



# STEP 3: CREATE EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Test embedding
vector = embedding_model.embed_query(
    chunks[0].page_content
)

print(f"\nEmbedding Dimensions: {len(vector)}")



# STEP 4: REBUILD CHROMA DATABASE
if os.path.exists("./chroma_db"):
    print("\nDeleting old Chroma DB...")
    shutil.rmtree("./chroma_db")

print("Creating fresh Chroma DB...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("\nVector DB Created Successfully!")



# STEP 5: VERIFY STORED DOCUMENTS
print("\nVerifying Chroma Storage...")

test_results = vectorstore.similarity_search(
    "Amazon S3 security",
    k=5
)

for i, doc in enumerate(test_results):

    print(f"\nResult {i+1}")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Page: {doc.metadata.get('page')}")
    print("-" * 40)
    print(doc.page_content[:300])