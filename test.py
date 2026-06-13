from rag.ingest import create_vectorstore

pdfs = [
    "docs/aws-guide.pdf",
    "docs/ec2-guide.pdf",
    "docs/s3-guide.pdf"
]

vectorstore = create_vectorstore(pdfs)

print("Vectorstore Created Successfully")

results = vectorstore.similarity_search(
    "What is Amazon S3?",
    k=3
)

for doc in results:
    print("\n----------------")
    print(doc.page_content[:300])