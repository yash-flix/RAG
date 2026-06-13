
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma

from rag.embeddings import embedding_model
from rag.graph import build_graph


# Load existing vector DB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

print("Vector DB loaded")


# Build LangGraph application
app = build_graph(vectorstore)

print("Graph compiled successfully\n")


# Chat Loop
config = {"configurable": {"thread_id": "user_1"}}

print("Document Assistant")
print("Type 'exit' to quit\n")

while True:

    question = input("Ask: ").strip()

    if not question:
        continue

    if question.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    result = app.invoke(
        {
            "question": question,
            "messages": [],
            "rewritten_question": "",
            "docs": [],
            "reranked_context": "",
            "answer": "",
            "relevance_score": 0.0
        },
        config=config
    )

    print(f"\n{'─'*55}")
    print(result["answer"])
    print(f"{'─'*55}\n")