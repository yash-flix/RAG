
from dotenv import load_dotenv
load_dotenv()

#  Imports 

from typing import TypedDict, Annotated, List
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Step 1: State schema 

class State(TypedDict):
    messages:            Annotated[List[BaseMessage], operator.add]
    question:            str
    rewritten_question:  str
    context:             str
    answer:              str
    relevance_score:     float


#  Step 2: Load components 

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
print("Vector DB loaded")

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

llm    = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# ChromaDB L2 distance threshold.
# Lower = closer match. Scores below this are considered relevant.
RELEVANCE_THRESHOLD = 0.8

SYSTEM_PROMPT = SystemMessage(content="""
You are an AWS expert.
Answer ONLY using the provided context.
If the answer is not present in the context, say:
"I could not find this information in the document."
At the end of your answer, mention the source document and page number used.
""")


#  Step 3: Helper 

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page_label", "Unknown")
        formatted.append(
            f"Source: {source}\nPage: {page}\n\nContent:\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


#  Step 4: Nodes 

# Node 1: rewrite_query 

rewrite_prompt = PromptTemplate.from_template("""
You are a query rewriting assistant.

Given the conversation history and the user's latest question,
rewrite the question into a fully standalone question that can be
understood without the history.

If the question is already standalone, return it unchanged.
Return ONLY the rewritten question — no explanation.

Conversation History:
{history}

Latest Question:
{question}
""")

def rewrite_query(state: State) -> dict:
    print("\n--- REWRITE NODE ---")

    # Build a readable history string from past messages
    history_lines = []
    for msg in state["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {msg.content}")
    history_text = "\n".join(history_lines) if history_lines else "None"

    formatted = rewrite_prompt.invoke({
        "question": state["question"],
        "history":  history_text
    })
    response = llm.invoke(formatted)
    rewritten = parser.invoke(response)

    print(f"Original : {state['question']}")
    print(f"Rewritten: {rewritten}")

    return {"rewritten_question": rewritten}


# Node 2: retrieve 
#

def retrieve(state: State) -> dict:
    print("\n--- RETRIEVE NODE ---")

    results = vectorstore.similarity_search_with_score(
        state["rewritten_question"],
        k=5
    )

    print("\nSimilarity Scores (L2 distance — lower = more relevant):")
    for i, (doc, score) in enumerate(results):
        print(f"  Chunk {i+1}: {score:.4f}")

    best_score = results[0][1]  # lowest distance = most relevant chunk
    docs = [doc for doc, _ in results]

    return {
        "context":  format_docs(docs),
        "relevance_score": best_score
    }


#  Node 3: generate 

def generate(state: State) -> dict:
    print("\n GENERATE NODE ")

    history = state["messages"]

    # Combine retrieved context + rewritten question as the latest human turn
    current_turn = HumanMessage(content=f"""
Context from documents:
{state['context']}

Question: {state['rewritten_question']}
""")

    # LLM sees: system prompt + full history + current turn
    messages_for_llm = [SYSTEM_PROMPT] + history + [current_turn]
    response         = llm.invoke(messages_for_llm)
    answer           = parser.invoke(response)

    print(f"\nAnswer:\n{answer}")


    return {
        "answer": answer,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer)
        ]
    }


#  Node 4: not_found 

def not_found(state: State) -> dict:
    print("\n--- NOT FOUND NODE ---")
    answer = "I could not find relevant information in the document."
    print(answer)

    # Still append to history so the LLM knows this question was unanswerable
    return {
        "answer": answer,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer)
        ]
    }


# Step 5: Router 

def route_question(state: State) -> str:
    score = state["relevance_score"]
    print(f"\nRouting — best L2 score: {score:.4f}  (threshold: {RELEVANCE_THRESHOLD})")

    if score < RELEVANCE_THRESHOLD:
        print("→ Relevant docs found → generate")
        return "generate"

    print("→ No relevant docs found → not_found")
    return "not_found"


#  Step 6: Build graph 

graph_builder = StateGraph(State)

graph_builder.add_node("rewrite",   rewrite_query)
graph_builder.add_node("retrieve",  retrieve)
graph_builder.add_node("generate",  generate)
graph_builder.add_node("not_found", not_found)

graph_builder.add_edge(START,      "rewrite")
graph_builder.add_edge("rewrite",  "retrieve")

graph_builder.add_conditional_edges(
    "retrieve",
    route_question,
    {
        "generate":  "generate",
        "not_found": "not_found"
    }
)

graph_builder.add_edge("generate",  END)
graph_builder.add_edge("not_found", END)

memory = MemorySaver()
app    = graph_builder.compile(checkpointer=memory)
print("Graph compiled successfully\n")


# Step 7: Chat loop

config = {"configurable": {"thread_id": "user_1"}}

print("AWS RAG Chatbot  |  memory + query rewriting + relevance gating")
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
            "messages": [],          # MemorySaver merges with saved history
            "rewritten_question": "",
            "context": "",
            "answer": "",
            "relevance_score": 0.0
        },
        config=config
    )

    print(f"\n{'─'*55}")
    print(result["answer"])
    print(f"{'─'*55}\n")