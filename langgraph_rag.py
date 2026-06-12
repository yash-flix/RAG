from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START , 
    END
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LLM
from langchain_groq import ChatGroq

# Prompt
from langchain_core.prompts import PromptTemplate

# Output Parser
from langchain_core.output_parsers import StrOutputParser

# Runnable Components
from langchain_core.runnables import RunnablePassthrough
 

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

print("Vector DB loaded")

retriever = vectorstore.as_retriever(
    search_type =  "mmr",
    search_kwargs = {"k" : 5 , "fetch_k": 20}
)




def format_docs(docs):

    formatted = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page_label",
            "Unknown"
        )

        formatted.append(
            f"""
Source: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted)

prompt = PromptTemplate.from_template(
"""
You are an AWS expert.

Answer ONLY using the provided context.

If the answer is not present in the context,
respond with:

"I could not find this information in the document."

At the end of your answer,
mention the source document and page number used.

Context:
{context}

Question:
{question}
"""
)
#Step 6: Load LLM

llm = ChatGroq(
    model = "llama-3.3-70b-versatile"
)

#Step 7 : Output parser 
parser = StrOutputParser()

#LangGraph state graph
class State(TypedDict):
    question:str
    rewritten_question : str
    context:str
    answer:str
    relevance_score: float

def retrieve(state: State):

    print("\n--- RETRIEVE NODE ---\n")

    print("Incoming State:")
    print(state)

    results = vectorstore.similarity_search_with_score(
        state["rewritten_question"],
        k=5
    )

    print("\nSimilarity Scores:\n")

    for i, (doc, score) in enumerate(results):
        print(f"Chunk {i+1}: {score}")

    best_score = results[0][1]

    docs = [
        doc
        for doc, score in results
    ]

    context = format_docs(docs)

    return {
        "context": context,
        "relevance_score": best_score
    }

def generate(state:State):
    print("Generating answer...")
    
    formated_prompt = prompt.invoke(
        {
            "question": state["question"],
             "context": state["context"]
        }
    )
    response = llm.invoke(formated_prompt)

    answer = parser.invoke(response)

    return {
        "answer": answer
    }

rewrite_prompt = PromptTemplate.from_template(
"""
You are a query rewriting assistant.

Convert the user's question into a standalone question
that can be understood without previous conversation.

If the question is already standalone,
return it unchanged.

Question:
{question}
"""
)

def rewrite_query(state: State):
    print("Rewriting question...")

    formated_prompt = rewrite_prompt.invoke(
        {
            "question": state["question"]
        }
    )
    response = llm.invoke(formated_prompt)

    rewritten_question = parser.invoke(response)

    return {
        "rewritten_question" : rewritten_question
    }

graph_builder = StateGraph(State)

def not_found(state: State):

    return {
        "answer":
        "I could not find relevant information in the document."
    }

def route_question(state: State):

    score = state["relevance_score"]

    print(f"\nRelevance Score: {score}")

    if score < 0.8:
        return "generate"

    return "not_found"


# Nodes

graph_builder.add_node(
    "rewrite",
    rewrite_query
)

graph_builder.add_node(
    "retrieve",
    retrieve
)

graph_builder.add_node(
    "generate",
    generate
)
graph_builder.add_node(
    "not_found",
    not_found
)

# Edges

graph_builder.add_edge(
    START,
    "rewrite"
)

graph_builder.add_edge(
    "rewrite",
    "retrieve"
)

graph_builder.add_conditional_edges(
    "retrieve",
    route_question,
    {
        "generate": "generate",
        "not_found": "not_found"
    }
)

graph_builder.add_edge(
    "generate",
    END
)

graph_builder.add_edge(
    "not_found",
    END
)


app = graph_builder.compile()

print("Graph compiled successfully")
print("Invoking graph...")

result = app.invoke(
    {
        "question": "What is Kubernetes?"
    }
)
print(result)



