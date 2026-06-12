from dotenv import load_dotenv

load_dotenv()

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

#Step 8: Build RAG Chain
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

# Step 9: Chat Loop

while True:

    question = input("\nAsk Question (type 'exit' to quit): ")

    if question.strip().lower() in [
        "exit",
        "quit",
        "bye"
    ]:
        break

    retrieved_docs = retriever.invoke(question)

    print("\nRETRIEVED DOCS\n")

    for i, doc in enumerate(retrieved_docs):

        print(f"Chunk {i+1}")
        print(f"Page: {doc.metadata.get('page_label')}")

        print(doc.page_content[:500])

        print("\n" + "=" * 50 + "\n")

    print("\nFinal Answer:\n")

    response = chain.invoke(question)

    print(response)