from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

parser = StrOutputParser()