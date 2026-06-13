from typing import TypedDict, Annotated, List
import operator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages:            Annotated[List[BaseMessage], operator.add]

    question:            str
    rewritten_question:  str


    reranked_context: str

    docs : list[Document]

    answer:              str
    relevance_score:     float
    