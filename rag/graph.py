from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from rag.state import State
from rag.prompts import (
    SYSTEM_PROMPT,
    rewrite_prompt
)

from rag.llm import (
    llm,
    parser
)

from rag.reranker import reranker

RERANK_THRESHOLD = 1.5


def build_graph(vectorstore):

    # ==========================================
    # Helper
    # ==========================================

    def format_docs(docs):

        formatted = []

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = (
                doc.metadata.get("page_label")
                or doc.metadata.get("page")
                or "Unknown"
            )

            formatted.append(
                f"Source: {source}\n"
                f"Page: {page}\n\n"
                f"Content:\n{doc.page_content}"
            )

        return "\n\n".join(formatted)

    # ==========================================
    # Rewrite Node
    # ==========================================

    def rewrite_query(state: State):

        print("\n===== REWRITE NODE =====")

        history_lines = []

        for msg in state["messages"]:

            role = (
                "User"
                if isinstance(msg, HumanMessage)
                else "Assistant"
            )

            history_lines.append(
                f"{role}: {msg.content}"
            )

        history_text = (
            "\n".join(history_lines)
            if history_lines
            else "None"
        )

        print("\n===== HISTORY =====")
        print(history_text)

        formatted_prompt = rewrite_prompt.invoke(
            {
                "question": state["question"],
                "history": history_text
            }
        )

        response = llm.invoke(
            formatted_prompt
        )

        rewritten = parser.invoke(
            response
        )

        print("\n===== REWRITTEN QUESTION =====")
        print(rewritten)

        return {
            "rewritten_question": rewritten
        }

    # ==========================================
    # Retrieve Node
    # ==========================================

    def retrieve(state: State):

        print("\n===== RETRIEVE NODE =====")

        results = (
            vectorstore.similarity_search_with_score(
                state["rewritten_question"],
                k=30
            )
        )

        print("\n===== RETRIEVED DOCS =====")

        for i, (doc, score) in enumerate(
            results[:10]
        ):

            print(
                f"\nChunk {i+1}"
            )

            print(
                f"Score: {score:.4f}"
            )

            print(
                f"Source: {doc.metadata.get('source')}"
            )

            print(
                doc.page_content[:200]
            )

        docs = [
            doc
            for doc, _ in results
        ]

        return {
            "docs": docs
        }

    # ==========================================
    # Rerank Node
    # ==========================================

    def rerank(state: State):

        print("\n===== RERANK NODE =====")

        query = state["rewritten_question"]

        docs = state["docs"]

        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        scores = reranker.predict(
            pairs
        )

        ranked = list(
            zip(docs, scores)
        )

        ranked.sort(
            key=lambda x: x[1],
            reverse=True
        )

        print(
            "\n===== TOP RERANKED DOCS ====="
        )

        for i, (doc, score) in enumerate(
            ranked[:10]
        ):

            print(
                f"\nRank {i+1}"
            )

            print(
                f"Score: {score:.4f}"
            )

            print(
                f"Source: {doc.metadata.get('source')}"
            )

            print(
                f"Page: {doc.metadata.get('page')}"
            )

        best_rerank_score = ranked[0][1]

        top_docs = [
            doc
            for doc, score in ranked[:10]
        ]

        context = format_docs(
            top_docs
        )

        return {
            "reranked_context": context,
            "docs": top_docs,
            "relevance_score": float(
                best_rerank_score
            )
        }

    # ==========================================
    # Generate Node
    # ==========================================

    def generate(state: State):

        print(
            "\n===== GENERATE NODE ====="
        )

        current_turn = HumanMessage(
            content=f"""
Context from documents:

{state['reranked_context']}

Question:
{state['rewritten_question']}
"""
        )

        messages_for_llm = (
            [SYSTEM_PROMPT]
            + state["messages"]
            + [current_turn]
        )

        response = llm.invoke(
            messages_for_llm
        )

        answer = parser.invoke(
            response
        )

        print(
            f"\nAnswer:\n{answer}"
        )

        return {
            "answer": answer,
            "messages": [
                HumanMessage(
                    content=state["question"]
                ),
                AIMessage(
                    content=answer
                )
            ]
        }

    # ==========================================
    # Not Found Node
    # ==========================================

    def not_found(state: State):

        print(
            "\n===== NOT FOUND NODE ====="
        )

        answer = (
            "I could not find relevant "
            "information in the document."
        )

        return {
            "answer": answer,
            "messages": [
                HumanMessage(
                    content=state["question"]
                ),
                AIMessage(
                    content=answer
                )
            ]
        }

    # ==========================================
    # Router
    # ==========================================

    def route_question(state: State):

        score = state["relevance_score"]

        print(
            f"\nBest reranker score: {score:.4f}"
        )

        if score > RERANK_THRESHOLD:

            print(
                "→ GENERATE"
            )

            return "generate"

        print(
            "→ NOT FOUND"
        )

        return "not_found"

    # ==========================================
    # Build Graph
    # ==========================================

    graph_builder = StateGraph(State)

    graph_builder.add_node(
        "rewrite",
        rewrite_query
    )

    graph_builder.add_node(
        "retrieve",
        retrieve
    )

    graph_builder.add_node(
        "rerank",
        rerank
    )

    graph_builder.add_node(
        "generate",
        generate
    )

    graph_builder.add_node(
        "not_found",
        not_found
    )

    graph_builder.add_edge(
        START,
        "rewrite"
    )

    graph_builder.add_edge(
        "rewrite",
        "retrieve"
    )

    graph_builder.add_edge(
        "retrieve",
        "rerank"
    )

    graph_builder.add_conditional_edges(
        "rerank",
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

    memory = MemorySaver()

    return graph_builder.compile(
        checkpointer=memory
    )