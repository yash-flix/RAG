from dotenv import load_dotenv

load_dotenv()

import os
import streamlit as st

from rag.ingest import create_vectorstore
from rag.graph import build_graph
import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)


st.set_page_config(
    page_title="Multi Document AI Assistant",
    page_icon="🤖",
    layout="wide"
)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "app_graph" not in st.session_state:
    st.session_state.app_graph = None



st.title("🤖 Multi Document AI Assistant")

st.write(
    "Upload PDF files and chat with them."
)

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Build Knowledge Base"):

    if not uploaded_files:
        st.error("Please upload PDF files.")
        st.stop()

    pdf_paths = []

    os.makedirs("uploads", exist_ok=True)

    with st.spinner("Processing PDFs..."):

        for uploaded_file in uploaded_files:

            save_path = os.path.join(
                "uploads",
                uploaded_file.name
            )

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_paths.append(save_path)

        vectorstore = create_vectorstore(
            pdf_paths
        )

        app_graph = build_graph(
            vectorstore
        )

        st.session_state.app_graph = (
            app_graph
        )

    st.success(
        "Knowledge Base Created Successfully!"
    )

# Chat Section

st.divider()

st.subheader("💬 Chat With Your Documents")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input(
    "Ask a question about your documents..."
)
if user_question:

    if st.session_state.app_graph is None:
        st.error(
            "Please build a knowledge base first."
        )
        st.stop()

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            history = []

            for msg in st.session_state.messages:

                if msg["role"] == "user":

                    history.append(
                        HumanMessage(
                            content=msg["content"]
                        )
                    )

                else:

                    history.append(
                        AIMessage(
                            content=msg["content"]
                        )
                    )

            result = st.session_state.app_graph.invoke(
                {
                    "question": user_question,
                    "messages": history,
                    "rewritten_question": "",
                    "docs": [],
                    "reranked_context": "",
                    "answer": "",
                    "relevance_score": 0.0
                },
                config={
                    "configurable": {
                        "thread_id": "streamlit_user"
                    }
                }
            )

            answer = result["answer"]

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )