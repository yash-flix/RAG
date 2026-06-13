from dotenv import load_dotenv

load_dotenv()

import os
import streamlit as st

from rag.ingest import create_vectorstore
from rag.graph import build_graph

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

st.set_page_config(
    page_title="DocMind",
    page_icon="◆",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600&family=Geist+Mono:wght@400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Global base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: #0a0a0b !important;
    font-family: 'Geist', system-ui, sans-serif !important;
    color: #d4d4d8 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Force sidebar always visible, never collapsed ── */
[data-testid="stSidebar"] {
    background: #0f0f10 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
    width: 260px !important;
    min-width: 260px !important;
    transform: none !important;
    visibility: visible !important;
    display: flex !important;
    flex-direction: column !important;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important;
    margin-left: 0 !important;
    visibility: visible !important;
}

/* Hide the collapse/expand toggle button */
[data-testid="collapsedControl"],
button[kind="header"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
    overflow-x: hidden !important;
    width: 260px !important;
}

/* ── Brand bar ── */
.dm-brand {
    padding: 22px 20px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.dm-wordmark {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: #fafafa;
    display: flex;
    align-items: center;
    gap: 12px;
    line-height: 1;
    margin-bottom: 6px;
}

.dm-wordmark .dm-logo {
    width: 36px; height: 36px;
    background: #fafafa;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    color: #09090b;
    font-weight: 700;
    flex-shrink: 0;
}

.dm-tagline {
    font-size: 10px;
    color: #3f3f46;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding-left: 48px;
}

/* ── Sidebar section ── */
.dm-section { padding: 20px 16px 10px; }

.dm-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #3f3f46;
    margin-bottom: 10px;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: #18181b !important;
    border: 1px dashed rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    padding: 18px 14px !important;
    transition: border-color 0.15s, background 0.15s !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(255,255,255,0.2) !important;
    background: #1c1c1f !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] > div,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #52525b !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 12px !important;
}

[data-testid="stFileUploader"] section {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #27272a !important;
    color: #a1a1aa !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    padding: 5px 12px !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #3f3f46 !important;
    color: #d4d4d8 !important;
}

/* ── Document pills ── */
.dm-doc-list { display: flex; flex-direction: column; gap: 4px; margin-top: 10px; }

.dm-doc-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: #18181b;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    font-size: 12px;
}

.dm-doc-pill .dm-doc-icon {
    width: 22px; height: 22px;
    background: #27272a;
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: 8px;
    font-weight: 700;
    flex-shrink: 0;
    color: #71717a;
}

.dm-doc-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
    color: #a1a1aa;
}

/* ── Build button ── */
.stButton > button {
    background: #fafafa !important;
    color: #09090b !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    width: 100% !important;
    letter-spacing: -0.1px !important;
    cursor: pointer !important;
    transition: background 0.1s !important;
    margin-top: 4px !important;
}

.stButton > button:hover { background: #e4e4e7 !important; }
.stButton > button:active { background: #d4d4d8 !important; }

/* ── Status chip ── */
.dm-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 11px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    margin-top: 4px;
    border: 1px solid rgba(255,255,255,0.06);
    background: #18181b;
    color: #a1a1aa;
}

.dm-status.dm-idle {
    background: transparent;
    color: #3f3f46;
    border-color: rgba(255,255,255,0.04);
}

.dm-status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dm-status.dm-ready .dm-status-dot { background: #fafafa; }
.dm-status.dm-idle .dm-status-dot  { background: #3f3f46; }

/* ── Divider ── */
.dm-rule {
    height: 1px;
    background: rgba(255,255,255,0.05);
    margin: 12px 0;
}

/* ── Session stats ── */
.dm-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    font-size: 12px;
}

.dm-stat-label { color: #3f3f46; }
.dm-stat-value { color: #71717a; font-weight: 500; }

/* ── Main content ── */
.main .block-container {
    padding: 40px 80px 120px !important;
    max-width: 1100px !important;
    width: 100% !important;
}

/* ── Page heading ── */
.dm-heading {
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.dm-heading h1 {
    font-size: 20px !important;
    font-weight: 500 !important;
    letter-spacing: -0.5px !important;
    color: #fafafa !important;
    line-height: 1.25 !important;
    margin-bottom: 6px !important;
}

.dm-heading p {
    font-size: 13px;
    color: #3f3f46;
    line-height: 1.6;
    max-width: 600px;
}

/* ── Empty state ── */
.dm-empty {
    padding: 64px 0 40px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}

.dm-empty-icon {
    width: 40px; height: 40px;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    color: #3f3f46;
    margin-bottom: 16px;
}

.dm-empty h2 {
    font-size: 15px;
    font-weight: 500;
    color: #3f3f46;
    margin-bottom: 6px;
}

.dm-empty p {
    font-size: 13px;
    color: #27272a;
    line-height: 1.65;
    max-width: 320px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0 0 24px !important;
    border: none !important;
    gap: 12px !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
[data-testid="stChatMessageContent"] {
    background: #18181b !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 11px 15px !important;
    color: #d4d4d8 !important;
    font-size: 13.5px !important;
    line-height: 1.7 !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
[data-testid="stChatMessageContent"] {
    background: transparent !important;
    border: none !important;
    border-left: 2px solid rgba(255,255,255,0.1) !important;
    border-radius: 0 !important;
    padding: 2px 0 2px 16px !important;
    color: #71717a !important;
    font-size: 13.5px !important;
    line-height: 1.75 !important;
}

[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    border-radius: 6px !important;
    width: 28px !important;
    height: 28px !important;
    font-size: 13px !important;
}

[data-testid="chatAvatarIcon-user"] {
    background: #2d1f3d !important;
    border: 1px solid #5b3a8a !important;
    color: #c084fc !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: #0f2a1e !important;
    border: 1px solid #166534 !important;
    color: #4ade80 !important;
}

/* ── Chat input bar ── */
[data-testid="stBottom"] {
    background: #0f0f10 !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    padding: 14px 80px !important;
    width: 100% !important;
}

/* The inner stChatInput wraps textarea + button in a single row — don't break it */
[data-testid="stChatInput"] {
    background: #18181b !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    width: 100% !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(255,255,255,0.22) !important;
}

[data-testid="stChatInputTextArea"] {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #d4d4d8 !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 13.5px !important;
    line-height: 1.55 !important;
    padding: 13px 16px !important;
    resize: none !important;
}

[data-testid="stChatInputTextArea"]::placeholder {
    color: #3f3f46 !important;
}

[data-testid="stChatInputSubmitButton"] button {
    background: #fafafa !important;
    border: none !important;
    border-radius: 7px !important;
    color: #09090b !important;
    width: 32px !important;
    height: 32px !important;
    cursor: pointer !important;
    transition: background 0.1s !important;
    margin-right: 8px !important;
}

[data-testid="stChatInputSubmitButton"] button:hover {
    background: #d4d4d8 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: #18181b !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-left: 2px solid #71717a !important;
    border-radius: 8px !important;
    color: #a1a1aa !important;
    font-size: 13px !important;
}

[data-testid="stAlert"][data-baseweb="notification"][kind="positive"] {
    border-left: 2px solid #fafafa !important;
    color: #d4d4d8 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #52525b !important; }
[data-testid="stSpinner"] > div {
    border-color: #3f3f46 transparent transparent transparent !important;
}

/* ── Custom avatar symbols ── */
[data-testid="chatAvatarIcon-user"] svg,
[data-testid="chatAvatarIcon-assistant"] svg {
    display: none !important;
}

[data-testid="chatAvatarIcon-user"]::after {
    content: "U";
    font-size: 11px;
    font-weight: 600;
    color: #c084fc;
    font-family: 'Geist', sans-serif;
}

[data-testid="chatAvatarIcon-assistant"]::after {
    content: "◆";
    font-size: 11px;
    color: #4ade80;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #27272a; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #3f3f46; }
</style>
""", unsafe_allow_html=True)

# ── Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "app_graph" not in st.session_state:
    st.session_state.app_graph = None

# ── Sidebar
with st.sidebar:

    st.markdown("""
    <div class="dm-brand">
        <div class="dm-wordmark">
            <div class="dm-logo">◆</div>
            DocMind
        </div>
        <div class="dm-tagline">RAG · LangGraph · Bedrock</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dm-section">
        <div class="dm-section-label">Documents</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

    if uploaded_files:
        pills_html = '<div class="dm-doc-list">'
        for f in uploaded_files:
            pills_html += f"""
            <div class="dm-doc-pill">
                <div class="dm-doc-icon">PDF</div>
                <span class="dm-doc-name" title="{f.name}">{f.name}</span>
            </div>"""
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)

    if st.button("Build knowledge base"):
        if not uploaded_files:
            st.error("Upload at least one PDF.")
            st.stop()

        pdf_paths = []
        os.makedirs("uploads", exist_ok=True)

        with st.spinner("Processing…"):
            for uploaded_file in uploaded_files:
                save_path = os.path.join("uploads", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_paths.append(save_path)

            vectorstore = create_vectorstore(pdf_paths)
            app_graph = build_graph(vectorstore)
            st.session_state.app_graph = app_graph

        st.success("Knowledge base ready.")

    if st.session_state.app_graph:
        st.markdown("""
        <div class="dm-status dm-ready">
            <span class="dm-status-dot"></span>
            Index active
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="dm-status dm-idle">
            <span class="dm-status-dot"></span>
            No index loaded
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown('<div class="dm-rule"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="dm-section">
            <div class="dm-section-label">Session</div>
        </div>
        """, unsafe_allow_html=True)

        turns = len([m for m in st.session_state.messages if m["role"] == "user"])
        docs_count = len(uploaded_files) if uploaded_files else 0

        st.markdown(f"""
        <div style="padding: 0 16px;">
            <div class="dm-stat-row">
                <span class="dm-stat-label">Exchanges</span>
                <span class="dm-stat-value">{turns}</span>
            </div>
            <div class="dm-stat-row">
                <span class="dm-stat-label">Documents</span>
                <span class="dm-stat-value">{docs_count}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Main
st.markdown("""
<div class="dm-heading">
    <h1>Ask your documents anything</h1>
    <p>Upload PDFs, build the index, then query across all of them in a single conversation.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    if st.session_state.app_graph is None:
        st.markdown("""
        <div class="dm-empty">
            <div class="dm-empty-icon">⬜</div>
            <h2>No index loaded</h2>
            <p>Upload PDFs in the sidebar and click "Build knowledge base" to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="dm-empty">
            <div class="dm-empty-icon">◆</div>
            <h2>Index ready</h2>
            <p>Type a question below — summaries, comparisons, specific extractions, anything.</p>
        </div>
        """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question about your documents…")

if user_question:

    if st.session_state.app_graph is None:
        st.error("Build a knowledge base first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner(""):

            history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                else:
                    history.append(AIMessage(content=msg["content"]))

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
                config={"configurable": {"thread_id": "streamlit_user"}}
            )

            answer = result["answer"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})