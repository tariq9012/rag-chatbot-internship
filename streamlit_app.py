"""
Task 4: Context-Aware Chatbot Using LangChain / RAG — Streamlit deployment

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from rag_chatbot import VectorStore, ConversationMemory, load_and_chunk, ask, USE_REAL_LLM

st.set_page_config(page_title="DevelopersHub Internship RAG Chatbot", page_icon="🤖")

st.title("🤖 Context-Aware RAG Chatbot")
st.caption("Ask anything about the DevelopersHub AI/ML Internship (Phase 2) tasks.")

if not USE_REAL_LLM:
    st.warning(
        "ANTHROPIC_API_KEY environment variable not set — running in **DEMO MODE** "
        "(extractive answers, no live LLM). Set the key before launching Streamlit "
        "for real generated answers.",
        icon="⚠️",
    )

# ---- Initialize vector store + memory once per session ----
if "store" not in st.session_state:
    chunks = load_and_chunk("knowledge_base.txt")
    st.session_state.store = VectorStore(chunks)
    st.session_state.memory = ConversationMemory(max_turns=5)
    st.session_state.display_history = []

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=5, value=3)
    if st.button("🗑️ Clear conversation"):
        st.session_state.memory.clear()
        st.session_state.display_history = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Knowledge base:** `knowledge_base.txt`\n\nVector store: TF-IDF + cosine similarity")

# ---- Render chat history ----
for turn in st.session_state.display_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Retrieved context"):
                for chunk, score in turn["sources"]:
                    st.markdown(f"**Score: {score:.3f}**\n\n{chunk[:300]}...")

# ---- Chat input ----
if question := st.chat_input("Ask a question about the internship tasks..."):
    st.session_state.display_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            answer, retrieved = ask(question, st.session_state.store, st.session_state.memory, k=k)
        st.markdown(answer)
        if retrieved:
            with st.expander("Retrieved context"):
                for chunk, score in retrieved:
                    st.markdown(f"**Score: {score:.3f}**\n\n{chunk[:300]}...")

    st.session_state.display_history.append(
        {"role": "assistant", "content": answer, "sources": retrieved}
    )
