# Task 4: Context-Aware Chatbot Using LangChain / RAG

**Internship:** DevelopersHub Corporation — AI/ML Engineering Internship Phase 2
**Author:** Tariq Jamil Khan

## Objective
Build a conversational chatbot that **remembers context** across turns and
**retrieves external information** from a custom knowledge base
(Retrieval-Augmented Generation), then deploy it with **Streamlit**.

## Dataset / Knowledge Base
A custom corpus (`knowledge_base.txt`) describing the DevelopersHub AI/ML
Internship Phase 2 program — all 5 task descriptions and submission requirements,
chunked into sections.

## Approach / Methodology
1. **Chunking** — corpus split into sections on `== Heading ==` markers.
2. **Vectorized document store** — chunks embedded with **TF-IDF** (scikit-learn)
   and indexed for **cosine-similarity** retrieval (a lightweight, fully-offline
   vector store; can be swapped for a neural-embedding store like
   sentence-transformers + FAISS without changing the rest of the pipeline).
3. **Retrieval** — top-k most relevant chunks fetched for each user question.
4. **Context memory** — `ConversationMemory` keeps a rolling window of the last
   5 conversation turns so follow-up questions (e.g. *"What about the deadline?"*)
   are resolved correctly.
5. **Generation** — retrieved chunks + conversation history + the question are
   sent to **Claude (Anthropic API)** to produce a grounded answer.
6. **Deployment** — `streamlit_app.py` wraps the same `ask()` function in an
   interactive chat UI with a sidebar (top-k slider, clear-conversation button,
   expandable "retrieved context" per answer).

> If `ANTHROPIC_API_KEY` is not set, the chatbot runs in a DEMO MODE (extractive
> answers from the top retrieved chunk) so the full pipeline stays reproducible offline.

## Key Results / Observations
- Direct factual questions (dataset names, deadlines) retrieve highly relevant
  chunks (cosine similarity scores 0.2–0.4+).
- Context memory correctly carries the topic forward for short follow-up
  questions that have no keywords of their own.
- The retrieval logic is fully decoupled from the UI — the same `rag_chatbot.py`
  module powers both the notebook demo and the Streamlit app.

## Project Structure
```
task4_rag_chatbot/
├── Task4_RAG_Chatbot.ipynb   # Full notebook (retrieval + memory + generation demo)
├── rag_chatbot.py            # Core RAG logic (VectorStore, ConversationMemory, ask())
├── streamlit_app.py          # Streamlit chat UI (deployment)
├── knowledge_base.txt        # Custom corpus
└── README.md
```

## How to Run
```bash
pip install scikit-learn numpy anthropic streamlit
export ANTHROPIC_API_KEY="your-key-here"   # optional — omit to run in demo mode

# CLI smoke test
python rag_chatbot.py

# Notebook
jupyter notebook Task4_RAG_Chatbot.ipynb

# Deployed chat app
streamlit run streamlit_app.py
```

## Skills Gained
- Conversational AI development
- Document chunking and vector search
- Retrieval-Augmented Generation (RAG)
- LLM integration and deployment (Streamlit)
