"""
Task 4: Context-Aware Chatbot Using LangChain / RAG
Core retrieval-augmented-generation (RAG) logic.

Author    : Tariq Jamil Khan
Internship: DevelopersHub Corporation - AI/ML Engineering (Phase 2)

Architecture
------------
1. The custom knowledge base (knowledge_base.txt) is split into chunks.
2. Chunks are turned into a vectorized document store using TF-IDF
   (sklearn) + cosine similarity -- a lightweight, fully-offline vector
   search that needs no external embedding-model download.
3. For each user question, the top-k most relevant chunks are retrieved.
4. Those chunks + the recent conversation history (context memory) are
   passed to an LLM (Claude, via the Anthropic API) to generate a
   grounded answer.

Set ANTHROPIC_API_KEY to use real LLM generation. Without it, the module
falls back to a simple extractive DEMO MODE so the RAG pipeline can still
be exercised end-to-end offline.
"""

import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

USE_REAL_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))

if USE_REAL_LLM:
    import anthropic
    _client = anthropic.Anthropic()
    _MODEL_NAME = "claude-sonnet-4-6"


# ---------------------------------------------------------------------
# 1. Vectorized document store
# ---------------------------------------------------------------------
class VectorStore:
    """A minimal TF-IDF based vectorized document store with cosine-similarity
    retrieval -- functionally equivalent to a neural-embedding vector store
    (e.g. FAISS + sentence-transformers) but fully offline/dependency-light."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query, k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]


def load_and_chunk(path="knowledge_base.txt"):
    """Split the corpus into chunks on '== Section ==' headers."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    raw_sections = re.split(r"\n(?=== )", text)
    chunks = [s.strip() for s in raw_sections if s.strip()]
    return chunks


# ---------------------------------------------------------------------
# 2. Conversation memory
# ---------------------------------------------------------------------
class ConversationMemory:
    """Keeps a rolling window of recent (user, assistant) turns so the
    chatbot remains context-aware across a multi-turn conversation."""

    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = []  # list of {"role": "user"/"assistant", "content": str}

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        # keep only the most recent N turns (N user + N assistant messages)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def as_text(self):
        lines = []
        for turn in self.history:
            speaker = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {turn['content']}")
        return "\n".join(lines)

    def clear(self):
        self.history = []


# ---------------------------------------------------------------------
# 3. RAG answer generation
# ---------------------------------------------------------------------
def build_rag_prompt(question, retrieved_chunks, memory: ConversationMemory):
    context = "\n\n---\n\n".join(c for c, _ in retrieved_chunks) or "No relevant context found."
    history_text = memory.as_text() or "(no previous conversation)"

    return f"""You are a helpful assistant answering questions about the DevelopersHub
Corporation AI/ML Engineering Internship program, using ONLY the context provided below.
If the answer isn't in the context, say you don't have that information.

Conversation so far:
{history_text}

Relevant context retrieved from the knowledge base:
{context}

Current question: {question}

Answer concisely and accurately, grounded in the context above."""


def _demo_fallback_answer(question, retrieved_chunks):
    """Extractive stand-in used only when no API key is configured."""
    if not retrieved_chunks:
        return "I don't have information about that in the knowledge base."
    best_chunk, _ = retrieved_chunks[0]
    # Return the most relevant paragraph (trim header line)
    lines = [l for l in best_chunk.split("\n") if l.strip()]
    body = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
    return f"[DEMO MODE - extractive answer] {body[:500]}"


def ask(question, store: VectorStore, memory: ConversationMemory, k=3):
    retrieved = store.retrieve(question, k=k)

    if USE_REAL_LLM:
        prompt = build_rag_prompt(question, retrieved, memory)
        response = _client.messages.create(
            model=_MODEL_NAME,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip()
    else:
        answer = _demo_fallback_answer(question, retrieved)

    memory.add("user", question)
    memory.add("assistant", answer)

    return answer, retrieved


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    chunks = load_and_chunk("knowledge_base.txt")
    store = VectorStore(chunks)
    memory = ConversationMemory()

    print(f"Loaded {len(chunks)} chunks into the vector store.\n")
    if not USE_REAL_LLM:
        print("WARNING: ANTHROPIC_API_KEY not set -> running in DEMO MODE.\n")

    test_questions = [
        "How many tasks do I need to complete for this internship?",
        "What dataset is used for the churn prediction task?",
        "What about the deadline?",  # follow-up, tests context memory
    ]

    for q in test_questions:
        answer, retrieved = ask(q, store, memory)
        print(f"User: {q}")
        print(f"Bot : {answer}")
        print(f"(retrieved {len(retrieved)} chunks, top score={retrieved[0][1]:.3f})" if retrieved else "(no chunks retrieved)")
        print("-" * 60)
