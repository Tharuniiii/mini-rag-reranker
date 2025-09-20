import streamlit as st
from src.baseline_search import Retriever
from src.reranker import HybridReranker

INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.json"
SQLITE_PATH = "data/chunks.sqlite"

retriever = Retriever(INDEX_PATH, META_PATH, SQLITE_PATH)
reranker = HybridReranker()

st.title("Mini RAG Reranker Demo")
query = st.text_input("Enter your query:")

if query:
    candidates = retriever.query(query, k=10)

    reranked = reranker.score(query, candidates)
    st.subheader("Reranked Results")
    for c in reranked:
        st.write(f"**{c.get('doc', 'unknown')}** — {c['chunk'][:200]}...")

