# src/test_reranker.py
import sys
from baseline_search import Retriever
from reranker import HybridReranker

# Paths to your data
INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.json"
SQLITE_PATH = "data/chunks.sqlite"

# Initialize retriever and reranker
retriever = Retriever(INDEX_PATH, META_PATH, SQLITE_PATH)

# Collect all chunks for BM25 fitting
import sqlite3
conn = sqlite3.connect(SQLITE_PATH)
cur = conn.cursor()
cur.execute('SELECT chunk FROM chunks')
all_chunks = [row[0] for row in cur.fetchall()]
conn.close()

reranker = HybridReranker()
reranker.fit_bm25(all_chunks)

print("\nEnter your query (or 'exit'):")
while True:
    query = input("\nquery: ").strip()
    if query.lower() == "exit":
        break

    # --- Use retriever.query (fixed) ---
    candidates = retriever.query(query, k=10)

    print("\n=== Baseline Search Results ===")
    for c in candidates:
        print(f"{c['score']:.4f} {c['doc']}")
        print(c['chunk'][:200].replace("\n", " "))
        print("---")

    reranked = reranker.score(query, candidates)

    print("\n=== Reranked Results ===")
    for c in reranked[:5]:
        print(f"{c['final_score']:.4f} {c['doc']}")
        print(c['chunk'][:200].replace("\n", " "))
        print("---")
