# Mini RAG + Reranker Sprint (industrial-safety-pdfs)

**What:** Small, reproducible Q&A service over a 20-PDF pack. Baseline = vector search (all-MiniLM-L6-v2). Rerankers = Hybrid (vector + BM25) and Learned (logistic regression).

**How to run (CPU-only)**

1. Create virtualenv and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
