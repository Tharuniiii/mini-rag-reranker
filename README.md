# 📌 Mini RAG + Reranker Sprint (industrial-safety-pdfs)

# 🚀 How to Run (CPU-only)
1️⃣ Setup
```
git clone https://github.com/Tharuniiii/mini-rag-reranker.git
cd mini-rag-reranker
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```
2️⃣ Prepare Data

Place your 20-PDFs in data/pdfs. Run:
```python src/index_pdfs.py```

3️⃣ Test Baseline + Reranker

Interactive test:
```python src/test_reranker.py```

Then enter queries like:
diagnostic coverage DC value determination  
requirements for category 2 safety systems  

4️⃣ Evaluate

Run evaluation on your source list:
```python src/evaluate.py data/sources.json --metrics```

5️⃣ Frontend (Optional)

To launch a simple UI using Streamlit:
```streamlit run app.py```

# 📂 Project Structure
```
mini-rag-reranker/
│
├── data/
│   ├── pdfs/             # Your input PDFs
│   └── sources.json      # Source metadata
├── src/
│   ├── index_pdfs.py     # Chunk + embed PDFs
│   ├── retriever.py      # Retriever implementation
│   ├── reranker.py       # Hybrid + learned reranker
│   ├── test_reranker.py  # CLI testing
│   └── evaluate.py       # Evaluation script
├── app.py                # Streamlit frontend
├── requirements.txt
└── README.md


