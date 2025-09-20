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
```
# Screenshots
<img width="1735" height="947" alt="Screenshot 2025-09-20 220639" src="https://github.com/user-attachments/assets/7d004545-abc4-4113-9786-6373f20174df" />
<img width="959" height="818" alt="Screenshot 2025-09-20 220412" src="https://github.com/user-attachments/assets/0dedfc70-f0df-4ad0-98ef-a0379cffb6cb" />
