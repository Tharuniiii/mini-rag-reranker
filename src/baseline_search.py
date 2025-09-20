# baseline_search.py
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import sqlite3

MODEL_NAME='all-MiniLM-L6-v2'

class Retriever:
    def __init__(self, index_path, meta_path, sqlite_path):
        self.index = faiss.read_index(index_path)
        with open(meta_path,'r') as f:
            obj = json.load(f)
        self.ids = obj['ids']
        self.meta = {int(m['id']):m for m in obj['meta']}
        self.model = SentenceTransformer(MODEL_NAME)
        self.conn = sqlite3.connect(sqlite_path)

    def query(self, q, k=5):
        q_emb = self.model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            chunk_id = int(self.ids[int(idx)])
            cur = self.conn.cursor()
            cur.execute('SELECT chunk, title, doc FROM chunks WHERE id=?', (chunk_id,))
            row = cur.fetchone()
            results.append({'id':chunk_id, 'score':float(score), 'chunk':row[0], 'title':row[1], 'doc':row[2]})
        return results

if __name__ == '__main__':
    import sys
    r = Retriever('data/faiss.index','data/meta.json','data/chunks.sqlite')
    q = ' '.join(sys.argv[1:]) or 'safety standard EN ISO 13849-1'
    res = r.query(q, k=5)
    for r in res:
        print(r['score'], r['title'])
        print(r['chunk'][:300].replace('\n',' '))
        print('---')