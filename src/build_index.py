# build_index.py
import argparse
import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MODEL_NAME = 'all-MiniLM-L6-v2'


def load_chunks(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute('SELECT id, doc, title, chunk FROM chunks')
    rows = cur.fetchall()
    conn.close()
    ids, docs, titles, chunks = zip(*rows)
    meta = [{'id': int(i), 'doc':d,'title':t} for i,d,t in zip(ids,docs,titles)]
    return list(ids), list(chunks), meta


def main(sqlite_path, index_path, meta_path):
    ids, chunks, meta = load_chunks(sqlite_path)
    model = SentenceTransformer(MODEL_NAME)
    print('Embedding', len(chunks), 'chunks...')
    embs = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embs)
    index.add(embs)
    faiss.write_index(index, index_path)
    # save embeddings ids->index mapping and meta
    with open(meta_path, 'w') as f:
        json.dump({'ids': ids, 'meta': meta}, f)
    print('Saved index to', index_path)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sqlite-path', required=True)
    p.add_argument('--index-path', required=True)
    p.add_argument('--meta-path', required=True)
    args = p.parse_args()
    main(args.sqlite_path, args.index_path, args.meta_path)