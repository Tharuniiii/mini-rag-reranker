# reranker.py
import math
import sqlite3
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression
import pickle

# --- Utility ---
def normalize_scores(arr):
    arr = np.array(arr)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

# --- Hybrid Reranker ---
class HybridReranker:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.bm25 = None
        self.docs_tokenized = None

    def fit_bm25(self, chunks):
        tokenized = [c.split() for c in chunks]
        self.docs_tokenized = tokenized
        self.bm25 = BM25Okapi(tokenized)

    def score(self, query, candidates):
        vec_scores = np.array([c['score'] for c in candidates])
        bm25_cand = []
        for c in candidates:
            overlap = len(set(query.split()) & set(c['chunk'].split()))
            bm25_cand.append(overlap)
        bm25_cand = np.array(bm25_cand, dtype=float)

        nv = normalize_scores(vec_scores)
        nk = normalize_scores(bm25_cand)
        final = self.alpha * nv + (1 - self.alpha) * nk

        for c, s in zip(candidates, final):
            c['final_score'] = float(s)
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)

# --- Learned Reranker ---
class LearnedReranker:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', random_state=42)

    def featurize(self, query, candidate):
        # Features: query length, chunk length, token overlap, title hit
        q_tokens = query.split()
        c_tokens = candidate['chunk'].split()
        qlen = len(q_tokens)
        clen = len(c_tokens)
        overlap = len(set(q_tokens) & set(c_tokens))
        title_hit = 1.0 if any(tok in candidate['title'] for tok in q_tokens) else 0.0
        return [candidate['score'], qlen, clen, overlap, title_hit]

    def prepare_features(self, query, candidates):
        X = [self.featurize(query, c) for c in candidates]
        return np.array(X)

    def fit(self, X, y):
        self.model.fit(X, y)

    def score(self, query, candidates):
        X = self.prepare_features(query, candidates)
        probs = self.model.predict_proba(X)[:, 1]
        for c, s in zip(candidates, probs):
            c['final_score'] = float(s)
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
