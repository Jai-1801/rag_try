# src/indexer.py
import faiss
import numpy as np
import os
import pickle
from typing import List, Dict

INDEX_PATH = 'faiss.index'
META_PATH = 'faiss_meta.pkl'

class FaissIndexer:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # using inner product
        self.metadatas = []

    def add(self, vectors: List[List[float]], metadatas: List[Dict]):
        """Add vectors and metadata to the index."""
        x = np.array(vectors).astype('float32')
        # normalize for cosine similarity
        faiss.normalize_L2(x)
        self.index.add(x)
        self.metadatas.extend(metadatas)

    def save(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, 'wb') as f:
            pickle.dump(self.metadatas, f)

    def load(self):
        """Load index and metadata from disk."""
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, 'rb') as f:
                self.metadatas = pickle.load(f)
            return True
        return False

    def query(self, vector, k=5):
        """Search for top k similar vectors."""
        v = np.array([vector]).astype('float32')
        faiss.normalize_L2(v)
        distances, idxs = self.index.search(v, k)
        results = []
        for score, idx in zip(distances[0], idxs[0]):
            if idx < len(self.metadatas):
                meta = self.metadatas[idx]
                results.append({'score': float(score), 'metadata': meta})
        return results