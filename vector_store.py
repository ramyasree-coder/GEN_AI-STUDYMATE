import os
import numpy as np
import faiss
import pickle
from typing import List, Tuple


class FaissVectorStore:
    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self._metadatas = []

    def add_documents(self, texts: List[str], embeddings: List[List[float]]):
        arr = np.array(embeddings, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch")
        self.index.add(arr)
        self._metadatas.extend(texts)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        q = np.array([query_embedding], dtype="float32")
        D, I = self.index.search(q, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._metadatas):
                continue
            results.append((self._metadatas[idx], float(dist)))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(self._metadatas, f)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            metadatas = pickle.load(f)
        index = faiss.read_index(os.path.join(path, "index.faiss"))
        inst = cls(embedding_dim=index.d)
        inst.index = index
        inst._metadatas = metadatas
        return inst
