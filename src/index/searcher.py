import numpy as np
import os
import pickle
from embeddings.openai_embeddings import get_embedding_model


try:
    import faiss
    HAS_FAISS = True
except:
    faiss = None
    HAS_FAISS = False


class Searcher:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.embedder = get_embedding_model()

        self.meta = pickle.load(open(os.path.join(index_dir, "meta.pkl"), "rb"))

        if HAS_FAISS:
            self.mode = "faiss"
            self.idx = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        else:
            self.mode = "numpy"
            self.idx = np.load(os.path.join(index_dir, "vectors.npy"))

    def search(self, query: str, k: int = 5):
        q = self.embedder.embed_query(query)
        q = np.array(q, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)

        if self.mode == "faiss":
            D, I = self.idx.search(q.reshape(1, -1), k)
            hits = [(float(D[0][i]), self.meta[I[0][i]]) for i in range(k)]
            return hits

        sims = self.idx @ q
        top = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in top]
