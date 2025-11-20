import os
import pickle
import numpy as np

from ingestion.section_extractor import extract_section_titles
from embeddings.openai_embeddings import get_embedding_model

try:
    import faiss
    HAS_FAISS = True
except:
    faiss = None
    HAS_FAISS = False


def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class IndexBuilder:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.embedder = get_embedding_model()

    def build(self, documents):
        page_texts = []
        metas = []

        for d in documents:
            text = d.page_content.strip()
            if not text:
                continue

            d.metadata["sections"] = extract_section_titles(text)
            page_texts.append(text)
            metas.append(d.metadata)

        # embeddings
        vectors = self.embedder.embed_documents(page_texts)
        vectors = normalize(np.array(vectors, dtype=np.float32))

        if HAS_FAISS:
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            faiss.write_index(index, os.path.join(self.index_dir, "index.faiss"))
        else:
            index = vectors
            np.save(os.path.join(self.index_dir, "vectors.npy"), vectors)

        pickle.dump(metas, open(os.path.join(self.index_dir, "meta.pkl"), "wb"))

        mode = "faiss" if HAS_FAISS else "numpy"
        return {
            "mode": mode,
            "index": index,
            "metadatas": metas,
        }
