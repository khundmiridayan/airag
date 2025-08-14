import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []

    def build_index(self, embeddings, texts):
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype('float32'))
        self.texts = texts

    def save(self, path='index_store/faiss_index'): 
        faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        with open(os.path.join(path, 'texts.pkl'), 'wb') as f:
            pickle.dump(self.texts, f)

    def load(self, path='index_store/faiss_index'):
        self.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        with open(os.path.join(path, 'texts.pkl'), 'rb') as f:
            self.texts = pickle.load(f)

    def query(self, embedding, top_k=3):
        D, I = self.index.search(np.array([embedding]).astype('float32'), top_k)
        return [self.texts[i] for i in I[0]]
