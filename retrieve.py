import os
import json
import numpy as np

class Retriever:
    """
    Retrieves relevant documents from the indexed knowledge base using cosine similarity.
    Supports multiple collections of data.
    """
    def __init__(self, index_path):
        # Load the precomputed index (embeddings and documents)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}. Please run ingest.py first.")
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        self.documents = index_data.get("documents", [])
        self.sources = index_data.get("sources", [])
        embeddings_list = index_data.get("embeddings", [])
        # Convert embeddings list to a numpy array for efficient similarity calculations
        self.embeddings = np.array(embeddings_list, dtype=float)
        # Normalize again to be safe (they should already be normalized)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.embeddings = self.embeddings / norms
        # Precompute transpose for faster dot products (optional)
        self.embeddings_T = self.embeddings.T

    def get_relevant(self, query, top_k=3):
        """
        Given a user query (text), return a list of top_k relevant documents with their sources.
        """
        # Embed the query using the same model as ingestion
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("SentenceTransformer not installed. Please install requirements.txt")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_emb = model.encode(query)
        # Normalize query vector
        if np.linalg.norm(query_emb) == 0:
            sim_scores = np.zeros(len(self.embeddings))
        else:
            query_emb = query_emb / np.linalg.norm(query_emb)
            # Compute cosine similarity as dot product (embeddings are unit-normalized)
            sim_scores = np.dot(self.embeddings, query_emb)
        if len(sim_scores) == 0:
            return []
        # Get indices of the top scores
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "source": self.sources[idx] if idx < len(self.sources) else None,
                "text": self.documents[idx],
                "score": float(sim_scores[idx])
            })
        return results
