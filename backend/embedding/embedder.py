"""Sentence-transformer embedder using all-MiniLM-L6-v2."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Generates text embeddings using a local sentence-transformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with the specified model.

        Args:
            model_name: HuggingFace model name. Defaults to all-MiniLM-L6-v2.
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {str(e)}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        try:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {str(e)}")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query: The query text.

        Returns:
            numpy array of shape (1, dimension).
        """
        return self.embed([query])
