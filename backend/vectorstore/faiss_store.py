"""FAISS-based vector store for similarity search."""

import faiss
import numpy as np


class FAISSStore:
    """Vector store using FAISS IndexFlatL2 for similarity search."""

    def __init__(self, dimension: int):
        """
        Initialize the FAISS store.

        Args:
            dimension: Dimensionality of the embedding vectors.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, texts: list[str], metadata: list[dict]) -> None:
        """
        Add vectors with associated text and metadata.

        Args:
            vectors: numpy array of shape (n, dimension).
            texts: List of text strings corresponding to each vector.
            metadata: List of metadata dicts corresponding to each vector.
        """
        if len(vectors) == 0:
            return

        vectors = vectors.astype(np.float32)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}"
            )

        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        """
        Search for the top-k most similar vectors.

        Args:
            query_vector: numpy array of shape (1, dimension).
            k: Number of results to return.

        Returns:
            List of dicts with keys: text, metadata, score.
        """
        if self.index.ntotal == 0:
            return []

        query_vector = query_vector.astype(np.float32)

        # Limit k to the number of stored vectors
        k = min(k, self.index.ntotal)

        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(distances[0][i]),
            })

        return results

    def clear(self) -> None:
        """Reset the index and all stored data."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []

    @property
    def total_vectors(self) -> int:
        """Return the total number of stored vectors."""
        return self.index.ntotal
