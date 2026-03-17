"""Retriever service for querying the FAISS vector store."""

from backend.embedding.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSStore


class Retriever:
    """Retrieves relevant chunks from the vector store based on a query."""

    def __init__(self, embedder: Embedder, store: FAISSStore):
        """
        Initialize the retriever.

        Args:
            embedder: Embedder instance for encoding queries.
            store: FAISSStore instance to search.
        """
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: The user's question.
            k: Number of chunks to retrieve.

        Returns:
            List of dicts with keys: text, metadata, score.
        """
        if self.store.total_vectors == 0:
            return []

        query_vector = self.embedder.embed_query(query)
        results = self.store.search(query_vector, k=k)
        return results
