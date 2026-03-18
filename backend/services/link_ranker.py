"""Ranks internal links by semantic similarity to a query."""

import numpy as np
from backend.embedding.embedder import Embedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one vector and a matrix of vectors."""
    a = a.flatten()
    norm_a = np.linalg.norm(a)
    norms_b = np.linalg.norm(b, axis=1)

    # Avoid division by zero
    safe_norms = np.where(norms_b == 0, 1e-10, norms_b)
    return np.dot(b, a) / (norm_a * safe_norms)


def rank_links(
    question: str,
    links: list[dict],
    embedder: Embedder,
    top_k: int = 5,
) -> list[dict]:
    """
    Return the top_k links most semantically relevant to the question.

    Args:
        question: The user's question.
        links: List of {"url": ..., "text": ...} dicts.
        embedder: The shared Embedder instance.
        top_k: How many links to return.

    Returns:
        Sorted list of the most relevant links (best first).
    """
    if not links:
        return []

    top_k = min(top_k, len(links))

    # Embed question
    q_vec = embedder.embed_query(question)  # shape (1, dim)

    # Embed "anchor text: url" for each link — richer signal than URL alone
    link_texts = [f"{l['text']}: {l['url']}" for l in links]
    link_vecs = embedder.embed(link_texts)   # shape (n, dim)

    scores = cosine_similarity(q_vec, link_vecs)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [links[i] for i in top_indices]