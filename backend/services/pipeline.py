"""Pipeline service that orchestrates the full URL processing flow."""

import asyncio

from backend.crawler.playwright_crawler import crawl_url
from backend.parser.html_parser import parse_html, chunk_text
from backend.embedding.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSStore


class Pipeline:
    """Orchestrates crawl → parse → chunk → embed → store."""

    def __init__(self, embedder: Embedder, store: FAISSStore):
        """
        Initialize the pipeline.

        Args:
            embedder: Embedder instance for generating vectors.
            store: FAISSStore instance for storing vectors.
        """
        self.embedder = embedder
        self.store = store

    async def process_url(self, url: str) -> dict:
        """
        Run the full pipeline: crawl, parse, chunk, embed, and store.

        Args:
            url: The URL to process.

        Returns:
            dict with keys: status, num_chunks, title
        """
        # Step 1: Crawl the URL
        html = await crawl_url(url)

        # Step 2: Parse HTML to clean text
        parsed = parse_html(html, url)

        # Step 3: Chunk the text
        chunks = chunk_text(parsed["text"], url)

        if not chunks:
            raise RuntimeError(f"No chunks generated from {url}")

        # Step 4: Generate embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        vectors = self.embedder.embed(chunk_texts)

        # Step 5: Build metadata
        metadata_list = [
            {"url": chunk["url"], "chunk_id": chunk["chunk_id"]}
            for chunk in chunks
        ]

        # Step 6: Clear old data and store new
        self.store.clear()
        self.store.add(vectors, chunk_texts, metadata_list)

        return {
            "status": "success",
            "num_chunks": len(chunks),
            "title": parsed["title"],
        }
