"""Pipeline service that orchestrates the full URL processing flow."""

import asyncio

from backend.crawler.playwright_crawler import crawl_url
from backend.parser.html_parser import parse_html, chunk_text, extract_internal_links
from backend.embedding.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSStore

from backend.llm.groq_client import GroqClient

class Pipeline:
    """Orchestrates crawl → parse → chunk → embed → store."""

    def __init__(self, embedder: Embedder, store: FAISSStore, llm: GroqClient = None):
        """
        Initialize the pipeline.

        Args:
            embedder: Embedder instance for generating vectors.
            store: FAISSStore instance for storing vectors.
            llm: Optional LLM client, used to generate summaries.
        """
        self.embedder = embedder
        self.store = store
        self.llm = llm

    async def process_url(self, start_url: str) -> dict:
        """
        Run the full pipeline for a single root URL.
        """
        try:
            # Scrape HTML
            html = await crawl_url(start_url)
            
            # Extraer enlaces internos
            internal_links = extract_internal_links(html, start_url)

            # Parsear texto limpio y crear chunks
            parsed = parse_html(html, start_url)
            chunks = chunk_text(parsed["text"], start_url)
            
            # Resume Page
            summary = "Summary not available"
            if self.llm and parsed.get("text"):
                summary = self.llm.summarize(parsed["text"][:8000]) # Avoid token limits

            main_title = parsed.get("title", "Unknown")

        except Exception as e:
            raise RuntimeError(f"Failed to process {start_url} - {str(e)}")

        if not chunks:
            raise RuntimeError(f"No chunks generated from {start_url}.")

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
            "title": main_title,
            "summary": summary,
            "internal_links": internal_links
        }
        
    async def process_incremental_url(self, url: str) -> str:
        """
        Process a sub-URL and APPEND its vectors to the existing store.
        Returns the parsed text so the LLM can use it immediately.
        """
        try:
            # 1. Scrape HTML
            html = await crawl_url(url)
            
            # 2. Parse clean text
            parsed = parse_html(html, url)
            text = parsed["text"]
            
            # 3. Create chunks
            chunks = chunk_text(text, url)
            
            if chunks:
                # 4. Generate embeddings
                chunk_texts = [chunk["text"] for chunk in chunks]
                vectors = self.embedder.embed(chunk_texts)

                # 5. Build metadata
                metadata_list = [
                    {"url": chunk["url"], "chunk_id": chunk["chunk_id"]}
                    for chunk in chunks
                ]

                # 6. Store in FAISS (Notice we DO NOT call self.store.clear() here)
                self.store.add(vectors, chunk_texts, metadata_list)
            
            return text

        except Exception as e:
            raise RuntimeError(f"Incremental processing failed for {url}: {str(e)}")