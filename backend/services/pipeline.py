"""Pipeline service that orchestrates the full URL processing flow."""

import asyncio

from backend.crawler.playwright_crawler import crawl_url
from backend.parser.html_parser import parse_html, chunk_text, extract_internal_links
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

    async def process_url(self, start_url: str, max_pages: int = 15) -> dict:
        """
        Run the full pipeline with a recursive crawler up to max_pages.
        """
        visited_urls = set()
        urls_to_visit = [start_url]
        all_chunks = []
        main_title = "Unknown"

        # Step 1 & 2 & 3: Crawl, Parse and Chunk loop
        while urls_to_visit and len(visited_urls) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            try:
                # Scrapear el HTML
                html = await crawl_url(current_url)
                visited_urls.add(current_url)

                # Extraer enlaces internos para la cola de visitas
                internal_links = extract_internal_links(html, start_url)
                for link in internal_links:
                    if link not in visited_urls and link not in urls_to_visit:
                        urls_to_visit.append(link)

                # Parsear texto limpio y crear chunks
                parsed = parse_html(html, current_url)
                chunks = chunk_text(parsed["text"], current_url)
                all_chunks.extend(chunks)

                # Guardar el título de la página principal
                if current_url == start_url:
                    main_title = parsed["title"]

            except Exception as e:
                print(f"Warning: Failed to process {current_url} - {str(e)}")

        if not all_chunks:
            raise RuntimeError(f"No chunks generated from {start_url} or its subpages.")

        # Step 4: Generate embeddings
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        vectors = self.embedder.embed(chunk_texts)

        # Step 5: Build metadata
        metadata_list = [
            {"url": chunk["url"], "chunk_id": chunk["chunk_id"]}
            for chunk in all_chunks
        ]

        # Step 6: Clear old data and store new
        self.store.clear()
        self.store.add(vectors, chunk_texts, metadata_list)

        return {
            "status": "success",
            "num_chunks": len(all_chunks),
            "pages_crawled": len(visited_urls),
            "title": main_title,
        }
