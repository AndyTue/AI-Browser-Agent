"""HTML parser and text chunker using BeautifulSoup."""

import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def parse_html(html: str, url: str) -> dict:
    """
    Parse raw HTML and extract clean text content and title.

    Args:
        html: Raw HTML string.
        url: Source URL for metadata.

    Returns:
        dict with keys: title, text, url
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove script and style elements
    for element in soup(["script", "style", "noscript", "iframe", "nav", "footer", "header"]):
        element.decompose()

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Extract visible text
    text = soup.get_text(separator=" ")

    # Normalize whitespace: collapse multiple spaces/newlines into single space
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise RuntimeError(f"No text content extracted from {url}")

    return {
        "title": title,
        "text": text,
        "url": url,
    }


def chunk_text(text: str, url: str, chunk_size: int = 2000, overlap: int = 200) -> list[dict]:
    """
    Split text into overlapping chunks.

    Args:
        text: The full text to chunk.
        url: Source URL for metadata.
        chunk_size: Target size of each chunk in characters.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of dicts with keys: text, url, chunk_id
    """
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size

        # If not at the end, try to break at a sentence boundary
        if end < len(text):
            # Look for the last period, question mark, or exclamation point
            last_break = text.rfind(". ", start, end)
            if last_break == -1:
                last_break = text.rfind("? ", start, end)
            if last_break == -1:
                last_break = text.rfind("! ", start, end)
            if last_break == -1:
                last_break = text.rfind(" ", start, end)

            if last_break > start:
                end = last_break + 1

        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunks.append({
                "text": chunk_text_content,
                "url": url,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

        # Move start forward by (end - overlap), but ensure we make progress
        start = max(start + 1, end - overlap)

    return chunks

def extract_internal_links(html: str, base_url: str) -> list[str]:
    """Extrae enlaces internos del mismo dominio."""
    soup = BeautifulSoup(html, "lxml")
    base_domain = urlparse(base_url).netloc
    links = set()
    
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        
        # Remover fragmentos (#) para evitar duplicados de la misma página
        full_url = full_url.split('#')[0]
        
        # Validar que pertenezca al mismo dominio y use http/https
        parsed_url = urlparse(full_url)
        if parsed_url.scheme in ["http", "https"] and parsed_url.netloc == base_domain:
            links.add(full_url)
            
    return list(links)