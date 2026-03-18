"""Cache service to store visited pages during a session."""

class PageCache:
    """Stores the extracted text of visited URLs to avoid re-scraping."""

    def __init__(self):
        self._cache: dict[str, str] = {}

    def get(self, url: str) -> str | None:
        """Retrieve text for a URL if it exists in cache."""
        return self._cache.get(url)

    def set(self, url: str, text: str) -> None:
        """Store extracted text for a URL."""
        self._cache[url] = text

    def clear(self) -> None:
        """Clear the cache (used when processing a new root URL)."""
        self._cache.clear()