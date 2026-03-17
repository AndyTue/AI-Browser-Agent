"""Playwright-based web crawler for fetching full-page HTML."""

import asyncio
from urllib.parse import urlparse

from playwright.async_api import async_playwright


async def crawl_url(url: str) -> str:
    """
    Crawl a URL using headless Chromium and return the full page HTML.

    Args:
        url: The URL to crawl.

    Returns:
        The full HTML content of the page.

    Raises:
        ValueError: If the URL is invalid.
        TimeoutError: If the page takes too long to load.
        RuntimeError: If crawling fails for any other reason.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}. Must include scheme (http/https) and domain.")

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()

            await page.goto(url, wait_until="networkidle", timeout=50000)

            # Wait a bit for any remaining JS rendering
            await page.wait_for_timeout(2000)

            html = await page.content()
            await browser.close()

            if not html or len(html.strip()) == 0:
                raise RuntimeError(f"Empty content received from {url}")

            return html

    except TimeoutError:
        raise TimeoutError(f"Timed out while loading {url} (30s limit)")
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to crawl {url}: {str(e)}")
