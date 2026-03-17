import httpx

API_BASE_URL = "http://localhost:8000"


def process_url(url: str) -> str:
    """Send a URL to the backend for processing."""
    if not url or not url.strip():
        return "Please enter a valid URL."

    try:
        response = httpx.post(
            f"{API_BASE_URL}/process",
            json={"url": url.strip()},
            timeout=600.0,
        )

        if response.status_code == 200:
            data = response.json()
            title = data.get("title", "Unknown")
            chunks = data.get("chunks_count", 0)
            return (
                f"**Successfully processed!**\n\n"
                f"**Title:** {title}\n"
                f"**Pages crawled:** {pages}\n"
                f"**Chunks created:** {chunks}\n"
                f"**URL:** {url}\n\n"
                f"You can now ask questions about this page."
            )
        else:
            detail = response.json().get("detail", "Unknown error")
            return f"**Error:** {detail}"

    except httpx.ConnectError:
        return "**Connection failed.** Make sure the backend is running on port 8000."
    except httpx.TimeoutException:
        return "**Request timed out.** The page might be too large or slow to load."
    except Exception as e:
        return f"**Unexpected error:** {str(e)}"


def chat(message: str, history: list) -> tuple:
    """Send a message to the backend and get a response."""
    if not message or not message.strip():
        return history, ""

    try:
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json={"question": message.strip()},
            timeout=800.0,
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No response received.")
            source = data.get("source_url", "")
            if source:
                answer += f"\n\n📎 *Source: {source}*"
        else:
            detail = response.json().get("detail", "Unknown error")
            answer = f"Error: {detail}"

    except httpx.ConnectError:
        answer = "Connection failed. Make sure the backend is running."
    except httpx.TimeoutException:
        answer = "Request timed out."
    except Exception as e:
        answer = f"Error: {str(e)}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""