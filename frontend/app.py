"""Gradio frontend for the AI Browser Chatbot."""

import gradio as gr
import httpx

API_BASE_URL = "http://localhost:8000"


def process_url(url: str) -> str:
    """Send a URL to the backend for processing."""
    if not url or not url.strip():
        return "⚠️ Please enter a valid URL."

    try:
        response = httpx.post(
            f"{API_BASE_URL}/process",
            json={"url": url.strip()},
            timeout=120.0,
        )

        if response.status_code == 200:
            data = response.json()
            title = data.get("title", "Unknown")
            chunks = data.get("chunks_count", 0)
            return (
                f"✅ **Successfully processed!**\n\n"
                f"📄 **Title:** {title}\n"
                f"🧩 **Chunks created:** {chunks}\n"
                f"🔗 **URL:** {url}\n\n"
                f"You can now ask questions about this page."
            )
        else:
            detail = response.json().get("detail", "Unknown error")
            return f"❌ **Error:** {detail}"

    except httpx.ConnectError:
        return "❌ **Connection failed.** Make sure the backend is running on port 8000."
    except httpx.TimeoutException:
        return "❌ **Request timed out.** The page might be too large or slow to load."
    except Exception as e:
        return f"❌ **Unexpected error:** {str(e)}"


def chat(message: str, history: list) -> tuple:
    """Send a message to the backend and get a response."""
    if not message or not message.strip():
        return history, ""

    try:
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json={"question": message.strip()},
            timeout=60.0,
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No response received.")
            source = data.get("source_url", "")
            if source:
                answer += f"\n\n📎 *Source: {source}*"
        else:
            detail = response.json().get("detail", "Unknown error")
            answer = f"❌ Error: {detail}"

    except httpx.ConnectError:
        answer = "❌ Connection failed. Make sure the backend is running."
    except httpx.TimeoutException:
        answer = "❌ Request timed out."
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""


# --- Build the Gradio UI ---
custom_css = """
.main-header {
    text-align: center;
    padding: 20px 0;
}
.main-header h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    margin-bottom: 5px;
}
.main-header p {
    color: #6b7280;
    font-size: 1.1em;
}
"""

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
)

with gr.Blocks(title="AI Browser Chatbot") as demo:

    # Header
    gr.HTML(
        """
        <div class="main-header">
            <h1>🌐 AI Browser Chatbot</h1>
            <p>Process any web page and chat with its content using AI</p>
        </div>
        """
    )

    # Section 1: URL Processing
    with gr.Group():
        gr.Markdown("### 📥 Process a Web Page")
        with gr.Row():
            url_input = gr.Textbox(
                label="URL",
                placeholder="https://example.com/article",
                scale=4,
                elem_id="url-input",
            )
            process_btn = gr.Button(
                "🚀 Process URL",
                variant="primary",
                scale=1,
                elem_id="process-btn",
            )
        status_output = gr.Markdown(
            value="*Enter a URL above and click Process to get started.*",
            label="Status",
            elem_id="status-output",
        )

    gr.Markdown("---")

    # Section 2: Chat
    with gr.Group():
        gr.Markdown("### 💬 Chat with the Page")
        chatbot = gr.Chatbot(
            height=450,
            placeholder="Process a URL first, then ask questions here...",
            elem_id="chatbot",
        )
        with gr.Row():
            msg_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about the processed page...",
                scale=4,
                elem_id="msg-input",
            )
            send_btn = gr.Button(
                "Send",
                variant="primary",
                scale=1,
                elem_id="send-btn",
            )

    # Wire up events
    process_btn.click(
        fn=process_url,
        inputs=[url_input],
        outputs=[status_output],
    )

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
    )

    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False, theme=custom_theme, css=custom_css)
