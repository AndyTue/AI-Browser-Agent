import gradio as gr
from api_client import process_url, chat

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

def build_ui() -> gr.Blocks:
    """Build and return the gradio interface"""
    with gr.Blocks(title="AI Browser Agent") as demo:
        
        # Header
        gr.HTML(
            """
            <div class="main-header">
                <h1>🌐 AI Browser Agent</h1>
                <p>Process any web page and chat with its content using AI</p>
            </div>
            """
        )

        # Section 1: URL Processing
        with gr.Group():
            gr.Markdown("### Process a Web")
            with gr.Row():
                url_input = gr.Textbox(
                    label="URL",
                    placeholder="https://example.com/article",
                    scale=4,
                    elem_id="url-input",
                )
                process_btn = gr.Button(
                    "Process URL",
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
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(
                height=450,
                placeholder="Process a URL first, then ask questions here...",
                elem_id="chatbot",
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Question",
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

    return demo