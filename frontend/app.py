"""Gradio frontend for the AI Browser Agent."""

from ui import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_port=7860, share=False)
