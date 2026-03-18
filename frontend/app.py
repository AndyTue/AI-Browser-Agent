"""Gradio frontend for the AI Browser Agent."""

from ui import build_ui, custom_theme, custom_css

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, theme=custom_theme, css=custom_css)
