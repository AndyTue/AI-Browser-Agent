"""Groq LLM client using llama3-8b-8192."""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqClient:
    """LLM client using Groq API with llama3-8b."""

    def __init__(self):
        """Initialize the Groq client with API key from environment."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise RuntimeError(
                "GROQ_API_KEY not set. Create a .env file with your Groq API key."
            )
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def generate(self, messages: list[dict], tools: list[dict] = None):
        """
        Generate a response using the provided messages and tools (for Tool Calling).

        Args:
            messages: List of conversation messages.
            tools: Optional list of tools to use.

        Returns:
            The LLM's response message object.
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1024,
            }
            if tools:
                kwargs["tools"] = tools
                
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}")

    def get_scrape_tool_schema(self) -> dict:
        """Return the schema for the scrape_url tool."""
        return {
            "type": "function",
            "function": {
                "name": "scrape_url",
                "description": "Navigates to an internal URL of the website to read its content and find answers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full internal URL to scrape."
                        }
                    },
                    "required": ["url"]
                }
            }
        }

    def summarize(self, text: str) -> str:
        """
        Generate a concise summary of the provided text.

        Args:
            text: The text to summarize (e.g., scraped from a web page).

        Returns:
            A summary string of the web.
        """
        system_prompt = (
            "You are a helpful AI assistant. Your task is to provide a very brief, "
            "concise summary (4 to 6 sentences maximum) of the provided text."
        )

        user_prompt = f"Please summarize the following text:\n\n{text}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: Summarization failed: {str(e)}")
            return "Summary unavailable."
