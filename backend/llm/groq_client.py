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

    def generate(self, context: str, history: str, question: str) -> str:
        """
        Generate a response using retrieved context, chat history, and the question.

        Args:
            context: Retrieved text chunks with URLs.
            history: Formatted conversation history.
            question: The user's current question.

        Returns:
            The LLM's response string.
        """
        system_prompt = (
            "You are a helpful AI assistant that answers questions based ONLY on the "
            "provided context from a web page. Follow these rules strictly:\n"
            "1. Use ONLY the information from the provided context to answer.\n"
            "2. Always cite the source URL in your answer.\n"
            "3. Do NOT make up or hallucinate any information.\n"
            "4. If the answer is not found in the context, say: "
            "\"I don't have enough information from the provided page to answer that question.\"\n"
            "5. Be concise and accurate."
        )

        user_prompt = (
            f"Context from web page:\n{context}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Question: {question}\n\n"
            "Answer based ONLY on the context above. Cite the source URL."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}")
