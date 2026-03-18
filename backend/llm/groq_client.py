"""Groq LLM client — llama-3.3-70b-versatile with manual tool calling."""

import os
import json
import re
from typing import Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """
    Normalised response object — always has `content`.
    If the model wants to call a tool, `tool_call` is set.
    """
    content: str
    tool_call: Optional[ToolCall] = None

    @property
    def wants_tool(self) -> bool:
        return self.tool_call is not None


class GroqClient:
    """LLM client using Groq API with manual JSON-based tool calling."""

    MODEL = "llama-3.3-70b-versatile"
    MAX_CONTEXT_CHARS = 12_000

    # Regex to pull the JSON action block from model output
    _ACTION_RE = re.compile(
        r'\{[^{}]*"action"\s*:\s*"scrape_url"[^{}]*\}',
        re.DOTALL,
    )

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise RuntimeError(
                "GROQ_API_KEY not set. Create a .env file with your Groq API key."
            )
        self.client = Groq(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, messages: list[dict]) -> LLMResponse:
        """
        Call the LLM and return a normalised LLMResponse.
        No native tool/function calling is used — the model signals tool
        use via a JSON block in its text output, which we parse manually.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                temperature=0,
                max_tokens=1024,
                # Deliberately NO tools/tool_choice — avoids all Groq
                # server-side tool-call parsing failures.
            )
            content = response.choices[0].message.content or ""
            return self._parse_response(content)

        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}")

    def get_tool_instructions(self) -> str:
        """
        Returns the system-prompt snippet that teaches the model how to
        signal a tool call using our manual JSON format.
        """
        return (
            "## Tool Use\n"
            "If you need to visit an internal page to find the answer, "
            "output ONLY the following JSON block and nothing else:\n"
            '{"action": "scrape_url", "url": "<full URL>"}\n\n'
            "Rules:\n"
            "- Output the JSON block alone, with no explanation before or after.\n"
            "- Only call scrape_url on URLs from the Available Internal Links list.\n"
            "- After receiving the page content, answer the user's question directly.\n"
            "- Do NOT output the JSON block if you already have enough context.\n"
        )

    def summarize(self, text: str) -> str:
        text = text[:6_000]
        system_prompt = (
            "You are a helpful AI assistant. Provide a very brief, "
            "concise summary (4 to 6 sentences maximum) of the provided text."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize:\n\n{text}"},
                ],
                temperature=0,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: Summarization failed: {str(e)}")
            return "Summary unavailable."

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, content: str) -> LLMResponse:
        """
        Check whether the model output contains a tool-call JSON block.
        Returns LLMResponse with tool_call populated if found.
        """
        match = self._ACTION_RE.search(content)
        if match:
            try:
                payload = json.loads(match.group())
                action = payload.get("action")
                if action == "scrape_url" and "url" in payload:
                    return LLMResponse(
                        content=content,
                        tool_call=ToolCall(
                            name="scrape_url",
                            arguments={"url": payload["url"]},
                        ),
                    )
            except json.JSONDecodeError:
                pass  # Fall through to plain text response

        return LLMResponse(content=content)