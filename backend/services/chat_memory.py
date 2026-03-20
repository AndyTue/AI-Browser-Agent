"""Chat memory service for maintaining conversation history."""


class ChatMemory:
    """Stores the last N conversation exchanges."""

    def __init__(self, max_exchanges: int = 5):
        """
        Initialize chat memory.

        Args:
            max_exchanges: Maximum number of exchanges to keep.
        """
        self.max_exchanges = max_exchanges
        self.exchanges: list[dict] = []

    def add(self, user_message: str, assistant_message: str) -> None:
        """
        Add a conversation exchange to memory.

        Args:
            user_message: The user's question.
            assistant_message: The assistant's response.
        """
        self.exchanges.append({
            "user": user_message,
            "assistant": assistant_message,
        })

        # Keep only the last N exchanges
        if len(self.exchanges) > self.max_exchanges:
            self.exchanges = self.exchanges[-self.max_exchanges:]

    def get_history(self) -> str:
        """
        Get formatted conversation history.

        Returns:
            Formatted string of conversation history.
        """
        if not self.exchanges:
            return "No previous conversation."

        lines = []
        for exchange in self.exchanges:
            lines.append(f"User: {exchange['user']}")
            lines.append(f"Assistant: {exchange['assistant']}")

        return "\n".join(lines)

    def get_history_summary(self, max_chars: int = 1200) -> str:
        """
        Get a compact conversation history suitable for LLM context.

        Each exchange is truncated individually, and the total output
        is capped at *max_chars* to limit token usage.

        Returns:
            Compact formatted string, or 'No previous conversation.'
        """
        if not self.exchanges:
            return "No previous conversation."

        lines = []
        for ex in self.exchanges:
            q = ex["user"][:120]
            a = ex["assistant"][:200]
            lines.append(f"Q: {q}{'...' if len(ex['user']) > 120 else ''} "
                         f"A: {a}{'...' if len(ex['assistant']) > 200 else ''}")

        summary = "\n".join(lines)
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit("\n", 1)[0]
        return summary

    def clear(self) -> None:
        """Clear all conversation history."""
        self.exchanges = []
