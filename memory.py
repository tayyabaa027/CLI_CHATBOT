"""
Sliding-window conversation memory with token-aware trimming.
Uses tiktoken's cl100k_base encoding as a token-count approximation —
not perfectly exact for Gemini, but close enough to budget against.
"""
import json
import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


class ConversationManager:
    def __init__(self, system_prompt: str | None = None, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _count(self, text: str) -> int:
        return len(_ENCODER.encode(text))

    def total_tokens(self) -> int:
        return sum(self._count(m["content"]) for m in self.messages)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self) -> None:
        """
        Drop the oldest non-system message first, repeatedly, until back
        under budget. If a single message is larger than the whole budget
        this won't fully save you — truncate giant inputs before they arrive.
        """
        while self.total_tokens() > self.max_tokens and len(self.messages) > 1:
            drop_index = 1 if self.messages[0]["role"] == "system" else 0
            if len(self.messages) <= drop_index + 1:
                break
            del self.messages[drop_index]

    def clear(self, system_prompt: str | None = None) -> None:
        """Wipe all messages and optionally restore the system prompt."""
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.messages, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            self.messages = json.load(f)
