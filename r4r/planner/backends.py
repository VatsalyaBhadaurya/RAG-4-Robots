"""LLM backends for the planner.

- OllamaChatBackend: local model via ollama.chat with structured decoding
  (format = ActionPlan JSON schema). Default model llama3.2:latest (D-04).
- MockBackend: returns scripted responses; lets us test the parse/validate/
  repair/fallback logic deterministically with no server.
"""

from __future__ import annotations

from typing import Callable, List, Optional


class LLMBackend:
    def chat(self, messages: List[dict], schema: Optional[dict] = None) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


class MockBackend(LLMBackend):
    """Returns `responses` in order (repeating the last once exhausted), or
    delegates to `handler(messages, call_index)` if provided. Use it to simulate
    good output, malformed-then-fixed output, or id fabrication."""

    def __init__(self, responses: Optional[List[str]] = None,
                 handler: Optional[Callable[[List[dict], int], str]] = None,
                 model: str = "mock") -> None:
        if responses is None and handler is None:
            raise ValueError("provide responses or handler")
        self._responses = responses or []
        self._handler = handler
        self._model = model
        self.calls = 0

    def chat(self, messages: List[dict], schema: Optional[dict] = None) -> str:
        i = self.calls
        self.calls += 1
        if self._handler is not None:
            return self._handler(messages, i)
        if not self._responses:
            return ""
        return self._responses[min(i, len(self._responses) - 1)]

    @property
    def name(self) -> str:
        return f"MockBackend({self._model})"


class OllamaChatBackend(LLMBackend):
    def __init__(self, model: str = "llama3.2:latest",
                 host: Optional[str] = None,
                 temperature: float = 0.0) -> None:
        import ollama  # lazy

        self._client = ollama.Client(host=host) if host else ollama
        self._model = model
        self._temperature = temperature

    def chat(self, messages: List[dict], schema: Optional[dict] = None) -> str:
        kwargs = {"model": self._model, "messages": messages,
                  "options": {"temperature": self._temperature}}
        # structured outputs: pass the JSON schema if available, else json mode
        kwargs["format"] = schema if schema else "json"
        resp = self._client.chat(**kwargs)
        return resp["message"]["content"]

    @property
    def name(self) -> str:
        return f"OllamaChatBackend({self._model})"
