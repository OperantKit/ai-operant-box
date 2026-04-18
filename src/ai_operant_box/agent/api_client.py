"""LLM API client wrappers (lazy ``anthropic`` import).

The Anthropic SDK is an *optional* dependency. Importing this module is safe
without it; only constructing :class:`AnthropicClient` actually triggers the
import, which raises ``RuntimeError`` if the package is missing.

API keys are read from the ``ANTHROPIC_API_KEY`` environment variable. They
are never accepted as positional constructor arguments to discourage
hardcoding.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, Protocol


class LLMClientProtocol(Protocol):
    """Minimal generate() interface used by :class:`LLMSubject`."""

    def generate(self, model: str, system: str, messages: Iterable[dict]) -> str:
        ...


class AnthropicClient:
    """Thin wrapper around the Anthropic Python SDK (lazy-imported).

    Parameters
    ----------
    default_model : str
        Default model identifier; can be overridden per :meth:`generate` call.
    max_tokens : int
        Maximum tokens to request per completion.
    api_key_env : str
        Name of the environment variable holding the API key. Default
        ``ANTHROPIC_API_KEY``.

    Notes
    -----
    The actual ``anthropic`` import happens inside ``__init__`` so that simply
    importing this module (and the package) does not require the SDK.
    """

    def __init__(
        self,
        default_model: str = "claude-3-5-sonnet-latest",
        max_tokens: int = 64,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise RuntimeError(
                "The 'anthropic' package is required to use AnthropicClient. "
                "Install it via `pip install ai-operant-box[llm]` or "
                "`pip install anthropic>=0.39`."
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env}' is not set. "
                "Anthropic API keys must never be hardcoded."
            )

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.default_model = default_model
        self.max_tokens = max_tokens

    def generate(
        self,
        model: str | None,
        system: str,
        messages: Iterable[dict],
    ) -> str:
        """Call Claude and return the first text block of the response."""
        msgs = list(messages)
        response: Any = self._client.messages.create(
            model=model or self.default_model,
            max_tokens=self.max_tokens,
            system=system,
            messages=msgs,
        )
        for block in getattr(response, "content", []):
            text = getattr(block, "text", None)
            if text:
                return str(text).strip()
        return ""


__all__ = ["AnthropicClient", "LLMClientProtocol"]
