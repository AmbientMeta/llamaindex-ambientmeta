"""Privacy-aware LLM wrapper for LlamaIndex.

Wraps any LlamaIndex LLM to automatically sanitize PII from inputs
and rehydrate responses, ensuring sensitive data never reaches the LLM.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CustomLLM
from pydantic import Field, PrivateAttr

from ambientmeta import AmbientMeta


class PrivacyLLM(CustomLLM):
    """A privacy-aware LLM wrapper that sanitizes PII before LLM calls.

    This wrapper intercepts all inputs, sanitizes PII using the AmbientMeta
    Privacy Gateway, forwards the sanitized text to the underlying LLM,
    and then rehydrates the response with original values.

    Example:
        from llamaindex_ambientmeta import PrivacyLLM
        from llama_index.llms.openai import OpenAI

        llm = PrivacyLLM(
            llm=OpenAI(model="gpt-4"),
            api_key="am_live_xxx",
        )

        # PII is automatically protected
        response = llm.complete("Contact john@acme.com about the deal")
    """

    llm: Any = Field(description="The underlying LlamaIndex LLM to wrap")
    api_key: str = Field(description="AmbientMeta API key")
    base_url: str = Field(
        default="https://api.ambientmeta.com",
        description="AmbientMeta API base URL",
    )
    entities: Optional[List[str]] = Field(
        default=None,
        description="Entity types to detect. None means all types.",
    )
    auto_rehydrate: bool = Field(
        default=True,
        description="Whether to automatically rehydrate responses.",
    )

    _client: Optional[AmbientMeta] = PrivateAttr(default=None)

    @property
    def _client_instance(self) -> AmbientMeta:
        if self._client is None:
            self._client = AmbientMeta(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    @property
    def metadata(self) -> LLMMetadata:
        return self.llm.metadata

    def _sanitize_text(self, text: str) -> tuple[str, str]:
        """Sanitize text and return (sanitized_text, session_id)."""
        result = self._client_instance.sanitize(text, entities=self.entities)
        return result.sanitized, result.session_id

    def _rehydrate_text(self, text: str, session_id: str) -> str:
        """Rehydrate text using the session ID."""
        result = self._client_instance.rehydrate(text, session_id)
        return result.text

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Complete a prompt with automatic PII protection."""
        sanitized, session_id = self._sanitize_text(prompt)

        response = self.llm.complete(sanitized, formatted=formatted, **kwargs)

        if self.auto_rehydrate:
            rehydrated = self._rehydrate_text(response.text, session_id)
            return CompletionResponse(text=rehydrated, raw=response.raw)

        return response

    def chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Chat with automatic PII protection."""
        sanitized_messages = []
        session_ids = []

        for msg in messages:
            sanitized, session_id = self._sanitize_text(msg.content)
            sanitized_messages.append(
                ChatMessage(role=msg.role, content=sanitized)
            )
            session_ids.append(session_id)

        response = self.llm.chat(sanitized_messages, **kwargs)

        if self.auto_rehydrate and session_ids:
            last_session = session_ids[-1]
            rehydrated = self._rehydrate_text(
                response.message.content, last_session
            )
            return ChatResponse(
                message=ChatMessage(
                    role=response.message.role, content=rehydrated
                ),
                raw=response.raw,
            )

        return response

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ):
        """Stream completion — sanitizes input, streams output as-is.

        Note: Rehydration is not applied to streamed tokens since the full
        response is needed for placeholder replacement. Use complete() if
        you need automatic rehydration.
        """
        sanitized, _ = self._sanitize_text(prompt)
        return self.llm.stream_complete(sanitized, formatted=formatted, **kwargs)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ):
        """Stream chat — sanitizes input, streams output as-is."""
        sanitized_messages = []
        for msg in messages:
            sanitized, _ = self._sanitize_text(msg.content)
            sanitized_messages.append(
                ChatMessage(role=msg.role, content=sanitized)
            )
        return self.llm.stream_chat(sanitized_messages, **kwargs)
