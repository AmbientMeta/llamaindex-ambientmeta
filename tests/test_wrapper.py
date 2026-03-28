"""Tests for llamaindex-ambientmeta wrapper."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)


class FakeLLM:
    """Minimal fake LLM for testing."""

    def __init__(self, response: str = "fake response"):
        self._response = response

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="fake")

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self._response)

    def chat(self, messages, **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=self._response)
        )


class TestPrivacyLLM:
    """Tests for the PrivacyLLM wrapper."""

    @patch("llamaindex_ambientmeta.wrapper.AmbientMeta")
    def test_complete_sanitizes_and_rehydrates(self, mock_client_class):
        """Test that complete() sanitizes input and rehydrates output."""
        from llamaindex_ambientmeta import PrivacyLLM

        mock_client = MagicMock()
        mock_client.sanitize.return_value = MagicMock(
            sanitized="Email [EMAIL_1] about the meeting",
            session_id="ses_123",
        )
        mock_client.rehydrate.return_value = MagicMock(
            text="I'll contact john@example.com tomorrow",
        )
        mock_client_class.return_value = mock_client

        fake_llm = FakeLLM(response="I'll contact [EMAIL_1] tomorrow")

        wrapper = PrivacyLLM(
            llm=fake_llm,
            api_key="am_test_xxx",
        )

        result = wrapper.complete("Email john@example.com about the meeting")

        mock_client.sanitize.assert_called_once_with(
            "Email john@example.com about the meeting",
            entities=None,
        )
        mock_client.rehydrate.assert_called_once()
        assert "john@example.com" in result.text

    @patch("llamaindex_ambientmeta.wrapper.AmbientMeta")
    def test_complete_no_rehydrate(self, mock_client_class):
        """Test that rehydration can be disabled."""
        from llamaindex_ambientmeta import PrivacyLLM

        mock_client = MagicMock()
        mock_client.sanitize.return_value = MagicMock(
            sanitized="Email [EMAIL_1]",
            session_id="ses_123",
        )
        mock_client_class.return_value = mock_client

        fake_llm = FakeLLM(response="Response with [EMAIL_1]")

        wrapper = PrivacyLLM(
            llm=fake_llm,
            api_key="am_test_xxx",
            auto_rehydrate=False,
        )

        result = wrapper.complete("Email john@example.com")

        mock_client.rehydrate.assert_not_called()
        assert "[EMAIL_1]" in result.text

    @patch("llamaindex_ambientmeta.wrapper.AmbientMeta")
    def test_chat_sanitizes_messages(self, mock_client_class):
        """Test that chat() sanitizes all messages."""
        from llamaindex_ambientmeta import PrivacyLLM

        mock_client = MagicMock()
        mock_client.sanitize.return_value = MagicMock(
            sanitized="Email [EMAIL_1]",
            session_id="ses_123",
        )
        mock_client.rehydrate.return_value = MagicMock(
            text="I'll contact john@example.com",
        )
        mock_client_class.return_value = mock_client

        fake_llm = FakeLLM(response="I'll contact [EMAIL_1]")

        wrapper = PrivacyLLM(
            llm=fake_llm,
            api_key="am_test_xxx",
        )

        messages = [
            ChatMessage(role=MessageRole.USER, content="Email john@example.com"),
        ]
        result = wrapper.chat(messages)

        mock_client.sanitize.assert_called_once()
        mock_client.rehydrate.assert_called_once()
        assert "john@example.com" in result.message.content

    @patch("llamaindex_ambientmeta.wrapper.AmbientMeta")
    def test_custom_entities(self, mock_client_class):
        """Test that custom entity types are forwarded."""
        from llamaindex_ambientmeta import PrivacyLLM

        mock_client = MagicMock()
        mock_client.sanitize.return_value = MagicMock(
            sanitized="Email [EMAIL_1]",
            session_id="ses_123",
        )
        mock_client.rehydrate.return_value = MagicMock(text="Email john@example.com")
        mock_client_class.return_value = mock_client

        fake_llm = FakeLLM(response="Email [EMAIL_1]")

        wrapper = PrivacyLLM(
            llm=fake_llm,
            api_key="am_test_xxx",
            entities=["EMAIL", "PHONE"],
        )

        wrapper.complete("Email john@example.com")

        mock_client.sanitize.assert_called_once_with(
            "Email john@example.com",
            entities=["EMAIL", "PHONE"],
        )


    @patch("llamaindex_ambientmeta.wrapper.AmbientMeta")
    def test_chat_no_rehydrate(self, mock_client_class):
        """Test chat() with auto_rehydrate=False."""
        from llamaindex_ambientmeta import PrivacyLLM

        mock_client = MagicMock()
        mock_client.sanitize.return_value = MagicMock(
            sanitized="Email [EMAIL_1]",
            session_id="ses_123",
        )
        mock_client_class.return_value = mock_client

        fake_llm = FakeLLM(response="I'll contact [EMAIL_1]")

        wrapper = PrivacyLLM(
            llm=fake_llm,
            api_key="am_test_xxx",
            auto_rehydrate=False,
        )

        messages = [
            ChatMessage(role=MessageRole.USER, content="Email john@example.com"),
        ]
        result = wrapper.chat(messages)

        mock_client.rehydrate.assert_not_called()
        assert "[EMAIL_1]" in result.message.content

    @patch("llamaindex_ambientmeta.wrapper.AmbientMeta")
    def test_chat_multiple_messages(self, mock_client_class):
        """Test chat() sanitizes all messages."""
        from llamaindex_ambientmeta import PrivacyLLM

        mock_client = MagicMock()
        mock_client.sanitize.side_effect = [
            MagicMock(sanitized="Hello [PERSON_1]", session_id="ses_1"),
            MagicMock(sanitized="Email [EMAIL_1]", session_id="ses_2"),
            MagicMock(sanitized="Call [PHONE_1]", session_id="ses_3"),
        ]
        mock_client.rehydrate.return_value = MagicMock(text="Done")
        mock_client_class.return_value = mock_client

        fake_llm = FakeLLM(response="Done")

        wrapper = PrivacyLLM(llm=fake_llm, api_key="am_test_xxx")

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="Hello John"),
            ChatMessage(role=MessageRole.USER, content="Email john@example.com"),
            ChatMessage(role=MessageRole.USER, content="Call 555-0123"),
        ]
        wrapper.chat(messages)

        assert mock_client.sanitize.call_count == 3

    @patch("llamaindex_ambientmeta.wrapper.AmbientMeta")
    def test_stream_complete(self, mock_client_class):
        """Test stream_complete() sanitizes input."""
        from llamaindex_ambientmeta import PrivacyLLM

        mock_client = MagicMock()
        mock_client.sanitize.return_value = MagicMock(
            sanitized="Email [EMAIL_1]",
            session_id="ses_123",
        )
        mock_client_class.return_value = mock_client

        fake_llm = FakeLLM()
        fake_llm.stream_complete = MagicMock(return_value=iter([]))

        wrapper = PrivacyLLM(llm=fake_llm, api_key="am_test_xxx")
        list(wrapper.stream_complete("Email john@example.com"))

        mock_client.sanitize.assert_called_once()
        fake_llm.stream_complete.assert_called_once_with(
            "Email [EMAIL_1]", formatted=False,
        )

    @patch("llamaindex_ambientmeta.wrapper.AmbientMeta")
    def test_stream_chat(self, mock_client_class):
        """Test stream_chat() sanitizes all messages."""
        from llamaindex_ambientmeta import PrivacyLLM

        mock_client = MagicMock()
        mock_client.sanitize.side_effect = [
            MagicMock(sanitized="Hello [PERSON_1]", session_id="ses_1"),
            MagicMock(sanitized="Email [EMAIL_1]", session_id="ses_2"),
        ]
        mock_client_class.return_value = mock_client

        fake_llm = FakeLLM()
        fake_llm.stream_chat = MagicMock(return_value=iter([]))

        wrapper = PrivacyLLM(llm=fake_llm, api_key="am_test_xxx")

        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello John"),
            ChatMessage(role=MessageRole.USER, content="Email john@example.com"),
        ]
        list(wrapper.stream_chat(messages))

        assert mock_client.sanitize.call_count == 2


class TestPrivacyPostprocessor:
    """Tests for the PrivacyPostprocessor."""

    @patch("llamaindex_ambientmeta.node_postprocessor.AmbientMeta")
    def test_postprocess_sanitizes_nodes(self, mock_client_class):
        """Test that node content is sanitized."""
        from llamaindex_ambientmeta import PrivacyPostprocessor
        from llama_index.core.schema import NodeWithScore, TextNode

        mock_client = MagicMock()
        mock_client.sanitize.return_value = MagicMock(
            sanitized="Contact [EMAIL_1] for details",
        )
        mock_client_class.return_value = mock_client

        postprocessor = PrivacyPostprocessor(api_key="am_test_xxx")

        nodes = [
            NodeWithScore(
                node=TextNode(text="Contact john@example.com for details"),
                score=0.9,
            ),
        ]

        result = postprocessor.postprocess_nodes(nodes)

        mock_client.sanitize.assert_called_once()
        assert "[EMAIL_1]" in result[0].node.get_content()

    @patch("llamaindex_ambientmeta.node_postprocessor.AmbientMeta")
    def test_postprocess_multiple_nodes(self, mock_client_class):
        """Test that multiple nodes are all sanitized."""
        from llamaindex_ambientmeta import PrivacyPostprocessor
        from llama_index.core.schema import NodeWithScore, TextNode

        mock_client = MagicMock()
        mock_client.sanitize.side_effect = [
            MagicMock(sanitized="Contact [EMAIL_1]"),
            MagicMock(sanitized="Call [PHONE_1]"),
        ]
        mock_client_class.return_value = mock_client

        postprocessor = PrivacyPostprocessor(api_key="am_test_xxx")

        nodes = [
            NodeWithScore(node=TextNode(text="Contact john@example.com"), score=0.9),
            NodeWithScore(node=TextNode(text="Call 555-0123"), score=0.8),
        ]

        result = postprocessor.postprocess_nodes(nodes)

        assert mock_client.sanitize.call_count == 2
        assert "[EMAIL_1]" in result[0].node.get_content()
        assert "[PHONE_1]" in result[1].node.get_content()

    @patch("llamaindex_ambientmeta.node_postprocessor.AmbientMeta")
    def test_postprocess_empty_node(self, mock_client_class):
        """Test that empty text nodes are skipped."""
        from llamaindex_ambientmeta import PrivacyPostprocessor
        from llama_index.core.schema import NodeWithScore, TextNode

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        postprocessor = PrivacyPostprocessor(api_key="am_test_xxx")

        nodes = [
            NodeWithScore(node=TextNode(text=""), score=0.5),
        ]

        result = postprocessor.postprocess_nodes(nodes)

        mock_client.sanitize.assert_not_called()
        assert len(result) == 1

    @patch("llamaindex_ambientmeta.node_postprocessor.AmbientMeta")
    def test_postprocessor_custom_entities(self, mock_client_class):
        """Test that custom entities are forwarded to sanitize."""
        from llamaindex_ambientmeta import PrivacyPostprocessor
        from llama_index.core.schema import NodeWithScore, TextNode

        mock_client = MagicMock()
        mock_client.sanitize.return_value = MagicMock(sanitized="Contact [EMAIL_1]")
        mock_client_class.return_value = mock_client

        postprocessor = PrivacyPostprocessor(
            api_key="am_test_xxx",
            entities=["EMAIL", "PHONE"],
        )

        nodes = [
            NodeWithScore(node=TextNode(text="Contact john@example.com"), score=0.9),
        ]

        postprocessor.postprocess_nodes(nodes)

        mock_client.sanitize.assert_called_once_with(
            "Contact john@example.com",
            entities=["EMAIL", "PHONE"],
        )
