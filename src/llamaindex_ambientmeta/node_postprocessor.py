"""Privacy-aware node postprocessor for LlamaIndex RAG pipelines.

Sanitizes PII in retrieved nodes before they are sent to the LLM,
and optionally rehydrates the LLM response.
"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field, PrivateAttr

from ambientmeta import AmbientMeta


class PrivacyPostprocessor(BaseNodePostprocessor):
    """Postprocessor that sanitizes PII in retrieved nodes.

    Use this in RAG pipelines to strip PII from retrieved context
    before it's included in the LLM prompt.

    Example:
        from llamaindex_ambientmeta import PrivacyPostprocessor
        from llama_index.core import VectorStoreIndex

        postprocessor = PrivacyPostprocessor(api_key="am_live_xxx")

        query_engine = index.as_query_engine(
            node_postprocessors=[postprocessor],
        )

        # Retrieved nodes are sanitized before the LLM sees them
        response = query_engine.query("What is John Smith's email?")
    """

    api_key: str = Field(description="AmbientMeta API key")
    base_url: str = Field(
        default="https://api.ambientmeta.com",
        description="AmbientMeta API base URL",
    )
    entities: Optional[List[str]] = Field(
        default=None,
        description="Entity types to detect. None means all types.",
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

    @classmethod
    def class_name(cls) -> str:
        return "PrivacyPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Sanitize PII in each node's text content."""
        for node_with_score in nodes:
            text = node_with_score.node.get_content()
            if text:
                result = self._client_instance.sanitize(
                    text, entities=self.entities
                )
                node_with_score.node.set_content(result.sanitized)

        return nodes
