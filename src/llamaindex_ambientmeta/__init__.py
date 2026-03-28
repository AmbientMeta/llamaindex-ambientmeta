"""LlamaIndex integration for AmbientMeta Privacy Gateway.

This package provides LlamaIndex-compatible components that automatically
sanitize PII before sending to LLMs and rehydrate responses.

Usage:
    from llamaindex_ambientmeta import PrivacyLLM
    from llama_index.llms.openai import OpenAI

    # Wrap any LlamaIndex LLM
    llm = PrivacyLLM(
        llm=OpenAI(model="gpt-4"),
        api_key="am_live_xxx",
    )

    # PII is automatically sanitized before the LLM call
    # and restored in the response
    response = llm.complete("Email john@example.com about the meeting")
"""

from llamaindex_ambientmeta.wrapper import PrivacyLLM
from llamaindex_ambientmeta.node_postprocessor import PrivacyPostprocessor

__all__ = ["PrivacyLLM", "PrivacyPostprocessor"]
__version__ = "0.1.0"
