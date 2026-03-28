# llamaindex-ambientmeta

[![PyPI](https://img.shields.io/pypi/v/llamaindex-ambientmeta)](https://pypi.org/project/llamaindex-ambientmeta/)
[![CI](https://github.com/AmbientMeta/llamaindex-ambientmeta/actions/workflows/ci.yml/badge.svg)](https://github.com/AmbientMeta/llamaindex-ambientmeta/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/AmbientMeta/llamaindex-ambientmeta/blob/main/LICENSE)

PII protection for LlamaIndex workflows. Sanitize sensitive data before it reaches your LLM. Restore originals after. The LLM never sees real data.

## Installation

```bash
pip install llamaindex-ambientmeta
```

## Quick Start

### LLM Wrapper

```python
from llamaindex_ambientmeta import PrivacyLLM
from llama_index.llms.openai import OpenAI

# Wrap any LlamaIndex LLM
llm = PrivacyLLM(
    llm=OpenAI(model="gpt-4"),
    api_key="am_live_xxx",  # Get your key at ambientmeta.com
)

# PII is automatically sanitized before the LLM call
# and restored in the response
response = llm.complete("Email john@example.com about the meeting")
# The LLM never sees the actual email address
```

### RAG Postprocessor

Strip PII from retrieved context before it reaches the LLM:

```python
from llamaindex_ambientmeta import PrivacyPostprocessor
from llama_index.core import VectorStoreIndex

postprocessor = PrivacyPostprocessor(api_key="am_live_xxx")

query_engine = index.as_query_engine(
    node_postprocessors=[postprocessor],
)

# Retrieved nodes are sanitized before the LLM sees them
response = query_engine.query("What is John Smith's email?")
```

## How It Works

```
Input: "Email john@acme.com about the merger"
     ↓ (sanitize)
LLM sees: "Email [EMAIL_1] about the merger"
     ↓ (LLM responds)
LLM output: "I'll contact [EMAIL_1] tomorrow"
     ↓ (rehydrate)
Output: "I'll contact john@acme.com tomorrow"
```

## Features

### Complete and Chat

```python
# Completion
response = llm.complete("Contact john@example.com")

# Chat
from llama_index.core.base.llms.types import ChatMessage, MessageRole

messages = [
    ChatMessage(role=MessageRole.USER, content="Email john@example.com"),
]
response = llm.chat(messages)
```

### Streaming

Streaming methods sanitize input but stream output as-is. Use `complete()` or `chat()` if you need automatic rehydration.

```python
# Streamed tokens are not rehydrated (full response needed for placeholder replacement)
for chunk in llm.stream_complete("Contact john@example.com"):
    print(chunk.text, end="")
```

### Custom Entity Types

```python
llm = PrivacyLLM(
    llm=OpenAI(),
    api_key="am_xxx",
    entities=["EMAIL", "PHONE", "SSN"],  # Only detect these types
)
```

### Disable Auto-Rehydration

```python
llm = PrivacyLLM(
    llm=OpenAI(),
    api_key="am_xxx",
    auto_rehydrate=False,
)

response = llm.complete("Email john@example.com")
# Response: "I'll contact [EMAIL_1] tomorrow"
```

## Supported Entity Types

See the [full entity type reference](https://docs.ambientmeta.com/api-reference) for the complete list. Common types include:

`PERSON`, `EMAIL`, `PHONE`, `SSN`, `CREDIT_CARD`, `LOCATION`, `ORGANIZATION`, `ADDRESS`, `DOB`, `IP_ADDRESS`, `URL`, `REFERENCE_ID`

## Documentation

- [Full Documentation](https://docs.ambientmeta.com/guides/llamaindex)
- [API Reference](https://docs.ambientmeta.com/api-reference)
- [AmbientMeta](https://ambientmeta.com)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
