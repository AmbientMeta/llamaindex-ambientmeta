# Changelog

## 0.1.0 — 2026-03-28

Initial release.

- `PrivacyLLM` — wrap any LlamaIndex LLM with automatic PII sanitization and rehydration
- `PrivacyPostprocessor` — sanitize PII in retrieved RAG nodes before the LLM sees them
- `complete()` and `chat()` with full sanitize/rehydrate round-trip
- `stream_complete()` and `stream_chat()` with input sanitization (output streamed as-is)
- Custom entity type filtering
- Configurable auto-rehydration (on by default)
