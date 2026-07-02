"""
Forces every fallback path so tests are deterministic and require zero
external services (no internet, no Ollama, no GPU). This is set before any
project module is imported.
"""
import os

os.environ.setdefault("FORCE_HASHING_EMBEDDER", "true")
os.environ.setdefault("FORCE_REGEX_NER", "true")
os.environ.setdefault("FORCE_MOCK_LLM", "true")
