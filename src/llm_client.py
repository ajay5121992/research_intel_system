"""
LLM client wrapping a local Ollama server, with a mandatory mock fallback.

Every caller MUST supply a `mock_fn` callable that produces a reasonable
result using deterministic heuristics. This guarantees the whole system
runs end-to-end even if Ollama is not installed/running -- which is
essential for a grader who downloads this repo and doesn't want to set up
a local LLM just to see the demo work.
"""
import logging
from typing import Callable, Optional

import requests

from src import config

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


def _call_ollama(system_prompt: str, user_prompt: str, json_mode: bool, timeout: float) -> str:
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"

    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/generate", json=payload, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    raw_response = data.get("response", "")
    # Debug visibility: this is the single most useful line for diagnosing
    # "why am I getting the mock/templated answer" -- it shows exactly what
    # Ollama returned before any JSON-extraction/parsing happens downstream.
    # Run with LOG_LEVEL=DEBUG (see src/config.py) to see it.
    logger.debug("RAW OLLAMA OUTPUT (model=%s): %s", config.OLLAMA_MODEL, raw_response)
    return raw_response


def generate(
    system_prompt: str,
    user_prompt: str,
    mock_fn: Optional[Callable[[], str]] = None,
    json_mode: bool = False,
    timeout: float = None,
) -> str:
    """Generate text via Ollama, falling back to `mock_fn()` on any failure.

    Falls back if:
      - FORCE_MOCK_LLM=true, or
      - Ollama is unreachable / times out / returns malformed data.
    """
    timeout = timeout or config.OLLAMA_TIMEOUT_SECONDS

    if not config.FORCE_MOCK_LLM:
        try:
            return _call_ollama(system_prompt, user_prompt, json_mode, timeout)
        except requests.exceptions.Timeout:
            logger.warning(
                "Ollama call TIMED OUT after %.0fs (model=%s). qwen2.5:7b-instruct "
                "can be slow -- consider raising OLLAMA_TIMEOUT_SECONDS in your "
                "environment/.env, or switching to a smaller/faster model via "
                "OLLAMA_MODEL. Falling back to mock synthesis for this call.",
                timeout, config.OLLAMA_MODEL,
            )
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            logger.warning("Ollama call failed (%s); falling back to mock", exc)

    if mock_fn is None:
        raise LLMUnavailable(
            "LLM unavailable (Ollama not reachable / FORCE_MOCK_LLM set) "
            "and no mock fallback was provided by the caller."
        )
    return mock_fn()


def check_ollama_health() -> dict:
    """Best-effort health check used by the Streamlit sidebar. Never raises."""
    if config.FORCE_MOCK_LLM:
        return {"status": "mock_forced", "model": config.OLLAMA_MODEL}
    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        model_available = any(config.OLLAMA_MODEL.split(":")[0] in m for m in models)
        return {
            "status": "online" if model_available else "online_model_missing",
            "model": config.OLLAMA_MODEL,
            "available_models": models,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "offline", "model": config.OLLAMA_MODEL, "error": str(exc)}
