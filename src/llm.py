"""Ollama client and communication module.

Handles HTTP communication with the local Ollama service (/api/chat),
enforces timeouts and temperature configurations, and maps HTTP/network
failures into clean domain exceptions.
"""

from typing import Optional
import requests

from src.config import AppConfig, load_config


class LLMError(Exception):
    """Base exception for LLM communication errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when the Ollama service cannot be reached."""
    pass


class LLMModelNotFoundError(LLMError):
    """Raised when the requested model is not pulled in Ollama."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when the Ollama request times out."""
    pass


def chat(
    messages: list[dict[str, str]],
    config: Optional[AppConfig] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: Optional[int] = None,
) -> str:
    """Send a chat completion request to the local Ollama instance.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        config: Optional AppConfig instance. Loaded if None.
        base_url: Optional override for Ollama base URL.
        model: Optional override for Ollama model name.
        temperature: Optional override for generation temperature.
        timeout: Optional override for request timeout in seconds.

    Returns:
        str: Assistant response message content.

    Raises:
        LLMConnectionError: If connection to Ollama is refused / unreachable.
        LLMModelNotFoundError: If the configured model is not found (HTTP 404).
        LLMTimeoutError: If the model takes longer than the timeout to respond.
        LLMError: For any other HTTP or parsing errors.
    """
    cfg = config or load_config()

    target_url = (base_url or cfg.ollama_base_url).rstrip("/") + "/api/chat"
    target_model = model or cfg.llm_model
    target_temp = temperature if temperature is not None else cfg.temperature
    target_timeout = timeout if timeout is not None else cfg.llm_timeout_seconds

    payload = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": target_temp,
        },
    }

    try:
        response = requests.post(
            target_url,
            json=payload,
            timeout=target_timeout,
        )
    except requests.exceptions.ConnectionError as e:
        raise LLMConnectionError(
            f"Ollama is not running (start it with 'ollama serve' or launch the Ollama app)"
        ) from e
    except requests.exceptions.Timeout as e:
        raise LLMTimeoutError(
            f"The model took too long to respond (timeout after {target_timeout} seconds)"
        ) from e
    except requests.exceptions.RequestException as e:
        raise LLMError(f"Network error communicating with Ollama: {e}") from e

    if response.status_code == 404:
        raise LLMModelNotFoundError(
            f"Model '{target_model}' not found in Ollama (run 'ollama pull {target_model}')"
        )
    elif response.status_code != 200:
        raise LLMError(
            f"Ollama returned HTTP error {response.status_code}: {response.text}"
        )

    try:
        data = response.json()
        if not isinstance(data, dict) or "message" not in data:
            raise ValueError("Missing 'message' field in response JSON")
        message = data["message"]
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError("Missing 'content' field in response message")
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError("'content' must be a string")
        return content
    except Exception as e:
        raise LLMError(f"Malformed response from Ollama: {e}") from e
