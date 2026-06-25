from __future__ import annotations

import json
import re
import threading

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scraper.config import Settings

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Cap how many chat calls are in flight at once. The OpenAI account is bound by
# a tokens-per-minute limit, so slamming it with one call per worker thread just
# trips 429s; a smaller in-flight ceiling smooths the burst and, combined with
# the SDK's Retry-After handling, keeps the run under the limit without dying.
_llm_semaphore = threading.Semaphore(8)


def configure_llm_concurrency(max_in_flight: int) -> None:
    global _llm_semaphore
    _llm_semaphore = threading.Semaphore(max(1, max_in_flight))


def _extract_json(raw_text: str) -> str:
    """Pull a JSON object out of a model reply.

    Local models (and occasionally hosted ones) wrap JSON in markdown fences or
    add prose around it. We strip fences and, failing that, grab the outermost
    ``{...}`` so parsing is robust regardless of the backend.
    """
    text = (raw_text or "").strip()
    if not text:
        return "{}"
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # base_url lets us target an OpenAI-compatible local server (e.g. Ollama
        # at http://localhost:11434/v1). Local servers ignore the key, so a
        # placeholder keeps the OpenAI SDK happy when no real key is needed.
        client_kwargs: dict = {
            "api_key": settings.openai_api_key or "local",
            # The SDK transparently retries 429/5xx and honors the server's
            # Retry-After, which is the cleanest way to ride out a tokens-per-
            # minute limit without bespoke backoff math.
            "max_retries": 6,
        }
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**client_kwargs)

    @retry(
        wait=wait_exponential(min=2, max=30),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type((ValueError, json.JSONDecodeError)),
        reraise=True,
    )
    def json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        max_output_tokens: int = 3000,
    ) -> BaseModel:
        with _llm_semaphore:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=max_output_tokens,
            )
        raw_text = response.choices[0].message.content or "{}"
        data = json.loads(_extract_json(raw_text))
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"LLM JSON failed schema validation: {exc}") from exc

    def text_response(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 2500,
    ) -> str:
        with _llm_semaphore:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_output_tokens,
            )
        return response.choices[0].message.content or ""
