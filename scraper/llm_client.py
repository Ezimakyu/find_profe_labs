from __future__ import annotations

import json

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from scraper.config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        max_output_tokens: int = 3000,
    ) -> BaseModel:
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
        data = json.loads(raw_text)
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
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_output_tokens,
        )
        return response.choices[0].message.content or ""

