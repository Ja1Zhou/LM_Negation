# Copyright 2026 Zhejian Zhou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Minimal OpenRouter chat-completions client used by the LLM-annotation step of
this release (attention-output annotation for Figures 5 and 9).

Design (borrowed from a tested in-house transport):
- base URL https://openrouter.ai/api/v1, key from $OPENROUTER_API_KEY (override with --api_key_env)
- provider pinning: the annotation model is pinned to a bf16 endpoint with
  allow_fallbacks=False, so a run fails loudly instead of silently switching to
  a differently-quantized endpoint (results would stop being comparable)
- reasoning stays enabled for openai/gpt-oss* (required by that model family);
  it is disabled for every other model so the reply is the bare JSON we parse
- bounded exponential backoff on 429 / 5xx / connection errors, then give up

Usage:
    client = OpenRouterClient()                       # openai/gpt-oss-120b, bf16-pinned
    text, meta = client.chat([{"role": "user", "content": prompt}])
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"  # the annotator used in the paper (Sec. 5.3)
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"

# canonical model id -> extra_body.provider. Endpoint slugs change over time;
# list current ones at https://openrouter.ai/<model>/providers before editing.
PROVIDER_ROUTING: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-120b": {"only": ["deepinfra/bf16"], "allow_fallbacks": False},
}


@dataclass
class ChatMeta:
    model: str
    provider: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    attempts: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class OpenRouterClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key_env: str = DEFAULT_KEY_ENV,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_s: float = 120.0,
        max_attempts: int = 8,
        provider_only: list[str] | None = None,
        allow_fallbacks: bool | None = None,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set. Put it in the environment (or a .env file "
                "loaded before import) — never commit it."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self._client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=timeout_s, max_retries=0)

        routing = PROVIDER_ROUTING.get(model.lower())
        if provider_only is not None:
            routing = {"only": list(provider_only), "allow_fallbacks": bool(allow_fallbacks)}
        elif routing is not None and allow_fallbacks is not None:
            routing = {**routing, "allow_fallbacks": bool(allow_fallbacks)}
        self.extra_body: dict[str, Any] = {}
        if routing:
            self.extra_body["provider"] = routing
        if not model.lower().startswith("openai/gpt-oss"):
            self.extra_body["reasoning"] = {"enabled": False}

    def describe(self) -> str:
        return f"model={self.model} extra_body={self.extra_body} T={self.temperature} max_tokens={self.max_tokens}"

    def chat(self, messages: list[dict], **overrides) -> tuple[str, ChatMeta]:
        """Return (assistant text, meta). Raises after max_attempts failures."""
        params = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        params.update(overrides)
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._client.chat.completions.create(**params)
                content = resp.choices[0].message.content
                if content is None:
                    raise RuntimeError(f"empty completion (finish_reason={resp.choices[0].finish_reason})")
                usage = getattr(resp, "usage", None)
                extra = getattr(resp, "model_extra", None) or {}
                meta = ChatMeta(
                    model=getattr(resp, "model", self.model),
                    provider=extra.get("provider"),
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    attempts=attempt,
                )
                return content.strip(), meta
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
            except APIStatusError as exc:
                last_exc = exc
                if exc.status_code < 500 and exc.status_code != 429:
                    raise  # 4xx other than rate limit: not retryable (bad key, bad model, no bf16 slot)
            delay = min(60.0, 2.0 ** (attempt - 1)) * (0.5 + random.random())
            print(f"[openrouter] attempt {attempt}/{self.max_attempts} failed: {last_exc}; retry in {delay:.1f}s")
            time.sleep(delay)
        raise RuntimeError(f"OpenRouter request failed after {self.max_attempts} attempts: {last_exc}")
