from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from eval_harness.core.interfaces import ModelBackend
from eval_harness.core.models import ModelResponse
from eval_harness.core.registry import BACKENDS


class BackendRequestError(RuntimeError):
    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


class OpenAICompatibleBackend(ModelBackend):
    def __init__(self, config: dict[str, Any], *, generation_config: dict[str, Any] | None = None):
        endpoint = config.get("endpoint")
        if config.get("endpoint_env"):
            endpoint = os.environ.get(config["endpoint_env"])
            if not endpoint:
                raise ValueError(f"required endpoint environment variable is unset: {config['endpoint_env']}")
        endpoints = config.get("endpoints") or ([endpoint] if endpoint else [])
        if not endpoints:
            raise ValueError("at least one backend endpoint is required")
        self.endpoints = [value.rstrip("/") for value in endpoints]
        self._endpoint_index = 0
        self._endpoint_lock = asyncio.Lock()
        self.model = config["model"]
        generation_config = generation_config or {}
        self.timeout = float(generation_config.get("timeout_sec", 300))
        self.retries = int(generation_config.get("retries", 3))
        self.api_key = config.get("api_key", "EMPTY")
        if config.get("api_key_env"):
            self.api_key = os.environ.get(config["api_key_env"], "")
            if not self.api_key:
                raise ValueError(
                    f"required API key environment variable is unset: {config['api_key_env']}"
                )
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def health_check(self) -> None:
        await asyncio.gather(*(self._health_check_endpoint(endpoint) for endpoint in self.endpoints))

    async def _health_check_endpoint(self, endpoint: str) -> None:
        candidates = [f"{endpoint.removesuffix('/v1')}/health", f"{endpoint}/models"]
        errors: list[str] = []
        for url in candidates:
            try:
                response = await self.client.get(url, timeout=10)
                if response.is_success:
                    return
                errors.append(f"{url}: HTTP {response.status_code}")
            except httpx.HTTPError as exc:
                errors.append(f"{url}: {type(exc).__name__}")
        raise BackendRequestError("backend health check failed: " + "; ".join(errors))

    async def _next_endpoint(self) -> str:
        async with self._endpoint_lock:
            endpoint = self.endpoints[self._endpoint_index % len(self.endpoints)]
            self._endpoint_index += 1
            return endpoint

    async def generate(
        self, messages: list[dict[str, Any]], generation_config: dict[str, Any]
    ) -> ModelResponse:
        excluded = {"n", "concurrency", "timeout_sec", "retries"}
        payload = {k: v for k, v in generation_config.items() if k not in excluded}
        payload.update({"model": self.model, "messages": messages, "n": 1})
        last_error: Exception | None = None
        last_endpoint: str | None = None
        for attempt in range(1, self.retries + 1):
            endpoint = await self._next_endpoint()
            last_endpoint = endpoint
            started = time.perf_counter()
            try:
                response = await self.client.post(f"{endpoint}/chat/completions", json=payload)
                if response.status_code >= 500 or response.status_code == 429:
                    raise BackendRequestError(
                        f"retryable HTTP {response.status_code}: {response.text[:1000]}"
                    )
                if response.status_code >= 400:
                    raise BackendRequestError(
                        f"non-retryable HTTP {response.status_code}: {response.text[:2000]}",
                        metadata={"endpoint": endpoint, "attempts": attempt},
                    )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise BackendRequestError("invalid response: no choices")
                choice = choices[0]
                message = choice.get("message") or {}
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                content = message.get("content")
                tool_calls = message.get("tool_calls")
                if not reasoning and not content and not tool_calls:
                    raise BackendRequestError("empty model response")
                return ModelResponse(
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=choice.get("finish_reason"),
                    usage=data.get("usage") or {},
                    raw_response=data,
                    metadata={
                        "endpoint": endpoint,
                        "attempt": attempt,
                        "latency_sec": time.perf_counter() - started,
                    },
                )
            except (httpx.TimeoutException, httpx.TransportError, BackendRequestError) as exc:
                last_error = exc
                if isinstance(exc, BackendRequestError) and str(exc).startswith("non-retryable"):
                    raise
                if attempt < self.retries:
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
                    continue
                break
            except (ValueError, KeyError) as exc:
                raise BackendRequestError(f"non-retryable response error: {exc}") from exc
        raise BackendRequestError(
            f"request failed after {self.retries} attempts: {last_error}",
            metadata={"endpoint": last_endpoint, "attempts": self.retries},
        )

    async def close(self) -> None:
        await self.client.aclose()


BACKENDS.register(
    "openai_compatible",
    lambda config, **kwargs: OpenAICompatibleBackend(
        config, generation_config=kwargs.get("generation_config")
    ),
)
