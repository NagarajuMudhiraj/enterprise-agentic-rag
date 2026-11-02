"""Unified LLM access with pooling, retries, provider failover and cost accounting.

Design notes:
  * One module-level AsyncClient is reused for every call. Creating a client per
    request (the previous behaviour) discarded the connection pool and paid a
    TLS handshake per LLM call.
  * Retries cover 429, 5xx, timeouts and transport errors with jittered
    exponential backoff. A 4xx that is not 429 is a client bug and is not retried.
  * If the primary provider is exhausted, the secondary is tried before failing.
"""

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.common.metrics import metrics
from src.common.config import settings

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


async def get_http_client() -> httpx.AsyncClient:
    """Return the shared, lazily-created HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        async with _client_lock:
            if _client is None or _client.is_closed:
                _client = httpx.AsyncClient(
                    timeout=httpx.Timeout(settings.LLM_TIMEOUT_SECONDS),
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                )
    return _client


async def close_http_client() -> None:
    """Release pooled connections on application shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


@dataclass(frozen=True)
class ProviderTarget:
    provider: str
    api_key: str
    base_url: str
    model: str


class LLMProvider:
    """
    Unified LLM provider supporting Groq, OpenAI, and a deterministic local
    fallback so the pipeline degrades instead of failing when no key is set.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        groq_key = api_key or settings.GROQ_API_KEY
        openai_key = settings.OPENAI_API_KEY

        self._targets: List[ProviderTarget] = []
        if groq_key:
            self._targets.append(
                ProviderTarget(
                    provider="groq",
                    api_key=groq_key,
                    base_url=(base_url or settings.GROQ_BASE_URL),
                    model=model or settings.GROQ_MODEL,
                )
            )
        if openai_key:
            self._targets.append(
                ProviderTarget(
                    provider="openai",
                    api_key=openai_key,
                    base_url=(base_url or settings.OPENAI_BASE_URL),
                    model=model or settings.LLM_MODEL,
                )
            )

        self.provider_type = self._targets[0].provider if self._targets else "local"
        self.api_key = self._targets[0].api_key if self._targets else None
        self.base_url = self._targets[0].base_url if self._targets else None
        self.model = self._targets[0].model if self._targets else "local-deterministic"
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

    # ------------------------------------------------------------------ public
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Invokes the LLM with optional tool specs.

        Returns a dict with `status` in {success, error, local}. Never raises for
        provider-side failures — callers implement their own degradation.
        """
        if not self._targets:
            return {
                "status": "local",
                "provider": "local",
                "content": None,
                "tool_calls": [],
            }

        attempts_allowed = settings.LLM_MAX_RETRIES if max_retries is None else max_retries
        errors: List[str] = []

        for target in self._targets:
            payload = self._build_payload(target, messages, tools, temperature, max_tokens, model)
            result = await self._call_with_retries(target, payload, attempts_allowed)
            if result.get("status") == "success":
                self._record_usage(target, result.get("usage") or {})
                return result
            errors.append(f"{target.provider}: {result.get('error', 'unknown error')}")
            if target is not self._targets[-1]:
                logger.warning(
                    "Provider failed, failing over",
                    extra={"provider": target.provider, "model": target.model},
                )

        return {"status": "error", "error": " | ".join(errors)}

    async def aclose(self) -> None:
        await close_http_client()

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _build_payload(
        target: ProviderTarget,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
        model_override: Optional[str],
    ) -> Dict[str, Any]:
        target_model = model_override or target.model
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if target.provider == "groq" and "gpt-oss" in target_model:
            # Keeps hidden reasoning short so the answer fits in max_tokens and TPM limits
            payload["reasoning_effort"] = "low"
        return payload

    async def _call_with_retries(
        self,
        target: ProviderTarget,
        payload: Dict[str, Any],
        max_retries: int,
    ) -> Dict[str, Any]:
        client = await get_http_client()
        url = f"{target.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {target.api_key}",
            "Content-Type": "application/json",
        }
        last_error = "unknown error"

        for attempt in range(max_retries + 1):
            started = asyncio.get_event_loop().time()
            try:
                resp = await client.post(url, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"transport error: {exc.__class__.__name__}"
                metrics.incr("llm_requests_total", {"provider": target.provider, "outcome": "transport_error"})
                if attempt < max_retries:
                    await self._sleep_before_retry(attempt, None)
                    continue
                logger.error(
                    "LLM transport failure after retries",
                    extra={"provider": target.provider, "attempt": attempt + 1},
                )
                return {"status": "error", "error": last_error}

            elapsed = asyncio.get_event_loop().time() - started
            metrics.observe("llm_request_duration_seconds", elapsed, {"provider": target.provider})

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    message = data["choices"][0]["message"]
                except (KeyError, IndexError, ValueError) as exc:
                    logger.error("Malformed LLM response: %s", exc)
                    metrics.incr("llm_requests_total", {"provider": target.provider, "outcome": "malformed"})
                    return {"status": "error", "error": f"Malformed provider response: {exc}"}

                metrics.incr("llm_requests_total", {"provider": target.provider, "outcome": "success"})
                return {
                    "status": "success",
                    "provider": target.provider,
                    "model": payload.get("model", target.model),
                    "content": message.get("content"),
                    "reasoning": message.get("reasoning"),
                    "tool_calls": message.get("tool_calls") or [],
                    "usage": data.get("usage", {}),
                }

            metrics.incr(
                "llm_requests_total",
                {"provider": target.provider, "outcome": f"http_{resp.status_code}"},
            )
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"

            if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                await self._sleep_before_retry(attempt, self._retry_after_seconds(resp))
                continue

            logger.warning(
                "LLM call failed",
                extra={"provider": target.provider, "attempt": attempt + 1},
            )
            return {"status": "error", "status_code": resp.status_code, "error": last_error}

        return {"status": "error", "error": last_error}

    async def _sleep_before_retry(self, attempt: int, retry_after: Optional[float]) -> None:
        if retry_after is not None:
            delay = min(retry_after, settings.LLM_MAX_BACKOFF_SECONDS)
        else:
            delay = min(2 ** attempt, settings.LLM_MAX_BACKOFF_SECONDS)
            delay += random.uniform(0, 0.5)  # jitter avoids synchronized retry storms
        logger.info("Retrying LLM call", extra={"attempt": attempt + 1})
        await asyncio.sleep(delay)

    def _record_usage(self, target: ProviderTarget, usage: Dict[str, Any]) -> None:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if not prompt_tokens and not completion_tokens:
            return

        cost = (
            prompt_tokens / 1000.0 * settings.LLM_INPUT_COST_PER_1K
            + completion_tokens / 1000.0 * settings.LLM_OUTPUT_COST_PER_1K
        )
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost

        metrics.incr("llm_tokens_total", {"direction": "input", "provider": target.provider}, prompt_tokens)
        metrics.incr("llm_tokens_total", {"direction": "output", "provider": target.provider}, completion_tokens)
        if cost:
            metrics.incr("llm_cost_usd_total", {"provider": target.provider}, cost)

        logger.info(
            "llm_call",
            extra={
                "provider": target.provider,
                "model": target.model,
                "tokens_in": prompt_tokens,
                "tokens_out": completion_tokens,
                "cost_usd": round(cost, 6),
            },
        )

    def usage_summary(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 6),
        }

    @staticmethod
    def _retry_after_seconds(resp: httpx.Response) -> Optional[float]:
        header = resp.headers.get("retry-after")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        match = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)(ms|s)", resp.text)
        if match:
            minutes = float(match.group(1) or 0)
            value = float(match.group(2))
            seconds = value / 1000.0 if match.group(3) == "ms" else value
            return minutes * 60 + seconds
        return None
