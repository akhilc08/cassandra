"""OpenAI transport for the time-machine forecaster.

Reuses `build_prompt` and `Forecast` from the Claude forecaster verbatim — only
the transport differs, so the prompt the model sees is byte-identical to the one
that produced the archived baseline. The Claude path is left untouched so
`data/baselines/claude-fable-5-2026-06-10/` stays reproducible.

Two properties this module must preserve, both load-bearing:

1. Market-blindness. `build_prompt` takes no price argument; nothing here adds one.
2. `failed=True` on any call or parse failure, never an exception and never a
   silent default. A sentinel p_model=0.5 against a market at 0.20 is edge 0.30 —
   inside the trading band — so a failure that reaches the strategy layer
   manufactures a confident bogus trade. Callers MUST check `.failed`.
"""

from __future__ import annotations

import json
import os

import structlog

from oracle.agents.forecaster import Forecast, build_prompt

logger = structlog.get_logger()

# Cheapest current tier that supports strict structured outputs. Recorded in
# every prediction row, so changing it invalidates a running pre-registration.
FORECAST_MODEL = "gpt-5.6-luna"

# Reasoning tokens bill as output tokens. Leaving effort unset risks defaulting
# to medium (~3x cost) — the docs put probability estimation from supplied
# evidence under "none". Set explicitly, always.
REASONING_EFFORT = "none"

# Stable across calls so the shared prompt prefix caches.
PROMPT_CACHE_KEY = "cassandra-forecast-v1"

# strict mode requires: every property in `required`, additionalProperties false.
# "none" MUST stay in the enum — it was 605 of 959 predictions in the baseline.
FORECAST_SCHEMA = {
    "type": "object",
    "properties": {
        "p_yes": {"type": "number"},
        "evidence_strength": {
            "type": "string",
            "enum": ["none", "weak", "moderate", "strong"],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["p_yes", "evidence_strength", "reasoning"],
    "additionalProperties": False,
}


def _failed(reason: str, raw: str = "") -> Forecast:
    return Forecast(
        p_model=0.5,
        evidence_strength="none",
        reasoning=reason,
        failed=True,
        model=FORECAST_MODEL,
        raw_response=raw[:2000],
    )


def parse_response(text: str) -> Forecast:
    """Parse a strict-schema reply. Bounds are enforced here, not in the schema:
    strict mode's numeric minimum/maximum support is not guaranteed."""
    try:
        data = json.loads(text)
        p = float(data["p_yes"])
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        logger.warning("forecaster_openai.parse_failed", text=text[:120])
        return _failed("parse failure", text)
    if p != p or p in (float("inf"), float("-inf")):  # NaN / inf survive json.loads
        logger.warning("forecaster_openai.non_finite", text=text[:120])
        return _failed("non-finite p_yes", text)
    return Forecast(
        p_model=max(0.01, min(0.99, p)),
        evidence_strength=str(data.get("evidence_strength", "none")),
        reasoning=str(data.get("reasoning", ""))[:1000],
        model=FORECAST_MODEL,
        raw_response=text[:2000],
    )


def _extract_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()
    # Fallback: walk output items for the first text content block.
    for item in getattr(resp, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            t = getattr(block, "text", None)
            if t:
                return str(t).strip()
    return ""


async def forecast(
    question: str,
    as_of: str,
    headlines: list[dict],
    world_events: list[dict] | None = None,
    description: str = "",
    timeout: int = 120,
    client=None,
) -> Forecast:
    """Market-blind P(YES) for one market as of `as_of`. Never raises."""
    from openai import AsyncOpenAI

    if client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            logger.warning("forecaster_openai.no_api_key")
            return _failed("OPENAI_API_KEY not set")
        client = AsyncOpenAI(api_key=key, timeout=timeout)

    prompt = build_prompt(question, as_of, headlines, world_events, description)
    try:
        resp = await client.responses.create(
            model=FORECAST_MODEL,
            reasoning={"effort": REASONING_EFFORT},
            input=[{"role": "user", "content": prompt}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "forecast",
                    "schema": FORECAST_SCHEMA,
                    "strict": True,
                }
            },
            prompt_cache_key=PROMPT_CACHE_KEY,
        )
    except Exception as e:  # noqa: BLE001 — must never propagate to the strategy layer
        logger.warning(
            "forecaster_openai.call_failed",
            question=question[:60],
            error=f"{type(e).__name__}: {e}"[:200],
        )
        return _failed(f"api call failed: {type(e).__name__}")

    text = _extract_text(resp)
    if not text:
        logger.warning("forecaster_openai.empty_response", question=question[:60])
        return _failed("empty response")
    return parse_response(text)
