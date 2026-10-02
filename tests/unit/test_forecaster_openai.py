"""Tests for the OpenAI transport (`oracle.agents.forecaster_openai`).

The transport must (a) record which model snapshot actually answered and whether
reasoning tokens were billed — the alias `gpt-5.6-luna` can be re-pointed by the
provider and an effort-default change would show up ONLY there — and (b) never
raise and never hand the strategy layer a sentinel probability with `failed`
unset. A fake client stands in for the SDK; no API key, no network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from oracle.agents import forecaster_openai as fo
from oracle.agents.forecaster import Forecast

SERVED = "gpt-5.6-luna-2026-08-01"
GOOD_TEXT = json.dumps({"p_yes": 0.23, "evidence_strength": "weak", "reasoning": "base rate"})


class FakeResponses:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class FakeClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.responses = FakeResponses(response, error)


def make_response(text: str, *, with_usage: bool = True, model: str | None = SERVED,
                  output_tokens: int = 60, reasoning_tokens: int = 0):
    resp = SimpleNamespace(output_text=text)
    if model is not None:
        resp.model = model
    if with_usage:
        resp.usage = SimpleNamespace(
            output_tokens=output_tokens,
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        )
    return resp


async def run(client) -> Forecast:
    return await fo.forecast(
        question="Will X happen by 2026-10-01?",
        as_of="2026-09-25",
        headlines=[],
        world_events=[{"date": "2026-09-20", "text": "something relevant"}],
        client=client,
    )


class TestProvenance:
    async def test_served_model_and_usage_are_recorded(self):
        client = FakeClient(make_response(GOOD_TEXT))
        fc = await run(client)
        assert fc.failed is False
        assert fc.p_model == pytest.approx(0.23)
        assert fc.evidence_strength == "weak"
        assert fc.model == fo.FORECAST_MODEL  # the alias we asked for ...
        assert fc.served_model == SERVED  # ... and the snapshot that answered
        assert fc.output_tokens == 60
        assert fc.reasoning_tokens == 0

    async def test_nonzero_reasoning_tokens_are_surfaced_not_hidden(self):
        # effort "none" must bill 0 reasoning tokens; a nonzero count is the one
        # signal that the provider changed the default under us.
        client = FakeClient(make_response(GOOD_TEXT, output_tokens=180, reasoning_tokens=119))
        fc = await run(client)
        assert fc.reasoning_tokens == 119
        assert fc.output_tokens == 180

    async def test_response_without_usage_still_parses(self):
        client = FakeClient(make_response(GOOD_TEXT, with_usage=False, model=None))
        fc = await run(client)
        assert fc.failed is False
        assert fc.p_model == pytest.approx(0.23)
        assert fc.served_model == ""
        assert fc.output_tokens is None
        assert fc.reasoning_tokens is None

    async def test_request_uses_the_pinned_alias_and_no_reasoning(self):
        client = FakeClient(make_response(GOOD_TEXT))
        await run(client)
        (call,) = client.responses.calls
        assert call["model"] == fo.FORECAST_MODEL == "gpt-5.6-luna"
        assert call["reasoning"] == {"effort": "none"}
        assert call["prompt_cache_key"] == fo.PROMPT_CACHE_KEY
        assert call["text"]["format"]["strict"] is True
        assert call["text"]["format"]["schema"] is fo.FORECAST_SCHEMA


class TestFailureIsNeverSilent:
    async def test_unparseable_text_keeps_failed_true(self):
        client = FakeClient(make_response("I think about 0.4, hard to say."))
        fc = await run(client)
        assert fc.failed is True
        assert fc.p_model == 0.5
        assert fc.reasoning == "parse failure"
        # Provenance is still recorded on a failed parse — that is exactly when
        # an operator wants to know which snapshot produced the garbage.
        assert fc.served_model == SERVED
        assert fc.output_tokens == 60

    async def test_non_finite_p_yes_keeps_failed_true(self):
        raw = '{"p_yes": NaN, "evidence_strength": "none", "reasoning": ""}'
        client = FakeClient(make_response(raw))
        fc = await run(client)
        assert fc.failed is True
        assert fc.p_model == 0.5

    async def test_empty_response_keeps_failed_true(self):
        client = FakeClient(SimpleNamespace(output_text="", output=[], model=SERVED))
        fc = await run(client)
        assert fc.failed is True
        assert fc.reasoning == "empty response"

    async def test_api_exception_is_swallowed_into_failed(self):
        client = FakeClient(error=RuntimeError("boom"))
        fc = await run(client)
        assert fc.failed is True
        assert fc.p_model == 0.5
        assert "RuntimeError" in fc.reasoning
        assert fc.served_model == ""
        assert fc.output_tokens is None

    async def test_missing_api_key_without_client_fails_before_any_call(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        fc = await fo.forecast("q", "2026-09-25", headlines=[], client=None)
        assert fc.failed is True
        assert "OPENAI_API_KEY" in fc.reasoning
