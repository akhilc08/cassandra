"""Time-machine forecaster — calibrated P(YES) estimate as of a past timestamp.

The forecaster sees only information available at prediction time T: dated world
events and news headlines from before T. It is deliberately BLIND to the market
price and price trend (see the comment on FORECAST_PROMPT below) — an earlier
design supplied the price as a prior and the model simply anchored to it.
Synthesis runs through the Claude Code CLI (`claude -p`).

The OpenAI transport lives in `forecaster_openai.py` and reuses `build_prompt`
and `Forecast` from here unchanged.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

# The forecaster is deliberately BLIND to the market price: shown the price,
# LLMs anchor to it almost perfectly (pilot run: every estimate within 1c of
# the market), which leaves no independent signal to trade on. Shrinkage
# toward the market happens mechanically in the strategy layer (alpha blend,
# tuned on the train split), not in the prompt.
FORECAST_PROMPT = """You are a superforecaster estimating the probability that a prediction market question resolves YES.

Today's date is {as_of}. Reason ONLY from information available on or before this date. Ignore any knowledge you may have of events after {as_of}.

QUESTION: {question}
{description_block}
WORLD EVENTS (Wikipedia Current Events, dated, before {as_of}):
{events_block}

RECENT NEWS HEADLINES (dated, seen before {as_of}):
{headlines_block}

Guidance:
- Start from the relevant base rate (how often do things like this happen?), then adjust for the specific evidence above and your general knowledge of the world as of {as_of}.
- Be decisive when the evidence is decisive; stay near the base rate when it is not.
- Be calibrated: of all questions where you say 0.80, about 80% should resolve YES.
- evidence_strength measures how much SPECIFIC information you have about THIS question: "none" = base rates only, "weak" = tangential evidence, "moderate" = relevant recent evidence, "strong" = near-decisive evidence.

Respond with ONLY a JSON object, no other text:
{{"p_yes": <float 0.01-0.99>, "evidence_strength": <"none"|"weak"|"moderate"|"strong">, "reasoning": "<2-3 sentences>"}}"""


# Pinned so results are reproducible and the knowledge cutoff (Jan 2026) is
# a known quantity — backtested markets all close after it. Recorded in every
# prediction row.
FORECAST_MODEL = "claude-fable-5"


@dataclass
class Forecast:
    p_model: float
    evidence_strength: str
    reasoning: str
    failed: bool = False  # call/parse failure — must be excluded from trading
    model: str = FORECAST_MODEL
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "p_model": self.p_model,
            "evidence_strength": self.evidence_strength,
            "reasoning": self.reasoning,
            "failed": self.failed,
            "model": self.model,
        }


def build_prompt(
    question: str,
    as_of: str,
    headlines: list[dict],
    world_events: list[dict] | None = None,
    description: str = "",
) -> str:
    if headlines:
        headlines_block = "\n".join(
            f"- [{h.get('seendate', '')[:10]}] {h.get('title', '')} ({h.get('domain', '')})"
            for h in headlines
        )
    else:
        headlines_block = "(no relevant news found)"
    if world_events:
        events_block = "\n".join(
            f"- [{e.get('date', '')}] {e.get('text', '')}" for e in world_events
        )
    else:
        events_block = "(no relevant world events found)"
    description_block = f"MARKET RULES: {description[:500]}\n" if description else ""
    return FORECAST_PROMPT.format(
        as_of=as_of,
        question=question,
        description_block=description_block,
        events_block=events_block,
        headlines_block=headlines_block,
    )


def parse_forecast(text: str) -> Forecast:
    """Parse the model's JSON reply; failures are flagged so the strategy
    layer can exclude them (a defaulted probability would create fake trades)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            p = float(data["p_yes"])
            p = max(0.01, min(0.99, p))
            return Forecast(
                p_model=p,
                evidence_strength=str(data.get("evidence_strength", "none")),
                reasoning=str(data.get("reasoning", ""))[:1000],
                raw_response=text[:2000],
            )
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            pass
    logger.warning("forecaster.parse_failed", text=text[:120])
    return Forecast(
        p_model=0.5,
        evidence_strength="none",
        reasoning="parse failure",
        failed=True,
        raw_response=text[:2000],
    )


async def _call_claude(prompt: str, timeout: int = 120, allow_web: bool = False) -> str:
    # In backtests all tools are disabled so the model cannot search the web
    # at "prediction time" and read post-resolution news — lookahead leakage.
    # In live forward-testing (allow_web=True) web search is legitimate:
    # there is no future to leak. cwd=/tmp and no setting sources keep
    # session CLAUDE.md context out of the forecast; the model is pinned.
    disallowed = "Bash,Read,Glob,Grep,Task,Agent,TodoWrite,Edit,Write"
    if not allow_web:
        disallowed = "WebSearch,WebFetch," + disallowed
    args = ["claude", "-p", prompt, "--model", FORECAST_MODEL,
            "--setting-sources", "", "--disallowedTools", disallowed]
    if allow_web:
        args += ["--allowedTools", "WebSearch,WebFetch"]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/tmp",
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed: {stderr.decode()[:200]}")
    return stdout.decode().strip()


async def forecast(
    question: str,
    as_of: str,
    headlines: list[dict],
    world_events: list[dict] | None = None,
    description: str = "",
    allow_web: bool = False,
    timeout: int = 120,
) -> Forecast:
    """Produce an independent (market-blind) P(YES) for one market as of `as_of`."""
    prompt = build_prompt(question, as_of, headlines, world_events, description)
    if allow_web:
        prompt += "\n\nYou MAY use web search to check the latest relevant news before answering."
    try:
        response = await _call_claude(prompt, timeout=timeout, allow_web=allow_web)
    except Exception as e:
        logger.warning("forecaster.claude_failed", question=question[:60], error=str(e))
        return Forecast(
            p_model=0.5,
            evidence_strength="none",
            reasoning=f"claude call failed: {e}",
            failed=True,
        )
    return parse_forecast(response)
