"""GDELT DOC 2.0 client — historical news with a hard date cutoff.

Free, no API key, full-text news archive back to 2017. Used by the
time-machine backtest so evidence is strictly limited to articles seen
before prediction time T (NewsAPI cannot do this: free tier is last-30-days
and the old code fetched news with no date bound at all — lookahead leakage).

API: https://api.gdeltproject.org/api/v2/doc/doc
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone

import httpx
import structlog

logger = structlog.get_logger()

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT allows roughly one request per ~10s per IP and penalty-boxes bursts;
# serialize and throttle all calls, and back off on 429.
_rate_lock = asyncio.Lock()
_THROTTLE_SECONDS = 10.0
_BACKOFF_SECONDS = 20.0
_MAX_RETRIES = 2

_STOPWORDS = {
    "a", "an", "the", "will", "be", "is", "are", "was", "were", "do", "does",
    "did", "to", "of", "in", "on", "at", "by", "for", "or", "and", "vs",
    "with", "than", "more", "less", "above", "below", "before", "after",
    "win", "happen", "announce", "announces", "reach", "hit", "end", "up",
    "down", "this", "that", "any", "have", "has", "what", "which", "who",
    "between",
}


def build_query(question: str, max_terms: int = 4) -> str:
    """Build a GDELT query from a market question.

    Keeps proper-noun-ish and distinctive terms; GDELT ANDs space-separated
    terms and requires multi-word phrases to be quoted.
    """
    # Keep $74,000 as one token, then strip punctuation except $ % . - inside tokens
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", question)
    cleaned = re.sub(r"[^\w\s$%.\-]", " ", cleaned)
    terms: list[str] = []
    for tok in cleaned.split():
        low = tok.lower().strip(".-")
        if not low or low in _STOPWORDS:
            continue
        if low.isdigit() and len(low) == 4 and low.startswith("20"):
            continue  # bare years match too much
        if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", low):
            continue  # literal dates never appear in headlines
        terms.append(tok.strip(".-"))
        if len(terms) >= max_terms:
            break
    if not terms:
        return f'"{question.strip()[:60]}"'
    return " ".join(f'"{t}"' if " " in t else t for t in terms)


def _fmt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def parse_articles(payload: dict, cutoff: datetime) -> list[dict]:
    """Extract articles, dropping anything GDELT saw at/after the cutoff.

    GDELT's enddatetime filter should already enforce this; we re-check
    seendate defensively because leakage here invalidates the backtest.
    """
    out = []
    for a in payload.get("articles", []):
        seen_raw = a.get("seendate", "")
        try:
            seen = datetime.strptime(seen_raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if seen >= cutoff:
            continue
        out.append({
            "title": a.get("title", ""),
            "seendate": seen.isoformat(),
            "domain": a.get("domain", ""),
            "url": a.get("url", ""),
        })
    return out


async def fetch_news_before(
    question: str,
    cutoff: datetime,
    lookback_days: int = 10,
    max_records: int = 12,
    client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Fetch news articles about `question` seen strictly before `cutoff`.

    Tries the full keyword query first, then relaxes to fewer terms if
    nothing matches.
    """
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    try:
        for max_terms in (4, 2):
            query = build_query(question, max_terms=max_terms)
            params = {
                "query": query,
                "mode": "artlist",
                "maxrecords": str(max_records),
                "startdatetime": _fmt(cutoff - timedelta(days=lookback_days)),
                "enddatetime": _fmt(cutoff - timedelta(minutes=1)),
                "format": "json",
                "sort": "datedesc",
            }
            payload = await _throttled_get(client, params)
            if payload is None:
                return []  # rate-limited out or hard failure — don't burn more slots
            articles = parse_articles(payload, cutoff)
            if articles:
                return articles
            # success but no matches → relax query and try once more
        return []
    finally:
        if own_client:
            await client.aclose()


async def _throttled_get(client: httpx.AsyncClient, params: dict) -> dict | None:
    """One GDELT request under the global throttle, with 429 backoff."""
    async with _rate_lock:
        for attempt in range(_MAX_RETRIES + 1):
            await asyncio.sleep(_THROTTLE_SECONDS)
            try:
                resp = await client.get(GDELT_API, params=params)
                if resp.status_code == 429:
                    logger.warning("gdelt.rate_limited", attempt=attempt)
                    await asyncio.sleep(_BACKOFF_SECONDS)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning("gdelt.fetch_failed", query=str(params.get("query"))[:60], error=str(e))
                return None
    return None
