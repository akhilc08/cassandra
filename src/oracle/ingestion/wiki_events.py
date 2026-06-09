"""Wikipedia Current Events — dated world-event evidence for the backtest.

Day pages are mutable: editors keep adding hindsight (final tallies,
confirmed winners) for days after the fact, so fetching the CURRENT
revision of a past day page leaks post-T information. To close that
channel, each day page D is pinned to its last revision before
D+1 00:00 UTC via the MediaWiki revisions API. Because the backtest only
uses pages dated strictly before prediction time T (D <= T_date - 1),
every pinned revision is guaranteed to have been authored before T.

This is the primary evidence source (GDELT rate limits make news
coverage spotty); pinned pages are cached on disk and shared across
markets.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import structlog

from oracle.ingestion.gdelt_client import _STOPWORDS

logger = structlog.get_logger()

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "cassandra-backtest/1.0 (prediction market research)"

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def day_slug(d: date) -> str:
    return f"{d.year}_{_MONTHS[d.month - 1]}_{d.day}"


def parse_day_page(raw_html: str) -> list[str]:
    """Extract event lines from a Current Events day page."""
    lines = []
    for li in re.findall(r"<li>(.*?)</li>", raw_html, re.DOTALL):
        txt = re.sub(r"<[^>]+>", "", li)
        txt = html_lib.unescape(txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        if 40 < len(txt) < 600:
            lines.append(txt)
    return lines


async def _pinned_revision_id(
    d: date, client: httpx.AsyncClient
) -> int | None:
    """Last revision of day page D saved before D+1 00:00 UTC."""
    rvstart = f"{(d + timedelta(days=1)).isoformat()}T00:00:00Z"
    resp = await client.get(
        WIKI_API,
        params={
            "action": "query",
            "prop": "revisions",
            "titles": f"Portal:Current_events/{day_slug(d)}",
            "rvlimit": "1",
            "rvdir": "older",
            "rvstart": rvstart,
            "format": "json",
            "formatversion": "2",
        },
        headers={"User-Agent": USER_AGENT},
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", [])
    if not pages or "revisions" not in pages[0]:
        return None
    return pages[0]["revisions"][0]["revid"]


async def fetch_day_events(
    d: date,
    client: httpx.AsyncClient,
    cache_dir: Path,
) -> list[str]:
    """Event lines for one day, pinned to the same-day revision, cached on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{d.isoformat()}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if isinstance(cached, dict) and "revid" in cached:
                return cached["lines"]
        except json.JSONDecodeError:
            pass
    try:
        revid = await _pinned_revision_id(d, client)
        if revid is None:
            # Page did not exist by end of day D — no contemporaneous evidence.
            cache_file.write_text(json.dumps({"revid": 0, "lines": []}))
            return []
        resp = await client.get(
            WIKI_API,
            params={
                "action": "parse",
                "oldid": str(revid),
                "prop": "text",
                "format": "json",
                "formatversion": "2",
            },
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        lines = parse_day_page(resp.json().get("parse", {}).get("text", ""))
    except Exception as e:
        logger.warning("wiki_events.fetch_failed", day=d.isoformat(), error=str(e))
        return []
    cache_file.write_text(json.dumps({"revid": revid, "lines": lines}))
    return lines


def question_terms(question: str, max_terms: int = 10) -> list[str]:
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", question)
    cleaned = re.sub(r"[^\w\s$%.\-]", " ", cleaned)
    terms = []
    for tok in cleaned.split():
        low = tok.lower().strip(".-")
        if not low or low in _STOPWORDS or len(low) < 3:
            continue
        if low.isdigit() and len(low) == 4 and low.startswith("20"):
            continue
        terms.append(low)
        if len(terms) >= max_terms:
            break
    return terms


def relevant_events(
    question: str,
    events_by_date: dict[str, list[str]],
    max_items: int = 8,
) -> list[dict]:
    """Pick event lines most relevant to the question by term overlap."""
    terms = question_terms(question)
    if not terms:
        return []
    min_hits = 1 if len(terms) <= 2 else 2
    scored: list[tuple[int, str, str]] = []
    for day, lines in events_by_date.items():
        for line in lines:
            low = line.lower()
            hits = sum(1 for t in terms if t in low)
            if hits >= min_hits:
                scored.append((hits, day, line))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [{"date": day, "text": line[:400]} for _, day, line in scored[:max_items]]


async def fetch_events_before(
    question: str,
    cutoff: datetime,
    client: httpx.AsyncClient,
    cache_dir: Path,
    lookback_days: int = 10,
    max_items: int = 8,
) -> list[dict]:
    """Relevant world events from day pages strictly before the cutoff date."""
    events_by_date: dict[str, list[str]] = {}
    # Only full days before the cutoff date — the cutoff day's page could
    # contain events from later that same day.
    last_day = cutoff.date() - timedelta(days=1)
    for i in range(lookback_days):
        d = last_day - timedelta(days=i)
        lines = await fetch_day_events(d, client, cache_dir)
        if lines:
            events_by_date[d.isoformat()] = lines
    return relevant_events(question, events_by_date, max_items=max_items)
