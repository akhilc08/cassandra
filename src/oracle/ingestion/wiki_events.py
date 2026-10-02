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

import asyncio
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
# Wikimedia's User-Agent policy requires a descriptive agent WITH contact
# information. Agents without it are throttled or refused at the edge, and
# cloud-runner IPs (GitHub Actions) are held to it strictly: the forward test
# ran six weeks with zero evidence on every decision under the old
# contact-free string. Keep the repo URL in here.
USER_AGENT = (
    "CassandraForwardTest/2.0 (https://github.com/akhilc08/oracle; "
    "prediction-market research; contact via GitHub issues) python-httpx"
)

# MediaWiki rate-limits anonymous bursts; throttle and back off on 429. The
# lock is (re)created inside the running loop so the module survives more than
# one asyncio.run() per process (tests, CLI subcommands).
_rate_lock: asyncio.Lock | None = None
_rate_lock_loop: asyncio.AbstractEventLoop | None = None
_THROTTLE_SECONDS = 1.0
_BACKOFF_SECONDS = 5.0
_MAX_ATTEMPTS = 3


def _lock() -> asyncio.Lock:
    global _rate_lock, _rate_lock_loop
    loop = asyncio.get_running_loop()
    if _rate_lock is None or _rate_lock_loop is not loop:
        _rate_lock = asyncio.Lock()
        _rate_lock_loop = loop
    return _rate_lock


async def _api_get(client: httpx.AsyncClient, params: dict) -> dict:
    # maxlag: when the database replicas lag, the API answers HTTP 200 with an
    # error envelope instead of piling on; treat it like a 429.
    params = {**params, "maxlag": "5"}
    async with _lock():
        for attempt in range(_MAX_ATTEMPTS):
            await asyncio.sleep(_THROTTLE_SECONDS)
            resp = await client.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 429:
                logger.warning("wiki_events.rate_limited", attempt=attempt)
                await asyncio.sleep(_BACKOFF_SECONDS * (attempt + 1))
                continue
            resp.raise_for_status()
            payload = resp.json()
            err = payload.get("error") if isinstance(payload, dict) else None
            if err:
                # An error envelope must never be mistaken for "page absent":
                # the caller would cache an empty day forever.
                if err.get("code") == "maxlag":
                    logger.warning("wiki_events.maxlag", attempt=attempt)
                    await asyncio.sleep(_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"wikipedia API error: {err.get('code')}: {err.get('info', '')[:120]}"
                )
            return payload
    raise RuntimeError("wikipedia API rate limit persisted after retries")

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
    payload = await _api_get(client, {
        "action": "query",
        "prop": "revisions",
        "titles": f"Portal:Current_events/{day_slug(d)}",
        "rvlimit": "1",
        "rvdir": "older",
        "rvstart": rvstart,
        "format": "json",
        "formatversion": "2",
    })
    pages = payload.get("query", {}).get("pages", [])
    if not pages or "revisions" not in pages[0]:
        return None
    return pages[0]["revisions"][0]["revid"]


async def fetch_day_events_status(
    d: date,
    client: httpx.AsyncClient,
    cache_dir: Path,
) -> tuple[list[str], str]:
    """Event lines for one day, pinned to the same-day revision, cached on disk.

    Returns (lines, status) with status in {"cache", "fetched", "absent",
    "failed"}. A failed day is NOT cached, so the next run retries it; "absent"
    (the page did not exist by end of day D) is cached as empty. Callers that
    need to know whether the evidence channel is alive must look at the status:
    "no relevant events" and "every fetch failed" both yield [] otherwise.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{d.isoformat()}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if isinstance(cached, dict) and "revid" in cached:
                return cached["lines"], "cache"
        except json.JSONDecodeError:
            pass
    try:
        revid = await _pinned_revision_id(d, client)
        if revid is None:
            # Page did not exist by end of day D — no contemporaneous evidence.
            cache_file.write_text(json.dumps({"revid": 0, "lines": []}))
            return [], "absent"
        payload = await _api_get(client, {
            "action": "parse",
            "oldid": str(revid),
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        })
        lines = parse_day_page(payload.get("parse", {}).get("text", ""))
    except httpx.HTTPStatusError as e:
        # The status code and body head are the one thing that tells an operator
        # WHY a runner cannot reach Wikipedia (403 = UA policy, 429 = rate limit).
        logger.warning(
            "wiki_events.fetch_failed", day=d.isoformat(),
            status=e.response.status_code, body=e.response.text[:200],
        )
        return [], "failed"
    except Exception as e:  # noqa: BLE001 — evidence is best-effort; never crash a scan
        logger.warning("wiki_events.fetch_failed", day=d.isoformat(),
                       error=f"{type(e).__name__}: {e}"[:200])
        return [], "failed"
    cache_file.write_text(json.dumps({"revid": revid, "lines": lines}))
    return lines, "fetched"


async def fetch_day_events(
    d: date,
    client: httpx.AsyncClient,
    cache_dir: Path,
) -> list[str]:
    """Event lines for one day (status-blind wrapper kept for the backtest)."""
    lines, _ = await fetch_day_events_status(d, client, cache_dir)
    return lines


async def fetch_days_before(
    cutoff: datetime,
    client: httpx.AsyncClient,
    cache_dir: Path,
    lookback_days: int = 10,
    max_consecutive_failures: int = 3,
) -> tuple[dict[str, list[str]], dict]:
    """Fetch (or load from cache) every full day page before the cutoff date, ONCE.

    The forward runner calls this once per scan and then matches each market's
    question against the shared `events_by_date` with `relevant_events`, instead
    of re-fetching ten day pages per candidate. Returns (events_by_date, stats)
    where stats = {"days_ok", "days_cached", "days_fetched", "days_absent",
    "days_failed", "days_skipped"}. After `max_consecutive_failures` failures in
    a row the remaining days are skipped without a request (circuit breaker:
    a blocked runner must not spend minutes in backoff), and counted as failed.
    """
    events_by_date: dict[str, list[str]] = {}
    stats = {"days_ok": 0, "days_cached": 0, "days_fetched": 0, "days_absent": 0,
             "days_failed": 0, "days_skipped": 0}
    consecutive_failures = 0
    last_day = cutoff.date() - timedelta(days=1)
    for i in range(lookback_days):
        d = last_day - timedelta(days=i)
        if consecutive_failures >= max_consecutive_failures:
            stats["days_skipped"] += 1
            stats["days_failed"] += 1
            continue
        lines, status = await fetch_day_events_status(d, client, cache_dir)
        if status == "failed":
            consecutive_failures += 1
            stats["days_failed"] += 1
            continue
        consecutive_failures = 0
        stats["days_ok"] += 1
        stats[f"days_{status}" if status != "cache" else "days_cached"] += 1
        if lines:
            events_by_date[d.isoformat()] = lines
    return events_by_date, stats


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
