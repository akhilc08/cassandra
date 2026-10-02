"""Tests for the Wikipedia evidence channel (`oracle.ingestion.wiki_events`).

These exist because the forward test ran six weeks with `n_wiki_events == 0` on
every one of its 38 decisions and nobody could tell from the artifacts whether
that meant "no relevant events" or "every fetch failed". The channel is now
status-aware, retried, cached and circuit-broken; every property below is one
that, when it silently broke, produced that outcome.

No network: every request goes through `httpx.MockTransport`; throttle and
backoff sleeps are zeroed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from oracle.ingestion import wiki_events as we

DAY = date(2026, 9, 20)
REVID = 1375926008
DAY_HTML = (
    "<div><ul>"
    "<li>Ukrainian forces launch a massive drone attack on Moscow Oblast, Russia; "
    "the Moscow Refinery is on fire following several direct hits. (Al Jazeera)</li>"
    "<li>short</li>"
    "<li>The Federal Reserve holds its benchmark rate steady at its September "
    "meeting, citing a cooling labor market. (<a href='#'>Reuters</a>)</li>"
    "</ul></div>"
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(we, "_THROTTLE_SECONDS", 0)
    monkeypatch.setattr(we, "_BACKOFF_SECONDS", 0)


def revisions_payload(revid: int = REVID) -> dict:
    return {"query": {"pages": [{"pageid": 1, "revisions": [{"revid": revid}]}]}}


def missing_page_payload() -> dict:
    return {"query": {"pages": [{"title": "Portal:Current events/x", "missing": True}]}}


def parse_payload(html: str = DAY_HTML) -> dict:
    return {"parse": {"text": html}}


def error_payload(code: str) -> dict:
    return {"error": {"code": code, "info": f"synthetic {code}"}}


def is_revisions(request: httpx.Request) -> bool:
    return request.url.params.get("prop") == "revisions"


class Recorder:
    """MockTransport handler that records every request and delegates to `handler(request, n)`."""

    def __init__(self, handler):
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request, len(self.requests))


def make_client(handler) -> tuple[httpx.AsyncClient, Recorder]:
    rec = Recorder(handler)
    return httpx.AsyncClient(transport=httpx.MockTransport(rec)), rec


def happy_path(request: httpx.Request, n: int) -> httpx.Response:
    if is_revisions(request):
        return httpx.Response(200, json=revisions_payload())
    return httpx.Response(200, json=parse_payload())


def cache_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("*.json"))


class TestFetchDayEventsStatus:
    async def test_403_is_failed_and_not_cached(self, tmp_path):
        client, rec = make_client(lambda r, n: httpx.Response(403, text="UA policy"))
        lines, status = await we.fetch_day_events_status(DAY, client, tmp_path)
        assert (lines, status) == ([], "failed")
        assert cache_files(tmp_path) == []
        assert len(rec.requests) == 1  # a 4xx other than 429 is not retried

    async def test_429_every_attempt_retries_max_attempts_then_fails(self, tmp_path):
        client, rec = make_client(lambda r, n: httpx.Response(429, text="slow down"))
        lines, status = await we.fetch_day_events_status(DAY, client, tmp_path)
        assert (lines, status) == ([], "failed")
        assert len(rec.requests) == we._MAX_ATTEMPTS
        assert cache_files(tmp_path) == []

    async def test_fetched_then_cache_hit_makes_no_request(self, tmp_path):
        client, rec = make_client(happy_path)
        lines, status = await we.fetch_day_events_status(DAY, client, tmp_path)
        assert status == "fetched"
        assert len(lines) == 2  # "short" is dropped by the length filter
        assert "Moscow Refinery" in lines[0]
        assert len(rec.requests) == 2  # one revisions lookup, one parse
        assert [is_revisions(r) for r in rec.requests] == [True, False]
        # The parse call must ask for the PINNED revision, not the current page.
        assert rec.requests[1].url.params.get("oldid") == str(REVID)

        cached = json.loads((tmp_path / f"{DAY.isoformat()}.json").read_text())
        assert cached["revid"] == REVID
        assert cached["lines"] == lines

        n_before = len(rec.requests)
        lines2, status2 = await we.fetch_day_events_status(DAY, client, tmp_path)
        assert (lines2, status2) == (lines, "cache")
        assert len(rec.requests) == n_before

    async def test_maxlag_envelope_is_retried_not_cached_as_absent(self, tmp_path):
        def handler(request, n):
            if n == 1:
                return httpx.Response(200, json=error_payload("maxlag"))
            return happy_path(request, n)

        client, rec = make_client(handler)
        lines, status = await we.fetch_day_events_status(DAY, client, tmp_path)
        assert status == "fetched"
        assert len(lines) == 2
        assert len(rec.requests) == 3
        cached = json.loads((tmp_path / f"{DAY.isoformat()}.json").read_text())
        assert cached["revid"] == REVID  # never 0: an error envelope is not "page absent"

    async def test_non_maxlag_error_envelope_is_failed_and_not_cached(self, tmp_path):
        client, rec = make_client(lambda r, n: httpx.Response(200, json=error_payload("badvalue")))
        lines, status = await we.fetch_day_events_status(DAY, client, tmp_path)
        assert (lines, status) == ([], "failed")
        assert cache_files(tmp_path) == []
        assert len(rec.requests) == 1

    async def test_missing_revisions_is_absent_and_cached_with_revid_zero(self, tmp_path):
        client, rec = make_client(lambda r, n: httpx.Response(200, json=missing_page_payload()))
        lines, status = await we.fetch_day_events_status(DAY, client, tmp_path)
        assert (lines, status) == ([], "absent")
        assert len(rec.requests) == 1  # no parse call for a page that did not exist
        cached = json.loads((tmp_path / f"{DAY.isoformat()}.json").read_text())
        assert cached == {"revid": 0, "lines": []}
        # Absent is a legitimate, cacheable answer: the next call must not re-ask.
        lines2, status2 = await we.fetch_day_events_status(DAY, client, tmp_path)
        assert (lines2, status2) == ([], "cache")
        assert len(rec.requests) == 1

    async def test_user_agent_carries_contact_url_and_maxlag_is_sent(self, tmp_path):
        client, rec = make_client(happy_path)
        await we.fetch_day_events_status(DAY, client, tmp_path)
        assert rec.requests, "no request was issued"
        for request in rec.requests:
            assert "https://github.com/akhilc08/oracle" in request.headers["user-agent"]
            assert request.url.params.get("maxlag") == "5"


class TestFetchDaysBefore:
    CUTOFF = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)

    async def test_circuit_breaker_stops_after_three_consecutive_failures(self, tmp_path):
        client, rec = make_client(lambda r, n: httpx.Response(403, text="UA policy"))
        events, stats = await we.fetch_days_before(self.CUTOFF, client, tmp_path, lookback_days=10)
        assert events == {}
        assert stats["days_ok"] == 0
        assert stats["days_failed"] == 10
        assert stats["days_skipped"] == 7
        assert len(rec.requests) == 3  # one per attempted day; 403 is not retried
        assert cache_files(tmp_path) == []

    async def test_cached_days_feed_relevant_events(self, tmp_path):
        """Design-doc invariant 3: the evidence channel, end to end, on committed cache."""
        last_day = self.CUTOFF.date() - timedelta(days=1)
        for i in range(10):
            d = last_day - timedelta(days=i)
            lines = we.parse_day_page(DAY_HTML) if d == date(2026, 9, 20) else []
            (tmp_path / f"{d.isoformat()}.json").write_text(
                json.dumps({"revid": REVID if lines else 0, "lines": lines})
            )
        client, rec = make_client(lambda r, n: httpx.Response(500, text="must not be called"))

        events, stats = await we.fetch_days_before(self.CUTOFF, client, tmp_path, lookback_days=10)
        assert len(rec.requests) == 0
        assert stats == {"days_ok": 10, "days_cached": 10, "days_fetched": 0,
                         "days_absent": 0, "days_failed": 0, "days_skipped": 0}
        assert list(events) == ["2026-09-20"]

        items = we.relevant_events(
            "Will the Moscow Refinery fire be extinguished by September 30?", events
        )
        assert len(items) > 0
        assert items[0]["date"] == "2026-09-20"
        assert "Moscow Refinery" in items[0]["text"]

    async def test_never_touches_the_cutoff_day(self, tmp_path):
        """Only full days strictly before the cutoff date: the cutoff day's page could
        contain events from later that same day (lookahead)."""
        client, rec = make_client(happy_path)
        await we.fetch_days_before(self.CUTOFF, client, tmp_path, lookback_days=2)
        titles = [r.url.params.get("titles") for r in rec.requests if is_revisions(r)]
        assert titles == [
            "Portal:Current_events/2026_September_24",
            "Portal:Current_events/2026_September_23",
        ]


class TestRateLock:
    def test_lock_survives_multiple_event_loops(self):
        """`asyncio.run()` twice in one process (tests, CLI subcommands) must not
        raise 'bound to a different event loop'. Contend the lock so it actually
        binds to the loop, then do it again on a fresh loop."""

        async def contend():
            async def hold():
                async with we._lock():
                    await asyncio.sleep(0)

            await asyncio.gather(hold(), hold(), hold())
            assert we._lock() is we._lock()
            return we._lock()

        first = asyncio.run(contend())
        second = asyncio.run(contend())
        assert first is not second


class TestCommittedCache:
    CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "forward" / "wiki_cache"

    def test_committed_day_pages_are_well_formed(self):
        """The runner's cache is committed so a cold runner starts warm; a malformed
        file would be silently re-fetched (or, worse, read as empty)."""
        files = cache_files(self.CACHE_DIR)
        if not files:
            pytest.skip("no committed wiki cache in this checkout")
        for f in files:
            date.fromisoformat(f.stem)  # file names are ISO dates
            payload = json.loads(f.read_text())
            assert isinstance(payload, dict) and "revid" in payload and "lines" in payload, f
            assert isinstance(payload["revid"], int), f
            assert isinstance(payload["lines"], list), f
