"""Unit tests for time-machine data plumbing: price-at-T, GDELT cutoff, parsing."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from oracle.agents.forecaster import parse_forecast
from oracle.ingestion.gdelt_client import build_query, parse_articles
from oracle.ingestion.price_history import last_price_at_or_before
from oracle.ingestion.wiki_events import day_slug, parse_day_page, relevant_events

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from backtest import parse_resolution, yes_token_id  # noqa: E402


class TestPriceAt:
    HISTORY = [
        {"t": 1000, "p": 0.40},
        {"t": 2000, "p": 0.45},
        {"t": 3000, "p": 0.55},
    ]

    def test_picks_last_point_at_or_before(self):
        assert last_price_at_or_before(self.HISTORY, 2500) == 0.45
        assert last_price_at_or_before(self.HISTORY, 2000) == 0.45
        assert last_price_at_or_before(self.HISTORY, 99999, tolerance_seconds=10**9) == 0.55

    def test_never_uses_future_price(self):
        assert last_price_at_or_before(self.HISTORY, 1500) == 0.40

    def test_stale_price_rejected(self):
        assert last_price_at_or_before(self.HISTORY, 3000 + 7 * 3600, tolerance_seconds=6 * 3600) is None

    def test_empty_history(self):
        assert last_price_at_or_before([], 1000) is None


class TestGdelt:
    def test_query_drops_stopwords_and_years(self):
        q = build_query("Will Bitcoin reach $150,000 by 2026?")
        assert "Will" not in q.split() and "2026" not in q
        assert "Bitcoin" in q

    def test_query_keeps_comma_numbers_drops_dates(self):
        q = build_query("Will Bitcoin be above $74,000 on 2026-02-07?")
        assert "$74000" in q and "2026-02-07" not in q

    def test_articles_after_cutoff_dropped(self):
        cutoff = datetime(2026, 4, 1, tzinfo=timezone.utc)
        payload = {"articles": [
            {"title": "before", "seendate": "20260331T120000Z", "domain": "a.com", "url": "u1"},
            {"title": "at cutoff", "seendate": "20260401T000000Z", "domain": "b.com", "url": "u2"},
            {"title": "after", "seendate": "20260402T000000Z", "domain": "c.com", "url": "u3"},
        ]}
        articles = parse_articles(payload, cutoff)
        assert [a["title"] for a in articles] == ["before"]

    def test_malformed_seendate_dropped(self):
        cutoff = datetime(2026, 4, 1, tzinfo=timezone.utc)
        payload = {"articles": [{"title": "bad", "seendate": "not-a-date"}]}
        assert parse_articles(payload, cutoff) == []


class TestForecastParse:
    def test_parses_clean_json(self):
        f = parse_forecast('{"p_yes": 0.72, "evidence_strength": "strong", "reasoning": "x"}')
        assert f.p_model == 0.72 and f.evidence_strength == "strong" and not f.failed

    def test_parses_json_with_surrounding_text(self):
        f = parse_forecast('Here you go:\n{"p_yes": 0.3, "evidence_strength": "weak", "reasoning": "y"}\nDone.')
        assert f.p_model == 0.3

    def test_clamps_extreme_probabilities(self):
        assert parse_forecast('{"p_yes": 1.0, "evidence_strength": "strong", "reasoning": ""}').p_model == 0.99
        assert parse_forecast('{"p_yes": 0.0, "evidence_strength": "strong", "reasoning": ""}').p_model == 0.01

    def test_garbage_is_flagged_failed(self):
        f = parse_forecast("I cannot answer that.")
        assert f.failed

    def test_missing_p_yes_is_flagged_failed(self):
        f = parse_forecast('{"evidence_strength": "weak", "reasoning": "no estimate"}')
        assert f.failed


class TestWikiEvents:
    def test_day_slug(self):
        from datetime import date
        assert day_slug(date(2026, 4, 5)) == "2026_April_5"

    def test_parse_day_page_strips_tags(self):
        html = "<ul><li>The <a href='/x'>Israeli</a> cabinet convenes about a ceasefire in Lebanon today.</li><li>short</li></ul>"
        lines = parse_day_page(html)
        assert lines == ["The Israeli cabinet convenes about a ceasefire in Lebanon today."]

    def test_relevant_events_requires_overlap(self):
        events = {
            "2026-04-14": [
                "Israel announces an extension of the ceasefire with Lebanon for two weeks.",
                "A volcano erupts in Iceland disrupting flights.",
            ],
        }
        picked = relevant_events("Israel announces Lebanon ceasefire extension by June 7?", events)
        assert len(picked) == 1
        assert "ceasefire" in picked[0]["text"]

    def test_relevant_events_empty_when_no_match(self):
        events = {"2026-04-14": ["A volcano erupts in Iceland disrupting flights."]}
        assert relevant_events("Will the Lakers win the NBA championship?", events) == []


class TestMarketParsing:
    def test_resolution_yes_first(self):
        m = {"outcomes": '["Yes", "No"]', "outcomePrices": '["1", "0"]'}
        assert parse_resolution(m) is True

    def test_resolution_no_first_order_flipped(self):
        m = {"outcomes": '["No", "Yes"]', "outcomePrices": '["1", "0"]'}
        assert parse_resolution(m) is False  # the NO outcome settled at 1

    def test_unresolved_returns_none(self):
        m = {"outcomes": '["Yes", "No"]', "outcomePrices": '["0.4", "0.6"]'}
        assert parse_resolution(m) is None

    def test_non_binary_returns_none(self):
        m = {"outcomes": '["Trump", "Biden"]', "outcomePrices": '["1", "0"]'}
        assert parse_resolution(m) is None

    def test_yes_token_follows_outcome_order(self):
        m = {"outcomes": '["No", "Yes"]', "clobTokenIds": '["111", "222"]'}
        assert yes_token_id(m) == "222"
