"""Unit tests for the forward paper-trading test — pure logic and INVARIANTS.

These exist because forward-test v1 shipped two silent failures:

1. It re-implemented the decision rule and omitted `max_divergence`, so it ran a
   looser strategy than the backtested one. `TestTheBand` is the guard.
2. It failed silently for seven weeks while the report still looked healthy.
   `TestFailedForecastInvariant` and `TestReportShape` guard that class of lie.

The shadow period added the failures that `TestScanGuards`, `TestScanIsolation`,
`TestSettle` and `TestPrereg` guard: a dead evidence channel nobody noticed, a
scan that lost a day's rows to one bad market, and a health record that did not
exist. No network, no API key: Gamma, the CLOB, Wikipedia and the forecaster are
in-memory fakes keyed on the URL.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import forward_test as ft  # noqa: E402

from oracle.agents.forecaster import Forecast  # noqa: E402
from oracle.agents.forecaster_openai import (  # noqa: E402
    FORECAST_MODEL,
    parse_response,
)
from oracle.evaluation.pnl import BetInput, StrategyParams, decide_trade  # noqa: E402

FROZEN = ft.load_frozen_params()


def bet(p_model: float, p_market: float, outcome: bool = True, market_id: str = "m1"):
    return BetInput(
        market_id=market_id,
        question="q",
        p_model=p_model,
        p_market=p_market,
        outcome=outcome,
        close_time="2026-08-10T00:00:00Z",
        category="other",
        evidence_strength="moderate",
    )


def write_baseline(tmp_path: Path, **overrides) -> Path:
    """A copy of the archived evaluation.json with `frozen_params` mutated."""
    frozen = {
        "alpha": 1.0,
        "threshold": 0.08,
        "slippage": 0.01,
        "min_price": 0.03,
        "max_price": 0.97,
        "evidence_gate": None,
        "max_divergence": 0.35,
        "stake": 100.0,
        "kelly_fraction": 0.25,
        "kelly_cap": 0.05,
    }
    frozen.update(overrides)
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps({"frozen_params": frozen}))
    return path


class TestLoadFrozenParams:
    """The params must come from the archived artifact, not from local constants."""

    def test_returns_the_pre_registered_values(self):
        p = ft.load_frozen_params()
        assert p.alpha == 1.0
        assert p.threshold == 0.08
        assert p.slippage == 0.01
        assert p.min_price == 0.03
        assert p.max_price == 0.97
        assert p.max_divergence == 0.35
        assert p.evidence_gate is None

    def test_ignores_unknown_keys_in_the_artifact(self, tmp_path, monkeypatch):
        path = write_baseline(tmp_path)
        payload = json.loads(path.read_text())
        payload["frozen_params"]["some_future_key"] = 7
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(ft, "BASELINE_EVAL", path)
        assert ft.load_frozen_params().threshold == 0.08

    @pytest.mark.parametrize(
        "key,bad",
        [
            ("alpha", 0.5),
            ("threshold", 0.05),
            ("slippage", 0.02),
            ("min_price", 0.01),
            ("max_price", 0.99),
            ("max_divergence", 1.0),  # the v1 bug, expressed as artifact drift
        ],
    )
    def test_drift_in_any_numeric_param_is_fatal(self, tmp_path, monkeypatch, key, bad):
        monkeypatch.setattr(ft, "BASELINE_EVAL", write_baseline(tmp_path, **{key: bad}))
        with pytest.raises(SystemExit) as exc:
            ft.load_frozen_params()
        assert key in str(exc.value)

    def test_evidence_gate_drift_is_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ft, "BASELINE_EVAL", write_baseline(tmp_path, evidence_gate="moderate"))
        with pytest.raises(SystemExit) as exc:
            ft.load_frozen_params()
        assert "evidence_gate" in str(exc.value)

    def test_tiny_drift_still_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ft, "BASELINE_EVAL", write_baseline(tmp_path, threshold=0.0801))
        with pytest.raises(SystemExit):
            ft.load_frozen_params()

    def test_missing_artifact_is_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ft, "BASELINE_EVAL", tmp_path / "nope.json")
        with pytest.raises(SystemExit) as exc:
            ft.load_frozen_params()
        assert "frozen params unavailable" in str(exc.value)


class TestTheBand:
    """THE most important test in this file.

    The frozen strategy trades iff 0.08 <= |p_model - p_market| <= 0.35 (with
    alpha=1.0 the blend is the model, so edge == p_model - p_market). v1 dropped
    the upper bound and traded everything above 0.08.

    Boundary cases below use pairs whose float difference lands exactly on the
    double nearest 0.08 / 0.35, so the inclusive bounds are genuinely exercised.
    """

    @pytest.mark.parametrize(
        "p_model,p_market,should_trade,why",
        [
            # --- lower bound (threshold = 0.08, comparison is `< threshold`) ---
            (0.579, 0.50, False, "|edge| 0.079 — just under the threshold"),
            (0.13, 0.05, True, "|edge| exactly 0.08 — the threshold is inclusive"),
            (0.421, 0.50, False, "|edge| 0.079 on the NO side"),
            (0.42, 0.50, True, "|edge| 0.08 on the NO side"),
            # --- inside the band ---
            (0.62, 0.50, True, "|edge| 0.12 — comfortably inside"),
            (0.30, 0.50, True, "|edge| 0.20 on the NO side"),
            # --- upper bound (max_divergence = 0.35, comparison is `> max_div`) ---
            (0.85, 0.50, True, "|edge| exactly 0.35 — max_divergence is inclusive"),
            (0.25, 0.60, True, "|edge| exactly 0.35 on the NO side"),
            (0.87, 0.50, False, "|edge| 0.37 — THE v1 BUG: must NOT trade"),
            (0.13, 0.50, False, "|edge| 0.37 on the NO side: must NOT trade"),
            (0.95, 0.40, False, "|edge| 0.55 — wild disagreement, never trade"),
        ],
    )
    def test_band(self, p_model, p_market, should_trade, why):
        t = decide_trade(bet(p_model, p_market), FROZEN)
        assert (t is not None) is should_trade, why

    def test_boundary_pairs_really_sit_on_the_bounds(self):
        # Guards the parametrization above against float drift making the
        # "exactly at the bound" cases quietly land inside the band instead.
        assert abs(0.13 - 0.05) == 0.08
        assert abs(0.42 - 0.50) >= 0.08
        assert abs(0.85 - 0.50) == 0.35
        assert abs(0.25 - 0.60) == 0.35
        assert abs(0.87 - 0.50) > 0.35

    def test_v1_regression_max_divergence_is_what_changes_the_answer(self):
        """The exact bug: identical inputs, one param dropped, opposite decision."""
        v1_like = StrategyParams(
            alpha=FROZEN.alpha,
            threshold=FROZEN.threshold,
            slippage=FROZEN.slippage,
            min_price=FROZEN.min_price,
            max_price=FROZEN.max_price,
        )  # max_divergence left at its 1.0 default — v1's omission
        b = bet(0.90, 0.50)
        assert decide_trade(b, v1_like) is not None, "v1 would have taken this trade"
        assert decide_trade(b, FROZEN) is None, "the frozen strategy must refuse it"

    def test_side_is_yes_when_model_above_market(self):
        t = decide_trade(bet(0.62, 0.50), FROZEN)
        assert t.side == "yes"
        assert t.entry_cost == pytest.approx(0.51)  # p_market + 1c slippage

    def test_side_is_no_when_model_below_market(self):
        t = decide_trade(bet(0.38, 0.50), FROZEN)
        assert t.side == "no"
        assert t.entry_cost == pytest.approx(0.51)  # (1 - p_market) + slippage

    @pytest.mark.parametrize(
        "p_market,should_trade",
        [(0.02, False), (0.03, True), (0.97, True), (0.98, False)],
    )
    def test_price_bounds_are_inclusive(self, p_market, should_trade):
        # edge fixed at +/-0.10 so only the price bound can decide
        p_model = p_market + 0.10 if p_market < 0.5 else p_market - 0.10
        t = decide_trade(bet(p_model, p_market), FROZEN)
        assert (t is not None) is should_trade

    def test_frozen_params_have_no_evidence_gate(self):
        # "none" was 605 of 959 baseline predictions; gating would silently
        # discard most of the traded universe.
        weak = BetInput("m", "q", 0.62, 0.50, True, "2026-08-10", evidence_strength="none")
        assert decide_trade(weak, FROZEN) is not None


class TestFailedForecastInvariant:
    """Why stage_scan must check `.failed` before building a BetInput."""

    def test_sentinel_would_manufacture_a_trade(self):
        # BetInput has no `failed` field. A failed forecast's p_model=0.5 against
        # a market at 0.20 is edge 0.30 — squarely inside the band.
        sentinel = decide_trade(bet(0.5, 0.20), FROZEN)
        assert sentinel is not None
        assert 0.08 <= abs(sentinel.edge) <= 0.35
        # ...which is exactly why the guard in stage_scan is load-bearing.

    def test_failed_forecasts_are_marked_and_neutral(self):
        for text in ["not json", '{"evidence_strength": "weak"}', '{"p_yes": NaN}']:
            f = parse_response(text)
            assert f.failed is True
            assert f.p_model == 0.5


class TestFillPrice:
    """v1 recorded the best ask regardless of size."""

    def test_walks_the_book_instead_of_taking_the_best_ask(self):
        book = {
            "asks": [
                {"price": "0.40", "size": "1"},      # 1 share of bait
                {"price": "0.50", "size": "1000"},   # where the size actually is
            ]
        }
        vwap, filled = ft.fill_price(book, 100.0)
        assert filled == pytest.approx(100.0)
        assert vwap != 0.40
        assert vwap > 0.49, "a 1-share offer must not set the fill for a $100 stake"
        # 1 share @ 0.40 + 199.2 shares @ 0.50 = $100 for 200.2 shares
        assert vwap == pytest.approx(100.0 / 200.2)

    def test_empty_book(self):
        assert ft.fill_price({}, 100.0) == (None, 0.0)
        assert ft.fill_price({"asks": []}, 100.0) == (None, 0.0)

    def test_partial_fill_is_reported_honestly(self):
        book = {"asks": [{"price": "0.50", "size": "10"}]}  # only $5 of depth
        vwap, filled = ft.fill_price(book, 100.0)
        assert vwap == pytest.approx(0.50)
        assert filled == pytest.approx(5.0)
        assert filled < 100.0

    def test_walks_cheapest_first_regardless_of_book_order(self):
        descending = {"asks": [{"price": "0.50", "size": "1000"}, {"price": "0.40", "size": "1"}]}
        ascending = {"asks": [{"price": "0.40", "size": "1"}, {"price": "0.50", "size": "1000"}]}
        assert ft.fill_price(descending, 100.0) == ft.fill_price(ascending, 100.0)

    def test_single_deep_level_fills_at_that_price(self):
        book = {"asks": [{"price": "0.25", "size": "10000"}]}
        vwap, filled = ft.fill_price(book, 100.0)
        assert vwap == pytest.approx(0.25)
        assert filled == pytest.approx(100.0)

    def test_malformed_and_zero_price_levels_are_skipped(self):
        book = {
            "asks": [
                {"price": "0.00", "size": "999"},   # free money is a data error
                {"size": "10"},                     # missing price
                {"price": "abc", "size": "10"},     # unparseable
                {"price": "0.60", "size": "1000"},
            ]
        }
        vwap, filled = ft.fill_price(book, 100.0)
        assert vwap == pytest.approx(0.60)
        assert filled == pytest.approx(100.0)


class TestBestPrices:
    """The CLOB returns asks descending; asks[0] is the WORST price."""

    def test_best_ask_is_the_minimum_not_the_first(self):
        book = {
            "bids": [{"price": "0.50", "size": "10"},
                     {"price": "0.54", "size": "10"},
                     {"price": "0.52", "size": "10"}],
            "asks": [{"price": "0.62", "size": "10"},
                     {"price": "0.60", "size": "10"},
                     {"price": "0.58", "size": "10"}],
        }
        bid, ask = ft.best_prices(book)
        assert ask == 0.58
        assert ask != float(book["asks"][0]["price"])
        assert bid == 0.54
        assert bid != float(book["bids"][0]["price"])
        assert bid < ask, "crossed book would mean we read the sides backwards"

    def test_empty_sides_are_none(self):
        assert ft.best_prices({}) == (None, None)
        assert ft.best_prices({"bids": [], "asks": []}) == (None, None)
        assert ft.best_prices({"bids": [{"price": "0.4", "size": "1"}]}) == (0.4, None)

    def test_levels_without_a_price_are_skipped(self):
        book = {"bids": [{"size": "5"}, {"price": "0.31", "size": "5"}],
                "asks": [{"size": "5"}, {"price": "0.33", "size": "5"}]}
        assert ft.best_prices(book) == (0.31, 0.33)

    def test_mid_from_best_prices_is_the_logged_p_market(self):
        book = {"bids": [{"price": "0.40", "size": "1"}], "asks": [{"price": "0.44", "size": "1"}]}
        bid, ask = ft.best_prices(book)
        assert (bid + ask) / 2 == pytest.approx(0.42)


class TestTokenIds:
    """Located by outcome NAME, never by index."""

    def test_reversed_outcomes_still_map_correctly(self):
        m = {"outcomes": '["No", "Yes"]', "clobTokenIds": '["tok_no", "tok_yes"]'}
        assert ft.token_ids(m) == ("tok_yes", "tok_no")

    def test_conventional_order(self):
        m = {"outcomes": '["Yes", "No"]', "clobTokenIds": '["tok_yes", "tok_no"]'}
        assert ft.token_ids(m) == ("tok_yes", "tok_no")

    def test_case_insensitive(self):
        m = {"outcomes": ["YES", "no"], "clobTokenIds": ["a", "b"]}
        assert ft.token_ids(m) == ("a", "b")

    def test_already_parsed_lists_work_too(self):
        m = {"outcomes": ["No", "Yes"], "clobTokenIds": ["b", "a"]}
        assert ft.token_ids(m) == ("a", "b")

    @pytest.mark.parametrize(
        "market",
        [
            {"outcomes": '["Yes", "No", "Maybe"]', "clobTokenIds": '["a", "b", "c"]'},
            {"outcomes": '["Up", "Down"]', "clobTokenIds": '["a", "b"]'},
            {"outcomes": '["Yes", "No"]'},                       # no token ids
            {"clobTokenIds": '["a", "b"]'},                      # no outcomes
            {"outcomes": '["Yes", "No"]', "clobTokenIds": '["a"]'},  # length mismatch
            {"outcomes": "garbage", "clobTokenIds": "garbage"},
            {},
        ],
    )
    def test_non_binary_or_malformed_returns_none(self, market):
        assert ft.token_ids(market) is None


class TestClusterKey:
    """The backtest's clusters were inert (event_id == market_id), which made the
    CI too narrow. Live, correlated markets must actually collapse."""

    def test_same_gamma_event_collapses(self):
        a = {"id": "1", "events": [{"id": "42", "slug": "some-match"}]}
        b = {"id": "2", "events": [{"id": "42", "slug": "some-match"}]}
        assert ft.cluster_key(a) == ft.cluster_key(b) == "event:42"

    def test_distinct_events_do_not_collide(self):
        a = {"id": "1", "events": [{"id": "42"}]}
        b = {"id": "2", "events": [{"id": "43"}]}
        assert ft.cluster_key(a) != ft.cluster_key(b)

    def test_event_id_wins_over_slug_fallback(self):
        a = {"id": "1", "events": [{"id": "42", "slug": "bitcoin-above-on-august-9"}]}
        assert ft.cluster_key(a) == "event:42"

    def test_daily_btc_ladder_collapses_via_slug_fallback(self):
        # No event id -> normalized slug. The daily strike ladders are one bet.
        a = {"id": "1", "category": "bitcoin-above-on-august-9-et"}
        b = {"id": "2", "category": "bitcoin-above-on-august-10-et"}
        assert ft.cluster_key(a) == ft.cluster_key(b) == "slug:bitcoin-above"

    def test_below_ladder_collapses_and_stays_distinct_from_above(self):
        below_a = {"id": "1", "category": "bitcoin-below-on-august-9-et"}
        below_b = {"id": "2", "category": "bitcoin-below-on-august-10-et"}
        above = {"id": "3", "category": "bitcoin-above-on-august-9-et"}
        assert ft.cluster_key(below_a) == ft.cluster_key(below_b) == "slug:bitcoin-below"
        assert ft.cluster_key(above) != ft.cluster_key(below_a)

    def test_one_match_many_markets_collapses(self):
        base = {"id": "1", "events": [{"slug": "epl-ars-che"}]}
        score = {"id": "2", "events": [{"slug": "epl-ars-che-exact-score"}]}
        more = {"id": "3", "events": [{"slug": "epl-ars-che-more-markets"}]}
        goals = {"id": "4", "events": [{"slug": "epl-ars-che-total-goals-25"}]}
        keys = {ft.cluster_key(m) for m in (base, score, more, goals)}
        assert keys == {"slug:epl-ars-che"}

    def test_different_slugs_do_not_collide(self):
        a = {"id": "1", "category": "fed-rate-september"}
        b = {"id": "2", "category": "us-election-2026"}
        assert ft.cluster_key(a) != ft.cluster_key(b)

    def test_key_is_namespaced_so_event_and_slug_cannot_collide(self):
        ev = {"id": "1", "events": [{"id": "bitcoin"}]}
        sl = {"id": "2", "category": "bitcoin"}
        assert ft.cluster_key(ev) == "event:bitcoin"
        assert ft.cluster_key(sl) == "slug:bitcoin"
        assert ft.cluster_key(ev) != ft.cluster_key(sl)

    def test_ladder_collapses_with_the_strike_embedded_in_the_slug(self):
        # Real Polymarket ladders carry the strike between "above" and "on"
        # (bitcoin-above-125000-on-august-9). An earlier regex only matched the
        # bare `-above-on-` form, so real ladders would NOT have collapsed and
        # the bootstrap CI would have been too narrow again — the exact defect
        # cluster_key exists to prevent.
        bare = {"id": "1", "category": "bitcoin-above-on-august-9"}
        strike_a = {"id": "2", "category": "bitcoin-above-125000-on-august-9"}
        strike_b = {"id": "3", "category": "bitcoin-above-130000-on-august-10"}
        keys = {ft.cluster_key(m) for m in (bare, strike_a, strike_b)}
        assert keys == {"slug:bitcoin-above"}

    def test_uncategorised_markets_do_not_all_collapse_into_one_cluster(self):
        # market_category() returns the literal "other", not "", for a market with
        # no category. Treating that as a real slug merged every such market into
        # a single cluster, corrupting the bootstrap and — via the 2-per-cluster
        # cap in fetch_live_candidates — throttling the scan to 2 markets a day.
        a = {"id": "901"}
        b = {"id": "902"}
        assert ft.cluster_key(a) != ft.cluster_key(b)
        assert ft.cluster_key(a) == "market:901"
        assert "other" not in ft.cluster_key(a)


class TestParseResponse:
    def test_valid_response(self):
        text = json.dumps({"p_yes": 0.73, "evidence_strength": "moderate", "reasoning": "because"})
        f = parse_response(text)
        assert f.failed is False
        assert f.p_model == pytest.approx(0.73)
        assert f.evidence_strength == "moderate"
        assert f.reasoning == "because"
        assert f.model == FORECAST_MODEL
        assert f.raw_response == text

    @pytest.mark.parametrize(
        "p_yes,expected",
        [(1.5, 0.99), (1.0, 0.99), (0.995, 0.99), (-0.2, 0.01), (0.0, 0.01), (0.5, 0.5)],
    )
    def test_probability_is_clamped_to_the_tradable_range(self, p_yes, expected):
        payload = {"p_yes": p_yes, "evidence_strength": "none", "reasoning": ""}
        f = parse_response(json.dumps(payload))
        assert f.failed is False
        assert f.p_model == pytest.approx(expected)
        assert 0.01 <= f.p_model <= 0.99

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_p_yes_fails_rather_than_clamping(self, literal):
        # json.loads happily accepts these; clamping NaN with min/max would have
        # returned 0.99, i.e. a maximally confident forecast out of nothing.
        text = f'{{"p_yes": {literal}, "evidence_strength": "strong", "reasoning": "x"}}'
        assert json.loads(text)["p_yes"] in (float("inf"), float("-inf")) or math.isnan(
            json.loads(text)["p_yes"]
        )
        f = parse_response(text)
        assert f.failed is True
        assert f.p_model == 0.5
        assert f.evidence_strength == "none"

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "not json at all",
            "{'p_yes': 0.6}",                                # single quotes
            '{"p_yes": 0.6',                                 # truncated
            "Here you go: {\"p_yes\": 0.6}",                 # prose wrapper
            '{"evidence_strength": "weak", "reasoning": "x"}',  # missing p_yes
            '{"p_yes": null}',
            '{"p_yes": "high"}',
            '{"p_yes": [0.6]}',
            '["p_yes", 0.6]',                                # right type, wrong shape
        ],
    )
    def test_unusable_responses_fail_closed(self, text):
        f = parse_response(text)
        assert f.failed is True
        assert f.p_model == 0.5

    def test_failure_still_records_the_raw_response(self):
        f = parse_response("garbage from the model")
        assert f.failed is True
        assert f.raw_response == "garbage from the model"

    def test_missing_optional_fields_default_without_failing(self):
        f = parse_response('{"p_yes": 0.4}')
        assert f.failed is False
        assert f.evidence_strength == "none"
        assert f.reasoning == ""

    def test_oversized_fields_are_truncated(self):
        long = "x" * 5000
        payload = {"p_yes": 0.4, "evidence_strength": "weak", "reasoning": long}
        f = parse_response(json.dumps(payload))
        assert len(f.reasoning) == 1000
        assert len(f.raw_response) == 2000


class TestCurrentMode:
    def test_shadow_without_a_preregistration(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ft, "PREREG_FILE", tmp_path / "preregistration.json")
        assert ft.current_mode() == "shadow"

    def test_official_once_preregistered(self, tmp_path, monkeypatch):
        prereg = tmp_path / "preregistration.json"
        prereg.write_text(json.dumps({"start": "2026-08-10", "params": FROZEN.to_dict()}))
        monkeypatch.setattr(ft, "PREREG_FILE", prereg)
        assert ft.current_mode() == "official"


class TestDecisionLog:
    def test_roundtrip_and_malformed_lines_are_skipped(self, tmp_path, monkeypatch):
        path = tmp_path / "decisions.jsonl"
        monkeypatch.setattr(ft, "DECISIONS_FILE", path)
        assert ft._load_decisions() == []

        rows = [
            {"decision_id": "1", "market_id": "1", "status": "open", "ts_decision": "2026-08-08"},
            {"decision_id": "2", "market_id": "2", "status": "no_trade",
             "ts_decision": "2026-08-09"},
        ]
        ft._rewrite_decisions(rows)
        assert ft._load_decisions() == rows

        with path.open("a") as fh:
            fh.write("{ truncated write from a killed process\n")
        assert ft._load_decisions() == rows  # the bad line is dropped, not fatal

    def test_rewrite_is_atomic_and_leaves_no_temp_files(self, tmp_path, monkeypatch):
        path = tmp_path / "decisions.jsonl"
        monkeypatch.setattr(ft, "DECISIONS_FILE", path)
        ft._rewrite_decisions([{"market_id": "1", "status": "settled"}])
        assert [p.name for p in tmp_path.iterdir()] == ["decisions.jsonl"]


# --- fakes for the stages ------------------------------------------------------
# The stages talk to Gamma, the CLOB, Wikipedia and OpenAI. Each is replaced by
# an in-memory stand-in keyed on the URL; nothing below opens a socket.

NOW = datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def market(mid, *, cluster=None, volume=50_000.0, hours=20.0, age_days=10.0) -> dict:
    cluster = cluster or f"ev{mid}"
    return {
        "id": str(mid),
        "question": f"Will candidate {mid} win the special election on 2026-10-02?",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": json.dumps([f"yes{mid}", f"no{mid}"]),
        "acceptingOrders": True,
        "volumeNum": volume,
        "endDate": iso(NOW + timedelta(hours=hours)),
        "createdAt": iso(NOW - timedelta(days=age_days)),
        "events": [{"id": cluster, "slug": f"event-{cluster}"}],
    }


def two_sided(bid="0.40", ask="0.44", size="1000") -> dict:
    return {"bids": [{"price": bid, "size": size}], "asks": [{"price": ask, "size": size}]}


def good_forecast(p_model=0.62, **overrides) -> Forecast:
    fields = {
        "p_model": p_model, "evidence_strength": "moderate", "reasoning": "r",
        "raw_response": "{}", "served_model": "gpt-5.6-luna-2026-08-01",
        "output_tokens": 60, "reasoning_tokens": 0,
    }
    fields.update(overrides)
    return Forecast(**fields)


def failed_forecast() -> Forecast:
    return Forecast(p_model=0.5, evidence_strength="none", reasoning="", failed=True)


class FakeResponse:
    def __init__(self, body, status_code=200):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body) if body is not None else "<not json>"

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeClient:
    """httpx.AsyncClient stand-in: every GET goes through `route(url, params)`."""

    def __init__(self, route, calls):
        self._route = route
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        return self._route(url, dict(params or {}))


def patch_data_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ft, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ft, "DECISIONS_FILE", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(ft, "SCANS_FILE", tmp_path / "scans.jsonl")
    monkeypatch.setattr(ft, "SUMMARY_FILE", tmp_path / "summary.json")
    monkeypatch.setattr(ft, "PREREG_FILE", tmp_path / "preregistration.json")
    monkeypatch.setattr(ft, "WIKI_CACHE", tmp_path / "wiki_cache")


class ScanWorld:
    """Everything a scan touches, faked, with knobs for each failure mode."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.calls: list[tuple[str, dict]] = []
        self.markets: list[dict] = []
        self.books: dict[str, FakeResponse] = {}     # token -> response; default two-sided
        self.gamma: dict[int, FakeResponse] = {}     # offset -> response; default markets/[]
        self.wiki = {"days_ok": 10, "days_cached": 9, "days_fetched": 1, "days_absent": 0,
                     "days_failed": 0, "days_skipped": 0}
        self.events = {iso(NOW)[:10]: ["Candidate 1 wins the special election after a recount."]}
        self.forecaster = lambda question: good_forecast()
        self.forecast_calls: list[dict] = []
        self.news = lambda question: ([], "empty")
        self.news_calls: list[str] = []
        patch_data_dir(tmp_path, monkeypatch)
        monkeypatch.setattr(ft, "_GAMMA_BACKOFF_SECONDS", 0.0)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr(ft.httpx, "AsyncClient",
                            lambda **kw: FakeClient(self.route, self.calls))
        monkeypatch.setattr(ft, "forecast", self._forecast)
        monkeypatch.setattr(ft, "fetch_days_before", self._fetch_days_before)
        monkeypatch.setattr(ft, "fetch_news_status", self._fetch_news_status)

    def route(self, url, params):
        if url.endswith("/markets"):
            offset = int(params.get("offset", 0))
            if offset in self.gamma:
                return self.gamma[offset]
            return FakeResponse(self.markets if offset == 0 else [])
        if url.endswith("/book"):
            return self.books.get(params["token_id"]) or FakeResponse(two_sided())
        raise AssertionError(f"unexpected GET {url}")

    async def _forecast(self, question, as_of, headlines, world_events=None, description=""):
        self.forecast_calls.append({"question": question, "world_events": world_events,
                                    "headlines": headlines})
        return self.forecaster(question)

    async def _fetch_news_status(self, question, cutoff, lookback_days=10, client=None):
        self.news_calls.append(question)
        return self.news(question)

    async def _fetch_days_before(self, cutoff, client, cache_dir, lookback_days=10):
        return self.events, dict(self.wiki)

    def scan(self, *argv) -> int:
        return ft.run_scan(ft.build_parser().parse_args(["scan", *argv]))

    def decisions(self) -> list[dict]:
        return ft._load_decisions()

    def last_scan(self) -> dict:
        return ft._load_jsonl(ft.SCANS_FILE)[-1]

    def gamma_calls(self) -> list:
        return [c for c in self.calls if c[0].endswith("/markets")]


@pytest.fixture
def world(tmp_path, monkeypatch) -> ScanWorld:
    return ScanWorld(tmp_path, monkeypatch)


class TestScanGuards:
    """Every way a scan can be unhealthy must be named in scans.jsonl and in the exit code."""

    def test_one_sided_book_is_skipped_not_an_outage(self, world):
        world.markets = [market(1)]
        world.books["yes1"] = FakeResponse({"bids": [{"price": "0.40", "size": "10"}], "asks": []})
        assert world.scan() == 0
        row = world.last_scan()
        assert row["status"] == "ok" and row["exit_code"] == 0
        assert row["n_illiquid"] == 1 and row["n_forecast_attempted"] == 0
        assert world.decisions() == []

    def test_all_forecasts_failed_is_an_outage(self, world):
        world.markets = [market(1), market(2)]
        world.forecaster = lambda q: failed_forecast()
        assert world.scan() == 1
        row = world.last_scan()
        assert row["status"] == "forecast_outage"
        assert row["n_forecast_attempted"] == 2 and row["n_forecast_ok"] == 0
        assert {d["status"] for d in world.decisions()} == {"forecast_failed"}

    def test_missing_api_key_exits_2_before_any_http(self, world, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY")
        world.markets = [market(1)]
        assert world.scan() == 2
        assert world.last_scan()["status"] == "no_api_key"
        assert world.calls == []

    def test_gamma_down_on_page_0_is_unavailable_not_empty(self, world):
        world.gamma[0] = FakeResponse({"error": "upstream"}, 500)
        assert world.scan() == 1
        assert world.last_scan()["status"] == "gamma_unavailable"
        assert len(world.gamma_calls()) == 3

    def test_gamma_down_on_a_later_page_is_truncation(self, world):
        world.markets = [market(1)]
        world.gamma[100] = FakeResponse(None, 502)
        assert world.scan() == 0
        row = world.last_scan()
        assert row["gamma_truncated"] is True and row["gamma_pages"] == 1
        assert row["status"] == "ok"

    def test_error_json_books_are_a_clob_outage_not_illiquidity(self, world):
        world.markets = [market(1), market(2)]
        for tok in ("yes1", "no1", "yes2", "no2"):
            world.books[tok] = FakeResponse({"error": "No orderbook exists"}, 404)
        assert world.scan() == 1
        row = world.last_scan()
        assert row["status"] == "clob_outage"
        assert row["n_book_failed"] == 2 and row["n_illiquid"] == 0

    def test_dead_evidence_channel_degrades_the_run_but_still_logs(self, world):
        world.markets = [market(1)]
        world.wiki.update(days_ok=0, days_cached=0, days_fetched=0, days_failed=10)
        world.events = {}
        assert world.scan() == 1
        assert world.last_scan()["status"] == "evidence_degraded"
        (row,) = world.decisions()
        assert row["evidence_ok"] is False and row["wiki_days_ok"] == 0
        assert row["status"] == "open"

    def test_healthy_run_logs_the_new_fields(self, world):
        world.markets = [market(1)]
        assert world.scan() == 0
        (row,) = world.decisions()
        assert row["status"] == "open" and row["side"] == "yes"
        assert row["evidence_ok"] is True and row["wiki_days_ok"] == 10
        assert row["n_wiki_events"] == 1
        assert row["served_model"] == "gpt-5.6-luna-2026-08-01"
        assert row["usage"] == {"output_tokens": 60, "reasoning_tokens": 0}
        assert row["scan_ts"] == row["ts_decision"]
        assert row["hours_to_close"] == pytest.approx(20.0, abs=0.05)
        assert row["params"]["scan_min_volume"] == 20_000.0
        assert row["params"]["primary_min_final_volume"] == 500_000.0
        assert row["params"]["window_hours"] == [12.0, 27.0]
        assert row["params"]["max_divergence"] == 0.35
        scan = world.last_scan()
        assert scan["n_open"] == 1 and scan["n_forecast_ok"] == 1 and scan["error"] is None

    def test_coverage_gap_is_measured_against_the_previous_run(self, world):
        args = ft.build_parser().parse_args(["scan"])
        previous = ft.new_scan_record(NOW - timedelta(hours=20), args)
        previous.update(status="ok", exit_code=0, duration_s=1.0)
        ft._append_jsonl(ft.SCANS_FILE, previous)
        world.scan()
        # previous window ended at -20h + 27h = +7h; this one starts at +12h.
        assert world.last_scan()["coverage_gap_hours"] == pytest.approx(5.0, abs=0.01)

    def test_a_crash_is_recorded_before_it_propagates(self, world, monkeypatch):
        async def boom(*a, **kw):
            raise RuntimeError("wikipedia client exploded")

        world.markets = [market(1)]
        monkeypatch.setattr(ft, "fetch_days_before", boom)
        with pytest.raises(RuntimeError):
            world.scan()
        row = world.last_scan()
        assert row["status"] == "crashed" and row["exit_code"] == 1
        assert "exploded" in row["error"]


class TestScanIsolation:
    """One bad market must cost exactly one market, and rows hit the disk as they are made."""

    def test_a_malformed_book_does_not_take_the_other_markets_down(self, world):
        world.markets = [market(1), market(2), market(3)]
        world.books["yes2"] = FakeResponse(["not", "a", "book"])
        world.books["yes3"] = FakeResponse({
            "bids": [{"price": "NaN", "size": "5"}, {"price": "0.40", "size": "100"}],
            "asks": [{"price": "Infinity", "size": "5"}, {"price": "0.44", "size": "100"}],
        })
        assert world.scan() == 0
        rows = world.decisions()
        assert sorted(r["market_id"] for r in rows) == ["1", "3"]
        assert world.last_scan()["n_market_failed"] == 1
        text = ft.DECISIONS_FILE.read_text()
        assert "NaN" not in text and "Infinity" not in text
        assert rows[1]["book"] == {"yes_bid": 0.40, "yes_ask": 0.44, "no_bid": 0.40, "no_ask": 0.44}

    def test_a_forecaster_bug_is_isolated_and_earlier_rows_survive(self, world):
        def forecaster(question):
            if "candidate 2" in question:
                raise RuntimeError("bug in the forecaster")
            return good_forecast()

        world.markets = [market(1), market(2), market(3)]
        world.forecaster = forecaster
        assert world.scan() == 0
        assert sorted(r["market_id"] for r in world.decisions()) == ["1", "3"]
        row = world.last_scan()
        assert row["n_market_failed"] == 1 and row["n_forecast_attempted"] == 3
        assert row["n_forecast_ok"] == 2


class TestNewsChannel:
    """--news gdelt feeds headlines to the forecaster and says on every row whether it did."""

    HEADLINE = {"title": "Candidate leads final poll", "seendate": "2026-10-01T00:00:00+00:00",
                "domain": "a.com", "url": "u"}

    def test_off_by_default_reproduces_the_replica(self, world):
        world.markets = [market(1)]
        assert world.scan() == 0
        assert world.news_calls == []
        assert world.forecast_calls[0]["headlines"] == []
        (row,) = world.decisions()
        assert row["news_status"] == "off" and row["n_headlines"] == 0
        assert row["params"]["news"] == "off"
        assert world.last_scan()["news"] == "off"

    def test_gdelt_headlines_reach_the_forecaster(self, world):
        world.markets = [market(1), market(2)]
        world.news = lambda q: ([self.HEADLINE], "ok") if "candidate 1" in q else ([], "empty")
        assert world.scan("--news", "gdelt") == 0
        by_q = {c["question"]: c["headlines"] for c in world.forecast_calls}
        assert by_q[market(1)["question"]] == [self.HEADLINE]
        assert by_q[market(2)["question"]] == []
        rows = {r["market_id"]: r for r in world.decisions()}
        assert rows["1"]["news_status"] == "ok" and rows["1"]["n_headlines"] == 1
        assert rows["2"]["news_status"] == "empty" and rows["2"]["n_headlines"] == 0
        scan = world.last_scan()
        assert scan["news"] == "gdelt" and scan["n_news_ok"] == 1 and scan["n_news_empty"] == 1

    def test_a_dead_news_channel_degrades_the_run_but_still_logs(self, world):
        world.markets = [market(1), market(2)]
        world.news = lambda q: ([], "failed")
        assert world.scan("--news", "gdelt") == 1
        scan = world.last_scan()
        assert scan["status"] == "news_degraded" and scan["n_news_failed"] == 2
        assert {r["news_status"] for r in world.decisions()} == {"failed"}
        assert len(world.decisions()) == 2

    def test_past_the_budget_markets_forecast_without_news(self, world, monkeypatch):
        monkeypatch.setattr(ft, "NEWS_BUDGET_SECONDS", 0.0)
        world.markets = [market(1)]
        world.news = lambda q: ([self.HEADLINE], "ok")
        assert world.scan("--news", "gdelt") == 0
        assert world.news_calls == []
        (row,) = world.decisions()
        assert row["news_status"] == "budget" and row["status"] == "open"
        assert world.last_scan()["n_news_budget"] == 1


class TestTradesPerScanCap:
    """One scan cannot open more than --max-trades-per-scan trades, whatever the clusters."""

    def test_trades_beyond_the_cap_are_logged_as_capped_no_trades(self, world):
        world.markets = [market(i) for i in range(1, 9)]
        assert world.scan() == 0
        rows = world.decisions()
        opened = [r for r in rows if r["status"] == "open"]
        capped = [r for r in rows if r.get("cap_skipped")]
        assert len(opened) == ft.MAX_TRADES_PER_SCAN == 6
        assert len(capped) == 2
        assert all(r["status"] == "no_trade" and r["side"] is None and r["stake"] == 0.0
                   and r["capped_side"] == "yes" for r in capped)
        scan = world.last_scan()
        assert scan["n_open"] == 6 and scan["n_capped"] == 2 and scan["n_no_trade"] == 2

    def test_the_cap_is_a_flag(self, world):
        world.markets = [market(i) for i in range(1, 4)]
        assert world.scan("--max-trades-per-scan", "1") == 0
        assert sum(r["status"] == "open" for r in world.decisions()) == 1
        assert world.decisions()[0]["params"]["max_trades_per_scan"] == 1

    def test_no_trade_decisions_do_not_use_up_the_cap(self, world):
        world.markets = [market(i) for i in range(1, 4)]
        world.forecaster = lambda q: good_forecast(0.43) if "candidate 1" in q else good_forecast()
        assert world.scan("--max-trades-per-scan", "2") == 0
        assert sum(r["status"] == "open" for r in world.decisions()) == 2
        assert world.last_scan()["n_capped"] == 0


class TestAlreadyAndClusters:
    """Dedupe by market, retry only forecast failures, cap concurrent exposure per cluster."""

    def candidates(self, markets, monkeypatch, tmp_path, logged=()):
        patch_data_dir(tmp_path, monkeypatch)
        for d in logged:
            ft._append_jsonl(ft.DECISIONS_FILE, d)
        def route(url, p):
            return FakeResponse(markets if p.get("offset") == "0" else [])

        client = FakeClient(route, [])
        args = ft.build_parser().parse_args(["scan"])
        out, meta = asyncio.run(ft.fetch_live_candidates(client, args, NOW))
        return [m["id"] for m in out], meta

    @staticmethod
    def logged(mid, status, cluster="event:42", hours=10.0) -> dict:
        return {"market_id": str(mid), "status": status, "cluster_key": cluster,
                "end_date": iso(NOW + timedelta(hours=hours))}

    def test_only_forecast_failed_rows_are_offered_again(self, monkeypatch, tmp_path):
        logged = [self.logged(1, "forecast_failed", "event:1"),
                  self.logged(2, "no_trade", "event:2"),
                  self.logged(3, "open", "event:3"),
                  self.logged(4, "settled", "event:4", hours=-5)]
        ids, meta = self.candidates([market(i) for i in range(1, 6)], monkeypatch, tmp_path, logged)
        assert ids == ["1", "5"]
        assert meta["n_raw"] == 5
        assert meta["gamma_pages"] == 2, "the empty terminating page was fetched too"

    def test_two_open_markets_in_a_cluster_block_a_third(self, monkeypatch, tmp_path):
        logged = [self.logged(10, "open"), self.logged(11, "no_trade")]
        raw = [market(7, cluster="42"), market(8, cluster="43")]
        ids, _ = self.candidates(raw, monkeypatch, tmp_path, logged)
        assert ids == ["8"]

    def test_closed_markets_free_their_cluster_slots(self, monkeypatch, tmp_path):
        logged = [self.logged(10, "open", hours=-10), self.logged(11, "no_trade", hours=-10)]
        raw = [market(7, cluster="42"), market(8, cluster="43")]
        ids, _ = self.candidates(raw, monkeypatch, tmp_path, logged)
        assert ids == ["7", "8"]

    def test_cap_applies_within_one_scan_too(self, monkeypatch, tmp_path):
        raw = [market(i, cluster="42") for i in range(1, 5)]
        ids, _ = self.candidates(raw, monkeypatch, tmp_path)
        assert ids == ["1", "2"]

    def test_window_and_floor_filters(self, monkeypatch, tmp_path):
        raw = [market(1, hours=20), market(2, hours=5), market(3, hours=30),
               market(4, volume=10_000.0), market(5, age_days=1.0),
               {**market(6), "endDate": "garbage"}]
        ids, _ = self.candidates(raw, monkeypatch, tmp_path)
        assert ids == ["1"]


# --- settle -------------------------------------------------------------------


def open_row(mid="1", *, side="yes", entry_cost=0.51, executable_cost=0.50,
             executable_notional=100.0, **extra) -> dict:
    return {
        "decision_id": mid, "market_id": mid, "mode": "shadow", "status": "open",
        "side": side, "stake": 100.0, "entry_cost": entry_cost,
        "executable_cost": executable_cost, "executable_notional": executable_notional,
        "end_date": iso(NOW - timedelta(hours=5)), "question": "q", "p_model": 0.62,
        "p_market_mid": 0.42, "cluster_key": "event:1", **extra,
    }


def gamma_market(outcome, *, volume=900_000.0, closed=True, **extra) -> dict:
    prices = {True: ["1", "0"], False: ["0", "1"], None: ["0.5", "0.5"]}[outcome]
    return {
        "id": "1", "closed": closed, "outcomes": '["Yes", "No"]',
        "outcomePrices": json.dumps(prices), "volumeNum": volume,
        "umaResolutionStatus": "resolved" if outcome is not None else "disputed", **extra,
    }


class SettleWorld:
    def __init__(self, tmp_path, monkeypatch):
        self.calls: list = []
        self.markets: dict[str, FakeResponse] = {}
        patch_data_dir(tmp_path, monkeypatch)
        monkeypatch.setattr(ft.httpx, "AsyncClient",
                            lambda **kw: FakeClient(self.route, self.calls))

    def route(self, url, params):
        mid = url.rsplit("/", 1)[-1]
        return self.markets.get(mid) or FakeResponse({"error": "not found"}, 404)

    def settle(self, rows: list[dict]) -> list[dict]:
        ft._rewrite_decisions(rows)
        asyncio.run(ft.stage_settle(None))
        return ft._load_decisions()


@pytest.fixture
def settle_world(tmp_path, monkeypatch) -> SettleWorld:
    return SettleWorld(tmp_path, monkeypatch)


class TestSettle:
    def test_full_fill_win(self, settle_world):
        settle_world.markets["1"] = FakeResponse(gamma_market(True))
        (d,) = settle_world.settle([open_row()])
        assert d["status"] == "settled" and d["won"] is True and d["outcome_yes"] is True
        assert d["pnl"] == pytest.approx(100 * 0.49 / 0.51)
        assert d["pnl_executable"] == pytest.approx(100.0)
        assert d["executable_fill_ratio"] == 1.0
        assert d["volume_final"] == 900_000.0
        assert d["ts_resolved"] and d["ts_settled"]

    @pytest.mark.parametrize("outcome,expected", [(True, 30.0), (False, -20.0)])
    def test_partial_fill_executable_pnl_is_on_the_filled_notional(
        self, settle_world, outcome, expected
    ):
        vwap, filled = ft.fill_price({"asks": [{"price": "0.40", "size": "50"}]}, 100.0)
        assert (vwap, filled) == (pytest.approx(0.40), pytest.approx(20.0))
        settle_world.markets["1"] = FakeResponse(gamma_market(outcome))
        (d,) = settle_world.settle([open_row(executable_cost=vwap, executable_notional=filled)])
        assert d["pnl_executable"] == pytest.approx(expected)
        assert d["executable_fill_ratio"] == pytest.approx(0.20)
        assert d["pnl"] == pytest.approx(100 * 0.49 / 0.51 if outcome else -100.0)

    def test_no_side_loses_when_yes_resolves(self, settle_world):
        settle_world.markets["1"] = FakeResponse(gamma_market(True))
        (d,) = settle_world.settle([open_row(side="no")])
        assert d["won"] is False and d["pnl"] == -100.0

    def test_void_shape_and_counters(self, settle_world, capsys):
        settle_world.markets["1"] = FakeResponse(gamma_market(None))
        (d,) = settle_world.settle([open_row()])
        assert d["status"] == "void" and d["outcome_yes"] is None and d["won"] is None
        assert d["pnl"] == 0.0 and d["pnl_per_dollar"] == 0.0 and d["ts_settled"]
        assert d["void_reason"] == {"outcomePrices": '["0.5", "0.5"]',
                                    "umaResolutionStatus": "disputed"}
        assert "settled 0, voided 1 of 1 open; resolved 0 non-trade rows" in capsys.readouterr().out

    def test_no_trade_rows_gain_an_outcome_but_keep_their_status(self, settle_world, capsys):
        settle_world.markets["1"] = FakeResponse(gamma_market(False))
        past = open_row(status="no_trade", side=None, stake=0.0)
        future = open_row("2", status="no_trade", side=None, stake=0.0,
                          end_date=iso(NOW + timedelta(hours=3)))
        rows = settle_world.settle([past, future])
        assert rows[0]["status"] == "no_trade" and rows[0]["outcome_yes"] is False
        assert rows[0]["volume_final"] == 900_000.0 and "ts_resolved" in rows[0]
        assert "pnl" not in rows[0]
        assert "outcome_yes" not in rows[1], "a market that has not closed is not fetched"
        assert [c[0].rsplit("/", 1)[-1] for c in settle_world.calls] == ["1"]
        assert "resolved 1 non-trade rows" in capsys.readouterr().out

    def test_resolved_rows_are_not_fetched_twice(self, settle_world):
        done = open_row(status="no_trade", side=None, stake=0.0, outcome_yes=None)
        settle_world.settle([done])
        assert settle_world.calls == []

    @pytest.mark.parametrize("response", [
        FakeResponse({"error": "unprocessable"}, 422),
        FakeResponse({"error": "no closed key here"}),
        FakeResponse(None, 200),
        FakeResponse(gamma_market(True, closed=False)),
    ])
    def test_unusable_or_unclosed_market_leaves_the_row_open(self, settle_world, response):
        settle_world.markets["1"] = response
        (d,) = settle_world.settle([open_row()])
        assert d["status"] == "open" and "outcome_yes" not in d


# --- report -------------------------------------------------------------------


def settled_row(i, *, p_model=0.70, outcome=None, evidence_ok=True, volume=600_000.0,
                volume_final=600_000.0, mode="shadow", **extra) -> dict:
    return {
        "status": "settled", "mode": mode, "market_id": f"m{i}", "question": "q",
        "p_model": p_model, "p_market_mid": 0.50,
        "outcome_yes": (i % 2 == 0) if outcome is None else outcome,
        "end_date": f"2026-08-{i + 1:02d}", "category": "other",
        "evidence_strength": "moderate", "cluster_key": f"event:{i}",
        "evidence_ok": evidence_ok, "volume": volume, "volume_final": volume_final, **extra,
    }


def scan_row(**overrides) -> dict:
    row = ft.new_scan_record(NOW - timedelta(hours=2), ft.build_parser().parse_args(["scan"]))
    row.update(status="ok", exit_code=0, n_candidates=3, n_forecast_attempted=3,
               n_forecast_ok=3, wiki_days_ok=10, wiki_days_cached=9, duration_s=12.0)
    row.update(overrides)
    return row


FULL_REPORT_KEYS = {
    "population", "n_settled", "n_trades", "total_pnl", "roi", "win_rate",
    "roi_ci_low", "roi_ci_high", "brier_blend_traded", "brier_market_traded",
    "max_drawdown", "n_clusters", "coin_baseline",
}


class TestReportShape:
    """summary.json is what the dashboard reads; its shape is a contract."""

    def test_no_settled_rows_reports_zeros_not_a_healthy_looking_table(self):
        decisions = [
            {"status": "open", "mode": "shadow", "market_id": "1"},
            {"status": "no_trade", "mode": "shadow", "market_id": "2"},
            {"status": "forecast_failed", "mode": "shadow", "market_id": "3"},
        ]
        for population in ft.POPULATIONS:
            assert ft._report(decisions, "shadow", population) == {
                "population": population, "n_settled": 0, "n_trades": 0,
                "coin_baseline": {"n_markets": 0, "n_trades": 0},
            }

    def test_coin_baseline_trades_every_resolved_market_at_half(self):
        # Market at 0.30, constant 0.50 -> edge +0.20 -> buys YES, whatever the model did.
        rows = [
            {**settled_row(0, outcome=True), "p_market_mid": 0.30},
            {**settled_row(1, outcome=False), "status": "no_trade", "p_market_mid": 0.30},
            {**settled_row(2), "status": "no_trade", "p_market_mid": 0.50},   # no edge
            {**settled_row(3), "status": "open", "p_market_mid": 0.30},       # unresolved
            {**settled_row(4, outcome=True), "status": "no_trade", "p_market_mid": 0.30,
             "volume_final": 100_000.0},                                     # not primary
        ]
        coin = ft._report(rows, "shadow", "primary")["coin_baseline"]
        assert coin["n_markets"] == 3 and coin["n_trades"] == 2
        win = (1 - 0.31) / 0.31
        assert coin["total_pnl"] == pytest.approx(100 * (win - 1))
        assert ft._report(rows, "shadow", "all")["coin_baseline"]["n_trades"] == 3

    def test_settled_rows_produce_the_full_metric_block(self):
        rep = ft._report([settled_row(i) for i in range(4)], "shadow", "all")
        assert set(rep) == FULL_REPORT_KEYS
        assert rep["n_settled"] == 4
        assert rep["n_trades"] == 4  # edge 0.20 is inside the band
        assert rep["n_clusters"] == 4
        assert rep["win_rate"] == 0.5
        assert "brier_blend" not in rep and "brier_market" not in rep

    def test_primary_requires_evidence_and_final_volume(self):
        decisions = [
            settled_row(0),                                   # in every population
            settled_row(1, evidence_ok=False),                # channel was dead
            settled_row(2, volume_final=100_000.0),           # too small at resolution
            settled_row(3, volume=20_000.0),                  # small at decision, big at close
        ]
        assert ft._report(decisions, "shadow", "primary")["n_settled"] == 2
        assert ft._report(decisions, "shadow", "secondary_at_decision")["n_settled"] == 3
        assert ft._report(decisions, "shadow", "all")["n_settled"] == 4

    def test_legacy_rows_without_the_new_fields_are_not_primary(self):
        legacy = settled_row(0)
        del legacy["evidence_ok"], legacy["volume_final"]
        assert ft._report([legacy], "shadow", "primary")["n_settled"] == 0
        assert ft._report([legacy], "shadow", "all")["n_settled"] == 1

    def test_modes_are_kept_separate(self):
        decisions = [settled_row(0), settled_row(1, mode="official")]
        assert ft._report(decisions, "shadow", "all")["n_settled"] == 1
        assert ft._report(decisions, "official", "all")["n_settled"] == 1

    def test_only_settled_rows_count_toward_pnl(self):
        still_open = {**settled_row(1), "status": "open"}
        assert ft._report([settled_row(0), still_open], "shadow", "all")["n_settled"] == 1

    def test_report_uses_the_frozen_params_not_a_local_copy(self):
        # A settled row 0.45 away from the market is outside max_divergence, so a
        # correctly-frozen report counts it as settled but never as a trade.
        rep = ft._report([settled_row(0, p_model=0.95)], "shadow", "all")
        assert rep["n_settled"] == 1
        assert rep["n_trades"] == 0

    def test_calibration_covers_every_resolved_row_not_just_trades(self):
        decisions = [
            {"mode": "shadow", "status": "settled", "p_model": 0.8, "p_market_mid": 0.5,
             "outcome_yes": True},
            {"mode": "shadow", "status": "no_trade", "p_model": 0.6, "p_market_mid": 0.99,
             "outcome_yes": True},
            {"mode": "shadow", "status": "forecast_failed", "p_model": 0.5, "p_market_mid": 0.5,
             "outcome_yes": False},                         # excluded: not a forecast
            {"mode": "shadow", "status": "void", "p_model": 0.5, "p_market_mid": 0.5,
             "outcome_yes": None},                          # excluded: no outcome
            {"mode": "shadow", "status": "no_trade", "p_model": 0.5, "p_market_mid": 0.5},
            {"mode": "official", "status": "settled", "p_model": 0.1, "p_market_mid": 0.1,
             "outcome_yes": True},                          # other mode
        ]
        c = ft._calibration(decisions, "shadow", FROZEN)
        assert c["n_resolved"] == 2 and c["n_in_band"] == 1
        assert c["brier_model"] == pytest.approx((0.04 + 0.16) / 2)
        assert c["brier_market"] == pytest.approx((0.25 + 0.0001) / 2)
        assert c["brier_model_in_band"] == pytest.approx(0.04)
        assert c["brier_market_in_band"] == pytest.approx(0.25)
        assert ft._calibration(decisions, "official", FROZEN)["n_resolved"] == 1
        assert ft._calibration([], "shadow", FROZEN) == {"n_resolved": 0}

    def write_and_report(self, decisions, scans, monkeypatch, tmp_path) -> dict:
        patch_data_dir(tmp_path, monkeypatch)
        ft._rewrite_decisions(decisions)
        for s in scans:
            ft._append_jsonl(ft.SCANS_FILE, s)
        assert ft.stage_report(None) == 0
        return json.loads(ft.SUMMARY_FILE.read_text())

    def test_summary_has_the_contract_keys(self, monkeypatch, tmp_path, capsys):
        luna = tmp_path / "luna.json"
        luna.write_text(json.dumps({"test": {"n_trades": 124, "roi": 0.3178,
                                             "roi_ci_90": [0.0692, 0.5865]}}))
        monkeypatch.setattr(ft, "LUNA_PRIMARY_EVAL", luna)
        still_open = {**settled_row(2), "status": "open"}
        del still_open["outcome_yes"]
        decisions = [settled_row(0), settled_row(1), still_open,
                     {"mode": "shadow", "status": "no_trade", "market_id": "x", "p_model": 0.4,
                      "p_market_mid": 0.5, "outcome_yes": False, "ts_decision": NOW.isoformat(),
                      "evidence_ok": True}]
        s = self.write_and_report(decisions, [scan_row()], monkeypatch, tmp_path)
        assert set(s) >= {
            "generated_at", "mode", "model", "min_volume", "universe", "schedule",
            "analysis_plan", "last_decision_ts", "counts", "n_decisions", "shadow",
            "official", "scan_health", "health", "baseline", "shadow_note",
        }
        assert s["min_volume"] == 20_000.0
        assert s["universe"] == {
            "scan_min_volume_to_date": 20_000.0, "primary_min_final_volume": 500_000.0,
            "secondary_min_volume_at_decision": 500_000.0, "min_age_days": 3.0,
            "max_open_per_cluster": 2, "max_trades_per_scan": 6, "evidence_ok_min_days": 8,
        }
        assert s["schedule"] == {"cron_utc": "17 */6 * * *", "window_hours": [12.0, 27.0],
                                 "nominal_horizon_hours": 24}
        assert s["analysis_plan"]["target_n_trades_primary"] == 200
        assert s["analysis_plan"]["interim"] is True
        for mode in ("shadow", "official"):
            assert set(s[mode]) == {"primary", "secondary_at_decision", "all", "calibration",
                                    "n_open"}
        assert s["shadow"]["primary"]["n_settled"] == 2
        assert s["shadow"]["n_open"] == 1
        assert s["shadow"]["calibration"]["n_resolved"] == 3
        assert s["official"]["all"] == {"population": "all", "n_settled": 0, "n_trades": 0,
                                        "coin_baseline": {"n_markets": 0, "n_trades": 0}}
        assert s["baseline"]["n_trades"] == 124 and s["baseline"]["test_roi"] == 0.3178
        assert s["baseline"]["test_ci_90"] == [0.0692, 0.5865]
        assert s["baseline"]["claude_reference"]["test_roi"] == 0.3052
        assert s["health"] == "ok"
        sh = s["scan_health"]
        assert sh["last_status"] == "ok" and sh["last_exit_code"] == 0
        assert 1.9 < sh["hours_since_last_scan"] < 2.1
        assert sh["runs_last_7d"] == 1 and sh["failed_runs_last_7d"] == 0
        assert sh["last_run"]["n_candidates"] == 3
        assert sh["evidence"] == {"last_wiki_days_ok": 10, "decisions_last_14d": 1,
                                  "decisions_with_evidence_last_14d": 1}
        out = capsys.readouterr().out
        assert "health: ok" in out and "wiki 10/10 day pages" in out
        assert "shadow/primary" in out and "official/all" in out

    def test_baseline_falls_back_to_the_literal_values(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ft, "LUNA_PRIMARY_EVAL", tmp_path / "missing.json")
        s = self.write_and_report([], [], monkeypatch, tmp_path)
        assert s["baseline"]["test_roi"] == 0.318 and s["baseline"]["n_trades"] == 124
        assert s["baseline"]["test_ci_90"] == [0.069, 0.586]

    def test_health_is_unknown_without_scans(self, monkeypatch, tmp_path):
        s = self.write_and_report([], [], monkeypatch, tmp_path)
        assert s["scan_health"] is None and s["health"] == "unknown"

    @pytest.mark.parametrize("status", sorted(ft.FAILED_SCAN_STATUSES))
    def test_hard_failures_are_failed(self, status, monkeypatch, tmp_path):
        s = self.write_and_report([], [scan_row(status=status, exit_code=1)], monkeypatch, tmp_path)
        assert s["health"] == "failed"
        assert s["scan_health"]["failed_runs_last_7d"] == 1

    @pytest.mark.parametrize("overrides", [
        {"status": "evidence_degraded", "exit_code": 1, "wiki_days_ok": 3},
        {"n_forecast_failed": 1},
        {"ts": (NOW - timedelta(hours=30)).isoformat()},
    ])
    def test_soft_failures_are_degraded(self, overrides, monkeypatch, tmp_path):
        s = self.write_and_report([], [scan_row(**overrides)], monkeypatch, tmp_path)
        assert s["health"] == "degraded"

    def test_latest_scan_decides_not_the_worst(self, monkeypatch, tmp_path):
        scans = [scan_row(status="crashed", exit_code=1, ts=(NOW - timedelta(hours=8)).isoformat()),
                 scan_row()]
        s = self.write_and_report([], scans, monkeypatch, tmp_path)
        assert s["health"] == "ok"
        assert s["scan_health"]["runs_last_7d"] == 2
        assert s["scan_health"]["failed_runs_last_7d"] == 1


class TestUniverseConstants:
    """Pre-registered universe and schedule — changing these invalidates the run."""

    def test_registered_constants(self):
        assert ft.SCAN_MIN_VOLUME == 20_000.0
        assert ft.PRIMARY_MIN_FINAL_VOLUME == 500_000.0
        assert ft.MIN_VOLUME == 500_000.0
        assert ft.MIN_AGE_DAYS == 3.0
        assert ft.STAKE == 100.0
        assert ft.MAX_OPEN_PER_CLUSTER == 2
        assert ft.WINDOW_HOURS == (12.0, 27.0)
        assert ft.CRON_UTC == "17 */6 * * *"
        assert ft.EVIDENCE_OK_MIN_DAYS == 8 and ft.WIKI_LOOKBACK_DAYS == 10
        assert ft.TARGET_N_TRADES_PRIMARY == 200
        assert ft.RETRYABLE_STATUSES == {"forecast_failed"}

    def test_scan_defaults_match_the_registered_universe(self):
        a = ft.build_parser().parse_args(["scan"])
        assert (a.min_hours, a.max_hours) == (12.0, 27.0)
        assert a.min_volume == 20_000.0
        assert a.max_markets == 150
        assert a.max_pages == 6
        assert a.concurrency == 4

    def test_the_committed_wiki_cache_is_where_the_scan_looks(self):
        assert ft.WIKI_CACHE == ft.DATA_DIR / "wiki_cache"
        assert ft.SCANS_FILE == ft.DATA_DIR / "scans.jsonl"


class TestNum:
    @pytest.mark.parametrize("raw,expected", [
        ("0.5", 0.5), (0.03, 0.03), ("0.999", 0.999), ("NaN", None), ("inf", None),
        ("0", None), ("1", None), ("1.5", None), (None, None), ("abc", None), ({}, None),
    ])
    def test_prices_are_finite_and_strictly_inside_the_unit_interval(self, raw, expected):
        assert ft._num(raw) == expected

    def test_book_helpers_survive_garbage(self):
        assert ft.best_prices(["not", "a", "dict"]) == (None, None)
        assert ft.best_prices({"bids": "x", "asks": [1, 2]}) == (None, None)
        assert ft.fill_price(None, 100.0) == (None, 0.0)
        assert ft.fill_price({"asks": [{"price": "NaN", "size": "9"}]}, 100.0) == (None, 0.0)


def prereg(**drift) -> dict:
    reg = {
        "strategy": {"alpha": 1.0, "threshold": 0.08, "slippage": 0.01, "min_price": 0.03,
                     "max_price": 0.97, "max_divergence": 0.35},
        "universe": {"scan_min_volume_to_date": 20_000.0, "primary_min_final_volume": 500_000.0},
        "schedule": {"window_hours": [12.0, 27.0]},
        "forecaster": {"model_alias": FORECAST_MODEL},
    }
    for key, value in drift.items():
        section, field = key.split("__")
        reg[section][field] = value
    return reg


class TestPrereg:
    """Once registered, the scan refuses to run anything but what was registered."""

    def validate(self, reg, tmp_path, monkeypatch, *argv):
        path = tmp_path / "preregistration.json"
        path.write_text(json.dumps(reg))
        monkeypatch.setattr(ft, "PREREG_FILE", path)
        ft.validate_prereg(FROZEN, ft.build_parser().parse_args(["scan", *argv]))

    def test_matching_registration_passes(self, tmp_path, monkeypatch):
        self.validate(prereg(), tmp_path, monkeypatch)

    @pytest.mark.parametrize("key,bad", [
        ("strategy__alpha", 0.5), ("strategy__threshold", 0.05), ("strategy__slippage", 0.02),
        ("strategy__min_price", 0.01), ("strategy__max_price", 0.99),
        ("strategy__max_divergence", 1.0), ("universe__scan_min_volume_to_date", 500_000.0),
        ("universe__primary_min_final_volume", 250_000.0),
        ("schedule__window_hours", [12.0, 36.0]), ("forecaster__model_alias", "gpt-5.6-mini"),
    ])
    def test_drift_in_any_registered_field_is_fatal(self, key, bad, tmp_path, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            self.validate(prereg(**{key: bad}), tmp_path, monkeypatch)
        assert key.replace("__", ".") in str(exc.value)

    def test_running_with_different_flags_is_drift(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            self.validate(prereg(), tmp_path, monkeypatch, "--min-volume", "500000")
        assert "scan_min_volume_to_date" in str(exc.value)

    def test_missing_field_is_fatal(self, tmp_path, monkeypatch):
        reg = prereg()
        del reg["schedule"]["window_hours"]
        with pytest.raises(SystemExit):
            self.validate(reg, tmp_path, monkeypatch)

    def test_scan_refuses_and_records_the_refusal(self, world):
        ft.PREREG_FILE.write_text(json.dumps(prereg(strategy__max_divergence=1.0)))
        world.markets = [market(1)]
        with pytest.raises(SystemExit):
            world.scan()
        row = world.last_scan()
        assert row["status"] == "crashed" and row["mode"] == "official"
        assert "max_divergence" in row["error"]
        assert world.decisions() == [] and world.forecast_calls == []
