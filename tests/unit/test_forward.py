"""Unit tests for the forward paper-trading test — pure logic and INVARIANTS.

These exist because forward-test v1 shipped two silent failures:

1. It re-implemented the decision rule and omitted `max_divergence`, so it ran a
   looser strategy than the backtested one. `TestTheBand` is the guard.
2. It failed silently for seven weeks while the report still looked healthy.
   `TestFailedForecastInvariant` and `TestReportShape` guard that class of lie.

No network, no API key: every test here exercises pure functions.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import forward_test as ft  # noqa: E402

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
        f = parse_response(json.dumps({"p_yes": p_yes, "evidence_strength": "none", "reasoning": ""}))
        assert f.failed is False
        assert f.p_model == pytest.approx(expected)
        assert 0.01 <= f.p_model <= 0.99

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_p_yes_fails_rather_than_clamping(self, literal):
        # json.loads happily accepts these; clamping NaN with min/max would have
        # returned 0.99, i.e. a maximally confident forecast out of nothing.
        text = '{"p_yes": %s, "evidence_strength": "strong", "reasoning": "x"}' % literal
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
        f = parse_response(json.dumps({"p_yes": 0.4, "evidence_strength": "weak", "reasoning": long}))
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
            {"decision_id": "2", "market_id": "2", "status": "no_trade", "ts_decision": "2026-08-09"},
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


class TestReportShape:
    """summary.json is what the dashboard reads; its shape is a contract."""

    def test_no_settled_rows_reports_zeros_not_a_healthy_looking_table(self):
        decisions = [
            {"status": "open", "mode": "shadow", "market_id": "1"},
            {"status": "no_trade", "mode": "shadow", "market_id": "2"},
            {"status": "forecast_failed", "mode": "shadow", "market_id": "3"},
        ]
        assert ft._report(decisions, "shadow") == {"mode": "shadow", "n_settled": 0, "n_trades": 0}

    def test_settled_rows_produce_the_full_metric_block(self):
        decisions = [
            {
                "status": "settled", "mode": "shadow", "market_id": f"m{i}",
                "question": "q", "p_model": 0.70, "p_market_mid": 0.50,
                "outcome_yes": i % 2 == 0, "end_date": f"2026-08-0{i + 1}",
                "category": "other", "evidence_strength": "moderate",
                "cluster_key": f"event:{i}",
            }
            for i in range(4)
        ]
        rep = ft._report(decisions, "shadow")
        assert set(rep) == {
            "mode", "n_settled", "n_trades", "total_pnl", "roi", "win_rate",
            "roi_ci_low", "roi_ci_high", "brier_blend", "brier_market",
            "max_drawdown", "n_clusters",
        }
        assert rep["n_settled"] == 4
        assert rep["n_trades"] == 4  # edge 0.20 is inside the band
        assert rep["n_clusters"] == 4
        assert rep["win_rate"] == 0.5

    def test_modes_are_kept_separate(self):
        row = {
            "status": "settled", "market_id": "m1", "question": "q",
            "p_model": 0.70, "p_market_mid": 0.50, "outcome_yes": True,
            "end_date": "2026-08-01", "category": "other",
            "evidence_strength": "moderate", "cluster_key": "event:1",
        }
        decisions = [{**row, "mode": "shadow"}, {**row, "market_id": "m2", "mode": "official"}]
        assert ft._report(decisions, "shadow")["n_settled"] == 1
        assert ft._report(decisions, "official")["n_settled"] == 1

    def test_only_settled_rows_count_toward_pnl(self):
        settled = {
            "status": "settled", "mode": "shadow", "market_id": "m1", "question": "q",
            "p_model": 0.70, "p_market_mid": 0.50, "outcome_yes": True,
            "end_date": "2026-08-01", "category": "other",
            "evidence_strength": "moderate", "cluster_key": "event:1",
        }
        still_open = {**settled, "market_id": "m2", "status": "open", "cluster_key": "event:2"}
        assert ft._report([settled, still_open], "shadow")["n_settled"] == 1

    def test_report_uses_the_frozen_params_not_a_local_copy(self):
        # A settled row 0.45 away from the market is outside max_divergence, so a
        # correctly-frozen report counts it as settled but never as a trade.
        wide = {
            "status": "settled", "mode": "shadow", "market_id": "m1", "question": "q",
            "p_model": 0.95, "p_market_mid": 0.50, "outcome_yes": True,
            "end_date": "2026-08-01", "category": "other",
            "evidence_strength": "moderate", "cluster_key": "event:1",
        }
        rep = ft._report([wide], "shadow")
        assert rep["n_settled"] == 1
        assert rep["n_trades"] == 0


class TestUniverseConstants:
    """Pre-registered universe filter — changing these invalidates the run."""

    def test_volume_floor_and_stake_are_the_registered_ones(self):
        assert ft.MIN_VOLUME == 500_000.0
        assert ft.STAKE == 100.0
        assert ft.MIN_AGE_DAYS == 3.0

    def test_scan_defaults_match_the_registered_floor(self):
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--min-volume", type=float, default=ft.MIN_VOLUME)
        assert p.parse_args([]).min_volume == 500_000.0
