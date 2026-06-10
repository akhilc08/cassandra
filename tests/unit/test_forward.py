"""Unit tests for the forward paper-trading test's pure logic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from forward_test import MIN_PRICE, TAU, best_prices, decide, settle_pnl, token_ids


class TestBookParsing:
    def test_best_prices(self):
        book = {
            "bids": [{"price": "0.44", "size": "10"}, {"price": "0.46", "size": "5"}],
            "asks": [{"price": "0.49", "size": "8"}, {"price": "0.47", "size": "3"}],
        }
        bid, ask = best_prices(book)
        assert bid == 0.46 and ask == 0.47

    def test_empty_book(self):
        assert best_prices({"bids": [], "asks": []}) == (None, None)


class TestDecision:
    def test_buys_yes_above_threshold(self):
        side, edge = decide(p_model=0.70, p_market_mid=0.50)
        assert side == "yes" and abs(edge - 0.20) < 1e-9

    def test_buys_no_below_threshold(self):
        side, _ = decide(p_model=0.30, p_market_mid=0.50)
        assert side == "no"

    def test_no_trade_within_tau(self):
        side, _ = decide(p_model=0.50 + TAU - 0.01, p_market_mid=0.50)
        assert side is None

    def test_no_trade_outside_price_bounds(self):
        side, _ = decide(p_model=0.50, p_market_mid=MIN_PRICE - 0.01)
        assert side is None


class TestSettlement:
    def test_yes_win(self):
        # buy YES at 0.47, resolves YES: profit = 100 * (1-0.47)/0.47
        assert abs(settle_pnl("yes", 0.47, True, 100.0) - 100 * (1 - 0.47) / 0.47) < 1e-9

    def test_yes_loss(self):
        assert settle_pnl("yes", 0.47, False, 100.0) == -100.0

    def test_no_win(self):
        assert abs(settle_pnl("no", 0.55, False, 100.0) - 100 * (1 - 0.55) / 0.55) < 1e-9

    def test_no_loss(self):
        assert settle_pnl("no", 0.55, True, 100.0) == -100.0


class TestTokenIds:
    def test_orders_by_outcome_name(self):
        m = {"outcomes": '["No", "Yes"]', "clobTokenIds": '["111", "222"]'}
        assert token_ids(m) == ("222", "111")

    def test_non_binary_none(self):
        m = {"outcomes": '["A", "B"]', "clobTokenIds": '["111", "222"]'}
        assert token_ids(m) is None
