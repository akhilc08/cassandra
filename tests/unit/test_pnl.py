"""Unit tests for P&L simulation — the math the profit claims rest on."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from oracle.evaluation.pnl import (
    BetInput,
    StrategyParams,
    _kelly_fraction,
    decide_trade,
    grid_search,
    simulate,
    simulate_rule,
)


def bet(p_model=0.7, p_market=0.5, outcome=True, market_id="m1", strength="moderate"):
    return BetInput(
        market_id=market_id,
        question="q",
        p_model=p_model,
        p_market=p_market,
        outcome=outcome,
        close_time="2026-05-01T00:00:00Z",
        evidence_strength=strength,
    )


class TestDecideTrade:
    def test_buys_yes_when_model_above_market(self):
        t = decide_trade(bet(p_model=0.7, p_market=0.5), StrategyParams(alpha=1.0, threshold=0.05))
        assert t is not None and t.side == "yes"
        assert t.entry_cost == 0.51  # market price + 1c slippage

    def test_buys_no_when_model_below_market(self):
        t = decide_trade(bet(p_model=0.3, p_market=0.5), StrategyParams(alpha=1.0, threshold=0.05))
        assert t is not None and t.side == "no"
        assert t.entry_cost == 0.51  # (1 - 0.5) + slippage

    def test_no_trade_below_threshold(self):
        t = decide_trade(bet(p_model=0.53, p_market=0.5), StrategyParams(alpha=1.0, threshold=0.05))
        assert t is None

    def test_blend_shrinks_edge(self):
        # alpha=0.5: blend = 0.6, edge = 0.1
        t = decide_trade(bet(p_model=0.7, p_market=0.5), StrategyParams(alpha=0.5, threshold=0.05))
        assert t is not None
        assert abs(t.p_blend - 0.6) < 1e-9

    def test_alpha_zero_never_trades(self):
        t = decide_trade(bet(p_model=0.99, p_market=0.5), StrategyParams(alpha=0.0, threshold=0.01))
        assert t is None

    def test_price_bounds_skip_settled_markets(self):
        t = decide_trade(bet(p_model=0.5, p_market=0.99), StrategyParams(alpha=1.0, threshold=0.05))
        assert t is None

    def test_evidence_gate_blocks_weak(self):
        params = StrategyParams(alpha=1.0, threshold=0.05, evidence_gate="moderate")
        assert decide_trade(bet(strength="weak"), params) is None
        assert decide_trade(bet(strength="strong"), params) is not None

    def test_max_divergence_skips_extreme_disagreement(self):
        params = StrategyParams(alpha=1.0, threshold=0.05, max_divergence=0.35)
        assert decide_trade(bet(p_model=0.95, p_market=0.06), params) is None
        assert decide_trade(bet(p_model=0.7, p_market=0.5), params) is not None

    def test_win_pnl_per_dollar(self):
        t = decide_trade(bet(p_model=0.8, p_market=0.5, outcome=True), StrategyParams(alpha=1.0))
        # cost 0.51, win: (1 - 0.51) / 0.51 per dollar
        assert abs(t.pnl_per_dollar - (1 - 0.51) / 0.51) < 1e-9

    def test_loss_pnl_per_dollar(self):
        t = decide_trade(bet(p_model=0.8, p_market=0.5, outcome=False), StrategyParams(alpha=1.0))
        assert t.pnl_per_dollar == -1.0


class TestKelly:
    def test_positive_edge(self):
        # p=0.6 at cost 0.5: b=1, f = (0.6 - 0.4)/1 = 0.2
        assert abs(_kelly_fraction(0.6, 0.5) - 0.2) < 1e-9

    def test_no_edge_is_zero(self):
        assert _kelly_fraction(0.5, 0.5) == 0.0

    def test_negative_edge_clamped_to_zero(self):
        assert _kelly_fraction(0.4, 0.5) == 0.0


class TestSimulate:
    def test_flat_stake_pnl(self):
        bets = [
            bet(p_model=0.8, p_market=0.5, outcome=True, market_id="a"),
            bet(p_model=0.8, p_market=0.5, outcome=False, market_id="b"),
        ]
        r = simulate(bets, StrategyParams(alpha=1.0, threshold=0.05, stake=100.0))
        assert r.n_trades == 2
        win = (1 - 0.51) / 0.51 * 100
        assert abs(r.total_pnl - (win - 100)) < 1e-6
        assert r.win_rate == 0.5

    def test_drawdown_tracks_peak_to_trough(self):
        bets = [
            BetInput("a", "q", 0.9, 0.5, True, "2026-01-01"),
            BetInput("b", "q", 0.9, 0.5, False, "2026-01-02"),
            BetInput("c", "q", 0.9, 0.5, False, "2026-01-03"),
        ]
        r = simulate(bets, StrategyParams(alpha=1.0, threshold=0.05, stake=100.0))
        assert r.max_drawdown == 200.0  # two consecutive full losses

    def test_brier_compares_blend_to_market(self):
        # model perfect, market clueless
        bets = [
            bet(p_model=0.95, p_market=0.5, outcome=True, market_id="a"),
            bet(p_model=0.05, p_market=0.5, outcome=False, market_id="b"),
        ]
        r = simulate(bets, StrategyParams(alpha=1.0, threshold=0.05))
        assert r.brier_blend < r.brier_market

    def test_no_trades_zero_pnl(self):
        r = simulate([bet(p_model=0.5, p_market=0.5)], StrategyParams(alpha=1.0, threshold=0.05))
        assert r.n_trades == 0 and r.total_pnl == 0.0

    def test_deterministic_bootstrap(self):
        bets = [bet(p_model=0.8, p_market=0.5, outcome=True, market_id=str(i)) for i in range(20)]
        r1 = simulate(bets, StrategyParams(alpha=1.0))
        r2 = simulate(bets, StrategyParams(alpha=1.0))
        assert r1.roi_ci_low == r2.roi_ci_low


class TestNoSideMath:
    def test_no_side_win_pnl(self):
        # model 0.2 vs market 0.7 → buy NO at 0.30 + 0.01 slippage; NO wins
        t = decide_trade(bet(p_model=0.2, p_market=0.7, outcome=False), StrategyParams(alpha=1.0))
        assert t.side == "no" and t.won
        assert abs(t.pnl_per_dollar - (1 - 0.31) / 0.31) < 1e-9

    def test_kelly_uses_no_side_probability(self):
        # p_blend = 0.2 → NO-side win prob is 0.8, not 0.2
        t = decide_trade(bet(p_model=0.2, p_market=0.7, outcome=False), StrategyParams(alpha=1.0))
        p_win = 1.0 - t.p_blend
        assert p_win == 0.8
        assert _kelly_fraction(p_win, t.entry_cost) > 0

    def test_kelly_guards_degenerate_costs(self):
        assert _kelly_fraction(0.9, 0.0) == 0.0
        assert _kelly_fraction(0.9, 1.0) == 0.0


class TestMechanicalRules:
    def test_favorite_includes_high_priced_favorites(self):
        # 0.96 favorite must be traded (the old recast skipped >0.94)
        bets = [bet(p_model=0.5, p_market=0.96, outcome=True, market_id="hi")]
        r = simulate_rule(bets, "favorite")
        assert r.n_trades == 1 and r.trades[0].side == "yes"

    def test_favorite_buys_no_side_longshots(self):
        bets = [bet(p_model=0.5, p_market=0.05, outcome=False, market_id="lo")]
        r = simulate_rule(bets, "favorite")
        assert r.n_trades == 1 and r.trades[0].side == "no" and r.trades[0].won

    def test_favorite_skips_mid_markets(self):
        bets = [bet(p_model=0.5, p_market=0.5, market_id="mid")]
        assert simulate_rule(bets, "favorite").n_trades == 0

    def test_always_no_trades_everything(self):
        bets = [bet(p_model=0.5, p_market=p, outcome=False, market_id=str(p)) for p in (0.1, 0.5, 0.9)]
        r = simulate_rule(bets, "always_no")
        assert r.n_trades == 3 and r.win_rate == 1.0


class TestClusterBootstrap:
    def test_single_event_cluster_widens_ci(self):
        # 20 trades: as 20 independent events vs 2 events of 10 — clustered CI must be wider
        wins = [bet(p_model=0.9, p_market=0.5, outcome=(i % 2 == 0), market_id=f"m{i}") for i in range(20)]
        independent = simulate(wins, StrategyParams(alpha=1.0))

        clustered_bets = [
            BetInput(f"m{i}", "q", 0.9, 0.5, i % 2 == 0, "2026-05-01", event_id=f"ev{i % 2}")
            for i in range(20)
        ]
        clustered = simulate(clustered_bets, StrategyParams(alpha=1.0))
        width_ind = independent.roi_ci_high - independent.roi_ci_low
        width_clu = clustered.roi_ci_high - clustered.roi_ci_low
        assert width_clu > width_ind


class TestGridSearch:
    def test_prefers_min_trades(self):
        # 20 markets with consistent model edge
        bets = [
            bet(p_model=0.75, p_market=0.55, outcome=(i % 4 != 0), market_id=str(i))
            for i in range(20)
        ]
        ranked = grid_search(bets, min_trades=5)
        assert ranked[0].n_trades >= 5

    def test_returns_all_cells_when_none_eligible(self):
        bets = [bet(p_model=0.5, p_market=0.5, market_id=str(i)) for i in range(5)]
        ranked = grid_search(bets, min_trades=5)
        assert len(ranked) > 0
