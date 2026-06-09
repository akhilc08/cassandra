"""P&L simulation for the time-machine backtest.

Profitability is measured against the real market price at prediction time:
buy YES at p_market (+slippage) when the blended forecast is sufficiently
above it, buy NO at 1-p_market (+slippage) when sufficiently below. Shares
settle at $1 (win) or $0 (lose). Polymarket charges no trading fee; slippage
is the conservative stand-in for spread/impact.

All functions are pure so the strategy surface can be grid-searched on the
train split and frozen for the test split.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class BetInput:
    """One market joined with its forecast — everything known at time T plus outcome."""

    market_id: str
    question: str
    p_model: float          # forecaster's P(YES) at T
    p_market: float         # market price (YES) at T
    outcome: bool           # resolved YES?
    close_time: str         # ISO timestamp, for chronological ordering
    category: str = "other"
    evidence_strength: str = "none"
    event_id: str = ""      # markets of the same event have correlated outcomes


@dataclass
class StrategyParams:
    alpha: float = 0.5          # blend weight on model: p = a*model + (1-a)*market
    threshold: float = 0.05     # min |p_blend - p_market| to trade
    slippage: float = 0.01      # added to entry cost per share
    min_price: float = 0.03     # skip near-settled markets
    max_price: float = 0.97
    evidence_gate: str | None = None  # if set, trade only when strength is in gate
    # When the model disagrees with the market by more than this, the model
    # is more likely missing information than the market being wrong — skip.
    max_divergence: float = 1.0
    stake: float = 100.0        # flat stake per trade (USD)
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.05     # max fraction of bankroll per trade

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "slippage": self.slippage,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "evidence_gate": self.evidence_gate,
            "max_divergence": self.max_divergence,
            "stake": self.stake,
            "kelly_fraction": self.kelly_fraction,
            "kelly_cap": self.kelly_cap,
        }


_STRENGTH_RANK = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}


@dataclass
class Trade:
    market_id: str
    question: str
    event_id: str
    side: str               # "yes" | "no"
    entry_cost: float       # cost per share incl. slippage
    p_blend: float
    p_market: float
    edge: float             # p_blend - p_market (signed)
    won: bool
    pnl_per_dollar: float   # profit per $1 staked
    close_time: str
    category: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "side": self.side,
            "entry_cost": round(self.entry_cost, 4),
            "p_blend": round(self.p_blend, 4),
            "p_market": round(self.p_market, 4),
            "edge": round(self.edge, 4),
            "won": self.won,
            "pnl_per_dollar": round(self.pnl_per_dollar, 4),
            "close_time": self.close_time,
            "category": self.category,
        }


@dataclass
class PnLReport:
    params: StrategyParams
    n_markets: int = 0
    n_trades: int = 0
    total_staked: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0                    # total_pnl / total_staked
    win_rate: float = 0.0
    avg_edge_taken: float = 0.0
    max_drawdown: float = 0.0           # on flat-stake cumulative P&L, in $
    kelly_final_bankroll: float = 0.0   # compounded, starting at 1000
    brier_blend: float = 0.0            # all markets, blended forecast
    brier_market: float = 0.0           # all markets, market price
    brier_blend_traded: float = 0.0     # traded subset only
    brier_market_traded: float = 0.0
    roi_ci_low: float = 0.0             # bootstrap 90% CI on per-trade ROI
    roi_ci_high: float = 0.0
    trades: list[Trade] = field(default_factory=list)

    def to_dict(self, include_trades: bool = False) -> dict[str, Any]:
        d = {
            "params": self.params.to_dict(),
            "n_markets": self.n_markets,
            "n_trades": self.n_trades,
            "total_staked": round(self.total_staked, 2),
            "total_pnl": round(self.total_pnl, 2),
            "roi": round(self.roi, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_edge_taken": round(self.avg_edge_taken, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "kelly_final_bankroll": round(self.kelly_final_bankroll, 2),
            "brier_blend": round(self.brier_blend, 4),
            "brier_market": round(self.brier_market, 4),
            "brier_blend_traded": round(self.brier_blend_traded, 4),
            "brier_market_traded": round(self.brier_market_traded, 4),
            "roi_ci_90": [round(self.roi_ci_low, 4), round(self.roi_ci_high, 4)],
        }
        if include_trades:
            d["trades"] = [t.to_dict() for t in self.trades]
        return d


def decide_trade(bet: BetInput, params: StrategyParams) -> Trade | None:
    """Apply the strategy to one market. Returns None when no trade."""
    if not (params.min_price <= bet.p_market <= params.max_price):
        return None
    if params.evidence_gate is not None:
        if _STRENGTH_RANK.get(bet.evidence_strength, 0) < _STRENGTH_RANK[params.evidence_gate]:
            return None
    if abs(bet.p_model - bet.p_market) > params.max_divergence:
        return None

    p_blend = params.alpha * bet.p_model + (1 - params.alpha) * bet.p_market
    edge = p_blend - bet.p_market
    if abs(edge) < params.threshold:
        return None

    if edge > 0:
        side, entry_cost, won = "yes", bet.p_market + params.slippage, bet.outcome
    else:
        side, entry_cost, won = "no", (1 - bet.p_market) + params.slippage, not bet.outcome

    if entry_cost >= 1.0 or entry_cost <= 0.0:
        return None

    # Stake $1: buy 1/cost shares, each settles at $1 or $0.
    pnl_per_dollar = (1.0 - entry_cost) / entry_cost if won else -1.0

    return Trade(
        market_id=bet.market_id,
        question=bet.question,
        event_id=bet.event_id or bet.market_id,
        side=side,
        entry_cost=entry_cost,
        p_blend=p_blend,
        p_market=bet.p_market,
        edge=edge,
        won=won,
        pnl_per_dollar=pnl_per_dollar,
        close_time=bet.close_time,
        category=bet.category,
    )


def _brier(pairs: list[tuple[float, bool]]) -> float:
    if not pairs:
        return 0.0
    return statistics.mean((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs)


def _kelly_fraction(p_win: float, entry_cost: float) -> float:
    """Kelly optimal fraction for a binary contract bought at entry_cost."""
    if entry_cost <= 0.0 or entry_cost >= 1.0:
        return 0.0
    b = (1.0 - entry_cost) / entry_cost  # net odds
    return max(0.0, (p_win * b - (1.0 - p_win)) / b)


def simulate(
    bets: list[BetInput],
    params: StrategyParams,
    bootstrap_iters: int = 2000,
    seed: int = 42,
) -> PnLReport:
    """Run the strategy over all markets, chronologically by close time."""
    report = PnLReport(params=params, n_markets=len(bets))
    ordered = sorted(bets, key=lambda b: b.close_time)

    trades: list[Trade] = []
    for bet in ordered:
        trade = decide_trade(bet, params)
        if trade:
            trades.append(trade)
    report.trades = trades
    report.n_trades = len(trades)

    # Brier on the full set: did blending beat the market at all?
    blend_pairs = [
        (params.alpha * b.p_model + (1 - params.alpha) * b.p_market, b.outcome) for b in bets
    ]
    market_pairs = [(b.p_market, b.outcome) for b in bets]
    report.brier_blend = _brier(blend_pairs)
    report.brier_market = _brier(market_pairs)

    if not trades:
        return report

    traded_ids = {t.market_id for t in trades}
    traded_bets = [b for b in bets if b.market_id in traded_ids]
    report.brier_blend_traded = _brier(
        [(params.alpha * b.p_model + (1 - params.alpha) * b.p_market, b.outcome) for b in traded_bets]
    )
    report.brier_market_traded = _brier([(b.p_market, b.outcome) for b in traded_bets])

    # Flat-stake P&L and drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cumulative += t.pnl_per_dollar * params.stake
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    report.total_staked = params.stake * len(trades)
    report.total_pnl = cumulative
    report.roi = cumulative / report.total_staked if report.total_staked else 0.0
    report.win_rate = sum(1 for t in trades if t.won) / len(trades)
    report.avg_edge_taken = statistics.mean(abs(t.edge) for t in trades)
    report.max_drawdown = max_dd

    # Fractional-Kelly compounding, $1000 start
    bankroll = 1000.0
    for t in trades:
        p_win = t.p_blend if t.side == "yes" else 1.0 - t.p_blend
        f = min(_kelly_fraction(p_win, t.entry_cost) * params.kelly_fraction, params.kelly_cap)
        stake = bankroll * f
        bankroll += stake * t.pnl_per_dollar
    report.kelly_final_bankroll = bankroll

    # Bootstrap 90% CI on mean per-trade ROI. Trades within one event have
    # correlated outcomes, so resample event clusters, not individual trades.
    rng = random.Random(seed)
    clusters: dict[str, list[float]] = {}
    for t in trades:
        clusters.setdefault(t.event_id, []).append(t.pnl_per_dollar)
    cluster_list = list(clusters.values())
    means = []
    for _ in range(bootstrap_iters):
        sample: list[float] = []
        for _ in cluster_list:
            sample.extend(cluster_list[rng.randrange(len(cluster_list))])
        means.append(statistics.mean(sample))
    means.sort()
    report.roi_ci_low = means[int(0.05 * len(means))]
    report.roi_ci_high = means[int(0.95 * len(means)) - 1]

    return report


def simulate_rule(
    bets: list[BetInput],
    rule: str,
    slippage: float = 0.01,
    stake: float = 100.0,
) -> PnLReport:
    """Mechanical no-LLM baselines, implemented directly (not via decide_trade,
    whose threshold/price-bound interactions silently skip e.g. favorites
    priced above 0.94).

    Rules:
      favorite  — buy whichever side trades at >= 0.90 (skip mid markets)
      always_no — buy NO on every market
    """
    params = StrategyParams(alpha=1.0, threshold=0.0, slippage=slippage, stake=stake)
    report = PnLReport(params=params, n_markets=len(bets))
    trades: list[Trade] = []
    for bet in sorted(bets, key=lambda b: b.close_time):
        if rule == "favorite":
            if bet.p_market >= 0.90:
                side = "yes"
            elif bet.p_market <= 0.10:
                side = "no"
            else:
                continue
        elif rule == "always_no":
            side = "no"
        else:
            raise ValueError(rule)
        entry_cost = (bet.p_market if side == "yes" else 1 - bet.p_market) + slippage
        if not (0.0 < entry_cost < 1.0):
            continue
        won = bet.outcome if side == "yes" else not bet.outcome
        trades.append(Trade(
            market_id=bet.market_id, question=bet.question,
            event_id=bet.event_id or bet.market_id, side=side,
            entry_cost=entry_cost, p_blend=bet.p_market, p_market=bet.p_market,
            edge=0.0, won=won,
            pnl_per_dollar=(1.0 - entry_cost) / entry_cost if won else -1.0,
            close_time=bet.close_time, category=bet.category,
        ))
    return _report_from_trades(report, trades, params)


def _report_from_trades(
    report: PnLReport, trades: list[Trade], params: StrategyParams
) -> PnLReport:
    """Fill P&L metrics for a fixed list of trades (no Brier/Kelly semantics)."""
    report.trades = trades
    report.n_trades = len(trades)
    if not trades:
        return report
    cumulative = peak = max_dd = 0.0
    for t in trades:
        cumulative += t.pnl_per_dollar * params.stake
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    report.total_staked = params.stake * len(trades)
    report.total_pnl = cumulative
    report.roi = cumulative / report.total_staked
    report.win_rate = sum(1 for t in trades if t.won) / len(trades)
    report.max_drawdown = max_dd

    rng = random.Random(42)
    clusters: dict[str, list[float]] = {}
    for t in trades:
        clusters.setdefault(t.event_id, []).append(t.pnl_per_dollar)
    cluster_list = list(clusters.values())
    means = []
    for _ in range(2000):
        sample: list[float] = []
        for _ in cluster_list:
            sample.extend(cluster_list[rng.randrange(len(cluster_list))])
        means.append(statistics.mean(sample))
    means.sort()
    report.roi_ci_low = means[int(0.05 * len(means))]
    report.roi_ci_high = means[int(0.95 * len(means)) - 1]
    return report


def grid_search(
    bets: list[BetInput],
    alphas: list[float] | None = None,
    thresholds: list[float] | None = None,
    evidence_gates: list[str | None] | None = None,
    max_divergences: list[float] | None = None,
    min_trades: int = 10,
    base_params: StrategyParams | None = None,
) -> list[PnLReport]:
    """Evaluate the strategy grid on (train) bets, best first.

    Ranked by the bootstrap lower bound of per-trade ROI — total P&L alone
    rewards lucky single trades.
    """
    alphas = alphas if alphas is not None else [0.25, 0.5, 0.75, 1.0]
    thresholds = thresholds if thresholds is not None else [0.03, 0.05, 0.08, 0.12]
    evidence_gates = evidence_gates if evidence_gates is not None else [None, "moderate"]
    max_divergences = max_divergences if max_divergences is not None else [1.0, 0.35]
    base = base_params or StrategyParams()

    reports = []
    for a in alphas:
        for tau in thresholds:
            for gate in evidence_gates:
                for max_div in max_divergences:
                    params = StrategyParams(
                        alpha=a,
                        threshold=tau,
                        slippage=base.slippage,
                        min_price=base.min_price,
                        max_price=base.max_price,
                        evidence_gate=gate,
                        max_divergence=max_div,
                        stake=base.stake,
                        kelly_fraction=base.kelly_fraction,
                        kelly_cap=base.kelly_cap,
                    )
                    reports.append(simulate(bets, params))

    eligible = [r for r in reports if r.n_trades >= min_trades]
    if not eligible:
        # No cell produced enough trades — don't let a lucky 1-trade cell win.
        logger.warning("grid_search.no_cell_meets_min_trades", min_trades=min_trades)
        eligible = [r for r in reports if r.n_trades >= 3] or reports
    eligible.sort(key=lambda r: r.roi_ci_low, reverse=True)
    return eligible
