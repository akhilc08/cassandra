"""Oracle Multi-Agent System — orchestrates research, quant, risk, and portfolio management.

Submodule imports are LAZY. `import oracle.agents.forecaster` used to drag in the
whole platform via this file — paper_trading pulls aiosqlite, and the agent modules
pull the rest — so the forward test could not run anywhere that had not installed the
full platform. It failed in CI for exactly that reason while passing locally, because
the dev venv happened to have aiosqlite.

`from oracle.agents import AgentSystem, ResearchAgent, ...` still works; the cost is
just deferred to first access (PEP 562).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:  # import-time only for type checkers, never at runtime
    from oracle.agents.base import BaseAgent

_LAZY_EXPORTS = {
    "BaseAgent": "oracle.agents.base",
    "ToolCache": "oracle.agents.cache",
    "get_cache": "oracle.agents.cache",
    "MessageBus": "oracle.agents.messages",
    "PaperTradingEngine": "oracle.agents.paper_trading",
    "PortfolioManagerAgent": "oracle.agents.portfolio_manager",
    "QuantitativeAgent": "oracle.agents.quantitative",
    "ResearchAgent": "oracle.agents.research",
    "RiskAgent": "oracle.agents.risk",
}

__all__ = ["AgentSystem", *_LAZY_EXPORTS]


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


logger = structlog.get_logger()


class AgentSystem:
    """Top-level orchestrator that initializes and manages all agents.

    Usage:
        system = AgentSystem()
        await system.start()
        # ... system runs until stopped ...
        await system.stop()
    """

    def __init__(self, db_path: str = "oracle.db", initial_cash: float = 10000.0) -> None:
        from oracle.agents.cache import get_cache
        from oracle.agents.messages import MessageBus
        from oracle.agents.paper_trading import PaperTradingEngine
        from oracle.agents.portfolio_manager import PortfolioManagerAgent
        from oracle.agents.quantitative import QuantitativeAgent
        from oracle.agents.research import ResearchAgent
        from oracle.agents.risk import RiskAgent

        self.bus = MessageBus()
        self.trading_engine = PaperTradingEngine(db_path=db_path, initial_cash=initial_cash)
        self.cache = get_cache()

        # Initialize agents
        self.research_agent = ResearchAgent(self.bus)
        self.quant_agent = QuantitativeAgent(self.bus)
        self.risk_agent = RiskAgent(self.bus)
        self.portfolio_manager = PortfolioManagerAgent(self.bus, self.trading_engine)

        self._agents: list[BaseAgent] = [
            self.research_agent,
            self.quant_agent,
            self.risk_agent,
            self.portfolio_manager,
        ]
        self._tasks: list[asyncio.Task] = []  # type: ignore[type-arg]
        self._running = False

    async def start(self) -> None:
        """Initialize trading engine and start all agent loops."""
        await self.trading_engine.initialize()

        for agent in self._agents:
            task = asyncio.create_task(agent.run(), name=f"agent-{agent.agent_id}")
            self._tasks.append(task)

        self._running = True
        logger.info(
            "agent_system.started",
            agents=[a.agent_id for a in self._agents],
        )

    async def stop(self) -> None:
        """Stop all agents and cancel their tasks."""
        for agent in self._agents:
            agent.stop()

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        self._running = False
        logger.info("agent_system.stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        """System-wide status summary."""
        return {
            "running": self._running,
            "agents": {a.agent_id: a.status() for a in self._agents},
            "bus": {
                "registered_agents": self.bus.registered_agents,
                "message_history_count": len(self.bus.history),
            },
            "portfolio": self.trading_engine.get_portfolio_state(),
            "cache": {
                "size": self.cache.size,
                "hits": self.cache.stats.hits,
                "misses": self.cache.stats.misses,
                "hit_rate": round(self.cache.stats.hit_rate, 3),
            },
        }

    async def evaluate_market(
        self,
        market_id: str,
        question: str,
        category: str = "other",
        hours_to_resolution: float | None = None,
    ) -> str:
        """Submit a market for evaluation through the full pipeline.

        Returns the trace_id for tracking.
        """
        return await self.portfolio_manager.evaluate_opportunity(
            market_id=market_id,
            question=question,
            category=category,
            hours_to_resolution=hours_to_resolution,
        )
