"""Polymarket CLOB price history client — market price at a historical timestamp.

Used by the time-machine backtest: the price a trader would actually have paid
at prediction time T, not the settlement price.

API: https://clob.polymarket.com/prices-history (public, no key).
"""

from __future__ import annotations

import httpx
import structlog

logger = structlog.get_logger()

CLOB_API = "https://clob.polymarket.com"


def last_price_at_or_before(
    history: list[dict], ts: int, tolerance_seconds: int = 6 * 3600
) -> float | None:
    """Pick the last traded price at or before ts from a price series.

    Returns None if the nearest point at or before ts is older than the
    tolerance (stale price = market wasn't really trading).
    """
    best_t, best_p = None, None
    for point in history:
        t, p = point.get("t"), point.get("p")
        if t is None or p is None:
            continue
        if t <= ts and (best_t is None or t > best_t):
            best_t, best_p = t, float(p)
    if best_t is None or ts - best_t > tolerance_seconds:
        return None
    return best_p


async def fetch_price_series(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity_minutes: int = 60,
    client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Fetch [{t, p}, ...] for a CLOB token between two unix timestamps."""
    params = {
        "market": token_id,
        "startTs": str(start_ts),
        "endTs": str(end_ts),
        "fidelity": str(fidelity_minutes),
    }
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    try:
        resp = await client.get(f"{CLOB_API}/prices-history", params=params)
        resp.raise_for_status()
        return resp.json().get("history", [])
    except Exception as e:
        logger.warning("price_history.fetch_failed", token_id=token_id[:16], error=str(e))
        return []
    finally:
        if own_client:
            await client.aclose()


async def fetch_price_at(
    token_id: str,
    ts: int,
    lookback_hours: int = 48,
    tolerance_seconds: int = 6 * 3600,
    client: httpx.AsyncClient | None = None,
) -> float | None:
    """Market price of a token at unix time ts (last trade at or before ts)."""
    history = await fetch_price_series(
        token_id, ts - lookback_hours * 3600, ts, fidelity_minutes=60, client=client
    )
    return last_price_at_or_before(history, ts, tolerance_seconds)
