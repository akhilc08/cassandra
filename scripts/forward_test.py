"""Forward paper-trading test — the live counterpart of the time-machine backtest.

Runs the SAME frozen system against live markets, logging every decision at
decision time (forecast, order book, fill price). This is the evidence the
backtest cannot provide: real books, real spreads, no selection hindsight.

Pre-registered 2026-06-10 from the 687-market train split (see
data/backtest/evaluation.json): alpha=1.0, tau=0.08, no evidence gate,
price bounds 0.03-0.97, flat $100 paper stakes. Success criteria at ~100
settled trades: positive P&L, cluster-bootstrap CI lower bound > -5%,
replica arm within its backtest CI. Do not retune while the test runs.

Two arms per market:
  replica  — wiki-events evidence only, exactly as backtested
  enhanced — web search enabled (legitimate live; no future to leak)

Usage (run daily, e.g. via cron):
    python scripts/forward_test.py scan      # find, forecast, decide, log
    python scripts/forward_test.py settle    # settle resolved decisions
    python scripts/forward_test.py report    # running P&L per arm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest import (  # noqa: E402
    EXCLUDE_KEYWORDS,
    GAMMA_API,
    USER_AGENT,
    _parse_dt,
    _parse_json_field,
    market_category,
    parse_resolution,
)
from oracle.agents.forecaster import FORECAST_MODEL, forecast  # noqa: E402
from oracle.ingestion.wiki_events import fetch_events_before  # noqa: E402

logger = structlog.get_logger()

CLOB_API = "https://clob.polymarket.com"
DATA_DIR = Path(__file__).parent.parent / "data" / "forward"
DECISIONS_FILE = DATA_DIR / "decisions.jsonl"

# --- Frozen strategy (pre-registered 2026-06-10; do not edit mid-test) ---
ALPHA = 1.0
TAU = 0.08
MIN_PRICE = 0.03
MAX_PRICE = 0.97
STAKE = 100.0
MIN_VOLUME = 20_000.0
MIN_AGE_DAYS = 3.0
ARMS = ("replica", "enhanced")


def token_ids(market: dict) -> tuple[str, str] | None:
    """(yes_token, no_token) located by outcome name."""
    outcomes = [str(o).lower() for o in _parse_json_field(market.get("outcomes"))]
    tokens = _parse_json_field(market.get("clobTokenIds"))
    if len(outcomes) == 2 and len(tokens) == 2 and "yes" in outcomes and "no" in outcomes:
        return str(tokens[outcomes.index("yes")]), str(tokens[outcomes.index("no")])
    return None


def best_prices(book: dict) -> tuple[float | None, float | None]:
    """(best_bid, best_ask) from a CLOB book payload."""
    bids = [float(b["price"]) for b in book.get("bids", []) if "price" in b]
    asks = [float(a["price"]) for a in book.get("asks", []) if "price" in a]
    return (max(bids) if bids else None, min(asks) if asks else None)


def decide(p_model: float, p_market_mid: float) -> tuple[str | None, float]:
    """Frozen decision rule. Returns (side or None, edge)."""
    edge = ALPHA * p_model + (1 - ALPHA) * p_market_mid - p_market_mid
    if not (MIN_PRICE <= p_market_mid <= MAX_PRICE):
        return None, edge
    if abs(edge) < TAU:
        return None, edge
    return ("yes" if edge > 0 else "no"), edge


def settle_pnl(side: str, entry_cost: float, outcome_yes: bool, stake: float) -> float:
    won = outcome_yes if side == "yes" else not outcome_yes
    return stake * ((1.0 - entry_cost) / entry_cost) if won else -stake


def _load_decisions() -> list[dict]:
    if not DECISIONS_FILE.exists():
        return []
    out = []
    for line in DECISIONS_FILE.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _rewrite_decisions(decisions: list[dict]) -> None:
    import os
    tmp = DECISIONS_FILE.with_suffix(f".jsonl.tmp{os.getpid()}")
    with tmp.open("w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")
    tmp.replace(DECISIONS_FILE)


async def fetch_live_candidates(client: httpx.AsyncClient, args) -> list[dict]:
    now = datetime.now(timezone.utc)
    lo = now + timedelta(hours=args.min_hours)
    hi = now + timedelta(hours=args.max_hours)
    params = {
        "active": "true",
        "closed": "false",
        "limit": "200",
        "order": "volumeNum",
        "ascending": "false",
        "end_date_min": lo.strftime("%Y-%m-%d"),
        "end_date_max": hi.strftime("%Y-%m-%d"),
        "volume_num_min": str(MIN_VOLUME),
    }
    resp = await client.get(f"{GAMMA_API}/markets", params=params)
    resp.raise_for_status()
    markets = resp.json()

    seen_market_ids = {d["market_id"] for d in _load_decisions()}
    seen_events: dict[str, int] = {}
    out = []
    for m in markets if isinstance(markets, list) else []:
        question = (m.get("question") or "").strip()
        if not question or any(kw in question.lower() for kw in EXCLUDE_KEYWORDS):
            continue
        if token_ids(m) is None:
            continue
        end_dt = _parse_dt(m.get("endDate", ""))
        created_dt = _parse_dt(m.get("createdAt", "") or m.get("startDate", ""))
        if not end_dt or not created_dt or not (lo <= end_dt <= hi):
            continue
        if (now - created_dt).total_seconds() < MIN_AGE_DAYS * 86400:
            continue
        if str(m.get("id")) in seen_market_ids:
            continue
        events = _parse_json_field(m.get("events"))
        event_id = str(events[0].get("id")) if events and isinstance(events[0], dict) else str(m.get("id"))
        if seen_events.get(event_id, 0) >= 2:
            continue
        seen_events[event_id] = seen_events.get(event_id, 0) + 1
        m["_event_id"] = event_id
        out.append(m)
        if len(out) >= args.max_markets:
            break
    return out


async def stage_scan(args) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        candidates = await fetch_live_candidates(client, args)
        print(f"{len(candidates)} new live candidates closing in {args.min_hours}-{args.max_hours}h")
        if not candidates:
            return

        semaphore = asyncio.Semaphore(args.concurrency)
        write_lock = asyncio.Lock()

        async def process(m: dict, out) -> None:
            market_id = str(m["id"])
            toks = token_ids(m)
            try:
                yes_book = (await client.get(f"{CLOB_API}/book", params={"token_id": toks[0]})).json()
                no_book = (await client.get(f"{CLOB_API}/book", params={"token_id": toks[1]})).json()
            except Exception as e:
                logger.warning("scan.book_failed", market_id=market_id, error=str(e))
                return
            yes_bid, yes_ask = best_prices(yes_book)
            no_bid, no_ask = best_prices(no_book)
            if yes_bid is None or yes_ask is None or no_ask is None:
                logger.info("scan.illiquid_book", market_id=market_id)
                return
            p_mid = (yes_bid + yes_ask) / 2

            wiki = await fetch_events_before(
                m["question"], now, client=client,
                cache_dir=Path(__file__).parent.parent / "data" / "backtest" / "wiki_events_pinned",
                lookback_days=10,
            )

            for arm in ARMS:
                async with semaphore:
                    f = await forecast(
                        question=m["question"],
                        as_of=now.date().isoformat(),
                        headlines=[],
                        world_events=wiki,
                        description=(m.get("description") or "")[:500],
                        allow_web=(arm == "enhanced"),
                        timeout=240 if arm == "enhanced" else 120,
                    )
                if f.failed:
                    logger.warning("scan.forecast_failed", market_id=market_id, arm=arm)
                    continue
                side, edge = decide(f.p_model, p_mid)
                entry_cost = None
                if side == "yes":
                    entry_cost = yes_ask
                elif side == "no":
                    entry_cost = no_ask
                if entry_cost is not None and not (0.0 < entry_cost < 1.0):
                    side, entry_cost = None, None
                record = {
                    "decision_id": f"{market_id}:{arm}",
                    "ts_decision": now.isoformat(),
                    "market_id": market_id,
                    "event_id": m["_event_id"],
                    "question": m["question"],
                    "category": market_category(m),
                    "end_date": m["endDate"],
                    "arm": arm,
                    "model": FORECAST_MODEL,
                    "p_model": f.p_model,
                    "evidence_strength": f.evidence_strength,
                    "reasoning": f.reasoning[:500],
                    "book": {"yes_bid": yes_bid, "yes_ask": yes_ask, "no_bid": no_bid, "no_ask": no_ask},
                    "p_market_mid": round(p_mid, 4),
                    "edge": round(edge, 4),
                    "side": side,
                    "entry_cost": entry_cost,
                    "stake": STAKE if side else 0.0,
                    "params": {"alpha": ALPHA, "tau": TAU, "min_price": MIN_PRICE, "max_price": MAX_PRICE},
                    "status": "open" if side else "no_trade",
                }
                async with write_lock:
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                tag = f"{side or 'no-trade':>8}"
                print(f"  [{arm:8}] {tag} mid={p_mid:.2f} model={f.p_model:.2f} | {m['question'][:52]}")

        with DECISIONS_FILE.open("a") as out:
            await asyncio.gather(*(process(m, out) for m in candidates))
    print(f"\nDecisions appended → {DECISIONS_FILE}")


async def stage_settle(args) -> None:
    decisions = _load_decisions()
    open_ids = {d["market_id"] for d in decisions if d["status"] == "open"}
    if not open_ids:
        print("No open positions to settle.")
        _print_report(decisions)
        return
    resolved: dict[str, bool | None] = {}
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        for market_id in open_ids:
            try:
                resp = await client.get(f"{GAMMA_API}/markets/{market_id}")
                resp.raise_for_status()
                m = resp.json()
            except Exception as e:
                logger.warning("settle.gamma_failed", market_id=market_id, error=str(e))
                continue
            if not m.get("closed"):
                continue
            resolved[market_id] = parse_resolution(m)  # None = ambiguous/disputed

    settled = 0
    for d in decisions:
        if d["status"] != "open" or d["market_id"] not in resolved:
            continue
        outcome = resolved[d["market_id"]]
        if outcome is None:
            d["status"] = "void"
            d["pnl"] = 0.0
        else:
            d["status"] = "settled"
            d["outcome_yes"] = outcome
            d["pnl"] = round(settle_pnl(d["side"], d["entry_cost"], outcome, d["stake"]), 2)
        d["ts_settled"] = datetime.now(timezone.utc).isoformat()
        settled += 1
    if settled:
        _rewrite_decisions(decisions)
    print(f"Settled {settled} positions.")
    _print_report(decisions)


def _print_report(decisions: list[dict]) -> None:
    print(f"\n=== Forward test report ({datetime.now(timezone.utc).date()}) ===")
    for arm in ARMS:
        rows = [d for d in decisions if d["arm"] == arm]
        trades = [d for d in rows if d["status"] in ("open", "settled", "void") and d.get("side")]
        settled = [d for d in rows if d["status"] == "settled"]
        pnl = sum(d.get("pnl", 0.0) for d in settled)
        staked = sum(d["stake"] for d in settled)
        wins = sum(1 for d in settled if d.get("pnl", 0) > 0)
        roi = pnl / staked if staked else 0.0
        print(f"  {arm:9} decisions={len(rows):4d} trades={len(trades):4d} open={sum(1 for d in rows if d['status']=='open'):3d} "
              f"settled={len(settled):3d} P&L=${pnl:+.2f} ROI={roi:+.1%} wins={wins}/{len(settled)}")
    print("  (success bar: ~100 settled trades, positive P&L, CI low > -5%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cassandra forward paper-trading test")
    parser.add_argument("stage", choices=["scan", "settle", "report"])
    parser.add_argument("--min-hours", type=int, default=12, help="min hours to scheduled close")
    parser.add_argument("--max-hours", type=int, default=36, help="max hours to scheduled close")
    parser.add_argument("--max-markets", type=int, default=25, help="max new markets per scan")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    async def run():
        if args.stage == "scan":
            await stage_scan(args)
        elif args.stage == "settle":
            await stage_settle(args)
        else:
            _print_report(_load_decisions())

    asyncio.run(run())


if __name__ == "__main__":
    main()
