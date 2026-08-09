"""Forward paper-trading test v2 — the live counterpart of the time-machine backtest.

Runs the SAME frozen system against live order books, logging every decision at
decision time (forecast, reasoning, book snapshot, fill). This is the evidence the
backtest cannot provide: real books, real spreads, no selection hindsight.

Design notes in docs/superpowers/specs/2026-08-09-forward-test-v2-design.md.

Two things v1 got wrong that this must not repeat:

1. v1 re-implemented the decision rule locally and omitted `max_divergence`, so it
   silently ran a looser strategy than the backtested one (175 vs 144 trades on the
   test split). This module imports `decide_trade` and asserts its params against
   `evaluation.json` at startup instead of declaring its own constants.
2. v1 failed silently: `claude -p` died on every cron invocation for seven weeks
   while the daily report still printed a healthy-looking table. A scan that
   forecasts nothing now exits non-zero.

Usage (run daily via GitHub Actions):
    python scripts/forward_test.py scan      # find, forecast, decide, log
    python scripts/forward_test.py settle    # settle resolved decisions
    python scripts/forward_test.py report    # metrics + CI -> summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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
from oracle.agents.forecaster_openai import FORECAST_MODEL, forecast  # noqa: E402
from oracle.evaluation.pnl import (  # noqa: E402
    BetInput,
    StrategyParams,
    decide_trade,
    simulate,
)
from oracle.ingestion.wiki_events import fetch_events_before  # noqa: E402

logger = structlog.get_logger()

CLOB_API = "https://clob.polymarket.com"
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "forward"
DECISIONS_FILE = DATA_DIR / "decisions.jsonl"
SUMMARY_FILE = DATA_DIR / "summary.json"
PREREG_FILE = DATA_DIR / "preregistration.json"
BASELINE_EVAL = ROOT / "data" / "baselines" / "claude-fable-5-2026-06-10" / "evaluation.json"
WIKI_CACHE = ROOT / "data" / "backtest" / "wiki_events_pinned"

# Pre-registered universe filter, new in v2 and justified on the TRAIN split:
# the $250k-500k bucket loses on both splits (-41.1% train, -85.0% test) while
# everything above $500k is strongly positive. See the design doc.
MIN_VOLUME = 500_000.0
MIN_AGE_DAYS = 3.0
STAKE = 100.0


def load_frozen_params() -> StrategyParams:
    """Load the frozen params from the archived baseline and fail loudly on drift.

    v1 declared ALPHA/TAU/MIN_PRICE/MAX_PRICE as local constants and simply forgot
    max_divergence. Deriving them from the artifact makes that class of bug
    impossible.
    """
    if not BASELINE_EVAL.exists():
        raise SystemExit(f"frozen params unavailable: {BASELINE_EVAL} missing")
    frozen = json.loads(BASELINE_EVAL.read_text())["frozen_params"]
    fields = StrategyParams.__dataclass_fields__
    params = StrategyParams(**{k: v for k, v in frozen.items() if k in fields})
    expected = {
        "alpha": 1.0,
        "threshold": 0.08,
        "slippage": 0.01,
        "min_price": 0.03,
        "max_price": 0.97,
        "max_divergence": 0.35,
    }
    for key, want in expected.items():
        got = getattr(params, key)
        if abs(got - want) > 1e-9:
            raise SystemExit(
                f"frozen param drift: {key}={got}, pre-registered {want}. "
                "Refusing to trade a strategy that is not the backtested one."
            )
    if params.evidence_gate is not None:
        raise SystemExit(f"frozen param drift: evidence_gate={params.evidence_gate}, expected None")
    return params


def cluster_key(market: dict) -> str:
    """Stable grouping key for the bootstrap.

    The backtest stored event_id == market_id for all 959 markets, which made its
    "cluster bootstrap" inert and its CI too narrow (10 correlated BTC-strike bets
    counted as 10 independent observations). Live we have the real Gamma event id;
    fall back to a normalized slug so daily strike ladders and the several markets
    on one match still collapse into one cluster.
    """
    events = _parse_json_field(market.get("events"))
    if events and isinstance(events[0], dict) and events[0].get("id"):
        return f"event:{events[0]['id']}"
    slug = market_category(market) or str(market.get("id"))
    slug = re.sub(r"-(exact-score|more-markets|correct-score|total-goals|btts).*$", "", slug)
    slug = re.sub(r"-above-on-.*$", "-above", slug)
    slug = re.sub(r"-below-on-.*$", "-below", slug)
    return f"slug:{slug}"


def token_ids(market: dict) -> tuple[str, str] | None:
    """(yes_token, no_token) located by outcome name, never by index."""
    outcomes = [str(o).lower() for o in _parse_json_field(market.get("outcomes"))]
    tokens = _parse_json_field(market.get("clobTokenIds"))
    if len(outcomes) == 2 and len(tokens) == 2 and "yes" in outcomes and "no" in outcomes:
        return str(tokens[outcomes.index("yes")]), str(tokens[outcomes.index("no")])
    return None


def best_prices(book: dict) -> tuple[float | None, float | None]:
    """(best_bid, best_ask). CLOB returns asks descending, so min/max, not [0]."""
    bids = [float(b["price"]) for b in book.get("bids", []) if "price" in b]
    asks = [float(a["price"]) for a in book.get("asks", []) if "price" in a]
    return (max(bids) if bids else None, min(asks) if asks else None)


def fill_price(book: dict, notional: float) -> tuple[float | None, float]:
    """Volume-weighted ask to buy `notional` dollars, walking the book.

    v1 recorded the best ask regardless of size, so a 1-share offer could set the
    recorded fill for a $100 stake. Returns (vwap, filled_notional); vwap is None
    when the book cannot absorb anything.
    """
    levels = []
    for a in book.get("asks", []):
        try:
            levels.append((float(a["price"]), float(a["size"])))
        except (KeyError, TypeError, ValueError):
            continue
    levels.sort(key=lambda x: x[0])
    spent = shares = 0.0
    for price, size in levels:
        if price <= 0:
            continue
        room = notional - spent
        if room <= 0:
            break
        take = min(size, room / price)
        spent += take * price
        shares += take
    if shares <= 0:
        return None, 0.0
    return spent / shares, spent


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
    tmp = DECISIONS_FILE.with_suffix(f".jsonl.tmp{os.getpid()}")
    with tmp.open("w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")
    tmp.replace(DECISIONS_FILE)


def current_mode() -> str:
    """'official' once a pre-registration is committed, else 'shadow'."""
    return "official" if PREREG_FILE.exists() else "shadow"


async def fetch_live_candidates(client: httpx.AsyncClient, args) -> list[dict]:
    now = datetime.now(timezone.utc)
    lo = now + timedelta(hours=args.min_hours)
    hi = now + timedelta(hours=args.max_hours)
    # Gamma's end_date_* are date-granular and reject min == max with a 422, so
    # widen by a day on each side and re-filter exactly in Python.
    params_base = {
        "active": "true",
        "closed": "false",
        "limit": "100",
        "order": "volumeNum",
        "ascending": "false",
        "end_date_min": (lo - timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date_max": (hi + timedelta(days=1)).strftime("%Y-%m-%d"),
        "volume_num_min": str(args.min_volume),
    }
    raw: list[dict] = []
    seen_ids: set[str] = set()
    for offset in range(0, args.max_pages * 100, 100):
        resp = await client.get(f"{GAMMA_API}/markets", params={**params_base, "offset": str(offset)})
        if resp.status_code != 200:
            logger.warning("scan.gamma_error", status=resp.status_code, offset=offset)
            break
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        for m in batch:
            mid = str(m.get("id"))
            if mid not in seen_ids:
                seen_ids.add(mid)
                raw.append(m)

    already = {d["market_id"] for d in _load_decisions()}
    seen_clusters: dict[str, int] = {}
    out = []
    for m in raw:
        question = (m.get("question") or "").strip()
        if not question or any(kw in question.lower() for kw in EXCLUDE_KEYWORDS):
            continue
        if token_ids(m) is None:
            continue
        if not m.get("acceptingOrders", True):
            continue
        if float(m.get("volumeNum") or 0) < args.min_volume:
            continue
        end_dt = _parse_dt(m.get("endDate", ""))
        created_dt = _parse_dt(m.get("createdAt", "") or m.get("startDate", ""))
        if not end_dt or not created_dt or not (lo <= end_dt <= hi):
            continue
        if (now - created_dt).total_seconds() < MIN_AGE_DAYS * 86400:
            continue
        if str(m.get("id")) in already:
            continue
        ckey = cluster_key(m)
        if seen_clusters.get(ckey, 0) >= 2:
            continue
        seen_clusters[ckey] = seen_clusters.get(ckey, 0) + 1
        m["_cluster_key"] = ckey
        out.append(m)
        if len(out) >= args.max_markets:
            break
    return out


async def stage_scan(args) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    params = load_frozen_params()
    mode = current_mode()
    now = datetime.now(timezone.utc)
    as_of = now.date().isoformat()

    n_forecast_ok = 0
    rows: list[dict] = []

    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        candidates = await fetch_live_candidates(client, args)
        print(f"{len(candidates)} candidates closing in {args.min_hours}-{args.max_hours}h "
              f"with volume >= ${args.min_volume:,.0f}  [mode={mode}]")
        if not candidates:
            # Not an error: a legitimately empty pool is expected at this floor.
            print("no candidates; nothing to forecast")
            return 0

        semaphore = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()

        async def process(m: dict) -> None:
            nonlocal n_forecast_ok
            market_id = str(m["id"])
            toks = token_ids(m)
            try:
                yes_book = (await client.get(f"{CLOB_API}/book", params={"token_id": toks[0]})).json()
                no_book = (await client.get(f"{CLOB_API}/book", params={"token_id": toks[1]})).json()
            except Exception as e:  # noqa: BLE001
                logger.warning("scan.book_failed", market_id=market_id, error=str(e)[:120])
                return
            yes_bid, yes_ask = best_prices(yes_book)
            no_bid, no_ask = best_prices(no_book)
            if yes_bid is None or yes_ask is None or no_ask is None:
                logger.info("scan.illiquid_book", market_id=market_id)
                return
            p_mid = (yes_bid + yes_ask) / 2

            wiki = await fetch_events_before(
                m["question"], now, client=client, cache_dir=WIKI_CACHE, lookback_days=10,
            )

            async with semaphore:
                f = await forecast(
                    question=m["question"],
                    as_of=as_of,
                    headlines=[],  # backtest ran --skip-gdelt; replica must match
                    world_events=wiki,
                    description=(m.get("description") or "")[:500],
                )

            base = {
                "decision_id": market_id,
                "ts_decision": now.isoformat(),
                "mode": mode,
                "market_id": market_id,
                "cluster_key": m.get("_cluster_key") or cluster_key(m),
                "question": m["question"],
                "category": market_category(m),
                "end_date": m.get("endDate"),
                "volume": float(m.get("volumeNum") or 0),
                "model": FORECAST_MODEL,
                "p_model": f.p_model,
                "evidence_strength": f.evidence_strength,
                "reasoning": f.reasoning,
                "raw_response": f.raw_response,  # v1 dropped this; the dashboard needs it
                "n_wiki_events": len(wiki or []),
                "book": {"yes_bid": yes_bid, "yes_ask": yes_ask,
                         "no_bid": no_bid, "no_ask": no_ask},
                "p_market_mid": p_mid,
                "params": {**params.__dict__, "min_volume": args.min_volume},
            }

            # Invariant: a failed forecast must never reach the strategy layer.
            # BetInput has no `failed` field, so a sentinel p_model=0.5 against a
            # market at 0.20 is edge 0.30 -- inside the band -- and would
            # manufacture a confident bogus trade.
            if f.failed:
                base.update({"status": "forecast_failed", "side": None, "stake": 0.0})
                async with lock:
                    rows.append(base)
                return

            n_forecast_ok += 1

            bet = BetInput(
                market_id=market_id,
                question=m["question"],
                p_model=f.p_model,
                p_market=p_mid,
                outcome=False,  # unknown until settle; not used by decide_trade
                close_time=m.get("endDate") or "",
                category=market_category(m),
                evidence_strength=f.evidence_strength,
                event_id=base["cluster_key"],
            )
            trade = decide_trade(bet, params)
            if trade is None:
                base.update({
                    "status": "no_trade",
                    "side": None,
                    "stake": 0.0,
                    "edge": f.p_model - p_mid,
                })
                async with lock:
                    rows.append(base)
                return

            # Replica cost (comparable to the backtest) and the executable cost
            # (depth-walked real ask) are logged side by side.
            side_book = yes_book if trade.side == "yes" else no_book
            vwap, filled = fill_price(side_book, STAKE)
            base.update({
                "status": "open",
                "side": trade.side,
                "edge": trade.edge,
                "entry_cost": trade.entry_cost,          # mid + slippage, replica
                "executable_cost": vwap,                 # depth-walked ask
                "executable_notional": filled,
                "stake": STAKE,
            })
            async with lock:
                rows.append(base)

        await asyncio.gather(*(process(m) for m in candidates))

    with DECISIONS_FILE.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    n_trades = sum(1 for r in rows if r["status"] == "open")
    n_failed = sum(1 for r in rows if r["status"] == "forecast_failed")
    print(f"logged {len(rows)} decisions: {n_trades} trades, "
          f"{sum(1 for r in rows if r['status'] == 'no_trade')} no-trade, {n_failed} failed")

    # v1's fatal flaw: every forecast failed for seven weeks while the report still
    # looked healthy. If we had candidates but not one forecast succeeded, that is
    # an infrastructure failure and the job must fail loudly.
    if n_forecast_ok == 0:
        print("ERROR: candidates found but zero forecasts succeeded", file=sys.stderr)
        return 1
    return 0


async def stage_settle(args) -> int:
    decisions = _load_decisions()
    open_rows = [d for d in decisions if d.get("status") == "open"]
    if not open_rows:
        print("no open positions to settle")
        return 0
    settled = 0
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        for d in open_rows:
            try:
                resp = await client.get(f"{GAMMA_API}/markets/{d['market_id']}")
                m = resp.json()
            except Exception as e:  # noqa: BLE001
                logger.warning("settle.fetch_failed", market_id=d["market_id"], error=str(e)[:120])
                continue
            if isinstance(m, list):
                m = m[0] if m else {}
            # Resolved markets still report active=true; `closed` is the only gate.
            if not m.get("closed"):
                continue
            outcome = parse_resolution(m)
            if outcome is None:
                d.update({"status": "void", "outcome_yes": None, "pnl": 0.0})
                settled += 1
                continue
            won = outcome if d["side"] == "yes" else (not outcome)
            cost = d["entry_cost"]
            d.update({
                "status": "settled",
                "outcome_yes": outcome,
                "won": won,
                "pnl": d["stake"] * ((1.0 - cost) / cost) if won else -d["stake"],
                "pnl_per_dollar": ((1.0 - cost) / cost) if won else -1.0,
                "ts_settled": datetime.now(timezone.utc).isoformat(),
            })
            if d.get("executable_cost"):
                ec = d["executable_cost"]
                d["pnl_executable"] = d["stake"] * ((1.0 - ec) / ec) if won else -d["stake"]
            settled += 1
    _rewrite_decisions(decisions)
    print(f"settled {settled} of {len(open_rows)} open positions")
    return 0


def _report(decisions: list[dict], mode: str | None) -> dict:
    rows = [d for d in decisions if d.get("status") == "settled"]
    if mode:
        rows = [d for d in rows if d.get("mode") == mode]
    if not rows:
        return {"mode": mode, "n_settled": 0, "n_trades": 0}
    bets = [
        BetInput(
            market_id=d["market_id"], question=d["question"], p_model=d["p_model"],
            p_market=d["p_market_mid"], outcome=bool(d["outcome_yes"]),
            close_time=d.get("end_date") or "", category=d.get("category", ""),
            evidence_strength=d.get("evidence_strength", "none"),
            event_id=d.get("cluster_key") or d["market_id"],
        )
        for d in rows
    ]
    params = load_frozen_params()
    rep = simulate(bets, params)
    return {
        "mode": mode,
        "n_settled": len(rows),
        "n_trades": rep.n_trades,
        "total_pnl": rep.total_pnl,
        "roi": rep.roi,
        "win_rate": rep.win_rate,
        "roi_ci_low": rep.roi_ci_low,
        "roi_ci_high": rep.roi_ci_high,
        "brier_blend": rep.brier_blend,
        "brier_market": rep.brier_market,
        "max_drawdown": rep.max_drawdown,
        "n_clusters": len({d.get("cluster_key") for d in rows}),
    }


def stage_report(args) -> int:
    decisions = _load_decisions()
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.get("status", "?")] = counts.get(d.get("status", "?"), 0) + 1
    last_ts = max((d.get("ts_decision", "") for d in decisions), default="")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": current_mode(),
        "model": FORECAST_MODEL,
        "min_volume": MIN_VOLUME,
        "last_decision_ts": last_ts,
        "counts": counts,
        "n_decisions": len(decisions),
        "shadow": _report(decisions, "shadow"),
        "official": _report(decisions, "official"),
        "baseline": {
            "source": "data/baselines/claude-fable-5-2026-06-10",
            "test_roi": 0.3052,
            "test_ci_90": [0.0797, 0.554],
            "note": "claude-fable-5, no volume floor; CI computed with inert clustering",
        },
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))

    print(f"=== Forward test ({summary['generated_at'][:10]}, mode={summary['mode']}) ===")
    print(f"  decisions={len(decisions)}  {counts}")
    for m in ("shadow", "official"):
        r = summary[m]
        if r.get("n_settled"):
            print(f"  {m:<9} settled={r['n_settled']:>4}  P&L=${r['total_pnl']:>+9,.0f}  "
                  f"ROI={r['roi']*100:>+6.1f}%  CI=[{r['roi_ci_low']*100:+.1f}%, "
                  f"{r['roi_ci_high']*100:+.1f}%]  clusters={r['n_clusters']}")
        else:
            print(f"  {m:<9} no settled trades yet")
    print(f"  -> {SUMMARY_FILE}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Cassandra forward paper-trading test v2")
    sub = p.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--min-hours", type=float, default=12.0)
    s.add_argument("--max-hours", type=float, default=36.0)
    s.add_argument("--min-volume", type=float, default=MIN_VOLUME)
    s.add_argument("--max-markets", type=int, default=40)
    s.add_argument("--max-pages", type=int, default=6)
    s.add_argument("--concurrency", type=int, default=4)

    sub.add_parser("settle")
    sub.add_parser("report")

    args = p.parse_args()
    if args.stage == "scan":
        return asyncio.run(stage_scan(args))
    if args.stage == "settle":
        return asyncio.run(stage_settle(args))
    return stage_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
