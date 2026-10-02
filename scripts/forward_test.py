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

The shadow period (2026-08-12..2026-09-24) found two more silent failures, fixed
here: the Wikipedia evidence channel was dead on every decision and nothing
checked it was alive; and the $500k volume floor was applied to volume at decision
time although the backtest measured it at resolution, which cut the live universe
to a tenth of the backtested one. Every scan now appends a health record to
data/forward/scans.jsonl, and `report` turns it into one health verdict that the
dashboard shows before any performance number.

Usage (run every six hours via GitHub Actions):
    python scripts/forward_test.py scan      # find, forecast, decide, log
    python scripts/forward_test.py settle    # resolve outcomes, settle open trades
    python scripts/forward_test.py report    # metrics + CI + health -> summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from datetime import UTC, date, datetime, timedelta
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
    _brier,
    decide_trade,
    simulate,
)
from oracle.ingestion.wiki_events import fetch_days_before, relevant_events  # noqa: E402

logger = structlog.get_logger()

CLOB_API = "https://clob.polymarket.com"
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "forward"
DECISIONS_FILE = DATA_DIR / "decisions.jsonl"
SCANS_FILE = DATA_DIR / "scans.jsonl"
SUMMARY_FILE = DATA_DIR / "summary.json"
PREREG_FILE = DATA_DIR / "preregistration.json"
# Committed, not gitignored: a cold cache on a fresh Actions runner plus a
# rate-limited Wikipedia is exactly how the evidence channel died in shadow mode.
WIKI_CACHE = DATA_DIR / "wiki_cache"
BASELINE_EVAL = ROOT / "data" / "baselines" / "claude-fable-5-2026-06-10" / "evaluation.json"
LUNA_PRIMARY_EVAL = ROOT / "data" / "backtest" / "evaluation.gpt-5.6-luna.primary.json"

# Three volume numbers, and why they differ. The $500k evidence (the $250k-500k
# bucket loses on both splits, everything above wins) was measured on FINAL
# volume at resolution, which is all the backtest ever saw. Shadow mode applied
# the same $500k to volume-TO-DATE at decision time, a day before close, when a
# game-day market carries ~10% of its final volume: that cut live supply to ~8%
# of the backtest-equivalent and excluded game-day sports, 76% of backtest
# trades. So the scan floor is the backtest's collect floor, the primary
# population is selected on final volume at report time, and the decision-time
# floor survives only as a secondary population.
SCAN_MIN_VOLUME = 20_000.0            # volume-to-date at scan == the backtest's collect floor
PRIMARY_MIN_FINAL_VOLUME = 500_000.0  # applied at REPORT time to volume_final
MIN_VOLUME = 500_000.0                # secondary population: volume at decision time
MIN_AGE_DAYS = 3.0
STAKE = 100.0
MAX_OPEN_PER_CLUSTER = 2
# A forecast_failed row is an infrastructure hiccup, not a decision: the market
# is offered again on the next scan. Every other status is final for that market.
RETRYABLE_STATUSES = {"forecast_failed"}

# Evidence channel: ten pinned Wikipedia day pages per scan, fetched once and
# shared by every candidate. Fewer than eight alive means the forecaster works
# from a different information set than the backtest did, so the rows are
# flagged (evidence_ok=false) and the run exits 1.
EVIDENCE_OK_MIN_DAYS = 8
WIKI_LOOKBACK_DAYS = 10
WIKI_CACHE_RETENTION_DAYS = 40

# Schedule: four scans a day. Actions cron drifts by hours, so the window is
# wider than the cadence (15h between runs still leaves no market unseen) and
# the realised horizon is (21, 27]h instead of the bimodal 14h/34h of v2.0.
CRON_UTC = "17 */6 * * *"
WINDOW_HOURS = (12.0, 27.0)
NOMINAL_HORIZON_HOURS = 24

# Analysis plan: one confirmatory look at the primary population. Everything
# before that is labelled interim, so the daily CI is not a peeking machine.
TARGET_N_TRADES_PRIMARY = 200
POPULATIONS = ("primary", "secondary_at_decision", "all")

# Scan statuses and exit codes (one row per run in scans.jsonl):
#   0  ok | no_candidates             (one-sided books skipped only is still ok)
#   1  forecast_outage | clob_outage | evidence_degraded | gamma_unavailable | crashed
#   2  no_api_key                     (checked before any HTTP)
FAILED_SCAN_STATUSES = {"crashed", "gamma_unavailable", "no_api_key", "forecast_outage",
                        "clob_outage"}
_GAMMA_ATTEMPTS = 3
_GAMMA_BACKOFF_SECONDS = 2.0

SHADOW_NOTE = (
    "Shadow rows 2026-08-12..2026-09-24 were forecast with the Wikipedia channel dead "
    "(n_wiki_events == 0 on every row) under a $500k volume-to-date floor; they are "
    "reported as shadow only and are excluded from the primary population."
)


class GammaUnavailable(Exception):  # noqa: N818 -- named for what it reports
    """Gamma could not serve the first page: the universe is unknown, not empty."""


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


def _differs(registered, running) -> bool:
    if isinstance(registered, (int, float)) and isinstance(running, (int, float)):
        return abs(registered - running) > 1e-9
    if isinstance(registered, list) and isinstance(running, list):
        return len(registered) != len(running) or any(
            _differs(a, b) for a, b in zip(registered, running)
        )
    return registered != running


def validate_prereg(params: StrategyParams, args) -> None:
    """Refuse to scan when the committed pre-registration and the running config disagree.

    Everything checked here is part of what was registered; silently running
    something else would make the official result unfalsifiable.
    """
    reg = json.loads(PREREG_FILE.read_text())
    running = {
        "strategy.alpha": params.alpha,
        "strategy.threshold": params.threshold,
        "strategy.slippage": params.slippage,
        "strategy.min_price": params.min_price,
        "strategy.max_price": params.max_price,
        "strategy.max_divergence": params.max_divergence,
        "universe.scan_min_volume_to_date": args.min_volume,
        "universe.primary_min_final_volume": PRIMARY_MIN_FINAL_VOLUME,
        "schedule.window_hours": [args.min_hours, args.max_hours],
        "forecaster.model_alias": FORECAST_MODEL,
    }
    for key, have in running.items():
        registered = reg
        for part in key.split("."):
            registered = registered.get(part) if isinstance(registered, dict) else None
        if registered is None or _differs(registered, have):
            raise SystemExit(
                f"pre-registration drift: {key}: registered {registered!r}, running {have!r}. "
                "Refusing to scan."
            )


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
    # market_category returns the literal "other" rather than an empty string when
    # a market has no category, so `x or fallback` never fires -- without this
    # check every uncategorised market collapses into one giant "other" cluster,
    # which both corrupts the bootstrap and (via the 2-per-cluster cap in
    # fetch_live_candidates) silently throttles the scan to 2 such markets a day.
    slug = market_category(market)
    if not slug or slug == "other":
        return f"market:{market.get('id')}"
    slug = re.sub(r"-(exact-score|more-markets|correct-score|total-goals|btts).*$", "", slug)
    # Strike ladders appear both as `bitcoin-above-on-april-1` and with the strike
    # embedded, `bitcoin-above-125000-on-august-9`. Collapse both to `bitcoin-above`.
    slug = re.sub(r"-(above|below)(-[\d.]+)?-on-.*$", r"-\1", slug)
    return f"slug:{slug}"


def token_ids(market: dict) -> tuple[str, str] | None:
    """(yes_token, no_token) located by outcome name, never by index."""
    outcomes = [str(o).lower() for o in _parse_json_field(market.get("outcomes"))]
    tokens = _parse_json_field(market.get("clobTokenIds"))
    if len(outcomes) == 2 and len(tokens) == 2 and "yes" in outcomes and "no" in outcomes:
        return str(tokens[outcomes.index("yes")]), str(tokens[outcomes.index("no")])
    return None


def _num(x) -> float | None:
    """A price strictly inside (0, 1), or None.

    CLOB levels arrive as strings and error bodies carry no levels at all; a NaN
    or 0 price that slipped through would poison the fill math and the JSON log.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and 0.0 < v < 1.0 else None


def _levels(book, side: str) -> list[dict]:
    if not isinstance(book, dict):
        return []
    return [lvl for lvl in (book.get(side) or []) if isinstance(lvl, dict)]


def best_prices(book) -> tuple[float | None, float | None]:
    """(best_bid, best_ask). CLOB returns asks descending, so min/max, not [0]."""
    bids = [p for p in (_num(b.get("price")) for b in _levels(book, "bids")) if p is not None]
    asks = [p for p in (_num(a.get("price")) for a in _levels(book, "asks")) if p is not None]
    return (max(bids) if bids else None, min(asks) if asks else None)


def fill_price(book, notional: float) -> tuple[float | None, float]:
    """Volume-weighted ask to buy `notional` dollars, walking the book.

    v1 recorded the best ask regardless of size, so a 1-share offer could set the
    recorded fill for a $100 stake. Returns (vwap, filled_notional); vwap is None
    when the book cannot absorb anything.
    """
    levels = []
    for a in _levels(book, "asks"):
        price = _num(a.get("price"))
        try:
            size = float(a["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if price is not None and math.isfinite(size):
            levels.append((price, size))
    levels.sort(key=lambda x: x[0])
    spent = shares = 0.0
    for price, size in levels:
        room = notional - spent
        if room <= 0:
            break
        take = min(size, room / price)
        spent += take * price
        shares += take
    if shares <= 0:
        return None, 0.0
    return spent / shares, spent


# --- JSONL plumbing -----------------------------------------------------------
# allow_nan=False everywhere: a NaN token makes the file unreadable to the
# dashboard's JSON.parse and to strict parsers, and silently hides a bad price.


def _dumps(row: dict) -> str:
    return json.dumps(row, allow_nan=False)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(_dumps(row) + "\n")
        f.flush()


def _load_decisions() -> list[dict]:
    return _load_jsonl(DECISIONS_FILE)


def _rewrite_decisions(decisions: list[dict]) -> None:
    tmp = DECISIONS_FILE.with_suffix(f".jsonl.tmp{os.getpid()}")
    with tmp.open("w") as f:
        for d in decisions:
            f.write(_dumps(d) + "\n")
    tmp.replace(DECISIONS_FILE)


def _last_scan() -> dict | None:
    scans = _load_jsonl(SCANS_FILE)
    return scans[-1] if scans else None


def current_mode() -> str:
    """'official' once a pre-registration is committed, else 'shadow'."""
    return "official" if PREREG_FILE.exists() else "shadow"


# --- scan: candidates ---------------------------------------------------------


async def _gamma_page(client: httpx.AsyncClient, params: dict, offset: int) -> list | None:
    """One page of Gamma markets, or None after _GAMMA_ATTEMPTS failures."""
    for attempt in range(_GAMMA_ATTEMPTS):
        try:
            page_params = {**params, "offset": str(offset)}
            resp = await client.get(f"{GAMMA_API}/markets", params=page_params)
            if resp.status_code != 200:
                raise ValueError(f"status {resp.status_code}")
            batch = resp.json()
            if not isinstance(batch, list):
                raise ValueError("body is not a list")
            return batch
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("scan.gamma_error", offset=offset, attempt=attempt, error=str(e)[:120])
            if attempt + 1 < _GAMMA_ATTEMPTS:
                await asyncio.sleep(_GAMMA_BACKOFF_SECONDS * (attempt + 1))
    return None


def _admit(m: dict, args, now: datetime, lo: datetime, hi: datetime) -> float | None:
    """Hours to close when the market is in the scan universe, else None.

    Raises TypeError/ValueError on a malformed record; the caller skips those.
    """
    question = (m.get("question") or "").strip()
    if not question or any(kw in question.lower() for kw in EXCLUDE_KEYWORDS):
        return None
    if token_ids(m) is None or not m.get("acceptingOrders", True):
        return None
    if float(m.get("volumeNum") or 0) < args.min_volume:
        return None
    end_dt = _parse_dt(m.get("endDate") or "")
    created_dt = _parse_dt(m.get("createdAt") or m.get("startDate") or "")
    if not end_dt or not created_dt or not (lo <= end_dt <= hi):
        return None
    if (now - created_dt).total_seconds() < MIN_AGE_DAYS * 86400:
        return None
    return (end_dt - now).total_seconds() / 3600


def _open_cluster_counts(logged: list[dict], now: datetime) -> dict[str, int]:
    """Clusters already carrying logged markets that have not closed yet.

    The per-cluster cap is on CONCURRENT exposure across runs, not per scan:
    with four scans a day a daily strike ladder would otherwise get 8 slots.
    """
    counts: dict[str, int] = {}
    for d in logged:
        if d.get("status") in RETRYABLE_STATUSES or not d.get("cluster_key"):
            continue  # re-offered below, and counted when it is
        end_dt = _parse_dt(d.get("end_date") or "")
        if end_dt and end_dt > now:
            counts[d["cluster_key"]] = counts.get(d["cluster_key"], 0) + 1
    return counts


async def fetch_live_candidates(
    client: httpx.AsyncClient, args, now: datetime
) -> tuple[list[dict], dict]:
    """Markets closing inside the window that we have not already decided on.

    Returns (candidates, meta); meta is the Gamma part of the scan record.
    Raises GammaUnavailable when even the first page cannot be fetched: an
    unknown universe must not be reported as an empty one (exit 0, "healthy").
    """
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
    meta = {"window_lo": args.min_hours, "window_hi": args.max_hours,
            "n_raw": 0, "gamma_pages": 0, "gamma_truncated": False}
    raw: list[dict] = []
    seen_ids: set[str] = set()
    for page in range(args.max_pages):
        batch = await _gamma_page(client, params_base, page * 100)
        if batch is None:
            if page == 0:
                raise GammaUnavailable(f"Gamma page 0 failed after {_GAMMA_ATTEMPTS} attempts")
            meta["gamma_truncated"] = True  # the long tail is missing; say so, carry on
            break
        meta["gamma_pages"] += 1
        if not batch:
            break
        for m in batch:
            mid = str(m.get("id")) if isinstance(m, dict) else None
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                raw.append(m)
    meta["n_raw"] = len(raw)

    logged = _load_decisions()
    already = {d.get("market_id") for d in logged if d.get("status") not in RETRYABLE_STATUSES}
    seen_clusters = _open_cluster_counts(logged, now)
    out: list[dict] = []
    for m in raw:
        try:
            hours_to_close = _admit(m, args, now, lo, hi)
        except (TypeError, ValueError):
            continue  # one malformed record is not worth aborting the scan over
        if hours_to_close is None or str(m.get("id")) in already:
            continue
        ckey = cluster_key(m)
        if seen_clusters.get(ckey, 0) >= MAX_OPEN_PER_CLUSTER:
            continue
        seen_clusters[ckey] = seen_clusters.get(ckey, 0) + 1
        m["_cluster_key"] = ckey
        m["_hours_to_close"] = hours_to_close
        out.append(m)
        if len(out) >= args.max_markets:
            break
    return out, meta


# --- scan: run ----------------------------------------------------------------


def new_scan_record(now: datetime, args) -> dict:
    """The scans.jsonl row, pessimistic by default: only a finished scan clears it."""
    return {
        "ts": now.isoformat(), "mode": current_mode(), "exit_code": 1, "status": "crashed",
        "window_lo": args.min_hours, "window_hi": args.max_hours,
        "scan_min_volume": args.min_volume,
        "n_raw": 0, "gamma_pages": 0, "gamma_truncated": False, "n_candidates": 0,
        "n_illiquid": 0, "n_book_failed": 0, "n_market_failed": 0,
        "n_forecast_attempted": 0, "n_forecast_ok": 0, "n_forecast_failed": 0,
        "n_open": 0, "n_no_trade": 0,
        "wiki_days_ok": 0, "wiki_days_cached": 0, "wiki_days_failed": 0,
        "coverage_gap_hours": 0.0, "duration_s": None, "error": None,
    }


def _coverage_gap_hours(record: dict, previous: dict | None) -> float:
    """Hours of close times no scan looked at, between the previous window and this one.

    Cron drift is hours, not minutes; with a 6h cadence and a [12h, 27h] window a
    gap opens only when runs are more than 15h apart. Recorded, not fatal.
    """
    if not previous:
        return 0.0
    try:
        prev_hi = datetime.fromisoformat(previous["ts"])
        prev_hi += timedelta(hours=float(previous["window_hi"]))
        this_lo = datetime.fromisoformat(record["ts"])
        this_lo += timedelta(hours=float(record["window_lo"]))
    except (KeyError, TypeError, ValueError):
        return 0.0
    return round(max(0.0, (this_lo - prev_hi).total_seconds() / 3600), 2)


def _prune_wiki_cache(now: datetime) -> None:
    """Drop day pages past the retention window so the committed cache stays small."""
    cutoff = (now - timedelta(days=WIKI_CACHE_RETENTION_DAYS)).date()
    for path in WIKI_CACHE.glob("*.json"):
        try:
            day = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if day < cutoff:
            path.unlink()


def _scan_verdict(record: dict) -> tuple[str, str | None]:
    """(status, error message) for a scan that got as far as the order books."""
    attempted, ok = record["n_forecast_attempted"], record["n_forecast_ok"]
    if attempted and not ok:
        # v1's fatal flaw, guarded on ATTEMPTS: a scan where every candidate had
        # a one-sided book attempted nothing and is not an outage.
        return "forecast_outage", "candidates found but not one forecast succeeded"
    if not attempted and record["n_book_failed"]:
        return "clob_outage", "candidates found but every order-book request failed"
    if record["wiki_days_ok"] < EVIDENCE_OK_MIN_DAYS:
        days = f"{record['wiki_days_ok']}/{WIKI_LOOKBACK_DAYS}"
        return ("evidence_degraded",
                f"only {days} Wikipedia day pages; rows logged with evidence_ok=false")
    return "ok", None


async def _scan(args, now: datetime, record: dict) -> int:
    """The scan proper; `record` is the scans.jsonl row, filled in as we go."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Before any HTTP: without a key every forecast fails, which would look like
    # a provider outage and spend Gamma/CLOB/Wikipedia quota for nothing.
    if not os.environ.get("OPENAI_API_KEY"):
        record.update(status="no_api_key", error="OPENAI_API_KEY is not set")
        print("ERROR: no_api_key: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2
    params = load_frozen_params()
    if PREREG_FILE.exists():
        validate_prereg(params, args)
    mode = record["mode"]
    as_of = now.date().isoformat()
    _prune_wiki_cache(now)
    rows: list[dict] = []

    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        try:
            candidates, meta = await fetch_live_candidates(client, args, now)
        except GammaUnavailable as e:
            record.update(status="gamma_unavailable", error=str(e))
            print(f"ERROR: gamma_unavailable: {e}", file=sys.stderr)
            return 1
        record.update(meta, n_candidates=len(candidates))
        print(f"{len(candidates)} candidates closing in {args.min_hours}-{args.max_hours}h "
              f"with volume >= ${args.min_volume:,.0f}  [mode={mode}]")

        # One prefetch per scan, shared by every candidate (v2.0 refetched ten
        # pages per market). Done even with no candidates so the cache stays
        # warm and the health record still says whether Wikipedia answered.
        events_by_date, wiki = await fetch_days_before(
            now, client, WIKI_CACHE, lookback_days=WIKI_LOOKBACK_DAYS,
        )
        record.update(wiki_days_ok=wiki["days_ok"], wiki_days_cached=wiki["days_cached"],
                      wiki_days_failed=wiki["days_failed"])
        print(f"wiki: {wiki['days_ok']}/{WIKI_LOOKBACK_DAYS} day pages "
              f"({wiki['days_cached']} cached, {wiki['days_fetched']} fetched, "
              f"{wiki['days_failed']} failed)")
        if not candidates:
            # Not an error: a legitimately empty pool is expected at this floor.
            record["status"] = "no_candidates"
            print("no candidates; nothing to forecast")
            return 0
        evidence_ok = wiki["days_ok"] >= EVIDENCE_OK_MIN_DAYS

        forecast_slots = asyncio.Semaphore(args.concurrency)
        book_slots = asyncio.Semaphore(8)
        lock = asyncio.Lock()

        async def log_row(row: dict) -> None:
            # Written the moment it exists: v2.0 buffered rows until after the
            # gather, so one crash lost the whole run's decisions.
            async with lock:
                _append_jsonl(DECISIONS_FILE, row)
                rows.append(row)

        async def get_book(token: str):
            async with book_slots:
                resp = await client.get(f"{CLOB_API}/book", params={"token_id": token})
            # The CLOB answers unknown/retired tokens with error JSON, which v2.0
            # parsed as an empty book and logged as "illiquid".
            if resp.status_code != 200:
                raise RuntimeError(f"clob {resp.status_code}: {resp.text[:80]}")
            return resp.json()

        async def process_market(market_id: str, m: dict) -> None:
            yes_tok, no_tok = token_ids(m)
            try:
                yes_book, no_book = await get_book(yes_tok), await get_book(no_tok)
            except Exception as e:  # noqa: BLE001
                logger.warning("scan.book_failed", market_id=market_id, error=str(e)[:120])
                record["n_book_failed"] += 1
                return
            if not isinstance(yes_book, dict) or not isinstance(no_book, dict):
                raise TypeError("clob returned a non-object book")
            yes_bid, yes_ask = best_prices(yes_book)
            no_bid, no_ask = best_prices(no_book)
            if yes_bid is None or yes_ask is None or no_ask is None:
                logger.info("scan.illiquid_book", market_id=market_id)
                record["n_illiquid"] += 1
                return
            p_mid = (yes_bid + yes_ask) / 2
            wiki_evidence = relevant_events(m["question"], events_by_date)

            record["n_forecast_attempted"] += 1
            async with forecast_slots:
                f = await forecast(
                    question=m["question"],
                    as_of=as_of,
                    headlines=[],  # backtest ran --skip-gdelt; replica must match
                    world_events=wiki_evidence,
                    description=(m.get("description") or "")[:500],
                )
            if (f.reasoning_tokens or 0) > 0:
                # Effort "none" must bill zero reasoning tokens; anything else
                # means the provider changed the default under the registration.
                logger.warning("scan.reasoning_tokens_nonzero", market_id=market_id,
                               served_model=f.served_model, reasoning_tokens=f.reasoning_tokens)

            base = {
                "decision_id": market_id,
                "ts_decision": now.isoformat(),
                "scan_ts": now.isoformat(),
                "mode": mode,
                "market_id": market_id,
                "cluster_key": m.get("_cluster_key") or cluster_key(m),
                "question": m["question"],
                "category": market_category(m),
                "end_date": m.get("endDate"),
                "hours_to_close": m.get("_hours_to_close"),
                "volume": float(m.get("volumeNum") or 0),
                "model": FORECAST_MODEL,
                "served_model": f.served_model,
                "usage": {"output_tokens": f.output_tokens,
                          "reasoning_tokens": f.reasoning_tokens},
                "p_model": f.p_model,
                "evidence_strength": f.evidence_strength,
                "reasoning": f.reasoning,
                "raw_response": f.raw_response,  # v1 dropped this; the dashboard needs it
                "n_wiki_events": len(wiki_evidence),
                "wiki_days_ok": wiki["days_ok"],
                "evidence_ok": evidence_ok,
                "book": {"yes_bid": yes_bid, "yes_ask": yes_ask,
                         "no_bid": no_bid, "no_ask": no_ask},
                "p_market_mid": p_mid,
                "params": {
                    **params.__dict__,
                    "min_volume": args.min_volume,
                    "scan_min_volume": args.min_volume,
                    "primary_min_final_volume": PRIMARY_MIN_FINAL_VOLUME,
                    "window_hours": [args.min_hours, args.max_hours],
                },
            }

            # Invariant: a failed forecast must never reach the strategy layer.
            # BetInput has no `failed` field, so a sentinel p_model=0.5 against a
            # market at 0.20 is edge 0.30 -- inside the band -- and would
            # manufacture a confident bogus trade.
            if f.failed:
                record["n_forecast_failed"] += 1
                base.update({"status": "forecast_failed", "side": None, "stake": 0.0})
                await log_row(base)
                return
            record["n_forecast_ok"] += 1

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
                record["n_no_trade"] += 1
                base.update({"status": "no_trade", "side": None, "stake": 0.0,
                             "edge": f.p_model - p_mid})
                await log_row(base)
                return

            # Replica cost (comparable to the backtest) and the executable cost
            # (depth-walked real ask) are logged side by side.
            side_book = yes_book if trade.side == "yes" else no_book
            vwap, filled = fill_price(side_book, STAKE)
            record["n_open"] += 1
            base.update({
                "status": "open",
                "side": trade.side,
                "edge": trade.edge,
                "entry_cost": trade.entry_cost,          # mid + slippage, replica
                "executable_cost": vwap,                 # depth-walked ask
                "executable_notional": filled,
                "stake": STAKE,
            })
            await log_row(base)

        async def process(m: dict) -> None:
            market_id = str(m.get("id"))
            try:
                await process_market(market_id, m)
            except Exception as e:  # noqa: BLE001
                # One bad market must not take the other 149 down with it.
                logger.warning("scan.market_failed", market_id=market_id,
                               error=f"{type(e).__name__}: {e}"[:160])
                record["n_market_failed"] += 1

        # process() already isolates failures; return_exceptions is the belt to
        # that brace, so a surprise in one task can never cancel its siblings.
        await asyncio.gather(*(process(m) for m in candidates), return_exceptions=True)

    print(f"logged {len(rows)} decisions: {record['n_open']} trades, "
          f"{record['n_no_trade']} no-trade, {record['n_forecast_failed']} failed  "
          f"(skipped: {record['n_illiquid']} illiquid, {record['n_book_failed']} book failures, "
          f"{record['n_market_failed']} market errors)")
    status, message = _scan_verdict(record)
    record["status"] = status
    if message is None:
        return 0
    record["error"] = message
    print(f"ERROR: {status}: {message}", file=sys.stderr)
    return 1


async def stage_scan(args, record: dict | None = None) -> int:
    """Run one scan and append its health record whatever happens."""
    now = datetime.now(UTC)
    record = record if record is not None else new_scan_record(now, args)
    record["ts"] = now.isoformat()
    started = time.monotonic()
    try:
        record["exit_code"] = await _scan(args, now, record)
    except BaseException as e:
        record.update(status="crashed", exit_code=1, error=repr(e)[:300])
        raise
    finally:
        record["coverage_gap_hours"] = _coverage_gap_hours(record, _last_scan())
        record["duration_s"] = round(time.monotonic() - started, 1)
        _append_jsonl(SCANS_FILE, record)
    return record["exit_code"]


def run_scan(args) -> int:
    """asyncio.run(stage_scan) plus the last line of defence for the health record.

    stage_scan records its own crashes; this catches the ones its finally cannot
    see (event-loop setup and teardown), so a dead scan is never invisible.
    """
    record = new_scan_record(datetime.now(UTC), args)
    try:
        return asyncio.run(stage_scan(args, record))
    except BaseException as e:
        if record["duration_s"] is None:
            record.update(status="crashed", exit_code=1, error=repr(e)[:300])
            _append_jsonl(SCANS_FILE, record)
        raise


# --- settle -------------------------------------------------------------------


def _settle_due(d: dict, now: datetime) -> bool:
    """Open rows always; no_trade/forecast_failed rows once their market has closed, once.

    Resolving the rows we did NOT trade is what makes calibration over every
    scanned market measurable; the shadow period never had it.
    """
    status = d.get("status")
    if status == "open":
        return True
    if status not in ("no_trade", "forecast_failed") or "outcome_yes" in d:
        return False
    end_dt = _parse_dt(d.get("end_date") or "")
    return bool(end_dt and end_dt < now)


async def _fetch_market(client: httpx.AsyncClient, market_id: str) -> dict | None:
    """The Gamma market record, or None (logged) when it cannot be trusted."""
    try:
        resp = await client.get(f"{GAMMA_API}/markets/{market_id}")
        m = resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("settle.fetch_failed", market_id=market_id, error=str(e)[:120])
        return None
    if isinstance(m, list):
        m = m[0] if m else None
    # A 404/422 body or an error envelope has no `closed` key. Reading it as
    # "not closed yet" would be harmless; reading it as a resolution would not.
    if not isinstance(m, dict) or "closed" not in m:
        logger.warning("settle.market_unavailable", market_id=market_id,
                       status=resp.status_code, body=resp.text[:80])
        return None
    return m


def _mark_resolved(d: dict, m: dict, outcome: bool | None, ts: str) -> None:
    volume_final = float(m.get("volumeNum") or 0)
    d.update({"outcome_yes": outcome, "ts_resolved": ts, "volume_final": volume_final})
    if outcome is None:
        d["void_reason"] = {"outcomePrices": m.get("outcomePrices"),
                            "umaResolutionStatus": m.get("umaResolutionStatus")}


def _settle_open(d: dict, outcome: bool | None, ts: str) -> None:
    if outcome is None:
        d.update({"status": "void", "won": None, "pnl": 0.0, "pnl_per_dollar": 0.0,
                  "ts_settled": ts})
        return
    won = outcome if d["side"] == "yes" else (not outcome)
    cost = d["entry_cost"]
    pnl_per_dollar = (1.0 - cost) / cost if won else -1.0
    d.update({
        "status": "settled",
        "won": won,
        "pnl": d["stake"] * pnl_per_dollar,
        "pnl_per_dollar": pnl_per_dollar,
        "ts_settled": ts,
    })
    # Executable P&L is on what the book actually sold us, not the nominal stake:
    # a $20 fill cannot win or lose $100.
    notional = float(d.get("executable_notional") or 0.0)
    cost_exec = d.get("executable_cost")
    d["executable_fill_ratio"] = notional / d["stake"] if d["stake"] else 0.0
    if cost_exec and notional > 0:
        d["pnl_executable"] = notional * ((1.0 - cost_exec) / cost_exec) if won else -notional
    else:
        d["pnl_executable"] = 0.0


async def stage_settle(args) -> int:
    decisions = _load_decisions()
    now = datetime.now(UTC)
    due = [d for d in decisions if _settle_due(d, now)]
    if not due:
        print("nothing to settle")
        return 0
    n_open = sum(1 for d in due if d.get("status") == "open")
    counts = {"settled": 0, "void": 0, "resolved": 0}
    slots = asyncio.Semaphore(4)

    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:

        async def settle_one(d: dict) -> None:
            async with slots:
                m = await _fetch_market(client, d["market_id"])
            # Resolved markets still report active=true; `closed` is the only gate.
            if not m or not m.get("closed"):
                return
            outcome = parse_resolution(m)
            _mark_resolved(d, m, outcome, now.isoformat())
            if d.get("status") != "open":
                counts["resolved"] += 1
                return
            _settle_open(d, outcome, now.isoformat())
            counts["void" if outcome is None else "settled"] += 1

        async def guarded(d: dict) -> None:
            try:
                await settle_one(d)
            except Exception as e:  # noqa: BLE001
                logger.warning("settle.market_failed", market_id=d.get("market_id"),
                               error=f"{type(e).__name__}: {e}"[:160])

        await asyncio.gather(*(guarded(d) for d in due))

    _rewrite_decisions(decisions)
    print(f"settled {counts['settled']}, voided {counts['void']} of {n_open} open; "
          f"resolved {counts['resolved']} non-trade rows")
    return 0


# --- report -------------------------------------------------------------------


def _settled(decisions: list[dict], mode: str, population: str) -> list[dict]:
    """Settled rows of one mode in one of the three analysis populations."""
    out = []
    for d in decisions:
        if d.get("status") != "settled" or d.get("mode") != mode:
            continue
        if population == "primary" and not (
            d.get("evidence_ok") and float(d.get("volume_final") or 0) >= PRIMARY_MIN_FINAL_VOLUME
        ):
            continue
        if population == "secondary_at_decision" and float(d.get("volume") or 0) < MIN_VOLUME:
            continue
        out.append(d)
    return out


def _report(decisions: list[dict], mode: str, population: str) -> dict:
    rows = _settled(decisions, mode, population)
    if not rows:
        return {"population": population, "n_settled": 0, "n_trades": 0}
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
    rep = simulate(bets, load_frozen_params())
    return {
        "population": population,
        "n_settled": len(rows),
        "n_trades": rep.n_trades,
        "total_pnl": rep.total_pnl,
        "roi": rep.roi,
        "win_rate": rep.win_rate,
        "roi_ci_low": rep.roi_ci_low,
        "roi_ci_high": rep.roi_ci_high,
        # Traded subset only; calibration over everything scanned is `calibration`.
        "brier_blend_traded": rep.brier_blend_traded,
        "brier_market_traded": rep.brier_market_traded,
        "max_drawdown": rep.max_drawdown,
        "n_clusters": len({d.get("cluster_key") for d in rows}),
    }


def _calibration(decisions: list[dict], mode: str, params: StrategyParams) -> dict:
    """Brier of model vs market over every resolved scanned market, traded or not."""
    rows = [d for d in decisions
            if d.get("mode") == mode and d.get("outcome_yes") is not None
            and d.get("status") != "forecast_failed"]
    if not rows:
        return {"n_resolved": 0}
    triples = [(d["p_model"], d["p_market_mid"], bool(d["outcome_yes"])) for d in rows]
    in_band = [t for t in triples if params.min_price <= t[1] <= params.max_price]
    return {
        "n_resolved": len(rows),
        "brier_model": _brier([(p, y) for p, _, y in triples]),
        "brier_market": _brier([(q, y) for _, q, y in triples]),
        "n_in_band": len(in_band),
        "brier_model_in_band": _brier([(p, y) for p, _, y in in_band]),
        "brier_market_in_band": _brier([(q, y) for _, q, y in in_band]),
    }


def _mode_block(decisions: list[dict], mode: str, params: StrategyParams) -> dict:
    block: dict = {p: _report(decisions, mode, p) for p in POPULATIONS}
    block["calibration"] = _calibration(decisions, mode, params)
    block["n_open"] = sum(1 for d in decisions
                          if d.get("mode") == mode and d.get("status") == "open")
    return block


def _within(ts: str | None, now: datetime, days: int) -> bool:
    dt = _parse_dt(ts or "")
    return bool(dt and dt >= now - timedelta(days=days))


def _scan_health(scans: list[dict], decisions: list[dict], now: datetime) -> dict | None:
    """What the scan records say about the pipeline, independent of any P&L."""
    if not scans:
        return None
    last = scans[-1]
    last_ts = _parse_dt(last.get("ts") or "")
    recent = [s for s in scans if _within(s.get("ts"), now, days=7)]
    recent_decisions = [d for d in decisions if _within(d.get("ts_decision"), now, days=14)]
    return {
        "last_scan_ts": last.get("ts"),
        "hours_since_last_scan": (round((now - last_ts).total_seconds() / 3600, 2)
                                  if last_ts else None),
        "last_status": last.get("status"),
        "last_exit_code": last.get("exit_code"),
        "runs_last_7d": len(recent),
        "failed_runs_last_7d": sum(1 for s in recent if s.get("exit_code") != 0),
        "last_run": last,
        "evidence": {
            "last_wiki_days_ok": last.get("wiki_days_ok"),
            "decisions_last_14d": len(recent_decisions),
            "decisions_with_evidence_last_14d": sum(
                1 for d in recent_decisions if d.get("evidence_ok")
            ),
        },
    }


def _health(scan_health: dict | None) -> str:
    """One word the dashboard shows before any number: liveness outranks performance."""
    if scan_health is None:
        return "unknown"
    if scan_health["last_status"] in FAILED_SCAN_STATUSES:
        return "failed"
    hours = scan_health["hours_since_last_scan"]
    stale = hours is None or hours > 24
    forecast_failures = (scan_health["last_run"].get("n_forecast_failed") or 0) > 0
    if scan_health["last_status"] == "evidence_degraded" or forecast_failures or stale:
        return "degraded"
    return "ok"


def _baseline() -> dict:
    """The like-for-like backtest reference, read from the artifact when it exists."""
    n_trades, roi, ci = 124, 0.318, [0.069, 0.586]
    if LUNA_PRIMARY_EVAL.exists():
        test = json.loads(LUNA_PRIMARY_EVAL.read_text()).get("test", {})
        n_trades = test.get("n_trades", n_trades)
        roi = test.get("roi", roi)
        ci = test.get("roi_ci_90", ci)
    return {
        "source": "data/backtest/evaluation.gpt-5.6-luna.primary.json",
        "test_roi": roi,
        "test_ci_90": ci,
        "n_trades": n_trades,
        "note": ("gpt-5.6-luna, frozen claude params, final volume >= $500k, event-clustered "
                 "CI: the like-for-like reference for the primary population"),
        "claude_reference": {
            "source": "data/baselines/claude-fable-5-2026-06-10",
            "test_roi": 0.3052,
            "test_ci_90": [0.0514, 0.5629],
        },
    }


def _health_line(summary: dict) -> str:
    sh = summary["scan_health"]
    if sh is None:
        return f"health: {summary['health']}  (no scans.jsonl yet)"
    run = sh["last_run"]
    hours = sh["hours_since_last_scan"]
    ago = f"{hours:.1f}h ago" if hours is not None else "unknown age"
    return (f"health: {summary['health']}  last scan {str(sh['last_scan_ts'])[:16]} ({ago}) "
            f"status={sh['last_status']} exit={sh['last_exit_code']}  "
            f"wiki {run.get('wiki_days_ok')}/{WIKI_LOOKBACK_DAYS} day pages "
            f"({run.get('wiki_days_cached')} cached, {run.get('wiki_days_failed')} failed)  "
            f"runs 7d={sh['runs_last_7d']} failed={sh['failed_runs_last_7d']}")


def _print_report(summary: dict) -> None:
    print(f"=== Forward test ({summary['generated_at'][:10]}, mode={summary['mode']}) ===")
    print(f"  {_health_line(summary)}")
    print(f"  decisions={summary['n_decisions']}  {summary['counts']}")
    for mode in ("shadow", "official"):
        for population in POPULATIONS:
            r = summary[mode][population]
            label = f"{mode}/{population}"
            if r["n_trades"]:
                print(f"  {label:<30} settled={r['n_settled']:>4}  P&L=${r['total_pnl']:>+9,.0f}  "
                      f"ROI={r['roi']*100:>+6.1f}%  CI=[{r['roi_ci_low']*100:+.1f}%, "
                      f"{r['roi_ci_high']*100:+.1f}%]  clusters={r['n_clusters']}")
            else:
                print(f"  {label:<30} settled={r['n_settled']:>4}  no trades yet")
    print(f"  -> {SUMMARY_FILE}")


def stage_report(args) -> int:
    now = datetime.now(UTC)
    decisions = _load_decisions()
    params = load_frozen_params()
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.get("status", "?")] = counts.get(d.get("status", "?"), 0) + 1
    scan_health = _scan_health(_load_jsonl(SCANS_FILE), decisions, now)

    summary = {
        "generated_at": now.isoformat(),
        "mode": current_mode(),
        "model": FORECAST_MODEL,
        "min_volume": SCAN_MIN_VOLUME,
        "universe": {
            "scan_min_volume_to_date": SCAN_MIN_VOLUME,
            "primary_min_final_volume": PRIMARY_MIN_FINAL_VOLUME,
            "secondary_min_volume_at_decision": MIN_VOLUME,
            "min_age_days": MIN_AGE_DAYS,
            "max_open_per_cluster": MAX_OPEN_PER_CLUSTER,
            "evidence_ok_min_days": EVIDENCE_OK_MIN_DAYS,
        },
        "schedule": {
            "cron_utc": CRON_UTC,
            "window_hours": list(WINDOW_HOURS),
            "nominal_horizon_hours": NOMINAL_HORIZON_HOURS,
        },
        "analysis_plan": {
            "target_n_trades_primary": TARGET_N_TRADES_PRIMARY,
            "look": ("one confirmatory look when the primary population reaches "
                     f"{TARGET_N_TRADES_PRIMARY} settled trades"),
            "interim": True,
        },
        "last_decision_ts": max((d.get("ts_decision", "") for d in decisions), default=""),
        "counts": counts,
        "n_decisions": len(decisions),
        "shadow": _mode_block(decisions, "shadow", params),
        "official": _mode_block(decisions, "official", params),
        "scan_health": scan_health,
        "health": _health(scan_health),
        "baseline": _baseline(),
        "shadow_note": SHADOW_NOTE,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, allow_nan=False))
    _print_report(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cassandra forward paper-trading test v2")
    sub = p.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--min-hours", type=float, default=WINDOW_HOURS[0])
    s.add_argument("--max-hours", type=float, default=WINDOW_HOURS[1])
    s.add_argument("--min-volume", type=float, default=SCAN_MIN_VOLUME)
    s.add_argument("--max-markets", type=int, default=150)
    s.add_argument("--max-pages", type=int, default=6)
    s.add_argument("--concurrency", type=int, default=4)

    sub.add_parser("settle")
    sub.add_parser("report")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.stage == "scan":
        return run_scan(args)
    if args.stage == "settle":
        return asyncio.run(stage_settle(args))
    return stage_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
