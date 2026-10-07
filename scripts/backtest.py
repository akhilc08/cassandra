"""Time-machine backtest — leak-free profitability evaluation against real prices.

Methodology:
  - Universe: resolved Polymarket markets that closed AFTER the model's
    knowledge cutoff (Feb 2026+), volume-filtered, max 2 per event.
  - Prediction time T = scheduled close (endDate) minus --hours-before.
    Only markets still trading at T are eligible — an implementable strategy.
  - Market price at T comes from the CLOB price-history API (the price a
    trader would actually have paid), NOT the settlement price.
  - Evidence = GDELT headlines seen strictly before T. No NewsAPI, no
    undated news, no lookahead.
  - Forecast = an LLM that is BLIND to the market price (shown it, models
    anchor to it); shrinkage toward the market happens in the strategy layer.
  - P&L: blend/threshold strategy tuned on the train window (closes before
    --split-date), frozen and evaluated out-of-sample on the test window.

Stages are cached to data/backtest/*.jsonl and resumable:
    python scripts/backtest.py collect  --target 250
    python scripts/backtest.py predict  --concurrency 4
    python scripts/backtest.py evaluate --split-date 2026-05-01
    python scripts/backtest.py all
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oracle.agents.forecaster import FORECAST_MODEL, forecast
from oracle.evaluation.pnl import (
    BetInput,
    StrategyParams,
    grid_search,
    simulate,
    simulate_rule,
)
from oracle.ingestion.gdelt_client import fetch_news_before
from oracle.ingestion.price_history import fetch_price_series, last_price_at_or_before
from oracle.ingestion.wiki_events import fetch_events_before

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_DIR = Path(__file__).parent.parent / "data" / "backtest"
MARKETS_FILE = DATA_DIR / "markets.jsonl"
PREDICTIONS_FILE = DATA_DIR / "predictions.jsonl"


PREDICTIONS_TAG = ""  # --predictions-tag: keeps ablation runs out of the main file


def predictions_file(provider: str) -> Path:
    """Per-provider prediction files.

    The claude path keeps the original filename so the archived baseline stays
    reproducible; a different provider must never append into it, or the two
    models' forecasts silently interleave in one file.
    """
    tag = f".{PREDICTIONS_TAG}" if PREDICTIONS_TAG else ""
    if provider == "claude" and not tag:
        return PREDICTIONS_FILE
    if provider == "claude":
        return DATA_DIR / f"predictions{tag}.jsonl"
    from oracle.agents.forecaster_openai import FORECAST_MODEL as OPENAI_MODEL
    return DATA_DIR / f"predictions.{OPENAI_MODEL}{tag}.jsonl"


def get_forecaster(provider: str):
    """(forecast_fn, model_id) for the requested provider."""
    if provider == "claude":
        from oracle.agents.forecaster import FORECAST_MODEL as m, forecast as fn
        return fn, m
    from oracle.agents.forecaster_openai import FORECAST_MODEL as m, forecast as fn
    return fn, m
EVALUATION_FILE = DATA_DIR / "evaluation.json"

# Markets a news-driven forecaster cannot trade: intraday price noise,
# token-launch FDV bets with no public information flow.
EXCLUDE_KEYWORDS = [
    "up or down", "fdv above", "fdv below", "after launch", "airdrop",
    "token by", "launch a token", "days after launch",
]

USER_AGENT = "cassandra-backtest/1.0"


def _parse_json_field(raw) -> list:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return raw or []


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace(" ", "T").replace("Z", "+00:00").replace("+00:00:00", "+00:00"))
    except ValueError:
        try:
            return datetime.fromisoformat(s.replace(" ", "T").replace("+00", "+00:00"))
        except ValueError:
            return None


def parse_resolution(market: dict) -> bool | None:
    """True = YES, False = NO, None = ambiguous/unresolved.

    outcomePrices[i] is the settlement price of outcomes[i]; the YES outcome
    is located by name rather than assumed to be first.
    """
    outcomes = [str(o).lower() for o in _parse_json_field(market.get("outcomes"))]
    prices = _parse_json_field(market.get("outcomePrices"))
    if len(outcomes) != 2 or len(prices) != 2 or "yes" not in outcomes:
        return None
    yes_idx = outcomes.index("yes")
    try:
        p_yes, p_no = float(prices[yes_idx]), float(prices[1 - yes_idx])
    except (ValueError, TypeError):
        return None
    if p_yes == 1.0 and p_no == 0.0:
        return True
    if p_yes == 0.0 and p_no == 1.0:
        return False
    return None


def yes_token_id(market: dict) -> str | None:
    outcomes = [str(o).lower() for o in _parse_json_field(market.get("outcomes"))]
    tokens = _parse_json_field(market.get("clobTokenIds"))
    if len(outcomes) == 2 and len(tokens) == 2 and "yes" in outcomes:
        return str(tokens[outcomes.index("yes")])
    return None


def market_category(market: dict) -> str:
    events = _parse_json_field(market.get("events"))
    if events and isinstance(events[0], dict):
        for key in ("category", "ticker", "slug"):
            v = events[0].get(key)
            if v:
                return str(v)[:40]
    return str(market.get("category") or "other")[:40]


def cluster_key(market: dict) -> str:
    """Stable grouping key for the bootstrap, for both live Gamma payloads and
    stored backtest records.

    The 959-market collect stored event_id == market_id for every row, so the
    "cluster bootstrap" resampled 959 clusters over 959 markets and its CI was
    too narrow: ten correlated "BTC below strike" bets, and the several markets
    on a single football match, each counted as independent observations.
    """
    events = _parse_json_field(market.get("events"))
    if events and isinstance(events[0], dict) and events[0].get("id"):
        return f"event:{events[0]['id']}"
    # market_category returns the literal "other" rather than "" when a market has
    # no category, so `x or fallback` never fires -- without this check every
    # uncategorised market collapses into one giant cluster.
    slug = market_category(market)
    if not slug or slug == "other":
        return f"market:{market.get('id') or market.get('market_id')}"
    slug = re.sub(r"-(exact-score|more-markets|correct-score|total-goals|btts).*$", "", slug)
    # Ladders appear as `bitcoin-above-on-april-1` and with the strike embedded,
    # `bitcoin-above-125000-on-august-9`. Collapse both to `bitcoin-above`.
    slug = re.sub(r"-(above|below)(-[\d.]+)?-on-.*$", r"-\1", slug)
    return f"slug:{slug}"


async def fetch_candidate_markets(
    client: httpx.AsyncClient,
    start: datetime,
    end: datetime,
    min_volume: float,
    min_duration_days: float,
    hours_before: int,
) -> list[dict]:
    """Sample eligible markets across weekly endDate windows in [start, end]."""
    candidates: list[dict] = []
    seen_events: dict[str, int] = {}
    window = start
    while window < end:
        window_end = min(window + timedelta(days=7), end)
        window_count = 0
        for offset in (0, 100, 200, 300):
            params = {
                "closed": "true",
                "limit": "100",
                "offset": str(offset),
                "order": "volumeNum",
                "ascending": "false",
                "end_date_min": window.strftime("%Y-%m-%d"),
                "end_date_max": window_end.strftime("%Y-%m-%d"),
                "volume_num_min": str(min_volume),
            }
            markets = None
            for attempt in range(3):
                await asyncio.sleep(0.3 * (1 + attempt * 10))
                try:
                    resp = await client.get(f"{GAMMA_API}/markets", params=params)
                    resp.raise_for_status()
                    markets = resp.json()
                    break
                except Exception as e:
                    logger.warning(
                        "collect.gamma_failed",
                        window=str(window.date()), offset=offset, attempt=attempt, error=str(e)[:120],
                    )
            if not isinstance(markets, list) or not markets:
                break
            for m in markets:
                question = (m.get("question") or "").strip()
                if not question or any(kw in question.lower() for kw in EXCLUDE_KEYWORDS):
                    continue
                if parse_resolution(m) is None or yes_token_id(m) is None:
                    continue
                end_dt = _parse_dt(m.get("endDate", ""))
                created_dt = _parse_dt(m.get("createdAt", "") or m.get("startDate", ""))
                closed_dt = _parse_dt(m.get("closedTime", ""))
                if not end_dt or not created_dt or not closed_dt:
                    continue
                t_pred = end_dt - timedelta(hours=hours_before)
                # Market must exist well before T and still be trading at T.
                if (t_pred - created_dt).total_seconds() < min_duration_days * 86400:
                    continue
                if closed_dt <= t_pred:
                    continue
                events = _parse_json_field(m.get("events"))
                event_id = str(events[0].get("id")) if events and isinstance(events[0], dict) else m.get("id", "")
                if seen_events.get(event_id, 0) >= 2:
                    continue
                seen_events[event_id] = seen_events.get(event_id, 0) + 1
                candidates.append(m)
                window_count += 1
            if len(markets) < 100:
                break
        print(f"  window {window.date()}: +{window_count} candidates ({len(candidates)} total)")
        window = window_end
    return candidates


def _trend_summary(history: list[dict], t_pred_ts: int) -> str:
    points = []
    for hours_ago in (72, 24, 6, 0):
        p = last_price_at_or_before(history, t_pred_ts - hours_ago * 3600)
        points.append((hours_ago, p))
    parts = [f"{h}h ago: {p:.2f}" if p is not None else f"{h}h ago: n/a" for h, p in points]
    parts[-1] = f"at prediction time: {points[-1][1]:.2f}" if points[-1][1] is not None else "now: n/a"
    return ", ".join(parts)


async def stage_collect(args) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    if MARKETS_FILE.exists():
        for line in MARKETS_FILE.read_text().splitlines():
            try:
                existing_ids.add(json.loads(line)["market_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    start = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)

    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        print(f"Sampling candidate markets {args.start_date} → {args.end_date} ...")
        candidates = await fetch_candidate_markets(
            client, start, end, args.min_volume, args.min_duration_days, args.hours_before
        )
        # Candidates arrive in chronological window order — shuffle (seeded)
        # so the collected sample covers the whole date range evenly.
        random.Random(42).shuffle(candidates)
        print(f"{len(candidates)} eligible candidates ({len(existing_ids)} already collected)")

        collected = 0
        with MARKETS_FILE.open("a") as out:
            for m in candidates:
                if collected + len(existing_ids) >= args.target:
                    break
                market_id = str(m.get("id"))
                if market_id in existing_ids:
                    continue
                end_dt = _parse_dt(m["endDate"])
                t_pred = end_dt - timedelta(hours=args.hours_before)
                t_pred_ts = int(t_pred.timestamp())

                token = yes_token_id(m)
                history = await fetch_price_series(
                    token, t_pred_ts - 96 * 3600, t_pred_ts, fidelity_minutes=60, client=client
                )
                p_market = last_price_at_or_before(history, t_pred_ts)
                if p_market is None or not (0.02 <= p_market <= 0.98):
                    continue

                wiki = await fetch_events_before(
                    m["question"], t_pred, client=client,
                    cache_dir=DATA_DIR / "wiki_events",
                    lookback_days=args.news_lookback_days,
                )
                headlines = []
                if not args.skip_gdelt:
                    headlines = await fetch_news_before(
                        m["question"], t_pred, lookback_days=args.news_lookback_days, client=client
                    )

                record = {
                    "market_id": market_id,
                    "question": m["question"],
                    "description": (m.get("description") or "")[:500],
                    "category": market_category(m),
                    "end_date": m["endDate"],
                    "closed_time": str(m.get("closedTime", "")),
                    "t_prediction": t_pred.isoformat(),
                    "p_market_at_t": p_market,
                    "price_trend": _trend_summary(history, t_pred_ts),
                    "outcome_yes": parse_resolution(m),
                    "volume": float(m.get("volumeNum") or 0),
                    "headlines": headlines,
                    "wiki_events": wiki,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                collected += 1
                print(
                    f"  [{collected}] {m['question'][:60]}  p@T={p_market:.2f} "
                    f"news={len(headlines)} wiki={len(wiki)}"
                )

    print(f"\nCollected {collected} new markets → {MARKETS_FILE}")


async def stage_refresh(args) -> None:
    """Re-derive the time-sensitive fields of an existing market sample:

    - market price at T with a tighter staleness tolerance (3h, not 6h)
    - wiki evidence from revision-pinned day pages (post-T edits excluded)
    - event_id for cluster-aware statistics

    Keeps the sample itself fixed so refreshing cannot cherry-pick markets.
    """
    rows = _load_jsonl(MARKETS_FILE)
    if not rows:
        print("Nothing to refresh.")
        return
    kept, dropped = [], 0
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        for i, row in enumerate(rows, 1):
            t_pred = datetime.fromisoformat(row["t_prediction"])
            t_pred_ts = int(t_pred.timestamp())

            try:
                resp = await client.get(f"{GAMMA_API}/markets/{row['market_id']}")
                resp.raise_for_status()
                m = resp.json()
            except Exception as e:
                logger.warning("refresh.gamma_failed", market_id=row["market_id"], error=str(e))
                dropped += 1
                continue

            events = _parse_json_field(m.get("events"))
            event_id = str(events[0].get("id")) if events and isinstance(events[0], dict) else row["market_id"]

            token = yes_token_id(m)
            history = await fetch_price_series(
                token, t_pred_ts - 96 * 3600, t_pred_ts, fidelity_minutes=60, client=client
            )
            p_market = last_price_at_or_before(history, t_pred_ts, tolerance_seconds=3 * 3600)
            if p_market is None or not (0.02 <= p_market <= 0.98):
                dropped += 1
                continue

            wiki = await fetch_events_before(
                row["question"], t_pred, client=client,
                cache_dir=DATA_DIR / "wiki_events_pinned",
                lookback_days=args.news_lookback_days,
            )
            row.update({
                "event_id": event_id,
                "p_market_at_t": p_market,
                "price_trend": _trend_summary(history, t_pred_ts),
                "wiki_events": wiki,
            })
            kept.append(row)
            if i % 25 == 0:
                print(f"  refreshed {i}/{len(rows)} (dropped {dropped})")

    import os
    tmp = MARKETS_FILE.with_suffix(f".jsonl.tmp{os.getpid()}")
    with tmp.open("w") as out:
        for row in kept:
            out.write(json.dumps(row) + "\n")
    tmp.replace(MARKETS_FILE)
    print(f"Refreshed {len(kept)} markets ({dropped} dropped) → {MARKETS_FILE}")


async def stage_probe(args) -> None:
    """Parametric-leakage probe: forecast the most market-surprising outcomes
    with ZERO evidence. If the evidence-blind model systematically calls the
    surprises the market missed, it remembers the future and the backtest is
    invalid for this model.
    """
    rows = _load_jsonl(MARKETS_FILE)
    surprising = sorted(
        rows,
        key=lambda r: abs((1.0 if r["outcome_yes"] else 0.0) - r["p_market_at_t"]),
        reverse=True,
    )[: args.probe_n]

    semaphore = asyncio.Semaphore(args.concurrency)
    results = []

    async def probe_one(m: dict) -> None:
        async with semaphore:
            f = await forecast(
                question=m["question"],
                as_of=m["t_prediction"][:10],
                headlines=[],
                world_events=[],
                description=m["description"],
            )
        y = 1.0 if m["outcome_yes"] else 0.0
        results.append({
            "question": m["question"],
            "p_market": m["p_market_at_t"],
            "p_blind": f.p_model,
            "outcome": m["outcome_yes"],
            "blind_brier": (f.p_model - y) ** 2,
            "market_brier": (m["p_market_at_t"] - y) ** 2,
            "failed": f.failed,
        })
        print(f"  mkt={m['p_market_at_t']:.2f} blind={f.p_model:.2f} actual={'YES' if y else 'NO '} | {m['question'][:60]}")

    await asyncio.gather(*(probe_one(m) for m in surprising))
    ok = [r for r in results if not r["failed"]]
    if not ok:
        print("Probe produced no usable forecasts.")
        return
    blind = sum(r["blind_brier"] for r in ok) / len(ok)
    market = sum(r["market_brier"] for r in ok) / len(ok)
    # "Knows" = confidently right about the side the market confidently
    # rejected. Near-0.5 estimates are generic uncertainty, not knowledge.
    knows = sum(
        1 for r in ok
        if (r["p_blind"] >= 0.75 if r["outcome"] else r["p_blind"] <= 0.25)
        and (r["p_market"] <= 0.25 if r["outcome"] else r["p_market"] >= 0.75)
    )
    print(f"\nProbe ({len(ok)} most surprising outcomes, evidence-blind, model={FORECAST_MODEL}):")
    print(f"  Brier blind={blind:.4f} vs market={market:.4f} (market should win big here)")
    print(f"  'knows the surprise' count: {knows}/{len(ok)} — should be ~0; >3 suggests parametric leakage")
    (DATA_DIR / "probe.json").write_text(json.dumps(
        {"model": FORECAST_MODEL, "blind_brier": blind, "market_brier": market,
         "knows_count": knows, "n": len(ok), "results": results}, indent=2))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


async def stage_predict(args) -> None:
    provider = getattr(args, "provider", "claude")
    forecast_fn, model_id = get_forecaster(provider)
    out_file = predictions_file(provider)

    markets = _load_jsonl(MARKETS_FILE)
    done_ids = {r["market_id"] for r in _load_jsonl(out_file)}
    todo = [m for m in markets if m["market_id"] not in done_ids]
    limit = getattr(args, "limit", 0)
    if limit:
        todo = todo[:limit]
    print(f"{len(markets)} markets, {len(done_ids)} already predicted, {len(todo)} to go "
          f"[provider={provider} model={model_id} → {out_file.name}]")
    if not todo:
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def predict_one(m: dict, out) -> None:
        async with semaphore:
            as_of = m["t_prediction"][:10]
            strip = getattr(args, "strip_headlines", False)
            kwargs = dict(
                question=m["question"],
                as_of=as_of,
                headlines=[] if strip else m["headlines"],
                world_events=m.get("wiki_events", []),
                description=m["description"],
            )
            result = await forecast_fn(**kwargs)
            if result.failed:
                result = await forecast_fn(**kwargs)
        record = {"market_id": m["market_id"], **result.to_dict()}
        async with write_lock:
            out.write(json.dumps(record) + "\n")
            out.flush()
        delta = result.p_model - m["p_market_at_t"]
        print(f"  {m['question'][:55]:<55} mkt={m['p_market_at_t']:.2f} model={result.p_model:.2f} ({delta:+.2f})")

    with out_file.open("a") as out:
        await asyncio.gather(*(predict_one(m, out) for m in todo))
    print(f"\nPredictions appended → {out_file}")


def _join_bets(
    split_date: str, provider: str = "claude", cluster: str = "stored",
    min_volume: float = 0.0,
) -> tuple[list[BetInput], list[BetInput]]:
    """cluster='stored' reproduces the published result exactly (event_id as
    collected, which equals market_id for all 959 rows and makes the bootstrap
    effectively per-trade). cluster='derived' groups correlated markets properly
    and yields the honest, wider CI.

    min_volume filters both splits on the market's FINAL volume as collected
    (the evaluation-universe rule); 0.0 keeps every market."""
    markets = {m["market_id"]: m for m in _load_jsonl(MARKETS_FILE)}
    train, test = [], []
    seen: set[str] = set()
    for p in _load_jsonl(predictions_file(provider)):
        m = markets.get(p["market_id"])
        if not m or p.get("failed") or p["market_id"] in seen:
            continue
        if float(m.get("volume") or 0.0) < min_volume:
            continue
        seen.add(p["market_id"])
        event_id = (cluster_key(m) if cluster == "derived"
                    else m.get("event_id", m["market_id"]))
        bet = BetInput(
            market_id=m["market_id"],
            question=m["question"],
            p_model=p["p_model"],
            p_market=m["p_market_at_t"],
            outcome=m["outcome_yes"],
            close_time=m["end_date"],
            category=m["category"],
            evidence_strength=p.get("evidence_strength", "none"),
            event_id=event_id,
        )
        (train if m["end_date"][:10] < split_date else test).append(bet)
    return train, test


_GAME_DAY_QUESTION = re.compile(r"^Will .+ win on \d{4}-\d{2}-\d{2}\?$")
_GAME_DAY_CATEGORY = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def is_game_day_market(question: str, category: str) -> bool:
    """Single-game sports markets ("Will X win on 2026-05-03?", or a
    league-team-team-date category slug) -- the bulk of the traded universe."""
    return bool(_GAME_DAY_QUESTION.match(question or "")
                or _GAME_DAY_CATEGORY.search(category or ""))


def load_params_from(path: Path) -> StrategyParams:
    """Import 'frozen_params' from another evaluation.json unchanged (no re-tune)."""
    frozen = json.loads(path.read_text())["frozen_params"]
    names = {f.name for f in dataclasses.fields(StrategyParams)}
    return StrategyParams(**{k: v for k, v in frozen.items() if k in names})


def _subgroup_report(bets: list[BetInput], params: StrategyParams) -> dict:
    r = simulate(bets, params)
    return {
        "n_markets": r.n_markets,
        "n_trades": r.n_trades,
        "roi": round(r.roi, 4),
        "roi_ci_90": [round(r.roi_ci_low, 4), round(r.roi_ci_high, 4)],
    }


def _print_report(name: str, r) -> None:
    print(f"\n--- {name} ---")
    print(f"  markets={r.n_markets} trades={r.n_trades} staked=${r.total_staked:.0f}")
    print(f"  P&L=${r.total_pnl:+.2f}  ROI={r.roi:+.1%}  (90% CI: {r.roi_ci_low:+.1%} … {r.roi_ci_high:+.1%})")
    print(f"  win_rate={r.win_rate:.1%}  avg|edge|={r.avg_edge_taken:.3f}  max_dd=${r.max_drawdown:.0f}")
    print(f"  kelly bankroll $1000 → ${r.kelly_final_bankroll:.0f}")
    print(f"  Brier (all):    blend={r.brier_blend:.4f}  market={r.brier_market:.4f}")
    print(f"  Brier (traded): blend={r.brier_blend_traded:.4f}  market={r.brier_market_traded:.4f}")


async def stage_evaluate(args) -> None:
    provider = getattr(args, "provider", "claude")
    eval_min_volume = float(getattr(args, "eval_min_volume", 0.0) or 0.0)
    params_from = getattr(args, "params_from", None)
    train, test = _join_bets(
        args.split_date, provider, getattr(args, "cluster", "stored"),
        min_volume=eval_min_volume,
    )
    print(f"Train: {len(train)} markets (close < {args.split_date}) | Test: {len(test)} markets "
          f"[provider={provider} eval_min_volume={eval_min_volume:g}]")
    if not train or not test:
        print("Not enough data on one side of the split — run collect/predict first.")
        return

    if params_from:
        # Transfer test: another model's frozen params applied unchanged, no
        # grid search, so nothing here is tuned on this provider's predictions.
        ranked = []
        best = load_params_from(Path(params_from))
        print(f"\nFrozen params imported from {params_from} (grid search skipped): "
              f"alpha={best.alpha} tau={best.threshold} gate={best.evidence_gate} "
              f"max_div={best.max_divergence}")
    else:
        print("\nGrid search on TRAIN (ranked by bootstrap ROI lower bound):")
        ranked = grid_search(train, min_trades=max(10, len(train) // 12))
        for r in ranked[:5]:
            p = r.params
            gate = p.evidence_gate or "-"
            print(
                f"  alpha={p.alpha:.2f} tau={p.threshold:.2f} gate={gate:<9} "
                f"trades={r.n_trades:3d} pnl=${r.total_pnl:+8.2f} roi={r.roi:+.1%} ci_low={r.roi_ci_low:+.1%}"
            )
        best = ranked[0].params
        print(f"\nFrozen params: alpha={best.alpha} tau={best.threshold} gate={best.evidence_gate}")

    train_report = simulate(train, best)
    test_report = simulate(test, best)
    _print_report("TRAIN (in-sample)", train_report)
    _print_report("TEST (out-of-sample)", test_report)

    # Baselines on test — does the LLM add anything beyond mechanical effects?
    fav = simulate_rule(test, "favorite", slippage=best.slippage)
    ano = simulate_rule(test, "always_no", slippage=best.slippage)
    # Pure-model baseline uses a fixed, pre-registered threshold (NOT the
    # tuned one) so the LLM-alone comparison is apples-to-apples.
    pure = simulate(test, StrategyParams(alpha=1.0, threshold=0.05, slippage=best.slippage))
    _print_report("BASELINE buy-favorite (test)", fav)
    _print_report("BASELINE always-no (test)", ano)
    _print_report("BASELINE pure-model alpha=1 tau=0.05 (test)", pure)
    # The same strategy fed a constant 0.50: what fading favourites earns with
    # no forecast at all. The model adds something only if it beats this.
    coin_train = simulate([dataclasses.replace(b, p_model=0.5) for b in train], best)
    coin = simulate([dataclasses.replace(b, p_model=0.5) for b in test], best)
    _print_report("BASELINE coin 0.50, same params (train)", coin_train)
    _print_report("BASELINE coin 0.50, same params (test)", coin)

    # Test split by market type: single-game sports vs everything else. The
    # live universe's composition differs from the backtest's, so the edge
    # has to be visible per subgroup, not just in aggregate.
    game_day = [b for b in test if is_game_day_market(b.question, b.category)]
    other = [b for b in test if not is_game_day_market(b.question, b.category)]
    subgroups = {
        "split": "test",
        "rule": "game_day_sports: question ~ '^Will .+ win on YYYY-MM-DD?$' or category ~ '-YYYY-MM-DD$'",
        "game_day_sports": _subgroup_report(game_day, best),
        "other": _subgroup_report(other, best),
    }
    print("\nTest split by subgroup:")
    for name in ("game_day_sports", "other"):
        sg = subgroups[name]
        print(f"  {name:<16} markets={sg['n_markets']:3d} trades={sg['n_trades']:3d} "
              f"roi={sg['roi']:+.1%}  (90% CI: {sg['roi_ci_90'][0]:+.1%} … {sg['roi_ci_90'][1]:+.1%})")

    # Edge by month — parametric leakage would show as edge decaying with
    # distance from the model's Jan 2026 training cutoff.
    print("\nEdge by close month (all markets, blend Brier vs market Brier):")
    by_month: dict[str, list] = {}
    for b in train + test:
        by_month.setdefault(b.close_time[:7], []).append(b)
    month_rows = {}
    for month in sorted(by_month):
        bs = by_month[month]
        a = best.alpha
        brier_blend = sum((a * b.p_model + (1 - a) * b.p_market - (1.0 if b.outcome else 0.0)) ** 2 for b in bs) / len(bs)
        brier_market = sum((b.p_market - (1.0 if b.outcome else 0.0)) ** 2 for b in bs) / len(bs)
        month_rows[month] = {"n": len(bs), "brier_blend": round(brier_blend, 4), "brier_market": round(brier_market, 4)}
        print(f"  {month}  n={len(bs):3d}  blend={brier_blend:.4f}  market={brier_market:.4f}  delta={brier_blend - brier_market:+.4f}")

    _, model_id = get_forecaster(provider)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "split_date": args.split_date,
        "n_train": len(train),
        "n_test": len(test),
        "edge_by_month": month_rows,
        "frozen_params": best.to_dict(),
        "grid_top5": [r.to_dict() for r in ranked[:5]],
        "train": train_report.to_dict(),
        "test": test_report.to_dict(include_trades=True),
        "baselines_test": {
            "buy_favorite": fav.to_dict(),
            "always_no": ano.to_dict(),
            "pure_model": pure.to_dict(),
            "coin_half": coin.to_dict(),
        },
        "baselines_train": {"coin_half": coin_train.to_dict()},
        "subgroups": subgroups,
    }
    cluster_mode = getattr(args, "cluster", "stored")
    output["cluster"] = cluster_mode
    if params_from:
        output["params_source"] = str(params_from)
    if eval_min_volume > 0:
        output["eval_min_volume"] = eval_min_volume
    stem = "evaluation" if provider == "claude" else f"evaluation.{model_id}"
    if cluster_mode != "stored":
        stem += f".{cluster_mode}"
    out_stem = getattr(args, "out_stem", None)
    if out_stem:
        stem = out_stem
    eval_file = DATA_DIR / f"{stem}.json"
    eval_file.write_text(json.dumps(output, indent=2))
    print(f"\nFull evaluation written → {eval_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cassandra time-machine backtest")
    parser.add_argument("stage", choices=["collect", "refresh", "probe", "predict", "evaluate", "all"])
    parser.add_argument("--probe-n", type=int, default=20, help="markets in the leakage probe")
    parser.add_argument("--target", type=int, default=250, help="markets to collect")
    parser.add_argument("--start-date", default="2026-02-01")
    parser.add_argument("--end-date", default="2026-06-07")
    parser.add_argument("--split-date", default="2026-05-01", help="train/test boundary on endDate")
    parser.add_argument("--hours-before", type=int, default=24, help="prediction time T = close - this")
    parser.add_argument("--min-volume", type=float, default=5000.0)
    parser.add_argument("--min-duration-days", type=float, default=3.0)
    parser.add_argument("--news-lookback-days", type=int, default=10)
    parser.add_argument("--skip-gdelt", action="store_true", help="skip GDELT (rate-limit penalty box)")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel forecaster calls")
    parser.add_argument("--provider", choices=["claude", "openai"], default="claude",
                        help="forecaster backend; openai writes to its own predictions/evaluation "
                             "files so the archived claude baseline is never overwritten")
    parser.add_argument("--cluster", choices=["stored", "derived"], default="stored",
                        help="bootstrap clustering: 'stored' reproduces the published CI "
                             "(inert -- event_id == market_id); 'derived' groups correlated "
                             "markets and gives the honest, wider CI")
    parser.add_argument("--limit", type=int, default=0,
                        help="predict: cap markets this run (0 = all); use for cheap smoke tests")
    parser.add_argument("--params-from", default=None, metavar="PATH",
                        help="evaluate: import 'frozen_params' from this evaluation.json and skip "
                             "the grid search (transfer test of another model's params)")
    parser.add_argument("--eval-min-volume", type=float, default=0.0, metavar="FLOAT",
                        help="evaluate: keep only markets whose FINAL volume >= this "
                             "(the evaluation-universe rule); 0 = all markets")
    parser.add_argument("--data-dir", default=None, metavar="PATH",
                        help="work on a separate sample under this directory (markets.jsonl, "
                             "predictions, evaluation) so the archived 959-market sample is untouched")
    parser.add_argument("--predictions-tag", default="", metavar="TAG",
                        help="predict/evaluate: suffix for the predictions file (ablation runs)")
    parser.add_argument("--strip-headlines", action="store_true",
                        help="predict: forecast with headlines removed (news ablation)")
    parser.add_argument("--out-stem", default=None, metavar="NAME",
                        help="evaluate: override the output filename stem under data/backtest/")
    args = parser.parse_args()
    global DATA_DIR, MARKETS_FILE, PREDICTIONS_FILE, EVALUATION_FILE, PREDICTIONS_TAG
    PREDICTIONS_TAG = args.predictions_tag
    if args.data_dir:
        DATA_DIR = Path(args.data_dir).resolve()
        MARKETS_FILE = DATA_DIR / "markets.jsonl"
        PREDICTIONS_FILE = DATA_DIR / "predictions.jsonl"
        EVALUATION_FILE = DATA_DIR / "evaluation.json"

    async def run():
        if args.stage in ("collect", "all"):
            await stage_collect(args)
        if args.stage == "refresh":
            await stage_refresh(args)
        if args.stage == "probe":
            await stage_probe(args)
        if args.stage in ("predict", "all"):
            await stage_predict(args)
        if args.stage in ("evaluate", "all"):
            await stage_evaluate(args)

    asyncio.run(run())


if __name__ == "__main__":
    main()
