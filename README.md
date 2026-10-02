# Cassandra

Autonomous AI prediction engine for [Polymarket](https://polymarket.com). Cassandra forecasts markets with a market-blind LLM, trades the disagreement against real prices, and measures itself with a leak-controlled backtest — end to end, no human in the loop.

**+30.5% out-of-sample ROI on 144 simulated trades (90% CI +5.1%…+56.3%, clear of zero) in a leak-controlled time-machine backtest against real market prices.** A live shadow forward test is running (see [Forward test](#forward-test-paper-trading)); the edge is not yet confirmed on live order books.

---

## How the predictor works

1. **Forecast (market-blind)** — a pinned LLM (`claude-fable-5`) estimates P(YES) from dated evidence and base rates, *without seeing the market price*. Shown the price, LLMs anchor to it within ±1¢ — which destroys the signal. Evidence comes from revision-pinned Wikipedia Current Events pages.
2. **Blend & decide** — the strategy layer shrinks the forecast toward the market (`p = α·p_model + (1−α)·p_market`) and buys the cheap side when the divergence exceeds a threshold τ. α and τ are tuned on a train split and frozen.
3. **Verify** — every claim of edge is checked against real prices at decision time via a time-machine backtest over 959 resolved markets.

The repo also contains the larger always-on platform (multi-source ingestion, hybrid retrieval over Neo4j + Qdrant, judge/hallucination/reflection agents, risk guardrails, war-room dashboard) — see [Live platform](#live-platform) below.

---

## Backtest results

A **time-machine backtest** against 959 resolved Polymarket markets (Feb–Jun 2026, volume ≥ $20k, max 2 markets per event). For each market, the pipeline travels to T = 24h before scheduled close and trades at the **real market price at T** (from the CLOB price-history API), settling at the actual resolution.

### Methodology — leakage controls

An earlier version of this backtest had every classic lookahead bug (settlement price used as entry price, news fetched at run time with no date bound). The current harness was rebuilt around one rule: *nothing the forecaster or the trade decision sees may postdate T*.

| Channel | Control |
|---------|---------|
| Entry price | CLOB `prices-history`: last trade ≤ T (max 3h stale), never `outcomePrices` |
| News evidence | Wikipedia Current Events day pages **pinned to their last revision before D+1 00:00 UTC** via the MediaWiki revisions API — post-hoc hindsight edits excluded (a current-revision fetch was worth ~10 extra lines/page of hindsight) |
| Model knowledge | Forecaster pinned to `claude-fable-5` (training cutoff Jan 2026); all backtested markets close Feb 2026+ |
| Model tools | `claude -p` runs with WebSearch/WebFetch/Bash/etc. explicitly disallowed |
| Leakage probe | Evidence-blind forecasts on the 20 most market-surprising outcomes: blind Brier 0.50 vs market 0.70, **1/18 strict "knows the surprise"** — and the model is confidently *wrong* about post-cutoff BTC prices, confirming its knowledge ends at the cutoff |
| Anchoring | The forecaster never sees the market price (shown the price, it anchored within ±1¢ on 12/12 pilot markets); shrinkage toward the market happens in the strategy layer |
| Overfitting | Strategy grid (α, τ, evidence gate, divergence cap) tuned on train (close < May 1, n=687), frozen, evaluated **once** on test (May–Jun, n=272); CIs use event-cluster bootstrap |

Strategy: blend `p = α·p_model + (1−α)·p_market`, buy the cheap side when `|p − p_market| > τ`, 1¢ slippage, flat $100 stakes. Frozen params from train: α=1.0, τ=0.08, no evidence gate.

### Out-of-sample results (test: May–Jun 2026, 272 markets)

| Metric | Strategy | always-NO | buy-favorite (≥90¢) | pure model (τ=0.05) |
|--------|----------|-----------|----------------------|----------------------|
| Trades | 144 | 272 | 46 | 192 |
| P&L (flat $100) | **+$4,395** | +$2,059 | −$118 | +$3,877 |
| ROI | **+30.5%** | +7.6% | −2.6% | +20.2% |
| 90% CI (event-cluster) | **+5.1% … +56.3%** | −4.6% … +18.0% | −10.1% … +2.4% | −0.6% … +44.5% |
| Win rate | 45.1% | 54.4% | 93.5% | 42.7% |
| Max drawdown | $1,170 | $2,011 | $174 | $1,670 |

Train (in-sample): +$6,760 on 342 trades (+19.8% ROI, CI low +2.9%). Quarter-Kelly compounding on test: $1,000 → $4,048.

**The strategy is the only variant whose CI excludes zero.** Mechanical effects don't
explain the result: always-NO made +7.6% but its CI spans zero, buying favorites *lost*
money, and the pure-model rule (+20.2%) also spans zero once correlated markets are
clustered properly.

> **CI correction (2026-08-10).** Earlier versions of this table reported
> **+8.0% … +55.4%**, from a bootstrap that clustered on the stored `event_id`. That
> field equals `market_id` for all 959 collected markets, so the "cluster bootstrap"
> resampled 959 clusters over 959 markets and was effectively per-trade — ten
> correlated "BTC below strike" bets and the several markets on one football match
> each counted as independent. Re-running with keys derived from the event slug
> (`--cluster derived`) gives the wider, honest interval above. Point estimates are
> unchanged; only the precision was overstated. The affected baseline claim (that the
> lower bound cleared every baseline's mean) has been removed — at +5.1% it does not
> clear always-NO's +7.6% point estimate. Reproduce either with
> `scripts/backtest.py evaluate --cluster {stored,derived}`.

### Honest read of the numbers

- **Long-shot book profile**: 109 of 144 entries are under 50¢; the top-10 winners contribute 109% of total P&L and the other 134 trades net −$407. Most trades lose small; winners pay 5–10:1. Expect long losing streaks at a 45% win rate.
- **Both sides are profitable** — NO: +$3,706 on 104 trades; YES: +$689 on 40 — so the edge is not just a NO-tilt riding a NO-heavy period (always-NO trailed by 23 points).
- **The model is not better calibrated than the market** (test Brier: blend 0.262 vs market 0.217). The P&L comes from selective, payoff-asymmetric disagreement on cheap sides — not from beating the market on average.
- **No cutoff-decay pattern**: the model's Brier gap vs the market is flat-to-shrinking from Feb (+0.069) through May (+0.043) — not what parametric leakage would produce.
- **The edge is not stale-crypto luck**: sports markets carry the test book (116/144 trades, +$2,798, +24.1% ROI) on stable team-strength priors; crypto-price markets contributed only 10 trades. Excluding the 5 extreme-divergence trades (>30¢) still leaves +$2,665 on 139 trades.
- An earlier 286-market run with different frozen params (α=0.75, τ=0.03) showed +23.2% test ROI with a CI spanning zero; scaling to 959 markets tightened the CI and the conclusion survived.

### Known limitations

- The test window is one regime (May–Jun 2026); a single period can favor a long-shot book. Forward (paper-trading) confirmation on live order books is still the bar before real capital. A **shadow** run has been live on GitHub Actions since 2026-08-10 (56 decisions, 4 settled as of 2026-10-01; dashboard: https://akhilc08.github.io/cassandra/). The shadow period found plumbing defects rather than a result: the Wikipedia evidence channel was dead on every decision (blocked User-Agent, cold cache, swallowed failures), the $500k floor was applied to volume-to-date instead of final volume (cutting supply to ~8% of the backtest-equivalent and excluding game-day sports markets), and a daily cron drifting 3–6h made the realised horizon bimodal instead of 24h. All are fixed in shadow; those rows never count. **Official** mode starts when `data/forward/preregistration.json` is committed — until then the backtest's blind spots (real spreads, executability, regime dependence) remain open.
- Market universe filtered on **final** volume (post-T information) — selection, not leakage, but live deployment would select on volume-to-date.
- Universe limited to markets still open 24h before *scheduled* close with clean YES/NO resolution; early-resolving and disputed markets are excluded.
- 1¢ flat slippage approximates execution; thin books would fill worse than the last-trade price.
- GDELT is rate-limited to uselessness from this network; evidence is Wikipedia-only (199/959 markets had relevant pinned events) plus the model's pre-cutoff knowledge.

---

## Reproducing the backtest

No Docker, no API keys — every data source (Polymarket Gamma, CLOB price history, Wikipedia revisions) is free; synthesis runs through the Claude Code CLI (`claude -p`, pinned model).

```bash
python3 -m venv .venv
.venv/bin/pip install httpx structlog pydantic-settings numpy scipy aiosqlite pytest pytest-asyncio

# stages are cached in data/backtest/ and resumable
.venv/bin/python scripts/backtest.py collect --target 1000 --min-volume 20000 --skip-gdelt
.venv/bin/python scripts/backtest.py refresh                 # pinned wiki revisions, event ids, 3h price tolerance
.venv/bin/python scripts/backtest.py probe                   # parametric-leakage probe (run before trusting results)
.venv/bin/python scripts/backtest.py predict --concurrency 8 # claude -p forecasts, resumable
.venv/bin/python scripts/backtest.py evaluate                # grid on train, frozen eval on test

.venv/bin/python -m pytest tests/unit/test_pnl.py tests/unit/test_time_machine.py
```

Full artifacts: `data/backtest/evaluation.json` (per-trade detail), `data/backtest/probe.json`.

---

## Forward test (paper trading)

`scripts/forward_test.py` runs the frozen strategy against live Polymarket order books from GitHub Actions (`.github/workflows/forward-test.yml`, cron `17 */6 * * *`), with `gpt-5.6-luna` as the forecaster behind the same market-blind prompt. Strategy code is *imported* from `src/oracle/evaluation/pnl.py`, never re-implemented, and the runner exits non-zero if its params differ from the frozen baseline. Results are committed back to `data/forward/` and published at https://akhilc08.github.io/cassandra/.

- **scan** — markets closing in 12–27h (nominal horizon 24h, the backtest's T) with volume-to-date ≥ $20k (the backtest's collect floor), ≥ 3 days old, ≤ 2 open per event cluster. Wikipedia day pages (last 10 days, revision-pinned, cache committed under `data/forward/wiki_cache`) are fetched once per run; each market gets a forecast and a `decide_trade` verdict. One row per attempted market in `decisions.jsonl` (`open`, `no_trade` or `forecast_failed`), each carrying `hours_to_close`, `evidence_ok`, the served model snapshot and token usage.
- **settle** — once a market closes, every row for it gets `outcome_yes`, `ts_resolved` and `volume_final`; `open` rows get P&L at the recorded book mid with 1¢ slippage (replica) and at the recorded ask (executable).
- **report** — `summary.json`, via the same `simulate()` and event-cluster bootstrap as the backtest.

Three populations are reported per mode: **primary** — evidence channel healthy at decision time (`evidence_ok`) and **final** volume ≥ $500k, the like-for-like match for the backtest's evaluation universe; **secondary** — volume ≥ $500k at decision time; **all** — every settled trade. Calibration (model vs market Brier) is measured over every resolved scanned market, traded or not.

Health: every scan appends a row to `data/forward/scans.jsonl`; the summary's `health` is `ok`, `degraded` (fewer than 8/10 Wikipedia day pages, forecast failures in the last run, or > 24h since the last scan), `failed` (Gamma, CLOB or forecaster outage, missing API key, crash) or `unknown`. Reading the dashboard: the banner is the health verdict plus staleness (warn > 24h, error > 48h since the last scan); **shadow** and **official** rows are labelled and never pooled; every CI is *interim* until the single pre-registered look at 200 settled primary trades. Analysis plan and pre-registration draft: `docs/superpowers/specs/2026-08-09-forward-test-v2-design.md`.

---

## Live platform

Beyond the backtested core, the repo ships a continuously-running platform:

1. **Ingest** — NewsAPI, Twitter/X, Reddit, government APIs (Congress, CourtListener), polling aggregators, YouTube/podcast audio (Whisper), chart images (vision)
2. **Store** — chunks and embeds content into Qdrant (vectors) and extracts entities into Neo4j (knowledge graph)
3. **Research** — a multi-agent system generates a structured thesis per market using hybrid retrieval (vector + BM25 + graph, RRF fusion, BGE re-ranking)
4. **Evaluate** — an LLM judge scores theses on 4 dimensions (gates: groundedness ≥ 7, reasoning ≥ 6, evidence ≥ 5); a hallucination detector verifies claims against sources at 0.75 cosine similarity
5. **Reflect** — a self-critique step checks for anchoring, recency, and confirmation biases
6. **Trade** — the Risk Agent enforces hard guardrails; the Portfolio Manager executes paper trades
7. **Learn** — post-resolution post-mortems classify predictions as good-process vs lucky

```
┌─────────────────────────────────────────────────────────┐
│                    Ingestion Layer                       │
│  News · Twitter · Reddit · Gov APIs · Audio · Vision    │
└───────────────────┬─────────────────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │   Knowledge Store  │
          │  Neo4j (graph)     │
          │  Qdrant (vectors)  │
          └─────────┬──────────┘
                    │
     ┌──────────────▼──────────────┐
     │     Hybrid Retrieval        │
     │  Vector · BM25 · Graph      │
     │  RRF Fusion · BGE Reranker  │
     └──────────────┬──────────────┘
                    │
     ┌──────────────▼──────────────────────────┐
     │            Agent Pipeline               │
     │  Forecaster → Reflection → Judge        │
     │  Hallucination Check → Risk → Trade     │
     └──────────────┬──────────────────────────┘
                    │
     ┌──────────────▼──────────────┐
     │       Observability         │
     │  Prometheus · Grafana       │
     │  LLM Tracer · SSE Dashboard │
     └─────────────────────────────┘
```

### Risk guardrails (hard limits)

| Rule | Threshold |
|------|-----------|
| Max single-market exposure | 10% of portfolio |
| Max category exposure | 30% of portfolio |
| Max risk on markets resolving within 24h | 5% of portfolio |
| Stop-loss trigger | 50% loss on any position |

### Tech stack

| Layer | Tech |
|-------|------|
| Forecaster | `claude-fable-5` via Claude Code CLI (pinned, market-blind) |
| Agent LLM (judge, hallucination, reflection) | claude-haiku-4-5 |
| API | FastAPI + uvicorn |
| Vector DB | Qdrant v1.9 |
| Graph DB | Neo4j 5.19 (APOC) |
| Embeddings | sentence-transformers (BAAI/bge-large-en-v1.5, 1024-dim) |
| Retrieval | vector + BM25 + graph traversal, RRF fusion (k=60), BGE-reranker-v2-m3 |
| Model routing | ComplexityClassifier (logistic regression) — local stub for simple queries, Claude for synthesis |
| Fine-tuning | Modal + LoRA (Mistral 7B Instruct, r=16) |
| Observability | Prometheus + Grafana, SSE war-room dashboard (React/Vite) |
| Cache | Qdrant-backed semantic cache + TTL tool cache |
| A/B testing | SQLite + two-sample t-test (min 30 samples/variant) |
| Containerization | Docker Compose |

### Running the platform

```bash
cp .env.example .env          # ORACLE_ANTHROPIC_API_KEY, ORACLE_NEWSAPI_KEY, ...
docker compose up -d          # Neo4j, Qdrant, Prometheus, Grafana
pip install uv && uv sync
uvicorn oracle.api.app:app --reload
cd frontend && npm install && npm run dev
```

API at `http://localhost:8000`, war-room dashboard at `http://localhost:5173`. Prometheus metrics at `/metrics` (`oracle_brier_score`, `oracle_accuracy_rate`, `oracle_portfolio_value`, `oracle_cost_per_prediction`, …).

Optional LoRA fine-tuning on resolved markets: `modal run src/oracle/training/modal_trainer.py` (A10G, ~2h; r=16, alpha=32, q/k/v/o projections).

---

## Project structure

```
scripts/
├── backtest.py      # time-machine backtest: collect → refresh → probe → predict → evaluate
└── forward_test.py  # live paper trading on GitHub Actions: scan → settle → report

src/oracle/
├── agents/          # forecaster (market-blind, pinned), research, reflection, risk, portfolio
├── api/             # FastAPI app, routes, SSE streaming
├── cache/           # TTL + semantic cache (Qdrant-backed)
├── evaluation/      # pnl (P&L sim, Kelly, cluster bootstrap), judge, hallucination, post-mortems
├── ingestion/       # price_history (CLOB), wiki_events (revision-pinned), gdelt, news, social, gov
├── knowledge/       # Neo4j + Qdrant clients, embeddings
├── observability/   # Prometheus metrics, LLM tracer
├── prompts/         # prompt registry, A/B testing
├── retrieval/       # vector, BM25, graph search, RRF fusion, re-ranker
├── routing/         # complexity classifier → model routing
└── training/        # Modal LoRA fine-tuning, synthetic data generator

data/
├── backtest/        # markets.jsonl, predictions.jsonl, evaluation.json, probe.json
└── forward/         # decisions.jsonl, scans.jsonl, summary.json, wiki_cache/ (committed)
```
