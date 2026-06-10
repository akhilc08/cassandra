# Cassandra

Autonomous AI prediction engine for [Polymarket](https://polymarket.com). Cassandra ingests multi-modal signals, reasons over a hybrid knowledge graph, and executes paper trades — end to end, no human in the loop.

**+30.5% out-of-sample ROI on 144 simulated trades (90% CI +8%…+55%, clear of zero) in a leak-controlled time-machine backtest against real market prices**

---

## What it does

Oracle runs a continuous prediction pipeline:

1. **Ingest** — pulls from NewsAPI, Twitter/X, Reddit, government APIs (Congress, CourtListener), polling aggregators, YouTube/podcast audio (Whisper), and chart images (vision)
2. **Store** — chunks and embeds content into Qdrant (vector DB) and extracts entities into Neo4j (knowledge graph)
3. **Research** — a multi-agent system generates a structured thesis for each market using hybrid retrieval
4. **Evaluate** — an LLM judge scores the thesis on 4 dimensions; a hallucination detector verifies every claim against sources
5. **Reflect** — a self-critique step checks for anchoring, recency, and confirmation biases before committing
6. **Trade** — the Risk Agent enforces hard guardrails, then the Portfolio Manager executes paper trades
7. **Learn** — post-resolution post-mortems classify predictions as good-process vs lucky, feeding back into calibration

---

## Architecture

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
     │  Research → Reflection → Judge          │
     │  Hallucination Check → Risk → Trade     │
     └──────────────┬──────────────────────────┘
                    │
     ┌──────────────▼──────────────┐
     │       Observability         │
     │  Prometheus · Grafana       │
     │  LLM Tracer · SSE Dashboard │
     └─────────────────────────────┘
```

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
| 90% CI (per-trade ROI) | **+8.0% … +55.4%** | −6.5% … +22.1% | −9.5% … +2.6% | +1.0% … +42.3% |
| Win rate | 45.1% | 54.4% | 93.5% | 42.7% |
| Max drawdown | $1,170 | $2,011 | $174 | $1,670 |

Train (in-sample): +$6,760 on 342 trades (+19.8% ROI, CI low +3.8%). Quarter-Kelly compounding on test: $1,000 → $4,048.

**The strategy's CI excludes zero** and its lower bound (+8.0%) clears every baseline's mean. Mechanical effects don't explain it: always-NO made +7.6% (CI spans zero) and buying favorites *lost* money in this period.

### Honest read of the numbers

- **Long-shot book profile**: 109 of 144 entries are under 50¢; the top-10 winners contribute 109% of total P&L and the other 134 trades net −$407. Most trades lose small; winners pay 5–10:1. Expect long losing streaks at a 45% win rate.
- **Both sides are profitable** — NO: +$3,706 on 104 trades; YES: +$689 on 40 — so the edge is not just a NO-tilt riding a NO-heavy period (always-NO trailed by 23 points).
- **The model is not better calibrated than the market** (test Brier: blend 0.262 vs market 0.217). The P&L comes from selective, payoff-asymmetric disagreement on cheap sides — not from beating the market on average.
- **No cutoff-decay pattern**: the model's Brier gap vs the market is flat-to-shrinking from Feb (+0.069) through May (+0.043) — not what parametric leakage would produce.
- An earlier 286-market run with different frozen params (α=0.75, τ=0.03) showed +23.2% test ROI with a CI spanning zero; scaling to 959 markets tightened the CI and the conclusion survived.

### Known limitations

- The test window is one regime (May–Jun 2026); a single period can favor a long-shot book. Forward (paper-trading) confirmation is still the bar before real capital.
- Market universe filtered on **final** volume (post-T information) — selection, not leakage, but live deployment would select on volume-to-date.
- Universe limited to markets still open 24h before *scheduled* close with clean YES/NO resolution; early-resolving and disputed markets are excluded.
- 1¢ flat slippage approximates execution; thin books would fill worse than the last-trade price.
- GDELT is rate-limited to uselessness from this network; evidence is Wikipedia-only (199/959 markets had relevant pinned events) plus the model's pre-cutoff knowledge.
- Reproducing: `scripts/backtest.py` stages are cached and resumable — `collect → refresh → probe → predict → evaluate`. Full artifacts in `data/backtest/evaluation.json`.

---

## Forward test (live paper trading)

The backtest's remaining blind spots — real spreads, executability, regime dependence — are covered by a forward paper-trading test running the **identical frozen system** against live markets:

- **Pre-registered 2026-06-10** (from the 687-market train split): α=1.0, τ=0.08, 3–97¢ bounds, flat $100 paper stakes. Not retuned while the test runs.
- **Fills at the real ask** from the live CLOB order book (stricter than the backtest's last-trade + 1¢), decision edge measured against the book mid.
- **Two arms per market**: `replica` (wiki-events evidence only, exactly as backtested — tests whether the backtest generalizes) and `enhanced` (web search enabled, legitimate live — tests whether richer evidence adds edge).
- Every decision is logged at decision time (forecast, reasoning, book snapshot, fill) to `data/forward/decisions.jsonl`, including no-trades.
- **Success bar** before any real capital: ~100 settled trades, positive P&L, cluster-bootstrap CI lower bound above −5%, replica arm consistent with its backtest CI.

```bash
# daily scan + settlement (e.g. cron at 14:00 UTC)
python scripts/forward_test.py scan
python scripts/forward_test.py settle    # also prints the running report
```

---

## Numbers

### Retrieval

| Metric | Value |
|--------|-------|
| Retrieval strategies | 3 (vector, BM25, graph traversal) |
| Fusion algorithm | Reciprocal Rank Fusion (k=60) |
| Re-ranker | BGE-reranker-v2-m3 (cross-encoder) |
| Embedding model | BAAI/bge-large-en-v1.5 (1024-dim) |
| Claim verification threshold | 0.75 cosine similarity |

### Evaluation pipeline

| Check | Model | Max tokens |
|-------|-------|-----------|
| LLM judge (4-dim scoring) | claude-3-5-haiku | 1,024 |
| Hallucination — claim extraction | claude-3-5-haiku | 1,024 |
| Hallucination — contradiction check | claude-3-5-haiku | 1,024 |
| Reflection / bias detection | claude-3-5-haiku | 512 |
| Research synthesis | claude-3-5-haiku | 1,024 |

Judge quality gates: groundedness ≥ 7/10, reasoning ≥ 6/10, evidence ≥ 5/10.

### Model routing

The `ComplexityClassifier` (logistic regression, trained on 500 synthetic samples) routes ~80% of queries to the local stub and only sends ~20% to Claude — keeping API costs low.

| Route | Share | Latency |
|-------|-------|---------|
| Local stub | ~80% | <1ms |
| Claude (haiku) | ~20% | ~500ms |

### Risk guardrails (hard limits)

| Rule | Threshold |
|------|-----------|
| Max single-market exposure | 10% of portfolio |
| Max category exposure | 30% of portfolio |
| Max risk on markets resolving within 24h | 5% of portfolio |
| Stop-loss trigger | 50% loss on any position |

### API cost (claude-3-5-haiku @ $0.80/1M input · $4.00/1M output)

| Scale | Research cycles/day | Daily cost |
|-------|-------------------|------------|
| 10 markets | ~30 | ~$0.35 |
| 50 markets | ~150 | ~$1.80 |
| 200 markets | ~1,000 | ~$12 |
| 500 markets | ~2,500 | ~$30 |

### Ingestion schedule

| Source | Interval |
|--------|---------|
| Polymarket markets | 60 seconds |
| News (NewsAPI) | 15 minutes |
| Reddit | 30 minutes |
| Government APIs | 6 hours |
| Polling aggregators | 12 hours |
| Audio (Whisper) | Daily |
| Twitter/X | Continuous streaming |

---

## Tech stack

| Layer | Tech |
|-------|------|
| API | FastAPI + uvicorn |
| Vector DB | Qdrant v1.9 |
| Graph DB | Neo4j 5.19 (APOC) |
| Embeddings | sentence-transformers (BGE large) |
| LLM | Anthropic claude-3-5-haiku |
| Fine-tuning | Modal + LoRA (Mistral 7B Instruct, r=16) |
| Observability | Prometheus + Grafana |
| Cache | Qdrant-backed semantic cache (1024-dim) |
| Frontend | React (Vite) — real-time war room dashboard |
| Streaming | Server-Sent Events (SSE) |
| A/B testing | SQLite + two-sample t-test (min 30 samples/variant) |
| Containerization | Docker Compose |

---

## Quickstart

```bash
# 1. Copy env
cp .env.example .env
# Fill in: ORACLE_ANTHROPIC_API_KEY, ORACLE_NEWSAPI_KEY, etc.

# 2. Start infrastructure
docker compose up -d

# 3. Install Python deps
pip install uv && uv sync

# 4. Run the API
uvicorn oracle.api.app:app --reload

# 5. Start the frontend
cd frontend && npm install && npm run dev
```

The API will be at `http://localhost:8000` and the war room dashboard at `http://localhost:5173`.

---

## Fine-tuning (optional)

Oracle ships a Modal-based LoRA pipeline for fine-tuning Mistral 7B on prediction market reasoning:

```bash
# Generate training data from resolved markets
python -m oracle.training.data_generator

# Launch fine-tune on Modal (A10G GPU, ~2h)
modal run src/oracle/training/modal_trainer.py
```

LoRA config: r=16, alpha=32, target modules: q/k/v/o projections, dropout=0.05.

---

## Observability

Prometheus metrics are exposed at `/metrics`. Key gauges:

- `oracle_brier_score` — calibration quality (lower is better)
- `oracle_accuracy_rate` — rolling prediction accuracy
- `oracle_cache_hit_rate` — tool cache efficiency
- `oracle_portfolio_value` — paper portfolio value
- `oracle_cost_per_prediction` — USD cost histogram
- `oracle_llm_latency_seconds` — per-model, per-agent latency

The React war room streams live agent activity, trade decisions, and evaluation scores via SSE.

---

## Project structure

```
src/oracle/
├── agents/          # Research, Reflection, Quant, Risk, Portfolio agents
├── api/             # FastAPI app, routes, SSE streaming
├── cache/           # TTL + semantic cache (Qdrant-backed)
├── evaluation/      # LLM judge, hallucination detector, calibration, post-mortems
├── ingestion/       # News, Twitter, Reddit, audio, vision, gov scrapers
├── knowledge/       # Neo4j + Qdrant clients, embeddings
├── observability/   # Prometheus metrics, LLM tracer
├── prompts/         # Prompt registry, A/B testing
├── retrieval/       # Vector, BM25, graph search, RRF fusion, re-ranker
├── routing/         # Complexity classifier → model routing
└── training/        # Modal LoRA fine-tuning, synthetic data generator
```
