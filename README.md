# Cassandra

Autonomous AI prediction engine for [Polymarket](https://polymarket.com). Cassandra ingests multi-modal signals, reasons over a hybrid knowledge graph, and executes paper trades — end to end, no human in the loop.

**+23% out-of-sample ROI on 57 simulated trades (90% CI −9%…+59%) in a leak-controlled time-machine backtest against real market prices**

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

A **time-machine backtest** against 286 resolved Polymarket markets (Feb–Jun 2026, median volume $1.1M, max 2 markets per event). For each market, the pipeline travels to T = 24h before scheduled close and trades at the **real market price at T** (from the CLOB price-history API), settling at the actual resolution.

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
| Overfitting | Strategy grid (α, τ, evidence gate, divergence cap) tuned on train (close < May 1, n=212), frozen, evaluated **once** on test (May–Jun, n=74); CIs use event-cluster bootstrap |

Strategy: blend `p = α·p_model + (1−α)·p_market`, buy the cheap side when `|p − p_market| > τ`, 1¢ slippage, flat $100 stakes. Frozen params from train: α=0.75, τ=0.03, no evidence gate.

### Out-of-sample results (test: May–Jun 2026)

| Metric | Strategy | always-NO | buy-favorite (≥90¢) | pure model (α=1, τ=0.05) |
|--------|----------|-----------|----------------------|--------------------------|
| Trades | 57 | 74 | 10 | 55 |
| P&L (flat $100) | **+$1,323** | +$838 | +$51 | +$1,361 |
| ROI | **+23.2%** | +11.3% | +5.1% | +24.7% |
| 90% CI (per-trade ROI) | −9.3% … +58.8% | −12.6% … +36.1% | +3.8% … +6.5% | −9.2% … +64.0% |
| Win rate | 49.1% | 58.1% | 100% | 49.1% |
| Max drawdown | $962 | $596 | $0 | $962 |

Train (in-sample): +$1,561 on 162 trades (+9.6% ROI). Quarter-Kelly compounding on test: $1,000 → $1,309.

### Honest read of the numbers

- **The 90% CI includes zero.** +23% ROI on 57 trades is *consistent with* edge, not proof of it. Roughly 200+ trades at this mean would be needed for significance.
- **Profit is concentrated and skewed**: the top 5 trades contribute +$2,085; the other 52 net −$762. The book wins by buying cheap sides (41/57 entries under 50¢) with ~7:1 payoffs, at a 49% win rate.
- **The NO side carried it**: +$1,278 from 40 NO trades vs +$45 from 17 YES trades, in a period whose base rate favored NO (mechanical always-NO made +11.3%). The strategy's excess over always-NO (+12pp) and its win on the YES side too are the actual evidence of model value.
- **The model is not better calibrated than the market** (test Brier: blend 0.214 vs market 0.202). The P&L comes from selective, payoff-asymmetric disagreement — not from beating the market on average.
- **No cutoff-decay pattern**: the model's Brier gap vs the market *shrinks* from Feb (+0.026) to May (+0.009) — the opposite of what parametric leakage would produce.

### Known limitations

- 57 test trades is a small book; the result needs forward (paper-trading) confirmation before real capital.
- Market universe filtered on **final** volume (post-T information) — selection, not leakage, but live deployment would select on volume-to-date.
- Universe limited to markets still open 24h before *scheduled* close with clean YES/NO resolution; early-resolving and disputed markets are excluded.
- 1¢ flat slippage approximates execution; thin books would fill worse than the last-trade price.
- GDELT is rate-limited to uselessness from this network; evidence is Wikipedia-only (60/286 markets had relevant pinned events) plus the model's pre-cutoff knowledge.
- Reproducing: `scripts/backtest.py` stages are cached and resumable — `collect → refresh → probe → predict → evaluate`. Full artifacts in `data/backtest/evaluation.json`.

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
