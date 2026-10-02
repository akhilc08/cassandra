# Cassandra — orientation for agents

Read this before touching anything.

## What is live

- A **forward paper-trading test runs from `main` on GitHub Actions**
  (`.github/workflows/forward-test.yml`, every 6h). Whatever is on `main` runs
  next cycle. Every run commits `data/forward/{decisions,scans}.jsonl`,
  `summary.json` and `wiki_cache/` back to `main`, so **always
  `git fetch && git merge --ff-only origin/main` first** — a checkout goes stale
  within hours.
- Dashboard: https://akhilc08.github.io/cassandra/ (GitHub Pages, built from
  `dashboard/index.html` + the data files; no build step).
- Mode is **shadow** until `data/forward/preregistration.json` exists. Do not
  create that file casually: it starts the official record. The gate checklist
  and the draft are in `docs/superpowers/specs/2026-08-09-forward-test-v2-design.md`.

## What matters

| Path | Role |
|---|---|
| `scripts/forward_test.py` | live runner: `scan` / `settle` / `report` |
| `src/oracle/evaluation/pnl.py` | frozen strategy (`decide_trade`, `simulate`, bootstrap). Never re-implement it. |
| `src/oracle/agents/forecaster.py`, `forecaster_openai.py` | market-blind prompt; OpenAI transport |
| `src/oracle/ingestion/wiki_events.py` | revision-pinned Wikipedia evidence |
| `scripts/backtest.py`, `data/backtest/` | leak-controlled time-machine backtest (the evidence for the strategy) |
| `data/baselines/claude-fable-5-2026-06-10/evaluation.json` | source of the frozen params; the runner refuses to start if they drift |
| `data/backtest/evaluation.gpt-5.6-luna.primary.json` | like-for-like reference for the forward test's primary population |

## What is NOT part of the forward test

`src/oracle/{api,cache,knowledge,observability,prompts,retrieval,routing,training}`,
`frontend/`, `docker/`, `docker-compose.yml` and most of `src/oracle/agents/` are
the earlier always-on platform (Neo4j/Qdrant RAG, judge/hallucination agents,
war-room UI). It is documented in the README under "Live platform", it has never
run in CI, `frontend/` does not build, and `tests/unit/test_chunker.py` /
`test_entity_resolver.py` need spaCy. `data/baselines/platform-2026-04-01/` holds
its superseded April 2026 results. Leave it alone unless asked.

## Rules

- Frozen strategy params (`alpha=1.0 threshold=0.08 slippage=0.01 min_price=0.03
  max_price=0.97 max_divergence=0.35 stake=100`) do not change. Universe,
  window and schedule changes are allowed only while in shadow mode and must be
  reflected in the pre-registration draft.
- Never put per-market skip rows in `decisions.jsonl`; run-level facts go in
  `scans.jsonl`. Never compute metrics in the dashboard; `report` owns them.
- Tests that must stay green (CI runs them before every scan):
  `.venv/bin/python -m pytest tests/unit/test_forward.py tests/unit/test_pnl.py tests/unit/test_time_machine.py tests/unit/test_wiki_events.py tests/unit/test_forecaster_openai.py -q`
- Runner deps are pinned in `requirements-forward.txt`; bump deliberately.
- Commit messages: one short line. Do not push to `main` without being asked.
