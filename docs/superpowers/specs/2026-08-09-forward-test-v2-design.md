# Forward test v2 — design

Status: approved to build. Supersedes the June forward test (deleted 2026-08-05).

## Why v1 died

It ran from a laptop crontab. macOS cron cannot reach the login Keychain where the
Claude Code OAuth token lives, so `claude -p` exited non-zero with empty stderr on
every invocation from 2026-06-14 onward. Seven weeks, zero forecasts, 3 settled
trades total. The pipeline was correct end to end; only the trigger was broken.

It also was not a replica: its `decide()` omitted `max_divergence`, so it ran a
looser rule than the one that produced +30.5%.

## What we are testing

Whether the backtested edge survives on live order books, using an OpenAI
forecaster in place of `claude-fable-5`.

The Claude baseline is frozen at `data/baselines/claude-fable-5-2026-06-10/`.

### Findings that shape the design

**1. The edge is driven by liquidity, not category.** Cross-tab of the test split:

| | <$500k | ≥$500k |
|---|---|---|
| sports | 14 tr, −100% | 101 tr, +41% |
| non-sports | 2 tr, +20% | 27 tr, +59% |

Non-sports liquid markets outperform sports at the same liquidity (27 trades,
15 wins, +58.7%). The earlier "the edge is sports" reading confused liquidity with
category — World Cup markets were simply the liquid ones in that window.

**2. A ≥$500k volume floor is supported on both splits, independently.**

| bucket | train (687) | test (272) |
|---|---|---|
| $62k–250k | 19 tr, +23.1% | — |
| $250k–500k | 24 tr, **−41.1%** | 16 tr, **−85.0%** |
| $500k–1M | 116 tr, +40.7% | 54 tr, +56.4% |
| $1M–5M | 155 tr, +19.2% | 63 tr, +42.0% |
| ≥$5M | 28 tr, −13.9% | 11 tr, +5.8% |

Cumulative at a ≥$500k floor: train +19.8% → **+24.4%**, test +30.5% → **+45.0%**.

**Honesty note, load-bearing:** the volume effect was *discovered* by inspecting the
test split, then confirmed on train. Train support makes the floor legitimate to
pre-register — it is the same basis on which α, τ, and `max_divergence` were chosen.
But **+45.0% must never be reported as a clean out-of-sample result**, because the
choice was test-informed. The forward test is what provides the clean test of it.

**3. The published CI is too narrow.** `event_id == market_id` for all 959 markets,
so the "cluster bootstrap" resamples 959 clusters over 959 markets — the clustering
is inert. Ten correlated "BTC below strike" bets and four markets on one football
match are all counted as independent.

Recomputed with a real cluster key (normalized `category` slug → 116 clusters over
144 test trades):

```
point estimate                    +30.5%
CI as published (inert)           [+8.0%, +55.4%]   (reproduces evaluation.json)
CI with real event clustering     [+5.1%, +56.3%]
```

The conclusion survives. Precision was overstated; the edge claim was not.

**4. ≥$5M is weak** (−13.9% train, +5.8% test). Plausible mechanism: the most liquid
markets are the most efficiently priced. Evidence is thin (39 trades combined), so
**no upper cap** — minimise parameter changes.

## Decisions

| decision | choice | basis |
|---|---|---|
| Universe | any category, **volume ≥ $500k** at decision time | finding 2 |
| Upper volume cap | none | finding 4, insufficient evidence |
| Cluster key | normalized `category` slug, never `market_id` | finding 3 |
| Model | `gpt-5.6-luna`, `reasoning={"effort":"none"}` | cheapest current tier; effort is the dominant cost lever |
| Output | Responses API, `text.format` strict `json_schema` | removes regex-scrape and the NaN hole |
| Arms | **one** (no web search) | v1's web arm took 0 trades from 8; hosted web_search is ~$60/mo and breaks the time-machine property |
| Compute | GitHub Actions, daily cron | free on public repos; secret store fixes the Keychain bug |
| Dashboard | single static HTML → GitHub Pages | `frontend/` has never built; `App.tsx` never existed |
| Storage | `decisions.jsonl` committed back to repo | free, versioned, auditable |
| Repo | public | required for free Actions + Pages |

Cost: **~$0.50/month** LLM, $0 infrastructure. Phase 0 re-baseline: **$0.32** via Batch.

## Frozen strategy

Unchanged from `evaluation.json` `frozen_params`, plus the new volume floor:

```
alpha=1.0  threshold=0.08  slippage=0.01
min_price=0.03  max_price=0.97  max_divergence=0.35
stake=$100 flat
min_volume=500_000                       ← new, pre-registered
```

Effective rule (α=1.0 ⇒ `edge == p_model − p_market`):

```
trade iff  volume ≥ $500k
      AND  0.03 ≤ p_market ≤ 0.97
      AND  0.08 ≤ |p_model − p_market| ≤ 0.35
buy YES if p_model > p_market else NO
```

**Fidelity is structural, not aspirational.** The runner imports `decide_trade`,
`BetInput`, `StrategyParams`, `simulate` from `src/oracle/evaluation/pnl.py`. It
does not reimplement them — that is exactly how v1 lost `max_divergence`. On
startup it asserts its literal params equal `evaluation.json`'s `frozen_params` and
exits non-zero on mismatch.

## Phases

```
0  re-baseline    959 cached markets → luna, Batch API, $0.32       GO/NO-GO
1  runner         OpenAI forecaster + imported strategy
2  automation     Actions workflow + Pages dashboard
3  shadow         daily, logged, labelled, not counted
4  official       pre-registration committed, counting starts
```

Phase 0 runs on historical data, so seasonality does not touch it. If luna forecasts
badly across 959 markets we learn that for $0.32 before building anything.

Phase 0 generates predictions **once**, then evaluates them three ways (evaluation is
free — pure computation over cached rows):

- **primary:** existing frozen params + ≥$500k floor — does the edge transfer?
- secondary: re-tune on clean Mar–Apr only (406 markets)
- secondary: re-tune on full train (687)

luna's knowledge cutoff is **2026-02-16** and the train split holds 281 Feb markets,
roughly half of which resolved before that cutoff and sit inside its training data.
Re-tuning on that is contaminated. The primary evaluation avoids it entirely by not
re-tuning. The test split (May–Jun) is clean under every variant.

**Go/no-go:** proceed if primary-evaluation test ROI is positive with a CI lower
bound above −5% under *real* clustering. Otherwise stop and reconsider the model.

### Phase 0 result — 2026-08-10: **GO**

959/959 markets forecast by `gpt-5.6-luna`, zero failures, actual cost **$0.18**.
Test-split results, event-clustered CI:

| variant | trades | ROI | 90% CI | |
|---|---|---|---|---|
| **PRIMARY** frozen params, ≥$500k | 124 | **+31.8%** | [+6.9%, +58.6%] | clear |
| frozen params, all volumes | 141 | +17.6% | [−5.4%, +41.4%] | spans zero |
| re-tuned on full train, ≥$500k | 197 | +36.9% | [+11.6%, +63.8%] | clear |
| re-tuned on clean Mar–Apr, ≥$500k | 160 | +19.1% | [−2.0%, +39.9%] | spans zero |

Findings:

- **The edge transfers, but only with the volume floor.** Without it luna's CI spans
  zero. The floor was selected on Claude's data and independently improves a different
  forecaster by a comparable margin (+17.6% → +31.8%) — corroboration that it is not
  test-set overfitting.
- **Do not re-tune.** The two re-tunes disagree violently (+36.9% vs +19.1%) on
  identical predictions, differing only in `max_divergence` (1.0 vs 0.35). The grid is
  flat and the choice is noise-driven. The pre-registered params stay.
- **luna is better calibrated than Claude yet makes less money** (Brier 0.2534 vs
  0.2616; traded 0.2337 vs 0.2464). Consistent with the edge coming from extreme
  disagreement: better calibration produces fewer extreme divergences to trade.
- `reasoning={"effort":"none"}` **is** honored on luna — 0 reasoning tokens vs 119 when
  omitted (60 vs 180 output tokens). Omitting it would have roughly tripled the bill.
- Strict `json_schema` on the Responses API works. Batch was not needed at this price.

**Remaining gate before Phase 4 (official).** Phase 3 shadow mode must complete at
least one full live cycle — decisions logged from a real scan, and at least one
settled — before a pre-registration is written. The first live `scan` returned 0
candidates (expected: ~1.3/day at this floor), so the plumbing has not yet been
exercised end to end on real data. `data/forward/preregistration.json` must not be
created until it has.

## Runner

`scripts/forward_test.py`, subcommands `scan` / `settle` / `report`.

**scan** — Gamma `/markets` (`active=true, closed=false`, volume floor, close 12–36h
out) → filter → wiki evidence pinned pre-T → luna forecast → `decide_trade` → append.

**settle** — Gamma `/markets/{id}`; `closed` is the only reliable gate (resolved
markets still report `active: true`); `parse_resolution` → P&L; ambiguous → `void`.

**report** — `simulate()` → `summary.json`. **All metrics and CIs computed in Python**,
never in JavaScript, so the frozen bootstrap contract (2000 iters, `Random(42)`,
including the off-by-one `means[int(0.95*n) - 1]`) is not re-implemented.

### Forecaster swap

`build_prompt` is pure and provider-agnostic — reused verbatim as the user message.
Only the transport changes:

```python
resp = client.responses.create(
    model="gpt-5.6-luna",
    reasoning={"effort": "none"},
    input=[{"role": "user", "content": build_prompt(...)}],
    text={"format": {"type": "json_schema", "name": "forecast",
                     "schema": FORECAST_SCHEMA, "strict": True}},
    prompt_cache_key="cassandra-forecast-v1",
)
```

Python clamp to [0.01, 0.99] stays. `raw_response` must now be **persisted** —
`Forecast.to_dict()` currently drops it, and it is what makes the dashboard's "why"
possible.

### Invariants, each with a test

1. **`failed` forecasts excluded before `BetInput` is built.** `BetInput` has no
   `failed` field, so nothing downstream can catch a sentinel `p_model=0.5`. Against
   a market at 0.20 that is edge 0.30 — inside the band — and manufactures a
   confident bogus trade.
2. **Market-blindness.** No price reaches the prompt. Enforced only structurally
   today, with zero test coverage. `description` is the one leak channel: truncated
   to 500 chars, never scrubbed.
3. **Evidence channel matches the backtest**: `headlines=[]` (the run used
   `--skip-gdelt`), wiki-only, `allow_web=False`.
4. **Params match `evaluation.json`** or the process exits non-zero.
5. **Cluster key is never `market_id`.**

### Entry price

Feed `decide_trade` the book **mid** with `slippage=0.01` so the number stays
comparable to the backtest, and separately log the real ask so an executable P&L can
be computed post hoc. Observed spreads from the 16 archived v1 decisions: median
1.0¢, p90 2.0¢ — the executable version is likely slightly *cheaper* than the frozen
assumption.

`best_prices` must walk the book to $100 notional; today it ignores depth, so a
1-share ask can set the recorded fill. Check `acceptingOrders`.

## Dashboard

One self-contained `index.html`, no build step, `fetch`es `decisions.jsonl` +
`summary.json` from the same origin. Shows:

- **pipeline state** — last run, candidates scanned, forecast, traded, skipped (with
  reasons), failures
- **why, per decision** — question, evidence given to the model, `p_model`, reasoning
  text, market price, edge, which band test passed or failed, side, fill
- **performance** — cumulative P&L, ROI, CI, win rate, calibration, breakdowns by
  category and volume bucket
- **staleness banner** — mandatory, not optional (see failure modes)
- **shadow vs official** — visually unmistakable

`frontend/`, `src/oracle/api/`, and `src/oracle/observability/` are untouched: an
abandoned Phase-7 branch, one commit, never run. `npm run build` fails on a missing
`App.tsx` that was never committed.

## Failure modes

| risk | mitigation |
|---|---|
| Actions cron drift (5–30+ min, runs occasionally dropped) | schedule `:17`, idempotent and date-keyed, never assume exactly-once |
| Scheduled workflows auto-disable after 60 days repo inactivity | daily commit-back should reset it — **undocumented**, so the staleness banner is the real defence |
| Concurrent runs racing the push | `concurrency: {group: pipeline, cancel-in-progress: false}`, `git pull --rebase --autostash`, retry once |
| Silent death (the v1 failure) | zero successful forecasts ⇒ **exit non-zero**; a run that forecasts nothing must never print a healthy report |
| Runner IP throttled by Gamma/CLOB | verify from a runner before building on it |

## Open items to verify in Phase 0

- `reasoning: {effort: "none"}` honored on luna specifically — its default is
  undocumented; omitting may silently give `medium` and ~3× cost. One call settles it.
- Batch API + strict structured outputs in combination — undocumented. Test with a
  2-request batch before the 959-market sweep.
- Gamma/CLOB reachability from an Actions runner IP.
- Live supply at ≥$500k. Measured 2026-08-05: 8 markets ≥$200k and 2 ≥$1M in a
  7-day window, so expect **~2–3 trades/week** — many months to 100 settled trades.
  This is the real cost of fidelity, and it is accepted: the infrastructure is free
  and unattended, so a long run costs only patience. Report interim CIs honestly.

## Prerequisite

An OpenAI API key, stored as the GitHub Actions secret `OPENAI_API_KEY`. This is
what permanently fixes the class of bug that killed v1 — no Keychain, no laptop.
