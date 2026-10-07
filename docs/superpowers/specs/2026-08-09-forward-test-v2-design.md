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
| Universe | scan: any category, **volume-to-date ≥ $20k** at scan (the backtest's collect floor); **PRIMARY** population: final volume ≥ $500k at resolution (the backtest's evaluation rule), evidence channel on; **SECONDARY**: volume ≥ $500k at decision time | finding 2 sets the $500k level; shadow-period finding (2026-09-25): the to-date floor cut supply to ~8% of the backtest-equivalent (0.86 vs 8.4 markets/day) and excluded game-day sports (76% of backtest trades) |
| Upper volume cap | none | finding 4, insufficient evidence |
| Cluster key | normalized `category` slug, never `market_id` | finding 3 |
| Model | `gpt-5.6-luna`, `reasoning={"effort":"none"}` | cheapest current tier; effort is the dominant cost lever |
| Output | Responses API, `text.format` strict `json_schema` | removes regex-scrape and the NaN hole |
| Arms | **one** (no web search) | v1's web arm took 0 trades from 8; hosted web_search is ~$60/mo and breaks the time-machine property |
| Compute | GitHub Actions, cron every 6h (`17 */6 * * *`) | free on public repos; secret store fixes the Keychain bug; 6h cadence keeps the realised horizon near 24h (see Runner) |
| Dashboard | single static HTML → GitHub Pages | `frontend/` has never built; `App.tsx` never existed |
| Storage | `decisions.jsonl` committed back to repo | free, versioned, auditable |
| Repo | public | required for free Actions + Pages |

Cost: **~$0.50/month** LLM, $0 infrastructure. Phase 0 re-baseline: **$0.32** via Batch.

## Frozen strategy

Unchanged from `evaluation.json` `frozen_params`:

```
alpha=1.0  threshold=0.08  slippage=0.01
min_price=0.03  max_price=0.97  max_divergence=0.35
stake=$100 flat
```

Effective rule (α=1.0 ⇒ `edge == p_model − p_market`):

```
trade iff  0.03 ≤ p_market ≤ 0.97
      AND  0.08 ≤ |p_model − p_market| ≤ 0.35
buy YES if p_model > p_market else NO
```

The ≥$500k rule is a **population** rule, applied the way the backtest applied it: to
**final** volume, at evaluation (`PRIMARY_MIN_FINAL_VOLUME`, report time). The scan
floor is the backtest's collect floor, volume-to-date ≥ $20k (`SCAN_MIN_VOLUME`). The
shadow period ran the $500k rule as a volume-to-date gate at decision time; that rule
survives only as the SECONDARY population (`MIN_VOLUME`). See *Shadow-period findings*.

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
3  shadow         every 6h, logged, labelled, not counted
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

*Update 2026-09-25:* the shadow period has since logged 38 decisions and settled 4, and
exposed five defects (see *Shadow-period findings*). All are fixed in shadow. The gate is
now the checklist under *Pre-registration (draft)*; the file still does not exist.

## Runner

`scripts/forward_test.py`, subcommands `scan` / `settle` / `report`, triggered by
`.github/workflows/forward-test.yml` on cron **`17 */6 * * *`** (00:17, 06:17, 12:17,
18:17 UTC).

**Window and horizon.** `scan` takes markets closing in **[12h, 27h]** (`--min-hours 12
--max-hours 27`, `--max-markets 150`). The nominal horizon is 24h — the backtest's T. At a
6h cadence each market is first seen with (21, 27]h to close, so the realised horizon
is unimodal around 24h instead of the 14h/34h bimodal split the daily cron produced
under [12h, 36h]. The 15h of window slack tolerates up to 15h between runs (two dropped
or drifted runs) before a market can pass through unseen; `coverage_gap_hours` in
`scans.jsonl` records it when one does.

**scan** — Gamma `/markets` (`active=true, closed=false`, volume-to-date ≥ $20k =
`SCAN_MIN_VOLUME`, close in-window, ≥ 3 days old, ≤ 2 open per cluster) → Wikipedia day
pages fetched **once per scan** (`fetch_days_before`, 10 days, committed cache under
`data/forward/wiki_cache`, 40-day retention) → per market `relevant_events` → luna
forecast → `decide_trade` → one row per attempted market (`open`, `no_trade` or
`forecast_failed`). Rows are written as each market completes, not after a gather, so
one market's exception cannot lose the run. CLOB status is checked (an error body is not
"illiquid"); one-sided books and other skips are *counted* in `scans.jsonl`, never
written as decision rows; `forecast_failed` rows are retried on a later scan
(`RETRYABLE_STATUSES`). Every row carries `scan_ts`, `hours_to_close`, `wiki_days_ok`,
`evidence_ok` (= `wiki_days_ok ≥ 8`), `served_model` and `usage`; `params` records the
scan floor, the primary floor and the window.

**settle** — Gamma `/markets/{id}`; `closed` is the only reliable gate (resolved
markets still report `active: true`); `parse_resolution` → P&L; ambiguous → `void`
(with `void_reason`). **Every** resolved row — `open`, `no_trade` and `forecast_failed`
alike — gets `outcome_yes`, `ts_resolved` and `volume_final`, so calibration over all
scanned markets and the PRIMARY population (final volume) are both measurable; only
`open` rows change status. `pnl_executable` is computed on `executable_notional` (not
the stake) and `executable_fill_ratio` is recorded.

**report** — `simulate()` → `summary.json` with three populations per mode
(`primary`, `secondary_at_decision`, `all`), a `calibration` block over every resolved
scanned row, `scan_health` from `scans.jsonl` and one `health` verdict. **All metrics
and CIs computed in Python**, never in JavaScript, so the frozen bootstrap contract
(2000 iters, `Random(42)`, including the off-by-one `means[int(0.95*n) - 1]`) is not
re-implemented.

**Health.** Each `scan` appends one row to `data/forward/scans.jsonl` in a `finally`:
status, exit code, window, counts (raw, Gamma pages, candidates, illiquid, book failed,
market failed, forecast attempted/ok/failed, open, no_trade), wiki days ok/cached/failed,
`coverage_gap_hours`, duration, error. Exit codes: **0** ok / no candidates / only
one-sided books skipped; **1** `forecast_outage` (attempted > 0, ok = 0), `clob_outage`
(candidates > 0, attempted = 0, book failures > 0), `evidence_degraded` (candidates > 0
and `wiki_days_ok < 8`; rows are still logged with `evidence_ok=false`),
`gamma_unavailable` (page 0 failed after 3 attempts); **2** `no_api_key` (checked before
any HTTP). `summary.health`: `failed` on crashed / gamma_unavailable / no_api_key /
forecast_outage / clob_outage; `degraded` on evidence_degraded, any forecast failure in
the last run, or > 24h since the last scan; else `ok`; `unknown` without `scans.jsonl`.
The dashboard heartbeat keys on `last_scan_ts` (warn > 24h, error > 48h; fallback
`generated_at`), not on `summary.generated_at` alone, so a failed scan followed by a
successful report can no longer read LIVE.

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
   `--skip-gdelt`), wiki-only, `allow_web=False`. Tested by
   `tests/unit/test_wiki_events.py` (status-aware fetch, 429/maxlag retry, no caching of
   failures, circuit breaker, compliant User-Agent, `fetch_days_before` →
   `relevant_events` end to end). `evidence_ok` is recorded per row (`wiki_days_ok ≥ 8`
   of 10 day pages); a run with candidates and fewer than 8/10 day pages exits 1
   (`evidence_degraded`) — its rows are logged, flagged, and excluded from the PRIMARY
   population.
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
| Actions cron drift (observed **3–6 hours** on the daily schedule, runs occasionally dropped) | schedule `:17` every 6h with a 15h window overlap, idempotent and date-keyed, never assume exactly-once; `coverage_gap_hours` logged per run |
| Scheduled workflows auto-disable after 60 days repo inactivity | the commit-back on every run should reset it — **undocumented**, so the staleness banner is the real defence |
| Concurrent runs racing the push | `concurrency: {group: forward-test-${{ github.ref }}, cancel-in-progress: false}` (per-ref), `git pull --rebase --autostash`, retry once |
| Silent death (the v1 failure) | zero successful forecasts out of > 0 attempted ⇒ **exit 1** (`forecast_outage`); a per-run record in `scans.jsonl`; `health` derived from the last scan, not from the report's timestamp |
| Evidence channel silently dead (the shadow failure: 0 wiki events on 38/38 rows) | compliant User-Agent with contact URL; cache committed to the repo; day pages prefetched once per scan with a status per day; `evidence_ok` on every row; exit 1 (`evidence_degraded`) when a run with candidates sees < 8/10 pages |
| False alarm on a one-sided book (both red shadow runs) | the outage guard counts *attempted* forecasts, not candidates; CLOB status checked so an error body is not logged as illiquid; skips counted in `scans.jsonl` |
| Runner IP throttled by Gamma/CLOB | verified from the runner during shadow; `gamma_unavailable` / `clob_outage` exit 1 rather than reporting an empty healthy scan |

## Open items to verify in Phase 0

- `reasoning: {effort: "none"}` honored on luna specifically — its default is
  undocumented; omitting may silently give `medium` and ~3× cost. One call settles it.
- Batch API + strict structured outputs in combination — undocumented. Test with a
  2-request batch before the 959-market sweep.
- Gamma/CLOB reachability from an Actions runner IP.

All three were settled by Phase 0 and the shadow period (effort honored; Batch not
needed; Gamma/CLOB reachable from the runner). Supply is addressed by the universe
change above and the sample-size target below.

## Analysis plan

**Primary endpoint.** Event-clustered 90% bootstrap CI (2000 iters, `Random(42)`,
cluster = `cluster_key`) on replica ROI — `simulate()` with the frozen params, book mid,
1¢ slippage, flat $100 — over settled rows of the PRIMARY population: `mode ==
official`, `evidence_ok`, `volume_final ≥ $500k`.

**Sample size.** Target **n = 200 settled primary trades** (`TARGET_N_TRADES_PRIMARY`).
Basis: the luna ≥$500k per-trade `pnl_per_dollar` distribution has SD ≈ 1.69. At n = 200
the 90% CI half-width is 1.645 × 1.69/√200 ≈ 0.20, so when the true ROI is +30% (the
backtest point estimate) the lower bound clears zero with ≈ 80% power.

**Looks.** **One** confirmatory look, at n = 200 settled primary trades or on
2027-09-30, whichever comes first. The dashboard and `summary.json` publish CIs
continuously, labelled *interim*; an interim "CI clear of zero" on any given day is not a
result — checking daily and stopping on the first clear interval is continuous peeking
and inflates the false-positive rate.

**Secondary endpoints** (reported, not confirmatory): SECONDARY population ROI (volume
≥ $500k at decision time); executable ROI (`pnl_executable` on `executable_notional`);
always-NO comparison over the same rows; Brier of the model vs the market over **all**
resolved scanned rows, traded or not (and the in-band subset, 0.03 ≤ mid ≤ 0.97); ROI by
`hours_to_close` bucket and by `evidence_ok`.

## Shadow-period findings (2026-09-25)

Shadow mode has run daily on Actions since 2026-08-10. As of 2026-09-25: **38 decisions
(2026-08-12 … 2026-09-24), 34 `no_trade`, 4 settled — +$69 on 4 trades, ROI +17.4%,
90% CI [−37.5%, +72.2%]**. None of this is a result; the period was for exercising the
plumbing, and it found five defects. All were fixed in shadow, before registration, so
the rows above stay shadow-only and outside every official population.

1. **Evidence channel dead on 38/38 rows** (`n_wiki_events == 0` every time). Cause: a
   contact-free User-Agent (Wikimedia's policy; refused or throttled from cloud-runner
   IPs) + a cold, gitignored cache rebuilt from nothing on every run + fetch failures
   swallowed as "no relevant events". Fix: compliant UA carrying the repo URL; a status
   per day (`fetch_day_events_status`: cache / fetched / absent / failed — failures are
   never cached); pages fetched once per scan; the cache committed under
   `data/forward/wiki_cache` (40-day retention); `wiki_days_ok` and `evidence_ok` on
   every row; exit 1 (`evidence_degraded`) when a run with candidates sees < 8/10 day
   pages. Tests: `tests/unit/test_wiki_events.py`.
2. **Universe mismatch.** The $500k floor was applied to volume-**to-date** at decision
   time; the backtest applied it to **final** volume at evaluation. Live supply was
   0.86 markets/day against a backtest-equivalent 8.4/day (~8%), and game-day sports
   markets — 76% of backtest trades, whose volume mostly arrives in the final day —
   were excluded outright. Fix: scan at the backtest's collect floor (volume-to-date
   ≥ $20k); `volume_final` recorded at settle for every resolved row; PRIMARY population
   = `evidence_ok` and `volume_final ≥ $500k` (the backtest's rule); the old gate
   survives as the SECONDARY population.
3. **Horizon and coverage.** The daily cron drifted 3–6h, so the [12h, 36h] window left
   coverage gaps and the realised horizon was bimodal (~14h / ~34h) rather than 24h; the
   month-end cohort closing 2026-09-01 fell in such a gap. Fix: cron `17 */6 * * *`,
   window [12h, 27h] (realised (21, 27]h), `hours_to_close` on every row,
   `coverage_gap_hours` on every scan.
4. **Both red runs were false alarms.** Each was a single candidate with a one-sided
   book: zero forecasts were *attempted*, and the "zero forecasts succeeded" guard
   fired anyway. Meanwhile CLOB status was never checked (an error body was logged as
   "illiquid"), skips left no record, a Gamma page-0 failure exited 0 looking healthy,
   one market's exception could abort the whole gather before any row was written, and
   `forecast_failed` rows were never retried. Fix: the guard counts attempts
   (`forecast_outage` = attempted > 0 and ok = 0); CLOB status checked; per-run counts in
   `scans.jsonl`; `gamma_unavailable` exits 1; rows written per market;
   `RETRYABLE_STATUSES`.
5. **No health record, no stopping rule, missing provenance.** The dashboard heartbeat
   keyed on `summary.generated_at`, so a failed scan followed by a successful report
   showed LIVE; `no_trade` rows never received outcomes, so calibration over scanned
   markets was unmeasured and `brier_*` were traded-subset only; the daily "CI clear of
   zero" was continuous peeking; the +31.8% luna PRIMARY figure existed in no artifact;
   the served model snapshot and token usage were not logged; CI deps were unpinned and
   tests never ran in CI. Fix: `scans.jsonl` + `health`; outcomes for every resolved row
   and a `calibration` block; the *Analysis plan* above;
   `data/backtest/evaluation.gpt-5.6-luna.primary.json` (gate item); `served_model` and
   `usage` per row; pinned CI deps with unit tests in the workflow.

## Shadow-period findings, part 2 (2026-10-07): the forecaster has no skill

From 2026-10-02 the shadow run went from +17% on 4 trades to **−61% on 44 (−$2,700)**;
PRIMARY −72% on 11, CI [−100%, −16%]. Nothing broke. The diagnosis:

1. **The model forecasts no better than a coin.** Brier: backtest 0.255, live 0.263;
   a constant 0.50 scores 0.250 (market: 0.208 backtest, 0.076 live). It answered
   exactly 0.50 on 399/959 backtest markets and "evidence: none" on 85%. Its inputs are
   a 2026-02-16 knowledge cutoff (8 months stale live), no headlines (the backtest ran
   `--skip-gdelt`, so the replica feeds `headlines=[]`), and Wikipedia events on about
   a third of markets.
2. **So the strategy fades favourites when the model knows nothing.** "Model 0.50,
   market 0.80" is a 0.30 edge inside the band, so it buys NO.
3. **The backtest edge was not the model's.** The same frozen params fed a constant
   0.50: test split (May–Jun) **+53.2%** vs luna's +31.8%; clean train (Feb 16–Apr)
   −5.7% vs +26.9%; live −22% vs −61%. The +31.8% that the registration rests on is
   mostly a period in which underdogs won, not forecasting skill. It is not evidence
   for this forecaster.
4. **Correlation amplified it.** One scan on 2026-10-04 opened 21 trades on the
   Brazilian elections, each in its own event cluster, so `max_open_per_cluster` never
   fired: 3 wins, −$1,647, 61% of all losses.

Changes, all shadow-only:

- **Coin baseline everywhere.** `report` adds `coin_baseline` (the frozen strategy
  fed p = 0.50 over every resolved scanned market in the population) to each population,
  and `calibration.brier_coin = 0.25`; `backtest.py evaluate` adds `coin_half` to the
  train and test baselines. The model adds something only if it beats both.
- **News channel** `scan --news gdelt`: GDELT headlines seen before the scan (10-day
  lookback, the backtest's source), headlines quoting Polymarket/Kalshi/odds dropped so
  the forecaster stays market-blind, `news_status` and `n_headlines` on every row,
  `n_news_*` per scan, `news_degraded` (exit 1) when every request failed, and a
  540 s budget after which the remaining markets forecast without headlines
  (`news_status = "budget"`). **Off by default** until the re-baseline below says
  it helps.
- **Per-scan trade cap** `MAX_TRADES_PER_SCAN = 6` (`--max-trades-per-scan`): trades
  past the cap, in the order forecasts complete, are logged as `no_trade` with
  `cap_skipped: true` and `capped_side`.
- **News re-baseline.** GDELT's DOC API reaches back only ~3 months and rate-limits
  the developer network, so the 959-market sample cannot get headlines.
  `.github/workflows/news-sample.yml` collects a fresh sample on an Actions runner
  (end dates 2026-07-15 … 2026-10-05, `backtest.py collect --data-dir
  data/backtest/news`). No OpenAI key is used there. The forecasts are a separate step:
  `predict` with headlines, `predict --strip-headlines --predictions-tag nonews` as the
  ablation, then `evaluate` with the frozen params. **Go criterion for `--news gdelt`:**
  with headlines, Brier < 0.25 and below the no-news ablation, and ROI above the coin
  baseline on the same markets. If it fails, news does not rescue this forecaster and
  the model itself has to change.

## Pre-registration (draft)

Commit the following as `data/forward/preregistration.json` — and nothing before it —
when every gate item below is met. `current_mode()` is existence-based: the moment the
file exists, rows are `official`. `validate_prereg()` runs at every scan start and exits
non-zero on drift of `strategy.*`, `universe.scan_min_volume_to_date`,
`universe.primary_min_final_volume`, `schedule.window_hours` or
`forecaster.model_alias`. `<…>` values are filled at commit time.

```json
{
  "schema": "cassandra-forward-prereg/1",
  "registered_at": "<ISO-8601 UTC timestamp of the registering commit>",
  "registered_commit": "<40-char sha of the commit that adds this file>",
  "forecaster": {
    "model_alias": "gpt-5.6-luna",
    "served_model_snapshot": "<served_model observed on the runner, e.g. gpt-5.6-luna-2026-08-01>",
    "reasoning_effort": "none",
    "expected_reasoning_tokens": 0,
    "prompt": "build_prompt verbatim (src/oracle/agents/forecaster.py FORECAST_PROMPT; no price, no trend)",
    "prompt_cache_key": "cassandra-forecast-v1"
  },
  "evidence": {
    "source": "wikipedia pinned day pages (last revision before D+1 00:00 UTC)",
    "lookback_days": 10,
    "max_items": 8,
    "headlines": "<[] or 'gdelt: 10-day lookback, odds headlines dropped, 540 s budget' -- per the news re-baseline>",
    "web_search": false,
    "row_is_evidence_ok_when": "wiki_days_ok >= 8",
    "cache": "data/forward/wiki_cache"
  },
  "strategy": {
    "source": "data/baselines/claude-fable-5-2026-06-10/evaluation.json#frozen_params",
    "alpha": 1.0,
    "threshold": 0.08,
    "slippage": 0.01,
    "min_price": 0.03,
    "max_price": 0.97,
    "max_divergence": 0.35,
    "evidence_gate": null,
    "stake_usd": 100,
    "entry_price": "book mid (yes_bid + yes_ask) / 2 fed to decide_trade with slippage 0.01 (replica); the depth-walked ask to $100 notional is logged separately for pnl_executable"
  },
  "universe": {
    "scan_min_volume_to_date": 20000,
    "primary_min_final_volume": 500000,
    "secondary_min_volume_at_decision": 500000,
    "min_age_days": 3,
    "max_open_per_cluster": 2,
    "max_trades_per_scan": 6,
    "cluster_key": "forward_test.cluster_key: Gamma event id when present, else normalized category slug with strike-ladder and sub-market suffixes collapsed; never market_id"
  },
  "schedule": {
    "cron_utc": "17 */6 * * *",
    "window_hours": [12, 27],
    "nominal_horizon_hours": 24
  },
  "endpoints": {
    "primary": "event-clustered 90% bootstrap CI (2000 iters, Random(42)) on replica ROI over settled PRIMARY rows: mode official, evidence_ok, volume_final >= 500000",
    "secondary": [
      "SECONDARY population ROI (volume >= 500000 at decision time)",
      "executable ROI (pnl_executable on executable_notional)",
      "always-NO comparison over the same rows",
      "coin baseline: the frozen strategy fed p = 0.50 over every resolved scanned PRIMARY market (summary coin_baseline); the forecaster must beat it",
      "Brier model vs market over all resolved scanned rows, and the in-band subset",
      "ROI by hours_to_close bucket",
      "ROI by evidence_ok"
    ]
  },
  "analysis_plan": {
    "target_n_trades_primary": 200,
    "look": "one confirmatory look at n = 200 settled primary trades or on 2027-09-30, whichever comes first",
    "interim_reporting": "CIs published continuously and labelled interim; no interim value is a result",
    "power_basis": "luna >= $500k per-trade pnl_per_dollar SD ~1.69; at n = 200 the 90% CI half-width is ~0.20, ~80% power for the lower bound to exclude 0 at true ROI +30%"
  },
  "invalidates_this_registration": [
    "any change to strategy.*, universe.* or schedule.window_hours",
    "forecaster.model_alias changes, or served_model stops matching served_model_snapshot's alias",
    "reasoning_tokens > 0 on any official row",
    "any change to FORECAST_PROMPT / build_prompt, or headlines / web search enabled",
    "a second confirmatory look, or moving the look date",
    "editing or deleting official rows in decisions.jsonl other than by settle"
  ],
  "shadow_period": {
    "from": "2026-08-10",
    "to": "<date of the last shadow scan>",
    "n_decisions": "<n>",
    "n_settled": "<n>",
    "note": "shadow rows were forecast with the Wikipedia channel dead (n_wiki_events == 0 on every row) under a $500k volume-to-date floor; they are excluded from every official population"
  }
}
```

**Gate — commit the file only when every item holds:**

- [ ] ≥ 3 consecutive scans on the Actions runner with `health == ok` and
      `wiki_days_ok ≥ 8`
- [ ] ≥ 1 full cycle under the new window and cadence: ≥ 1 decision and ≥ 1 settled
      PRIMARY row (`evidence_ok`, `volume_final ≥ $500k`)
- [ ] `data/backtest/evaluation.gpt-5.6-luna.primary.json` committed (the +31.8% figure
      lives in an artifact, not only in this document)
- [ ] CI unit tests green on pinned dependencies
- [ ] `served_model` observed on the runner and copied into `served_model_snapshot`
- [ ] The forecaster beats the coin baseline somewhere clean: the news re-baseline
      passes its go criterion, or a replacement model does (2026-10-07 finding: luna
      without news does not)

## Prerequisite

An OpenAI API key, stored as the GitHub Actions secret `OPENAI_API_KEY`. This is
what permanently fixes the class of bug that killed v1 — no Keychain, no laptop.
