# DESIGN.md — Cassandra dashboard

Visual system for `dashboard/index.html` (single self-contained file, GitHub Pages).

## Theme

**Light primary, full dark support** via `prefers-color-scheme`.

Scene: a laptop screen, any hour, reading small dense numbers for under a minute. Dark
was rejected as the *category reflex* for anything finance-adjacent, and light-editorial
was rejected as the second-order reflex. What is left is what the thing actually is: a
paper-adjacent lab record with a status light on it.

Base is a true low-chroma neutral, not the cream/sand band. Chroma is tinted marginally
toward the accent's hue, never toward warmth by default.

## Color strategy

**Restrained.** Neutral surface, one accent, semantic states only. Color is never
decoration here; every colored pixel means something.

The one committed use of color: **mode**. Shadow data is rendered in a deliberately
provisional register (muted, dashed rules, hatched interval fills). Official data is
rendered in full ink. A glance distinguishes them without reading a word.

### Roles

| token | role |
|---|---|
| `--bg` / `--surface` / `--surface-2` | page, panel, inset |
| `--ink` / `--ink-2` / `--ink-3` | primary / secondary / tertiary text |
| `--accent` | current selection, primary action, the live heartbeat |
| `--ok` `--warn` `--danger` | settled-positive, staleness warning, failure and staleness error |
| `--rule` / `--rule-strong` | hairlines that carry structure in place of cards |

Contrast floor: 4.5:1 body, 3:1 large. Numeric columns use `--ink`, never `--ink-3`.

## Typography

One family: system sans for everything, plus system mono for **all numerics**. No
display face. Fixed rem scale, ratio ~1.2. Product UI, not a landing page.

- `--fs-page` 1.25rem / 600 — the single page title
- `--fs-section` 0.9375rem / 600 — section heads
- `--fs-body` 0.875rem / 400
- `--fs-label` 0.75rem / 500 — column heads and meta
- Numerics: mono, `font-variant-numeric: tabular-nums`, right-aligned in columns

Uppercase is confined to short status words and column labels. No section eyebrows.

## Layout

**Single column, rule-separated. No cards.** Cards were the previous version's mistake:
a grid of boxes that reads as empty scaffolding on the many days with no data. Hairline
rules give structure that survives having nothing in it.

Order is the user's question order: heartbeat, then performance, then the decision log.

- Page max-width 1180px, generous side gutters
- Decision log is a CSS grid that collapses to a stacked record on narrow screens
- Wide content scrolls inside its own container; the body never scrolls horizontally

## Signature elements

**The heartbeat.** A persistent status strip at the top, always the first thing read:
freshness of the last run, a live/stale/dead verdict, and time since. Under 36h calm,
36-72h warning, over 72h a loud failure. It carries a slow two-second pulse when live,
which is the page's only ambient motion and doubles as proof the page itself rendered.

**The interval bar.** ROI is drawn as its confidence interval on a shared axis with a
tick at the point estimate and a marked zero line, not as a big number. If the interval
straddles zero, that is immediately visible. This is the honest primitive of the whole
project rendered as a graphic, and it replaces the banned hero metric.

**The verdict line.** Every decision row states in plain words which rule bound it:
`edge 0.041 < 0.08 threshold`, `edge 0.412 > 0.35 divergence cap`, `price 0.985 outside
0.03-0.97`. This is the "why" the product exists to answer.

## Motion

150-250ms, `ease-out-quart`. State only: row expansion, filter changes, the heartbeat
pulse. No page-load choreography. Everything is visible by default and never gated on a
transition firing. `prefers-reduced-motion: reduce` drops the pulse to a static dot and
makes expansion instant.

## Empty state

Designed as a first-class view. It reports what the pipeline is waiting for and what
the volume floor implies about expected frequency, so zero decisions reads as a
measurement rather than a malfunction. It never uses the word "yet" alone.
