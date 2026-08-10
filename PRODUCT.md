# PRODUCT.md — Cassandra

> Inferred from the codebase and working session on 2026-08-10 rather than from an
> interview, at the user's request. Correct anything that misreads the intent.

## Register

**product** — the surface is a monitoring instrument for an automated pipeline. Design
serves the task; earned familiarity beats novelty.

## What it is

Cassandra forecasts Polymarket prediction markets with a market-blind LLM, trades the
disagreement against real order books, and measures itself honestly. The dashboard is
the window onto the live forward paper-trading test.

## Users & purpose

One user: the author, checking on his own experiment. Opens it on a laptop, briefly,
every day or few days. He is not trading from it; he is invigilating it.

He arrives with one question, in this order of urgency:

1. **Is it still alive?** The previous version of this pipeline died silently for seven
   weeks while its report still printed a healthy-looking table. Liveness is the
   product's first job, not a footnote.
2. **Did it do anything, and why?** For any single decision: what evidence it had, what
   probability it produced, what the market said, and which rule made the call.
3. **Is it making money?** With the uncertainty attached. A point estimate without its
   interval is a lie in this domain.

## The dominant design fact

**Most visits will show nothing happened.** At the pre-registered $500k volume floor
only ~1.3 markets/day qualify, so many days log zero decisions. The empty state is the
*default* state, not an edge case. It has to look deliberate and be informative, and
the design cannot lean on data being present.

This is why the hero element is liveness, not P&L.

## Personality

Instrument, not terminal. Honest, quiet, legible. It should feel like a well-kept lab
record: dense where density helps, calm everywhere else, and never overselling a
result. Uncertainty is displayed as prominently as the estimate.

## Anti-references

- **Trading terminals.** Dark chrome, green/red glow, tickers, candlesticks. This is a
  slow scientific experiment, not a desk.
- **The hero-metric dashboard.** Big ROI number over a card grid of stats. Wrong twice
  over: it is the saturated template, and most days there is no number to show.
- **Crypto-project aesthetics.** Gradients, glass, neon. The result is fragile and the
  presentation must not imply otherwise.
- **Apologetic empty states.** "No data yet" with a shrug illustration. Emptiness here
  carries real information and should be reported as such.

## Accessibility

None specified. Defaults applied: WCAG AA contrast for body text, full keyboard
operation of the decision log, visible focus, `prefers-reduced-motion` honored, and
state never signalled by color alone (shape, label, and position carry it too).

## Design principles

1. **Liveness outranks performance.** Staleness is the loudest thing on the page.
2. **Never state a result without its uncertainty.** The interval is the number.
3. **Shadow and official must be unmistakable.** Provisional data must never read as a
   verdict.
4. **Every decision is explainable in one expansion.** No hunting.
5. **Compute nothing in the browser.** All P&L and CIs come from Python, where the
   frozen bootstrap contract lives.
