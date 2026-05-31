# YentlGuard

**Mechanistic interpretability for clinical-triage LLM bias** 

YentlGuard probes how clinical-triage language models shift confidence under
demographic and sycophancy pressure — and surfaces it with measurable signals:

- **ΔM** — confidence-margin shift between paired vignettes
- **CRR** — confidence recovery rate after a corrective prompt
- **TAR** — thought-allocation ratio across reasoning traces
- **Sycophancy gap** — divergence under social pressure

### How to drive this console

1. Pick a **starter prompt** below the composer, or type your own.
2. Watch the **Agent Flow** trace stream in real time — every supervisor →
   sub-agent → tool hop is shown with timing and status.
3. When an analysis finishes, the **report opens automatically** in the right
   panel (zoom, fullscreen, open, download from its toolbar).

> Running without GCP credentials? The console drops into a self-contained
> **mock mode** that exercises the full multi-agent flow with demo data.