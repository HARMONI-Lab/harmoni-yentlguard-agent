# YentlGuard · Chainlit Interface

Demo-optimized agent UI for HARMONI Lab's YentlGuard mechanistic interpretability framework.

## File structure

```
yentlguard_ui/
├── app.py                          ← main Chainlit application
├── .chainlit/
│   └── config.toml                 ← theme, sidebar width, feature flags
└── public/
    ├── theme.css                   ← HARMONI Lab dark instrument theme
    ├── reports/                    ← symlink → results/ (auto-created on startup)
    └── elements/
        ├── ReportViewer.jsx        ← iframe-based analysis report panel
        ├── InstrumentPanel.jsx     ← sidebar status display before first report
        └── MetricPulse.jsx         ← inline ΔM / CRR / TAR metric strip
```

## Setup

```bash
# From the project root (yentlguard/)
pip install chainlit

# Copy the UI directory alongside your yentlguard package
# (or run from within yentlguard_ui/ with yentlguard on PYTHONPATH)
cd yentlguard_ui
PYTHONPATH=.. chainlit run app.py
```

Opens at http://localhost:8000

## What the UI does

**Left panel — chat**
- Streaming agent responses with token-by-token output
- Tool calls shown as collapsible instrument-readout steps
  - `[BQ]` prefix → teal border (BigQuery tools)
  - `[RUN]` prefix → coral border (runner tools)  
  - `[MCP]` prefix → violet border (Phoenix MCP tools)
  - `[PHX]` prefix → violet border (Phoenix function tools)
- MetricPulse strip appears below any response containing ΔM / CRR / TAR values
  - Color-coded: teal = good, amber = ambiguous, coral = alert threshold

**Right panel — ElementSidebar**
- Before any report: InstrumentPanel shows system config (model, project, Phoenix URL)
- After `analyze_run` completes: ReportViewer loads the HTML report in an iframe
  - Updates automatically — no page refresh needed
  - Report is the full self-contained YentlGuard HTML with all H1–H4 tables,
    sycophancy analysis, gate statistics, and cross-model pivot

## Running without GCP (UI development)

If `yentlguard` is not installed or GCP credentials are absent, the app falls back
to a mock runner that simulates agent responses with realistic fake data.
All UI elements (tool steps, MetricPulse, sidebar) work identically in mock mode.

```bash
# No credentials needed — mock mode activates automatically
chainlit run app.py
```

## Demo script

1. Start: `PYTHONPATH=.. chainlit run app.py`
2. First prompt: *"What experiments do I have?"*
   → Shows `[BQ] LIST EXPERIMENTS` tool step, returns experiment summary
3. Follow-up: *"What prompt will be used if I run another experiment?"*
   → Shows `[MCP] LATEST PROMPT` × 4 tool steps
4. Trigger analysis: *"Run analyze_run on run-id <uuid>"*
   → Shows `[RUN] ANALYZE RUN` tool step, report loads into right panel automatically
5. Annotate: *"Annotate spans from that run with sycophancy verdicts"*
   → Shows `[BQ] SYCOPHANCY VERDICT` then `[PHX] ANNOTATE SPANS` steps

## Theming

All colors in `public/theme.css` use CSS variables at `:root`.
To change the accent: update `--teal`, `--coral`, `--violet`.
The sidebar width is set in `.chainlit/config.toml` as `element_sidebar_width`.
