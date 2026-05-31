"""
YentlGuard HTML report generator.

Produces a single self-contained HTML file from an AnalysisResult.
All CSS, JS, and data are inlined — no external dependencies at render time.

Design: dark scientific instrument. Monospace data. Teal/coral accents.
Dense information layout that screenshots cleanly for papers.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from yentlguard.eval.analyze import AnalysisResult

# ── Colour tokens (HARMONI Lab palette) ───────────────────────────────────────
TEAL = "#1D9E75"
CORAL = "#D85A30"
VIOLET = "#7F77DD"
GRAY = "#888780"
BG = "#0f1117"
BG2 = "#161b22"
BG3 = "#1c2128"
BORDER = "#30363d"
TEXT = "#e6edf3"
TEXT2 = "#8b949e"


def _df_to_html(df: pd.DataFrame, max_rows: int = 200) -> str:
    """Render a DataFrame as a styled HTML table."""
    if df is None or df.empty:
        return '<p class="no-data">No data available for this analysis.</p>'

    display = df.head(max_rows)
    rows_html = ""
    for _, row in display.iterrows():
        cells = ""
        for col in display.columns:
            val = row[col]
            cls = ""
            if isinstance(val, float):
                formatted = f"{val:.4f}" if abs(val) < 1000 else f"{val:.2f}"
            elif isinstance(val, bool):
                formatted = "✓" if val else "✗"
                cls = "bool-true" if val else "bool-false"
            else:
                formatted = str(val) if pd.notna(val) else "—"
            cells += f'<td class="{cls}">{formatted}</td>'
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in display.columns)
    note = (
        f'<p class="truncation-note">Showing first {max_rows} of {len(df)} rows.</p>'
        if len(df) > max_rows
        else ""
    )
    return f"""
<div class="table-scroll">
  <table>
    <thead><tr>{headers}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
{note}
""".strip()


def _metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<span class="card-sub">{sub}</span>' if sub else ""
    return f"""
<div class="metric-card">
  <span class="card-label">{label}</span>
  <span class="card-value">{value}</span>
  {sub_html}
</div>""".strip()


def _section(title: str, hypothesis: str, content: str, section_id: str) -> str:
    return f"""
<section id="{section_id}">
  <div class="section-header">
    <h2>{title}</h2>
    <p class="hypothesis">{hypothesis}</p>
  </div>
  {content}
</section>""".strip()


def _overview_cards(result: AnalysisResult) -> str:
    df = result.overview
    if df.empty:
        return '<p class="no-data">No overview data.</p>'

    cards = ""
    for _, row in df.iterrows():
        label = row.get("label") or row.get("experiment_id", "")[:8]
        model = row.get("model_version", "")
        budget = row.get("thinking_budget", "—")
        acc = row.get("accuracy")
        dm = row.get("mean_delta_m")
        tar = row.get("mean_tar")
        crr = row.get("mean_crr")
        n = row.get("n_vignettes", "?")
        fired = row.get("n_gate_fired", 0)

        cards += f"""
<div class="overview-card">
  <div class="overview-model">{model}</div>
  <div class="overview-label">{label} · budget: {budget} · n={n}</div>
  <div class="overview-metrics">
    <div class="ov-m"><span>accuracy</span><strong>{f"{acc:.1%}" if acc is not None else "—"}</strong></div>
    <div class="ov-m"><span>mean ΔM</span><strong>{f"{dm:.4f}" if dm is not None else "—"}</strong></div>
    <div class="ov-m"><span>mean TAR</span><strong>{f"{tar:.4f}" if tar is not None else "—"}</strong></div>
    <div class="ov-m"><span>mean CRR</span><strong>{f"{crr:.4f}" if crr is not None else "—"}</strong></div>
    <div class="ov-m"><span>gate fires</span><strong>{int(fired) if pd.notna(fired) else 0}</strong></div>
  </div>
</div>"""
    return f'<div class="overview-grid">{cards}</div>'


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@400;700;800&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --teal:   {TEAL};
  --coral:  {CORAL};
  --violet: {VIOLET};
  --gray:   {GRAY};
  --bg:     {BG};
  --bg2:    {BG2};
  --bg3:    {BG3};
  --border: {BORDER};
  --text:   {TEXT};
  --text2:  {TEXT2};
  --mono:   'JetBrains Mono', monospace;
  --display:'Syne', sans-serif;
}}

html {{ scroll-behavior: smooth; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  font-size: 13px;
  line-height: 1.6;
  min-height: 100vh;
}}

/* ── Header ────────────────────────────────────────────────────────────── */
.report-header {{
  padding: 48px 64px 36px;
  border-bottom: 1px solid var(--border);
  position: relative;
  overflow: hidden;
}}
.report-header::before {{
  content: '';
  position: absolute;
  top: -60px; left: -60px;
  width: 320px; height: 320px;
  background: radial-gradient(circle, {TEAL}18 0%, transparent 70%);
  pointer-events: none;
}}
.header-lab {{
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.18em;
  color: var(--teal);
  text-transform: uppercase;
  margin-bottom: 12px;
}}
.header-title {{
  font-family: var(--display);
  font-size: 36px;
  font-weight: 800;
  color: var(--text);
  line-height: 1.1;
  margin-bottom: 8px;
}}
.header-subtitle {{
  font-size: 13px;
  color: var(--text2);
  max-width: 640px;
}}
.header-meta {{
  margin-top: 20px;
  display: flex;
  gap: 32px;
  flex-wrap: wrap;
}}
.meta-item {{
  display: flex;
  flex-direction: column;
  gap: 2px;
}}
.meta-label {{ font-size: 10px; letter-spacing: 0.1em; color: var(--teal); text-transform: uppercase; }}
.meta-value {{ font-size: 13px; color: var(--text); }}

/* ── Nav ─────────────────────────────────────────────────────────────── */
nav {{
  position: sticky;
  top: 0;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 0 64px;
  z-index: 100;
  display: flex;
  gap: 0;
  overflow-x: auto;
}}
nav a {{
  color: var(--text2);
  text-decoration: none;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 14px 18px;
  display: block;
  border-bottom: 2px solid transparent;
  white-space: nowrap;
  transition: color 0.15s, border-color 0.15s;
}}
nav a:hover {{ color: var(--teal); border-bottom-color: var(--teal); }}

/* ── Layout ──────────────────────────────────────────────────────────── */
main {{ padding: 48px 64px; max-width: 1400px; }}

section {{
  margin-bottom: 64px;
  padding-bottom: 64px;
  border-bottom: 1px solid var(--border);
}}
section:last-child {{ border-bottom: none; }}

.section-header {{
  margin-bottom: 28px;
}}
h2 {{
  font-family: var(--display);
  font-size: 22px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 8px;
}}
.hypothesis {{
  font-size: 12px;
  color: var(--teal);
  font-style: italic;
  max-width: 800px;
  line-height: 1.5;
}}

/* ── Overview grid ───────────────────────────────────────────────────── */
.overview-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
  margin-bottom: 32px;
}}
.overview-card {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px 24px;
}}
.overview-model {{
  font-family: var(--display);
  font-size: 15px;
  font-weight: 700;
  color: var(--teal);
  margin-bottom: 4px;
}}
.overview-label {{
  font-size: 11px;
  color: var(--text2);
  margin-bottom: 16px;
}}
.overview-metrics {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 8px;
}}
.ov-m {{
  display: flex;
  flex-direction: column;
  gap: 2px;
}}
.ov-m span {{
  font-size: 10px;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
.ov-m strong {{
  font-size: 14px;
  color: var(--text);
  font-weight: 500;
}}

/* ── Tables ──────────────────────────────────────────────────────────── */
.table-scroll {{
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 12px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}}
thead tr {{
  background: var(--bg3);
  border-bottom: 1px solid var(--border);
}}
th {{
  padding: 10px 14px;
  text-align: left;
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text2);
  white-space: nowrap;
  font-weight: 500;
}}
td {{
  padding: 8px 14px;
  border-bottom: 1px solid {BORDER}55;
  color: var(--text);
  white-space: nowrap;
  font-size: 12px;
}}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: {BG3}; }}
td.bool-true  {{ color: var(--teal); }}
td.bool-false {{ color: var(--coral); }}

.truncation-note {{
  font-size: 11px;
  color: var(--text2);
  margin-top: 6px;
  font-style: italic;
}}
.no-data {{
  color: var(--text2);
  font-style: italic;
  padding: 16px 0;
}}

/* ── Metric cards ────────────────────────────────────────────────────── */
.metric-cards {{
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 28px;
}}
.metric-card {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--teal);
  border-radius: 6px;
  padding: 14px 20px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 140px;
}}
.card-label {{
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text2);
}}
.card-value {{
  font-size: 22px;
  font-weight: 500;
  color: var(--teal);
  font-family: var(--display);
}}
.card-sub {{
  font-size: 11px;
  color: var(--text2);
}}

/* ── Run IDs ──────────────────────────────────────────────────────────── */
.run-ids {{
  margin-bottom: 32px;
}}
.run-id-row {{
  display: flex;
  gap: 12px;
  align-items: baseline;
  margin-bottom: 6px;
}}
.run-id-badge {{
  font-size: 10px;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 2px 8px;
  color: var(--violet);
  letter-spacing: 0.06em;
}}
.run-id-label {{
  font-size: 12px;
  color: var(--text2);
}}

/* ── Coral accent for Pass 2 / CRR sections ──────────────────────────── */
#h4 .section-header h2 {{ color: var(--coral); }}
#h4 .metric-card {{ border-left-color: var(--coral); }}
#h4 .card-value {{ color: var(--coral); }}

/* ── Footer ──────────────────────────────────────────────────────────── */
footer {{
  padding: 32px 64px;
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--text2);
  display: flex;
  justify-content: space-between;
}}
footer a {{ color: var(--teal); text-decoration: none; }}
"""


def generate_html_report(
    result: AnalysisResult,
    output_path: Path,
    experiment_ids: list[str],
) -> Path:
    """
    Generate the YentlGuard Analysis Report.

    Parameters
    ----------
    result:
        Computed AnalysisResult object containing all tables.
    output_path:
        Directory to write the report into.
    experiment_ids:
        List of experiment IDs included in this analysis.

    Returns
    -------
    Path to the generated HTML file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"yentlguard_analysis_{timestamp}.html"
    out_file = output_path / filename

    # ── Overview summary cards ─────────────────────────────────────────────
    n_vignettes = int(result.overview["n_vignettes"].sum()) if not result.overview.empty else 0
    n_interventions = int(result.raw_pass2.shape[0]) if not result.raw_pass2.empty else 0
    models = result.overview["model_version"].unique().tolist() if not result.overview.empty else []
    mean_crr = result.h4_crr["mean_crr"].mean() if not result.h4_crr.empty else None

    summary_cards = f"""
<div class="metric-cards">
  {_metric_card("Vignettes analyzed", str(n_vignettes))}
  {_metric_card("Models compared", str(len(models)), " · ".join(models))}
  {_metric_card("Interventions triggered", str(n_interventions), "Pass 2 corrections")}
  {_metric_card("Mean CRR", f"{mean_crr:.4f}" if mean_crr is not None else "—", "Confidence Recovery Rate")}
</div>"""

    # ── Run ID list ────────────────────────────────────────────────────────
    experiment_id_html = '<div class="run-ids">'
    for rid in experiment_ids:
        # Check if we have a label for this experiment_id
        label = result.run_labels.get(rid, rid[:8])
        experiment_id_html += f"""
          <span class="run-id-pill" title="{rid}">{label}</span>
        """
    experiment_id_html += "</div>"

    # ── Sections ───────────────────────────────────────────────────────────
    sections = []

    sections.append(
        _section(
            "Overview",
            "Aggregate accuracy, ΔM, TAR, and CRR per model and experiment batch.",
            experiment_id_html
            + summary_cards
            + _overview_cards(result)
            + _df_to_html(result.overview),
            "overview",
        )
    )

    sections.append(
        _section(
            "H1 — Reasoning Mitigation Effect",
            "Does scaling the thinking budget from low → medium → high reduce Perturbation Sensitivity Score? "
            "A decreasing PSS with higher budget supports the hypothesis that extended reasoning "
            "actively suppresses surface-level demographic token associations.",
            _df_to_html(result.h1_thinking_budget),
            "h1",
        )
    )

    h2_content = _df_to_html(result.h2_tar_friction)
    if result.h2_tar_friction.empty:
        h2_content = (
            '<div class="metric-card" style="border-left-color: var(--gray);">'
            '<span class="card-label">TAR Undefined</span>'
            '<p style="color: var(--text2); margin-top: 8px;">'
            "Thought Allocation Ratio (TAR) requires a thinking budget. "
            "The models in this analysis were run with <code>thinking_budget=None</code>, "
            "so TAR and cognitive friction cannot be measured.</p>"
            "</div>"
        )

    sections.append(
        _section(
            "H2 — Demographic Cognitive Friction",
            "Does the presence of a demographic label trigger higher Thought Allocation Ratio? "
            "If female chest-pain presentations produce higher TAR than male presentations, "
            "the model is spending more reasoning compute reconciling demographic schema before committing.",
            h2_content,
            "h2",
        )
    )

    sections.append(
        _section(
            "H3 — Mathematical Boundary Invariance",
            "Does Gemini 3.1 Pro maintain wider ΔM under demographic perturbation than 2.5 Pro, "
            "particularly at the safety-critical ESI 2 ↔ 3 boundary? "
            "A consistently wider margin under perturbation indicates the newer model "
            "commits more firmly to triage decisions regardless of demographic signal.",
            _df_to_html(result.h3_delta_m),
            "h3",
        )
    )

    sections.append(
        _section(
            "H4 — Selective Surgery via CRR",
            "Does vital-sign-foregrounding corrective re-prompting recover the nb_ambiguous "
            "confidence baseline? CRR = 1.0 indicates full recovery; CRR < 0.1 indicates "
            "the demographic token's influence on confidence cannot be overcome by prompt intervention alone.",
            _df_to_html(result.h4_crr),
            "h4",
        )
    )

    sections.append(
        _section(
            "Sycophancy Control Analysis",
            "CRR (Pass 2 corrective) vs. three demographically-blind distractor prompts. "
            "The crr_vs_distractor_gap is the key signal: a large positive gap means "
            "the corrective prompt's explicit demographic suppression is doing real "
            "mechanistic work beyond generic authoritative re-prompting. "
            "A gap near zero is evidence that CRR is measuring directive compliance, "
            "not genuine debiasing — the primary methodological threat to validity "
            "of the Selective Surgery framing.",
            _df_to_html(result.sycophancy),
            "sycophancy",
        )
    )

    sections.append(
        _section(
            "Gate Statistics",
            "Distribution of correction gate decisions across models, budgets, and clinical categories. "
            "Gate fire rate = proportion of female/nb vignettes where ΔM fell below threshold, "
            "triggering a corrective re-prompt.",
            _df_to_html(result.gate_stats),
            "gate",
        )
    )

    sections.append(
        _section(
            "Cross-model vignette pivot",
            "Vignette-level side-by-side comparison across all model versions in this analysis. "
            "Use this table to identify specific vignette IDs where models diverge — "
            "high-disagreement vignettes are candidates for qualitative case study.",
            _df_to_html(result.cross_model, max_rows=100),
            "pivot",
        )
    )

    body_content = "\n".join(sections)
    nav_links = "".join(
        [
            '<a href="#overview">Overview</a>',
            '<a href="#h1">H1 · Reasoning</a>',
            '<a href="#h2">H2 · Friction</a>',
            '<a href="#h3">H3 · Boundary</a>',
            '<a href="#h4">H4 · Recovery</a>',
            '<a href="#sycophancy">Sycophancy</a>',
            '<a href="#gate">Gate stats</a>',
            '<a href="#pivot">Vignette pivot</a>',
        ]
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YentlGuard Analysis Report — {generated_at}</title>
<style>{CSS}</style>
</head>
<body>

<header class="report-header">
  <div class="header-lab">YentlGuard</div>
  <h1 class="header-title">Mechanistic Interpretability<br>Analysis Report</h1>
  <p class="header-subtitle">
    Token-level confidence margins, Thought Allocation Ratios, and Confidence Recovery Rates
    across Gemini model generations on YentlBench clinical triage vignettes.
  </p>
  <div class="header-meta">
    <div class="meta-item">
      <span class="meta-label">Generated</span>
      <span class="meta-value">{generated_at}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Models</span>
      <span class="meta-value">{" · ".join(models) if models else "—"}</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">Run IDs</span>
      <span class="meta-value">{len(experiment_ids)} experiment batch(es)</span>
    </div>
  </div>
</header>

<nav>{nav_links}</nav>

<main>{body_content}</main>

<footer>
  <span>YentlGuard · <a href="https://harmonilab.org">HARMONI Lab</a> · harmonilab.org</span>
  <span>Generated {generated_at}</span>
</footer>

</body>
</html>"""

    out_file.write_text(html, encoding="utf-8")
    return out_file
