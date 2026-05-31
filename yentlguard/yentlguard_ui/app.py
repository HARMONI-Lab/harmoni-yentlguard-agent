"""
YentlGuard Chainlit Interface
HARMONI Lab — demo-optimized agent UI

Architecture:
    Left  — streaming chat with instrument-readout tool call steps
    Right — ElementSidebar showing live analysis report (iframe via CustomElement)
             updates automatically when analyze_run completes

Multi-agent event handling:
    root_agent is a supervisor; sub-agents (data_analyst, observability,
    experiment_runner) do the actual tool calls. ADK emits events tagged with
    the author (sub-agent name). This app renders tool calls from any agent
    identically — badge is derived from tool name, not agent name.

Run:
    chainlit run app.py
"""

import asyncio
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

import chainlit as cl

# ── ADK runner setup ──────────────────────────────────────────────────────────
# Import lazily so the app starts even without a full YentlGuard install,
# falling back to a mock runner for UI development.

try:
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types
    from yentlguard.agent.yentlguard_agent.agent import root_agent

    _runner = InMemoryRunner(agent=root_agent, app_name="yentlguard")
    _ADK_AVAILABLE = True
except ImportError:
    _runner = None
    _ADK_AVAILABLE = False

# ── Report file watcher ───────────────────────────────────────────────────────
# analyze_run in runner_tools defaults to "yentlguard_analysis" as output_dir.
# We watch both that default AND the env-overridable YENTLGUARD_RESULTS_DIR so
# either path works without reconfiguration.

_RESULTS_DIRS = [
    Path(os.environ.get("YENTLGUARD_RESULTS_DIR", "results")),
    Path("yentlguard_analysis"),
]
for _d in _RESULTS_DIRS:
    _d.mkdir(exist_ok=True)

# Chainlit serves public/ as static. Symlink both report dirs into public/reports/
# so the iframe can load them regardless of which dir analyze_run wrote to.
_PUBLIC_REPORTS = Path("public/reports")
_PUBLIC_REPORTS.mkdir(parents=True, exist_ok=True)

for _results_dir in _RESULTS_DIRS:
    _link = _PUBLIC_REPORTS / _results_dir.name
    if not _link.exists():
        try:
            _link.symlink_to(_results_dir.resolve())
        except OSError:
            pass  # Windows fallback: files will be copied on demand


def _find_latest_report() -> Path | None:
    """Scan all watched directories for the most recently modified report HTML."""
    candidates = []
    for d in _RESULTS_DIRS:
        candidates.extend(d.glob("yentlguard_analysis_*.html"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


async def _push_report_to_sidebar(report_path: Path) -> None:
    """Load an analysis HTML into the right-panel sidebar via iframe."""
    # Route through the symlinked public/reports/<dir>/<file> path
    parent_name = report_path.parent.name
    rel = f"/public/reports/{parent_name}/{report_path.name}"
    report_el = cl.CustomElement(
        name="ReportViewer",
        props={
            "src": rel,
            "title": report_path.stem.replace("yentlguard_analysis_", "Analysis "),
            "timestamp": datetime.fromtimestamp(
                report_path.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC"),
        },
        display="inline",
    )
    await cl.ElementSidebar.set_title("ANALYSIS REPORT")
    await cl.ElementSidebar.set_elements([report_el], key="report-panel")


async def _push_welcome_sidebar() -> None:
    """Show the instrument panel in the sidebar before any report exists."""
    welcome_el = cl.CustomElement(
        name="InstrumentPanel",
        props={
            "status": "READY",
            "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"),
            "project": os.environ.get("YENTLGUARD_GCP_PROJECT", "—"),
            "phoenix": os.environ.get("PHOENIX_BASE_URL", "—"),
        },
        display="inline",
    )
    await cl.ElementSidebar.set_title("HARMONI LAB · YENTLGUARD")
    await cl.ElementSidebar.set_elements([welcome_el], key="instrument-panel")


# ── Metric extraction ─────────────────────────────────────────────────────────

_METRIC_PATTERNS = {
    "delta_m": re.compile(r"ΔM[=:\s]+([0-9.]+)"),
    "crr":     re.compile(r"CRR[=:\s]+([0-9.]+)"),
    "tar":     re.compile(r"TAR[=:\s]+([0-9.]+)"),
    "gap":     re.compile(r"gap[=:\s]+([0-9.]+)"),
    "pss":     re.compile(r"PSS[=:\s]+([0-9.]+)"),
}


def _extract_metrics(text: str) -> dict[str, str]:
    found = {}
    for name, pat in _METRIC_PATTERNS.items():
        m = pat.search(text)
        if m:
            found[name] = m.group(1)
    return found


# ── Tool call rendering ───────────────────────────────────────────────────────

_TOOL_META: dict[str, tuple[str, str]] = {
    # BigQuery
    "list_experiments":            ("BQ",  "LIST EXPERIMENTS"),
    "get_pss_summary":             ("BQ",  "PSS SUMMARY"),
    "get_gate_fire_rate":          ("BQ",  "GATE FIRE RATE"),
    "get_sycophancy_verdict":      ("BQ",  "SYCOPHANCY VERDICT"),
    "query_bigquery":              ("BQ",  "CUSTOM QUERY"),
    # Runner
    "triage_vignette":             ("RUN", "TRIAGE VIGNETTE"),
    "run_baseline":                ("RUN", "BASELINE PASS"),
    "run_experiment":              ("RUN", "EXPERIMENT"),
    "analyze_run":                 ("RUN", "ANALYZE RUN"),
    # Phoenix function tools
    "annotate_spans_with_verdicts":("PHX", "ANNOTATE SPANS"),
    "push_prompt_version":         ("PHX", "PUSH PROMPT"),
    "list_prompt_versions":        ("PHX", "LIST PROMPTS"),
    "create_anomaly_dataset":      ("PHX", "ANOMALY DATASET"),
    # Phoenix MCP
    "list-projects":               ("MCP", "LIST PROJECTS"),
    "get-project":                 ("MCP", "GET PROJECT"),
    "list-traces":                 ("MCP", "LIST TRACES"),
    "get-trace":                   ("MCP", "GET TRACE"),
    "get-spans":                   ("MCP", "GET SPANS"),
    "get-span-annotations":        ("MCP", "SPAN ANNOTATIONS"),
    "list-annotation-configs":     ("MCP", "ANNOTATION CONFIGS"),
    "list-sessions":               ("MCP", "LIST SESSIONS"),
    "get-session":                 ("MCP", "GET SESSION"),
    "list-prompts":                ("MCP", "LIST PROMPTS"),
    "get-prompt":                  ("MCP", "GET PROMPT"),
    "get-latest-prompt":           ("MCP", "LATEST PROMPT"),
    "get-prompt-by-identifier":    ("MCP", "GET PROMPT"),
    "get-prompt-version":          ("MCP", "PROMPT VERSION"),
    "list-prompt-versions":        ("MCP", "PROMPT VERSIONS"),
    "get-prompt-version-by-tag":   ("MCP", "PROMPT BY TAG"),
    "list-prompt-version-tags":    ("MCP", "PROMPT TAGS"),
    "add-prompt-version-tag":      ("MCP", "TAG PROMPT"),
    "upsert-prompt":               ("MCP", "UPSERT PROMPT"),
    "list-datasets":               ("MCP", "LIST DATASETS"),
    "get-dataset":                 ("MCP", "GET DATASET"),
    "get-dataset-examples":        ("MCP", "DATASET EXAMPLES"),
    "get-dataset-experiments":     ("MCP", "DATASET EXPERIMENTS"),
    "add-dataset-examples":        ("MCP", "ADD EXAMPLES"),
    "list-experiments-for-dataset":("MCP", "LIST EXPERIMENTS"),
    "get-experiment-by-id":        ("MCP", "GET EXPERIMENT"),
}

# Sub-agent names → display prefix (shown in step name alongside tool badge)
_AGENT_PREFIX: dict[str, str] = {
    "data_analyst_agent":      "analyst",
    "observability_agent":     "observ",
    "experiment_runner_agent": "runner",
    "yentlguard_agent":        "",
}


def _tool_label(name: str) -> tuple[str, str]:
    return _TOOL_META.get(name, ("TOOL", name.upper().replace("_", " ")))


def _format_tool_output(raw: str) -> str:
    """Pretty-print JSON tool output, truncating large payloads."""
    try:
        data = json.loads(raw)
        pretty = json.dumps(data, indent=2)
        lines = pretty.splitlines()
        if len(lines) > 40:
            return "\n".join(lines[:40]) + f"\n\n… {len(lines) - 40} more lines"
        return pretty
    except (json.JSONDecodeError, TypeError):
        if len(raw) > 800:
            return raw[:800] + f"\n\n… {len(raw) - 800} more chars"
        return raw


def _step_name(tool_name: str, author: str | None) -> str:
    badge, label = _tool_label(tool_name)
    prefix = _AGENT_PREFIX.get(author or "", "")
    if prefix:
        return f"[{badge}] {label}  ·  {prefix}"
    return f"[{badge}] {label}"


# ── ADK event helpers ─────────────────────────────────────────────────────────

def _get_tool_name(event) -> str | None:
    """Extract tool name from an ADK event regardless of event shape."""
    # Function tool call
    if hasattr(event, "tool_call") and event.tool_call:
        return getattr(event.tool_call, "name", None)
    # MCP tool call (surfaces as a content part with function_call)
    if hasattr(event, "content") and event.content:
        for part in getattr(event.content, "parts", []):
            if hasattr(part, "function_call") and part.function_call:
                return getattr(part.function_call, "name", None)
    return None


def _get_tool_args(event) -> dict | None:
    if hasattr(event, "tool_call") and event.tool_call:
        return getattr(event.tool_call, "args", None)
    if hasattr(event, "content") and event.content:
        for part in getattr(event.content, "parts", []):
            if hasattr(part, "function_call") and part.function_call:
                return getattr(part.function_call, "args", None)
    return None


def _get_tool_result(event) -> str | None:
    if hasattr(event, "tool_result") and event.tool_result:
        tr = event.tool_result
        return str(tr.output) if hasattr(tr, "output") else str(tr)
    if hasattr(event, "content") and event.content:
        for part in getattr(event.content, "parts", []):
            if hasattr(part, "function_response") and part.function_response:
                resp = part.function_response
                raw = getattr(resp, "response", None) or getattr(resp, "output", None)
                if raw is not None:
                    return str(raw)
    return None


def _get_text_parts(event) -> list[str]:
    texts = []
    if hasattr(event, "content") and event.content:
        for part in getattr(event.content, "parts", []):
            # Skip parts that are tool calls/responses
            if hasattr(part, "function_call") and part.function_call:
                continue
            if hasattr(part, "function_response") and part.function_response:
                continue
            if hasattr(part, "text") and part.text:
                texts.append(part.text)
    return texts


def _get_author(event) -> str | None:
    return getattr(event, "author", None)


def _is_final(event) -> bool:
    """True if this event is the terminal event in the turn."""
    return getattr(event, "is_final_response", False) or getattr(event, "turn_complete", False)


# ── Chainlit lifecycle ────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    session_id = secrets.token_hex(8)
    cl.user_session.set("session_id", session_id)

    if _ADK_AVAILABLE:
        await _runner.session_service.create_session(
            app_name="yentlguard",
            user_id="demo_user",
            session_id=session_id,
        )

    existing_report = _find_latest_report()
    if existing_report:
        await _push_report_to_sidebar(existing_report)
    else:
        await _push_welcome_sidebar()

    await cl.Message(
        content=(
            "**YentlGuard** · HARMONI Lab\n\n"
            "Mechanistic interpretability for clinical triage LLM bias. "
            "Instrument: Gemini · Metrics: ΔM, TAR, CRR, sycophancy gap.\n\n"
            "Try: *\"What experiments do I have?\"* or "
            "*\"What prompt will be used if I run now?\"*"
        ),
        author="YentlGuard",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")

    if not _ADK_AVAILABLE:
        await _handle_mock(message.content)
        return

    response_msg = cl.Message(content="", author="YentlGuard")
    await response_msg.send()

    full_text = ""
    # Stack of (Step, tool_name) — supports nested or sequential tool calls
    # from multiple sub-agents in the same turn.
    open_steps: list[tuple[cl.Step, str]] = []

    async for event in _runner.run_async(
        user_id="demo_user",
        session_id=session_id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=message.content)],
        ),
    ):
        author = _get_author(event)
        tool_name = _get_tool_name(event)
        tool_result = _get_tool_result(event)
        text_parts = _get_text_parts(event)

        # ── Tool call — open a step ───────────────────────────────────────
        if tool_name and not tool_result:
            args = _get_tool_args(event)
            step = cl.Step(
                name=_step_name(tool_name, author),
                type="tool",
                show_input=True,
            )
            await step.__aenter__()
            try:
                step.input = f"```json\n{json.dumps(args, indent=2)}\n```" if args else "—"
            except Exception:
                step.input = str(args or "")
            open_steps.append((step, tool_name))

        # ── Tool result — close the matching step ────────────────────────
        if tool_result and open_steps:
            step, completed_tool = open_steps.pop()
            step.output = f"```\n{_format_tool_output(tool_result)}\n```"
            await step.__aexit__(None, None, None)

            if completed_tool == "analyze_run":
                await asyncio.sleep(0.8)  # give filesystem time to flush
                new_report = _find_latest_report()
                if new_report:
                    await _push_report_to_sidebar(new_report)
                    await cl.Message(
                        content=f"Report loaded → right panel · `{new_report.name}`",
                        author="YentlGuard",
                    ).send()

        # ── Text streaming ────────────────────────────────────────────────
        # Only stream text from the supervisor (root) or the final sub-agent
        # response — skip intermediate tool call descriptions from sub-agents.
        if text_parts:
            is_supervisor = author in ("yentlguard_agent", None)
            is_final = _is_final(event)
            if is_supervisor or is_final:
                for chunk in text_parts:
                    await response_msg.stream_token(chunk)
                    full_text += chunk

    await response_msg.update()

    metrics = _extract_metrics(full_text)
    if metrics:
        metric_el = cl.CustomElement(
            name="MetricPulse",
            props={"metrics": metrics},
            display="inline",
        )
        await cl.Message(content="", elements=[metric_el], author="YentlGuard").send()


# ── Mock runner ───────────────────────────────────────────────────────────────

async def _handle_mock(query: str):
    """Simulates agent responses for UI development without GCP credentials."""
    await asyncio.sleep(0.3)

    async with cl.Step(name="[BQ] LIST EXPERIMENTS", type="tool") as step:
        step.input = '```json\n{"limit": 5}\n```'
        await asyncio.sleep(0.4)
        step.output = (
            '```json\n[\n  {\n'
            '    "experiment_id": "a1b2c3d4-e5f6-...",\n'
            '    "label": "gemini-2.5-pro medium female",\n'
            '    "vignette_count": 70,\n'
            '    "created_at": "2026-05-01T14:22:00Z"\n'
            '  }\n]\n```'
        )

    msg = cl.Message(content="", author="YentlGuard")
    await msg.send()
    demo_text = (
        "Found **1 experiment batch**.\n\n"
        "Experiment `a1b2c3d4` — gemini-2.5-pro · medium budget · "
        "female + nb_label_only variants · 70 vignettes · May 1 2026.\n\n"
        "Gate fire rate: **34%** across female vignettes. "
        "Mean ΔM=1.42, mean CRR=0.71, sycophancy gap=0.28 — ambiguous range. "
        "The corrective prompt is recovering some confidence but not cleanly separating "
        "from the distractor controls.\n\n"
        "Suggest: `get_sycophancy_verdict` on this experiment_id for per-vignette breakdown."
    )
    for chunk in [demo_text[i:i+8] for i in range(0, len(demo_text), 8)]:
        await msg.stream_token(chunk)
        await asyncio.sleep(0.01)
    await msg.update()

    metric_el = cl.CustomElement(
        name="MetricPulse",
        props={"metrics": {"delta_m": "1.42", "crr": "0.71", "gap": "0.28"}},
        display="inline",
    )
    await cl.Message(content="", elements=[metric_el], author="YentlGuard").send()
