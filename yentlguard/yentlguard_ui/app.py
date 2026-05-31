"""
YentlGuard · Chainlit Interface  —  v2 "demo edition"
Mechanistic interpretability for clinical-triage LLM bias

Targets Chainlit >= 2.0  (uses ElementSidebar, CustomElement, set_starters).

What changed vs v1
------------------
• Live "Agent Flow" trace ... every supervisor -> sub-agent -> tool hop is shown
                              as it streams (inline AgentFlow custom element).
• Orchestration step tree .. ORCHESTRATION -> <agent> -> <tool> nested steps,
                              each tool call timed + status-marked.
• Starter prompt buttons ... one-click demo prompts via @cl.set_starters.
• Metric gauges ............ delta-M / CRR / TAR / gap / PSS as threshold-
                              coloured radial gauges (MetricPulse).
• Report toolbar ........... zoom / fullscreen / open / download (ReportViewer).
• Mock mode upgraded ....... writes a self-contained demo report and exercises
                              the full multi-agent flow with no GCP creds.

Run:
    PYTHONPATH=.. chainlit run app.py
"""

import asyncio
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import chainlit as cl

# Triple-backtick fence built without literal backticks (keeps this file easy to
# embed in docs); functionally identical to a normal code fence.
_FENCE = chr(96) * 3


# -- ADK runner setup ----------------------------------------------------------
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


# -- Report file watcher -------------------------------------------------------
_RESULTS_DIRS = [
    Path(os.environ.get("YENTLGUARD_RESULTS_DIR", "results")),
    Path("yentlguard_analysis"),
]
for _d in _RESULTS_DIRS:
    _d.mkdir(exist_ok=True)

_PUBLIC_REPORTS = Path("public/reports")
_PUBLIC_REPORTS.mkdir(parents=True, exist_ok=True)

for _results_dir in _RESULTS_DIRS:
    _link = _PUBLIC_REPORTS / _results_dir.name
    if not _link.exists():
        try:
            _link.symlink_to(_results_dir.resolve())
        except OSError:
            pass  # Windows fallback: files copied on demand


def _find_latest_report():
    candidates = []
    for d in _RESULTS_DIRS:
        candidates.extend(d.glob("yentlguard_analysis_*.html"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


async def _push_report_to_sidebar(report_path: Path) -> None:
    parent_name = report_path.parent.name
    rel = f"/public/reports/{parent_name}/{report_path.name}"
    # Embed the report HTML directly (srcDoc) so it never depends on Chainlit
    # static-file serving / symlinks (which can return {"detail":"Invalid filename"}).
    try:
        html = report_path.read_text(encoding="utf-8")
    except OSError:
        html = ""
    report_el = cl.CustomElement(
        name="ReportViewer",
        props={
            "html": html,
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
    # Remember the active report so it stays pinned on later turns.
    cl.user_session.set("current_report", str(report_path))


async def _push_welcome_sidebar() -> None:
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
    await cl.ElementSidebar.set_title("YENTLGUARD")
    await cl.ElementSidebar.set_elements([welcome_el], key="instrument-panel")


async def _ensure_sidebar() -> None:
    """Keep the analysis panel pinned across turns.

    Chainlit can drop / collapse the ElementSidebar when a new message starts,
    which makes the report appear to "close". Re-assert whatever should show:
    the active report if one has loaded, otherwise the welcome panel.
    """
    current = cl.user_session.get("current_report")
    if current and Path(current).exists():
        await _push_report_to_sidebar(Path(current))
        return
    latest = _find_latest_report()
    if latest:
        await _push_report_to_sidebar(latest)
    else:
        await _push_welcome_sidebar()


# -- Metric extraction ---------------------------------------------------------
_METRIC_PATTERNS = {
    "delta_m": re.compile(r"ΔM[=:\s]+([0-9.]+)"),
    "crr":     re.compile(r"CRR[=:\s]+([0-9.]+)"),
    "tar":     re.compile(r"TAR[=:\s]+([0-9.]+)"),
    "gap":     re.compile(r"gap[=:\s]+([0-9.]+)"),
    "pss":     re.compile(r"PSS[=:\s]+([0-9.]+)"),
}


def _extract_metrics(text: str):
    found = {}
    for name, pat in _METRIC_PATTERNS.items():
        m = pat.search(text)
        if m:
            found[name] = m.group(1)
    return found


# -- Tool registry -------------------------------------------------------------
_TOOL_META = {
    # BigQuery
    "list_experiments":             ("BQ",  "LIST EXPERIMENTS"),
    "get_pss_summary":              ("BQ",  "PSS SUMMARY"),
    "get_gate_fire_rate":           ("BQ",  "GATE FIRE RATE"),
    "get_sycophancy_verdict":       ("BQ",  "SYCOPHANCY VERDICT"),
    "query_bigquery":               ("BQ",  "CUSTOM QUERY"),
    # Runner
    "triage_vignette":              ("RUN", "TRIAGE VIGNETTE"),
    "run_baseline":                 ("RUN", "BASELINE PASS"),
    "run_experiment":               ("RUN", "EXPERIMENT"),
    "analyze_run":                  ("RUN", "ANALYZE RUN"),
    # Phoenix function tools
    "annotate_spans_with_verdicts": ("PHX", "ANNOTATE SPANS"),
    "push_prompt_version":          ("PHX", "PUSH PROMPT"),
    "list_prompt_versions":         ("PHX", "LIST PROMPTS"),
    "create_anomaly_dataset":       ("PHX", "ANOMALY DATASET"),
    # Phoenix MCP
    "list-projects":                ("MCP", "LIST PROJECTS"),
    "get-project":                  ("MCP", "GET PROJECT"),
    "list-traces":                  ("MCP", "LIST TRACES"),
    "get-trace":                    ("MCP", "GET TRACE"),
    "get-spans":                    ("MCP", "GET SPANS"),
    "get-span-annotations":         ("MCP", "SPAN ANNOTATIONS"),
    "list-annotation-configs":      ("MCP", "ANNOTATION CONFIGS"),
    "list-sessions":                ("MCP", "LIST SESSIONS"),
    "get-session":                  ("MCP", "GET SESSION"),
    "list-prompts":                 ("MCP", "LIST PROMPTS"),
    "get-prompt":                   ("MCP", "GET PROMPT"),
    "get-latest-prompt":            ("MCP", "LATEST PROMPT"),
    "get-prompt-by-identifier":     ("MCP", "GET PROMPT"),
    "get-prompt-version":           ("MCP", "PROMPT VERSION"),
    "list-prompt-versions":         ("MCP", "PROMPT VERSIONS"),
    "get-prompt-version-by-tag":    ("MCP", "PROMPT BY TAG"),
    "list-prompt-version-tags":     ("MCP", "PROMPT TAGS"),
    "add-prompt-version-tag":       ("MCP", "TAG PROMPT"),
    "upsert-prompt":                ("MCP", "UPSERT PROMPT"),
    "list-datasets":                ("MCP", "LIST DATASETS"),
    "get-dataset":                  ("MCP", "GET DATASET"),
    "get-dataset-examples":         ("MCP", "DATASET EXAMPLES"),
    "get-dataset-experiments":      ("MCP", "DATASET EXPERIMENTS"),
    "add-dataset-examples":         ("MCP", "ADD EXAMPLES"),
    "list-experiments-for-dataset": ("MCP", "LIST EXPERIMENTS"),
    "get-experiment-by-id":         ("MCP", "GET EXPERIMENT"),
}

# Sub-agent names -> short display label
_AGENT_PREFIX = {
    "data_analyst_agent":      "analyst",
    "observability_agent":     "observ",
    "experiment_runner_agent": "runner",
    "yentlguard_agent":        "supervisor",
}

# Tool family -> accent colour (mirrors theme.css; consumed by AgentFlow.jsx)
FAMILY_COLOR = {"BQ": "#1D9E75", "RUN": "#D85A30", "MCP": "#7F77DD",
                "PHX": "#7F77DD", "TOOL": "#8b949e"}


def _tool_label(name):
    return _TOOL_META.get(name, ("TOOL", name.upper().replace("_", " ")))


def _agent_label(author):
    return _AGENT_PREFIX.get(author or "", "") or (author or "supervisor")


def _format_tool_output(raw):
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


def _step_name(tool_name, author):
    badge, label = _tool_label(tool_name)
    prefix = _AGENT_PREFIX.get(author or "", "")
    if prefix and prefix != "supervisor":
        return f"[{badge}] {label}  ·  {prefix}"
    return f"[{badge}] {label}"


# -- ADK event helpers ---------------------------------------------------------
def _get_tool_name(event):
    if hasattr(event, "tool_call") and event.tool_call:
        return getattr(event.tool_call, "name", None)
    if hasattr(event, "content") and event.content:
        for part in getattr(event.content, "parts", []):
            if hasattr(part, "function_call") and part.function_call:
                return getattr(part.function_call, "name", None)
    return None


def _get_tool_args(event):
    if hasattr(event, "tool_call") and event.tool_call:
        return getattr(event.tool_call, "args", None)
    if hasattr(event, "content") and event.content:
        for part in getattr(event.content, "parts", []):
            if hasattr(part, "function_call") and part.function_call:
                return getattr(part.function_call, "args", None)
    return None


def _get_tool_result(event):
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


def _get_text_parts(event):
    texts = []
    if hasattr(event, "content") and event.content:
        for part in getattr(event.content, "parts", []):
            if hasattr(part, "function_call") and part.function_call:
                continue
            if hasattr(part, "function_response") and part.function_response:
                continue
            if hasattr(part, "text") and part.text:
                texts.append(part.text)
    return texts


def _get_author(event):
    return getattr(event, "author", None)


def _is_final(event):
    return getattr(event, "is_final_response", False) or getattr(
        event, "turn_complete", False
    )


# -- Starter prompts (onboarding) ----------------------------------------------
@cl.set_starters
async def starters():
    return [
        cl.Starter(
            label="What experiments do I have?",
            message="What experiments do I have?",
        ),
        cl.Starter(
            label="What prompt fires if I run now?",
            message="What prompt will be used if I run another experiment right now?",
        ),
        cl.Starter(
            label="Analyze my latest run",
            message="Run analyze_run on my most recent run and load the report.",
        ),
        cl.Starter(
            label="Sycophancy verdict breakdown",
            message="Give me the sycophancy verdict breakdown for my latest experiment.",
        ),
        cl.Starter(
            label="Annotate spans with verdicts",
            message="Annotate the spans from my latest run with sycophancy verdicts.",
        ),
    ]


# -- Chainlit lifecycle --------------------------------------------------------
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

    # IMPORTANT: do NOT send a cl.Message() here. Chainlit only renders the
    # @cl.set_starters prompts while the thread is EMPTY — sending any message
    # on chat start makes the thread non-empty and hides the starters. The
    # welcome copy lives in chainlit.md, which shows on the empty starter screen.


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")

    # Pin the report/instrument panel so it doesn't close when a turn starts.
    await _ensure_sidebar()

    if not _ADK_AVAILABLE:
        await _handle_mock(message.content)
        return

    turn_start = time.monotonic()

    # Live agent-flow trace (inline, updates as events stream in).
    flow_state = {"events": [], "running": True, "elapsed": 0.0,
                  "agents": 0, "tools": 0}
    flow_el = cl.CustomElement(name="AgentFlow", props=dict(flow_state),
                               display="inline")
    flow_msg = cl.Message(content="", elements=[flow_el], author="YentlGuard")
    await flow_msg.send()

    async def _refresh_flow():
        flow_state["elapsed"] = round(time.monotonic() - turn_start, 1)
        flow_el.props = dict(flow_state)
        try:
            await flow_el.update()
        except Exception:
            pass

    # Orchestration step tree: ORCHESTRATION -> <agent> -> <tool>.
    orchestration = cl.Step(name="◆ ORCHESTRATION", type="run")
    await orchestration.send()

    response_msg = cl.Message(content="", author="YentlGuard")
    await response_msg.send()

    full_text = ""
    open_tools = []          # stack of {step, tool, t0, ev}
    agent_steps = {}         # author -> its run step
    seen_authors = set()

    def _push_event(**kw):
        flow_state["events"].append(kw)
        return len(flow_state["events"]) - 1

    async def _ensure_agent_step(author):
        key = author or "yentlguard_agent"
        if key not in agent_steps:
            label = _agent_label(author)
            astep = cl.Step(name=f"▸ {label}", type="run",
                            parent_id=orchestration.id)
            await astep.send()
            agent_steps[key] = astep
            if key not in seen_authors:
                seen_authors.add(key)
                flow_state["agents"] = len(seen_authors)
                _push_event(kind="agent", agent=label, status="active")
                await _refresh_flow()
        return agent_steps[key]

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

        # -- Tool call -> open a nested step + flow node --------------------
        if tool_name and not tool_result:
            astep = await _ensure_agent_step(author)
            badge, label = _tool_label(tool_name)
            args = _get_tool_args(event)
            step = cl.Step(name=_step_name(tool_name, author), type="tool",
                           parent_id=astep.id, show_input=True)
            await step.send()
            try:
                step.input = (f"{_FENCE}json\n{json.dumps(args, indent=2)}\n{_FENCE}"
                              if args else "—")
            except Exception:
                step.input = str(args or "")
            await step.update()

            ev = _push_event(kind="tool", agent=_agent_label(author),
                             badge=badge, label=label, tool=tool_name,
                             status="running", duration=None)
            flow_state["tools"] += 1
            await _refresh_flow()
            open_tools.append({"step": step, "tool": tool_name,
                               "t0": time.monotonic(), "ev": ev})

        # -- Tool result -> close the matching step + flow node ------------
        if tool_result and open_tools:
            entry = open_tools.pop()
            step = entry["step"]
            dur = time.monotonic() - entry["t0"]
            step.output = f"{_FENCE}\n{_format_tool_output(tool_result)}\n{_FENCE}"
            await step.update()

            node = flow_state["events"][entry["ev"]]
            node["status"] = "done"
            node["duration"] = round(dur, 2)
            await _refresh_flow()

            if entry["tool"] == "analyze_run":
                await asyncio.sleep(0.8)  # let the filesystem flush
                new_report = _find_latest_report()
                if new_report:
                    await _push_report_to_sidebar(new_report)
                    await cl.Message(
                        content=f"📊 Report loaded → right panel · `{new_report.name}`",
                        author="YentlGuard",
                    ).send()

        # -- Text streaming ------------------------------------------------
        if text_parts:
            is_supervisor = author in ("yentlguard_agent", None)
            if is_supervisor or _is_final(event):
                for chunk in text_parts:
                    await response_msg.stream_token(chunk)
                    full_text += chunk

    await response_msg.update()

    # Close any dangling steps and the orchestration summary.
    for entry in open_tools:
        try:
            entry["step"].output = "(interrupted)"
            await entry["step"].update()
        except Exception:
            pass
    for astep in agent_steps.values():
        astep.output = "done"
        await astep.update()
    orchestration.output = (
        f"{flow_state['agents']} agent(s) · {flow_state['tools']} tool call(s) · "
        f"{round(time.monotonic() - turn_start, 1)}s"
    )
    await orchestration.update()

    flow_state["running"] = False
    await _refresh_flow()

    metrics = _extract_metrics(full_text)
    if metrics:
        metric_el = cl.CustomElement(name="MetricPulse",
                                     props={"metrics": metrics}, display="inline")
        await cl.Message(content="", elements=[metric_el],
                         author="YentlGuard").send()

    # Re-assert the report panel so it remains visible after the turn.
    await _ensure_sidebar()


# -- Mock runner (no GCP creds needed) -----------------------------------------
_DEMO_REPORT_HTML = """<!doctype html><html><head><meta charset='utf-8'>
<title>YentlGuard Analysis (demo)</title>
<style>
  body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:28px;
       color:#1c2128;background:#fff;}
  h1{font-size:22px;margin:0 0 4px;} h2{font-size:15px;margin:24px 0 8px;
     color:#0f1117;border-bottom:2px solid #1D9E75;padding-bottom:4px;}
  .sub{color:#6b7280;font-size:12px;margin-bottom:20px;}
  .cards{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;}
  .card{flex:1;min-width:120px;border:1px solid #e5e7eb;border-radius:8px;
        padding:12px 14px;}
  .card .k{font-size:11px;color:#6b7280;text-transform:uppercase;
           letter-spacing:.08em;} .card .v{font-size:22px;font-weight:700;}
  .teal{color:#1D9E75;} .amber{color:#b8860b;} .coral{color:#D85A30;}
  table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px;}
  th,td{border:1px solid #e5e7eb;padding:8px 10px;text-align:left;}
  th{background:#f3f4f6;}
</style></head><body>
<h1>YentlGuard — Analysis Report (DEMO)</h1>
<div class='sub'>gemini-2.5-pro · medium budget · 70 vignettes · generated in mock mode</div>
<div class='cards'>
  <div class='card'><div class='k'>Mean ΔM</div><div class='v amber'>1.42</div></div>
  <div class='card'><div class='k'>Mean CRR</div><div class='v teal'>0.71</div></div>
  <div class='card'><div class='k'>Sycophancy gap</div><div class='v coral'>0.28</div></div>
  <div class='card'><div class='k'>Gate fire rate</div><div class='v amber'>34%</div></div>
</div>
<h2>Per-variant summary</h2>
<table><tr><th>Variant</th><th>ΔM</th><th>CRR</th><th>Verdict</th></tr>
<tr><td>female</td><td>1.61</td><td>0.68</td><td>ambiguous</td></tr>
<tr><td>nb_label_only</td><td>1.23</td><td>0.74</td><td>recovered</td></tr>
<tr><td>control</td><td>0.18</td><td>0.97</td><td>clean</td></tr></table>
<h2>Notes</h2>
<p>This is a self-contained placeholder served by the Chainlit mock runner so the
report panel demonstrates end-to-end without GCP credentials.</p>
</body></html>"""


def _write_demo_report() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _RESULTS_DIRS[0] / f"yentlguard_analysis_{ts}.html"
    out.write_text(_DEMO_REPORT_HTML, encoding="utf-8")
    return out


async def _handle_mock(query: str):
    """Exercise the full multi-agent flow with realistic fake data."""
    turn_start = time.monotonic()

    flow_state = {"events": [], "running": True, "elapsed": 0.0,
                  "agents": 0, "tools": 0}
    flow_el = cl.CustomElement(name="AgentFlow", props=dict(flow_state),
                               display="inline")
    await cl.Message(content="", elements=[flow_el], author="YentlGuard").send()

    async def _refresh():
        flow_state["elapsed"] = round(time.monotonic() - turn_start, 1)
        flow_el.props = dict(flow_state)
        try:
            await flow_el.update()
        except Exception:
            pass

    orchestration = cl.Step(name="◆ ORCHESTRATION", type="run")
    await orchestration.send()

    # Scripted: supervisor -> analyst(BQ) -> observ(MCP) -> runner(RUN).
    script = [
        ("data_analyst_agent",  "list_experiments",   '{"limit": 5}',
         '[{"experiment_id":"a1b2c3d4","label":"gemini-2.5-pro medium female",'
         '"vignette_count":70}]'),
        ("observability_agent", "get-spans",           '{"project":"yentlguard"}',
         '{"spans":128,"flagged":44}'),
        ("experiment_runner_agent", "analyze_run",     '{"run_id":"a1b2c3d4"}',
         '{"status":"ok","report":"written"}'),
    ]

    for author, tool, args_json, out_json in script:
        label = _agent_label(author)
        if label not in {e.get("agent") for e in flow_state["events"]
                         if e["kind"] == "agent"}:
            flow_state["agents"] += 1
            flow_state["events"].append({"kind": "agent", "agent": label,
                                         "status": "active"})
            await _refresh()
        astep = cl.Step(name=f"▸ {label}", type="run",
                        parent_id=orchestration.id)
        await astep.send()

        badge, tlabel = _tool_label(tool)
        step = cl.Step(name=_step_name(tool, author), type="tool",
                       parent_id=astep.id, show_input=True)
        await step.send()
        step.input = f"{_FENCE}json\n{args_json}\n{_FENCE}"
        await step.update()
        idx = len(flow_state["events"])
        flow_state["events"].append({"kind": "tool", "agent": label,
                                     "badge": badge, "label": tlabel,
                                     "status": "running", "duration": None})
        flow_state["tools"] += 1
        await _refresh()
        await asyncio.sleep(0.6)
        step.output = f"{_FENCE}json\n{_format_tool_output(out_json)}\n{_FENCE}"
        await step.update()
        flow_state["events"][idx]["status"] = "done"
        flow_state["events"][idx]["duration"] = 0.6
        await astep.update()
        await _refresh()

        if tool == "analyze_run":
            report = _write_demo_report()
            await asyncio.sleep(0.3)
            await _push_report_to_sidebar(report)
            await cl.Message(
                content=f"📊 Report loaded → right panel · `{report.name}`",
                author="YentlGuard",
            ).send()

    orchestration.output = (f"{flow_state['agents']} agent(s) · "
                            f"{flow_state['tools']} tool call(s) · "
                            f"{round(time.monotonic() - turn_start, 1)}s")
    await orchestration.update()
    flow_state["running"] = False
    await _refresh()

    msg = cl.Message(content="", author="YentlGuard")
    await msg.send()
    demo_text = (
        "Found **1 experiment batch**. Experiment `a1b2c3d4` — gemini-2.5-pro · "
        "medium budget · female + nb_label_only · 70 vignettes.\n\n"
        "Gate fire rate **34%** across female vignettes. "
        "Mean ΔM=1.42, mean CRR=0.71, sycophancy gap=0.28 — ambiguous range. "
        "The corrective prompt recovers some confidence but does not cleanly "
        "separate from the distractor controls.\n\n"
        "Suggested next: `get_sycophancy_verdict` on this experiment_id."
    )
    for chunk in [demo_text[i:i + 8] for i in range(0, len(demo_text), 8)]:
        await msg.stream_token(chunk)
        await asyncio.sleep(0.01)
    await msg.update()

    metric_el = cl.CustomElement(
        name="MetricPulse",
        props={"metrics": {"delta_m": "1.42", "crr": "0.71",
                           "tar": "0.93", "gap": "0.28", "pss": "0.40"}},
        display="inline",
    )
    await cl.Message(content="", elements=[metric_el],
                     author="YentlGuard").send()

    # Re-assert the report panel so it remains visible after the mock turn.
    await _ensure_sidebar()