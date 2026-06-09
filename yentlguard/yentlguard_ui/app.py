"""YentlGuard · Chainlit Interface  —  v2 "demo edition"

Mechanistic interpretability for clinical-triage LLM bias
Targets Chainlit >= 2.0  (uses ElementSidebar, CustomElement, set_starters).

Report loading model
--------------------
A single session variable `current_report` is the source of truth for what is
shown in the sidebar. Its value is one of:
  • "welcome"                  -> the welcome screen
  • a gs:// or https:// URI    -> a report served through the proxy route
  • a local file path string   -> a report read from disk (mock mode)

Everything goes through `show_report(target)`, which is idempotent: if `target`
is already what's shown it does nothing (this is the anti-flicker guard), and it
is the ONLY place that writes `current_report`. The `_render_*` helpers only
draw; they never touch session state, so the two can't drift apart.

Run:    PYTHONPATH=.. chainlit run app.py
"""

import asyncio
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import chainlit as cl
from google.cloud import storage as gcs

# Triple-backtick fence built without literal backticks (keeps this file easy to
# embed in docs); functionally identical to a normal code fence.
_FENCE = chr(96) * 3

# Module logger. All UI-side debug is prefixed with [ui] so it's easy to grep in
# the terminal feed and pinpoint where report loading stops.
logger = logging.getLogger("yentlguard.ui")

# -- ADK runner setup ----------------------------------------------------------
try:
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types
    from yentlguard.agent.yentlguard_agent.agent import root_agent

    _runner = InMemoryRunner(agent=root_agent, app_name="yentlguard")
    _ADK_AVAILABLE = True
except ImportError:
    _runner = None
    _ADK_AVAILABLE = False

from chainlit.server import app as cl_app
from fastapi import Response


@cl_app.get("/report-proxy/{bucket_name}/{blob_name:path}")
async def proxy_report(bucket_name: str, blob_name: str):
    """Proxy HTML reports from GCS to bypass Chainlit websocket size limits and CORS/Iframe restrictions."""
    logger.info("[ui] proxy_report: GET bucket=%s blob=%s", bucket_name, blob_name)
    try:
        client = gcs.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        content = blob.download_as_string()
        logger.info("[ui] proxy_report: served %d bytes for %s", len(content), blob_name)
        return Response(content=content, media_type="text/html")
    except Exception as e:
        logger.exception("[ui] proxy_report: FAILED for %s: %s", blob_name, e)
        return Response(content=f"Error loading report: {e}", status_code=500)


# -- Welcome HTML --------------------------------------------------------------
_WELCOME_HTML = """<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>Welcome to YentlGuard</title>
<style>
  body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:28px;
       color:#e6edf3;background:#0f1117;}
  h1{font-size:22px;margin:0 0 4px;}
  h2{font-size:15px;margin:24px 0 8px;
     color:#e6edf3;border-bottom:2px solid #1D9E75;padding-bottom:4px;}
  p { line-height: 1.5; font-size: 14px; }
  ul { line-height: 1.5; font-size: 14px; }
</style>
</head>
<body>
<h1>YentlGuard</h1>
<p><strong>Mechanistic interpretability for clinical-triage LLM bias</strong></p>
<p>YentlGuard probes how clinical-triage language models shift confidence under demographic and sycophancy pressure and surfaces it with measurable signals:</p>
<ul>
  <li><strong>\u0394M</strong> confidence-margin shift between paired vignettes</li>
  <li><strong>CRR</strong> confidence recovery rate after a corrective prompt</li>
  <li><strong>TAR</strong>  thought-allocation ratio across reasoning traces</li>
  <li><strong>Sycophancy gap</strong>  divergence under social pressure</li>
</ul>
<h2>How to drive this console</h2>
<p>1. Type a prompt in the chat window to the left.<br>2. Watch the <strong>Agent Flow</strong> trace stream in real time.<br>3. When an analysis finishes, the report will replace this view automatically.</p>
</body>
</html>"""


_SAMPLES_MARKDOWN = """- *List available Phoenix datasets*
- *What are the non-zero gate fire rates for the last week?*
- *Generate a sycophancy verdict for the non-zero gate fire rates for the last week.*
- *Run a test for gemini-2.5-flash with low thinking budget for male vignettes.*
- *Create an anomaly dataset with likely sycophancy verdicts for the latest run.*
- *Generate a full analysis report for the latest run.*
"""

# -- Render helpers: draw ONLY, never write session state ----------------------
async def _render_welcome() -> None:
    el = cl.CustomElement(
        name="ReportViewer",
        props={"html": _WELCOME_HTML, "src": "", "title": "Welcome", "timestamp": ""},
        display="side",
    )
    samples_el = cl.Text(name="Sample Requests", content=_SAMPLES_MARKDOWN, display="side")
    await cl.ElementSidebar.set_title("ABOUT YENTLGUARD")
    await cl.ElementSidebar.set_elements([samples_el, el], key=f"welcome-{uuid4().hex[:8]}")


async def _render_uri(report_uri: str) -> None:
    if report_uri.startswith("gs://"):
        _, _, bucket_name, *parts = report_uri.split("/")
    else:  # https://storage.googleapis.com/...
        path = report_uri[len("https://storage.googleapis.com/"):]
        bucket_name, *parts = path.split("/")
    blob_name = "/".join(parts)
    logger.info("[ui] _render_uri: uri=%s -> bucket=%s blob=%s", report_uri, bucket_name, blob_name)

    # Download the report HTML in-process and hand it to ReportViewer via the
    # `html` prop -- the SAME path the welcome screen uses. We deliberately do
    # NOT point an iframe at /report-proxy/... : that relative URL resolves
    # against the Chainlit app's own origin and is shadowed by Chainlit's SPA
    # catch-all route, so the iframe just reloads the whole app -- which is the
    # infinite "windows inside windows" nesting.
    try:
        html = gcs.Client().bucket(bucket_name).blob(blob_name).download_as_text()
        logger.info("[ui] _render_uri: downloaded %d chars for %s", len(html), blob_name)
    except Exception as e:
        logger.exception("[ui] _render_uri: FAILED to download %s: %s", blob_name, e)
        html = "<p style='color:#e6edf3;font-family:sans-serif;padding:24px'>Failed to load report: " + str(e) + "</p>"

    el = cl.CustomElement(
        name="ReportViewer",
        props={
            "html": html,
            "src": "",
            "title": "Analysis Report",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        },
        display="side",
    )
    samples_el = cl.Text(name="Sample Requests", content=_SAMPLES_MARKDOWN, display="side")
    await cl.ElementSidebar.set_title("ANALYSIS REPORT")
    await cl.ElementSidebar.set_elements([samples_el, el], key=f"report-{uuid4().hex[:8]}")


async def _render_path(report_path: Path) -> None:
    html = report_path.read_text(encoding="utf-8")
    el = cl.CustomElement(
        name="ReportViewer",
        props={
            "html": html,
            "src": "",
            "title": report_path.stem.replace("yentlguard_analysis_", "Analysis "),
            "timestamp": datetime.fromtimestamp(
                report_path.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC"),
        },
        display="side",
    )
    samples_el = cl.Text(name="Sample Requests", content=_SAMPLES_MARKDOWN, display="side")
    await cl.ElementSidebar.set_title("ANALYSIS REPORT")
    await cl.ElementSidebar.set_elements([samples_el, el], key=f"report-{uuid4().hex[:8]}")


# -- The ONE entry point. Everything goes through here. ------------------------
async def show_report(target: str | None) -> None:
    """Show `target` in the sidebar. Idempotent: a no-op if it's already shown.

    target: "welcome", a gs://.../https:// report URI, or a local file path.
    This is the only function that writes `current_report`.
    """
    target = target or "welcome"
    logger.info("[ui] show_report: requested=%r current=%r",
                target, cl.user_session.get("current_report"))

    # Already on screen -> don't touch the DOM (anti-flicker guard).
    if target == cl.user_session.get("current_report"):
        logger.info("[ui] show_report: no-op, already showing %r", target)
        return

    if target == "welcome":
        logger.info("[ui] show_report: branch=WELCOME")
        await _render_welcome()
    elif target.startswith(("gs://", "https://storage.googleapis.com/")):
        logger.info("[ui] show_report: branch=URI %s", target)
        await _render_uri(target)
    elif Path(target).exists():
        logger.info("[ui] show_report: branch=LOCAL_PATH %s", target)
        await _render_path(Path(target))
    else:  # a path was passed but the file is gone -> fall back gracefully
        logger.warning("[ui] show_report: target %r is neither a storage URI nor an existing path -> falling back to WELCOME", target)
        await _render_welcome()
        target = "welcome"

    cl.user_session.set("current_report", target)
    logger.info("[ui] show_report: current_report=%r", target)


# GCS location of generated reports (override via env if your bucket differs).
_REPORT_BUCKET = os.environ.get("YENTLGUARD_BUCKET", "yentlguard-analysis")
_REPORT_PREFIX = os.environ.get("YENTLGUARD_REPORT_PREFIX", "reports/")


def _latest_gcs_report() -> str | None:
    """Return the https URL of the newest .html report in the GCS bucket, or
    None if the bucket has no reports yet. Reports live only in GCS
    (gs://<bucket>/reports/...), so this is what the welcome screen uses to
    auto-load the most recent report on startup.
    """
    try:
        client = gcs.Client()
        blobs = [
            b for b in client.list_blobs(_REPORT_BUCKET, prefix=_REPORT_PREFIX)
            if b.name.endswith(".html")
        ]
        if not blobs:
            logger.warning("[ui] _latest_gcs_report: NO .html reports under gs://%s/%s",
                           _REPORT_BUCKET, _REPORT_PREFIX)
            return None
        latest = max(blobs, key=lambda b: (b.updated or b.time_created, b.name))
        # Build the public https URL by concatenation (no f-string) so there is
        # no chance of a brace-escaping mistake. _render_uri proxies it.
        report_url = "https://storage.googleapis.com/" + _REPORT_BUCKET + "/" + latest.name
        logger.info("[ui] _latest_gcs_report: %d report(s); latest=%s", len(blobs), report_url)
        return report_url
        # (superseded) url = f"https://storage.googleapis.com/{_REPORT_BUCKET}/{latest.name}"
        # (superseded) return f"https://storage.googleapis.com/{_REPORT_BUCKET}/{latest.name}"
    except Exception as e:
        logger.exception("[ui] _latest_gcs_report: FAILED to list bucket %s: %s",
                         _REPORT_BUCKET, e)
        return None


# -- Metric extraction ---------------------------------------------------------
_METRIC_PATTERNS = {
    "delta_m": re.compile(r"\u0394M[=:\s]+([0-9.]+)"),
    "crr":     re.compile(r"CRR[=:\s]+([0-9.]+)"),
    "tar":     re.compile(r"TAR[=:\s]+([0-9.]+)"),
    "gap":     re.compile(r"gap[=:\s]+([0-9.]+)"),
    "delta_m_degradation": re.compile(r"delta_m_degradation[=:\s]+([0-9.]+)"),
}


def _extract_metrics(text: str):
    found = {}
    for name, pat in _METRIC_PATTERNS.items():
        m = pat.search(text)
        if m:
            found[name] = m.group(1)
    return found


_REPORT_URL_RE = re.compile(r"(?:gs://|https://storage\.googleapis\.com/)\S+?\.html")


def _find_report_uri(raw) -> str | None:
    """Pull a GCS .html report URL out of a tool result (string or JSON).

    Robust to the exact result key/nesting: just look for a storage URL ending
    in .html, preferring one under a /reports/ path.
    """
    if not raw:
        return None
    text = raw if isinstance(raw, str) else json.dumps(raw)
    matches = _REPORT_URL_RE.findall(text)
    if not matches:
        return None
    for m in matches:
        if "/reports/" in m:
            return m
    return matches[0]


# -- Tool registry -------------------------------------------------------------
_TOOL_META = {
    # BigQuery
    "list_experiments":                ("BQ",  "LIST EXPERIMENTS"),
    "get_delta_m_degradation_summary": ("BQ",  "delta_m_degradation SUMMARY"),
    "get_gate_fire_rate":              ("BQ",  "GATE FIRE RATE"),
    "get_sycophancy_verdict":          ("BQ",  "SYCOPHANCY VERDICT"),
    "query_bigquery":                  ("BQ",  "CUSTOM QUERY"),
    # Runner
    "triage_vignette":                 ("RUN", "TRIAGE VIGNETTE"),
    "run_baseline":                    ("RUN", "BASELINE PASS"),
    "run_experiment":                  ("RUN", "EXPERIMENT"),
    "analyze_run":                     ("RUN", "ANALYZE RUN"),
    # Phoenix function tools
    "annotate_spans_with_verdicts":    ("PHX", "ANNOTATE SPANS"),
    "push_prompt_version":             ("PHX", "PUSH PROMPT"),
    "list_prompt_versions":            ("PHX", "LIST PROMPTS"),
    "create_anomaly_dataset":          ("PHX", "ANOMALY DATASET"),
    # Phoenix MCP
    "list-projects":                   ("MCP", "LIST PROJECTS"),
    "get-project":                     ("MCP", "GET PROJECT"),
    "list-traces":                     ("MCP", "LIST TRACES"),
    "get-trace":                       ("MCP", "GET TRACE"),
    "get-spans":                       ("MCP", "GET SPANS"),
    "get-span-annotations":            ("MCP", "SPAN ANNOTATIONS"),
    "list-annotation-configs":         ("MCP", "ANNOTATION CONFIGS"),
    "list-sessions":                   ("MCP", "LIST SESSIONS"),
    "get-session":                     ("MCP", "GET SESSION"),
    "list-prompts":                    ("MCP", "LIST PROMPTS"),
    "get-prompt":                      ("MCP", "GET PROMPT"),
    "get-latest-prompt":               ("MCP", "LATEST PROMPT"),
    "get-prompt-by-identifier":        ("MCP", "GET PROMPT"),
    "get-prompt-version":              ("MCP", "PROMPT VERSION"),
    "list-prompt-versions":            ("MCP", "PROMPT VERSIONS"),
    "get-prompt-version-by-tag":       ("MCP", "PROMPT BY TAG"),
    "list-prompt-version-tags":        ("MCP", "PROMPT TAGS"),
    "add-prompt-version-tag":          ("MCP", "TAG PROMPT"),
    "upsert-prompt":                   ("MCP", "UPSERT PROMPT"),
    "list-datasets":                   ("MCP", "LIST DATASETS"),
    "get-dataset":                     ("MCP", "GET DATASET"),
    "get-dataset-examples":            ("MCP", "DATASET EXAMPLES"),
    "get-dataset-experiments":         ("MCP", "DATASET EXPERIMENTS"),
    "add-dataset-examples":            ("MCP", "ADD EXAMPLES"),
    "list-experiments-for-dataset":    ("MCP", "LIST EXPERIMENTS"),
    "get-experiment-by-id":            ("MCP", "GET EXPERIMENT"),
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
            return "\n".join(lines[:40]) + f"\n\n\u2026 {len(lines) - 40} more lines"
        return pretty
    except (json.JSONDecodeError, TypeError):
        if len(raw) > 800:
            return raw[:800] + f"\n\n\u2026 {len(raw) - 800} more chars"
        return raw


def _step_name(tool_name, author):
    badge, label = _tool_label(tool_name)
    prefix = _AGENT_PREFIX.get(author or "", "")
    if prefix and prefix != "supervisor":
        return f"[{badge}] {label}  \u00b7  {prefix}"
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
    import json

    def _to_str(val):
        if isinstance(val, (dict, list)):
            return json.dumps(val)
        # Handle protobuf MapComposite or Struct which might act like dicts
        if hasattr(val, "items") and callable(getattr(val, "items")):
            try:
                return json.dumps(dict(val))
            except Exception:
                pass
        return str(val)

    if hasattr(event, "tool_result") and event.tool_result:
        tr = event.tool_result
        return _to_str(tr.output) if hasattr(tr, "output") else str(tr)
    if hasattr(event, "content") and event.content:
        for part in getattr(event.content, "parts", []):
            if hasattr(part, "function_response") and part.function_response:
                resp = part.function_response
                raw = getattr(resp, "response", None) or getattr(resp, "output", None)
                if raw is not None:
                    return _to_str(raw)
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


# -- Chainlit lifecycle --------------------------------------------------------
@cl.on_chat_start
async def on_start():
    session_id = secrets.token_hex(8)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("current_report", None)
    if _ADK_AVAILABLE:
        await _runner.session_service.create_session(
            app_name="yentlguard",
            user_id="demo_user",
            session_id=session_id,
        )

    # Auto-load the latest report from the GCS bucket; else show welcome.
    startup_target = _latest_gcs_report() or "welcome"
    logger.info("[ui] on_start: session=%s startup_target=%r", session_id, startup_target)
    await show_report(startup_target)


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")

    if not _ADK_AVAILABLE:
        await _handle_mock(message.content)
        return

    turn_start = time.monotonic()
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

    orchestration = cl.Step(name="\u25c6 ORCHESTRATION", type="run")
    await orchestration.send()

    response_msg = cl.Message(content="", author="YentlGuard")
    await response_msg.send()

    full_text = ""
    open_tools = []
    agent_steps = {}
    seen_authors = set()

    def _push_event(**kw):
        flow_state["events"].append(kw)
        return len(flow_state["events"]) - 1

    async def _ensure_agent_step(author):
        key = author or "yentlguard_agent"
        if key not in agent_steps:
            label = _agent_label(author)
            astep = cl.Step(name=f"\u25b8 {label}", type="run",
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

        if tool_name and not tool_result:
            astep = await _ensure_agent_step(author)
            badge, label = _tool_label(tool_name)
            args = _get_tool_args(event)
            step = cl.Step(name=_step_name(tool_name, author), type="tool",
                           parent_id=astep.id, show_input=True)
            await step.send()
            try:
                step.input = (f"{_FENCE}json\n{json.dumps(args, indent=2)}\n{_FENCE}"
                              if args else "\u2014")
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
                try:
                    import json, ast
                    try:
                        data = json.loads(tool_result)
                    except json.JSONDecodeError:
                        data = ast.literal_eval(tool_result)
                    if isinstance(data, list) and len(data) > 0:
                        data = data[0]

                    # Robust: scan the result for a GCS .html URL regardless of
                    # which key/nesting analyze_run uses; fall back to the key.
                    new_report_uri = _find_report_uri(tool_result) or (
                        data.get("report_uri") if isinstance(data, dict) else None
                    )
                    logger.info("[ui] analyze_run: detected report_uri=%r", new_report_uri)
                    if not new_report_uri:
                        logger.warning("[ui] analyze_run: no report URL in tool_result; head=%s",
                                       str(tool_result)[:300])
                    if new_report_uri:
                        await show_report(new_report_uri)
                        await cl.Message(
                            content=f"Report loaded \u2192 right panel \u00b7 `{new_report_uri.split('/')[-1]}`",
                            author="YentlGuard",
                        ).send()
                except Exception as e:
                    logger.exception("[ui] analyze_run: FAILED to load report to sidebar: %s", e)

        if text_parts:
            is_supervisor = author in ("yentlguard_agent", None)
            if is_supervisor or _is_final(event):
                for chunk in text_parts:
                    await response_msg.stream_token(chunk)
                    full_text += chunk

    await response_msg.update()

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
        f"{flow_state['agents']} agent(s) \u00b7 {flow_state['tools']} tool call(s) \u00b7 "
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


# -- Mock runner (no GCP creds needed) -----------------------------------------
_DEMO_REPORT_HTML = """<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>YentlGuard Analysis (demo)</title>
<style>
  body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:28px;
       color:#e6edf3;background:#0f1117;}
  h1{font-size:22px;margin:0 0 4px;} h2{font-size:15px;margin:24px 0 8px;
     color:#e6edf3;border-bottom:2px solid #1D9E75;padding-bottom:4px;}
  .sub{color:#8b949e;font-size:12px;margin-bottom:20px;}
  .cards{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;}
  .card{flex:1;min-width:120px;border:1px solid #30363d;border-radius:8px;
        padding:12px 14px;background:#161b22;}
  .card .k{font-size:11px;color:#8b949e;text-transform:uppercase;
           letter-spacing:.08em;} .card .v{font-size:22px;font-weight:700;}
  .teal{color:#1D9E75;} .amber{color:#E0A33E;} .coral{color:#D85A30;}
  table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px;}
  th,td{border:1px solid #30363d;padding:8px 10px;text-align:left;}
  th{background:#161b22;}
</style>
</head>
<body>
<h1>YentlGuard \u2014 Analysis Report (DEMO)</h1>
<div class='sub'>gemini-2.5-pro \u00b7 medium budget \u00b7 70 vignettes \u00b7 generated in mock mode</div>
<div class='cards'>
  <div class='card'><div class='k'>Mean \u0394M</div><div class='v amber'>1.42</div></div>
  <div class='card'><div class='k'>Mean CRR</div><div class='v teal'>0.71</div></div>
  <div class='card'><div class='k'>Sycophancy gap</div><div class='v coral'>0.28</div></div>
  <div class='card'><div class='k'>Gate fire rate</div><div class='v amber'>34%</div></div>
</div>
<h2>Per-variant summary</h2>
<table><tr><th>Variant</th><th>\u0394M</th><th>CRR</th><th>Verdict</th></tr>
<tr><td>female</td><td>1.61</td><td>0.68</td><td>ambiguous</td></tr>
<tr><td>nb_label_only</td><td>1.23</td><td>0.74</td><td>recovered</td></tr>
<tr><td>control</td><td>0.18</td><td>0.97</td><td>clean</td></tr>
</table>
<h2>Notes</h2>
<p>This is a self-contained placeholder served by the Chainlit mock runner so the
report panel demonstrates end-to-end without GCP credentials.</p>
</body>
</html>"""


def _write_demo_report() -> Path:
    mock_dir = Path("yentlguard_analysis")
    mock_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = mock_dir / f"yentlguard_analysis_{ts}.html"
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

    orchestration = cl.Step(name="\u25c6 ORCHESTRATION", type="run")
    await orchestration.send()

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
        astep = cl.Step(name=f"\u25b8 {label}", type="run",
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
            await show_report(str(report))
            await cl.Message(
                content=f"Report loaded \u2192 right panel \u00b7 `{report.name}`",
                author="YentlGuard",
            ).send()

    orchestration.output = (f"{flow_state['agents']} agent(s) \u00b7 "
                            f"{flow_state['tools']} tool call(s) \u00b7 "
                            f"{round(time.monotonic() - turn_start, 1)}s")
    await orchestration.update()
    flow_state["running"] = False
    await _refresh()

    msg = cl.Message(content="", author="YentlGuard")
    await msg.send()
    demo_text = (
        "Found **1 experiment batch**. Experiment `a1b2c3d4` \u2014 gemini-2.5-pro \u00b7 "
        "medium budget \u00b7 female + nb_label_only \u00b7 70 vignettes.\n\n"
        "Gate fire rate **34%** across female vignettes. "
        "Mean \u0394M=1.42, mean CRR=0.71, sycophancy gap=0.28 \u2014 ambiguous range. "
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
                           "tar": "0.93", "gap": "0.28", "delta_m_degradation": "0.40"}},
        display="inline",
    )
    await cl.Message(content="", elements=[metric_el],
                     author="YentlGuard").send()