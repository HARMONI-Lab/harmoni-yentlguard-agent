"""
Phoenix MCP toolset configuration for the YentlGuard ADK agent.

@arizeai/phoenix-mcp runs as a Node child process launched via npx.
ADK manages the subprocess lifecycle through StdioConnectionParams.

Requires: Node.js + npx in PATH.
Returns None if PHOENIX_API_KEY is not set or Node is unavailable —
the agent runs with BigQuery and Phoenix function tools only.

Tool surface exposed (full read + targeted write, as of @arizeai/phoenix-mcp v4.x):

    Projects:
        list-projects, get-project

    Traces:
        list-traces, get-trace

    Spans & Annotations:
        get-spans               — retrieve spans for a trace; supports basic filtering
                                  NOTE: custom attribute filtering (e.g. yentlguard.vignette_id)
                                  is NOT supported — use BigQuery for indexed metric lookups
        get-span-annotations    — retrieve human/LLM annotations on a specific span

    Annotation Configs:
        list-annotation-configs — inspect scoring rubrics and label schemas defined in Phoenix;
                                  useful before calling annotate-span to confirm valid label names

    Sessions:                   — not used in YentlGuard (no conversational flows), included
        list-sessions           for completeness in case multi-turn eval flows are added
        get-session

    Prompts (full versioning surface):
        list-prompts            — browse all stored prompt names
        get-prompt              — fetch latest version of a named prompt
        get-latest-prompt       — alias; fetches latest version
        get-prompt-by-identifier— fetch by name or ID
        get-prompt-version      — fetch a specific version by ID
        list-prompt-versions    — all versions of a named prompt (use before run_experiment)
        get-prompt-version-by-tag — fetch by tag (e.g. "production", "experiment-v2")
        list-prompt-version-tags — see all tags on a named prompt
        add-prompt-version-tag  — tag a version (e.g. promote to "production")
        upsert-prompt           — create or update a prompt version

    Datasets:
        list-datasets           — browse corpus and anomaly subset datasets
        get-dataset             — metadata + schema for a specific dataset
        get-dataset-examples    — retrieve actual vignette rows from a dataset;
                                  use to inspect what's inside before a targeted re-run
        get-dataset-experiments — list experiments that ran against a given dataset;
                                  the primary cross-reference from dataset → run history
        add-dataset-examples    — extend an existing dataset with new vignette rows

    Experiments:
        list-experiments-for-dataset — list all experiments for a specific dataset;
                                       preferred over BQ list_experiments when a
                                       dataset_id is known (Phoenix-native view includes
                                       eval metadata, timing, and annotations)
        get-experiment-by-id    — full experiment record including metadata,
                                  outputs, and any Phoenix-stored annotations

Base URL derivation:
    Strips /v1/traces suffix if PHOENIX_COLLECTOR_ENDPOINT was set instead
    of PHOENIX_BASE_URL. Both variables are supported.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Full tool filter — every tool the YentlGuard agent should be able to call.
# Grouped by domain for readability; order does not affect ADK behaviour.
PHOENIX_MCP_TOOL_FILTER: list[str] = [
    # ── Projects ──────────────────────────────────────────────────────────
    "list-projects",
    "get-project",
    # ── Traces ────────────────────────────────────────────────────────────
    "list-traces",
    "get-trace",
    # ── Spans & Annotations ───────────────────────────────────────────────
    "get-spans",  # was "list-spans" — renamed in v4.x
    "get-span-annotations",  # was "get-span"   — now returns annotations for a span
    # ── Annotation Configs ────────────────────────────────────────────────
    "list-annotation-configs",  # NEW — inspect scoring rubrics before annotating
    # ── Sessions ──────────────────────────────────────────────────────────
    "list-sessions",  # NEW — included for completeness / future use
    "get-session",  # NEW
    # ── Prompts (full versioning surface) ────────────────────────────────
    "list-prompts",
    "get-prompt",
    "get-latest-prompt",  # NEW — explicit latest-version fetch
    "get-prompt-by-identifier",  # NEW — fetch by name or ID
    "get-prompt-version",  # NEW — fetch a specific version by ID
    "list-prompt-versions",  # was missing from prior filter
    "get-prompt-version-by-tag",  # NEW — fetch by tag
    "list-prompt-version-tags",  # NEW — see all tags on a prompt
    "add-prompt-version-tag",  # NEW — promote a version (e.g. to "production")
    "upsert-prompt",  # write: create or update prompt version
    # ── Datasets ──────────────────────────────────────────────────────────
    "list-datasets",
    "get-dataset",
    "get-dataset-examples",  # NEW — inspect actual vignette rows
    "get-dataset-experiments",  # NEW — cross-reference dataset → experiment history
    "add-dataset-examples",  # write: extend a dataset with new rows
    # ── Experiments ───────────────────────────────────────────────────────
    "list-experiments-for-dataset",  # was "list-experiments" — correct v4.x name
    "get-experiment-by-id",  # was "get-experiment" — correct v4.x name
]


def build_phoenix_mcp_toolset():
    """
    Return a configured McpToolset for @arizeai/phoenix-mcp, or None if
    PHOENIX_API_KEY is not set or Node/npx is unavailable.
    """
    api_key = os.environ.get("PHOENIX_API_KEY", "").strip()
    if not api_key:
        logger.info(
            "PHOENIX_API_KEY not set — Phoenix MCP toolset disabled. "
            "Phoenix function tools (annotate_spans_with_verdicts, "
            "push_prompt_version, etc.) remain available."
        )
        return None

    # Derive base URL — strip /v1/traces if collector endpoint was supplied
    raw = os.environ.get(
        "PHOENIX_BASE_URL",
        os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com"),
    )
    base_url = raw.split("/v1/")[0].rstrip("/")

    try:
        from google.adk.tools.mcp_tool import McpToolset
        from google.adk.tools.mcp_tool.mcp_session_manager import (
            StdioConnectionParams,
        )
        from mcp import StdioServerParameters
    except ImportError as e:
        logger.warning("google-adk not installed — Phoenix MCP toolset disabled: %s", e)
        return None

    logger.info(
        "Phoenix MCP toolset → %s (%d tools)",
        base_url,
        len(PHOENIX_MCP_TOOL_FILTER),
    )

    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@arizeai/phoenix-mcp@latest",
                    "--baseUrl",
                    base_url,
                    "--apiKey",
                    api_key,
                ],
            )
        ),
        tool_filter=PHOENIX_MCP_TOOL_FILTER,
    )
