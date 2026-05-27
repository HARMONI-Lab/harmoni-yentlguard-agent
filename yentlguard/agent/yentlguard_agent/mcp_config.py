"""
Phoenix MCP toolset configuration for the YentlGuard ADK agent.

@arizeai/phoenix-mcp runs as a Node child process launched via npx.
ADK manages the subprocess lifecycle through StdioConnectionParams.

Requires: Node.js + npx in PATH.
Returns None if PHOENIX_API_KEY is not set or Node is unavailable —
the agent runs with BigQuery and Phoenix function tools only.

Tool surface exposed (full read + targeted write):
    Read:  list-projects, get-project
           list-traces, get-trace
           list-spans, get-span
           list-datasets, get-dataset
           list-experiments, get-experiment
           list-prompts, get-prompt
    Write: upsert-prompt          (prompt versioning)
           add-dataset-examples   (extend anomaly datasets)
           annotate-span          (low-level span annotation)

Base URL derivation:
    Strips /v1/traces suffix if PHOENIX_COLLECTOR_ENDPOINT was set instead
    of PHOENIX_BASE_URL. Both variables are supported.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


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
        os.environ.get(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com"
        ),
    )
    base_url = raw.split("/v1/")[0].rstrip("/")

    try:
        from google.adk.tools.mcp_tool import McpToolset
        from google.adk.tools.mcp_tool.mcp_session_manager import (
            StdioConnectionParams,
        )
    except ImportError as e:
        logger.warning(
            "google-adk not installed — Phoenix MCP toolset disabled: %s", e
        )
        return None

    logger.info("Phoenix MCP toolset → %s", base_url)

    return McpToolset(
        connection_params=StdioConnectionParams(
            command="npx",
            args=[
                "-y",
                "@arizeai/phoenix-mcp@latest",
                "--baseUrl", base_url,
                "--apiKey", api_key,
            ],
        ),
        tool_filter=[
            # ── Read ────────────────────────────────────────────────────
            "list-projects",
            "get-project",
            "list-traces",
            "get-trace",
            "list-spans",
            "get-span",
            "list-datasets",
            "get-dataset",
            "list-experiments",
            "get-experiment",
            "list-prompts",
            "get-prompt",
            # ── Write ───────────────────────────────────────────────────
            # Prompt versioning — used by push_prompt_version tool and
            # directly by the agent when the user asks to update a prompt
            "upsert-prompt",
            # Dataset extension — used by create_anomaly_dataset and
            # directly when the user wants to add examples to an existing set
            "add-dataset-examples",
            # Low-level span annotation — prefer annotate_spans_with_verdicts
            # function tool which handles BQ→Phoenix pairing automatically;
            # this is available for direct single-span annotation
            "annotate-span",
        ],
    )
