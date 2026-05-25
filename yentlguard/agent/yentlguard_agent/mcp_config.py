"""
Phoenix MCP toolset configuration for the YentlGuard ADK agent.

@arizeai/phoenix-mcp runs as a Node child process launched via npx.
ADK manages the full subprocess lifecycle through StdioConnectionParams —
the process starts when the toolset is initialized and is terminated when
the agent session ends.

Requires: Node.js + npx available in PATH.
If absent, build_phoenix_mcp_toolset() returns None and the agent runs
with BigQuery tools only (still fully functional for metric queries).

What Phoenix MCP tools are used for (v4.x):
    list-projects, get-project     — project inventory
    list-traces, get-trace         — trace exploration by time window
    list-spans, get-span           — span drill-down on specific vignettes

What Phoenix MCP tools are NOT used for (use BigQuery instead):
    Span filtering by custom attribute (yentlguard.vignette_id etc.)
    Metric aggregation (PSS, TAR distributions, CRR means)
    Cross-run comparisons

The tool_filter list intentionally excludes write operations
(upsert-prompt, add-prompt-version-tag). Add them explicitly when
prompt versioning via the agent is needed.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def build_phoenix_mcp_toolset():
    """
    Return a configured McpToolset for @arizeai/phoenix-mcp, or None if
    PHOENIX_API_KEY is not set or Node is unavailable.
    """
    api_key = os.environ.get("PHOENIX_API_KEY", "").strip()
    if not api_key:
        logger.info(
            "PHOENIX_API_KEY not set — Phoenix MCP toolset disabled. "
            "BigQuery tools remain available for all metric queries."
        )
        return None

    # Derive base URL from whichever env var is set.
    # Strip /v1/traces suffix if the collector endpoint was provided instead.
    raw = os.environ.get(
        "PHOENIX_BASE_URL",
        os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com"),
    )
    base_url = raw.split("/v1/")[0].rstrip("/")

    try:
        from google.adk.tools.mcp_tool import McpToolset
        from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    except ImportError as e:
        logger.warning("google-adk not installed — Phoenix MCP toolset disabled: %s", e)
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
        # Read-only tools only. Extend this list to add write operations.
        tool_filter=[
            "list-projects",
            "get-project",
            "list-traces",
            "get-trace",
            "list-spans",
            "get-span",
            "list-datasets",
            "list-experiments",
            "get-experiment",
        ],
    )
