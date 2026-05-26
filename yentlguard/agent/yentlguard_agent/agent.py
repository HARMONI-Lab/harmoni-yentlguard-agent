"""
YentlGuard root ADK agent.

Exposes the full agentic loop: experiment planning, run execution,
BigQuery metric analysis, Phoenix span exploration, and hypothesis
interpretation across H1–H5.

Run via:
    yentlguard agent                    # launches adk web (browser UI)
    yentlguard agent --query "..."      # single-turn, prints and exits
    adk web yentlguard/agent/yentlguard_agent  # direct ADK entrypoint
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from functools import cached_property

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

from yentlguard.telemetry.phoenix import setup_phoenix_tracing
from yentlguard.config import GCP_PROJECT_ID, GCP_LOCATION

# batch=False: flush spans per turn for interactive adk web sessions.
# CLI runs (yentlguard run, yentlguard baseline) keep batch=True via their
# own setup_phoenix_tracing() calls.
setup_phoenix_tracing(batch=False)

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.models import Gemini
from google.genai import Client

from yentlguard.agent.yentlguard_agent.mcp_config import build_phoenix_mcp_toolset
from yentlguard.agent.yentlguard_agent.prompt import SYSTEM_INSTRUCTION
from yentlguard.agent.yentlguard_agent.tools.bq_tools import (
    get_gate_fire_rate,
    get_pss_summary,
    get_sycophancy_verdict,
    list_experiments,
    query_bigquery,
)
from yentlguard.agent.yentlguard_agent.tools.runner_tools import (
    analyze_run,
    run_baseline,
    run_experiment,
)

logger = logging.getLogger(__name__)

_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

class VertexGemini(Gemini):
    @cached_property
    def api_client(self) -> Client:
        return Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION,
        )

_tools = [
    FunctionTool(func=query_bigquery),
    FunctionTool(func=list_experiments),
    FunctionTool(func=get_pss_summary),
    FunctionTool(func=get_sycophancy_verdict),
    FunctionTool(func=get_gate_fire_rate),
    FunctionTool(func=run_baseline),
    FunctionTool(func=run_experiment),
    FunctionTool(func=analyze_run),
]

_phoenix_mcp = build_phoenix_mcp_toolset()
if _phoenix_mcp is not None:
    _tools.append(_phoenix_mcp)
    logger.info("Phoenix MCP toolset attached.")
else:
    logger.info(
        "Phoenix MCP toolset not available — "
        "set PHOENIX_API_KEY to enable span exploration tools."
    )

root_agent = Agent(
    model=VertexGemini(model=_model),
    name="yentlguard_agent",
    instruction=SYSTEM_INSTRUCTION,
    tools=_tools,
)
