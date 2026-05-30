"""
YentlGuard root ADK agent.

Multi-agent architecture:
    Root Supervisor  — Plans and delegates to sub-agents via ADK transfer
    Data Analyst     — BigQuery metrics, PSS, sycophancy verdicts
    Observability    — Phoenix MCP, trace/span exploration, prompt versions
    Runner           — Baseline runs, experiments, and reporting
"""

from __future__ import annotations

import logging
import os
import sys
from functools import cached_property
from pathlib import Path

# Force basic config so ADK agent logs stream directly to the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

from yentlguard.config import GCP_LOCATION, GCP_PROJECT_ID
from yentlguard.telemetry.phoenix import setup_phoenix_tracing

setup_phoenix_tracing(project_name="yentlguard-agent", batch=False)

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from google.genai import Client

from yentlguard.agent.yentlguard_agent.mcp_config import build_phoenix_mcp_toolset
from yentlguard.agent.yentlguard_agent.prompt import (
    DATA_ANALYST_INSTRUCTION,
    EXPERIMENT_RUNNER_INSTRUCTION,
    OBSERVABILITY_INSTRUCTION,
    SUPERVISOR_INSTRUCTION,
)
from yentlguard.agent.yentlguard_agent.tools.bq_tools import (
    get_gate_fire_rate,
    get_pss_summary,
    get_sycophancy_verdict,
    list_experiments,
    query_bigquery,
)
from yentlguard.agent.yentlguard_agent.tools.phoenix_tools import (
    annotate_spans_with_verdicts,
    create_anomaly_dataset,
    list_prompt_versions,
    push_prompt_version,
)
from yentlguard.agent.yentlguard_agent.tools.runner_tools import (
    analyze_run,
    run_baseline,
    run_experiment,
    triage_vignette,
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


# --- DATA ANALYST AGENT ---
data_analyst_tools = [
    FunctionTool(func=query_bigquery),
    FunctionTool(func=list_experiments),
    FunctionTool(func=get_pss_summary),
    FunctionTool(func=get_sycophancy_verdict),
    FunctionTool(func=get_gate_fire_rate),
]

data_analyst_agent = Agent(
    model=VertexGemini(model=_model),
    name="data_analyst_agent",
    description="Handles BigQuery metric aggregation, statistical thresholds, PSS, CRR, and computes sycophancy verdicts.",
    instruction=DATA_ANALYST_INSTRUCTION,
    tools=data_analyst_tools,
)

# --- OBSERVABILITY & PROMPT ENGINEER AGENT ---
observability_tools = [
    FunctionTool(func=annotate_spans_with_verdicts),
    FunctionTool(func=push_prompt_version),
    FunctionTool(func=list_prompt_versions),
    FunctionTool(func=create_anomaly_dataset),
]

_phoenix_mcp = build_phoenix_mcp_toolset()
if _phoenix_mcp is not None:
    observability_tools.append(_phoenix_mcp)
    logger.info("Phoenix MCP toolset attached to observability_agent.")
else:
    logger.info("Phoenix MCP toolset unavailable — set PHOENIX_API_KEY to enable.")

observability_agent = Agent(
    model=VertexGemini(model=_model),
    name="observability_agent",
    description="Handles Arize Phoenix integration. Manages prompt versioning, trace/span exploration, anomaly datasets, and writing span annotations.",
    instruction=OBSERVABILITY_INSTRUCTION,
    tools=observability_tools,
)

# --- EXPERIMENT RUNNER AGENT ---
runner_tools = [
    FunctionTool(func=triage_vignette),
    FunctionTool(func=run_baseline),
    FunctionTool(func=run_experiment),
    FunctionTool(func=analyze_run),
]

experiment_runner_agent = Agent(
    model=VertexGemini(model=_model),
    name="experiment_runner_agent",
    description="Safely orchestrates long-running Gemini triage evaluations, calculates costs, and generates HTML/CSV reports.",
    instruction=EXPERIMENT_RUNNER_INSTRUCTION,
    tools=runner_tools,
)

# --- ROOT SUPERVISOR AGENT ---
root_agent = Agent(
    model=VertexGemini(model=_model),
    name="yentlguard_agent",
    instruction=SUPERVISOR_INSTRUCTION,
    sub_agents=[
        data_analyst_agent,
        observability_agent,
        experiment_runner_agent,
    ],
)
