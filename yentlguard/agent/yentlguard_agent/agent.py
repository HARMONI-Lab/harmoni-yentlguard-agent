"""
YentlGuard root ADK agent.

Tool inventory:
    BigQuery tools     — metric queries, PSS, sycophancy verdicts, gate rates
    Runner tools       — run_baseline, run_experiment, analyze_run
    Phoenix tools      — span annotation, prompt versioning, anomaly datasets
    Phoenix MCP tools  — trace/span exploration, list-experiments, list-prompts
                         (@arizeai/phoenix-mcp via npx stdio process)
"""

from __future__ import annotations

import logging
import os
from functools import cached_property
from pathlib import Path

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
from yentlguard.agent.yentlguard_agent.prompt import SYSTEM_INSTRUCTION
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
    # BigQuery — metric aggregation
    FunctionTool(func=query_bigquery),
    FunctionTool(func=list_experiments),
    FunctionTool(func=get_pss_summary),
    FunctionTool(func=get_sycophancy_verdict),
    FunctionTool(func=get_gate_fire_rate),
    # Runner — experiment execution
    FunctionTool(func=run_baseline),
    FunctionTool(func=run_experiment),
    FunctionTool(func=analyze_run),
    # Phoenix — span annotation, prompt management, anomaly datasets
    FunctionTool(func=annotate_spans_with_verdicts),
    FunctionTool(func=push_prompt_version),
    FunctionTool(func=list_prompt_versions),
    FunctionTool(func=create_anomaly_dataset),
]

# Phoenix MCP toolset — npx @arizeai/phoenix-mcp stdio process
# Provides: list-traces, get-trace, list-spans, get-span,
#           list-projects, list-datasets, get-dataset,
#           list-experiments, get-experiment,
#           list-prompts, get-prompt, upsert-prompt,
#           add-dataset-examples, annotate-span
_phoenix_mcp = build_phoenix_mcp_toolset()
if _phoenix_mcp is not None:
    _tools.append(_phoenix_mcp)
    logger.info("Phoenix MCP toolset attached.")
else:
    logger.info("Phoenix MCP toolset unavailable — set PHOENIX_API_KEY to enable.")

root_agent = Agent(
    model=VertexGemini(model=_model),
    name="yentlguard_agent",
    instruction=SYSTEM_INSTRUCTION,
    tools=_tools,
)
