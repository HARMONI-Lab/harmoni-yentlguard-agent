"""
YentlGuard Baseline Lookup.

Queries for nb_ambiguous baseline ΔM values.
The correction gate in YentlGuardRunner calls this client to retrieve
the historical ΔM for a given vignette under the nb_ambiguous condition,
which serves as the recovery target for CRR computation.
"""

import logging
from abc import ABC, abstractmethod
import pandas as pd
from google.cloud import bigquery
from yentlguard.config import RUNS_TABLE, GCP_PROJECT_ID

# MCP Client imports for future/current usage
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)

class BaselineLookup(ABC):
    """Abstract base class for retrieving nb_ambiguous baseline ΔM values."""

    @abstractmethod
    def get_baseline_delta_m(
        self,
        vignette_id: str,
        variant: str = "nb_ambiguous",
        model_version: str | None = None,
    ) -> float:
        """Retrieve the mean ΔM."""
        pass

    @abstractmethod
    def get_span_history(
        self,
        vignette_id: str,
        variant: str | None = None,
        model_version: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return raw span records for a vignette for broader analysis."""
        pass


class BQBackend(BaselineLookup):
    """
    Client for retrieving nb_ambiguous baseline ΔM values from BigQuery.
    
    NOTE: The official Arize Phoenix MCP server (@arizeai/phoenix-mcp) does not 
    support filtering spans by custom attributes (like yentlguard.vignette_id).
    Fetching all spans and filtering locally is O(N) and prohibitively slow.
    
    Therefore, this backend bypasses Phoenix MCP entirely and queries the BigQuery
    RUNS_TABLE directly, which contains the exact same baseline data and executes
    in milliseconds. The interface remains the same to satisfy YentlGuardRunner.
    """

    def __init__(self, project_name: str = "yentlguard"):
        self.project_name = project_name
        self.client = bigquery.Client(project=GCP_PROJECT_ID)

    def get_baseline_delta_m(
        self,
        vignette_id: str,
        variant: str = "nb_ambiguous",
        model_version: str | None = None,
    ) -> float:
        """Retrieve the mean ΔM from BigQuery."""
        query = f"""
            SELECT AVG(delta_m) as avg_dm
            FROM `{RUNS_TABLE}`
            WHERE vignette_id = @vignette_id
              AND demographic_variant = @variant
              AND pass_number = 1
        """
        
        query_params = [
            bigquery.ScalarQueryParameter("vignette_id", "STRING", str(vignette_id)),
            bigquery.ScalarQueryParameter("variant", "STRING", str(variant))
        ]
        
        if model_version:
            query += " AND model_version = @model_version"
            query_params.append(bigquery.ScalarQueryParameter("model_version", "STRING", str(model_version)))

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        
        try:
            df = self.client.query(query, job_config=job_config).to_dataframe()
        except Exception as e:
            raise RuntimeError(f"BigQuery baseline lookup failed for {vignette_id}/{variant}: {e}") from e

        if df.empty or pd.isna(df["avg_dm"].iloc[0]):
            raise ValueError(
                f"No baseline found in BigQuery for vignette_id={vignette_id}, "
                f"variant={variant}. Run the baseline command first."
            )

        mean_delta_m = float(df["avg_dm"].iloc[0])
        logger.debug(
            "BQ baseline: vignette=%s variant=%s delta_m=%.4f",
            vignette_id, variant, mean_delta_m
        )
        return mean_delta_m

    def get_span_history(
        self,
        vignette_id: str,
        variant: str | None = None,
        model_version: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        raise NotImplementedError("Span history lookup not supported via BQBackend.")


class MCPBackend(BaselineLookup):
    """
    Client for retrieving nb_ambiguous baseline ΔM values via Phoenix MCP.
    """

    def __init__(self, mcp_endpoint: str = "", project_name: str = "yentlguard", api_key: str | None = None):
        self.project_name = project_name
        self.mcp_endpoint = mcp_endpoint
        self.api_key = api_key

    async def _call_tool(self, name: str, arguments: dict) -> list[dict]:
        import json
        async with sse_client(self.mcp_endpoint) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return [
                    json.loads(c.text) if c.text.strip().startswith("{") else {"raw": c.text}
                    for c in result.content
                ]

    def _run(self, coro):
        import asyncio
        return asyncio.run(coro)

    def get_baseline_delta_m(
        self,
        vignette_id: str,
        variant: str = "nb_ambiguous",
        model_version: str | None = None,
    ) -> float:
        spans = self.get_span_history(vignette_id, variant, model_version)
        if not spans:
            raise ValueError(
                f"No Phoenix spans found for vignette_id={vignette_id}, "
                f"variant={variant}. Run the baseline command first."
            )

        delta_m_values = [
            s.get("attributes", {}).get("yentlguard.delta_m")
            for s in spans
            if isinstance(s, dict) and s.get("attributes", {}).get("yentlguard.delta_m") is not None
        ]

        if not delta_m_values:
            raise ValueError(
                f"Spans found for {vignette_id}/{variant} but none contain "
                f"yentlguard.delta_m attribute. Verify YentlGuard span annotation."
            )

        mean_delta_m = sum(delta_m_values) / len(delta_m_values)
        return mean_delta_m

    def get_span_history(
        self,
        vignette_id: str,
        variant: str | None = None,
        model_version: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Return raw span records for a vignette for broader analysis.

        Useful for TAR distribution comparison and PSS computation.
        """
        arguments: dict = {
            "project_name": self.project_name,
            "filters": {"attributes.yentlguard.vignette_id": vignette_id},
            "limit": limit,
        }
        if variant:
            arguments["filters"]["attributes.yentlguard.demographic_variant"] = variant
        if model_version:
            arguments["filters"]["attributes.yentlguard.model_version"] = model_version

        try:
            return self._run(self._call_tool("get_spans", arguments)) # type: ignore
        except RuntimeError:
            raise
        except Exception as e:
            # Unpack ExceptionGroup for Python 3.11+ to surface root causes (e.g. ConnectError)
            if hasattr(e, "exceptions"):
                root_error = e.exceptions[0] if e.exceptions else e
            else:
                root_error = e
            raise RuntimeError(
                f"MCPBackend span history query failed for {vignette_id}: {root_error}"
            ) from e
