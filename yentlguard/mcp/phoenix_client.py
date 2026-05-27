"""
YentlGuard Phoenix MCP client.

Queries Arize Phoenix via MCP for nb_ambiguous baseline ΔM values.
The correction gate in YentlGuardRunner calls this client to retrieve
the historical ΔM for a given vignette under the nb_ambiguous condition,
which serves as the recovery target for CRR computation.

Uses the correct mcp>=1.0.0 transport pattern:
    sse_client(url) → two anyio streams
    ClientSession(read_stream, write_stream) → session
"""

import logging
import pandas as pd
from google.cloud import bigquery
from yentlguard.config import RUNS_TABLE, GCP_PROJECT_ID

logger = logging.getLogger(__name__)

class PhoenixMCPClient:
    """
    Client for retrieving nb_ambiguous baseline ΔM values.
    
    NOTE: The official Arize Phoenix MCP server (@arizeai/phoenix-mcp) does not 
    support filtering spans by custom attributes (like yentlguard.vignette_id).
    Fetching all spans and filtering locally is O(N) and prohibitively slow.
    
    Therefore, this client bypasses Phoenix MCP entirely and queries the BigQuery
    RUNS_TABLE directly, which contains the exact same baseline data and executes
    in milliseconds. The interface remains the same to satisfy YentlGuardRunner.
    """

    def __init__(self, mcp_endpoint: str = "", project_name: str = "yentlguard", api_key: str | None = None):
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
            return self._run(self._call_tool("get_spans", arguments))
        except RuntimeError:
            raise
        except Exception as e:
            # Unpack ExceptionGroup for Python 3.11+ to surface root causes (e.g. ConnectError)
            if hasattr(e, "exceptions"):
                root_error = e.exceptions[0] if e.exceptions else e
            else:
                root_error = e
            raise RuntimeError(
                f"PhoenixMCPClient span history query failed for {vignette_id}: {root_error}"
            ) from e
