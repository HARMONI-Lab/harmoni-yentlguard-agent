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

from yentlguard.config import GCP_PROJECT_ID, RUNS_TABLE

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
            bigquery.ScalarQueryParameter("variant", "STRING", str(variant)),
        ]

        if model_version:
            query += " AND model_version = @model_version"
            query_params.append(
                bigquery.ScalarQueryParameter("model_version", "STRING", str(model_version))
            )

        job_config = bigquery.QueryJobConfig(query_parameters=query_params)

        try:
            df = self.client.query(query, job_config=job_config).to_dataframe()
        except Exception as e:
            raise RuntimeError(
                f"BigQuery baseline lookup failed for {vignette_id}/{variant}: {e}"
            ) from e

        if df.empty or pd.isna(df["avg_dm"].iloc[0]):
            raise ValueError(
                f"No baseline found in BigQuery for vignette_id={vignette_id}, "
                f"variant={variant}. Run the baseline command first."
            )

        mean_delta_m = float(df["avg_dm"].iloc[0])
        logger.debug(
            "BQ baseline: vignette=%s variant=%s delta_m=%.4f", vignette_id, variant, mean_delta_m
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
