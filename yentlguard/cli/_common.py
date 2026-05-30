import logging
import os

logger = logging.getLogger("yentlguard.cli")

_DEFAULT_PHOENIX_MCP_ENDPOINT = "https://app.phoenix.arize.com"


def _get_completed_vignettes(model: str, budget: str, variant: str) -> set[str]:
    # Retained for baseline.py / resumption use. cmd_run no longer calls this
    # because run_experiment iterates the full dataset.
    from google.cloud import bigquery
    from yentlguard.config import GCP_PROJECT_ID, RUNS_TABLE

    client = bigquery.Client(project=GCP_PROJECT_ID)
    query = f"""
        SELECT DISTINCT vignette_id
        FROM `{RUNS_TABLE}`
        WHERE model_version = @model
          AND thinking_budget = @budget
          AND demographic_variant = @variant
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("model", "STRING", model),
            bigquery.ScalarQueryParameter("budget", "STRING", budget),
            bigquery.ScalarQueryParameter("variant", "STRING", variant),
        ]
    )
    try:
        df = client.query(query, job_config=job_config).to_dataframe()
        return set(df["vignette_id"].astype(str).tolist())
    except Exception as e:
        logger.warning("Failed to check completed vignettes: %s", e)
        return set()


def _build_phoenix_components():
    """
    Instantiate PhoenixPromptManager, PhoenixDatasetManager, and
    PhoenixExperimentRegistry. Returns (prompt_mgr, dataset_mgr, expt_registry).

    NOTE: cmd_run (Option A) no longer uses expt_registry, but cmd_baseline
    still calls expt_registry.register(...). So this keeps returning the
    3-tuple; run.py simply ignores the third element. All three degrade
    gracefully when Phoenix is unreachable.
    """
    from yentlguard.mcp.phoenix_manager import (
        PhoenixDatasetManager,
        PhoenixExperimentRegistry,
        PhoenixPromptManager,
    )

    base_url = os.environ.get("PHOENIX_BASE_URL", "http://localhost:6006")
    api_key = os.environ.get("PHOENIX_API_KEY", "")

    prompt_mgr = PhoenixPromptManager(base_url=base_url, api_key=api_key)
    dataset_mgr = PhoenixDatasetManager(base_url=base_url, api_key=api_key)
    expt_registry = PhoenixExperimentRegistry(base_url=base_url, api_key=api_key)

    return prompt_mgr, dataset_mgr, expt_registry

def _extract_experiment_id(experiment) -> str | None:
    """Pull the Phoenix experiment id from a RanExperiment (object or dict)."""
    exp_id = getattr(experiment, "id", None) or getattr(experiment, "experiment_id", None)
    if exp_id is None and isinstance(experiment, dict):
        exp_id = experiment.get("id") or experiment.get("experiment_id")
    if exp_id is None:
        # Last resort: lift it off the first run.
        runs = getattr(experiment, "runs", None)
        if runs is None and isinstance(experiment, dict):
            runs = experiment.get("runs")
        if runs:
            first = runs[0]
            exp_id = getattr(first, "experiment_id", None) or (
                first.get("experiment_id") if isinstance(first, dict) else None
            )
    return exp_id