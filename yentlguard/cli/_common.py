import logging
import os

logger = logging.getLogger("yentlguard.cli")

_DEFAULT_PHOENIX_MCP_ENDPOINT = "https://app.phoenix.arize.com"


def _build_phoenix_components():
    """
    Instantiate PhoenixPromptManager and PhoenixDatasetManager. Returns (prompt_mgr, dataset_mgr).
    """
    from yentlguard.mcp.phoenix_manager import (
        PhoenixDatasetManager,
        PhoenixPromptManager,
    )

    base_url = os.environ.get("PHOENIX_BASE_URL", os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
    api_key = os.environ.get("PHOENIX_API_KEY", "")

    prompt_mgr = PhoenixPromptManager(base_url=base_url, api_key=api_key)
    dataset_mgr = PhoenixDatasetManager(base_url=base_url, api_key=api_key)
    return prompt_mgr, dataset_mgr


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
