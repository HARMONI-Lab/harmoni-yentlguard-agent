"""
Experiment execution tools for the YentlGuard ADK agent.

These wrap the existing CLI commands (cmd_baseline, cmd_run, cmd_analyze)
so the agent can trigger and monitor runs without shelling out. They return
structured JSON summaries rather than printing to stdout.

IMPORTANT: run_baseline and run_experiment are long-running operations.
    - run_baseline: ~3–5 min for 70 vignettes (1 Vertex AI call per vignette)
    - run_experiment: varies; each gate-fired vignette spawns 4 parallel
      Vertex AI calls (corrective + 3 distractors)

The agent should state the estimated scope and confirm with the user before
calling run_experiment. run_baseline is lower risk but still incurs GCP cost.

Both return an experiment_id on completion that can be passed directly to BigQuery
analysis tools (get_pss_summary, get_sycophancy_verdict, etc.) and to Phoenix
MCP tools (list-experiments-for-dataset, get-experiment-by-id) via the
phoenix_dataset_id stored in the BQ experiments table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_baseline(
    model: str = "gemini-2.5-pro",
    budget: str = "medium",
    dataset_path: str = "dataset_output/dataset_quintets.csv",
) -> str:
    """
    Run the nb_ambiguous baseline pass for a model + thinking budget tier.

    Populates the BigQuery runs table with pass_number=1 rows for the neutral
    demographic condition. These baseline ΔM values are the recovery target
    for CRR computation in all subsequent corrective runs.

    Must be completed before calling run_experiment on the same model+budget
    combination if CRR computation is needed.

    Args:
        model: Gemini model string (e.g., "gemini-2.5-pro", "gemini-2.5-flash").
        budget: Thinking budget tier — "low", "medium", or "high".
        dataset_path: Path to dataset_quintets.csv produced by yentlbench prepare.
                      Default assumes CWD is the project root.

    Returns:
        JSON object with status, model, budget, and experiment_id on success.
        Error string prefixed with "Error:" on failure.
    """
    from yentlguard.config import validate

    try:
        validate()
    except RuntimeError as e:
        return f"Error: GCP config incomplete — {e}"

    if not Path(dataset_path).exists():
        return (
            f"Error: dataset not found at {dataset_path}. "
            "Run: yentlbench prepare  (requires MIMIC-IV-ED data)"
        )

    import argparse
    from yentlguard.cli import cmd_baseline

    args = argparse.Namespace(model=model, budget=budget, dataset=dataset_path, skip_shutdown=True)
    try:
        experiment_id = cmd_baseline(args)
        return json.dumps(
            {"status": "complete", "model": model, "budget": budget, "experiment_id": experiment_id}
        )
    except Exception as e:
        logger.error("run_baseline failed: %s", e)
        return f"Error: baseline run failed — {e}"


def run_experiment(
    model: str,
    variants: list[str],
    budgets: list[str],
    label: str,
    dataset_path: str = "dataset_output/dataset_quintets.csv",
    threshold: float = 1.0,
    notes: str | None = None,
) -> str:
    """
    Execute a two-pass mechanistic run for specified demographic variants and
    thinking budget tiers. Writes all results to BigQuery and returns the
    experiment_id for subsequent analysis.

    Each vignette where the correction gate fires (ΔM < threshold AND
    demographic token present) spawns four parallel Vertex AI calls:
    corrective + distractors 3a/3b/3c. Confirm the scope with the user
    before calling — this incurs real GCP cost.

    The returned experiment_id can be used with:
        - BigQuery tools: get_pss_summary, get_sycophancy_verdict, get_gate_fire_rate
        - Phoenix MCP: query_bigquery to retrieve phoenix_dataset_id from the
          experiments table, then list-experiments-for-dataset and
          get-experiment-by-id for the Phoenix-native experiment view.

    Valid variants: "male", "female", "nb_ambiguous", "nb_label_only"
    Valid budgets:  "low", "medium", "high"

    Args:
        model: Gemini model string.
        variants: Demographic variants to run, e.g. ["female", "nb_label_only"].
        budgets: Thinking budget tiers, e.g. ["low", "medium", "high"].
        label: Human-readable experiment label stored in the experiments table.
        dataset_path: Path to dataset_quintets.csv.
        threshold: ΔM threshold for the correction gate (default 1.0 nat).
                   Lower values fire the gate less aggressively.
        notes: Optional free-text notes stored with the experiment batch.

    Returns:
        JSON object with status and experiment_id on success.
        Error string prefixed with "Error:" on failure.
    """
    from yentlguard.config import validate

    try:
        validate()
    except RuntimeError as e:
        return f"Error: GCP config incomplete — {e}"

    if not Path(dataset_path).exists():
        return (
            f"Error: dataset not found at {dataset_path}. "
            "Run: yentlbench prepare  (requires MIMIC-IV-ED data)"
        )

    valid_variants = {"male", "female", "nb_ambiguous", "nb_label_only"}
    invalid = [v for v in variants if v not in valid_variants]
    if invalid:
        return f"Error: invalid variants {invalid}. Valid: {sorted(valid_variants)}"

    valid_budgets = {"low", "medium", "high"}
    invalid_b = [b for b in budgets if b not in valid_budgets]
    if invalid_b:
        return f"Error: invalid budgets {invalid_b}. Valid: {sorted(valid_budgets)}"

    import argparse
    from yentlguard.cli import cmd_run
    import os

    args = argparse.Namespace(
        model=model,
        variants=variants,
        budget=budgets,
        label=label,
        dataset=dataset_path,
        threshold=threshold,
        notes=notes,
        phoenix_mcp_endpoint=os.environ.get(
            "PHOENIX_MCP_ENDPOINT", "https://app.phoenix.arize.com"
        ),
        skip_shutdown=True,
    )
    try:
        experiment_id = cmd_run(args)
        return json.dumps({
            "status": "complete",
            "experiment_id": experiment_id,
            "model": model,
            "variants": variants,
            "budgets": budgets,
            "next_steps": (
                "To get the Phoenix experiment view: call query_bigquery to retrieve "
                "phoenix_dataset_id from the experiments table for this experiment_id, then "
                "call list-experiments-for-dataset with that dataset_id."
            ),
        })
    except Exception as e:
        logger.error("run_experiment failed: %s", e)
        return f"Error: experiment run failed — {e}"

def analyze_run(
    experiment_ids: list[str],
    output_dir: str = "yentlguard_analysis",
) -> str:
    """
    Generate the full HTML analysis report and raw CSV outputs for one or more
    completed experiment batches.

    This wraps the 'yentlguard analyze' CLI command. It queries BigQuery for
    all metrics (PSS, H1, H3, H5), applies statistical tests, and writes
    the results to disk for inspection.

    Args:
        experiment_ids: One or more experiment batch IDs to include.
        output_dir: Directory to write the HTML report and CSVs to.

    Returns:
        JSON object with output_dir, experiment_ids, and status on success.
    
    Pull completed run data from BigQuery, compute H1-H5 summary statistics,
    and write a self-contained HTML report plus CSV files to the output directory.

    Use this when the user asks for a full report on one or more completed runs.
    For targeted metric queries (PSS, sycophancy verdicts, gate rates), use the
    BigQuery tools directly — they return results faster without writing files.

    Args:
        experiment_ids: One or more experiment batch IDs to include.
        output_dir: Directory to write the HTML report and CSVs to.

    Returns:
        JSON object with output_dir, experiment_ids, and status on success.
    """
    from yentlguard.config import validate

    try:
        validate()
    except RuntimeError as e:
        return f"Error: GCP config incomplete — {e}"

    import argparse
    from yentlguard.cli import cmd_analyze

    args = argparse.Namespace(
        experiment_ids=experiment_ids,
        output_dir=output_dir,
    )
    try:
        cmd_analyze(args)
        return json.dumps(
            {
                "status": "complete",
                "output_dir": output_dir,
                "experiment_ids": experiment_ids,
                "next_steps": f"Analysis written to {output_dir}/. You can read the HTML report or CSVs to interpret the results.",
            }
        )
    except Exception as e:
        logger.error("analyze_run failed: %s", e)
        return f"Error: analysis failed — {e}"
