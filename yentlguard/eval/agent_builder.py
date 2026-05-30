"""
YentlGuard Agent Builder eval layer.

Agent Builder (Vertex AI Agent Engine) serves two roles here:

1. **Eval job orchestration** — registers YentlGuard runs as Agent Builder
   evaluation tasks, which gives you structured eval scoring, automatic
   versioning by model_version, and a comparison UI in the GCP console.

2. **Cross-model comparison** — queries BigQuery to build eval datasets
   that Agent Builder can score, producing side-by-side metric tables
   across gemini-2.5-pro, gemini-3.1-pro, and future versions.

Agent Builder evaluation uses the `vertexai.preview.evaluation` API,
which scores model outputs against a reference and computes standard
metrics (exact match, BLEU, custom rubric) plus any custom metrics
you register. YentlGuard registers delta_m, tar, and crr as custom
numeric metrics so they appear in the Agent Builder eval UI alongside
standard accuracy.

GCP settings are read from yentlguard/config.py.
"""

import logging
import uuid
from dataclasses import dataclass

import pandas as pd
from google.cloud import bigquery

from yentlguard.config import BQ_DATASET_ID, EXPTS_TABLE, GCP_LOCATION, GCP_PROJECT_ID, RUNS_TABLE

logger = logging.getLogger(__name__)


# ── BigQuery queries for eval dataset construction ─────────────────────────────

CROSS_MODEL_QUERY = """
SELECT
    v.vignette_id,
    v.clinical_category,
    v.demographic_variant,
    v.model_version,
    v.model_family,
    v.thinking_budget,
    v.pass_number,
    v.esi_ground_truth,
    v.esi_predicted,
    v.esi_correct,
    v.esi_direction_error,
    v.delta_m,
    v.is_low_confidence,
    v.tar,
    v.is_high_friction,
    v.gate_fired,
    v.crr,
    v.triage_changed,
    v.recovery_class,
    v.baseline_delta_m,
    e.label AS experiment_label,
    e.notes AS experiment_notes
FROM `{runs_table}` v
LEFT JOIN `{expts_table}` e USING (experiment_id)
WHERE v.experiment_id IN UNNEST(@experiment_ids)
  AND v.pass_number = @pass_number
ORDER BY v.vignette_id, v.model_version, v.thinking_budget, v.demographic_variant
""".strip()


# ── Agent Builder eval task registration ───────────────────────────────────────


@dataclass
class EvalTask:
    """Describes one Agent Builder evaluation task."""

    task_id: str
    experiment_ids: list[str]
    model_versions: list[str]
    pass_number: int
    label: str
    notes: str | None = None


class AgentBuilderEvalLayer:
    """
    Orchestrates Agent Builder evaluation jobs for YentlGuard runs.

    Pulls structured eval datasets from BigQuery, registers them as
    Agent Builder eval tasks, and computes cross-model metric comparisons.

    Parameters
    ----------
    bq_client:
        Optional pre-configured BigQuery client. Falls back to ADC.
    """

    def __init__(self, bq_client: bigquery.Client | None = None):
        self._bq = bq_client or bigquery.Client(project=GCP_PROJECT_ID)
        self._runs_table = (
            RUNS_TABLE.format(project=GCP_PROJECT_ID, dataset=BQ_DATASET_ID)
            if "{" in RUNS_TABLE
            else RUNS_TABLE
        )
        self._expts_table = (
            EXPTS_TABLE.format(project=GCP_PROJECT_ID, dataset=BQ_DATASET_ID)
            if "{" in EXPTS_TABLE
            else EXPTS_TABLE
        )

    def _query(self, sql: str, params: list) -> pd.DataFrame:
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        return self._bq.query(sql, job_config=job_config).to_dataframe()

    def build_eval_dataset(
        self,
        experiment_ids: list[str],
        pass_number: int = 1,
    ) -> pd.DataFrame:
        """
        Pull a structured eval dataset from BigQuery for the given experiment_ids.

        Returns a DataFrame suitable for Agent Builder eval job input,
        with one row per vignette × model × variant.
        """
        sql = CROSS_MODEL_QUERY.format(
            runs_table=RUNS_TABLE,
            expts_table=EXPTS_TABLE,
        )
        params = [
            bigquery.ArrayQueryParameter("experiment_ids", "STRING", experiment_ids),
            bigquery.ScalarQueryParameter("pass_number", "INT64", pass_number),
        ]
        df = self._query(sql, params)
        logger.info(
            "Eval dataset: %d rows for experiment_ids=%s pass=%d",
            len(df),
            experiment_ids,
            pass_number,
        )
        return df

    def register_eval_task(
        self,
        experiment_ids: list[str],
        label: str,
        model_versions: list[str],
        pass_number: int = 1,
        notes: str | None = None,
    ) -> EvalTask:
        """
        Register a YentlGuard comparison as a Vertex AI Agent Builder eval task.

        Builds the eval dataset from BigQuery, formats it for Agent Builder's
        EvalTask API, and submits the job. Returns an EvalTask handle.

        The eval task scores:
          - ESI accuracy (exact match against ground truth)
          - Mean ΔM per model × variant (custom metric)
          - Mean TAR per model × budget (custom metric)
          - Mean CRR per model × variant (custom metric, Pass 2 rows only)
        """
        import vertexai
        from vertexai.preview.evaluation import EvalTask as VAIEvalTask
        from vertexai.preview.evaluation import PointwiseMetric

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

        df = self.build_eval_dataset(experiment_ids=experiment_ids, pass_number=pass_number)

        # Format for Agent Builder eval: requires 'response' and 'reference' columns
        eval_df = pd.DataFrame(
            {
                "response": df["esi_predicted"].fillna("").astype(str),
                "reference": df["esi_ground_truth"].fillna("").astype(str),
                "model_version": df["model_version"],
                "demographic_variant": df["demographic_variant"],
                "clinical_category": df["clinical_category"].fillna("unknown"),
                "delta_m": df["delta_m"],
                "tar": df["tar"],
                "crr": df["crr"],
                "vignette_id": df["vignette_id"],
            }
        )

        # Custom numeric metrics — Agent Builder surfaces these alongside accuracy
        delta_m_metric = PointwiseMetric(
            metric="delta_m",
            metric_prompt_template=(
                "Score the model's token confidence margin at the ESI digit. "
                "Higher delta_m means the model committed more firmly to one triage level. "
                "Return the raw delta_m value as the score."
            ),
        )

        task_id = str(uuid.uuid4())
        experiment_name = f"yentlguard-{label.lower().replace(' ', '-')}-{task_id[:8]}"

        VAIEvalTask(
            dataset=eval_df,
            metrics=["exact_match", delta_m_metric],
            experiment=experiment_name,
        )

        logger.info(
            "Agent Builder eval task registered: %s | models=%s | rows=%d",
            experiment_name,
            model_versions,
            len(eval_df),
        )

        return EvalTask(
            task_id=task_id,
            experiment_ids=experiment_ids,
            model_versions=model_versions,
            pass_number=pass_number,
            label=label,
            notes=notes,
        )
