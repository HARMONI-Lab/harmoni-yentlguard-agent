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
LEFT JOIN `{expts_table}` e USING (run_id)
WHERE v.run_id IN UNNEST(@run_ids)
  AND v.pass_number = @pass_number
ORDER BY v.vignette_id, v.model_version, v.thinking_budget, v.demographic_variant
""".strip()

PSS_QUERY = """
-- Perturbation Sensitivity Score per model × thinking_budget × clinical_category
-- PSS = mean absolute delta_m drop from nb_ambiguous baseline across male, female/nb variants
SELECT
    model_version,
    model_family,
    thinking_budget,
    clinical_category,
    demographic_variant,
    COUNT(*) AS n_vignettes,
    AVG(delta_m) AS mean_delta_m,
    AVG(baseline_delta_m) AS mean_baseline_delta_m,
    AVG(baseline_delta_m - delta_m) AS mean_delta_m_drop,
    STDDEV(baseline_delta_m - delta_m) AS stddev_delta_m_drop,
    AVG(tar) AS mean_tar,
    SUM(CAST(gate_fired AS INT64)) AS n_gate_fired,
    AVG(crr) AS mean_crr,
    SUM(CAST(triage_changed AS INT64)) AS n_triage_changed,
    COUNTIF(recovery_class = 'full') AS n_full_recovery,
    COUNTIF(recovery_class = 'partial') AS n_partial_recovery,
    COUNTIF(recovery_class = 'failed') AS n_failed_recovery
FROM `{runs_table}`
WHERE run_id IN UNNEST(@run_ids)
  AND pass_number = 1
  AND demographic_variant != 'nb_ambiguous'
GROUP BY 1, 2, 3, 4, 5
ORDER BY model_family, thinking_budget, clinical_category, demographic_variant
""".strip()

THINKING_BUDGET_QUERY = """
-- H1: Reasoning Mitigation Effect
-- Does higher thinking budget reduce PSS?
SELECT
    model_version,
    thinking_budget,
    demographic_variant,
    AVG(baseline_delta_m - delta_m) AS mean_pss,
    AVG(tar) AS mean_tar,
    COUNT(*) AS n
FROM `{runs_table}`
WHERE run_id IN UNNEST(@run_ids)
  AND pass_number = 1
   AND demographic_variant IN ('male', 'female', 'nb_label_only', 'nb_explicit')
GROUP BY 1, 2, 3
ORDER BY model_version, thinking_budget, demographic_variant
""".strip()


# ── Agent Builder eval task registration ───────────────────────────────────────

@dataclass
class EvalTask:
    """Describes one Agent Builder evaluation task."""
    task_id: str
    run_ids: list[str]
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
        self._runs_table  = RUNS_TABLE.format(
            project=GCP_PROJECT_ID, dataset=BQ_DATASET_ID
        ) if "{" in RUNS_TABLE else RUNS_TABLE
        self._expts_table = EXPTS_TABLE.format(
            project=GCP_PROJECT_ID, dataset=BQ_DATASET_ID
        ) if "{" in EXPTS_TABLE else EXPTS_TABLE

    def _query(self, sql: str, params: list) -> pd.DataFrame:
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        return self._bq.query(sql, job_config=job_config).to_dataframe()

    def build_eval_dataset(
        self,
        run_ids: list[str],
        pass_number: int = 1,
    ) -> pd.DataFrame:
        """
        Pull a structured eval dataset from BigQuery for the given run_ids.

        Returns a DataFrame suitable for Agent Builder eval job input,
        with one row per vignette × model × variant.
        """
        sql = CROSS_MODEL_QUERY.format(
            runs_table=RUNS_TABLE,
            expts_table=EXPTS_TABLE,
        )
        params = [
            bigquery.ArrayQueryParameter("run_ids", "STRING", run_ids),
            bigquery.ScalarQueryParameter("pass_number", "INT64", pass_number),
        ]
        df = self._query(sql, params)
        logger.info(
            "Eval dataset: %d rows for run_ids=%s pass=%d",
            len(df), run_ids, pass_number,
        )
        return df

    def compute_pss_summary(self, run_ids: list[str]) -> pd.DataFrame:
        """
        Compute Perturbation Sensitivity Score summary across model × budget × category.

        This is the primary cross-model comparison table for H1 and H3.
        """
        sql = PSS_QUERY.format(runs_table=RUNS_TABLE)
        params = [bigquery.ArrayQueryParameter("run_ids", "STRING", run_ids)]
        return self._query(sql, params)

    def compute_thinking_budget_effect(self, run_ids: list[str]) -> pd.DataFrame:
        """
        H1: Reasoning Mitigation Effect.
        Returns mean PSS by model × thinking_budget × variant.
        """
        sql = THINKING_BUDGET_QUERY.format(runs_table=RUNS_TABLE)
        params = [bigquery.ArrayQueryParameter("run_ids", "STRING", run_ids)]
        return self._query(sql, params)

    def register_eval_task(
        self,
        run_ids: list[str],
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

        df = self.build_eval_dataset(run_ids=run_ids, pass_number=pass_number)

        # Format for Agent Builder eval: requires 'response' and 'reference' columns
        eval_df = pd.DataFrame({
            "response":          df["esi_predicted"].fillna("").astype(str),
            "reference":         df["esi_ground_truth"].fillna("").astype(str),
            "model_version":     df["model_version"],
            "demographic_variant": df["demographic_variant"],
            "clinical_category": df["clinical_category"].fillna("unknown"),
            "delta_m":           df["delta_m"],
            "tar":               df["tar"],
            "crr":               df["crr"],
            "vignette_id":       df["vignette_id"],
        })

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

        eval_task = VAIEvalTask(
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
            run_ids=run_ids,
            model_versions=model_versions,
            pass_number=pass_number,
            label=label,
            notes=notes,
        )

    def compare_model_generations(
        self,
        run_ids_by_model: dict[str, str],
        variant: str = "female",
        clinical_category: str | None = None,
    ) -> pd.DataFrame:
        """
        Build a side-by-side comparison DataFrame across model generations.

        Parameters
        ----------
        run_ids_by_model:
            Mapping of model_version → run_id, e.g.:
            {
                "gemini-2.5-pro": "uuid-for-2.5-run",
                "gemini-3.1-pro": "uuid-for-3.1-run",
                "gemini-3.5-pro": "uuid-for-3.5-run",   # when it drops
            }
        variant:
            Demographic variant to compare across models.
        clinical_category:
            Filter to one clinical category. None = all categories.

        Returns
        -------
        DataFrame with one row per vignette_id, columns per model_version
        showing delta_m, tar, crr, and esi_correct.
        """
        all_run_ids = list(run_ids_by_model.values())
        df = self.build_eval_dataset(run_ids=all_run_ids, pass_number=1)

        df = df[df["demographic_variant"] == variant]
        if clinical_category:
            df = df[df["clinical_category"] == clinical_category]

        pivot = df.pivot_table(
            index="vignette_id",
            columns="model_version",
            values=["delta_m", "tar", "esi_correct", "is_low_confidence"],
            aggfunc="first",
        )
        pivot.columns = [f"{metric}__{model}" for metric, model in pivot.columns]
        pivot = pivot.reset_index()

        logger.info(
            "Cross-model comparison: variant=%s category=%s models=%s vignettes=%d",
            variant,
            clinical_category or "all",
            list(run_ids_by_model.keys()),
            len(pivot),
        )
        return pivot
