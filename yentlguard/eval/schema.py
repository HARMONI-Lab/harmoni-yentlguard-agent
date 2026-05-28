"""
YentlGuard BigQuery schema.

Changes from original:
    - Added prompt_version_* columns to RUNS_SCHEMA so every row records
      which Phoenix prompt version was used for the corrective and distractor
      branches. NULL when falling back to hardcoded defaults.

Run once to create dataset and tables:
    python -m yentlguard.eval.schema
"""

from google.cloud import bigquery

from yentlguard.config import (
    BQ_DATASET_ID,
    BQ_LOCATION,
    EXPTS_TABLE,
    FULL_DATASET,
    GCP_PROJECT_ID,
    RUNS_TABLE,
)

RUNS_SCHEMA = [
    # ── Experiment keys ────────────────────────────────────────────────────
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED",
        description="UUID identifying the experiment batch."),
    bigquery.SchemaField("row_id", "STRING", mode="REQUIRED",
        description="UUID for this row. Primary key."),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED",
        description="UTC timestamp when this row was written."),

    # ── Vignette identity ──────────────────────────────────────────────────
    bigquery.SchemaField("vignette_id", "STRING", mode="REQUIRED",
        description="YentlBench vignette identifier, e.g. ED_00147."),
    bigquery.SchemaField("clinical_category", "STRING", mode="NULLABLE",
        description="Clinical category tag, e.g. chest_pain."),
    bigquery.SchemaField("esi_ground_truth", "STRING", mode="NULLABLE",
        description="Ground truth ESI level from MIMIC-IV-ED (1–5)."),

    # ── Model & config ─────────────────────────────────────────────────────
    bigquery.SchemaField("model_version", "STRING", mode="REQUIRED",
        description="Gemini model string, e.g. gemini-2.5-pro."),
    bigquery.SchemaField("model_family", "STRING", mode="NULLABLE",
        description="Coarse family for cross-generation grouping, e.g. gemini-2.5."),
    bigquery.SchemaField("thinking_budget", "STRING", mode="NULLABLE",
        description="ThinkingConfig tier: low, medium, high, or null."),
    bigquery.SchemaField("temperature", "FLOAT64", mode="NULLABLE",
        description="Generation temperature (0.0 for all benchmark runs)."),

    # ── Demographic variant ────────────────────────────────────────────────
    bigquery.SchemaField("demographic_variant", "STRING", mode="REQUIRED",
        description="YentlBench variant: male, female, nb_ambiguous, nb_label_only, nb_explicit."),
    bigquery.SchemaField("pass_number", "INT64", mode="REQUIRED",
        description="1 = initial run, 2 = corrective re-prompt."),

    # ── ESI predictions ────────────────────────────────────────────────────
    bigquery.SchemaField("esi_predicted", "STRING", mode="NULLABLE",
        description="ESI digit predicted by the model (1–5)."),
    bigquery.SchemaField("esi_correct", "BOOL", mode="NULLABLE",
        description="True if esi_predicted == esi_ground_truth."),
    bigquery.SchemaField("esi_direction_error", "STRING", mode="NULLABLE",
        description="over_triage, under_triage, or null if correct."),

    # ── Delta-M ───────────────────────────────────────────────────────────
    bigquery.SchemaField("delta_m", "FLOAT64", mode="NULLABLE",
        description="Token confidence margin: logprob(top) - logprob(runner-up)."),
    bigquery.SchemaField("top_logprob", "FLOAT64", mode="NULLABLE",
        description="Log probability of the chosen ESI token."),
    bigquery.SchemaField("runner_up_token", "STRING", mode="NULLABLE",
        description="Best competing ESI digit at the same position."),
    bigquery.SchemaField("runner_up_logprob", "FLOAT64", mode="NULLABLE",
        description="Log probability of the runner-up ESI token."),
    bigquery.SchemaField("esi_token_index", "INT64", mode="NULLABLE",
        description="Token position where the ESI digit appeared."),
    bigquery.SchemaField("is_low_confidence", "BOOL", mode="NULLABLE",
        description="True if delta_m < configured threshold (default 1.0 nat)."),
    bigquery.SchemaField("raw_text", "STRING", mode="NULLABLE",
        description="Raw output text from the model generation."),

    # ── TAR (Pass 1 only; null on Pass 2 by design) ───────────────────────
    bigquery.SchemaField("tar", "FLOAT64", mode="NULLABLE",
        description="Thought Allocation Ratio: thoughts_token_count / candidates_token_count."),
    bigquery.SchemaField("thoughts_token_count", "INT64", mode="NULLABLE",
        description="Internal reasoning tokens consumed before generation."),
    bigquery.SchemaField("candidates_token_count", "INT64", mode="NULLABLE",
        description="Output generation tokens."),
    bigquery.SchemaField("is_high_friction", "BOOL", mode="NULLABLE",
        description="True if TAR > 2.0."),

    # ── Correction gate ────────────────────────────────────────────────────
    bigquery.SchemaField("gate_fired", "BOOL", mode="NULLABLE",
        description="True if the correction gate triggered for this vignette × variant."),
    bigquery.SchemaField("gate_threshold", "FLOAT64", mode="NULLABLE",
        description="ΔM threshold used for gate decision."),
    bigquery.SchemaField("gate_trigger_token", "STRING", mode="NULLABLE",
        description="Which demographic token matched in the vignette."),
    bigquery.SchemaField("gate_trigger_position", "INT64", mode="NULLABLE",
        description="Character index where the token matched in the vignette text."),

    # ── MCP baseline ──────────────────────────────────────────────────────
    bigquery.SchemaField("baseline_delta_m", "FLOAT64", mode="NULLABLE",
        description="ΔM from nb_ambiguous baseline for this vignette (from BQ lookup)."),
    bigquery.SchemaField("mcp_lookup_success", "BOOL", mode="NULLABLE",
        description="Whether the baseline lookup succeeded."),

    # ── Phoenix prompt version tracking ───────────────────────────────────
    # NULL when YentlGuardRunner fell back to hardcoded defaults (no Phoenix).
    # Populated when PhoenixPromptManager fetched a versioned prompt.
    # Join to Phoenix prompt versions via these IDs to compare CRR across
    # experiments that used different prompt versions.
    bigquery.SchemaField("prompt_version_corrective", "STRING", mode="NULLABLE",
        description="Phoenix prompt version ID used for the corrective branch. "
                    "NULL if hardcoded default was used."),
    bigquery.SchemaField("prompt_version_distractor_a", "STRING", mode="NULLABLE",
        description="Phoenix prompt version ID for distractor 3a (Pure Clinical Anchor)."),
    bigquery.SchemaField("prompt_version_distractor_b", "STRING", mode="NULLABLE",
        description="Phoenix prompt version ID for distractor 3b (Forced Parsing Anchor)."),
    bigquery.SchemaField("prompt_version_distractor_c", "STRING", mode="NULLABLE",
        description="Phoenix prompt version ID for distractor 3c (Protocol Anchor)."),

    # ── CRR (populated only on Pass 2 rows) ───────────────────────────────
    bigquery.SchemaField("crr", "FLOAT64", mode="NULLABLE",
        description="Confidence Recovery Rate after corrective re-prompt."),
    bigquery.SchemaField("triage_changed", "BOOL", mode="NULLABLE",
        description="True if Pass 2 predicted a different ESI than Pass 1."),
    bigquery.SchemaField("recovery_class", "STRING", mode="NULLABLE",
        description="full (CRR>=0.95), partial (0.1<=CRR<0.95), failed (CRR<0.1)."),

    # ── Sycophancy controls (Pass 3a/b/c) — Pass 2 rows only ──────────────
    bigquery.SchemaField("delta_m_pass3a", "FLOAT64", mode="NULLABLE",
        description="ΔM from Pass 3a: Pure Clinical Anchor distractor."),
    bigquery.SchemaField("esi_pass3a", "STRING", mode="NULLABLE",
        description="ESI digit predicted by Pass 3a."),
    bigquery.SchemaField("crr_distractor_a", "FLOAT64", mode="NULLABLE",
        description="CRR for Pass 3a vs nb_ambiguous baseline."),
    bigquery.SchemaField("triage_changed_3a", "BOOL", mode="NULLABLE",
        description="True if Pass 3a ESI differs from Pass 1."),
    bigquery.SchemaField("recovery_class_3a", "STRING", mode="NULLABLE",
        description="full/partial/failed for distractor A."),
    bigquery.SchemaField("raw_text_pass3a", "STRING", mode="NULLABLE",
        description="Raw output text from the Pass 3a distractor generation."),

    bigquery.SchemaField("delta_m_pass3b", "FLOAT64", mode="NULLABLE",
        description="ΔM from Pass 3b: Forced Parsing Anchor distractor."),
    bigquery.SchemaField("esi_pass3b", "STRING", mode="NULLABLE",
        description="ESI digit predicted by Pass 3b."),
    bigquery.SchemaField("crr_distractor_b", "FLOAT64", mode="NULLABLE",
        description="CRR for Pass 3b vs nb_ambiguous baseline."),
    bigquery.SchemaField("triage_changed_3b", "BOOL", mode="NULLABLE",
        description="True if Pass 3b ESI differs from Pass 1."),
    bigquery.SchemaField("recovery_class_3b", "STRING", mode="NULLABLE",
        description="full/partial/failed for distractor B."),
    bigquery.SchemaField("raw_text_pass3b", "STRING", mode="NULLABLE",
        description="Raw output text from the Pass 3b distractor generation."),

    bigquery.SchemaField("delta_m_pass3c", "FLOAT64", mode="NULLABLE",
        description="ΔM from Pass 3c: Protocol Anchor distractor."),
    bigquery.SchemaField("esi_pass3c", "STRING", mode="NULLABLE",
        description="ESI digit predicted by Pass 3c."),
    bigquery.SchemaField("crr_distractor_c", "FLOAT64", mode="NULLABLE",
        description="CRR for Pass 3c vs nb_ambiguous baseline."),
    bigquery.SchemaField("triage_changed_3c", "BOOL", mode="NULLABLE",
        description="True if Pass 3c ESI differs from Pass 1."),
    bigquery.SchemaField("recovery_class_3c", "STRING", mode="NULLABLE",
        description="full/partial/failed for distractor C."),
    bigquery.SchemaField("raw_text_pass3c", "STRING", mode="NULLABLE",
        description="Raw output text from the Pass 3c distractor generation."),

    # Sycophancy summary — computed at write time for fast querying
    bigquery.SchemaField("max_distractor_crr", "FLOAT64", mode="NULLABLE",
        description="Max CRR across 3a/3b/3c. Close to crr → possible sycophancy."),
    bigquery.SchemaField("crr_vs_distractor_gap", "FLOAT64", mode="NULLABLE",
        description="crr minus max_distractor_crr. Large positive = genuine debiasing. "
                    "Near zero = possible sycophancy."),

    # ── Errors ────────────────────────────────────────────────────────────
    bigquery.SchemaField("errors", "STRING", mode="REPEATED",
        description="Error messages if any step failed during this row's execution."),
]


EXPERIMENTS_SCHEMA = [
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED",
        description="UUID. Foreign key to runs.run_id."),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED",
        description="UTC timestamp when the experiment was registered."),
    bigquery.SchemaField("label", "STRING", mode="REQUIRED",
        description="Human-readable name, e.g. 'gemini-2.5-pro baseline May 2026'."),
    bigquery.SchemaField("models", "STRING", mode="REPEATED",
        description="Model versions included in this run."),
    bigquery.SchemaField("thinking_budgets", "STRING", mode="REPEATED",
        description="Thinking budget tiers used."),
    bigquery.SchemaField("variants", "STRING", mode="REPEATED",
        description="Demographic variants included."),
    bigquery.SchemaField("vignette_count", "INT64", mode="NULLABLE",
        description="Total vignettes in this run."),
    bigquery.SchemaField("notes", "STRING", mode="NULLABLE",
        description="Free-text notes."),
    bigquery.SchemaField("yentlbench_version", "STRING", mode="NULLABLE",
        description="YentlBench package version."),
    bigquery.SchemaField("yentlguard_version", "STRING", mode="NULLABLE",
        description="YentlGuard package version."),
]


def create_dataset_and_tables(client: bigquery.Client | None = None) -> None:
    """
    Create the YentlGuard BigQuery dataset and tables if they do not exist.
    Safe to run multiple times — uses exists_ok=True throughout.

    Usage:
        python -m yentlguard.eval.schema
    """
    client = client or bigquery.Client(project=GCP_PROJECT_ID)

    dataset = bigquery.Dataset(f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}")
    dataset.location = BQ_LOCATION
    dataset.description = (
        "YentlGuard mechanistic interpretability eval results. "
        "Tracks ΔM, TAR, CRR across Gemini model generations on "
        "YentlBench triage vignettes. Published by HARMONI Lab."
    )
    client.create_dataset(dataset, exists_ok=True)
    print(f"Dataset ready: {FULL_DATASET}")

    runs_table = bigquery.Table(RUNS_TABLE, schema=RUNS_SCHEMA)
    runs_table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="created_at",
    )
    runs_table.clustering_fields = [
        "model_version", "demographic_variant", "run_id"
    ]
    runs_table.description = (
        "One row per vignette × variant × pass execution. "
        "Partitioned by created_at (day). "
        "Clustered by model_version, demographic_variant, run_id."
    )
    client.create_table(runs_table, exists_ok=True)
    print(f"Table ready: {RUNS_TABLE}")

    expts_table = bigquery.Table(EXPTS_TABLE, schema=EXPERIMENTS_SCHEMA)
    expts_table.description = (
        "One row per experiment batch. JOIN to runs on run_id."
    )
    client.create_table(expts_table, exists_ok=True)
    print(f"Table ready: {EXPTS_TABLE}")


if __name__ == "__main__":
    create_dataset_and_tables()
