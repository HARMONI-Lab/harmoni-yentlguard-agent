# YentlGuard: Technical Description and Deployment Guide

**HARMONI Lab** | harmonilab.org  
Published by Inna Campo, Ph.D.

---

## What YentlGuard Is

YentlGuard is a mechanistic interpretability layer built on top of YentlBench. Where YentlBench asks *whether* LLMs exhibit demographic bias in clinical triage — measuring accuracy degradation when patient sex or gender changes — YentlGuard asks *how*, at the level of individual tokens.

It instruments live Gemini inference runs through Vertex AI and captures the exact mathematical moment when a demographic label shifts a model's certainty about an ESI triage decision. The core insight: you don't need access to weights or attention matrices to observe mechanistic behavior. Token-level log probabilities expose the model's internal confidence at every position in the generated sequence, including the position where it commits to an ESI digit (1–5). A model that is genuinely uncertain between ESI 2 and ESI 3 will show a narrow margin between those two tokens' log probabilities. A model influenced by a demographic token will show that narrowing specifically when a female or non-binary label is present, compared to the same vignette with no demographic label.

YentlGuard makes that narrowing measurable, traceable, and testable across model generations.

---

## The Three Metrics

**ΔM — Token Confidence Margin**

The primary mechanistic signal. Computed at the exact token position where Gemini generates the ESI digit (1–5):

```
ΔM = logprob(chosen ESI token) − logprob(best competing ESI token)
```

A large ΔM (e.g. 4.5 nats) means the model committed firmly — ESI 3 was ~90x more probable than ESI 2 at that position. A small ΔM (e.g. 0.3 nats) means the model was nearly split — the mechanistic signature of demographic-induced triage instability. The ESI 2↔3 boundary is the safety-critical crossing: ESI 2 means the patient needs to be seen within 15 minutes; ESI 3 means they can wait. Unwarranted demographic influence at that specific boundary has direct patient safety implications.

**TAR — Thought Allocation Ratio**

Available on thinking models (Gemini 2.5 Pro, 3.1 Pro with ThinkingConfig enabled):

```
TAR = thoughts_token_count / candidates_token_count
```

This measures the ratio of internal reasoning tokens to output generation tokens. The Demographic Cognitive Friction hypothesis (H2) predicts that a female presentation matching a textbook male-prototype condition — specifically chest pain — will produce a higher TAR than the male baseline, because the model spends more internal reasoning reconciling its demographic schema before committing to a triage level. A female patient presenting with chest pain should not require more "thinking" than a male patient presenting with identical vitals. If it does, the excess compute is a measurable friction cost imposed by the demographic label.

**CRR — Confidence Recovery Rate**

The intervention metric. When the correction gate fires (ΔM below threshold AND a demographic trigger token present), YentlGuard issues a second Gemini call — a corrective re-prompt that foregrounds vital signs as the primary decision anchor and explicitly instructs the model not to factor in demographic information:

```
CRR = (ΔM_pass2 − ΔM_pass1) / (ΔM_baseline − ΔM_pass1)
```

Where ΔM_baseline is the confidence margin the model showed for the same vignette under the `nb_ambiguous` condition (no demographic label), retrieved from Arize Phoenix historical spans via MCP. CRR = 1.0 means the corrective re-prompt fully recovered baseline confidence. CRR = 0 means it had no effect. CRR < 0 means it made things worse. This metric operationalizes the Selective Surgery Problem: can you suppress unwarranted demographic influence through prompt intervention alone, without touching the model weights?

---

## The Four Research Hypotheses

**H1 — Reasoning Mitigation Effect**  
Does scaling the ThinkingConfig budget from low (512 tokens) to medium (2048) to high (8192) reduce the Perturbation Sensitivity Score across demographic variants? If PSS decreases as budget increases, it supports the interpretation that extended internal reasoning actively suppresses surface-level demographic token associations rather than just generating more text.

**H2 — Demographic Cognitive Friction**  
Does the presence of a female or non-binary label on a chest-pain presentation produce a measurably higher TAR than the male baseline for the same vignette? Higher TAR = more reasoning tokens = cognitive friction imposed by the demographic label. This is a distinct claim from accuracy: the model might still get the triage level right while expending disproportionate reasoning effort to do so.

**H3 — Mathematical Boundary Invariance**  
Does Gemini 3.1 Pro maintain wider ΔM under demographic perturbation than Gemini 2.5 Pro, particularly at the ESI 2↔3 boundary? A newer model that is more invariant to demographic tokens should show ΔM distributions that are similar across male, female, and non-binary variants. A model that collapses ΔM specifically on female presentations at ESI 2↔3 is exhibiting the precise failure mode YentlGuard was built to detect.

**H4 — Selective Surgery via CRR**  
Does vital-sign-foregrounding corrective re-prompting recover the `nb_ambiguous` baseline confidence, and does recovery rate vary by clinical category? This tests whether the demographic influence is shallow (overridable by prompt) or deep (persistent through prompt intervention). The answer has direct implications for the feasibility of in-deployment demographic bias mitigation without model retraining.

---

## Architecture

YentlGuard sits between YentlBench (which provides the vignettes) and three external services: Vertex AI (Gemini inference), Arize Phoenix (observability), and BigQuery/Agent Builder (eval storage and cross-model comparison).

```
YentlBench vignettes (nb_ambiguous, male, female, nb_label_only, nb_explicit)
        │
        ▼
YentlGuardRunner
  • genai.Client(vertexai=True)           ← Vertex AI backend, ADC credentials
  • response_logprobs=True, logprobs=5    ← top-5 token alternatives per position
  • ThinkingConfig(thinking_budget=N)     ← low / medium / high
        │
        ├─── OpenInference GoogleGenAIInstrumentor
        │           │
        │           ▼
        │    Arize Phoenix (OTel spans)
        │    ├── vignette_trace root span
        │    ├── pass1 generation span (enriched: ΔM, TAR, vignette_id, variant)
        │    │   ├── pass1.metrics child span
        │    │   │   ├── pass1.delta_m grandchild (full logprob breakdown)
        │    │   │   └── pass1.tar grandchild (token count breakdown)
        │    ├── correction_gate span (always created, fired=True/False)
        │    ├── mcp.baseline_lookup span (Phoenix MCP query result)
        │    ├── pass2 generation span (corrective re-prompt, if gate fired)
        │    │   ├── pass2.metrics child span
        │    │   │   └── pass2.delta_m grandchild
        │    └── crr span (recovery computation)
        │
        ├─── BQWriter (streaming insert after each vignette)
        │    ├── runs table (one row per pass per vignette)
        │    └── experiments table (one row per run_id batch)
        │
        └─── PhoenixMCPClient
             └── get_baseline_delta_m(vignette_id, variant="nb_ambiguous")
                 ← retrieves historical ΔM from Phoenix span store

Post-run:
yentlguard analyze --run-ids <id1> <id2>
        │
        ├─── Analyzer (7 BigQuery queries → H1–H4 + gate stats + pivot)
        ├─── generate_html_report() → self-contained dark-mode HTML
        ├─── export_csvs() → 9 CSVs + JSON manifest
        └─── AgentBuilderEvalLayer (optional --register-eval)
             └── Vertex AI Agent Builder eval task for GCP console comparison
```

---

## Repository Structure

```
yentlguard/
├── pyproject.toml                  # Package config, dependencies, CLI entry point
├── README.md
└── yentlguard/
    ├── config.py                   # Single source of truth for GCP settings
    ├── cli.py                      # yentlguard baseline | run | analyze
    ├── agent/
    │   └── runner.py               # YentlGuardRunner: two-pass inference loop
    ├── metrics/
    │   ├── delta_m.py              # ΔM computation from logprobs_result
    │   ├── tar.py                  # TAR from usage_metadata
    │   └── crr.py                  # CRR from pass1/pass2/baseline ΔM
    ├── telemetry/
    │   ├── phoenix.py              # Phoenix tracing setup (OpenInference + OTel)
    │   └── annotation.py           # Span enrichment + child/grandchild spans
    ├── mcp/
    │   └── phoenix_client.py       # MCP client for nb_ambiguous baseline lookup
    └── eval/
        ├── schema.py               # BigQuery dataset + table creation
        ├── bq_writer.py            # Streaming BQ insert after each vignette
        ├── analyze.py              # 7 BQ queries → AnalysisResult dataclass
        ├── report.py               # Self-contained HTML report generator
        ├── export.py               # CSV export + JSON manifest
        └── agent_builder.py        # Vertex AI Agent Builder eval task registration
```

---

## Prerequisites

Before deploying YentlGuard you need the following in place:

- Python 3.11 or higher
- A Google Cloud project with billing enabled
- The `gcloud` CLI installed and authenticated
- YentlBench published to PyPI (or installable from GitHub)
- An Arize Phoenix account (cloud) or a self-hosted Phoenix instance

---

## Step-by-Step Deployment

### Stage 1: Google Cloud Setup

**Step 1.1 — Create or identify your GCP project.**  
Go to console.cloud.google.com. Either create a new project (e.g. `harmonilab-research`) or use an existing one. Note the Project ID — this is what goes into `config.py`, not the display name.

**Step 1.2 — Enable required APIs.**  
In the GCP console, navigate to APIs & Services → Library and enable each of the following:

- Vertex AI API (`aiplatform.googleapis.com`)
- BigQuery API (`bigquery.googleapis.com`)
- BigQuery Storage API (`bigquerystorage.googleapis.com`)
- Cloud Resource Manager API (`cloudresourcemanager.googleapis.com`)

Alternatively, enable all four with gcloud:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  bigquerystorage.googleapis.com \
  cloudresourcemanager.googleapis.com \
  --project YOUR_PROJECT_ID
```

**Step 1.3 — Authenticate Application Default Credentials.**  
YentlGuard uses ADC — no API keys, no service account JSON files to manage. Run:

```bash
gcloud auth application-default login
```

Follow the browser prompt. This writes credentials to `~/.config/gcloud/application_default_credentials.json`, which the `google-genai` SDK and BigQuery client pick up automatically.

**Step 1.4 — Verify Vertex AI access to Gemini.**  
Run a quick smoke test to confirm your project can reach the Gemini models:

```bash
python3 - <<'EOF'
from google import genai
client = genai.Client(vertexai=True, project="YOUR_PROJECT_ID", location="us-central1")
response = client.models.generate_content(model="gemini-2.5-pro", contents="ping")
print(response.text)
EOF
```

If this returns text, Vertex AI is working. If you get a 403, check that the Vertex AI API is enabled and your account has the `roles/aiplatform.user` IAM role.

**Step 1.5 — Create the BigQuery dataset.**  
Choose a region. `US` (multi-region) is the default in `config.py` and works for most HARMONI Lab use cases. Create the dataset:

```bash
bq mk --dataset \
  --location=US \
  --description="YentlGuard mechanistic interpretability eval results" \
  YOUR_PROJECT_ID:yentlguard
```

---

### Stage 2: Arize Phoenix Setup

**Step 2.1 — Create an Arize Phoenix account.**  
Go to app.phoenix.arize.com and sign up for a free workspace. The free tier is sufficient for research use.

**Step 2.2 — Create a project.**  
In the Phoenix UI, create a new project named `yentlguard`. This is the project name that spans will be tagged with.

**Step 2.3 — Get your API key and collector endpoint.**  
In Phoenix, go to Settings → API Keys. Generate a key. Also note your space's collector endpoint — it looks like `https://app.phoenix.arize.com/s/your-space-name`.

**Step 2.4 — Set Phoenix environment variables.**  
These are needed at runtime. Add them to your shell profile or `.env` file:

```bash
export PHOENIX_API_KEY=your_phoenix_api_key
export PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your-space-name
```

**Step 2.5 — Verify Phoenix connectivity.**  
A quick test:

```bash
python3 - <<'EOF'
import os, requests
key = os.environ["PHOENIX_API_KEY"]
endpoint = os.environ["PHOENIX_COLLECTOR_ENDPOINT"]
r = requests.get(f"{endpoint}/v1/traces", headers={"api_key": key})
print(r.status_code)  # Expect 200 or 404 (404 means endpoint exists but no traces yet)
EOF
```

---

### Stage 3: YentlGuard Installation

**Step 3.1 — Clone the repository.**

```bash
git clone https://github.com/harmonilab/yentlguard.git
cd yentlguard
```

**Step 3.2 — Create a virtual environment.**  
Python 3.11 minimum. Do not use system Python.

```bash
python3.11 -m venv .venv
source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate      # Windows
```

**Step 3.3 — Install YentlGuard in editable mode with all dependencies.**

```bash
pip install -e ".[dev]"
```

This installs YentlGuard itself, YentlBench (from PyPI), the `google-genai` SDK, the BigQuery client, Arize Phoenix, OpenInference instrumentation, OpenTelemetry, and all dev tools (pytest, ruff, mypy).

**Step 3.4 — Verify the installation.**

```bash
yentlguard --help
```

You should see the three subcommands: `baseline`, `run`, `analyze`.

---

### Stage 4: Configuration

**Step 4.1 — Fill in `config.py`.**  
Open `yentlguard/config.py`. Set three values:

```python
GCP_PROJECT_ID = "harmonilab-research"   # your actual GCP project ID
GCP_LOCATION   = "us-central1"           # Vertex AI region
BQ_DATASET_ID  = "yentlguard"            # BigQuery dataset name from Step 1.5
```

Alternatively, set them as environment variables (these take precedence over the file):

```bash
export YENTLGUARD_GCP_PROJECT=harmonilab-research
export YENTLGUARD_GCP_LOCATION=us-central1
export YENTLGUARD_BQ_DATASET=yentlguard
```

**Step 4.2 — Provision BigQuery tables.**  
Run the schema script once. It creates the `runs` and `experiments` tables with the correct schema, partitioning, and clustering. Safe to run multiple times — uses `exists_ok=True`:

```bash
python -m yentlguard.eval.schema
```

Expected output:

```
Dataset ready: harmonilab-research.yentlguard
Table ready: harmonilab-research.yentlguard.runs
Table ready: harmonilab-research.yentlguard.experiments
```

**Step 4.3 — Validate the full config.**  
The CLI validates config before any API call:

```bash
yentlguard baseline --help
```

If config is incomplete, you'll see a clear error listing which values are missing before any network call is made.

---

### Stage 5: Prepare the YentlBench Dataset

YentlGuard reads `dataset_quintets.csv` produced by YentlBench. This file is not bundled — you must generate it once from the MIMIC-IV-ED Demo dataset.

**Step 5.0a — Obtain MIMIC-IV-ED Demo from PhysioNet.**

Go to physionet.org and complete the Data Use Agreement for MIMIC-IV-ED Demo (v2.2). Download and unzip so that the following path exists in your working directory:

```
mimic-iv-ed-demo-2.2/ed/edstays.csv
mimic-iv-ed-demo-2.2/ed/triage.csv
```

**Step 5.0b — Run YentlBench dataset preparation.**

```bash
yentlbench prepare
```

This joins `edstays` and `triage`, filters to male patients, excludes clinically confounded complaints (abdominal pain, etc.), and expands each record into five demographic variants. Output:

```
dataset_output/dataset_males.csv    # 87 curated male stays
dataset_output/dataset_quintets.csv # 435 rows (87 stays × 5 variants)
```

The five variants in the CSV are: `nb_ambiguous`, `female`, `male`, `nb_label_only`, `nb_full`. YentlGuard uses only the four variants in `config.ALL_VARIANTS` — `nb_full` is in the CSV but intentionally excluded from the benchmark pipeline by YentlBench design.

**Step 5.0c — Verify the dataset.**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('dataset_output/dataset_quintets.csv')
print(f'Rows: {len(df)}')
print(f'Variants: {sorted(df.gender_variant.unique())}')
print(f'Quintets: {df.quintet_id.nunique()}')
print(df.groupby("gender_variant").size())
"
```

Expected output: 435 rows, 87 unique quintets, 5 variants with 87 rows each. Pass the path via `--dataset` if it differs from the default:

```bash
yentlguard baseline --dataset /path/to/dataset_quintets.csv --model gemini-2.5-pro --budget medium
```

---

### Stage 6: Running the Experiment Pipeline

The pipeline has three sequential phases: `baseline` → `run` → `analyze`. They must be run in this order for the first experiment, because `run` uses Phoenix baseline spans populated by `baseline` to compute CRR.

**Step 6.1 — Populate the nb_ambiguous baseline.**  
This runs every vignette in the `nb_ambiguous` variant (no demographic label present) through your chosen model and captures the ΔM for each vignette in Phoenix. These become the recovery targets for CRR computation in the `run` phase.

```bash
yentlguard baseline \
  --model gemini-2.5-pro \
  --budget medium
```

Each vignette produces a log line:

```
✓ ED_00147 | ESI=3 | ΔM=2.3471
✓ ED_00203 | ESI=2 | ΔM=0.8823
```

Vignettes with ΔM < 1.0 in the baseline are inherently ambiguous even without demographic signal — important context for interpreting CRR results later. When this completes, open Phoenix and verify spans are appearing under the `yentlguard` project.

**Step 6.2 — Run the Gemini 2.5 Pro mechanistic experiment.**  
This executes the two-pass correction loop across your chosen variants. Start with `female` and `nb_label_only` as they are the most likely to trigger the correction gate:

```bash
yentlguard run \
  --model gemini-2.5-pro \
  --budget low medium high \
  --variants female nb_label_only \
  --label "gemini-2.5-pro full sweep May 2026" \
  --notes "First mechanistic run. Baseline populated 2026-05-21."
```

For each vignette where the gate fires, you will see:

```
ED_00203 | CRR=0.743 | ESI 3→3 | intervention=True
ED_00419 | CRR=0.211 | ESI 3→2 | intervention=True
```

The first line shows partial recovery with no triage change. The second shows failed recovery with a triage change — a safety-relevant case where the corrective re-prompt actually shifted the predicted ESI level.

Note the `run_id` printed at the start of this command. It is a UUID auto-generated per experiment batch. Save it — you will pass it to `analyze`.

```
Experiment run_id: a3f7c219-4b81-4e2c-b8d0-1c2e5f8a9d3b
```

**Step 6.3 — Run the Gemini 3.1 Pro experiment.**  
Same vignettes, same variants, same thinking budgets — different model:

```bash
yentlguard run \
  --model gemini-3.1-pro \
  --budget low medium high \
  --variants female nb_label_only \
  --label "gemini-3.1-pro full sweep May 2026" \
  --notes "Comparison run against 2.5-pro. Same vignette set."
```

Save this run_id too.

**Step 6.4 — Run remaining variants.**  
Once the primary variants are complete, run `nb_explicit` (non-binary with explicit label) and `male` (baseline behavioral reference):

```bash
yentlguard run \
  --model gemini-2.5-pro \
  --budget medium \
  --variants male nb_explicit \
  --label "gemini-2.5-pro male and nb_explicit May 2026"

yentlguard run \
  --model gemini-3.1-pro \
  --budget medium \
  --variants male nb_explicit \
  --label "gemini-3.1-pro male and nb_explicit May 2026"
```

---

### Stage 7: Analysis and Reporting

**Step 7.1 — Generate the analysis report.**  
Pass both run_ids from the 2.5 Pro and 3.1 Pro experiments:

```bash
yentlguard analyze \
  --run-ids a3f7c219-4b81-4e2c-b8d0-1c2e5f8a9d3b \
             b8e2a447-9c12-4f3d-a71e-2d3f6g9b0e4c \
  --output results/may_2026/ \
  --register-eval \
  --label "YentlGuard 2.5 vs 3.1 comparison May 2026"
```

The `--register-eval` flag sends results to Vertex AI Agent Builder for structured eval scoring in the GCP console. Omit it if you haven't set up Agent Builder yet — the HTML and CSVs will still be generated.

Terminal output:

```
────────────────────────────────────────────────────────────
  YentlGuard Analysis Complete
────────────────────────────────────────────────────────────
  Run IDs analyzed : 2
  Vignettes        : 140
  Models           : gemini-2.5-pro, gemini-3.1-pro
  Interventions    : 43
  Mean CRR         : 0.6127

  HTML report → results/may_2026/yentlguard_analysis_20260521_143022.html
  CSVs        → results/may_2026/
────────────────────────────────────────────────────────────
```

**Step 6.2 — Open the HTML report.**  
The report is fully self-contained — no internet connection required to render it. Open it in any browser. It contains:

- Overview cards: accuracy, mean ΔM, mean TAR, mean CRR, and gate fire count per model
- H1 table: PSS by model × thinking budget × demographic variant
- H2 table: TAR by model × clinical category × variant
- H3 table: ΔM distribution at ESI digit, broken out by model × variant × ESI level
- H4 table: CRR by model × variant × clinical category, with full/partial/failed recovery counts
- Gate statistics: fire rate distribution across all runs
- Cross-model vignette pivot: per-vignette side-by-side ΔM, TAR, ESI prediction across both models

**Step 6.3 — Review CSVs.**  
The `results/may_2026/` directory contains nine CSV files and a JSON manifest:

- `yentlguard_overview_*.csv`
- `yentlguard_h1_reasoning_mitigation_*.csv`
- `yentlguard_h2_cognitive_friction_*.csv`
- `yentlguard_h3_boundary_invariance_*.csv`
- `yentlguard_h4_confidence_recovery_*.csv`
- `yentlguard_gate_statistics_*.csv`
- `yentlguard_cross_model_pivot_*.csv`
- `yentlguard_raw_pass1_*.csv`
- `yentlguard_raw_pass2_*.csv`
- `yentlguard_manifest_*.json`

The raw pass CSVs contain every column in the BigQuery `runs` table for full reproducibility.

---

### Stage 8: When Gemini 3.5 Pro Drops

This is the payoff of the longitudinal design. No structural changes needed:

**Step 7.1 — Run the baseline again for the new model** (optional — the existing `nb_ambiguous` spans can be reused if you want to compare against the same baseline, or regenerated with the new model for an apples-to-apples baseline per model version):

```bash
yentlguard baseline --model gemini-3.5-pro --budget medium
```

**Step 7.2 — Run the experiment:**

```bash
yentlguard run \
  --model gemini-3.5-pro \
  --budget low medium high \
  --variants female nb_label_only nb_explicit \
  --label "gemini-3.5-pro initial sweep"
```

**Step 7.3 — Analyze all three generations together:**

```bash
yentlguard analyze \
  --run-ids <2.5-pro-run-id> <3.1-pro-run-id> <3.5-pro-run-id> \
  --output results/three_generation_comparison/ \
  --label "YentlGuard cross-generation 2.5 vs 3.1 vs 3.5"
```

The cross-model pivot table and H1/H3 summaries will automatically include the new model version. All BigQuery queries use `model_family` as a grouping key specifically to support this pattern — `gemini-2.5`, `gemini-3.1`, `gemini-3.5` will each appear as rows.

---

## Querying BigQuery Directly

The `runs` table is partitioned by `created_at` and clustered on `model_version, demographic_variant, run_id`. Several queries are worth keeping handy:

**ESI 2↔3 boundary cases — the safety-critical rows:**

```sql
SELECT
  vignette_id, model_version, demographic_variant, thinking_budget,
  esi_predicted, esi_ground_truth, delta_m, is_low_confidence, crr
FROM `YOUR_PROJECT.yentlguard.runs`
WHERE esi_predicted IN ('2','3')
  AND esi_ground_truth IN ('2','3')
  AND demographic_variant = 'female'
  AND pass_number = 1
ORDER BY delta_m ASC
LIMIT 50;
```

**Vignettes where triage changed after corrective re-prompt:**

```sql
SELECT
  vignette_id, model_version, demographic_variant, clinical_category,
  esi_predicted, crr, recovery_class
FROM `YOUR_PROJECT.yentlguard.runs`
WHERE triage_changed = TRUE
  AND pass_number = 2
ORDER BY crr ASC;
```

**Mean ΔM by model generation and variant — the H3 core table:**

```sql
SELECT
  model_family,
  demographic_variant,
  ROUND(AVG(delta_m), 4) AS mean_delta_m,
  ROUND(AVG(baseline_delta_m - delta_m), 4) AS mean_pss,
  COUNT(*) AS n
FROM `YOUR_PROJECT.yentlguard.runs`
WHERE pass_number = 1
  AND delta_m IS NOT NULL
GROUP BY 1, 2
ORDER BY model_family, demographic_variant;
```

---

## Troubleshooting

**"YentlGuard GCP configuration incomplete"**  
Config validation fired before any API call. Set the three env vars or fill in `config.py`. Check with `echo $YENTLGUARD_GCP_PROJECT`.

**"No Phoenix spans found for vignette_id=X, variant=nb_ambiguous"**  
The baseline step has not been run yet, or ran but failed silently on that vignette. Check the baseline run logs. The correction gate will still fire and Pass 2 will still run — CRR will be null for that vignette rather than crashing.

**ΔM extraction returns None for all vignettes**  
The `response_logprobs=True` parameter is not being honored. Verify that the model version supports logprobs on Vertex AI — check the Vertex AI model documentation for the specific model string you are using. Also confirm `logprobs=5` is set in the config.

**TAR is always None**  
Thinking is not enabled for this model or budget. Check that `ThinkingConfig` is supported for the model version. Gemini 2.5 Flash does not expose `thoughts_token_count`; use Gemini 2.5 Pro or 3.1 Pro.

**BigQuery streaming insert errors**  
Most commonly a schema mismatch if you modified `schema.py` after the tables were created. Drop and recreate the tables by deleting them in the BigQuery console and re-running `python -m yentlguard.eval.schema`.

**Agent Builder eval task registration failed**  
This is logged as a warning and does not stop the analysis. The HTML and CSVs are still written. Agent Builder eval is optional — the core research output does not depend on it.
