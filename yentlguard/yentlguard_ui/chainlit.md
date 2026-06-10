# YentlGuard

**Mechanistic interpretability layer for clinical triage LLM bias.**

YentlGuard provides a structured evaluation pipeline and an advanced **Multi-Agent Framework** to analyze, detect, and mitigate gender bias in clinical triage scenarios powered by Large Language Models (LLMs). Specifically designed for reasoning models like Google's Gemini series, YentlGuard goes beyond basic evaluation by providing an interactive AI agent to conduct deep mechanistic interpretability research on the fly.

By integrating **Arize Phoenix** for telemetry and **Google BigQuery** for robust analytical storage, YentlGuard powers end-to-end research workflows: from prompt engineering and span tracing to statistical analysis and sycophancy detection.

---

## Key Features

- **Mechanistic Evaluation (Two-Pass Pipeline):** Execute structured, multi-pass mechanistic tests (baseline and variant branches) against YentlBench clinical vignettes to measure bias at a token and thought level.
- **Advanced Agent Framework:** An interactive AI research assistant powered by Google ADK (`google-adk`), orchestrating specialized sub-agents for data analysis, observability, and experiment execution.
- **Arize Phoenix Integration:** Comprehensive tracing, span annotation, prompt versioning, and dataset management directly accessible via the Observability sub-agent and MCP (Model Context Protocol).
- **BigQuery Storage & Analytics:** Centralized, immutable storage of evaluation results, allowing the agent to dynamically compute complex hypotheses, degradation metrics, and gate fire rates.
- **Automated Reporting:** Automatically pull BigQuery run data, compute metrics, and generate detailed HTML reports and CSV summaries of experimental runs.

---

## The YentlGuard Agent Framework

The crown jewel of YentlGuard is its **Multi-Agent Architecture** designed to assist researchers in evaluating LLM bias. It utilizes a root supervisor that intelligently delegates tasks to specialized domain agents based on user queries:

### 1. Root Supervisor Agent
Acts as the orchestrator. It parses the user's research request, formulates an execution plan, and transfers control to the appropriate sub-agents via ADK transfer, synthesizing their findings into a final, direct report.

### 2. Data Analyst Agent
Your BigQuery and statistics expert. It specializes in:
- Extracting BigQuery metrics and calculating **ΔM (Token Confidence Margin)** degradation.
- Computing **TAR (Thought Allocation Ratio)** for reasoning tokens (pass 1).
- Analyzing **CRR (Confidence Recovery Rate)**.
- Evaluating **Sycophancy Verdicts**: Distinguishing between genuine debiasing (CRR gap > 0.3) and sycophantic compliance (CRR gap < 0.1).
- Monitoring safety gate fire rates across models.

### 3. Observability & Prompt Engineer Agent
Your Arize Phoenix integration expert. It specializes in:
- **Trace/Span Exploration:** Finding specific vignette execution traces and drill-downs.
- **Span Annotation:** Non-destructively annotating Phoenix spans with calculated sycophancy verdicts for visual analysis.
- **Prompt Versioning:** Iterating, pushing, and tagging prompt templates (e.g., corrective vs. distractor prompts) mapped directly to Phoenix.
- **Anomaly Datasets:** Automatically creating Phoenix datasets for edge cases (e.g., when the safety gate fires >60% of the time).

### 4. Experiment Runner Agent
Safely orchestrates long-running Gemini evaluations. It handles:
- Running `nb_ambiguous` baselines to seed Phoenix spans.
- Executing multi-variant two-pass experiments (e.g., male, female, nb_label_only).
- Estimating costs and budget caps.

---

## Mechanistic Metrics Tracked

YentlGuard introduces key metrics to understand the "how" and "why" behind an LLM's triage decisions:

- **ΔM (Token Confidence Margin):** Measures the delta in logprobs/confidence margins between passes.
- **TAR (Thought Allocation Ratio):** Measures the proportion of reasoning tokens dedicated to processing gender variables versus clinical facts.
- **CRR (Confidence Recovery Rate):** Determines how much confidence the model recovers when challenged with a distractor prompt vs a genuine clinical prompt. Used heavily to detect model *sycophancy*.

---

## Installation

Requires Python 3.11+. We strongly recommend using a virtual environment.

```bash
# Clone the repository
git clone https://github.com/HARMONI-Lab/harmoni-yentlguard-agent.git
cd yentlguard

# Install the package with core dependencies
pip install .

# Or, install with optional dependencies (dev, notebook, ui)
pip install .[dev,notebook,ui]
```

## Configuration

YentlGuard requires environment variables for GCP and Arize Phoenix configuration. Create a `.env` file in the root directory:

```env
# GCP Configuration
YENTLGUARD_GCP_PROJECT=your-gcp-project-id
YENTLGUARD_GCP_LOCATION=us-central1
YENTLGUARD_BQ_DATASET=your-bq-dataset-id
YENTLGUARD_GCS_BUCKET=your-gcs-bucket-name

# Arize Phoenix Configuration
PHOENIX_API_KEY=your_phoenix_api_key
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your_workspace/ # Or local: http://localhost:6006
PHOENIX_MCP_ENDPOINT=https://app.phoenix.arize.com/s/your_workspace/

# Default Evaluation Model
GEMINI_MODEL=gemini-2.5-pro
```

*(Note: Ensure you have `gcloud auth application-default login` configured for Vertex/BigQuery access).*

---

## Infrastructure Setup

To successfully run experiments and use the UI, follow these steps to provision the necessary infrastructure.

### 1. Google Cloud Platform (GCP) Preparation
*   **Enable APIs:** Ensure the **Vertex AI API**, **BigQuery API**, and **Cloud Storage API** are enabled in your GCP project.
*   **Authentication:** Set up your environment with Application Default Credentials so the CLI scripts can access GCP:
    ```bash
    gcloud auth application-default login
    gcloud config set project <your-project-id>
    ```

### 2. Automated Infrastructure Provisioning (BigQuery & GCS)
*   **BigQuery Automation:** YentlGuard includes a schema migration script to automatically provision the required BigQuery dataset and tables. Run:
    ```bash
    python -m yentlguard.eval.schema
    ```
*   **Cloud Storage (Optional but Recommended):** The GCS bucket (defined by `YENTLGUARD_GCS_BUCKET` in `.env`) is used to store analysis reports and dead-letter queues. 
    ```bash
    gsutil mb -l us-central1 gs://<your-bucket-name>
    ```

### 3. Arize Phoenix Configuration
*   **Hosting:** You can use [Arize Cloud](https://app.phoenix.arize.com/) or run a local instance: `docker run -p 6006:6006 arizeai/phoenix`.
*   **MCP Requirements:** Running the YentlGuard agent locally requires Node.js installed on your system to support the `@arizeai/phoenix-mcp` package (as indicated in the provided Dockerfile).

### 4. Dataset Management in Phoenix
Phoenix needs to know about your vignettes to trace the experiments correctly.
*   **Upload Corpus & Prompts:** Use the included setup script to push both your YentlBench dataset (`dataset_quintets.csv`) and the default prompt templates to Phoenix in one command:
    ```bash
    python -m yentlguard.mcp.setup_phoenix --dataset dataset_output/dataset_quintets.csv
    ```
*   **Creating Splits (Crucial Manual UI Step):** Open the Phoenix UI, navigate to the `yentlbench-quintets-all-variants` dataset, click on the **Splits** tab, and manually assign the data into splits based on the demographic variants (e.g., `nb_ambiguous`, `male`, `female`, `nb_label_only`). 


### 5. Deploying the YentlGuard UI (Cloud Run)
YentlGuard provides a `cloudbuild.yaml` file to deploy the Chainlit UI directly to Google Cloud Run.
*   **Store Secrets:** First, save your Phoenix API key to GCP Secret Manager:
    ```bash
    gcloud secrets create PHOENIX_API_KEY --replication-policy="automatic" --data-file=- <<< "your-api-key"
    ```
*   **Deploy:** Run the build and deployment process:
    ```bash
    gcloud builds submit --config cloudbuild.yaml
    ```

---

## CLI Commands & Workflow

YentlGuard is driven primarily via its CLI: `yentlguard`.

### 1. Start the YentlGuard Agent
Launch the interactive ADK agent to perform research, run queries, or orchestrate evaluations.
```bash
# Open interactive ADK web session/terminal
yentlguard agent

```

### 2. Seed Prompts
Seed Arize Phoenix with the default corrective and distractor prompt templates used during the two-pass mechanistic tests:
```bash
yentlguard prompts
```

### 3. Generate Baselines
Populate Phoenix with baseline reasoning spans using the non-binary ambiguous (`nb_ambiguous`) vignettes:
```bash
yentlguard baseline --model gemini-2.5-flash --budget medium
```

### 4. Execute Runs
Execute the full two-pass mechanistic runs against specific variants:
```bash
yentlguard run --model gemini-2.5-flash --budget medium --variants female
```

### 5. Analyze & Report
Pull evaluation data from BigQuery, compute summaries and hypotheses (H1-H5), and output detailed HTML/CSV reports:
```bash
yentlguard analyze --experiment-ids <exp_id_1> <exp_id_2> --output results/
```
*(Note: `yentlguard report` is a direct alias for `yentlguard analyze`)*

---

## Development & Testing

Tools for contributors modifying YentlGuard:

- **Linting & Formatting:** `ruff check .`
- **Type Checking:** `mypy .`
- **Testing:** `pytest tests/`
- **Database Migrations:** `python -m yentlguard.eval.schema`

##  License
This project is licensed under the MIT License.

## Authors
- Inna Rytsareva (<inna@harmonilab.org>)
