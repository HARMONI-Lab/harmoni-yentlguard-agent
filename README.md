# YentlGuard

**Mechanistic interpretability layer for clinical triage LLM bias — built on [YentlBench](https://github.com/harmonilab/yentlbench)**.

YentlGuard provides a structured evaluation pipeline and an agent framework to analyze, detect, and mitigate gender bias in clinical triage scenarios powered by Large Language Models (LLMs), focusing specifically on the capabilities of Google Vertex AI and Gemini models.

It leverages [Arize Phoenix](https://phoenix.arize.com/) for deep telemetry, prompt management, and experiment tracking, while persisting evaluation runs directly to Google BigQuery for further analysis and reporting.

## 🚀 Features

- **Mechanistic Evaluation:** Run two-pass mechanistic tests (baseline and variants) using YentlBench vignettes.
- **Arize Phoenix Integration:** Comprehensive tracing, span annotation, dataset management, and prompt versioning.
- **BigQuery Storage:** Centralized storage of evaluation results and experiments for easy querying and reporting.
- **ADK Agent Framework:** Built-in AI Agent powered by `google-adk` for interaction and deeper analysis.
- **Detailed Analytics:** Automatically pull BigQuery run data, compute hypotheses (H1–H5), and generate HTML reports and CSV summaries.

## 📦 Installation

Requires Python 3.11+. We recommend using a virtual environment.

```bash
# Clone the repository
git clone https://github.com/harmonilab/yentlguard.git
cd yentlguard

# Install the package with core dependencies
pip install .

# Or, install with optional dependencies (dev, notebook, ui)
pip install .[dev,notebook,ui]
```

## ⚙️ Configuration

YentlGuard requires environment variables for GCP and Arize Phoenix configuration.

Create a `.env` file in the root directory (or export these directly):

```bash
# GCP Configuration
YENTLGUARD_GCP_PROJECT=your-gcp-project-id
YENTLGUARD_GCP_LOCATION=us-central1
YENTLGUARD_BQ_DATASET=your-bq-dataset-id

# Arize Phoenix Configuration
PHOENIX_API_KEY=your_phoenix_api_key
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your_workspace/ # Or local: http://localhost:6006
PHOENIX_MCP_ENDPOINT=https://app.phoenix.arize.com/s/your_workspace/

# Default Agent Model
GEMINI_MODEL=gemini-2.5-pro
```

Alternatively, you can edit `yentlguard/config.py` directly, though using environment variables is recommended.

## 🛠️ Usage / CLI Commands

YentlGuard is driven primarily via its CLI: `yentlguard`.

### 1. Seed Prompts
Seed Arize Phoenix with the default corrective and distractor prompt templates:
```bash
yentlguard prompts
```

### 2. Generate Baselines
Populate Phoenix with baseline spans using the `nb_ambiguous` (non-binary ambiguous) vignettes.
```bash
yentlguard baseline --model gemini-2.5-pro --budget medium
```

### 3. Execute Runs
Execute the two-pass mechanistic runs against specific variants (e.g., male, female, nb_label_only).
```bash
yentlguard run --model gemini-2.5-pro --budget medium --variants male female
```

### 4. Analyze & Report
Pull evaluation data from BigQuery, compute summaries and hypotheses, and write out HTML reports and CSVs.
```bash
yentlguard analyze --experiment-ids <exp_id_1> <exp_id_2> --output results/
```
*(Note: `yentlguard report` is an alias for `yentlguard analyze`)*

### 5. Launch the ADK Agent
Start the YentlGuard interactive AI agent.
```bash
# Open interactive ADK web session
yentlguard agent

# Or run a single query
yentlguard agent --query "Analyze the recent triage run for bias."
```

## 📊 Development & Testing

If you are developing YentlGuard, you can use the provided tools:

- **Linting & Formatting:** `ruff check .`
- **Type Checking:** `mypy .`
- **Testing:** `pytest tests/`

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

## 🤝 Authors

- Inna Rytsareva (<inna@harmonilab.org>) - [HarmoniLab](https://harmonilab.org)
