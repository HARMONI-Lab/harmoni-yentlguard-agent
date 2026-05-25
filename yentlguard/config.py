"""
YentlGuard GCP configuration.

Single source of truth for project, location, and dataset settings.
All modules import from here rather than declaring their own placeholders.

Fill in the values below before running any YentlGuard commands.
These can also be set via environment variables — env vars take precedence.

Environment variables:
    YENTLGUARD_GCP_PROJECT   — GCP project ID
    YENTLGUARD_GCP_LOCATION  — Vertex AI region
    YENTLGUARD_BQ_DATASET    — BigQuery dataset ID
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── FILL THESE IN ──────────────────────────────────────────────────────────────
GCP_PROJECT_ID = os.environ.get("YENTLGUARD_GCP_PROJECT", "YOUR_GCP_PROJECT_ID")
GCP_LOCATION   = os.environ.get("YENTLGUARD_GCP_LOCATION", "YOUR_GCP_LOCATION")   # e.g. "us-central1"
BQ_DATASET_ID  = os.environ.get("YENTLGUARD_BQ_DATASET",  "YOUR_BQ_DATASET_ID")   # e.g. "yentlguard"
BQ_LOCATION    = "US"   # BigQuery dataset region — usually fine to leave as US
# ──────────────────────────────────────────────────────────────────────────────

# Derived table references — do not edit these
FULL_DATASET = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}"
RUNS_TABLE   = f"{FULL_DATASET}.runs"
EXPTS_TABLE  = f"{FULL_DATASET}.experiments"


def validate() -> None:
    """
    Raise if any required config placeholder is still unfilled.
    Call at CLI startup before any GCP calls.
    """
    missing = []
    if GCP_PROJECT_ID == "YOUR_GCP_PROJECT_ID":
        missing.append("YENTLGUARD_GCP_PROJECT (or edit config.py GCP_PROJECT_ID)")
    if GCP_LOCATION == "YOUR_GCP_LOCATION":
        missing.append("YENTLGUARD_GCP_LOCATION (or edit config.py GCP_LOCATION)")
    if BQ_DATASET_ID == "YOUR_BQ_DATASET_ID":
        missing.append("YENTLGUARD_BQ_DATASET (or edit config.py BQ_DATASET_ID)")
    if missing:
        raise RuntimeError(
            "YentlGuard GCP configuration incomplete. Set the following:\n"
            + "\n".join(f"  • {m}" for m in missing)
        )
