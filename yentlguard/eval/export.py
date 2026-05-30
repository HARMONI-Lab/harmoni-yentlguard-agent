"""
YentlGuard CSV export.

Writes all AnalysisResult tables to a timestamped output directory.
One CSV per analysis table, plus a manifest file listing what was written
and the experiment_ids included.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from yentlguard.eval.analyze import AnalysisResult

logger = logging.getLogger(__name__)

# Maps attribute name on AnalysisResult → output filename stem
EXPORT_TABLES = {
    "overview": "overview",
    "h1_thinking_budget": "h1_reasoning_mitigation",
    "h2_tar_friction": "h2_cognitive_friction",
    "h3_delta_m": "h3_boundary_invariance",
    "h4_crr": "h4_confidence_recovery",
    "sycophancy": "sycophancy_control",
    "gate_stats": "gate_statistics",
    "cross_model": "cross_model_pivot",
    "raw_pass1": "raw_pass1",
    "raw_pass2": "raw_pass2",
}


def export_csvs(
    result: AnalysisResult,
    output_path: Path,
    timestamp: str | None = None,
) -> dict[str, Path]:
    """
    Write all AnalysisResult DataFrames to CSV files.

    Parameters
    ----------
    result:
        Computed AnalysisResult from Analyzer.run().
    output_path:
        Directory to write CSVs into. Created if absent.
    timestamp:
        Timestamp string for filenames. Auto-generated if None.

    Returns
    -------
    Dict mapping table name → written file Path.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    written: dict[str, Path] = {}

    for attr, stem in EXPORT_TABLES.items():
        df: pd.DataFrame = getattr(result, attr, None)
        if df is None or df.empty:
            logger.info("CSV export: skipping %s (empty)", stem)
            continue

        path = output_path / f"yentlguard_{stem}_{timestamp}.csv"
        df.to_csv(path, index=False)
        written[attr] = path
        logger.info("CSV written: %s (%d rows)", path.name, len(df))

    # Write manifest
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_ids": result.experiment_ids,
        "run_labels": result.run_labels,
        "files": {k: str(v.name) for k, v in written.items()},
        "errors": result.errors,
    }
    manifest_path = output_path / f"yentlguard_manifest_{timestamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written: %s", manifest_path.name)

    return written
