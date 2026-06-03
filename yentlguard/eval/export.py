"""
YentlGuard CSV export.

Writes all AnalysisResult tables to a timestamped output directory.
One CSV per analysis table, plus a manifest file listing what was written
and the experiment_ids included.
"""

import io
import json
import logging
import zipfile
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage as gcs

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
    bucket_name: str,
    timestamp: str | None = None,
) -> dict[str, str]:
    """
    Write all AnalysisResult DataFrames to CSV files and zip them to GCS.

    Parameters
    ----------
    result:
        Computed AnalysisResult from Analyzer.run().
    bucket_name:
        GCS bucket name to write artifacts into.
    timestamp:
        Timestamp string for filenames. Auto-generated if None.

    Returns
    -------
    Dict mapping table name / artifact name → GCS URI.
    """
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        bucket.create()
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    written: dict[str, str] = {}
    
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for attr, stem in EXPORT_TABLES.items():
            df: pd.DataFrame = getattr(result, attr, None)
            if df is None or df.empty:
                logger.info("CSV export: skipping %s (empty)", stem)
                continue

            csv_name = f"yentlguard_{stem}_{timestamp}.csv"
            zf.writestr(csv_name, df.to_csv(index=False))
            written[attr] = csv_name
            logger.info("CSV added to zip: %s (%d rows)", csv_name, len(df))

    zip_blob_name = f"exports/yentlguard_exports_{timestamp}.zip"
    zip_blob = bucket.blob(zip_blob_name)
    zip_blob.upload_from_string(zip_buf.getvalue(), content_type="application/zip")
    
    exports_uri = f"https://storage.googleapis.com/{bucket_name}/{zip_blob_name}"
    written["exports_zip"] = exports_uri
    logger.info("Exports zip written: %s", exports_uri)

    # Write manifest
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_ids": result.experiment_ids,
        "run_labels": result.run_labels,
        "files": written,
        "errors": result.errors,
    }
    
    manifest_blob_name = f"exports/yentlguard_manifest_{timestamp}.json"
    manifest_blob = bucket.blob(manifest_blob_name)
    manifest_blob.upload_from_string(json.dumps(manifest, indent=2), content_type="application/json")
    
    manifest_uri = f"https://storage.googleapis.com/{bucket_name}/{manifest_blob_name}"
    written["manifest"] = manifest_uri
    logger.info("Manifest written: %s", manifest_uri)

    return written
