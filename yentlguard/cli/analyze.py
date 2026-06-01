import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from yentlguard.eval.analyze import Analyzer
from yentlguard.eval.export import export_csvs
from yentlguard.eval.report import generate_html_report

logger = logging.getLogger("yentlguard.cli")


def cmd_report(args: argparse.Namespace) -> None:
    cmd_analyze(args)


def cmd_analyze(args: argparse.Namespace) -> None:
    experiment_ids: list[str] = args.experiment_ids
    output_path = Path(args.output)

    if not experiment_ids:
        logger.error("No --experiment-ids provided.")
        return

    logger.info("Pulling data for %d experiment_id(s) from BigQuery...", len(experiment_ids))

    analyzer = Analyzer()
    result = analyzer.run(experiment_ids=experiment_ids)

    if result.raw_pass1.empty:
        logger.warning("No data found for experiment_ids=%s.", experiment_ids)
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    html_path = generate_html_report(
        result=result, output_path=output_path, experiment_ids=experiment_ids
    )
    logger.info("HTML report: %s", html_path)

    csv_files = export_csvs(result=result, output_path=output_path, timestamp=timestamp)
    logger.info("Wrote %d CSV files to %s", len(csv_files), output_path)



    print("\n" + "─" * 60)
    print("  YentlGuard Analysis Complete")
    print("─" * 60)
    print(f"  Experiment IDs analyzed : {len(experiment_ids)}")
    if not result.overview.empty:
        print(f"  Vignettes        : {int(result.overview['n_vignettes'].sum())}")
        models_str = ", ".join(result.overview["model_version"].unique().tolist())
        print(f"  Models           : {models_str}")
    print(f"  Interventions    : {len(result.raw_pass2)}")
    if not result.h4_crr.empty and result.h4_crr["mean_crr"].notna().any():
        print(f"  Mean CRR         : {result.h4_crr['mean_crr'].mean():.4f}")
    print(f"\n  HTML report → {html_path}")
    print(f"  CSVs        → {output_path}")
    print("─" * 60 + "\n")
