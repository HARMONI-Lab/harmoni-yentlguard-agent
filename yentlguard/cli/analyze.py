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
    run_ids: list[str] = args.run_ids
    output_path = Path(args.output)

    if not run_ids:
        logger.error("No --run-ids provided.")
        return

    logger.info("Pulling data for %d run_id(s) from BigQuery...", len(run_ids))

    analyzer = Analyzer()
    result = analyzer.run(run_ids=run_ids)

    if result.raw_pass1.empty:
        logger.warning("No data found for run_ids=%s.", run_ids)
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    html_path = generate_html_report(
        result=result, output_path=output_path, run_ids=run_ids
    )
    logger.info("HTML report: %s", html_path)

    csv_files = export_csvs(
        result=result, output_path=output_path, timestamp=timestamp
    )
    logger.info("Wrote %d CSV files to %s", len(csv_files), output_path)

    if args.register_eval:
        from yentlguard.eval.agent_builder import AgentBuilderEvalLayer

        try:
            layer = AgentBuilderEvalLayer()
            models = result.overview["model_version"].unique().tolist()
            task = layer.register_eval_task(
                run_ids=run_ids,
                label=args.label or f"yentlguard-analyze-{timestamp}",
                model_versions=models,
                notes=args.notes,
            )
            logger.info(
                "Agent Builder eval task registered: %s", task.task_id
            )
        except Exception as e:
            logger.warning("Agent Builder registration failed (non-fatal): %s", e)

    print("\n" + "─" * 60)
    print("  YentlGuard Analysis Complete")
    print("─" * 60)
    print(f"  Run IDs analyzed : {len(run_ids)}")
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
