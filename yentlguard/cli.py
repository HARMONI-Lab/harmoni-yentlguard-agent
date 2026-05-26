"""
YentlGuard CLI

Commands:
    baseline    Run nb_ambiguous vignettes through a model to populate Phoenix
                baseline spans. Must be run before any corrective loop.

    run         Execute two-pass mechanistic runs across all demographic variants
                for a vignette set, triggering correction gate where applicable.

    analyze     Pull completed run data from BigQuery, compute H1–H4 summary
                statistics, and write a self-contained HTML report + CSV files.

    report      Alias for analyze (backward compatibility).

    agent       Launch the YentlGuard ADK agent for interactive experiment
                planning, analysis, and hypothesis interpretation.

Usage:
    yentlguard baseline --model gemini-2.5-pro --budget medium

    yentlguard run \\
        --model gemini-2.5-pro --budget medium \\
        --variants male female nb_label_only \\
        --label "gemini-2.5 baseline May 2026"

    yentlguard run \\
        --model gemini-3.1-pro --budget low medium high \\
        --variants male female nb_label_only

    yentlguard analyze \\
        --run-ids <run_id_1> <run_id_2> \\
        --output results/ \\
        --register-eval

    yentlguard agent                        # launches adk web (browser UI)
    yentlguard agent --query "..."          # single-turn, prints and exits
"""

import argparse
import logging
import os
import yentlguard.config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("yentlguard.cli")

# Default Phoenix MCP endpoint — space-scoped cloud URL or base URL
# Override with --phoenix-mcp-endpoint for a different space or local Phoenix.
_DEFAULT_PHOENIX_MCP_ENDPOINT = "https://app.phoenix.arize.com"



def _get_completed_vignettes(model: str, budget: str, variant: str) -> set[str]:
    from google.cloud import bigquery
    from yentlguard.config import RUNS_TABLE, GCP_PROJECT_ID
    client = bigquery.Client(project=GCP_PROJECT_ID)
    query = f"""
        SELECT DISTINCT vignette_id
        FROM `{RUNS_TABLE}`
        WHERE model_version = @model
          AND thinking_budget = @budget
          AND demographic_variant = @variant
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("model", "STRING", model),
            bigquery.ScalarQueryParameter("budget", "STRING", budget),
            bigquery.ScalarQueryParameter("variant", "STRING", variant),
        ]
    )
    try:
        df = client.query(query, job_config=job_config).to_dataframe()
        return set(df["vignette_id"].astype(str).tolist())
    except Exception as e:
        import logging
        logging.getLogger("yentlguard.cli").warning("Failed to check completed vignettes: %s", e)
        return set()


def cmd_baseline(args: argparse.Namespace) -> None:
    """Populate Phoenix with nb_ambiguous baseline spans."""
    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    logger.info("Initializing Phoenix tracing...")
    provider = setup_phoenix_tracing()

    runner = YentlGuardRunner(
        model_version=args.model,
        thinking_budget=args.budget,
        phoenix_mcp_client=None,  # baseline pass: no MCP lookup needed
    )

    import pathlib as _pathlib
    import uuid as _uuid

    import pandas as _pd
    from yentlbench.local_runner.prompt import build_prompt as _build_prompt

    from yentlguard.eval.bq_writer import BQWriter

    dataset_path = _pathlib.Path(args.dataset)
    if not dataset_path.exists():
        logger.error(
            "Dataset not found: %s\n"
            "Run: yentlbench prepare  (requires MIMIC-IV-ED data)\n"
            "Or set --dataset to point at your dataset_quintets.csv",
            dataset_path,
        )
        raise SystemExit(1)

    df = _pd.read_csv(dataset_path)
    df = df[df["acuity"].notna()]
    df_variant = df[df["gender_variant"] == "nb_ambiguous"]
    logger.info("Loaded %d nb_ambiguous vignettes from %s", len(df_variant), dataset_path)

    run_id = str(_uuid.uuid4())
    logger.info("Baseline BQ run_id: %s", run_id)

    with BQWriter(run_id=run_id, gate_threshold=1.0) as bq:
        bq.register_experiment(
            label=f"baseline {args.model} {args.budget}",
            models=[args.model],
            thinking_budgets=[args.budget],
            variants=["nb_ambiguous"],
            vignette_count=len(df_variant),
            notes="Baseline pass for nb_ambiguous",
        )

        completed = _get_completed_vignettes(args.model, args.budget, "nb_ambiguous")
        if completed:
            logger.info("Found %d already completed vignettes for %s %s nb_ambiguous. Skipping.", len(completed), args.model, args.budget)

        for _, row in df_variant.iterrows():
            vignette = row.to_dict()
            vignette_id = str(int(vignette["source_stay_id"]))
            if vignette_id in completed:
                continue
            text = _build_prompt(vignette, "nb_ambiguous")
            run = runner.run(
                vignette_id=vignette_id,
                vignette_text=text,
                demographic_variant="nb_ambiguous",
            )

            esi_gt = str(int(vignette["acuity"])) if not _pd.isna(vignette.get("acuity")) else None
            cat = str(vignette.get("chiefcomplaint", "")) or None
            bq.write(run=run, esi_ground_truth=esi_gt, clinical_category=cat)

            status = "✓" if not run.errors else "✗"
            dm = run.pass1_delta_m.delta_m if run.pass1_delta_m and run.pass1_delta_m.delta_m else None
            logger.info(
                "%s %s | ESI=%s | ΔM=%.4f",
                status,
                vignette_id,
                run.pass1_esi or "?",
                dm or 0.0,
            )

    logger.info("Baseline run complete. Spans available in Phoenix project: yentlguard")
    if provider:
        provider.force_flush()


def cmd_run(args: argparse.Namespace) -> None:
    """Execute two-pass mechanistic runs for specified variants."""
    import uuid

    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.eval.bq_writer import BQWriter
    from yentlguard.mcp.phoenix_client import PhoenixMCPClient
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    provider = setup_phoenix_tracing()

    mcp_client = PhoenixMCPClient(mcp_endpoint=args.phoenix_mcp_endpoint)
    logger.info("Phoenix MCP endpoint: %s", args.phoenix_mcp_endpoint)

    run_id = args.run_id or str(uuid.uuid4())
    logger.info("Experiment run_id: %s", run_id)

    import pathlib as _pathlib

    import pandas as _pd
    from yentlbench.local_runner.prompt import build_prompt as _build_prompt

    dataset_path = _pathlib.Path(args.dataset)
    if not dataset_path.exists():
        logger.error(
            "Dataset not found: %s\n"
            "Run: yentlbench prepare  (requires MIMIC-IV-ED data)\n"
            "Or set --dataset to point at your dataset_quintets.csv",
            dataset_path,
        )
        raise SystemExit(1)

    df_all = _pd.read_csv(dataset_path)
    df_all = df_all[df_all["acuity"].notna()]

    n_per_variant = len(df_all[df_all["gender_variant"] == args.variants[0]])

    with BQWriter(run_id=run_id, gate_threshold=args.threshold) as bq:

        bq.register_experiment(
            label=args.label or f"{args.model} {','.join(args.budget)} {','.join(args.variants)}",
            models=[args.model],
            thinking_budgets=args.budget,
            variants=args.variants,
            vignette_count=n_per_variant * len(args.variants) * len(args.budget),
            notes=args.notes,
        )

        for budget in args.budget:
            runner = YentlGuardRunner(
                model_version=args.model,
                thinking_budget=budget,
                delta_m_threshold=args.threshold,
                phoenix_mcp_client=mcp_client,
            )

            for variant in args.variants:
                vignettes_df = df_all[df_all["gender_variant"] == variant]
                completed = _get_completed_vignettes(args.model, budget, variant)

                if completed:
                    vignettes_df = vignettes_df[~vignettes_df["source_stay_id"].astype(int).astype(str).isin(completed)]
                    logger.info("Skipped %d already completed vignettes.", len(completed))

                if vignettes_df.empty:
                    logger.info("All vignettes already completed for model=%s | budget=%s | variant=%s. Skipping.", args.model, budget, variant)
                    continue

                logger.info(
                    "Running %d vignettes | model=%s | budget=%s | variant=%s",
                    len(vignettes_df), args.model, budget, variant,
                )
                for _, row in vignettes_df.iterrows():
                    vignette = row.to_dict()
                    vignette_id = str(int(vignette["source_stay_id"]))
                    text = _build_prompt(vignette, variant)
                    esi_gt = str(int(vignette["acuity"])) if not _pd.isna(vignette.get("acuity")) else None
                    clinical_cat = str(vignette.get("chiefcomplaint", "")) or None
                    run = runner.run(
                        vignette_id=vignette_id,
                        vignette_text=text,
                        demographic_variant=variant,
                    )
                    bq.write(
                        run=run,
                        esi_ground_truth=esi_gt,
                        clinical_category=clinical_cat,
                    )
                    if run.crr:
                        dist_crrs = [
                            r.crr for r in [
                                run.crr_distractor_a,
                                run.crr_distractor_b,
                                run.crr_distractor_c,
                            ] if r is not None
                        ]
                        max_dist = max(dist_crrs) if dist_crrs else None
                        gap_str = f" | gap={run.crr.crr - max_dist:.3f}" if max_dist is not None else ""
                        logger.info(
                            "  %s | CRR=%.3f%s | ESI %s->%s | intervention=%s",
                            vignette_id,
                            run.crr.crr,
                            gap_str,
                            run.pass1_esi,
                            run.pass2_esi,
                            run.intervention_triggered,
                        )

    logger.info("Run complete. Query results: SELECT * FROM `%s` WHERE run_id = '%s'", "runs", run_id)
    if provider:
        provider.force_flush()


def cmd_report(args: argparse.Namespace) -> None:
    """Alias for analyze — kept for backward compatibility."""
    cmd_analyze(args)


def cmd_analyze(args: argparse.Namespace) -> None:
    """
    Pull completed run data from BigQuery, compute all summary statistics,
    and write a self-contained HTML report + CSV files to the output directory.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    from yentlguard.eval.analyze import Analyzer
    from yentlguard.eval.export import export_csvs
    from yentlguard.eval.report import generate_html_report

    run_ids: list[str] = args.run_ids
    output_path = Path(args.output)

    if not run_ids:
        logger.error("No --run-ids provided. Pass at least one run_id to analyze.")
        return

    logger.info("Pulling data for %d run_id(s) from BigQuery...", len(run_ids))

    analyzer = Analyzer()
    result = analyzer.run(run_ids=run_ids)

    if result.raw_pass1.empty:
        logger.warning(
            "No data found for run_ids=%s. "
            "Verify run_ids exist in BigQuery and the runs table is populated.",
            run_ids,
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info("Generating HTML report...")
    html_path = generate_html_report(
        result=result,
        output_path=output_path,
        run_ids=run_ids,
    )
    logger.info("HTML report: %s", html_path)

    logger.info("Exporting CSVs...")
    csv_files = export_csvs(
        result=result,
        output_path=output_path,
        timestamp=timestamp,
    )
    logger.info("Wrote %d CSV files to %s", len(csv_files), output_path)

    if args.register_eval:
        from yentlguard.eval.agent_builder import AgentBuilderEvalLayer
        logger.info("Registering Agent Builder eval task...")
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
                "Agent Builder eval task registered: %s | models=%s",
                task.task_id,
                task.model_versions,
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
        mean_crr = result.h4_crr["mean_crr"].mean()
        print(f"  Mean CRR         : {mean_crr:.4f}")
    print(f"\n  HTML report → {html_path}")
    print(f"  CSVs        → {output_path}")
    print("─" * 60 + "\n")


def cmd_agent(args: argparse.Namespace) -> None:
    """
    Launch the YentlGuard ADK agent.

    Without --query: opens adk web (browser UI) — recommended for interactive
    analysis sessions where you want to explore results, ask follow-up questions,
    or plan follow-up experiments.

    With --query TEXT: runs a single agent turn and exits. Useful for scripting
    or one-shot questions (e.g. from a Makefile or CI step).
    """
    import asyncio
    import secrets
    import subprocess
    import sys
    from pathlib import Path

    if args.query:
        # Single-turn mode
        from google.adk.runners import InMemoryRunner
        from google.genai import types
        from yentlguard.agent.yentlguard_agent.agent import root_agent

        async def _run_single_turn(query: str) -> None:
            runner = InMemoryRunner(agent=root_agent, app_name="yentlguard")
            session_id = secrets.token_hex(8)
            await runner.session_service.create_session(
                app_name="yentlguard",
                user_id="cli_user",
                session_id=session_id,
            )
            async for event in runner.run_async(
                user_id="cli_user",
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=query)],
                ),
            ):
                if hasattr(event, "content") and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            print(part.text, end="", flush=True)
            print()

        asyncio.run(_run_single_turn(args.query))

    else:
        # Interactive adk web mode
        agent_dir = str(
            (
                Path(__file__).parent
                / "agent"
                / "yentlguard_agent"
            ).resolve()
        )
        logger.info("Launching adk web → %s", agent_dir)
        result = subprocess.run(
            [sys.executable, "-m", "google.adk.cli", "web", agent_dir],
            check=False,
        )
        sys.exit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yentlguard",
        description="Mechanistic interpretability layer for YentlBench triage bias analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── baseline ──────────────────────────────────────────────────────────────
    p_baseline = sub.add_parser("baseline", help="Populate Phoenix nb_ambiguous baseline spans.")
    p_baseline.add_argument("--model", default="gemini-2.5-pro")
    p_baseline.add_argument("--budget", default="medium", choices=["low", "medium", "high"])
    p_baseline.add_argument(
        "--dataset",
        default="dataset_output/dataset_quintets.csv",
        help="Path to dataset_quintets.csv produced by: yentlbench prepare",
    )
    p_baseline.set_defaults(func=cmd_baseline)

    # ── run ───────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Execute two-pass mechanistic runs.")
    p_run.add_argument("--model", required=True, help="e.g. gemini-2.5-pro or gemini-3.1-pro")
    p_run.add_argument(
        "--budget", nargs="+", default=["medium"],
        choices=["low", "medium", "high"],
        help="Thinking budget tier(s). Multiple values run sequentially.",
    )
    p_run.add_argument(
        "--variants", nargs="+",
        default=["male", "female", "nb_label_only"],
        choices=["male", "female", "nb_ambiguous", "nb_label_only"],
    )
    p_run.add_argument(
        "--dataset",
        default="dataset_output/dataset_quintets.csv",
        help="Path to dataset_quintets.csv produced by: yentlbench prepare",
    )
    p_run.add_argument(
        "--threshold", type=float, default=1.0,
        help="ΔM threshold below which correction gate fires (default: 1.0 nat).",
    )
    p_run.add_argument(
        "--phoenix-mcp-endpoint",
        default=os.environ.get("PHOENIX_MCP_ENDPOINT", _DEFAULT_PHOENIX_MCP_ENDPOINT),
        help=(
            "Phoenix Base URL. "
            "Cloud default: https://app.phoenix.arize.com "
            "Local Phoenix: http://localhost:6006"
        ),
    )
    p_run.add_argument(
        "--run-id", default=None,
        help="Experiment batch UUID. Auto-generated if not provided.",
    )
    p_run.add_argument(
        "--label", default=None,
        help="Human-readable experiment label for BigQuery experiments table.",
    )
    p_run.add_argument(
        "--notes", default=None,
        help="Free-text notes about this experiment batch.",
    )
    p_run.set_defaults(func=cmd_run)

    # ── analyze ───────────────────────────────────────────────────────────────
    p_analyze = sub.add_parser(
        "analyze",
        help="Pull BigQuery run data, compute summaries, write HTML report + CSVs.",
    )
    p_analyze.add_argument(
        "--run-ids", nargs="+", required=True,
        help="One or more experiment batch run_ids to include in this analysis.",
    )
    p_analyze.add_argument(
        "--output", default="results/",
        help="Output directory for HTML report and CSVs (default: results/).",
    )
    p_analyze.add_argument(
        "--register-eval", action="store_true", default=False,
        help="Register results as an Agent Builder eval task in Vertex AI.",
    )
    p_analyze.add_argument(
        "--label", default=None,
        help="Label for the Agent Builder eval task (used with --register-eval).",
    )
    p_analyze.add_argument(
        "--notes", default=None,
        help="Free-text notes attached to the Agent Builder eval task.",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # ── report (alias) ────────────────────────────────────────────────────────
    p_report = sub.add_parser(
        "report",
        help="Alias for analyze. Kept for backward compatibility.",
    )
    p_report.add_argument("--run-ids", nargs="+", required=True)
    p_report.add_argument("--output", default="results/")
    p_report.add_argument("--register-eval", action="store_true", default=False)
    p_report.add_argument("--label", default=None)
    p_report.add_argument("--notes", default=None)
    p_report.set_defaults(func=cmd_report)

    # ── agent ─────────────────────────────────────────────────────────────────
    p_agent = sub.add_parser(
        "agent",
        help=(
            "Launch the YentlGuard ADK agent. "
            "Omit --query to open adk web (browser UI). "
            "Pass --query TEXT for a single-turn CLI mode."
        ),
    )
    p_agent.add_argument(
        "--query", default=None,
        metavar="TEXT",
        help=(
            "Run a single agent turn and exit. "
            "Omit to launch the interactive adk web browser UI."
        ),
    )
    p_agent.set_defaults(func=cmd_agent)

    return parser


def main() -> None:
    from yentlguard.config import validate
    parser = build_parser()
    args = parser.parse_args()
    # agent command does its own config validation inside tool calls,
    # not at startup — allows launching the agent even before GCP is configured.
    if args.command in ("run", "baseline", "analyze", "report"):
        try:
            validate()
        except RuntimeError as e:
            print(f"\n{e}\n")
            raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":
    main()
