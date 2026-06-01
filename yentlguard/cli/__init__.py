"""
YentlGuard CLI

Commands:
    baseline    Run nb_ambiguous vignettes to populate baseline spans.
    run         Execute two-pass mechanistic runs.
    analyze     Pull BigQuery data, compute H1–H5, write HTML + CSVs.
    report      Alias for analyze.
    agent       Launch the YentlGuard ADK agent.
    prompts     Seed Phoenix with default prompt templates.

Phoenix MCP integration:
    baseline and run now wire in PhoenixPromptManager (prompt versioning),
    PhoenixDatasetManager (vignette corpus upload), and
    PhoenixExperimentRegistry (experiment registration) when Phoenix env
    vars are present. All three are non-fatal — if Phoenix is unavailable
    the run proceeds with hardcoded defaults and BQ-only storage.
"""

import argparse
import logging
import os

from ._common import _DEFAULT_PHOENIX_MCP_ENDPOINT
from .agent import cmd_agent
from .analyze import cmd_analyze, cmd_report
from .baseline import cmd_baseline
from .prompts import cmd_prompts
from .run import cmd_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("yentlguard.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yentlguard",
        description="Mechanistic interpretability layer for YentlBench triage bias analysis.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose DEBUG logging")
    parser.add_argument("--log-file", type=str, default=None, help="Path to write execution logs")
    sub = parser.add_subparsers(dest="command", required=True)

    # baseline
    p_baseline = sub.add_parser("baseline", help="Populate Phoenix nb_ambiguous baseline spans.")
    p_baseline.add_argument("--model", default="gemini-2.5-pro")
    p_baseline.add_argument("--budget", default="medium", choices=["low", "medium", "high"])
    p_baseline.add_argument("--dataset", default="dataset_output/dataset_quintets.csv")
    p_baseline.set_defaults(func=cmd_baseline)

    # run
    p_run = sub.add_parser("run", help="Execute two-pass mechanistic runs.")
    p_run.add_argument("--model", required=True)
    p_run.add_argument(
        "--budget",
        nargs="+",
        default=["medium"],
        choices=["low", "medium", "high"],
    )
    p_run.add_argument(
        "--variants",
        nargs="+",
        default=["male", "female", "nb_label_only"],
        choices=["male", "female", "nb_ambiguous", "nb_label_only"],
    )
    p_run.add_argument("--dataset", default="dataset_output/dataset_quintets.csv")
    p_run.add_argument("--threshold", type=float, default=1.0)
    p_run.add_argument(
        "--phoenix-mcp-endpoint",
        default=os.environ.get("PHOENIX_MCP_ENDPOINT", _DEFAULT_PHOENIX_MCP_ENDPOINT),
    )
    p_run.add_argument("--experiment-id", default=None)
    p_run.add_argument("--label", default=None)
    p_run.add_argument("--notes", default=None)
    p_run.set_defaults(func=cmd_run)

    # analyze
    p_analyze = sub.add_parser(
        "analyze",
        help="Pull BigQuery run data, compute summaries, write HTML + CSVs.",
    )
    p_analyze.add_argument("--experiment-ids", nargs="+", required=True)
    p_analyze.add_argument("--output", default="results/")
    p_analyze.add_argument("--register-eval", action="store_true", default=False)
    p_analyze.add_argument("--label", default=None)
    p_analyze.add_argument("--notes", default=None)
    p_analyze.set_defaults(func=cmd_analyze)

    # report (alias)
    p_report = sub.add_parser("report", help="Alias for analyze.")
    p_report.add_argument("--experiment-ids", nargs="+", required=True)
    p_report.add_argument("--output", default="results/")
    p_report.add_argument("--register-eval", action="store_true", default=False)
    p_report.add_argument("--label", default=None)
    p_report.add_argument("--notes", default=None)
    p_report.set_defaults(func=cmd_report)

    # prompts
    p_prompts = sub.add_parser(
        "prompts",
        help="Seed Phoenix with the default corrective and distractor prompts.",
    )
    p_prompts.set_defaults(func=cmd_prompts)

    # agent
    p_agent = sub.add_parser(
        "agent",
        help="Launch the YentlGuard ADK agent.",
    )
    p_agent.add_argument(
        "--query",
        default=None,
        metavar="TEXT",
        help="Single-turn query. Omit to launch adk web.",
    )
    p_agent.set_defaults(func=cmd_agent)

    return parser


def main() -> None:
    from yentlguard.config import validate

    parser = build_parser()
    args = parser.parse_args()

    # Apply logging configuration based on args
    log_level = logging.DEBUG if args.verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if args.command in ("run", "baseline", "analyze", "report"):
        try:
            validate()
        except RuntimeError as e:
            print(f"\n{e}\n")
            raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":
    main()
