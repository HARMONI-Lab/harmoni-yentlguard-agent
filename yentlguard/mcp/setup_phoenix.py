"""
YentlGuard Phoenix setup script.

Run once before your first experiment to seed Phoenix with:
    1. Default corrective and distractor prompt versions
    2. Vignette corpus as a Phoenix dataset (optional — requires dataset CSV)

After this runs, YentlGuardRunner will fetch prompts from Phoenix at run time
and every prompt version will be tracked alongside BigQuery metric rows.

Usage:
    python -m yentlguard.mcp.setup_phoenix
    python -m yentlguard.mcp.setup_phoenix --dataset dataset_output/dataset_quintets.csv
    python -m yentlguard.mcp.setup_phoenix --prompts-only
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("yentlguard.phoenix_setup")


def push_default_prompts() -> None:
    from yentlguard.mcp.phoenix_manager import PhoenixPromptManager

    mgr = PhoenixPromptManager()

    if not mgr._available:
        logger.error("Phoenix is not reachable. Check PHOENIX_BASE_URL and PHOENIX_API_KEY.")
        sys.exit(1)

    logger.info("Pushing default prompt versions to Phoenix...")
    mgr.push_all_defaults()
    logger.info("Done. Verify at: %s", mgr._base_url)


def push_corpus(dataset_path: str) -> None:
    import pathlib

    import pandas as pd
    from yentlguard.prompting.prompt import build_prompt

    from yentlguard.mcp.phoenix_manager import PhoenixDatasetManager

    path = pathlib.Path(dataset_path)
    if not path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    df = df[df["acuity"].notna()]

    logger.info("Building corpus rows for all variants...")
    variants = ["nb_ambiguous", "male", "female", "nb_label_only"]
    rows = []
    for variant in variants:
        vdf = df[df["gender_variant"] == variant].copy()
        if vdf.empty:
            logger.info("  variant=%s: 0 rows — skipping", variant)
            continue
        vdf["vignette_text"] = vdf.apply(lambda r: build_prompt(r.to_dict(), variant), axis=1)
        vdf["esi_ground_truth"] = vdf["acuity"].apply(
            lambda v: str(int(v)) if pd.notna(v) else None
        )
        vdf["clinical_category"] = vdf.get("chiefcomplaint", pd.Series(dtype=str)).fillna("")
        vdf["source_stay_id"] = vdf["source_stay_id"].astype(str)
        vdf["demographic_variant"] = variant
        rows.append(
            vdf[
                [
                    "source_stay_id",
                    "vignette_text",
                    "demographic_variant",
                    "clinical_category",
                    "esi_ground_truth",
                ]
            ]
        )
        logger.info("  variant=%s: %d rows", variant, len(vdf))

    if not rows:
        logger.error("No rows built. Check gender_variant column in dataset.")
        sys.exit(1)

    corpus_df = pd.concat(rows, ignore_index=True)
    logger.info("Total corpus rows: %d", len(corpus_df))

    mgr = PhoenixDatasetManager()
    dataset_id = mgr.push_vignette_corpus(
        df=corpus_df,
        dataset_name="yentlbench-quintets-all-variants",
    )
    if dataset_id:
        logger.info("Corpus uploaded. Phoenix dataset_id=%s", dataset_id)
    else:
        logger.warning("Corpus upload failed. Check Phoenix connection.")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Seed Phoenix with YentlGuard default prompts and vignette corpus."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to dataset_quintets.csv. If omitted, only prompts are pushed.",
    )
    parser.add_argument(
        "--prompts-only",
        action="store_true",
        default=False,
        help="Push only prompt defaults, skip corpus upload.",
    )
    args = parser.parse_args()

    push_default_prompts()

    if not args.prompts_only and args.dataset:
        push_corpus(args.dataset)
    elif not args.prompts_only and not args.dataset:
        logger.info(
            "No --dataset supplied — skipping corpus upload. "
            "Re-run with --dataset path/to/dataset_quintets.csv to upload the corpus."
        )


if __name__ == "__main__":
    main()
