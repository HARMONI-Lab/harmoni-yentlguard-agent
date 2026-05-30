import argparse
import logging
import uuid

from ._common import _build_phoenix_components

logger = logging.getLogger("yentlguard.cli")

DATASET_NAME = "yentlbench-quintets-all-variants"
BASELINE_VARIANT = "nb_ambiguous"
BASELINE_SPLIT = "nb_ambiguous"  # create + assign this split in the Phoenix UI first


def cmd_baseline(args: argparse.Namespace) -> str:
    """Run the nb_ambiguous baseline as a linked Phoenix experiment over a Split.

    Uses a dataset Split so the experiment runs over ONLY the nb_ambiguous
    examples — no empty runs from skipped variants.
    """
    import pandas as _pd

    from phoenix.client import Client

    from yentlbench.local_runner.prompt import build_prompt as _build_prompt
    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.eval.bq_writer import BQWriter
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    provider = setup_phoenix_tracing(project_name="yentlguard-runs")

    # Registry unused here; ignore the third element of the 3-tuple.
    prompt_mgr, dataset_mgr, _ = _build_phoenix_components()

    df_all = dataset_mgr.get_vignettes_df()
    if df_all.empty:
        logger.error(
            "Failed to load vignettes dataset from Phoenix. Ensure the corpus "
            "'%s' is uploaded first (run setup_phoenix.py).",
            DATASET_NAME,
        )
        raise SystemExit(1)

    row_by_id = {
        str(int(r["source_stay_id"])): r.to_dict()
        for _, r in df_all.iterrows()
    }

    # Load ONLY the examples assigned to the split. Requires arize-phoenix
    # >= 12.7.0 and a split named BASELINE_SPLIT created in the UI.
    split_name = getattr(args, "split", None) or BASELINE_SPLIT
    client = Client()
    dataset = client.datasets.get_dataset(dataset=DATASET_NAME, splits=[split_name])
    examples = getattr(dataset, "examples", None) or []
    if len(examples) == 0:
        logger.error(
            "Split '%s' has no examples. Create it in the Phoenix UI (dataset "
            "page -> Splits) and assign the nb_ambiguous examples to it.",
            split_name,
        )
        raise SystemExit(1)
    logger.info("Loaded %d examples from split '%s'.", len(examples), split_name)

    runner = YentlGuardRunner(
        model_version=args.model,
        thinking_budget=args.budget,
        baseline_lookup=None,
        prompt_manager=prompt_mgr,
    )

    bq_experiment_id = uuid.uuid4().hex
    label = f"baseline {args.model} {args.budget}"

    with BQWriter(experiment_id=bq_experiment_id, gate_threshold=1.0) as bq:
        bq.register_experiment(
            label=label,
            models=[args.model],
            thinking_budgets=[args.budget],
            variants=[BASELINE_VARIANT],
            vignette_count=len(examples),
            notes="Baseline pass for nb_ambiguous (split)",
        )


        def task(input, metadata, _runner=runner, _bq=bq, _bq_id=bq_experiment_id):
            inp = input or {}
            meta = metadata or {}

            # Split already restricts to nb_ambiguous; keep a defensive default.
            variant = inp.get("demographic_variant", BASELINE_VARIANT)

            sid = str(int(meta["source_stay_id"]))
            vignette = row_by_id.get(sid)
            if vignette is None:
                return None

            text = _build_prompt(vignette, variant)

            run = _runner.run(
                vignette_id=sid,
                vignette_text=text,
                demographic_variant=variant,
                experiment_id=_bq_id,
            )

            esi_gt = (
                str(int(float(vignette["esi_ground_truth"])))
                if not _pd.isna(vignette.get("esi_ground_truth")) and vignette.get("esi_ground_truth")
                else None
            )
            cat = str(vignette.get("chiefcomplaint", "") or vignette.get("clinical_category", "")) or None
            _bq.write(run=run, esi_ground_truth=esi_gt, clinical_category=cat)

            dm = (
                run.pass1_delta_m.delta_m
                if run.pass1_delta_m and run.pass1_delta_m.delta_m
                else None
            )
            status = "✓" if not run.errors else "✗"
            logger.info("%s %s | ESI=%s | ΔM=%.4f", status, sid, run.pass1_esi or "?", dm or 0.0)

            return {
                "pass1_esi": run.pass1_esi,
                "baseline_delta_m": dm,
                "errors": run.errors,
            }


        experiment = client.experiments.run_experiment(
            dataset=dataset, task=task, experiment_name=label,
            experiment_metadata={
                "model": args.model,
                "thinking_budget": args.budget,
                "variants": [BASELINE_VARIANT],
                "split": split_name,
                "bq_experiment_id": bq_experiment_id,
                "notes": "Baseline pass for nb_ambiguous (split)",
            },
            concurrency=getattr(args, "concurrency", 4),
        )

    phoenix_experiment_id = (
        getattr(experiment, "id", None)
        or getattr(experiment, "experiment_id", None)
    )
    logger.info(
        "Baseline complete. phoenix_id=%s bq_id=%s split=%s",
        phoenix_experiment_id, bq_experiment_id, split_name,
    )

    if provider is not None and not getattr(args, "skip_shutdown", False):
        try:
            provider.force_flush(timeout_millis=10000)
        except Exception as e:
            logger.debug("force_flush warning (non-fatal): %s", e)
        provider.shutdown()

    return phoenix_experiment_id or bq_experiment_id