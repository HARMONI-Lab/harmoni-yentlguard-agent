import argparse
import logging
import threading

from phoenix.client import Client

from ._common import _build_phoenix_components, _extract_experiment_id

logger = logging.getLogger("yentlguard.cli")

DATASET_NAME = "yentlbench-quintets-all-variants"
BASELINE_VARIANT = "nb_ambiguous"
BASELINE_SPLIT = "nb_ambiguous"  # create + assign in the Phoenix UI first


def cmd_baseline(args: argparse.Namespace) -> str:
    """Run the nb_ambiguous baseline as a linked Phoenix experiment (client API),
    writing BigQuery rows under the SAME experiment id Phoenix assigns.
    """
    import pandas as _pd
    from yentlbench.local_runner.prompt import build_prompt as _build_prompt

    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.eval.bq_writer import BQWriter
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    provider = setup_phoenix_tracing(project_name="yentlguard-runs")
    prompt_mgr, dataset_mgr, _ = _build_phoenix_components()
    client = Client()

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

    # Load ONLY the split's examples (no empty runs from other variants).
    split_name = getattr(args, "split", None) or BASELINE_SPLIT
    dataset = client.datasets.get_dataset(dataset=DATASET_NAME, splits=[split_name])
    examples = getattr(dataset, "examples", None) or []
    if not examples:
        logger.error(
            "Split '%s' has no examples. Create + assign it in the Phoenix UI "
            "(dataset page -> Splits) first.",
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

    label = f"baseline {args.model} {args.budget}"
    collected: list[tuple] = []
    lock = threading.Lock()

    def task(input, metadata, _runner=runner):
        # Split already restricts to nb_ambiguous; keep a defensive default.
        variant = input.get("demographic_variant", BASELINE_VARIANT)
        stay_id = str(int(metadata["source_stay_id"]))
        vignette = row_by_id.get(stay_id)
        if vignette is None:
            return {"error": f"stay_id {stay_id} not in corpus"}

        text = _build_prompt(vignette, variant)
        run = _runner.run(
            vignette_id=stay_id,
            vignette_text=text,
            demographic_variant=variant,
            # No experiment_id: assigned by Phoenix after the run; spans link natively.
        )
        esi_gt = (
            str(int(float(vignette["esi_ground_truth"])))
            if not _pd.isna(vignette.get("esi_ground_truth")) and vignette.get("esi_ground_truth")
            else None
        )
        cat = str(vignette.get("chiefcomplaint", "") or vignette.get("clinical_category", "")) or None
        dm = (
            run.pass1_delta_m.delta_m
            if run.pass1_delta_m and run.pass1_delta_m.delta_m
            else None
        )
        with lock:
            collected.append((run, esi_gt, cat))
        status = "✓" if not run.errors else "✗"
        logger.info("%s %s | ESI=%s | ΔM=%.4f", status, stay_id, run.pass1_esi or "?", dm or 0.0)
        return {
            "pass1_esi": run.pass1_esi,
            "baseline_delta_m": dm,
            "errors": run.errors,
        }

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        experiment_name=label,
        concurrency=getattr(args, "concurrency", 4),
    )

    phoenix_id = _extract_experiment_id(experiment)

    with BQWriter(experiment_id=phoenix_id, gate_threshold=1.0) as bq:
        bq.register_experiment(
            label=label,
            models=[args.model],
            thinking_budgets=[args.budget],
            variants=[BASELINE_VARIANT],
            vignette_count=len(collected),
            notes="Baseline pass for nb_ambiguous (split)",
        )
        for run, esi_gt, cat in collected:
            bq.write(run=run, esi_ground_truth=esi_gt, clinical_category=cat)

    logger.info(
        "Baseline complete. phoenix_id=%s split=%s (%d BQ rows)",
        phoenix_id, split_name, len(collected),
    )

    if provider is not None and not getattr(args, "skip_shutdown", False):
        try:
            provider.force_flush(timeout_millis=10000)
        except Exception as e:
            logger.debug("force_flush warning (non-fatal): %s", e)
        provider.shutdown()

    return str(phoenix_id)