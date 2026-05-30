import argparse
import logging
import threading

from phoenix.client import Client

from ._common import _build_phoenix_components, _extract_experiment_id

logger = logging.getLogger("yentlguard.cli")

DATASET_NAME = "yentlbench-quintets-all-variants"


def cmd_run(args: argparse.Namespace) -> str:
    """Run two-pass experiments (one per model/budget/variant) on arize-phoenix
    16.x via the phoenix.client API, with every BigQuery row tagged using the
    SAME experiment id Phoenix assigns.

    The Phoenix id only exists after run_experiment() returns, so results are
    collected during the run and written to BigQuery afterward keyed by that id
    (no separate bq uuid).
    """
    import pandas as _pd
    from yentlbench.local_runner.prompt import build_prompt as _build_prompt

    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.eval.bq_writer import BQWriter
    from yentlguard.mcp.baseline_lookup import BQBackend
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    provider = setup_phoenix_tracing(project_name="yentlguard-runs")

    prompt_mgr, dataset_mgr, _ = _build_phoenix_components()
    mcp_client = BQBackend(project_name="yentlguard")
    client = Client()  # PHOENIX_BASE_URL / PHOENIX_API_KEY from env

    df_all = dataset_mgr.get_vignettes_df()
    if df_all.empty:
        logger.error(
            "Failed to load vignettes dataset from Phoenix. Ensure the corpus "
            "'%s' is uploaded first.",
            DATASET_NAME,
        )
        return ""

    row_by_id = {
        str(int(r["source_stay_id"])): r.to_dict()
        for _, r in df_all.iterrows()
    }

    experiment_ids: list[str] = []

    for budget in args.budget:
        runner = YentlGuardRunner(
            model_version=args.model,
            thinking_budget=budget,
            delta_m_threshold=args.threshold,
            baseline_lookup=mcp_client,
            prompt_manager=prompt_mgr,
        )

        for variant in args.variants:
            # Load ONLY this variant's examples via its Split (server-side).
            dataset = client.datasets.get_dataset(dataset=DATASET_NAME, splits=[variant])
            examples = getattr(dataset, "examples", None) or []
            if not examples:
                logger.warning(
                    "Split '%s' returned 0 examples — create/assign it in the "
                    "Phoenix UI. Skipping.",
                    variant,
                )
                continue

            label = args.label or f"{args.model} {budget} {variant}"

            # Collect during the run; write to BQ afterward with Phoenix's id.
            collected: list[tuple] = []
            lock = threading.Lock()

            def task(input, metadata, _variant=variant, _runner=runner):
                stay_id = str(int(metadata["source_stay_id"]))
                vignette = row_by_id.get(stay_id)
                if vignette is None:
                    return {"error": f"stay_id {stay_id} not in corpus"}

                text = _build_prompt(vignette, _variant)
                run = _runner.run(
                    vignette_id=stay_id,
                    vignette_text=text,
                    demographic_variant=_variant,
                    # No experiment_id here: it doesn't exist yet. Phoenix links
                    # this task's spans to the experiment run natively.
                )
                esi_gt = (
                    str(int(vignette["acuity"]))
                    if not _pd.isna(vignette.get("acuity"))
                    else None
                )
                clinical_cat = str(vignette.get("chiefcomplaint", "")) or None
                with lock:
                    collected.append((run, esi_gt, clinical_cat))
                return {
                    "pass1_esi": run.pass1_esi,
                    "pass2_esi": run.pass2_esi,
                    "crr": run.crr.crr if run.crr else None,
                    "intervention_triggered": run.intervention_triggered,
                    "errors": run.errors,
                }

            logger.info("Running experiment: %s | %d examples", label, len(examples))
            experiment = client.experiments.run_experiment(
                dataset=dataset,
                task=task,
                experiment_name=label,
                concurrency=getattr(args, "concurrency", 4),
            )

            # Phoenix experiment id — now available. Tag BQ with the SAME id.
            phoenix_id = _extract_experiment_id(experiment)
            experiment_ids.append(str(phoenix_id))

            with BQWriter(experiment_id=phoenix_id, gate_threshold=args.threshold) as bq:
                bq.register_experiment(
                    label=label,
                    models=[args.model],
                    thinking_budgets=[budget],
                    variants=[variant],
                    vignette_count=len(collected),
                    notes=args.notes,
                )
                for run, esi_gt, clinical_cat in collected:
                    bq.write(
                        run=run,
                        esi_ground_truth=esi_gt,
                        clinical_category=clinical_cat,
                    )

            logger.info(
                "Finished %s (phoenix id=%s, %d BQ rows)",
                label, phoenix_id, len(collected),
            )

    if provider is not None and hasattr(provider, "force_flush"):
        try:
            provider.force_flush(timeout_millis=10000)
        except Exception as flush_err:
            logger.debug("force_flush warning (non-fatal): %s", flush_err)
    if provider and not getattr(args, "skip_shutdown", False):
        provider.shutdown()

    logger.info(
        "Run complete. %d experiment(s): %s",
        len(experiment_ids), ", ".join(experiment_ids),
    )
    return ",".join(experiment_ids)