import argparse
import logging
import uuid

from phoenix.client import Client

from ._common import _build_phoenix_components

logger = logging.getLogger("yentlguard.cli")

DATASET_NAME = "yentlbench-quintets-all-variants"


def cmd_run(args: argparse.Namespace) -> str:
    """Run two-pass mechanistic experiments, one per (model, budget, variant).

    Each variant is loaded server-side through its dataset Split, so
    run_experiment nests every task span under the experiment (proper
    trace<->experiment linkage) and each variant shows up as its own
    comparable experiment in Phoenix.
    """
    import pandas as _pd
    from yentlbench.local_runner.prompt import build_prompt as _build_prompt

    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.eval.bq_writer import BQWriter
    from yentlguard.mcp.baseline_lookup import BQBackend
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    provider = setup_phoenix_tracing(project_name="yentlguard-runs")

    # expt_registry is intentionally dropped: run_experiment creates and links
    # the experiment for us, so the empty-shell registry is no longer needed.
    prompt_mgr, dataset_mgr, _ = _build_phoenix_components()

    mcp_client = BQBackend(project_name="yentlguard")
    client = Client()

    # Load the full corpus once for prompt-building + ground-truth lookup.
    # run_experiment iterates the Split's examples; we map back to the full row
    # by source_stay_id so _build_prompt receives the format it expects.
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
            dataset = client.datasets.get_dataset(dataset=DATASET_NAME, splits=[variant])            examples = getattr(dataset, "examples", None) or []
            if not examples:
                logger.warning(
                    "Split '%s' returned 0 examples — did you create/assign it "
                    "in the Phoenix UI? Skipping.",
                    variant,
                )
                continue

            label = args.label or f"{args.model} {budget} {variant}"
            bq_experiment_id = uuid.uuid4().hex

            logger.info(
                "Running experiment: %s | %d examples | bq_experiment_id=%s",
                label, len(examples), bq_experiment_id,
            )

            with BQWriter(
                experiment_id=bq_experiment_id,
                gate_threshold=args.threshold,
            ) as bq:
                bq.register_experiment(
                    label=label,
                    models=[args.model],
                    thinking_budgets=[budget],
                    variants=[variant],
                    vignette_count=len(examples),
                    notes=args.notes,
                )

                def task(metadata, _variant=variant, _runner=runner, _bq=bq, _exp_id=bq_experiment_id):
                    stay_id = str(int(metadata["source_stay_id"]))
                    vignette = row_by_id.get(stay_id)
                    if vignette is None:
                        return {"error": f"stay_id {stay_id} not in corpus"}

                    text = _build_prompt(vignette, _variant)
                    esi_gt = (
                        str(int(vignette["acuity"]))
                        if not _pd.isna(vignette.get("acuity"))
                        else None
                    )
                    clinical_cat = str(vignette.get("chiefcomplaint", "")) or None

                    # Pass experiment_id so yentlguard.experiment_id is written on
                    # every span (needed for annotate_spans_with_verdicts + BQ).
                    run = _runner.run(
                        vignette_id=stay_id,
                        vignette_text=text,
                        demographic_variant=_variant,
                        experiment_id=_exp_id,
                    )
                    _bq.write(
                        run=run,
                        esi_ground_truth=esi_gt,
                        clinical_category=clinical_cat,
                    )
                    return {
                        "pass1_esi": run.pass1_esi,
                        "pass2_esi": run.pass2_esi,
                        "crr": run.crr.crr if run.crr else None,
                        "intervention_triggered": run.intervention_triggered,
                        "errors": run.errors,
                    }

                experiment = client.experiments.run_experiment(
                    dataset=dataset, task=task, experiment_name=label,
                )

            exp_id = getattr(experiment, "id", None) or getattr(experiment, "experiment_id", None)
            if exp_id is None and isinstance(experiment, dict):
                exp_id = experiment.get("id") or experiment.get("experiment_id")
            
            experiment_ids.append(str(exp_id))
            logger.info("Finished experiment %s (phoenix id=%s)", label, exp_id)

    # Flush spans before shutdown so every task span reaches Phoenix.
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