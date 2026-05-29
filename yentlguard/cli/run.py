import argparse
import logging
from ._common import _build_phoenix_components, _get_completed_vignettes

logger = logging.getLogger("yentlguard.cli")

def cmd_run(args: argparse.Namespace) -> str:
    """Execute two-pass mechanistic runs for specified variants."""
    import pathlib as _pathlib
    import asyncio

    import pandas as _pd
    from yentlbench.local_runner.prompt import build_prompt as _build_prompt

    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.eval.bq_writer import BQWriter
    from yentlguard.mcp.baseline_lookup import BQBackend
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    provider = setup_phoenix_tracing()

    prompt_mgr, dataset_mgr, expt_registry = _build_phoenix_components()

    mcp_client = BQBackend(project_name="yentlguard")

    # Fetch dataset metadata to resolve undefined variables
    df_all = dataset_mgr.get_vignettes_df()
    phoenix_dataset_id = dataset_mgr.dataset_id
    if df_all.empty:
        logger.error("Failed to load vignettes dataset.")
        return ""
    n_per_variant = len(df_all) // len(df_all["gender_variant"].unique())

    # Register the experiment in Phoenix FIRST to obtain the official experiment_id.
    # Phoenix is now a hard dependency.
    experiment_id = expt_registry.register(
        label=(
            args.label
            or f"{args.model} {','.join(args.budget)} {','.join(args.variants)}"
        ),
        dataset_id=phoenix_dataset_id,
        model_version=args.model,
        thinking_budget=args.budget[0] if args.budget else None,
        variants=args.variants,
        vignette_count=(
            n_per_variant * len(args.variants) * len(args.budget)
        ),
        notes=args.notes,
    )
    logger.info("Experiment registered in Phoenix. experiment_id: %s", experiment_id)

    with BQWriter(
        experiment_id=experiment_id,
        gate_threshold=args.threshold,
    ) as bq:
        bq.register_experiment(
            label=(
                args.label
                or f"{args.model} {','.join(args.budget)} {','.join(args.variants)}"
            ),
            models=[args.model],
            thinking_budgets=args.budget,
            variants=args.variants,
            vignette_count=(
                n_per_variant * len(args.variants) * len(args.budget)
            ),
            notes=args.notes,
        )

        for budget in args.budget:
            runner = YentlGuardRunner(
                model_version=args.model,
                thinking_budget=budget,
                delta_m_threshold=args.threshold,
                baseline_lookup=mcp_client,
                prompt_manager=prompt_mgr,
            )

            for variant in args.variants:
                vignettes_df = df_all[df_all["gender_variant"] == variant]
                completed = _get_completed_vignettes(args.model, budget, variant)

                if completed:
                    vignettes_df = vignettes_df[
                        ~vignettes_df["source_stay_id"]
                        .astype(int)
                        .astype(str)
                        .isin(completed)
                    ]
                    logger.info(
                        "Skipped %d already completed vignettes.", len(completed)
                    )

                if vignettes_df.empty:
                    logger.info(
                        "All vignettes done for model=%s budget=%s variant=%s.",
                        args.model, budget, variant,
                    )
                    continue

                logger.info(
                    "Running %d vignettes | model=%s | budget=%s | variant=%s",
                    len(vignettes_df), args.model, budget, variant,
                )

                async def _process_variant(
                    _vignettes_df=vignettes_df,
                    _variant=variant,
                    _runner=runner,
                    _exp_id=experiment_id,
                ):
                    sem = asyncio.Semaphore(4)

                    async def process_row(row):
                        async with sem:
                            vignette = row.to_dict()
                            vignette_id = str(int(vignette["source_stay_id"]))
                            text = _build_prompt(vignette, _variant)
                            esi_gt = (
                                str(int(vignette["acuity"]))
                                if not _pd.isna(vignette.get("acuity"))
                                else None
                            )
                            clinical_cat = (
                                str(vignette.get("chiefcomplaint", "")) or None
                            )
                            # Pass experiment_id so yentlguard.experiment_id is written on
                            # every span — required for annotate_spans_with_verdicts.
                            run = await asyncio.to_thread(
                                _runner.run,
                                vignette_id=vignette_id,
                                vignette_text=text,
                                demographic_variant=_variant,
                                experiment_id=_exp_id,
                            )
                            bq.write(
                                run=run,
                                esi_ground_truth=esi_gt,
                                clinical_category=clinical_cat,
                            )
                            if run.crr:
                                dist_crrs = [
                                    r.crr
                                    for r in [
                                        run.crr_distractor_a,
                                        run.crr_distractor_b,
                                        run.crr_distractor_c,
                                    ]
                                    if r is not None
                                ]
                                max_dist = max(dist_crrs) if dist_crrs else None
                                gap_str = (
                                    f" | gap={run.crr.crr - max_dist:.3f}"
                                    if max_dist is not None
                                    else ""
                                )
                                logger.info(
                                    "  %s | CRR=%.3f%s | ESI %s→%s | triggered=%s",
                                    vignette_id,
                                    run.crr.crr,
                                    gap_str,
                                    run.pass1_esi,
                                    run.pass2_esi,
                                    run.intervention_triggered,
                                )

                    tasks = [process_row(row) for _, row in _vignettes_df.iterrows()]
                    await asyncio.gather(*tasks)

                asyncio.run(_process_variant())

    logger.info(
        "Run complete. experiment_id=%s  "
        "Query: SELECT * FROM `%s` WHERE experiment_id = '%s'",
        experiment_id, "runs", experiment_id,
    )
    if provider and not getattr(args, "skip_shutdown", False):
        provider.shutdown()
        
    return experiment_id
