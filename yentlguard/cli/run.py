import argparse
import asyncio
import logging

from phoenix.client import AsyncClient

from ._common import _build_phoenix_components, _extract_experiment_id

logger = logging.getLogger("yentlguard.cli")

DATASET_NAME = "yentlbench-quintets-all-variants"


def cmd_run(args: argparse.Namespace) -> str:
    """
    Execute YentlGuard mechanistic runs with loop-safe async handling.
    
    Works whether or not the calling thread already has a running event loop:
      - No running loop (plain CLI) -> asyncio.run() on this thread
      - Running loop (ADK run_experiment tool / Jupyter) -> drive on a fresh
        loop in a worker thread, so asyncio.run() is never called in a live loop
        
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments containing run configuration.
        
    Returns
    -------
    str
        Comma-separated list of experiment IDs that were executed.
    """
    coro = _cmd_run_async(args)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _cmd_run_async(args: argparse.Namespace) -> str:
    import pandas as _pd

    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.eval.bq_writer import BQWriter
    from yentlguard.mcp.baseline_lookup import BQBackend
    from yentlguard.prompting.prompt import build_prompt as _build_prompt
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    provider = setup_phoenix_tracing(project_name="yentlguard-runs")
    prompt_mgr, dataset_mgr = _build_phoenix_components()
    mcp_client = BQBackend(project_name="yentlguard")
    client = AsyncClient()

    df_all = dataset_mgr.get_vignettes_df()
    if df_all.empty:
        logger.error(
            "Failed to load vignettes dataset from Phoenix. Ensure the corpus "
            "'%s' is uploaded first.",
            DATASET_NAME,
        )
        return ""

    row_by_id = {str(int(r["source_stay_id"])): r.to_dict() for _, r in df_all.iterrows()}

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
            dataset = await client.datasets.get_dataset(dataset=DATASET_NAME, splits=[variant])
            examples = getattr(dataset, "examples", None) or []
            if not examples:
                logger.warning(
                    "Split '%s' returned 0 examples — create/assign it in the "
                    "Phoenix UI. Skipping.",
                    variant,
                )
                continue

            label = args.label or f"{args.model} {budget} {variant}"
            collected: list[tuple] = []
            lock = asyncio.Lock()

            # Bind variant + runner as defaults so each task closes over its own.
            async def task(input, metadata, _variant=variant, _runner=runner):
                stay_id = str(int(metadata["source_stay_id"]))
                vignette = row_by_id.get(stay_id)
                if vignette is None:
                    return {"error": f"stay_id {stay_id} not in corpus"}

                text = _build_prompt(vignette, _variant)
                run = await _runner.arun(
                    vignette_id=stay_id,
                    vignette_text=text,
                    demographic_variant=_variant,
                )
                esi_gt = (
                    str(int(vignette["acuity"]))
                    if vignette.get("acuity") is not None and not _pd.isna(vignette.get("acuity"))
                    else None
                )
                clinical_cat = str(vignette.get("chiefcomplaint", "")) or None
                async with lock:
                    collected.append((run, esi_gt, clinical_cat))

                dm = run.pass1_delta_m
                return {
                    "pass1_esi": run.pass1_esi,
                    "pass2_esi": run.pass2_esi,
                    "crr": run.crr.crr if run.crr else None,
                    "intervention_triggered": run.intervention_triggered,
                    "pass1_delta_m": dm.delta_m if dm else None,
                    "pass1_top_logprob": dm.top_logprob if dm else None,
                    "pass1_runner_up_logprob": dm.runner_up_logprob if dm else None,
                    "errors": run.errors,
                }

            logger.info("Running experiment: %s | %d examples", label, len(examples))
            experiment = await client.experiments.run_experiment(
                dataset=dataset,
                task=task,
                experiment_name=label,
                concurrency=getattr(args, "concurrency", 4),
            )

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
                    bq.write(run=run, esi_ground_truth=esi_gt, clinical_category=clinical_cat)

            logger.info(
                "Finished %s (phoenix id=%s, %d BQ rows)",
                label,
                phoenix_id,
                len(collected),
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
        len(experiment_ids),
        ", ".join(experiment_ids),
    )
    return ",".join(experiment_ids)
