import argparse
import asyncio
import logging

from phoenix.client import AsyncClient

from ._common import _build_phoenix_components, _extract_experiment_id

logger = logging.getLogger("yentlguard.cli")

DATASET_NAME = "yentlbench-quintets-all-variants"
BASELINE_VARIANT = "nb_ambiguous"
BASELINE_SPLIT = "nb_ambiguous"


def cmd_baseline(args: argparse.Namespace) -> str:
    """Loop-safe sync entrypoint.

    Works whether or not the calling thread already has a running loop:
      - no running loop (plain CLI)  -> asyncio.run() on this thread
      - running loop (ADK run_baseline tool / Jupyter) -> drive on a fresh loop
        in a worker thread, so asyncio.run() is never called inside a live loop
    """
    coro = _cmd_baseline_async(args)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _cmd_baseline_async(args: argparse.Namespace) -> str:
    """nb_ambiguous baseline as a linked Phoenix experiment via AsyncClient.

    Each task awaits runner.arun() directly on the SAME event loop, so there is
    no worker-thread bridge: OTel context is preserved and every span (LLM
    generation + ΔM children) nests under the experiment run.
    """
    import pandas as _pd
    from yentlguard.prompting.prompt import build_prompt as _build_prompt

    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.eval.bq_writer import BQWriter
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    provider = setup_phoenix_tracing(project_name="yentlguard-runs")
    prompt_mgr, dataset_mgr, _ = _build_phoenix_components()
    client = AsyncClient()  # PHOENIX_BASE_URL / PHOENIX_API_KEY from env

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

    split_name = getattr(args, "split", None) or BASELINE_SPLIT
    dataset = await client.datasets.get_dataset(dataset=DATASET_NAME, splits=[split_name])
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
    lock = asyncio.Lock()

    async def task(input, metadata, _runner=runner):
        # Split already restricts to nb_ambiguous; keep a defensive default.
        variant = input.get("demographic_variant", BASELINE_VARIANT)
        stay_id = str(int(metadata["source_stay_id"]))
        vignette = row_by_id.get(stay_id)
        if vignette is None:
            return {"error": f"stay_id {stay_id} not in corpus"}

        text = _build_prompt(vignette, variant)
        # Await arun() directly — same loop, so the runner's spans nest under
        # this task's span (no thread hop, no asyncio.run, no nest_asyncio).
        run = await _runner.arun(
            vignette_id=stay_id,
            vignette_text=text,
            demographic_variant=variant,
        )

        esi_gt = (
            str(int(float(vignette["esi_ground_truth"])))
            if vignette.get("esi_ground_truth") is not None
            and not _pd.isna(vignette.get("esi_ground_truth"))
            else None
        )
        cat = (
            str(vignette.get("chiefcomplaint", "") or vignette.get("clinical_category", ""))
            or None
        )
        async with lock:
            collected.append((run, esi_gt, cat))

        dm = run.pass1_delta_m
        return {
            "pass1_esi": run.pass1_esi,
            "baseline_delta_m": dm.delta_m if dm else None,
            "pass1_top_logprob": dm.top_logprob if dm else None,
            "pass1_runner_up_logprob": dm.runner_up_logprob if dm else None,
            "pass1_runner_up_token": dm.runner_up_token if dm else None,
            "errors": run.errors,
        }

    experiment = await client.experiments.run_experiment(
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
            notes="Baseline pass for nb_ambiguous (async split)",
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