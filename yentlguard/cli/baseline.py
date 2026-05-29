import argparse
import logging
from ._common import _build_phoenix_components, _get_completed_vignettes

logger = logging.getLogger("yentlguard.cli")

def cmd_baseline(args: argparse.Namespace) -> str:
    """Populate Phoenix with nb_ambiguous baseline spans."""
    import pandas as _pd
    from yentlbench.local_runner.prompt import build_prompt as _build_prompt

    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.eval.bq_writer import BQWriter
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    provider = setup_phoenix_tracing(project_name="yentlguard-runs")

    prompt_mgr, dataset_mgr, expt_registry = _build_phoenix_components()

    # Fetch vignettes from Phoenix dataset - no local CSV fallback
    df_all = dataset_mgr.get_vignettes_df()
    if df_all.empty:
        logger.error(
            "Failed to load vignettes dataset from Phoenix. "
            "Ensure the vignette corpus 'yentlbench-quintets-all-variants' is uploaded to Phoenix first. "
            "Run setup_phoenix.py to upload the dataset."
        )
        raise SystemExit(1)
    
    # Filter for nb_ambiguous variant
    df_variant = df_all[df_all["gender_variant"] == "nb_ambiguous"]
    if df_variant.empty:
        logger.error(
            "No nb_ambiguous vignettes found in Phoenix dataset. "
            "Ensure the corpus contains this variant."
        )
        raise SystemExit(1)
        
    logger.info(
        "Loaded %d nb_ambiguous vignettes from Phoenix dataset", len(df_variant)
    )

    # Use the existing Phoenix dataset ID
    phoenix_dataset_id = dataset_mgr.dataset_id
    
    # Ensure necessary columns exist with proper data types
    df_variant = df_variant.copy()
    df_variant["source_stay_id"] = df_variant["source_stay_id"].astype(str)
    df_variant["esi_ground_truth"] = df_variant["esi_ground_truth"].apply(
        lambda v: str(int(float(v))) if _pd.notna(v) and v != '' else None
    )

    runner = YentlGuardRunner(
        model_version=args.model,
        thinking_budget=args.budget,
        baseline_lookup=None,
        prompt_manager=prompt_mgr,
    )

    # Register the experiment in Phoenix FIRST to obtain the official experiment_id.
    # Phoenix is now a hard dependency.
    experiment_id = expt_registry.register(
        label=f"baseline {args.model} {args.budget}",
        dataset_id=phoenix_dataset_id,
        model_version=args.model,
        thinking_budget=args.budget,
        variants=["nb_ambiguous"],
        vignette_count=len(df_variant),
        notes="Baseline pass for nb_ambiguous",
    )
    logger.info("Baseline registered in Phoenix. experiment_id: %s", experiment_id)

    with BQWriter(
        experiment_id=experiment_id,
        gate_threshold=1.0,
    ) as bq:
        bq.register_experiment(
            label=f"baseline {args.model} {args.budget}",
            models=[args.model],
            thinking_budgets=[args.budget],
            variants=["nb_ambiguous"],
            vignette_count=len(df_variant),
            notes="Baseline pass for nb_ambiguous",
        )

        completed = _get_completed_vignettes(
            args.model, args.budget, "nb_ambiguous"
        )
        if completed:
            logger.info(
                "Skipping %d already completed vignettes.", len(completed)
            )

        for _, row in df_variant.iterrows():
            vignette = row.to_dict()
            vignette_id = str(vignette["source_stay_id"])
            if vignette_id in completed:
                continue
            text = _build_prompt(vignette, "nb_ambiguous")
            # Pass experiment_id so yentlguard.experiment_id is written on every span —
            # required for annotate_spans_with_verdicts to locate spans by experiment_id.
            run = runner.run(
                vignette_id=vignette_id,
                vignette_text=text,
                demographic_variant="nb_ambiguous",
                experiment_id=experiment_id,
            )

            esi_gt = (
                str(int(float(vignette["esi_ground_truth"])))
                if not _pd.isna(vignette.get("esi_ground_truth")) and vignette.get("esi_ground_truth")
                else None
            )
            cat = str(vignette.get("chiefcomplaint", "") or vignette.get("clinical_category", "")) or None
            bq.write(run=run, esi_ground_truth=esi_gt, clinical_category=cat)

            dm = (
                run.pass1_delta_m.delta_m
                if run.pass1_delta_m and run.pass1_delta_m.delta_m
                else None
            )
            status = "✓" if not run.errors else "✗"
            logger.info(
                "%s %s | ESI=%s | ΔM=%.4f",
                status, vignette_id, run.pass1_esi or "?", dm or 0.0,
            )

    logger.info("Baseline complete. experiment_id=%s", experiment_id)
    if provider and not getattr(args, "skip_shutdown", False):
        provider.shutdown()
        
    return experiment_id
