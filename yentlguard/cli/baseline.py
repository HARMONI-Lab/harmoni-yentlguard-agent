import argparse
import logging
from ._common import _build_phoenix_components, _get_completed_vignettes

logger = logging.getLogger("yentlguard.cli")

def cmd_baseline(args: argparse.Namespace) -> str:
    """Populate Phoenix with nb_ambiguous baseline spans."""
    import pathlib as _pathlib

    import pandas as _pd
    from yentlbench.local_runner.prompt import build_prompt as _build_prompt

    from yentlguard.agent.runner import YentlGuardRunner
    from yentlguard.eval.bq_writer import BQWriter
    from yentlguard.telemetry.phoenix import setup_phoenix_tracing

    provider = setup_phoenix_tracing()

    prompt_mgr, dataset_mgr, expt_registry = _build_phoenix_components()

    dataset_path = _pathlib.Path(args.dataset)
    if not dataset_path.exists():
        logger.error(
            "Dataset not found: %s\n"
            "Run: yentlbench prepare  (requires MIMIC-IV-ED data)",
            dataset_path,
        )
        raise SystemExit(1)

    df = _pd.read_csv(dataset_path)
    df = df[df["acuity"].notna()]
    df_variant = df[df["gender_variant"] == "nb_ambiguous"]
    logger.info(
        "Loaded %d nb_ambiguous vignettes from %s", len(df_variant), dataset_path
    )

    corpus_df = df_variant.copy()
    corpus_df["vignette_text"] = corpus_df.apply(
        lambda r: _build_prompt(r.to_dict(), "nb_ambiguous"), axis=1
    )
    corpus_df["esi_ground_truth"] = corpus_df["acuity"].apply(
        lambda v: str(int(v)) if _pd.notna(v) else None
    )
    corpus_df["clinical_category"] = corpus_df.get(
        "chiefcomplaint", _pd.Series(dtype=str)
    ).fillna("")
    corpus_df["source_stay_id"] = corpus_df["source_stay_id"].astype(str)
    corpus_df["demographic_variant"] = "nb_ambiguous"

    phoenix_dataset_id = dataset_mgr.push_vignette_corpus(
        df=corpus_df[[
            "source_stay_id",
            "vignette_text",
            "demographic_variant",
            "clinical_category",
            "esi_ground_truth",
        ]],
        dataset_name=f"yentlbench-nb-ambiguous-{args.model}-{args.budget}",
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
            vignette_id = str(int(vignette["source_stay_id"]))
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
                str(int(vignette["acuity"]))
                if not _pd.isna(vignette.get("acuity"))
                else None
            )
            cat = str(vignette.get("chiefcomplaint", "")) or None
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
    if provider:
        provider.shutdown()
        
    return experiment_id
