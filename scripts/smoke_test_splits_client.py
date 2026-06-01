"""Smoke test (phoenix-client) for the Split-based run — client-only installs.

Uses the lightweight arize-phoenix-client API (no full arize-phoenix SDK):
    from phoenix.client import Client
    client.datasets.get_dataset(dataset=NAME, splits=[...])
    client.experiments.run_experiment(dataset=, task=, experiment_name=)

The phoenix-client task binds arguments BY NAME: declare `input` and/or
`metadata` and Phoenix passes the example's input/metadata dicts.

Usage:
    python -m scripts.smoke_test_splits_client                    # free check
    python -m scripts.smoke_test_splits_client --live --limit 2   # 2 real runs
    python -m scripts.smoke_test_splits_client --split female       # other split
"""

import argparse
import sys

DATASET_NAME = "yentlbench-quintets-all-variants"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="nb_ambiguous")
    ap.add_argument(
        "--limit", type=int, default=2, help="Max examples to actually run in --live mode."
    )
    ap.add_argument(
        "--live", action="store_true", help="Invoke the real YentlGuardRunner (costs model calls)."
    )
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--budget", default="low")
    ap.add_argument("--threshold", type=float, default=1.0)
    args = ap.parse_args()

    # 1) import check ----------------------------------------------------
    try:
        from phoenix.client import Client
    except Exception as e:
        print(f"FAIL: cannot import phoenix.client ({e}).")
        print("      -> pip install -U 'arize-phoenix-client>=2.7.0'")
        return 1
    client = Client()  # reads PHOENIX_BASE_URL / PHOENIX_API_KEY from env
    if not hasattr(client, "experiments") or not hasattr(client, "datasets"):
        print("FAIL: this arize-phoenix-client lacks datasets/experiments.")
        print("      -> pip install -U 'arize-phoenix-client>=2.7.0'")
        return 1
    print("OK   phoenix.client datasets + experiments available")

    # 2) split loads & has examples --------------------------------------
    try:
        dataset = client.datasets.get_dataset(dataset=DATASET_NAME, splits=[args.split])
    except TypeError as e:
        print(f"FAIL: get_dataset(splits=...) unsupported on this client ({e}).")
        print("      -> pip install -U 'arize-phoenix-client>=2.7.0'")
        return 1
    examples = getattr(dataset, "examples", None) or []
    if not examples:
        print(f"FAIL: split '{args.split}' returned 0 examples.")
        print("      -> create + assign the split in the Phoenix UI first.")
        return 1
    print(f"OK   split '{args.split}' loaded   ({len(examples)} examples)")

    # 3) optional real-runner wiring -------------------------------------
    runner = None
    row_by_id = {}
    if args.live:
        from yentlguard.agent.runner import YentlGuardRunner
        from yentlguard.cli._common import _build_phoenix_components

        prompt_mgr, dataset_mgr, _ = _build_phoenix_components()
        runner = YentlGuardRunner(
            model_version=args.model,
            thinking_budget=args.budget,
            delta_m_threshold=args.threshold,
            baseline_lookup=None,
            prompt_manager=prompt_mgr,
        )
        df_all = dataset_mgr.get_vignettes_df()
        row_by_id = {
            (str(int(r["source_stay_id"])), r.get("demographic_variant", r.get("gender_variant", ""))): r.to_dict()
            for _, r in df_all.iterrows()
        }

    ran = {"n": 0}

    # phoenix-client binds task params BY NAME: input + metadata are the
    # example's input/metadata dicts. (If your client errors on `metadata`,
    # switch the signature to `def task(input, example):` and read
    # example["metadata"].)
    def task(input, metadata):
        if not args.live or ran["n"] >= args.limit:
            return {"smoke": "stub", "variant": args.split}
        ran["n"] += 1
        stay_id = str(int(metadata["source_stay_id"]))
        vignette = row_by_id.get((stay_id, args.split)) if row_by_id else None
        
        text = input.get("vignette_text")
        if not text and vignette:
            text = vignette.get("vignette_text")
        if not text:
            text = input.get("vignette_text", "")

        run = runner.run(
            vignette_id=stay_id,
            vignette_text=text,
            demographic_variant=args.split,
            experiment_id="smoke-test",
        )
        return {
            "pass1_esi": run.pass1_esi,
            "pass2_esi": run.pass2_esi,
            "crr": run.crr.crr if run.crr else None,
        }

    # 4) run_experiment over the split -----------------------------------
    # Keep args minimal for client compatibility; add experiment_metadata /
    # concurrency only if your client version accepts them.
    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        experiment_name=f"SMOKE {args.split} ({'live' if args.live else 'dry'})",
    )

    # RanExperiment may be a dict or an object depending on client version.
    runs = getattr(experiment, "runs", None)
    if runs is None and isinstance(experiment, dict):
        runs = experiment.get("runs") or experiment.get("task_runs")
    n = len(runs) if runs is not None else getattr(experiment, "task_runs_count", "?")
    print(f"OK   run_experiment created experiment   runs={n}")

    print()
    print("Next: open Phoenix -> Experiments and confirm the new experiment")
    print("lists its runs nested underneath (proves task spans are linked).")
    print("Phoenix UI: http://localhost:6006  (or your PHOENIX_BASE_URL)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
