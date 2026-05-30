import re

with open("yentlguard/cli/_common.py", "r") as f:
    content = f.read()

# Remove _get_completed_vignettes
content = re.sub(r'def _get_completed_vignettes.*?(?=\ndef _build_phoenix_components)', '', content, flags=re.DOTALL)

# Modify _build_phoenix_components
content = re.sub(r'PhoenixDatasetManager,\n\s+PhoenixExperimentRegistry,\n\s+PhoenixPromptManager,', 'PhoenixDatasetManager,\n        PhoenixPromptManager,', content)
content = re.sub(r'Instantiate PhoenixPromptManager, PhoenixDatasetManager, and\n\s+PhoenixExperimentRegistry\. Returns \(prompt_mgr, dataset_mgr, expt_registry\)\.\n\n\s+NOTE: cmd_run \(Option A\) no longer uses expt_registry, but cmd_baseline\n\s+still calls expt_registry\.register\(\.\.\.\)\. So this keeps returning the\n\s+3-tuple; run\.py simply ignores the third element\. All three degrade\n\s+gracefully when Phoenix is unreachable\.', 'Instantiate PhoenixPromptManager and PhoenixDatasetManager. Returns (prompt_mgr, dataset_mgr).', content)
content = re.sub(r'expt_registry = PhoenixExperimentRegistry\(base_url=base_url, api_key=api_key\)\n\n\s+return prompt_mgr, dataset_mgr, expt_registry', 'return prompt_mgr, dataset_mgr', content)

with open("yentlguard/cli/_common.py", "w") as f:
    f.write(content)

with open("yentlguard/cli/run.py", "r") as f:
    run_content = f.read()
run_content = run_content.replace('prompt_mgr, dataset_mgr, _ = _build_phoenix_components()', 'prompt_mgr, dataset_mgr = _build_phoenix_components()')
with open("yentlguard/cli/run.py", "w") as f:
    f.write(run_content)

with open("yentlguard/cli/baseline.py", "r") as f:
    base_content = f.read()
base_content = base_content.replace('prompt_mgr, dataset_mgr, _ = _build_phoenix_components()', 'prompt_mgr, dataset_mgr = _build_phoenix_components()')
with open("yentlguard/cli/baseline.py", "w") as f:
    f.write(base_content)

