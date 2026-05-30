import re
with open("yentlguard/eval/agent_builder.py", "r") as f:
    content = f.read()

# We also want to remove PSS_QUERY and THINKING_BUDGET_QUERY constants
content = re.sub(r'PSS_QUERY = """.*?"""\.strip\(\)', '', content, flags=re.DOTALL)
content = re.sub(r'THINKING_BUDGET_QUERY = """.*?"""\.strip\(\)', '', content, flags=re.DOTALL)

# Delete methods
content = re.sub(r'    def compute_pss_summary\(self, experiment_ids: list\[str\]\) -> pd\.DataFrame:.*?    def register_eval_task', '    def register_eval_task', content, flags=re.DOTALL)
content = re.sub(r'    def compare_model_generations\(.*?return pivot\n', '', content, flags=re.DOTALL)

# Also fix eval_task unused variable assignment
content = re.sub(r'eval_task = VAIEvalTask\(', 'VAIEvalTask(', content)

with open("yentlguard/eval/agent_builder.py", "w") as f:
    f.write(content)
