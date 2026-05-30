import re
with open("yentlguard/agent/yentlguard_agent/tools/runner_tools.py", "r") as f:
    content = f.read()

content = content.replace("import asyncio\n", "")
content = re.sub(r'# Register as an ADK tool.*?triage_vignette_tool = FunctionTool\(triage_vignette\)\n?', '', content, flags=re.DOTALL)

with open("yentlguard/agent/yentlguard_agent/tools/runner_tools.py", "w") as f:
    f.write(content)
