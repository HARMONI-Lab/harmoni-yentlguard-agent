import re
with open("yentlguard/mcp/phoenix_manager.py", "r") as f:
    content = f.read()

# Delete PhoenixExperimentRegistry class
content = re.sub(r'class PhoenixExperimentRegistry:.*?$(?=\nclass |\Z)', '', content, flags=re.DOTALL | re.MULTILINE)

with open("yentlguard/mcp/phoenix_manager.py", "w") as f:
    f.write(content)
