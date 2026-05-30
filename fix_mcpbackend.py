import re
with open("yentlguard/mcp/baseline_lookup.py", "r") as f:
    content = f.read()

# Delete MCPBackend class
content = re.sub(r'class MCPBackend\(BaselineLookup\):.*?$(?=\nclass |\Z)', '', content, flags=re.DOTALL | re.MULTILINE)

with open("yentlguard/mcp/baseline_lookup.py", "w") as f:
    f.write(content)
