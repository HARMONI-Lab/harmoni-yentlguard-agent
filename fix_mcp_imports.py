with open("yentlguard/mcp/baseline_lookup.py", "r") as f:
    content = f.read()

content = content.replace('from mcp.client.session import ClientSession\n', '')
content = content.replace('# MCP Client imports for future/current usage\nfrom mcp.client.sse import sse_client\n', '')

with open("yentlguard/mcp/baseline_lookup.py", "w") as f:
    f.write(content)
