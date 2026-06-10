FROM python:3.11-slim
WORKDIR /app

ARG PHOENIX_MCP_VERSION=4.0.13   
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @arizeai/phoenix-mcp@${PHOENIX_MCP_VERSION} && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir .[ui]

ENV HOST=0.0.0.0
ENV PORT=8080

EXPOSE 8080
WORKDIR /app/yentlguard/yentlguard_ui

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8080", "-h"]
