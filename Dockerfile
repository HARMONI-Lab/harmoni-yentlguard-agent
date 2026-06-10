FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @arizeai/phoenix-mcp@latest && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir .[ui]

ENV HOST=0.0.0.0
ENV PORT=8080
# Target the writable in-memory directory for MCP initialization
ENV NPM_CONFIG_CACHE=/tmp/.npm 

EXPOSE 8080
WORKDIR /app/yentlguard/yentlguard_ui

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8080", "-h"]
