FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .[ui]
ENV HOST=0.0.0.0
ENV PORT=8080
EXPOSE 8080
CMD ["chainlit", "run", "yentlguard/yentlguard_ui/app.py", "--host", "0.0.0.0", "--port", "8080", "-h"]
