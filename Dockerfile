# Runtime image for the Streamlit RAG app.
# The Ollama LLM runs OUTSIDE this container; point the app at it via
# OLLAMA_BASE_URL (e.g. http://host.docker.internal:11434).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application source (see .dockerignore for what is excluded).
COPY . .

EXPOSE 8501

# Default target for a local Ollama running on the Docker host.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
