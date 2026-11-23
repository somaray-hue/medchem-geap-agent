# Production container for MedChem GEAP Enterprise Agent
# Runs as a non-root user with RDKit + Google ADK + OpenTelemetry instrumentation.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# System libraries required by RDKit rendering and SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libexpat1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged runtime user
RUN groupadd --gid 10001 agentuser \
    && useradd --uid 10001 --gid agentuser --shell /bin/bash --create-home agentuser

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy agent source, DeepMind cheminformatics skills, and eval manifests
COPY medchem_agent/ ./medchem_agent/
COPY eval/ ./eval/

RUN mkdir -p /app/data && chown -R agentuser:agentuser /app

USER agentuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1

CMD ["uvicorn", "medchem_agent.main:fastapi_app", "--host", "0.0.0.0", "--port", "8080"]
