# Inspection API on a CPU: the exported OpenVINO model, no torch and no anomalib.
# The model (exports/) and the thresholds and scores (results/) are mounted at run time,
# see compose.yaml; nothing trained is baked into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
COPY configs ./configs
COPY scripts ./scripts
# Last, so a new commit does not invalidate the dependency layer.
ARG VQI_GIT_COMMIT=unknown
ENV VQI_GIT_COMMIT=${VQI_GIT_COMMIT}

EXPOSE 8000
CMD ["uvicorn", "--factory", "vqi.api:app_from_env", "--host", "0.0.0.0", "--port", "8000"]
