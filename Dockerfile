# Serving image for the churn API.
#
# Training is intentionally absent from this image: the deployed service loads a
# committed artifact and serves it. That keeps the image small and the memory
# footprint inside Render's free 512 MB.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# cpu-only torch-free requirements: sklearn, xgboost, shap, fastapi
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install -e ".[api]"

# The calibrated artifact ships inside the image (no persistent disk on free tier)
COPY artifacts/model_calibrated/ ./artifacts/model_calibrated/
COPY configs/ ./configs/

# Non-root, and bound to 0.0.0.0 so the platform proxy can reach it.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Single worker: two would exceed 512 MB once SHAP is loaded (measured 362.7 MB).
CMD ["uvicorn", "churn.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
