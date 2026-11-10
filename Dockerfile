FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/usr/local

ENV MODEL_BACKEND=onnx
ENV MODEL_ONNX_PATH=/app/models/furniture_detector.onnx
ENV MODEL_HOST="0.0.0.0"
ENV MODEL_PORT="8001"
ENV MODEL_WORKERS_COUNT=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra serve --frozen --no-dev --no-install-project

COPY models/furniture_detector.onnx /app/models/
COPY src/ /app/src/

RUN uv sync --extra serve --frozen --no-dev

EXPOSE 8001

CMD ["sh", "-c", "python -m uvicorn room_furniture_detector.api:app --host ${MODEL_HOST} --port ${MODEL_PORT} --workers ${MODEL_WORKERS_COUNT}"]
