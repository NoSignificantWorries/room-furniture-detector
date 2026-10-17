FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --extra serve --frozen --no-dev

COPY src/ /app/src/

# CMD ["python", "main.py"]
