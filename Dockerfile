# ============================================================
# Multi-Agent Orchestration Platform - Dockerfile
# Single image used by both `api` and `celery` services.
# The entrypoint is chosen via docker-compose `command:`.
# ============================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install uv via the official installer (fast, deterministic dep management).
# uv is used both to install dependencies and (optionally) to run the app.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first (better layer caching).
# Copy only manifest files so code changes don't bust the dep layer.
COPY pyproject.toml ./
COPY app/__init__.py app/__init__.py

# `uv sync` creates a .venv and installs all deps from pyproject.toml.
# Using --frozen would require uv.lock; we don't freeze here to keep the
# example frictionless. For production, commit uv.lock and use --frozen.
RUN uv sync || uv pip install --system .

# Copy the rest of the application code.
COPY app/ app/
COPY .env.example .env.example

# Default command = run the API. Override in compose for the celery service.
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]