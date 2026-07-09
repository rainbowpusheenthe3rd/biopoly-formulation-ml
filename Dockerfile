# Self-contained image: installs from the uv lockfile, then generates the synthetic
# dataset and trains the forward model so the API serves a live model on `up`.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    BIOPOLY_TRACKING_BACKEND=noop

# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer (cached unless the lockfile changes)
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Bake data + a trained champion model into the image
RUN uv run biopoly-generate && uv run biopoly-train

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "biopoly.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
