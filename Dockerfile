FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS dependencies
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_CACHE_DIR=/opt/uv-cache
RUN python -m pip install --no-cache-dir uv==0.13.9
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/opt/uv-cache uv sync --frozen --no-dev --no-install-project

FROM dependencies AS api-build
COPY src ./src
RUN --mount=type=cache,target=/opt/uv-cache uv sync --frozen --no-dev --no-editable --extra rerank

FROM dependencies AS ui-build
COPY src ./src
RUN --mount=type=cache,target=/opt/uv-cache uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS runtime
RUN groupadd --gid 1000 groundmark \
    && useradd --uid 1000 --gid 1000 --create-home groundmark \
    && mkdir -p /app/.cache /app/.artifacts \
    && chown -R groundmark:groundmark /app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="" HF_HOME=/home/groundmark/.cache/huggingface \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    OMP_NUM_THREADS=4
USER groundmark

FROM runtime AS api
COPY --from=api-build /app/.venv /app/.venv
EXPOSE 8100
HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8100/health', timeout=2)"]
CMD ["uvicorn", "groundmark.api:app", "--host", "0.0.0.0", "--port", "8100"]

FROM runtime AS ui
COPY --from=ui-build /app/.venv /app/.venv
COPY src/groundmark/ui.py ./src/groundmark/ui.py
COPY .streamlit/config.toml ./.streamlit/config.toml
EXPOSE 8601
HEALTHCHECK --interval=5s --timeout=3s --retries=10 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8601/_stcore/health', timeout=2)"]
CMD ["streamlit", "run", "src/groundmark/ui.py", "--server.address", "0.0.0.0", "--server.port", "8601", "--server.headless", "true"]
