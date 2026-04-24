FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev

COPY stage4_foundry_hosted.py .

# Vendored LangGraph hosting runtime (from langchain-azure PR #501)
COPY _vendor/ /app/_vendor/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8088

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8088/health')" || exit 1

CMD ["python", "-u", "stage4_foundry_hosted.py"]
