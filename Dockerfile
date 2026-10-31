FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 mrmem && useradd --uid 10001 --gid mrmem --create-home mrmem
WORKDIR /app
COPY pyproject.toml README.md requirements-server.txt ./
COPY deploy/constraints.txt ./deploy/constraints.txt
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements-server.txt && python -m pip check
COPY scripts/smoke_offline.py scripts/container_entrypoint.py ./scripts/
COPY deploy/runtime.server.json ./deploy/runtime.server.json
RUN mkdir -p /var/lib/mrmem && chown mrmem:mrmem /var/lib/mrmem
USER mrmem
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3).close()"
ENTRYPOINT ["python", "scripts/container_entrypoint.py"]
