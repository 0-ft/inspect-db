FROM python:3.12-slim-bookworm AS python-base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

FROM ghcr.io/charmbracelet/vhs
COPY --from=python-base /bin/uv /bin/uvx /bin/
COPY --from=python-base /usr/local/bin/python3.12 /usr/local/bin/python3.12
COPY --from=python-base /usr/local/lib/python3.12 /usr/local/lib/python3.12

RUN apt-get update && apt-get install -y git

RUN uv python install 3.12

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy


WORKDIR /app
COPY uv.lock pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-install-project --no-dev

COPY src /app/src

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

COPY tests /app/tests
COPY demo.tape /app/

# ENTRYPOINT ["/bin/bash"]

ENTRYPOINT ["vhs", "demo.tape", "-o", "/output/demo.gif"]