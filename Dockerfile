FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync

COPY . .

EXPOSE 8050

CMD ["uv", "run", "python", "app.py"] 