FROM python:3.13-slim

WORKDIR /app

# システム依存関係のインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 依存関係を先にインストール（レイヤーキャッシュ活用）
COPY pyproject.toml README.md ./
RUN uv pip install --system -e ".[dev]"

# ソースコードとデータをコピー
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/

CMD ["bash"]
