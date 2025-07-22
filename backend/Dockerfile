# 最新のセキュリティパッチが適用されたベースイメージの指定
FROM python:3.10-slim AS base

# ビルド引数
ARG PIP_CACHE_DIR=/root/.cache/pip
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.5.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_CACHE_DIR=${PIP_CACHE_DIR} \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

# ビルド依存関係のインストール
FROM base AS builder

# システム依存関係のインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Poetryのインストール
RUN pip install --no-cache-dir poetry==$POETRY_VERSION

# 依存関係をインストール
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    # 不要なキャッシュを削除
    find /usr/local -type f -name '*.pyc' -delete && \
    find /usr/local -type d -name '__pycache__' -exec rm -rf {} +

# モデルダウンロード用のステージ
FROM base AS model-downloader

# システム依存関係のインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 必要なPythonパッケージをインストール
COPY --from=builder /usr/local /usr/local
# 必要なパッケージのみをインストールしてビルドサイズを削減
RUN pip install --no-cache-dir torch transformers sentence-transformers

# モデルをダウンロード
RUN mkdir -p /app/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface/hub

# 必要なファイルをコピー
COPY download_models.py models.yaml strategies.yaml /app/

# 必要なモデルのみをダウンロード（実際に使用するモデルのみに絞る）
RUN python /app/download_models.py

# 最終的なイメージ
FROM base

# システム依存関係のインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# アプリケーションディレクトリの作成
WORKDIR /app

# 依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# モデルをコピー
COPY --from=model-downloader /app/.cache /app/.cache

# NLTKリソースをダウンロード
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt

# アプリケーションコードをコピー
COPY app/ /app/app/
COPY models.yaml /app/models.yaml
COPY strategies.yaml /app/strategies.yaml

# ログとアップロード用ディレクトリの作成
RUN mkdir -p /app/logs /app/uploads

# ヘルスチェック用のスクリプト
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# アプリケーションの実行
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
