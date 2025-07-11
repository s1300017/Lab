# 最新のセキュリティパッチが適用されたベースイメージを使用
FROM python:3.10-slim-bookworm

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .

RUN apt-get update && apt-get install -y libpq-dev gcc chromium \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# NLTK punktリソースを必ずDL
# NLTK壊れリソースを全ディレクトリから削除
RUN rm -rf /usr/local/share/nltk_data/tokenizers/punkt* /usr/local/lib/nltk_data/tokenizers/punkt* /root/nltk_data/tokenizers/punkt* /usr/share/nltk_data/tokenizers/punkt*
# punktを確実に再ダウンロード
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt

# モデルディレクトリを作成
RUN mkdir -p /app/models

# アプリケーションコードをコピー
COPY ./app /app

# 環境変数を設定
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# モデルファイルをコピー（存在する場合のみ）
RUN if [ -d "/app/models" ]; then \
        echo "Models directory exists"; \
    else \
        echo "Models directory not found"; \
        mkdir -p /app/models; \
    fi

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
