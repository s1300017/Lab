# 最新Ollama（1.0.0以降）をビルドするためのDockerfile
# 必要に応じてバージョンは適宜修正してください
FROM golang:1.24 as builder

# gitをインストール
RUN apt-get update && apt-get install -y git

# Ollamaリポジトリをclone
WORKDIR /build
RUN git clone https://github.com/jmorganca/ollama.git
WORKDIR /build/ollama

# 最新リリースのタグをcheckout（例: v1.21.0）
RUN git fetch --tags && git checkout $(git describe --tags $(git rev-list --tags --max-count=1))


# Goで直接バイナリをビルドし、バージョン情報も埋め込む
# バージョン情報を正しく埋め込むため、exportではなく同一RUN行で変数を使う
# mageではなくgo buildで明示的にビルド
WORKDIR /build/ollama
RUN OLLAMA_VERSION=$(git describe --tags $(git rev-list --tags --max-count=1)) && \
    sed -i "s/var Version string = \".*\"/var Version string = \"${OLLAMA_VERSION}\"/" version/version.go && \
    go build -o ollama . && \
    echo "ビルド後バージョン確認:" && ./ollama --version

# ランタイム用イメージ
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y ca-certificates curl

WORKDIR /app

# builderからバイナリをコピー
COPY --from=builder /build/ollama/ollama /usr/local/bin/ollama

# モデル保存用ディレクトリ
RUN mkdir -p /root/.ollama

EXPOSE 11434

ENTRYPOINT ["ollama"]
CMD ["serve"]
