# Apple Silicon GPU (Metal) 対応のOllamaビルド用Dockerfile
# M4 ProのMetal GPU加速を活用した最適化ビルド
FROM --platform=linux/arm64 golang:1.24 as builder

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
# Apple Metal GPU加速対応のビルド設定
RUN OLLAMA_VERSION=$(git describe --tags $(git rev-list --tags --max-count=1)) && \
    sed -i "s/var Version string = \".*\"/var Version string = \"${OLLAMA_VERSION}\"/" version/version.go && \
    CGO_ENABLED=1 GOOS=linux GOARCH=arm64 \
    go build -tags metal -ldflags "-s -w -X github.com/jmorganca/ollama/version.Version=${OLLAMA_VERSION}" -o ollama . && \
    echo "ビルド後バージョン確認:" && ./ollama --version

# Apple Silicon用ランタイムイメージ（ARM64最適化）
FROM --platform=linux/arm64 ubuntu:22.04
RUN apt-get update && apt-get install -y ca-certificates curl

WORKDIR /app

# builderからバイナリをコピー
COPY --from=builder /build/ollama/ollama /usr/local/bin/ollama

# モデル保存用ディレクトリ
RUN mkdir -p /root/.ollama

EXPOSE 11434

ENTRYPOINT ["ollama"]
CMD ["serve"]
