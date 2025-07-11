# ollama-custom ビルド手順（日本語）

このディレクトリには最新版OllamaのDockerイメージを自前でビルドするためのDockerfileが入っています。

## 手順（自動化スクリプトも後で用意します）

1. このディレクトリでDockerイメージをビルド

```sh
docker build -t ollama-custom:latest .
```

2. docker-compose.yml の `image:` を `ollama-custom:latest` に変更

3. モデルpull例（Ollama 1.0.0以降対応モデルがpull可能）

```sh
docker run --rm -it -v $PWD/ollama_data:/root/.ollama -p 11434:11434 ollama-custom:latest serve &
sleep 5
docker exec -it <コンテナ名> ollama pull nous-hermes2-yi-jp
```

またはcomposeでcommand指定

```yaml
command: -c "ollama serve & sleep 5 && ollama pull nous-hermes2-yi-jp && wait"
```

## 注意
- 公式リリースタグは https://github.com/jmorganca/ollama/releases を参照
- 最新タグは自動でcheckoutされます
- モデルpullには数GBの空き容量が必要

---
何か問題があればこのREADMEの内容を参考にしてください。
