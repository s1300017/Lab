#!/bin/sh
set -e

# サーバーをバックグラウンドで起動
ollama serve &
OLLAMA_PID=$!

# サーバー起動待ち（最大60秒リトライ）
for i in $(seq 1 60); do
  if curl -sf http://localhost:11434/api/tags > /dev/null; then
    break
  fi
  sleep 1
done

# 既にモデルが存在する場合はpullしない
if ! ollama list | grep -q 'llama2:7b'; then
  ollama pull llama2:7b
fi
if ! ollama list | grep -q 'mistral:latest'; then
  ollama pull mistral
fi
if ! ollama list | grep -q 'llama3:latest'; then
  ollama pull llama3
fi

# サーバーをプリロード（mistral:latestの初回遅延を防ぐ）
curl -sf -X POST http://localhost:11434/api/generate -d '{"model":"mistral:latest","prompt":"ping","stream":false}' > /dev/null || true

# サーバーをフォアグラウンドで維持
wait $OLLAMA_PID
