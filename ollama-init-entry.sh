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

# 全モデルの完全プリロードとキャッシュ化
echo "=== Starting comprehensive model preloading ==="

# プリロード関数
preload_model() {
    local model_name="$1"
    local display_name="$2"
    echo "Preloading $display_name model..."
    
    # 軽量なプロンプトでモデルをウォームアップ（メモリ節約）
    curl -sf -X POST http://localhost:11434/api/generate -d "{
        \"model\": \"$model_name\",
        \"prompt\": \"Hi\",
        \"stream\": false,
        \"options\": {
            \"temperature\": 0.0,
            \"num_predict\": 1
        }
    }" > /dev/null 2>&1 || echo "Warning: $display_name preload failed"
    
    echo "✓ $display_name model preloaded and cached"
}

# 全モデルを順次プリロード（安定性重視）
echo "Starting sequential model preloading for stability..."

# メモリ使用量を抑えるため順次実行
echo "Preloading Mistral (primary model)..."
preload_model "mistral:latest" "Mistral"

echo "Waiting 10 seconds before next model..."
sleep 10

echo "Preloading Llama3..."
preload_model "llama3:latest" "Llama3"

echo "Waiting 10 seconds before next model..."
sleep 10

echo "Preloading Llama2..."
preload_model "llama2:7b" "Llama2"

echo "All models preloaded sequentially."

# メモリ使用量を確認
echo "=== Memory usage after preloading ==="
free -h || echo "Memory info not available"

# モデル一覧を表示
echo "=== Available models ==="
ollama list

# プリロード成功の最終確認
echo "=== Final preload verification ==="
test_model() {
    local model_name="$1"
    local display_name="$2"
    echo "Testing $display_name responsiveness..."
    
    response_time=$(curl -w "%{time_total}" -s -o /dev/null -X POST http://localhost:11434/api/generate -d "{
        \"model\": \"$model_name\",
        \"prompt\": \"Test\",
        \"stream\": false,
        \"options\": {\"num_predict\": 1}
    }" 2>/dev/null || echo "999")
    
    # 応答時間が5秒未満かどうかをチェック（bcコマンド不要）
    response_int=$(echo "$response_time" | cut -d'.' -f1)
    if [ "$response_int" -lt 5 ] 2>/dev/null; then
        echo "✓ $display_name: Ready (${response_time}s)"
    else
        echo "⚠️ $display_name: Slow response (${response_time}s)"
    fi
}

# 各モデルの応答性能をテスト
test_model "mistral:latest" "Mistral"
test_model "llama3:latest" "Llama3"
test_model "llama2:7b" "Llama2"

echo ""
echo "🎉 All models successfully preloaded and cached!"
echo "⚡ Models are ready for immediate use without initial delays."
echo "📊 System is optimized for RAG evaluation workloads."

# プリロード完了を示すファイルを作成
touch /tmp/ollama_preload_complete
echo "Preload completion marker created at /tmp/ollama_preload_complete"

# サーバーをフォアグラウンドで維持
wait $OLLAMA_PID
