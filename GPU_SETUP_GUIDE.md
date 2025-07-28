# Mac GPU加速 RAGシステム セットアップガイド

## 🚀 概要

Apple M4 ProのMetal GPU加速を活用したRAG評価システムの完全セットアップガイドです。
Docker版OllamaからネイティブOllamaに変更することで、**約4倍の性能向上**を実現します。

## 📊 パフォーマンス比較

| 構成 | 処理時間 | GPU使用 | 性能 |
|------|----------|---------|------|
| Docker版Ollama | 55秒 | CPU処理 | ベースライン |
| ネイティブOllama | 13.4秒 | Metal GPU | **4倍高速** |

## 🛠️ セットアップ手順

### 1. ネイティブOllamaのインストール

```bash
# Homebrewでインストール
brew install ollama
```

### 2. Docker版Ollamaの停止

```bash
# 既存のDocker版Ollamaを停止
docker-compose stop ollama
```

### 3. ネイティブOllamaの起動

```bash
# GPU最適化設定でネイティブOllama起動
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_NUM_CTX=4096 \
OLLAMA_KEEP_ALIVE=24h \
OLLAMA_MAX_LOADED_MODELS=3 \
OLLAMA_HOST=0.0.0.0:11434 \
ollama serve &
```

### 4. 必要なモデルのダウンロード

```bash
# 主要モデルをダウンロード
ollama pull mistral:latest
ollama pull llama3:latest
ollama pull llama2:7b
```

### 5. バックエンドの再起動

```bash
# バックエンドを再起動してネイティブOllamaに接続
docker-compose restart backend
```

## ⚙️ 最適化設定

### ネイティブOllama環境変数

| 変数 | 値 | 効果 |
|------|----|----- |
| `OLLAMA_FLASH_ATTENTION` | 1 | Flash Attention有効化 |
| `OLLAMA_NUM_CTX` | 4096 | コンテキストサイズ拡張 |
| `OLLAMA_KEEP_ALIVE` | 24h | 24時間メモリ保持 |
| `OLLAMA_MAX_LOADED_MODELS` | 3 | 複数モデル同時保持 |
| `OLLAMA_HOST` | 0.0.0.0:11434 | 外部アクセス許可 |

### バックエンド接続設定

```python
# Docker内バックエンドからホストのネイティブOllamaに接続
return OllamaLLM(model="mistral:latest", base_url="http://host.docker.internal:11434")
```

## 🔍 GPU加速の確認

### Metal GPU初期化ログ

```
load_tensors: offloaded 33/33 layers to GPU
ggml_metal_init: GPU name: Apple M4 Pro
ggml_metal_init: GPU family: MTLGPUFamilyApple9 (1009)
ggml_metal_init: hasUnifiedMemory = true
llama_context: flash_attn = 1
Metal KV buffer size = 1024.00 MiB
```

### 性能テスト

```bash
# Mistralモデルでの性能テスト
time curl -X POST http://localhost:11434/api/generate \
  -d '{"model":"mistral:latest","prompt":"テスト","stream":false,"options":{"num_predict":50}}'
```

## 🏗️ システム構成

```
┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │
│   (Docker)      │◄──►│   (Docker)      │
│   Streamlit     │    │   FastAPI       │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐
│   Database      │    │     Ollama      │
│   (Docker)      │    │   (Native)      │
│   PostgreSQL    │    │  Metal GPU      │
└─────────────────┘    └─────────────────┘
```

## 📈 期待される効果

- **推論速度**: 4倍高速化（55秒 → 13.4秒）
- **応答性**: 20秒以内での確実な処理
- **メモリ効率**: 48GB統合メモリの最適活用
- **安定性**: ネイティブ環境での安定動作
- **GPU活用**: 33/33レイヤーをMetal GPU処理

## 🚨 注意事項

1. **Docker制限**: DockerコンテナはMetal GPUに直接アクセスできません
2. **メモリ使用**: 3モデル同時保持で約13GB RAM使用
3. **初回読み込み**: 初回モデル読み込みに12-15秒必要
4. **ポート競合**: ネイティブOllamaは11434ポートを使用

## 🔧 トラブルシューティング

### モデルが見つからない場合

```bash
# モデル一覧を確認
ollama list

# 不足しているモデルをダウンロード
ollama pull mistral:latest
```

### バックエンド接続エラー

```bash
# ネイティブOllamaの動作確認
curl -s http://localhost:11434/api/version

# バックエンドの再起動
docker-compose restart backend
```

### GPU使用状況の確認

```bash
# GPU使用率を監視
sudo powermetrics --samplers gpu_power -n 1 -i 1000
```

## 🎯 完了確認

1. ✅ ネイティブOllamaが起動している
2. ✅ 必要なモデルがダウンロード済み
3. ✅ バックエンドが正常に接続している
4. ✅ GPU加速が有効になっている
5. ✅ フロントエンドでRAG処理が高速化されている

これで、Apple M4 ProのMetal GPU加速を活用した高性能RAG評価システムが完成しました！
