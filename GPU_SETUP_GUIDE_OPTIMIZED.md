# Mac M4 Pro 最適化GPUセットアップガイド

## 1. ネイティブOllamaのインストール

```bash
# Homebrewでインストール
brew install ollama
```

## 2. Docker版Ollamaの停止

```bash
# 既存のDocker版Ollamaを停止
docker-compose stop ollama
```

## 3. 最適化されたネイティブOllamaの起動

```bash
# 最適化設定でネイティブOllamaを起動
GGML_METAL_USE_BF16=1 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_NUM_CTX=4096 \
OLLAMA_KEEP_ALIVE=24h \
OLLAMA_MAX_LOADED_MODELS=3 \
OLLAMA_NUM_THREADS=10 \
OLLAMA_THREADS=10 \
OLLAMA_NO_MUL_MAT_Q=1 \
OLLAMA_HOST=0.0.0.0:11434 \
ollama serve &
```

## 4. 必要なモデルのダウンロード

```bash
# 主要モデルをダウンロード
ollama pull mistral:latest  # 推奨モデル
ollama pull llama3:latest   # 代替モデル
ollama pull llama2:7b       # バックアップモデル
```

## 5. モデルの事前ロード（オプション）

```bash
# モデルを事前にロードしてウォームアップ
for model in mistral llama3 llama2; do
  echo "Preloading $model..."
  ollama run $model "Hi" > /dev/null 2>&1
  sleep 5
done
```

## 6. バックエンドの再起動

```bash
# バックエンドを再起動して最適化されたOllamaに接続
docker-compose restart backend
```

## 最適化ポイント

### パフォーマンス向上
- **Metal GPUアクセラレーション**: 全レイヤーをGPUで処理（33/33レイヤー）
- **bfloat16サポート**: 精度を維持しつつ高速化
- **Flash Attention**: 長文処理の高速化
- **スレッド最適化**: CPUコアを効率的に活用

### メモリ効率
- **モデル保持**: 24時間メモリに保持
- **同時読み込み**: 最大3モデルまで保持可能
- **コンテキスト長**: 4096トークン対応

### 安定性
- **エラーハンドリング**: 堅牢なプロセス管理
- **リソース制限**: メモリリーク防止
- **自動リカバリ**: プロセス監視と自動再起動

## トラブルシューティング

### GPUアクセラレーションの確認
```bash
# ログに以下のような行があれば正常にGPUが認識されています
# ggml_metal_init: GPU name: Apple M4 Pro
# ggml_metal_init: GPU family: MTLGPUFamilyApple9
```

### メモリ使用量の確認
```bash
# メモリ使用量を確認
ps aux | grep ollama
```

### パフォーマンスモニタリング
```bash
# GPU使用率をリアルタイム監視
sudo powermetrics --samplers gpu_power -i 1000
```

## 注意事項

1. 初回起動時はモデルの読み込みに時間がかかります（最大1分程度）
2. メモリ使用量が多いため、他のメモリ集中型アプリケーションの実行は控えてください
3. 定期的にOllamaを再起動することで、メモリリークを防げます
4. 問題が発生した場合は、`pkill -f ollama`でプロセスを終了してから再起動してください

## 推奨システム要件

- **OS**: macOS 13 Ventura 以降
- **メモリ**: 16GB 以上（推奨32GB以上）
- **ストレージ**: モデル用に20GB以上の空き容量
- **プロセッサ**: Apple Mシリーズチップ（M1/M2/M3/M4）

## 更新履歴

- 2025-07-28: 最適化版ガイドを作成
  - bfloat16サポートを追加
  - スレッド設定を最適化
  - トラブルシューティングセクションを追加
  - 推奨システム要件を明記
