# Windows RTX 3080 Ti 最適化GPUセットアップガイド

## 1. 前提条件の確認

### システム要件
- **OS**: Windows 11 64-bit (21H2以降推奨)
- **GPU**: NVIDIA RTX 3080 Ti (VRAM 12GB)
- **CPU**: Intel Core i9-11900K
- **メモリ**: 32GB以上推奨
- **ストレージ**: SSDに50GB以上の空き容量
- **ドライバ**: NVIDIAドライバ 535.98以降

### 必要なソフトウェア
1. [NVIDIA CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-downloads)
2. [cuDNN 8.9.x for CUDA 12.x](https://developer.nvidia.com/cudnn)
3. [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
4. [WSL2 (Windows Subsystem for Linux 2)](https://learn.microsoft.com/ja-jp/windows/wsl/install)
5. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## 2. 環境設定

### NVIDIAドライバの確認
```powershell
# NVIDIAドライバのバージョン確認
nvidia-smi
```

### CUDAのインストール
1. CUDA Toolkitインストーラーを実行
2. カスタムインストールを選択し、以下のコンポーネントをインストール:
   - CUDA Runtime
   - CUDA Developer Tools
   - CUDA Samples
   - NVIDIA Nsight Systems
   - NVIDIA Nsight Compute
   - NVIDIA Visual Profiler

### cuDNNのインストール
1. NVIDIAアカウントでログインしてcuDNNをダウンロード
2. ダウンロードしたZIPを解凍し、中身をCUDAインストールディレクトリにコピー
   - `cuda\bin` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`
   - `cuda\include` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include`
   - `cuda\lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib`

## 3. DockerとWSL2の設定

### WSL2の有効化
```powershell
# WSL2を有効化
wsl --install
wsl --set-default-version 2

# Ubuntu 22.04をインストール
wsl --install -d Ubuntu-22.04
```

### Docker Desktopの設定
1. Docker Desktopを起動
2. Settings > Resources > WSL Integration でUbuntuを有効化
3. Settings > Kubernetes でKubernetesを無効化（不要な場合）
4. Settings > Docker Engine で以下の設定を追加:
```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

### NVIDIA Container Toolkitのインストール
```powershell
# WSL2 Ubuntuで実行
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 4. Ollamaのセットアップ（CUDA版）

### Docker Composeの設定
`docker-compose.gpu.yml` を作成:
```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - OLLAMA_KEEP_ALIVE=24h
      - OLLAMA_MAX_LOADED_MODELS=2
      - OLLAMA_NUM_CTX=4096
      - OLLAMA_NUM_THREAD=16
      - OLLAMA_NO_MUL_MAT_Q=1
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

### サービス起動
```powershell
docker-compose -f docker-compose.gpu.yml up -d
```

## 5. モデルのダウンロード

```powershell
# WSL2内で実行
docker exec -it ollama_ollama_1 ollama pull mistral:latest
docker exec -it ollama_ollama_1 ollama pull llama3:latest
```

## 6. パフォーマンス最適化

### NVIDIA設定の調整
1. NVIDIA コントロールパネルを開く
2. "3D設定の管理"を選択
3. 以下の設定を適用:
   - 電源管理モード: 最高パフォーマンス
   - テクスチャ フィルタリング - 品質: 高パフォーマンス
   - スレッド化最適化: ON
   - 電源管理モード: 優先パフォーマンス

### Windows電源設定
```powershell
# 高パフォーマンスモードに設定
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

## 7. トラブルシューティング

### GPU認識の確認
```powershell
# WSL2内で実行
nvidia-smi
```

### メモリリークの監視
```powershell
# タスクマネージャーでGPUメモリ使用量を確認
# またはWSL2内で
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

### パフォーマンスモニタリング
```powershell
# WSL2内で
nvtop
```

## 8. 推奨設定

### 環境変数
```env
# バッチサイズの調整（VRAM 12GBの場合）
OLLAMA_BATCH_SIZE=512
# スレッド数（i9-11900Kは16スレッド）
OLLAMA_NUM_THREAD=16
# コンテキスト長
OLLAMA_NUM_CTX=4096
# モデル保持時間
OLLAMA_KEEP_ALIVE=24h
```

### 推奨起動オプション
```yaml
# docker-compose.gpu.yml のenvironmentセクションに追加
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - OLLAMA_KEEP_ALIVE=24h
  - OLLAMA_MAX_LOADED_MODELS=2
  - OLLAMA_NUM_CTX=4096
  - OLLAMA_NUM_THREAD=16
  - OLLAMA_NO_MUL_MAT_Q=1
  - OLLAMA_BATCH_SIZE=512
```

## 9. パフォーマンスベンチマーク

### テストコマンド
```powershell
# 推論速度テスト
docker exec -it ollama_ollama_1 ollama run mistral "こんにちは、あなたは誰ですか？"

# ベンチマーク実行
docker exec -it ollama_ollama_1 ollama run mistral --verbose
```

### 期待されるパフォーマンス
- **推論速度**: 30-50 tokens/秒（RTX 3080 Ti）
- **VRAM使用量**: 8-10GB（Mistral 7B）
- **コンテキスト長**: 4096トークンまで安定

## 10. 更新履歴

- 2025-07-28: 初版リリース
  - RTX 3080 Ti + i9-11900K向け最適化設定
  - CUDA 12.1 + cuDNN 8.9.x対応
  - WSL2 + Docker環境構築ガイド
  - パフォーマンスチューニング手順
