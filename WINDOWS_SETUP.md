# Windows環境セットアップガイド (RTX 3080 Ti + i9-11900K)

このガイドでは、Windows環境（RTX 3080 Ti + i9-11900K）でRAGアプリケーションをセットアップする手順を説明します。

## 前提条件

- Windows 10/11 64-bit (21H2以降推奨)
- NVIDIA RTX 3080 Ti GPU (ドライバ 535.98以降)
- WSL2 (Windows Subsystem for Linux 2)
- Docker Desktop for Windows
- NVIDIA Container Toolkit
- 管理者権限

## セットアップ手順

### 1. 必要なソフトウェアのインストール

1. **WSL2の有効化**
   ```powershell
   # 管理者としてPowerShellを開き、以下のコマンドを実行
   wsl --install
   wsl --set-default-version 2
   ```
   - 再起動が必要です

2. **Docker Desktop for Windowsのインストール**
   - [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) をダウンロードしてインストール
   - インストール時に「WSL 2 ベースのエンジンを使用する」を有効化

3. **NVIDIA Container Toolkitのインストール**
   ```powershell
   # WSL2 Ubuntuを起動
   wsl
   
   # NVIDIA Container Toolkitをインストール
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   exit
   ```

### 2. アプリケーションのセットアップ

1. **リポジトリのクローン**
   ```powershell
   # 任意のディレクトリで実行
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **セットアップスクリプトの実行**
   ```powershell
   # 管理者としてPowerShellを開き、リポジトリのルートディレクトリで実行
   .\setup-windows.ps1
   ```
   - 初回実行時はDockerイメージのビルドに時間がかかります
   - 完了後、自動的にアプリケーションが起動します

### 3. アプリケーションへのアクセス

- **フロントエンド**: http://localhost:8501
- **バックエンドAPI**: http://localhost:8000
- **データベース管理ツール (Adminer)**: http://localhost:8080
  - サーバー: `db`
  - ユーザー名: `postgres`
  - パスワード: `postgres`
  - データベース: `ragdb`

## よくある問題と解決策

### GPUが認識されない場合

1. **NVIDIAドライバの確認**
   ```powershell
   nvidia-smi
   ```
   - 正しく表示されない場合は、最新のNVIDIAドライバをインストールしてください

2. **Dockerの設定確認**
   - Docker Desktopの設定 > Docker Engine に以下が含まれていることを確認:
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

### メモリ不足エラーが発生する場合

`docker-compose.windows.yml` の `backend` サービスに以下の設定を追加:

```yaml
depends_on:
  db:
    condition: service_healthy
mem_limit: 12g  # システムのメモリに応じて調整
mem_reservation: 8g
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## アプリケーションの管理

### アプリケーションの起動

```powershell
docker-compose -f docker-compose.windows.yml up -d
```

### アプリケーションの停止

```powershell
docker-compose -f docker-compose.windows.yml down
```

### ログの確認

```powershell
# 全サービスのログ
docker-compose -f docker-compose.windows.yml logs -f

# 特定のサービスのログ
docker-compose -f docker-compose.windows.yml logs -f backend
```

### データベースのバックアップ

```powershell
# バックアップの作成
docker-compose -f docker-compose.windows.yml exec -T db pg_dump -U postgres ragdb > backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').sql

# リストア
Get-Content backup_20230728_123456.sql | docker-compose -f docker-compose.windows.yml exec -T db psql -U postgres ragdb
```

## パフォーマンスチューニング

### GPUメモリ使用量の最適化

`docker-compose.windows.yml` の `backend` サービスに以下の環境変数を追加:

```yaml
environment:
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  - TF_FORCE_GPU_ALLOW_GROWTH=true
```

### バッチサイズの調整

`backend/.env` ファイルに以下を追加:

```
# バッチサイズ (VRAM 12GBの場合)
BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
```

## トラブルシューティング

### コンテナが起動しない場合

1. ログを確認:
   ```powershell
   docker-compose -f docker-compose.windows.yml logs
   ```

2. コンテナの状態を確認:
   ```powershell
   docker-compose -f docker-compose.windows.yml ps -a
   ```

3. コンテナを再構築:
   ```powershell
   docker-compose -f docker-compose.windows.yml build --no-cache
   docker-compose -f docker-compose.windows.yml up -d
   ```

### データベース接続エラー

1. データベースが起動しているか確認:
   ```powershell
   docker-compose -f docker-compose.windows.yml ps db
   ```

2. データベースのログを確認:
   ```powershell
   docker-compose -f docker-compose.windows.yml logs db
   ```

3. データベースを再起動:
   ```powershell
   docker-compose -f docker-compose.windows.yml restart db
   ```

## サポート

問題が解決しない場合は、以下の情報を添えてご連絡ください:

1. 実行したコマンドとエラーメッセージ
2. `docker-compose -f docker-compose.windows.yml logs` の出力
3. `nvidia-smi` の出力
4. `docker info` の出力
