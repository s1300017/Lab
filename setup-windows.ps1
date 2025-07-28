# Windows環境セットアップスクリプト
# 必要なソフトウェアのインストールと環境設定を行います

# 管理者権限で実行されているか確認
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "管理者権限で実行してください。" -ForegroundColor Red
    exit 1
}

# 必要な環境変数を設定
$env:COMPOSE_DOCKER_CLI_BUILD=1
$env:DOCKER_BUILDKIT=1

# 必要なディレクトリを作成
$directories = @(
    ".\local_models",
    ".\backend\logs",
    ".\backend\uploads"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# .envファイルが存在しない場合は作成
if (-not (Test-Path ".\.env")) {
    @"
# データベース設定
POSTGRES_DB=ragdb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# バックエンド設定
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_API_BASE_URL=http://host.docker.internal:11434

# フロントエンド設定
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
"@ | Out-File -FilePath ".\.env" -Encoding UTF8
    
    Write-Host "Created .env file. Please update the OPENAI_API_KEY and other settings." -ForegroundColor Yellow
}

# WSL2のインストール確認
$wslStatus = wsl -l -v 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL2 is not installed. Installing WSL2..." -ForegroundColor Yellow
    # WSL2のインストール
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    
    Write-Host "Please restart your computer and run this script again after installing WSL2." -ForegroundColor Yellow
    Write-Host "Download WSL2 kernel update: https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi" -ForegroundColor Cyan
    exit 1
}

# Docker Desktopのインストール確認
$dockerPath = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerPath) {
    Write-Host "Docker Desktop is not installed. Please install Docker Desktop for Windows with WSL2 backend." -ForegroundColor Red
    Write-Host "Download Docker Desktop: https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe" -ForegroundColor Cyan
    exit 1
}

# NVIDIA Container Toolkitのインストール確認
$nvidiaDocker = docker info 2>&1 | Select-String -Pattern "nvidia" -Quiet
if (-not $nvidiaDocker) {
    Write-Host "NVIDIA Container Toolkit is not properly configured." -ForegroundColor Yellow
    Write-Host "Please install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker" -ForegroundColor Cyan
    exit 1
}

# 必要なDockerネットワークを作成
docker network create rag_network 2>$null

# イメージのビルド
Write-Host "Building Docker images..." -ForegroundColor Cyan
docker-compose -f docker-compose.windows.yml build

# データベースの初期化
Write-Host "Initializing database..." -ForegroundColor Cyan
docker-compose -f docker-compose.windows.yml up -d db

# データベースの準備が完了するのを待機
Write-Host "Waiting for database to be ready..." -ForegroundColor Cyan
$maxRetries = 30
$retryCount = 0
$dbReady = $false

while (-not $dbReady -and $retryCount -lt $maxRetries) {
    try {
        $result = docker-compose -f docker-compose.windows.yml exec -T db pg_isready -U postgres -d ragdb
        if ($LASTEXITCODE -eq 0) {
            $dbReady = $true
            Write-Host "Database is ready!" -ForegroundColor Green
        }
    } catch {
        # エラーは無視してリトライ
    }
    
    if (-not $dbReady) {
        $retryCount++
        Write-Progress -Activity "Waiting for database" -Status "Retry $retryCount of $maxRetries" -PercentComplete (($retryCount / $maxRetries) * 100)
        Start-Sleep -Seconds 2
    }
}

if (-not $dbReady) {
    Write-Host "Failed to start database. Please check the logs with: docker-compose -f docker-compose.windows.yml logs db" -ForegroundColor Red
    exit 1
}

# アプリケーションの起動
Write-Host "Starting application..." -ForegroundColor Cyan
docker-compose -f docker-compose.windows.yml up -d

# アプリケーションの起動を確認
Write-Host "Checking application status..." -ForegroundColor Cyan
$appReady = $false
$retryCount = 0
$maxRetries = 30

while (-not $appReady -and $retryCount -lt $maxRetries) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $appReady = $true
            Write-Host "Application is ready!" -ForegroundColor Green
        }
    } catch {
        # エラーは無視してリトライ
    }
    
    if (-not $appReady) {
        $retryCount++
        Write-Progress -Activity "Waiting for application" -Status "Retry $retryCount of $maxRetries" -PercentComplete (($retryCount / $maxRetries) * 100)
        Start-Sleep -Seconds 2
    }
}

if (-not $appReady) {
    Write-Host "Application failed to start. Please check the logs with: docker-compose -f docker-compose.windows.yml logs" -ForegroundColor Red
    exit 1
}

# 完了メッセージ
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Frontend:    http://localhost:8501" -ForegroundColor Cyan
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Adminer:     http://localhost:8080" -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop the application, run:" -ForegroundColor Yellow
Write-Host "  docker-compose -f docker-compose.windows.yml down" -ForegroundColor White
Write-Host ""
Write-Host "To view logs, run:" -ForegroundColor Yellow
Write-Host "  docker-compose -f docker-compose.windows.yml logs -f" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
