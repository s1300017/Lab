# Windows で Ollama + DeepSeek-OCR をセットアップする手順

このドキュメントは、**Windows + RTX 3080 Ti + i9 + 32GB RAM** を想定して、
DeepSeek-OCR などの AI モデルを **Ollama** 上で動かし、
将来 Mac 側の RAG システムからリモート呼び出しできるようにするためのセットアップ手順をまとめたものです。

> ※ 現時点では Mac 側コードからの自動切り替えはまだ実装していません。
> ここでは「Windows マシンを Ollama サーバーとして立てる」ところまでを対象とします。

---

## 前提条件

- Windows 10 / 11（64bit）
- 管理者権限を持つユーザー
- GPU: RTX 3080 Ti（CUDA 対応）
- NVIDIA ドライバがインストール済みで、GPU が正常に認識されていること
- インターネット接続があること

---

## 1. NVIDIA ドライバと CUDA 周りを確認

1. **NVIDIA ドライバのバージョン確認**

   - デスクトップ右クリック → 「NVIDIA コントロール パネル」
   - もしくは Windows 検索で `nvidia` と入力し、バージョンを確認

2. **コマンドラインから GPU を確認**（任意）
   - PowerShell または `cmd` で次を実行:
     ```powershell
     nvidia-smi
     ```
   - RTX 3080 Ti が正しく表示されていれば OK です。

> Ollama は Windows 版の場合、内部的に CUDA を使って GPU を有効化します。
> ここで GPU が見えていないと、CPU 実行になってしまいます。

---

## 2. Ollama for Windows をインストール

1. ブラウザで公式サイトへアクセス:

   - https://ollama.com/download

2. **Windows 用インストーラ** (`OllamaSetup.exe` など) をダウンロードし、実行します。

3. インストール完了後、スタートメニューから **Ollama** を起動します。

   - 初回起動時にバックグラウンドサービスとして常駐します。

4. PowerShell または `cmd` を開き、次で動作確認:

   ```powershell
   ollama --version
   ```

   バージョンが表示されれば OK です。

---

## 3. DeepSeek-OCR モデルをダウンロード

### 3-1. DeepSeek-OCR モデルを pull

DeepSeek-OCR 用の Ollama モデル名は、環境に合わせて選択します（例）。
ここでは仮に `deepseek-ocr` というモデル名が公開されている前提で記述します。

1. ターミナルで次を実行:

   ```powershell
   ollama pull deepseek-ocr
   ```

2. ダウンロードが完了すると、`C:\Users\<ユーザー名>\.ollama\models\` 配下に
   モデルのファイルが保存されます。

> 実際のモデル名は Ollama Hub（https://ollama.com/library）で
> 「deepseek」「ocr」などで検索して確認してください。
> `deepseek-coder` など別モデルと混同しないよう注意します。

### 3-2. テスト実行

簡単な画像ファイルに対して、OCR が動くかを確認します。

1. 適当な画像（スクリーンショットなど）を `C:\temp\test.png` に保存。

2. 次のようなコマンドでテストします（モデルに応じて API は異なる場合があります）。
   モデルの README にサンプルがあれば、それに従ってください。

   ```powershell
   ollama run deepseek-ocr --image "C:\temp\test.png"
   ```

3. コンソールにテキストが返ってくれば成功です。

---

## 4. RAG システムから呼び出すための前提設定

将来、Mac 側の RAG システムから Windows Ollama を叩くために、
以下の前提を満たしておきます。

1. **Windows マシンのローカル IP アドレスを固定 or メモ**

   - 例: `192.168.0.50` など
   - `ipconfig` コマンドで確認:
     ```powershell
     ipconfig
     ```

2. **Windows ファイアウォールで 11434 ポートを許可**

   - 「Windows Defender ファイアウォール」→「受信の規則」→「新しい規則」
   - ポート 11434/TCP を許可（ローカルネットワーク内からのアクセスを想定）

3. **Ollama サーバーが常駐していることを確認**

   - 通常は Ollama アプリを起動しておけば、
     `http://localhost:11434` で HTTP サーバーが立ち上がっています。

4. **Mac からの疎通確認（後日）**

   - Mac 側から、次のような HTTP リクエストでテスト予定です（例）:

     ```bash
     curl http://192.168.0.50:11434/api/tags
     ```

   - モデル一覧が JSON で返ってくれば、Mac → Windows Ollama 間の通信は OK です。

---

## 5. 将来の Mac 側 RAG との接続イメージ

このドキュメントの段階では、**まだ Mac 側のコード変更は行いません**。
今後予定している接続イメージだけ簡単に書いておきます。

- `.env` に、Windows 側の URL を定義:

  ```env
  OLLAMA_BASE_URL_WINDOWS=http://192.168.0.50:11434
  ```

- backend（FastAPI）側で、

  - `mac_local` モード → `OLLAMA_BASE_URL_MAC` を使用
  - `windows_gpu` モード → `OLLAMA_BASE_URL_WINDOWS` を使用

- frontend（Streamlit）側に「推論先（Mac / Windows）」の切替 UI を用意し、
  選択結果に応じて backend のモードを切り替える API を叩く予定です。

---

## 6. トラブルシューティングのメモ

- **`ollama pull deepseek-ocr` でエラーが出る場合**

  - ネットワーク（プロキシ・VPN）を確認
  - モデル名が正しいか（Ollama Hub で確認）

- **GPU が使われているか確認したい場合**

  - モデル推論中に `nvidia-smi` を実行し、
    GPU 使用率・メモリ使用量に変化があるかを見る

- **ポート 11434 に Mac からアクセスできない場合**
  - Windows ファイアウォールの設定（受信規則）を確認
  - LAN 上で IP アドレスが変わっていないかを確認

このファイルは「Windows 側の準備だけ」をまとめたもので、
Mac 側 RAG システムとの連携は別途実装します。必要になったら、
この手順をベースに、実際の接続設定や UI からの切替手順を追記していきます。
