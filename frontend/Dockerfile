FROM python:3.10-slim

# 環境変数の設定
ENV LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    TZ=Asia/Tokyo \
    PYTHONIOENCODING=utf-8 \
    FONTCONFIG_PATH=/etc/fonts \
    FONTCONFIG_FILE=/etc/fonts/fonts.conf

WORKDIR /app

# 日本語フォントと必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    fonts-ipafont \
    fonts-ipafont-gothic \
    fonts-ipafont-mincho \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fonts-takao \
    fonts-vlgothic \
    fontconfig \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# フォントキャッシュの更新
RUN mkdir -p /usr/share/fonts/opentype/ \
    && fc-cache -f -v

# フォント設定ファイルのコピー
COPY fonts/ /usr/share/fonts/opentype/

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . /app

# フォント設定の確認
RUN python -c "import matplotlib; print('Font cache directory:', matplotlib.get_cachedir())"
RUN python -c "import matplotlib.font_manager; print('Available fonts:', [f.name for f in matplotlib.font_manager.fontManager.ttflist if 'ipag' in f.name.lower() or 'noto' in f.name.lower()])"

CMD ["streamlit", "run", "app.py"]
