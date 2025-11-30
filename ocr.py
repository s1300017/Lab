# deepseek_ocr_simple.py
# 素の Ollama HTTP API を使って DeepSeek-OCR に画像を投げるサンプルスクリプト（英語プロンプト）

import base64
import json
import pathlib
from io import BytesIO

import requests
from PIL import Image
from requests.exceptions import ReadTimeout


def encode_image_to_base64(image_path: str, max_size: int = 1600, quality: int = 85) -> str:
    """画像ファイルを圧縮（リサイズ＆再エンコード）して base64 文字列に変換する関数。

    max_size: 長辺の最大ピクセル数。これを超える場合はアスペクト比を維持して縮小する。
    quality: JPEG の画質。数値が大きいほど高画質・高容量。
    """
    p = pathlib.Path(image_path)
    # Pillow で画像を開いて、長辺 max_size に収まるように縮小
    with Image.open(p) as img:
        img = img.convert("RGB")  # OCR 用なので RGB に正規化
        # サムネイル関数でアスペクト比を維持したまま縮小
        img.thumbnail((max_size, max_size))

        # メモリ上に JPEG として書き出し（PNG などでもサイズ削減のため JPEG に統一）
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()

    return base64.b64encode(data).decode("utf-8")


def call_deepseek_ocr(
    image_path: str,
    ollama_base_url: str = "http://localhost:11434",
    model_name: str = "deepseek-ocr",
) -> dict:
    """DeepSeek-OCR モデルを素の Ollama API から呼び出す。

    :param image_path: OCR したい画像ファイルへのパス
    :param ollama_base_url: Ollama サーバーの URL（例: http://localhost:11434 や http://192.168.x.x:11434）
    :param model_name: 使用するモデル名（Ollama 側で pull 済みの DeepSeek-OCR モデル名）
    :return: レスポンスの JSON を dict で返す
    """
    image_b64 = encode_image_to_base64(image_path)

    # Ollama の /api/chat エンドポイントを使用して、GUI と近い呼び出し形式に合わせる
    url = f"{ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are an OCR engine. Read all text in the image and output only the plain text. "
                    "Do not explain anything, just output the recognized text."
                ),
                "images": [image_b64],
            }
        ],
        "stream": True,
    }

    # ストリーミングレスポンスを逐次読み取り、message.content を結合しつつリアルタイム表示する
    try:
        full_text_chunks: list[str] = []
        events: list[dict] = []
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # JSON でパースできない行はスキップ
                    continue
                events.append(event)
                # /api/chat のストリーミングレスポンスでは message.content にトークンが積まれていく
                message = event.get("message") or {}
                delta = message.get("content")
                if delta:
                    # 受信したテキストを即時にコンソールへ出力
                    print(delta, end="", flush=True)
                    full_text_chunks.append(delta)
                if event.get("done"):
                    break

        return {
            "events": events,
            "response": "".join(full_text_chunks),
        }
    except ReadTimeout:
        print("DeepSeek-OCR からの応答がタイムアウトしました。初回ロード中の可能性があります。")
        return {}


def main():
    # TODO: ここを手元のファイル名・URL に合わせて変更してください
    image_path = "/Users/ryutaro/Downloads/s1300017.png"

    # Mac ローカルの Ollama に投げる場合
    # ollama_base_url = "http://localhost:11434"

    # Windows 側の Ollama に投げる場合（例: IP は環境に合わせて変更）
    # ollama_base_url = "http://192.168.0.50:11434"

    ollama_base_url = "http://localhost:11434"

    result = call_deepseek_ocr(
        image_path=image_path,
        ollama_base_url=ollama_base_url,
        model_name="deepseek-ocr",  # 必要なら deepseek-ocr:latest などに変更
    )

    # Ollama の /api/generate のレスポンスは JSON 形式で、
    # 生成テキストは通常 result["response"] に入っている
    print("=== Raw JSON ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== OCR Result Text ===")
    print(result.get("response", ""))


if __name__ == "__main__":
    main()