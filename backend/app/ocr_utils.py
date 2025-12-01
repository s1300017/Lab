from __future__ import annotations

"""PDF/OCR 関連のユーティリティ関数群。

main.py から切り出したレイアウト抽出処理や DeepSeek OCR, 画像キャプション生成などを提供する。
"""

import io
import os
import base64
import html
import json
import re
import logging

import requests
from pdf2image import convert_from_bytes
from PIL import Image

from .ollama_config import get_ollama_base_url
from .settings import get_ollama_image_caption_env_defaults

try:
    from pypdf import PdfReader  # layoutモード対応版
    HAS_PYPDF_LAYOUT = True
except ImportError:  # フォールバック: 旧PyPDF2
    from PyPDF2 import PdfReader  # type: ignore
    HAS_PYPDF_LAYOUT = False


logger = logging.getLogger(__name__)


def extract_pdf_text_layout(contents: bytes) -> str:
    """PyPDFのレイアウトモード（layout=True/extraction_mode="layout"）で抽出"""
    reader = PdfReader(io.BytesIO(contents))
    pages_text = []
    for page_index, page in enumerate(reader.pages):
        text = ""
        extract_errors: list[str] = []

        # layout対応APIを順に試行
        for kwargs in (
            {"layout": True},
            {"extraction_mode": "layout"},
            {},
        ):
            try:
                text_candidate = page.extract_text(**kwargs)
                if text_candidate:
                    text = text_candidate
                    break
            except TypeError as e:  # 未対応引数
                extract_errors.append(str(e))
                continue
            except Exception as e:  # 想定外
                extract_errors.append(str(e))
                continue

        if not text:
            logger.warning(
                "[警告] layout抽出に失敗 (page=%d). errors=%s",
                page_index,
                extract_errors if extract_errors else "なし",
            )
            text = ""

        pages_text.append(text)

    joined = "\n\n".join(pages_text)
    if HAS_PYPDF_LAYOUT:
        logger.info("[重要] PyPDF layoutモードで抽出成功: %d文字", len(joined))
    else:
        logger.warning("[警告] PyPDF layout未対応バージョン。標準抽出結果: %d文字", len(joined))
    return joined


def _normalize_deepseek_ocr_html(text: str) -> str:
    """DeepSeek OCR 応答中の HTML 断片をプレーンテキストに正規化する。"""
    s = html.unescape(str(text))
    if "<" not in s and ">" not in s:
        return s.strip()
    s = re.sub(r"</tr\\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</td\\s*>", " | ", s, flags=re.IGNORECASE)
    s = re.sub(r"<.*?>", "", s)
    lines = [line.strip(" |") for line in s.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def run_deepseek_ocr_via_ollama_for_pdf(
    contents: bytes,
    *,
    model: str | None = None,
    prompt: str | None = None,
    max_pages: int | None = None,
    dpi: int = 150,
    timeout: int = 60,
    image_output_dir=None,
    resize_max: int | None = None,
    jpeg_quality: int = 85,
) -> str:
    """PDFバイト列をOllama deepseek-ocrに渡してOCRテキストを取得するユーティリティ。"""

    try:
        pages = convert_from_bytes(contents, dpi=dpi)
    except Exception as e:  # noqa: BLE001
        logger.warning("[警告] pdf2imageによるページ画像化に失敗: %s. Ollama DeepSeek OCR処理を中止します。", e)
        raise

    if max_pages is None:
        max_pages_env = os.getenv("OLLAMA_DEEPSEEK_MAX_PAGES")
        if max_pages_env:
            try:
                max_pages = int(max_pages_env)
            except ValueError:
                max_pages = None
    if max_pages is not None:
        pages = pages[:max_pages]

    model_name = model or os.getenv("OLLAMA_DEEPSEEK_OCR_MODEL", "deepseek-ocr:latest")
    ocr_prompt = prompt or os.getenv("OLLAMA_DEEPSEEK_OCR_PROMPT", "")

    text_blocks: list[str] = []

    for idx, pil_img in enumerate(pages):
        try:
            img = pil_img.convert("RGB")
            if resize_max is not None:
                try:
                    if max(img.size) > resize_max:
                        img.thumbnail((resize_max, resize_max))
                except Exception as resize_err:  # noqa: BLE001
                    logger.warning("[警告] DeepSeek OCR画像リサイズに失敗 (page=%d): %s", idx + 1, resize_err)

            buf = io.BytesIO()
            try:
                img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            except Exception as jpeg_err:  # noqa: BLE001
                logger.warning(
                    "[警告] DeepSeek OCR画像JPEGエンコードに失敗 (page=%d): %s. PNGで再試行します。",
                    idx + 1,
                    jpeg_err,
                )
                buf = io.BytesIO()
                img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            if image_output_dir is not None:
                try:
                    image_output_dir.mkdir(parents=True, exist_ok=True)
                    save_path = image_output_dir / f"page_{idx + 1:03d}.jpg"
                    with save_path.open("wb") as f:
                        f.write(image_bytes)
                except Exception as save_err:  # noqa: BLE001
                    logger.warning("[警告] DeepSeek OCRページ画像の保存に失敗 (page=%d): %s", idx + 1, save_err)

            page_text = _run_deepseek_ocr_for_image_bytes(
                image_bytes,
                model=model_name,
                prompt=ocr_prompt,
                resize_max=None,
                jpeg_quality=jpeg_quality,
                timeout=timeout,
            )
            label = f"PAGE {idx + 1}"
            page_text_str = str(page_text).strip()
            if page_text_str:
                text_blocks.append(f"[{label}]\n{page_text_str}")
            else:
                logger.warning("[警告] deepseek-ocr(Ollama)から空テキスト (page=%d)", idx + 1)
        except Exception as e:  # noqa: BLE001
            logger.warning("[警告] deepseek-ocr(Ollama)によるOCR処理に失敗 (page=%d): %s", idx + 1, e)

    if not text_blocks:
        raise RuntimeError("Ollama DeepSeek OCRから有効なテキストを取得できませんでした。")

    return "\n\n".join(text_blocks).strip()


def _run_deepseek_ocr_for_image_bytes(
    image_bytes: bytes,
    *,
    model: str | None = None,
    prompt: str | None = None,
    resize_max: int | None = None,
    jpeg_quality: int = 85,
    timeout: int = 300,
) -> str:
    """単一画像バイト列に対して DeepSeek-OCR (Ollama) を実行するヘルパー。"""

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"DeepSeek OCR画像の読み込みに失敗しました: {e}") from e

    if resize_max is not None:
        try:
            if max(img.size) > resize_max:
                img.thumbnail((resize_max, resize_max))
        except Exception as resize_err:  # noqa: BLE001
            logger.warning("[警告] DeepSeek OCR画像リサイズに失敗: %s", resize_err)

    buf = io.BytesIO()
    try:
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    except Exception as jpeg_err:  # noqa: BLE001
        logger.warning("[警告] DeepSeek OCR画像JPEGエンコードに失敗: %s. PNGで再試行します。", jpeg_err)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    encoded_bytes = buf.getvalue()

    img_b64 = base64.b64encode(encoded_bytes).decode("utf-8")

    base_url = get_ollama_base_url().rstrip("/")
    chat_url = f"{base_url}/api/chat"
    model_name = model or os.getenv("OLLAMA_DEEPSEEK_OCR_MODEL", "deepseek-ocr:latest")
    cli_ocr_prompt = (
        "You are an OCR engine. Read all text in the image and output only the plain text. "
        "Do not explain anything, just output the recognized text."
    )
    ocr_prompt = prompt or cli_ocr_prompt

    body = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": ocr_prompt,
                "images": [img_b64],
            }
        ],
        "stream": True,
    }

    chunks: list[str] = []
    with requests.post(chat_url, json=body, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event.get("message") or {}
            delta = message.get("content")
            if delta:
                chunks.append(delta)
            if event.get("done"):
                break

    text = "".join(chunks)
    text = _normalize_deepseek_ocr_html(text)
    return text.strip()


def generate_image_captions_from_pdf(
    contents: bytes,
    max_pages: int | None = None,
    dpi: int = 150,
    timeout: int = 30,
) -> list[dict]:
    """PDFページごとに画像を生成し、llava 等で日本語キャプションを生成するユーティリティ。"""

    captions: list[dict] = []

    max_pages, timeout, caption_model, caption_prompt = get_ollama_image_caption_env_defaults(
        default_max_pages=max_pages,
        default_timeout=timeout,
    )

    try:
        pages = convert_from_bytes(contents, dpi=dpi)
    except Exception as e:  # noqa: BLE001
        logger.warning("[警告] pdf2imageによるページ画像化に失敗: %s. 画像キャプション処理をスキップします。", e)
        return captions

    if max_pages is not None:
        pages = pages[:max_pages]

    base_url = get_ollama_base_url().rstrip("/")
    chat_url = f"{base_url}/api/chat"

    for idx, pil_img in enumerate(pages):
        try:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            img_b64 = base64.b64encode(png_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{img_b64}"

            body = {
                "model": caption_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image_url": data_url},
                            {"type": "text", "text": caption_prompt},
                        ],
                    }
                ],
                "stream": False,
            }

            resp = requests.post(chat_url, json=body, timeout=timeout)
            resp.raise_for_status()
            j = resp.json()
            caption = (
                j.get("message", {}).get("content")
                or j.get("response")
                or "(説明なし)"
            )
            captions.append({"page": idx + 1, "caption": str(caption).strip()})
        except Exception as e:  # noqa: BLE001
            logger.warning("[警告] llavaによる画像キャプション生成に失敗 (page=%d): %s", idx + 1, e)

    return captions


def cleanse_pdf_text(text: str) -> str:
    """PDF抽出テキストを表構造を維持しつつノイズ除去するユーティリティ。

    主な処理内容:
    - ページ全体で頻出するヘッダ/フッタ相当のボイラープレート行を検出して除去
    - 明らかなページ番号行 ("1/10", "Page 3" など) を除去
    - 表っぽい行はタブ区切りに正規化してひとまとまりのブロックとして残す
    - 連続ハイフネーションによる単語分割 (exam-\nple) を簡易的に解消
    - 過剰な空行を1行までに圧縮
    """
    import re

    # --- 1. 行単位に分割 ---
    lines = text.splitlines()

    # --- 2. 行末ハイフネーションを簡易的に解消 ---
    merged_lines: list[str] = []
    for line in lines:
        if merged_lines:
            prev = merged_lines[-1]
            if prev.rstrip().endswith("-") and re.match(r"^[A-Za-zぁ-んァ-ン一-龥0-9]", line.lstrip() or ""):
                merged_lines[-1] = prev.rstrip()[:-1] + line.lstrip()
                continue
        merged_lines.append(line)
    lines = merged_lines

    # --- 3. 頻出行に基づくボイラープレート候補の検出 ---
    def _normalize_for_boilerplate(s: str) -> str:
        s_norm = re.sub(r"\s+", " ", s.strip())
        return s_norm.lower()

    freq: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) < 5 or len(stripped) > 200:
            continue
        norm = _normalize_for_boilerplate(stripped)
        freq[norm] = freq.get(norm, 0) + 1

    boilerplate_norms = {k for k, v in freq.items() if v >= 3}

    # --- 4. 実際のクレンジング処理 ---
    page_number_pattern = re.compile(
        r"^(?:"
        r"\d+\s*$"
        r"|"
        r"\d{1,4}\s*/\s*\d{1,4}"
        r"|"
        r"page\s+\d{1,4}(?:\s*/\s*\d{1,4})?"
        r")$",
        re.IGNORECASE,
    )

    cleaned: list[str] = []
    table_buffer: list[str] = []

    table_border_pattern = re.compile(r"^[\s\-=~_`+*•·―—┄┅┈┉┌┐└┘┬┴┼┤├╴╶╷╵╸╹╺╻╾╿]+$")
    table_delimiter_pattern = re.compile(r"[\|│┃┆┊┋┇┈┉┌┐└┘┬┴┼┤├]")

    def flush_table() -> None:
        if not table_buffer:
            return
        normalized_rows: list[str] = []
        for raw in table_buffer:
            row = table_delimiter_pattern.sub("\t", raw)
            row = re.sub(r"\s{2,}", "\t", row.strip())
            row = re.sub(r"\t{2,}", "\t", row)
            normalized_rows.append(row)
        if cleaned and cleaned[-1] != "":
            cleaned.append("")
        cleaned.append("\n".join(normalized_rows))
        cleaned.append("")
        table_buffer.clear()

    prev_blank = False
    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush_table()
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
            continue

        if page_number_pattern.match(stripped):
            flush_table()
            continue

        norm = _normalize_for_boilerplate(stripped)
        if norm in boilerplate_norms:
            flush_table()
            continue

        if table_border_pattern.match(stripped):
            flush_table()
            continue

        if table_delimiter_pattern.search(stripped):
            table_buffer.append(stripped)
            prev_blank = False
            continue

        space_chunks = re.findall(r"\s{2,}", line)
        if len(space_chunks) >= 2:
            table_buffer.append(stripped)
            prev_blank = False
            continue

        flush_table()
        cleaned.append(stripped)
        prev_blank = False

    flush_table()

    while len(cleaned) > 1 and cleaned[-1] == "" and cleaned[-2] == "":
        cleaned.pop()

    return "\n".join(cleaned)
