from __future__ import annotations

import os
from pathlib import Path
import logging
import json
from datetime import datetime, timezone
import contextvars

POSTGRES_DB = os.environ.get("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "rag_password")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "db")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")

_DATABASE_URL = os.environ.get("DATABASE_URL")
if _DATABASE_URL:
    DB_URL = _DATABASE_URL
else:
    DB_URL = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdf"
EXTRACTED_DIR = DATA_DIR / "extracted"
IMAGES_DIR = DATA_DIR / "images"

# データ保存用ディレクトリを初期化
for _p in (PDF_DIR, EXTRACTED_DIR, IMAGES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# LLM と埋め込みモデルのデフォルト設定
DEFAULT_LLM_NAME = "gpt-oss"
DEFAULT_EMBEDDING_MODEL = "huggingface_bge_small"

SUPPORTED_EMBEDDING_MODELS = {
    # OpenAIモデル
    "openai",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    # HuggingFaceモデル
    "huggingface_bge_small",
    "huggingface_bge_large",
    "huggingface_miniLM",
    "huggingface_mpnet_base",
    "huggingface_multi_qa_minilm",
    "huggingface_multi_qa_mpnet",
    "huggingface_paraphrase_multilingual",
    "huggingface_distiluse_multilingual",
    "huggingface_xlm_r",
    "jina-embeddings-v4",
    # Ollama埋め込みモデル
    "nomic-embed-text",
    "mxbai-embed-large",
    "mxbai-embed-large:335m",
    "all-minilm",
    "bge-m3",
    "qwen3-embedding",
    "snowflake-arctic-embed2",
    "jina-embeddings-v3",
}


def resolve_ocr_image_compression(ocr_image_compression: str | None) -> tuple[str, int, int]:
    """OCR画像圧縮モードと、それに対応する resize_max / jpeg_quality を環境変数込みで解決する。"""

    mode = (ocr_image_compression or "balanced").lower()
    if mode not in {"light", "balanced", "high"}:
        mode = "balanced"

    if mode == "light":
        resize_key = "OLLAMA_DEEPSEEK_IMG_MAX_LIGHT"
        resize_default = "1200"
        jpeg_key = "OLLAMA_DEEPSEEK_JPEG_Q_LIGHT"
        jpeg_default = "70"
    elif mode == "high":
        resize_key = "OLLAMA_DEEPSEEK_IMG_MAX_HIGH"
        resize_default = "2048"
        jpeg_key = "OLLAMA_DEEPSEEK_JPEG_Q_HIGH"
        jpeg_default = "92"
    else:  # balanced
        resize_key = "OLLAMA_DEEPSEEK_IMG_MAX_BALANCED"
        resize_default = "1600"
        jpeg_key = "OLLAMA_DEEPSEEK_JPEG_Q_BALANCED"
        jpeg_default = "85"

    try:
        resize_max = int(os.getenv(resize_key, resize_default))
    except Exception:  # noqa: BLE001
        resize_max = int(resize_default)

    try:
        jpeg_quality = int(os.getenv(jpeg_key, jpeg_default))
    except Exception:  # noqa: BLE001
        jpeg_quality = int(jpeg_default)

    return mode, resize_max, jpeg_quality


def get_ollama_deepseek_timeout(default_seconds: int) -> int:
    """DeepSeek OCR 用のタイムアウト値を環境変数から取得する（不正値はデフォルトにフォールバック）。"""

    try:
        return int(os.getenv("OLLAMA_DEEPSEEK_TIMEOUT", str(default_seconds)))
    except Exception:  # noqa: BLE001
        return default_seconds


def get_ollama_image_caption_env_defaults(
    *,
    default_max_pages: int | None = None,
    default_timeout: int = 30,
) -> tuple[int | None, int, str, str]:
    """画像キャプション生成用のページ数・タイムアウト・モデル・プロンプトのデフォルトを環境変数込みで解決する。"""

    max_pages: int | None = default_max_pages
    try:
        max_pages_env = os.getenv("OLLAMA_IMAGE_CAPTION_MAX_PAGES", "").strip()
        if max_pages_env:
            max_pages = int(max_pages_env)
    except Exception:  # noqa: BLE001
        pass

    timeout = default_timeout
    try:
        timeout_env = os.getenv("OLLAMA_IMAGE_CAPTION_TIMEOUT", "").strip()
        if timeout_env:
            timeout = int(timeout_env)
    except Exception:  # noqa: BLE001
        pass

    caption_model = os.getenv("OLLAMA_IMAGE_CAPTION_MODEL", "llava:7b")
    caption_prompt = os.getenv(
        "OLLAMA_IMAGE_CAPTION_PROMPT",
        "この画像の内容を日本語で簡潔に説明してください。",
    )

    return max_pages, timeout, caption_model, caption_prompt


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower()


# HTTP リクエスト単位の request_id を保持するための ContextVar
REQUEST_ID_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id",
    default=None,
)


class JsonFormatter(logging.Formatter):
    """標準 logging 用のシンプルな JSON フォーマッタ。"""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        log_record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # file_id / job_id / experiment_id などの識別子や共通タグが extra 経由で渡されていればログに含める
        for key in ("file_id", "pdf_file_id", "job_id", "experiment_id", "chat_log_id", "component", "endpoint"):
            value = getattr(record, key, None)
            if value is not None:
                log_record[key] = value
        # request_id は LogRecord に直接指定されている場合か、ContextVar から取得して付与する
        request_id = getattr(record, "request_id", None)
        if request_id is None:
            try:
                request_id = REQUEST_ID_CTX.get()
            except LookupError:  # pragma: no cover - 安全側
                request_id = None
        if request_id is not None:
            log_record["request_id"] = request_id
        if record.exc_info:
            # 例外スタックトレースも必要に応じて含める
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record, ensure_ascii=False)


def configure_logging() -> None:
    """アプリ共通の logging 設定を初期化するヘルパー。既に設定済みなら何もしない。"""

    root = logging.getLogger()
    if root.handlers:
        # 他所ですでに設定されている場合は上書きしない
        return

    root.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler()
    if LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            ),
        )
    root.addHandler(handler)


# settings 読み込み時に一度だけ logging を初期化
configure_logging()
