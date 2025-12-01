from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Dict

import base64
import hashlib
import io
import json
import os
import re
import tempfile
import textwrap
import threading
import time
import uuid
import logging

from fastapi import HTTPException
from sqlalchemy import create_engine, text

from .ocr_utils import (
    extract_pdf_text_layout,
    cleanse_pdf_text,
    run_deepseek_ocr_via_ollama_for_pdf,
    generate_image_captions_from_pdf,
)
from .persistence_utils import persist_pdf_upload_to_db
from .chunk_utils import generate_default_chunks_for_storage
from .llm_ragas_utils import (
    _extract_answer_text,
    split_sentences,
    build_rag_answer_prompt,
    extract_relevant_context,
    evaluate_answer_quality,
    regenerate_answer_with_context,
)
from .settings import (
    DB_URL,
    DATA_DIR,
    PDF_DIR,
    EXTRACTED_DIR,
    resolve_ocr_image_compression,
    get_ollama_deepseek_timeout,
)


logger = logging.getLogger(__name__)


# DB 設定とパス類（main.py と同等の環境変数/構成を利用）
engine = create_engine(DB_URL)
IMAGES_DIR = DATA_DIR / "images"


# MLX DeepSeek OCR (任意機能): main.py と同様に安全にインポートする
try:
    from .mlx_deepseek_ocr_check import (
        DEFAULT_OCR_PROMPT,
        DEFAULT_PHOTO_PROMPT,
        run_deepseek_ocr,
    )

    MLX_OCR_AVAILABLE = True
except Exception:  # noqa: BLE001
    MLX_OCR_AVAILABLE = False
    DEFAULT_OCR_PROMPT = (
        "Please transcribe every visible character from this image in Japanese. "
        "Do not describe the scene. If any part is unreadable, leave it blank without guessing."
    )
    DEFAULT_PHOTO_PROMPT = (
        "Please describe the content of this image in Japanese briefly and clearly."
    )


class UploadJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class UploadJobState:
    job_id: str
    status: UploadJobStatus = UploadJobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_requested: bool = False
    progress: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None


_UPLOAD_JOBS: Dict[str, UploadJobState] = {}
_UPLOAD_JOBS_LOCK = threading.Lock()


def _db_create_upload_job(job_id: str, status: UploadJobStatus, progress: str) -> None:
    from .main import jst_now_str  # 遅延インポート

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO upload_jobs (job_id, status, progress, cancel_requested, created_at, updated_at)
                    VALUES (:job_id, :status, :progress, FALSE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (job_id) DO UPDATE
                    SET status = EXCLUDED.status,
                        progress = EXCLUDED.progress,
                        cancel_requested = EXCLUDED.cancel_requested,
                        updated_at = CURRENT_TIMESTAMP
                    """
                ),
                {"job_id": job_id, "status": status.value, "progress": progress},
            )
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[%s][ERROR] upload_jobs insert/update失敗: %s",
            jst_now_str(),
            e,
            extra={"job_id": job_id},
        )


def _db_update_upload_job(
    job_id: str,
    *,
    status: Optional[UploadJobStatus] = None,
    progress: Optional[str] = None,
    cancel_requested: Optional[bool] = None,
    error: Optional[str] = None,
    result: Optional[dict] = None,
    file_id: Optional[str] = None,
) -> None:
    from .main import jst_now_str  # 遅延インポート

    fields: list[str] = []
    params: Dict[str, Any] = {"job_id": job_id}

    if status is not None:
        fields.append("status = :status")
        params["status"] = status.value
    if progress is not None:
        fields.append("progress = :progress")
        params["progress"] = progress
    if cancel_requested is not None:
        fields.append("cancel_requested = :cancel_requested")
        params["cancel_requested"] = cancel_requested
    if error is not None:
        fields.append("error = :error")
        params["error"] = error
    if result is not None:
        fields.append("result_json = :result_json")
        params["result_json"] = json.dumps(result, ensure_ascii=False)
    if file_id is not None:
        fields.append("file_id = :file_id")
        params["file_id"] = file_id

    if not fields:
        return

    set_clause = ", ".join(fields) + ", updated_at = CURRENT_TIMESTAMP"
    sql = f"UPDATE upload_jobs SET {set_clause} WHERE job_id = :job_id"
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), params)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[%s][ERROR] upload_jobs更新失敗: %s",
            jst_now_str(),
            e,
            extra={"job_id": job_id},
        )


class UploadJobCancelled(Exception):
    pass


def get_upload_job(job_id: str) -> Optional[UploadJobState]:
    with _UPLOAD_JOBS_LOCK:
        return _UPLOAD_JOBS.get(job_id)


def set_upload_job(job: UploadJobState) -> None:
    with _UPLOAD_JOBS_LOCK:
        _UPLOAD_JOBS[job.job_id] = job
    _db_create_upload_job(job.job_id, job.status, job.progress)


def update_job_progress(job_id: Optional[str], message: str) -> None:
    if not job_id:
        return
    with _UPLOAD_JOBS_LOCK:
        job = _UPLOAD_JOBS.get(job_id)
        if not job:
            return
        job.progress = message
        job.updated_at = time.time()
    _db_update_upload_job(job_id, progress=message)


def mark_job_cancel_requested(job_id: str) -> None:
    with _UPLOAD_JOBS_LOCK:
        job = _UPLOAD_JOBS.get(job_id)
        if not job:
            return
        job.cancel_requested = True
        job.updated_at = time.time()
    _db_update_upload_job(job_id, cancel_requested=True)


def is_job_cancelled(job_id: Optional[str]) -> bool:
    if not job_id:
        return False
    with _UPLOAD_JOBS_LOCK:
        job = _UPLOAD_JOBS.get(job_id)
        return bool(job and job.cancel_requested)


def raise_if_job_cancelled(job_id: Optional[str]) -> None:
    if is_job_cancelled(job_id):
        raise UploadJobCancelled(f"upload job cancelled: {job_id}")


#############################################
# ここから PDF アップロードパイプライン本体
#############################################


def run_pdf_upload_pipeline_sync(
    *,
    contents: bytes,
    file_name: str,
    cleanse: bool,
    question_llm_model: str,
    answer_llm_model: str,
    generate_image_captions: bool,
    ocr_engine: str,
    ocr_quality: str = "balanced",
    ocr_image_compression: str = "balanced",
    job_id: Optional[str] = None,
) -> dict:
    """PDFアップロード処理パイプラインを同期的に実行するヘルパー関数。

    ここでは PDF→テキスト抽出（+クレンジング）までを行い、その結果のみを
    JSON/ストレージ/DB に保存する。QA生成やチャンク生成は別APIで行う。
    """

    from .main import jst_now_str  # 遅延インポートで循環依存を回避

    file_hash = hashlib.sha256(contents).hexdigest()
    warning_message = ""
    file_id = str(uuid.uuid4())

    update_job_progress(job_id, "PDFファイルを読み込み中です…")
    raise_if_job_cancelled(job_id)

    logger.info(
        "[%s][重要] upload_job パイプライン開始: file_name=%s, file_id=%s, size=%d",
        jst_now_str(),
        file_name,
        file_id,
        len(contents),
        extra={"file_id": file_id, "job_id": job_id, "component": "upload", "endpoint": "upload_pipeline"},
    )

    pdf_stream = io.BytesIO(contents)
    logger.info(
        "[重要] BytesIOストリーム作成完了: %dバイト",
        pdf_stream.getbuffer().nbytes,
        extra={"file_id": file_id, "job_id": job_id, "component": "upload", "endpoint": "upload_pipeline"},
    )

    try:
        update_job_progress(job_id, "PDFテキストを抽出中です（OCRエンジン選択中）…")
        text = ""
        normalized_engine = (ocr_engine or "").lower()
        ocr_quality = (ocr_quality or "balanced").lower()
        if ocr_quality not in {"fast", "balanced", "high"}:
            ocr_quality = "balanced"

        normalized_compression, resize_max, jpeg_quality = resolve_ocr_image_compression(
            ocr_image_compression,
        )
        ocr_image_compression = normalized_compression

        use_mlx_ocr = normalized_engine in {"mlx", "deepseek_mlx"}
        use_ollama_ocr = normalized_engine in {"ollama_deepseek", "deepseek_ocr", "deepseek-ocr"}
        if use_ollama_ocr:
            actual_ocr_engine = "ollama_deepseek"
        elif use_mlx_ocr:
            actual_ocr_engine = "mlx"
        else:
            actual_ocr_engine = "pypdf"

        if normalized_engine == "deepseek":
            warning_message += "DeepSeek OCR (PyTorch版) は廃止されました。MLX版スクリプトをご利用ください。\n"
            logger.warning("[警告] DeepSeek OCR (PyTorch) はサポート終了のためMLX版にフォールバックします。")
            use_mlx_ocr = True

        extracted_captions = ""
        mlx_image_captions: list[dict[str, str | int | None]] = []

        if use_ollama_ocr:
            raise_if_job_cancelled(job_id)
            update_job_progress(job_id, "Ollama DeepSeek OCR でPDFテキストを抽出中です…")
            if ocr_quality == "fast":
                quality_max_pages = 5
            elif ocr_quality == "high":
                quality_max_pages = None
            else:
                quality_max_pages = 20
            try:
                text = run_deepseek_ocr_via_ollama_for_pdf(
                    contents,
                    model=os.getenv("OLLAMA_DEEPSEEK_OCR_MODEL"),
                    prompt=os.getenv("OLLAMA_DEEPSEEK_OCR_PROMPT", DEFAULT_OCR_PROMPT),
                    max_pages=quality_max_pages,
                    dpi=int(os.getenv("OLLAMA_DEEPSEEK_DPI", "300")),
                    timeout=get_ollama_deepseek_timeout(600),
                    image_output_dir=IMAGES_DIR / file_id,
                    resize_max=resize_max,
                    jpeg_quality=jpeg_quality,
                )
            except Exception as e:  # noqa: BLE001
                warning_message += "Ollama DeepSeek OCR によるテキスト抽出に失敗したため、PyPDFベースにフォールバックします。\n"
                logger.warning(
                    "[警告] Ollama DeepSeek OCR失敗のためPyPDFへフォールバックします: %s",
                    e,
                    extra={"file_id": file_id, "job_id": job_id, "component": "upload", "endpoint": "upload_pipeline"},
                )
                use_ollama_ocr = False
                actual_ocr_engine = "pypdf"

        if not use_mlx_ocr and not use_ollama_ocr:
            raise_if_job_cancelled(job_id)
            update_job_progress(job_id, "PyPDF でテキストを抽出中です…")
            text = extract_pdf_text_layout(contents)
            auto_threshold = int(os.getenv("OCR_AUTO_MIN_CHARS", "200"))
            if len(text.strip()) < auto_threshold:
                logger.warning(
                    "[警告] PyPDF抽出が短いためMLX OCRへフォールバック: %d < %d",
                    len(text.strip()),
                    auto_threshold,
                    extra={"file_id": file_id, "job_id": job_id, "component": "upload", "endpoint": "upload_pipeline"},
                )
                use_mlx_ocr = True
            else:
                actual_ocr_engine = "pypdf"

        if use_mlx_ocr and not MLX_OCR_AVAILABLE:
            logger.warning(
                "[警告] MLX DeepSeek OCRモジュールが利用できないため、PyPDFベースにフォールバックします。",
                extra={"file_id": file_id, "job_id": job_id, "component": "upload", "endpoint": "upload_pipeline"},
            )
            use_mlx_ocr = False
            actual_ocr_engine = "pypdf"

        if use_mlx_ocr:
            raise_if_job_cancelled(job_id)
            update_job_progress(job_id, "MLX DeepSeek OCR でPDFテキストを抽出中です…")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                tmp_pdf.write(contents)
                tmp_pdf_path = Path(tmp_pdf.name)
            try:
                ocr_results = run_deepseek_ocr(
                    input_path=tmp_pdf_path,
                    model_path=os.getenv("MLX_DEEPSEEK_MODEL", "quocnguyen/DeepSeek-OCR-bf16-mlx"),
                    prompt=os.getenv("MLX_DEEPSEEK_PROMPT", DEFAULT_OCR_PROMPT),
                    describe_photos=bool(
                        os.getenv("MLX_DEEPSEEK_DESCRIBE_PHOTOS", "0")
                        not in {"0", "false", "False"}
                    ),
                    photo_prompt=os.getenv("MLX_DEEPSEEK_PHOTO_PROMPT", DEFAULT_PHOTO_PROMPT),
                    max_tokens=int(os.getenv("MLX_DEEPSEEK_MAX_TOKENS", "3000")),
                    temperature=float(os.getenv("MLX_DEEPSEEK_TEMPERATURE", "0.0")),
                    max_pages=None,
                    dpi=int(os.getenv("MLX_DEEPSEEK_DPI", "300")),
                    margin=int(os.getenv("MLX_DEEPSEEK_MARGIN", "16")),
                    contrast=float(os.getenv("MLX_DEEPSEEK_CONTRAST", "1.0")),
                    color_mode=os.getenv("MLX_DEEPSEEK_COLOR_MODE", "grayscale"),
                    image_format=os.getenv("MLX_DEEPSEEK_IMAGE_FORMAT", "png"),
                    binarize_threshold=(
                        lambda v: int(v) if v else None
                    )(os.getenv("MLX_DEEPSEEK_BINARIZE_THRESHOLD", "")),
                    sharpen=bool(
                        os.getenv("MLX_DEEPSEEK_SHARPEN", "0")
                        not in {"0", "false", "False"}
                    ),
                    ocr_min_length=int(os.getenv("MLX_DEEPSEEK_OCR_MIN_LENGTH", "20")),
                    fallback_text_crop_ratio=float(
                        os.getenv("MLX_DEEPSEEK_FALLBACK_TEXT_CROP_RATIO", "0.4")
                    ),
                    save_images_dir=None,
                    verbose=False,
                )
            finally:
                tmp_pdf_path.unlink(missing_ok=True)

            text_blocks: list[str] = []
            caption_lines: list[str] = []
            for page in ocr_results:
                raise_if_job_cancelled(job_id)
                if page.ocr.strip():
                    text_blocks.append(f"[{page.label}]\n{page.ocr.strip()}")
                if page.photo and page.photo.strip():
                    caption_lines.append(
                        f"- {page.label if page.page is None else f'p{page.page}'}: {page.photo.strip()}"
                    )
                    mlx_image_captions.append(
                        {
                            "source": "mlx_deepseek",
                            "page": page.page,
                            "label": page.label,
                            "caption": page.photo.strip(),
                        }
                    )
            if text_blocks:
                text = "\n\n".join(text_blocks).strip()
            else:
                text = ""
            extracted_captions = "\n".join(caption_lines)
            actual_ocr_engine = "mlx"

        if cleanse:
            raise_if_job_cancelled(job_id)
            update_job_progress(job_id, "抽出テキストをクレンジング処理中です…")
            logger.info(
                "[重要] クレンジング処理を実施します",
                extra={"file_id": file_id, "job_id": job_id, "component": "upload", "endpoint": "upload_pipeline"},
            )
            text = cleanse_pdf_text(text)

        sample_text = text[:3000] if len(text) > 3000 else text
        logger.info(
            "[重要] PDF抽出完了: 合計%d文字, サンプル=%s...",
            len(text),
            sample_text[:100],
            extra={"file_id": file_id, "job_id": job_id, "component": "upload", "endpoint": "upload_pipeline"},
        )
    except Exception as pdf_error:  # noqa: BLE001
        logger.error(
            "[重要] PDF処理エラー: %s",
            pdf_error,
            extra={"file_id": file_id, "job_id": job_id, "component": "upload", "endpoint": "upload_pipeline"},
        )
        raise RuntimeError(f"PDF処理エラー: {str(pdf_error)}") from pdf_error

    raise_if_job_cancelled(job_id)

    # --- ここからは抽出結果のみを保存するフェーズ（QA生成は別APIで実施） ---
    update_job_progress(job_id, "抽出結果をJSONとして保存中です…")
    extracted_path = EXTRACTED_DIR / f"{file_id}.json"
    with open(extracted_path, "w", encoding="utf-8") as f_json:
        json.dump(
            {
                "text": sample_text,
                "questions": [],
                "answers": [],
                "qa_meta": [],
                "file_name": file_name,
                "settings": {
                    "cleanse_used": bool(cleanse),
                    "generate_image_captions": bool(generate_image_captions),
                    "ocr_engine_selected": normalized_engine or "auto",
                    "ocr_engine_used": actual_ocr_engine,
                },
                "image_captions": [],
            },
            f_json,
            ensure_ascii=False,
        )

    raise_if_job_cancelled(job_id)
    update_job_progress(job_id, "PDFファイルをストレージに保存中です…")
    pdf_path = PDF_DIR / f"{file_id}.pdf"
    with open(pdf_path, "wb") as f_pdf:
        f_pdf.write(contents)

    raise_if_job_cancelled(job_id)
    update_job_progress(job_id, "データベースにPDF情報を保存中です…")
    persist_pdf_upload_to_db(
        file_id=file_id,
        file_name=file_name or f"{file_id}.pdf",
        original_name=file_name or f"{file_id}.pdf",
        file_size=len(contents),
        storage_path=str(pdf_path),
        cleanse_used=cleanse,
        question_llm_model=question_llm_model,
        answer_llm_model=answer_llm_model,
        chunks=[],
        questions=[],
        answers=[],
        qa_meta=[],
        file_hash=file_hash,
        ocr_engine_used=actual_ocr_engine,
        ocr_engine_selected=normalized_engine or "auto",
    )
    logger.info(
        "[%s][INFO] upload_job 抽出フェーズでの永続化処理が完了しました",
        jst_now_str(),
        extra={"file_id": file_id, "job_id": job_id},
    )

    update_job_progress(job_id, "PDF抽出フェーズが完了しました。")
    return {
        "file_id": file_id,
        "text": sample_text,
        "questions": [],
        "answers": [],
        "qa_meta": [],
        "file_name": file_name,
        "warning": warning_message,
        "ocr_engine_used": actual_ocr_engine,
        "ocr_engine_selected": normalized_engine or "auto",
    }


def start_upload_job(
    *,
    contents: bytes,
    file_name: str,
    cleanse: bool,
    question_llm_model: str,
    answer_llm_model: str,
    generate_image_captions: bool,
    ocr_engine: str,
    ocr_quality: str = "balanced",
    ocr_image_compression: str = "balanced",
) -> dict:
    """PDFアップロード処理をバックグラウンドジョブとして開始するサービス関数。"""

    job_id = str(uuid.uuid4())
    job = UploadJobState(
        job_id=job_id,
        status=UploadJobStatus.PENDING,
        progress="ジョブを受け付けました。キューに登録されています。",
    )
    set_upload_job(job)

    def worker() -> None:
        update_job_progress(job_id, "PDF処理パイプラインを開始しました…")
        with _UPLOAD_JOBS_LOCK:
            current = _UPLOAD_JOBS.get(job_id)
            if not current:
                return
            current.status = UploadJobStatus.RUNNING
            current.updated_at = time.time()
        _db_update_upload_job(
            job_id,
            status=UploadJobStatus.RUNNING,
            progress="PDF処理パイプラインを開始しました…",
        )
        try:
            result = run_pdf_upload_pipeline_sync(
                contents=contents,
                file_name=file_name,
                cleanse=cleanse,
                question_llm_model=question_llm_model,
                answer_llm_model=answer_llm_model,
                generate_image_captions=generate_image_captions,
                ocr_engine=ocr_engine,
                ocr_quality=ocr_quality,
                ocr_image_compression=ocr_image_compression,
                job_id=job_id,
            )
            with _UPLOAD_JOBS_LOCK:
                current = _UPLOAD_JOBS.get(job_id)
                if not current:
                    return
                if current.cancel_requested:
                    current.status = UploadJobStatus.CANCELLED
                    if not current.error:
                        current.error = "ユーザーによりキャンセルされました。"
                else:
                    current.status = UploadJobStatus.COMPLETED
                    current.result = result
                    current.error = None
                current.updated_at = time.time()
            file_id_inner = result.get("file_id") if isinstance(result, dict) else None
            _db_update_upload_job(
                job_id,
                status=current.status,
                progress=current.progress,
                error=current.error,
                result=result,
                file_id=file_id_inner,
            )
        except UploadJobCancelled as ce:  # noqa: BLE001
            with _UPLOAD_JOBS_LOCK:
                current = _UPLOAD_JOBS.get(job_id)
                if not current:
                    return
                current.status = UploadJobStatus.CANCELLED
                current.error = str(ce)
                current.updated_at = time.time()
            _db_update_upload_job(
                job_id,
                status=UploadJobStatus.CANCELLED,
                error=str(ce),
            )
        except Exception as e:  # noqa: BLE001
            with _UPLOAD_JOBS_LOCK:
                current = _UPLOAD_JOBS.get(job_id)
                if not current:
                    return
                current.status = UploadJobStatus.ERROR
                current.error = str(e)
                current.updated_at = time.time()
            _db_update_upload_job(
                job_id,
                status=UploadJobStatus.ERROR,
                error=str(e),
            )

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


def get_upload_job_status(job_id: str) -> dict:
    """メモリまたはDBからアップロードジョブの状態を取得する。"""

    job = get_upload_job(job_id)
    if job:
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "cancel_requested": job.cancel_requested,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    # メモリ上にジョブが無い場合は、DBテーブル(upload_jobs)から状態を取得
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT job_id, status, progress, cancel_requested, error,
                               result_json, created_at, updated_at
                        FROM upload_jobs
                        WHERE job_id = :job_id
                        """
                    ),
                    {"job_id": job_id},
                )
                .mappings()
                .first()
            )
        if not row:
            raise HTTPException(
                status_code=404, detail="指定されたジョブIDは存在しません。",
            )

        result_obj = None
        if row["result_json"]:
            try:
                result_obj = json.loads(row["result_json"])
            except Exception:  # noqa: BLE001
                result_obj = None

        created_ts = row["created_at"].timestamp() if row["created_at"] is not None else None
        updated_ts = row["updated_at"].timestamp() if row["updated_at"] is not None else None

        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "progress": row["progress"] or "",
            "cancel_requested": bool(row["cancel_requested"]),
            "result": result_obj,
            "error": row["error"],
            "created_at": created_ts,
            "updated_at": updated_ts,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        from .main import jst_now_str  # 遅延インポート

        logger.error("[%s][ERROR] upload_job_status DB参照中にエラー: %s", jst_now_str(), e)
        raise HTTPException(
            status_code=500, detail="ジョブ状態取得中にサーバエラーが発生しました。",
        ) from e


def cancel_upload_job(job_id: str) -> dict:
    """実行中のアップロードジョブにキャンセルフラグを立てる。"""

    job = get_upload_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="指定されたジョブIDは存在しません。")

    mark_job_cancel_requested(job_id)
    update_job_progress(
        job_id,
        "キャンセル要求を受け付けました。現在実行中のステップが完了し次第、中断されます。",
    )

    job = get_upload_job(job_id)
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "cancel_requested": job.cancel_requested,
    }


def get_extracted_data(file_id: str) -> dict:
    """指定file_idの抽出テキスト・QA・ファイル名・PDF本体を返すサービス関数。"""

    extracted_path = EXTRACTED_DIR / f"{file_id}.json"
    if not extracted_path.exists():
        raise HTTPException(status_code=404, detail=f"file_id={file_id}の抽出データが見つかりません")
    with open(extracted_path, "r", encoding="utf-8") as f_json:
        data = json.load(f_json)

    pdf_path = PDF_DIR / f"{file_id}.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f_pdf:
            data["pdf_bytes_base64"] = base64.b64encode(f_pdf.read()).decode("utf-8")

    if "file_name" not in data:
        data["file_name"] = f"{file_id}.pdf"
    return data


def generate_qa_for_existing_pdf(
    *,
    file_id: str,
    question_llm_model: str,
    answer_llm_model: str,
    initial_warning: str = "",
) -> dict:
    """抽出済みテキストを用いて既存PDFに対するQAを生成し、JSON/DBを更新するサービス関数。"""

    from . import main as main_module  # 遅延インポートで循環依存を回避

    init_generation_llm = main_module.init_generation_llm
    jst_now_str = main_module.jst_now_str

    try:
        extracted_path = EXTRACTED_DIR / f"{file_id}.json"
        if not extracted_path.exists():
            raise HTTPException(status_code=404, detail=f"file_id={file_id}の抽出データが見つかりません")
        with open(extracted_path, "r", encoding="utf-8") as f_json:
            data = json.load(f_json)

        text = data.get("text") or ""
        if not text.strip():
            raise HTTPException(status_code=400, detail="抽出テキストが空のためQAを生成できません。")

        settings = data.get("settings") or {}
        cleanse_used = bool(settings.get("cleanse_used", False))
        generate_image_captions = bool(settings.get("generate_image_captions", True))
        ocr_engine_selected = settings.get("ocr_engine_selected") or "auto"
        ocr_engine_used = settings.get("ocr_engine_used") or ocr_engine_selected
        file_name = data.get("file_name") or f"{file_id}.pdf"
        warning_message = (data.get("warning") or "") + initial_warning

        pdf_path = PDF_DIR / f"{file_id}.pdf"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"file_id={file_id}のPDF本体が見つかりません")
        with open(pdf_path, "rb") as f_pdf:
            contents = f_pdf.read()
        file_hash = hashlib.sha256(contents).hexdigest()

        sample_text = text[:3000] if len(text) > 3000 else text
        context_sentences = split_sentences(text)
        logger.info(
            "[%s][重要] generate_qa: 抽出テキスト長=%d  サンプル=%s...",
            jst_now_str(),
            len(text),
            sample_text[:100],
        )

        image_captions: list[dict] = []
        if generate_image_captions:
            try:
                image_captions = generate_image_captions_from_pdf(contents)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[警告] generate_qa: 画像キャプション生成中に例外: %s. キャプションなしで続行します。",
                    e,
                )

        extracted_captions = ""
        mlx_image_captions: list[dict] = []

        logger.info("[重要] generate_qa: LLM質問生成開始")
        llm_q_instance, resolved_question_llm = init_generation_llm(
            question_llm_model,
            purpose="question_generation",
            temperature=0.2,
            top_p=0.85,
            num_predict=320,
            max_tokens=320,
        )
        logger.info("[重要] generate_qa: 質問生成LLM=%s", resolved_question_llm)

        prompt_q = textwrap.dedent(
            f"""
            あなたはPDF文書の内容を確認する質問を生成する専門アシスタントです。
            以下の制約を守り、日本語で代表的な質問を5件作成してください。
            - 質問は具体的かつ本文に直接基づく内容にすること。
            - 文書に記載がない推測的な質問は避けること。
            - 質問は箇条書きではなく、1行の文章形式で記述すること。

            ### 文書抜粋
            {text[:1800]}

            ### 出力形式
            質問のみを1行ずつ列挙してください。
            """
        ).strip()

        try:
            questions_resp = llm_q_instance.invoke(prompt_q)
            raw_questions_text = _extract_answer_text(questions_resp)
            logger.info("[重要] generate_qa: LLM質問生成レスポンス長=%d", len(raw_questions_text))
            questions = [q.strip() for q in raw_questions_text.split("\n") if q.strip()]
            logger.info("[重要] generate_qa: 質問リスト生成完了 件数=%d", len(questions))
        except Exception as e:  # noqa: BLE001
            logger.error("[重要] generate_qa: LLM質問生成例外: %s", e)
            questions = []

        answers: list[str] = []
        llm_a_instance = None
        resolved_answer_llm: Optional[str] = None

        if not questions:
            logger.info("[重要] generate_qa: 正規表現によるQA/箇条書き抽出開始")
            bullets = re.findall(r"^[\*\-\d\.]+\s*(.+)", text, re.MULTILINE)
            qas = re.findall(r"Q[\d：: ]*(.+?)\nA[\d：: ]*(.+?)(?=\nQ|\n\Z)", text, re.DOTALL)
            if qas:
                questions = [q.strip() for q, a in qas]
                answers = [a.strip() for q, a in qas]
            elif bullets:
                questions = bullets[:5]
                answers = ["該当内容を本文から要約してください。"] * len(questions)
            else:
                paras = [p.strip() for p in text.split("\n") if p.strip()]
                questions = [f"{p[:20]}について説明してください。" for p in paras[:5]]
                answers = ["該当内容を本文から要約してください。"] * len(questions)
            resolved_answer_llm = resolved_question_llm
        else:
            llm_a_instance, resolved_answer_llm = init_generation_llm(
                answer_llm_model,
                purpose="answer_generation",
                temperature=0.25,
                top_p=0.85,
                num_predict=640,
                max_tokens=640,
            )
            logger.info("[重要] generate_qa: 回答生成LLM=%s", resolved_answer_llm)

            for i, q in enumerate(questions):
                try:
                    prompt_a = build_rag_answer_prompt(sample_text, q)
                    answer_resp = llm_a_instance.invoke(prompt_a, max_tokens=640)
                    normalized_answer = _extract_answer_text(answer_resp)
                    answer = normalized_answer.strip().split("\n")[0]
                    if answer and answer[-1] not in {"。", "！", "？", ".", "!", "?"}:
                        answer = f"{answer}。"
                    logger.info("[重要] generate_qa: LLM回答%d生成完了 文字数=%d", i + 1, len(answer))
                    answers.append(answer)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    logger.error("[重要] generate_qa: LLM回答%d生成例外: %s", i + 1, e)
                    traceback.print_exc()
                    answers.append("該当内容を本文から要約してください。")

        if len(questions) < 5:
            logger.warning("[警告] generate_qa: 質問数が不足 (%d件)。フォールバックで補完します。", len(questions))
            fallback_needed = 5 - len(questions)
            paras = [p.strip() for p in text.split("\n") if p.strip()]
            fallback_questions: list[str] = []
            for para in paras:
                candidate = f"{para[:20]}について説明してください。"
                if candidate not in questions and candidate not in fallback_questions:
                    fallback_questions.append(candidate)
                if len(fallback_questions) >= fallback_needed:
                    break
            while len(fallback_questions) < fallback_needed:
                fallback_questions.append("本文の主要な論点について説明してください。")
            logger.info("[重要] generate_qa: フォールバック質問を追加 %s", fallback_questions)
            for fallback_q in fallback_questions:
                questions.append(fallback_q)
                if llm_a_instance is not None:
                    try:
                        prompt_a = build_rag_answer_prompt(sample_text, fallback_q)
                        answer_resp = llm_a_instance.invoke(prompt_a, max_tokens=640)
                        normalized_answer = _extract_answer_text(answer_resp)
                        fallback_answer = normalized_answer.strip().split("\n")[0]
                        if fallback_answer and fallback_answer[-1] not in {"。", "！", "？", ".", "!", "?"}:
                            fallback_answer = f"{fallback_answer}。"
                    except Exception as e:  # noqa: BLE001
                        import traceback

                        logger.error("[重要] generate_qa: フォールバック質問の回答生成例外: %s", e)
                        traceback.print_exc()
                        fallback_answer = "本文を要約してください。"
                else:
                    fallback_answer = "本文を要約してください。"
                answers.append(fallback_answer)

        if not questions or not answers:
            logger.warning("[重要] generate_qa: ダミーQAセットを返却（questions/answersが空）")
            questions = ["この文書の主題は何ですか？"]
            answers = ["本文を要約してください。"]
            if resolved_answer_llm is None:
                resolved_answer_llm = resolved_question_llm

        qa_meta: list[dict] = []
        try:
            for idx, (question, answer) in enumerate(zip(questions, answers)):
                context_snippet = extract_relevant_context(question, context_sentences, max_sentences=6)
                context_lines = context_snippet.splitlines()
                quality = evaluate_answer_quality(answer, context_lines)
                retry_count = 0
                regenerated_answer = answer

                if quality.get("needs_retry") and llm_a_instance is not None:
                    refined = regenerate_answer_with_context(
                        question,
                        context_snippet or sample_text,
                        llm_a_instance,
                        max_tokens=640,
                    )
                    refined_quality = evaluate_answer_quality(refined, context_lines)
                    if refined_quality.get("score", 0.0) >= quality.get("score", 0.0):
                        regenerated_answer = refined
                        quality = refined_quality
                        retry_count = 1
                        answers[idx] = regenerated_answer

                qa_meta.append(
                    {
                        "score": quality.get("score", 0.0),
                        "is_auto_fixed": retry_count > 0,
                        "is_dummy_answer": quality.get("is_dummy", False),
                        "quality": quality,
                        "context_snippet": context_snippet,
                        "retry_count": retry_count,
                        "candidates": [regenerated_answer],
                        "candidate_scores": [quality.get("score", 0.0)],
                    }
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("[警告] generate_qa: qa_meta生成時に例外: %s。全件デフォルト値を設定します", e)
            qa_meta = [
                {
                    "score": 1.0,
                    "is_auto_fixed": False,
                    "is_dummy_answer": False,
                    "quality": {"score": 1.0, "is_dummy": False, "needs_retry": False},
                    "context_snippet": "",
                    "retry_count": 0,
                    "candidates": [a],
                    "candidate_scores": [1.0],
                }
                for a in answers
            ]

        combined_captions: list[str] = []
        if extracted_captions:
            combined_captions.append(extracted_captions)
        if image_captions:
            combined_captions.append(
                "\n".join([f"- p{c.get('page')}: {c.get('caption', '')}" for c in image_captions])
            )

        if combined_captions:
            captions_text = "\n".join([block for block in combined_captions if block.strip()])
            combined_text = text + "\n\n【画像キャプション】\n" + captions_text
        else:
            combined_text = text

        chunks_for_storage = generate_default_chunks_for_storage(combined_text)
        logger.info("[重要] generate_qa: チャンク数=%d", len(chunks_for_storage))

        with open(extracted_path, "w", encoding="utf-8") as f_json:
            json.dump(
                {
                    "text": sample_text,
                    "questions": questions,
                    "answers": answers,
                    "qa_meta": qa_meta,
                    "file_name": file_name,
                    "settings": {
                        "cleanse_used": bool(cleanse_used),
                        "generate_image_captions": bool(generate_image_captions),
                        "ocr_engine_selected": ocr_engine_selected,
                        "ocr_engine_used": ocr_engine_used,
                    },
                    "image_captions": image_captions,
                },
                f_json,
                ensure_ascii=False,
            )

        persist_pdf_upload_to_db(
            file_id=file_id,
            file_name=file_name,
            original_name=file_name,
            file_size=len(contents),
            storage_path=str(pdf_path),
            cleanse_used=cleanse_used,
            question_llm_model=resolved_question_llm,
            answer_llm_model=resolved_answer_llm or resolved_question_llm,
            chunks=chunks_for_storage,
            questions=questions,
            answers=answers,
            qa_meta=qa_meta,
            file_hash=file_hash,
            ocr_engine_used=ocr_engine_used,
            ocr_engine_selected=ocr_engine_selected,
        )
        logger.info(
            "[%s][INFO] generate_qa: 永続化処理が完了しました (file_id=%s)",
            jst_now_str(),
            file_id,
            extra={"file_id": file_id},
        )

        return {
            "file_id": file_id,
            "text": sample_text,
            "questions": questions,
            "answers": answers,
            "qa_meta": qa_meta,
            "file_name": file_name,
            "warning": warning_message,
            "ocr_engine_used": ocr_engine_used,
            "ocr_engine_selected": ocr_engine_selected,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback

        logger.error(
            "[%s][重要] generate_qa全体例外: %s",
            jst_now_str(),
            e,
            extra={"file_id": file_id},
        )
        logger.debug(traceback.format_exc(), extra={"file_id": file_id})
        raise HTTPException(status_code=500, detail=str(e)) from e


def upload_and_generate_qa(
    *,
    contents: bytes,
    file_name: str,
    cleanse: bool,
    question_llm_model: str,
    answer_llm_model: str,
    generate_image_captions: bool,
    ocr_engine: str,
    ocr_image_compression: str = "balanced",
) -> dict:
    """同期API用: アップロードとQA生成を一括で行う高レベルサービス。"""

    extract_result = run_pdf_upload_pipeline_sync(
        contents=contents,
        file_name=file_name,
        cleanse=cleanse,
        question_llm_model=question_llm_model,
        answer_llm_model=answer_llm_model,
        generate_image_captions=generate_image_captions,
        ocr_engine=ocr_engine,
        ocr_quality="balanced",
        ocr_image_compression=ocr_image_compression,
        job_id=None,
    )
    file_id = extract_result.get("file_id")
    if not file_id:
        raise HTTPException(status_code=500, detail="PDF抽出に失敗しました。file_id が取得できませんでした。")

    initial_warning = str(extract_result.get("warning") or "")
    return generate_qa_for_existing_pdf(
        file_id=file_id,
        question_llm_model=question_llm_model,
        answer_llm_model=answer_llm_model,
        initial_warning=initial_warning,
    )
