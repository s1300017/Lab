from datetime import datetime
from pytz import timezone
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import hashlib
import time
from enum import Enum
from dataclasses import dataclass, field


def jst_now_str():
    return datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')


from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import threading

# --- PDF・抽出データ保存用ディレクトリのグローバル定義 ---
import uuid
import json
import html
import re
import textwrap
from difflib import SequenceMatcher
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PDF_DIR = DATA_DIR / "pdf"
EXTRACTED_DIR = DATA_DIR / "extracted"
IMAGES_DIR = DATA_DIR / "images"
PDF_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print(f"[{jst_now_str()}] === FastAPI main.py 起動開始 [テスト用] ===")

# データベース接続設定
POSTGRES_DB = os.environ.get("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "rag_password")
DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@db:5432/{POSTGRES_DB}"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPIアプリケーションの初期化
app = FastAPI()


# --- Ollama接続先の推論モード管理（mac_local / windows_gpu） ---
_ALLOWED_INFERENCE_MODES = {"mac_local", "windows_gpu"}
_INFERENCE_MODE_LOCK = threading.Lock()
_INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "mac_local")
if _INFERENCE_MODE not in _ALLOWED_INFERENCE_MODES:
    _INFERENCE_MODE = "mac_local"


def get_inference_mode() -> str:
    """現在の推論モードを取得する（スレッドセーフ）。"""
    with _INFERENCE_MODE_LOCK:
        return _INFERENCE_MODE


def set_inference_mode(mode: str) -> None:
    """推論モードを更新する（スレッドセーフ）。"""
    if mode not in _ALLOWED_INFERENCE_MODES:
        raise ValueError(f"未対応の推論モードです: {mode}")
    global _INFERENCE_MODE
    with _INFERENCE_MODE_LOCK:
        _INFERENCE_MODE = mode


def get_ollama_base_url() -> str:
    """推論モードに応じてOllamaのベースURLを返すヘルパー。"""
    mode = get_inference_mode()
    mac_url = os.environ.get("OLLAMA_BASE_URL_MAC") or os.environ.get("OLLAMA_BASE_URL") or "http://ollama:11434"
    if mode == "windows_gpu":
        windows_url = os.environ.get("OLLAMA_BASE_URL_WINDOWS")
        if windows_url:
            return windows_url
    return mac_url


class UploadJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


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
    """PDFアップロードジョブの状態を upload_jobs テーブルに登録する。"""
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
    except Exception as e:
        print(f"[{jst_now_str()}][ERROR] upload_jobs insert/update失敗: {e}")


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
    """upload_jobs テーブルの指定ジョブを更新する。"""
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

    # updated_at は常に現在時刻で上書き
    set_clause = ", ".join(fields) + ", updated_at = CURRENT_TIMESTAMP"
    sql = f"UPDATE upload_jobs SET {set_clause} WHERE job_id = :job_id"
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), params)
    except Exception as e:
        print(f"[{jst_now_str()}][ERROR] upload_jobs更新失敗: {e}")


class UploadJobCancelled(Exception):
    pass


def _get_upload_job(job_id: str) -> Optional[UploadJobState]:
    with _UPLOAD_JOBS_LOCK:
        return _UPLOAD_JOBS.get(job_id)


def _set_upload_job(job: UploadJobState) -> None:
    with _UPLOAD_JOBS_LOCK:
        _UPLOAD_JOBS[job.job_id] = job
    # メモリ上の状態とあわせて DB 側にも登録
    _db_create_upload_job(job.job_id, job.status, job.progress)


def _update_job_progress(job_id: Optional[str], message: str) -> None:
    if not job_id:
        return
    with _UPLOAD_JOBS_LOCK:
        job = _UPLOAD_JOBS.get(job_id)
        if not job:
            return
        job.progress = message
        job.updated_at = time.time()
    # 進捗を DB 側にも反映
    _db_update_upload_job(job_id, progress=message)


def _mark_job_cancel_requested(job_id: str) -> None:
    with _UPLOAD_JOBS_LOCK:
        job = _UPLOAD_JOBS.get(job_id)
        if not job:
            return
        job.cancel_requested = True
        job.updated_at = time.time()
    # キャンセルフラグを DB 側にも反映
    _db_update_upload_job(job_id, cancel_requested=True)


def _is_job_cancelled(job_id: Optional[str]) -> bool:
    if not job_id:
        return False
    with _UPLOAD_JOBS_LOCK:
        job = _UPLOAD_JOBS.get(job_id)
        return bool(job and job.cancel_requested)


def _raise_if_job_cancelled(job_id: Optional[str]) -> None:
    if _is_job_cancelled(job_id):
        raise UploadJobCancelled(f"upload job cancelled: {job_id}")

# --- Dockerヘルスチェック用エンドポイント ---
@app.get("/health")
def health_check():
    """Docker用のシンプルなヘルスチェックAPI"""
    return {"status": "ok"}

# サーバ起動時にデータベースを初期化
@app.on_event("startup")
async def startup_event():
    print(f"[{jst_now_str()}] [DEBUG] startup_event呼び出し")
    print(f"[{jst_now_str()}] [DEBUG] DB_URL = {os.getenv('DATABASE_URL')}")
    
    # データベース接続をテスト
    max_retries = 5
    retry_delay = 5  # 秒
    
    for attempt in range(max_retries):
        try:
            print(f"[{jst_now_str()}] [DEBUG] データベース接続を試行中... (試行 {attempt + 1}/{max_retries})")
            init_db()
            print(f"[{jst_now_str()}] [DEBUG] データベース初期化に成功しました")
            break
        except Exception as e:
            print(f"[{jst_now_str()}] [ERROR] データベース初期化エラー (試行 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"[{jst_now_str()}] [CRITICAL] データベース初期化に失敗しました。最大試行回数に達しました。")
                raise
            import time
            time.sleep(retry_delay)

    # --- 起動時にOllamaモデルをバックグラウンドでウォームアップ（任意）---
    try:
        def _warmup():
            try:
                import urllib.request, urllib.error
                import json as _json
                base_url = get_ollama_base_url()
                targets = [
                    {"model": "mistral:latest", "prompt": "ping", "stream": False},
                    {"model": "llama3:latest", "prompt": "ping", "stream": False},
                    {"model": "gpt-oss:20b", "prompt": "ping", "stream": False},
                ]
                for body in targets:
                    try:
                        req = urllib.request.Request(
                            url=f"{base_url.rstrip('/')}/api/generate",
                            data=_json.dumps(body).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            _ = resp.read()
                        print(f"[ウォームアップ] {body['model']} 成功")
                    except Exception as we:
                        print(f"[ウォームアップ警告] {body.get('model')} 失敗: {we}")
            except Exception as e:
                print(f"[ウォームアップ初期化失敗] {e}")

        threading.Thread(target=_warmup, daemon=True).start()
    except Exception as e:
        print(f"[ウォームアップ起動失敗] {e}")

def init_db():
    print(f"[{jst_now_str()}] [DEBUG] init_db呼び出し")
    try:
        print(f"[{jst_now_str()}] [DEBUG] データベース接続テスト開始")
        with engine.connect() as conn:
            print(f"[{jst_now_str()}] [DEBUG] データベース接続成功")
            
            # トランザクションを開始
            with conn.begin():
                # テーブルが存在するか確認
                result = conn.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'embeddings');"
                ))
                table_exists = result.scalar()
                
                if not table_exists:
                    print(f"[{jst_now_str()}] [INFO] embeddingsテーブルを作成します")
                    conn.execute(text("""
                        CREATE TABLE embeddings (
                            id SERIAL PRIMARY KEY,
                            text TEXT NOT NULL,
                            embedding_model TEXT NOT NULL,
                            chunk_strategy TEXT NOT NULL,
                            chunk_size INTEGER,
                            chunk_overlap INTEGER,
                            avg_chunk_len FLOAT,
                            num_chunks INTEGER,
                            overall_score FLOAT,
                            faithfulness FLOAT,
                            answer_relevancy FLOAT,
                            context_recall FLOAT,
                            context_precision FLOAT,
                            answer_correctness FLOAT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """))
                    print(f"[{jst_now_str()}] [INFO] embeddingsテーブルを作成しました")
                else:
                    print(f"[{jst_now_str()}] [INFO] embeddingsテーブルは既に存在します")

                # --- PDF履歴用テーブル作成（存在しない場合のみ実行） ---
                print(f"[{jst_now_str()}] [INFO] PDF・評価履歴用テーブルを確認します")
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS pdf_files (
                        id TEXT PRIMARY KEY,
                        file_name TEXT,
                        original_name TEXT,
                        file_size BIGINT,
                        storage_path TEXT,
                        cleanse_used BOOLEAN,
                        question_llm_model TEXT,
                        answer_llm_model TEXT,
                        file_hash TEXT,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    ALTER TABLE pdf_files
                    ADD COLUMN IF NOT EXISTS file_hash TEXT
                """))
                conn.execute(text("""
                    ALTER TABLE pdf_files
                    ADD COLUMN IF NOT EXISTS ocr_engine_used TEXT
                """))
                conn.execute(text("""
                    ALTER TABLE pdf_files
                    ADD COLUMN IF NOT EXISTS ocr_engine_selected TEXT
                """))
                conn.execute(text("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_pdf_files_file_hash
                    ON pdf_files(file_hash)
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS pdf_chunks (
                        id SERIAL PRIMARY KEY,
                        pdf_file_id TEXT REFERENCES pdf_files(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        content_hash TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))

                # アップロードジョブ状態管理テーブル（upload_jobs）を作成
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS upload_jobs (
                        job_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        progress TEXT,
                        cancel_requested BOOLEAN DEFAULT FALSE,
                        error TEXT,
                        result_json TEXT,
                        file_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS generated_questions (
                        id SERIAL PRIMARY KEY,
                        pdf_file_id TEXT REFERENCES pdf_files(id) ON DELETE CASCADE,
                        question TEXT NOT NULL,
                        answer TEXT,
                        question_model TEXT,
                        answer_model TEXT,
                        meta_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        id SERIAL PRIMARY KEY,
                        pdf_file_id TEXT REFERENCES pdf_files(id) ON DELETE SET NULL,
                        experiment_name TEXT,
                        parameters TEXT,
                        status TEXT,
                        total_combinations INTEGER,
                        completed_combinations INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS experiment_results (
                        id SERIAL PRIMARY KEY,
                        experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
                        embedding_model TEXT,
                        chunk_strategy TEXT,
                        chunk_size INTEGER,
                        chunk_overlap INTEGER,
                        num_chunks INTEGER,
                        avg_chunk_len INTEGER,
                        overall_score FLOAT,
                        faithfulness FLOAT,
                        answer_relevancy FLOAT,
                        context_recall FLOAT,
                        context_precision FLOAT,
                        answer_correctness FLOAT,
                        answer_similarity FLOAT,
                        details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_logs (
                        id SERIAL PRIMARY KEY,
                        pdf_file_id TEXT REFERENCES pdf_files(id) ON DELETE SET NULL,
                        user_message TEXT NOT NULL,
                        assistant_message TEXT NOT NULL,
                        llm_model_used TEXT,
                        embedding_model TEXT,
                        scope TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))

                # モデル選択状態を永続化するためのテーブル
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS model_selection (
                        id INTEGER PRIMARY KEY,
                        llm_model TEXT,
                        embedding_model TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))

                # コミットは自動的に行われる
                
    except Exception as e:
        print(f"[{jst_now_str()}] [ERROR] データベース初期化エラー: {str(e)}")
        # エラーの詳細をログに出力
        import traceback
        print(f"[{jst_now_str()}] [ERROR] スタックトレース:\n{traceback.format_exc()}")
        raise
    for route in app.routes:
        print(f"[{jst_now_str()}] [ROUTE]", route.path, route.methods)

# サーバ起動時にルート一覧を出力
import threading

def print_routes():
    import time
    time.sleep(2)  # サーバ起動待ち
    print(f"[{jst_now_str()}] === FastAPI登録ルート一覧 ===")
    for route in app.routes:
        print(f"[{jst_now_str()}] [ROUTE]", route.path, route.methods)
threading.Thread(target=print_routes, daemon=True).start()

# --- PDFアップロード＆QA自動生成API ---
from fastapi import UploadFile, File

try:
    from pypdf import PdfReader  # layoutモード対応版
    HAS_PYPDF_LAYOUT = True
except ImportError:  # フォールバック: 旧PyPDF2
    from PyPDF2 import PdfReader
    HAS_PYPDF_LAYOUT = False


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
            print(
                f"[警告] layout抽出に失敗 (page={page_index}). "
                f"errors={extract_errors if extract_errors else 'なし'}"
            )
            text = ""

        pages_text.append(text)

    joined = "\n\n".join(pages_text)
    if HAS_PYPDF_LAYOUT:
        print(f"[重要] PyPDF layoutモードで抽出成功: {len(joined)}文字")
    else:
        print(f"[警告] PyPDF layout未対応バージョン。標準抽出結果: {len(joined)}文字")
    return joined


# --- PDF削除API ---
@app.delete("/pdf/{file_id}")
def delete_pdf(file_id: str):
    """指定されたPDFのDBレコードとファイルを削除する。"""
    print(f"[{jst_now_str()}][INFO] delete_pdf呼び出し: file_id={file_id}")
    try:
        with engine.begin() as conn:
            pdf_row = conn.execute(
                text("SELECT storage_path FROM pdf_files WHERE id = :id"),
                {"id": file_id}
            ).fetchone()
            if not pdf_row:
                raise HTTPException(status_code=404, detail="指定されたPDFは存在しません。")

            conn.execute(text("DELETE FROM pdf_chunks WHERE pdf_file_id = :id"), {"id": file_id})
            conn.execute(text("DELETE FROM generated_questions WHERE pdf_file_id = :id"), {"id": file_id})
            conn.execute(text("DELETE FROM pdf_files WHERE id = :id"), {"id": file_id})

        storage_path = pdf_row.storage_path
        try:
            if storage_path and Path(storage_path).exists():
                Path(storage_path).unlink()
                print(f"[{jst_now_str()}][INFO] ストレージPDF削除完了: {storage_path}")
        except Exception as file_err:
            print(f"[{jst_now_str()}][警告] ストレージPDF削除失敗: {file_err}")

        extracted_path = EXTRACTED_DIR / f"{file_id}.json"
        if extracted_path.exists():
            try:
                extracted_path.unlink()
                print(f"[{jst_now_str()}][INFO] 抽出データ削除完了: {extracted_path}")
            except Exception as json_err:
                print(f"[{jst_now_str()}][警告] 抽出データ削除失敗: {json_err}")

        # 画像化したPDFページのディレクトリも削除
        images_dir = IMAGES_DIR / file_id
        if images_dir.exists():
            try:
                import shutil
                shutil.rmtree(images_dir)
                print(f"[{jst_now_str()}][INFO] 画像ディレクトリ削除完了: {images_dir}")
            except Exception as img_err:
                print(f"[{jst_now_str()}][警告] 画像ディレクトリ削除失敗: {img_err}")

        return {"status": "deleted", "file_id": file_id}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[{jst_now_str()}][警告] delete_pdfで例外: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="PDF削除中にエラーが発生しました。")

def _run_pdf_upload_pipeline_sync(
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
    import io

    file_hash = hashlib.sha256(contents).hexdigest()
    warning_message = ""
    file_id = str(uuid.uuid4())

    _update_job_progress(job_id, "PDFファイルを読み込み中です…")
    _raise_if_job_cancelled(job_id)

    print(
        f"[{jst_now_str()}][重要] upload_job パイプライン開始: "
        f"file_name={file_name}, file_id={file_id}, size={len(contents)}"
    )

    pdf_stream = io.BytesIO(contents)
    print(f"[重要] BytesIOストリーム作成完了: {pdf_stream.getbuffer().nbytes}バイト")

    try:
        _update_job_progress(job_id, "PDFテキストを抽出中です（OCRエンジン選択中）…")
        text = ""
        normalized_engine = (ocr_engine or "").lower()
        ocr_quality = (ocr_quality or "balanced").lower()
        if ocr_quality not in {"fast", "balanced", "high"}:
            ocr_quality = "balanced"

        ocr_image_compression = (ocr_image_compression or "balanced").lower()
        if ocr_image_compression not in {"light", "balanced", "high"}:
            ocr_image_compression = "balanced"

        if ocr_image_compression == "light":
            resize_max = int(os.getenv("OLLAMA_DEEPSEEK_IMG_MAX_LIGHT", "1200"))
            jpeg_quality = int(os.getenv("OLLAMA_DEEPSEEK_JPEG_Q_LIGHT", "70"))
        elif ocr_image_compression == "high":
            resize_max = int(os.getenv("OLLAMA_DEEPSEEK_IMG_MAX_HIGH", "2048"))
            jpeg_quality = int(os.getenv("OLLAMA_DEEPSEEK_JPEG_Q_HIGH", "92"))
        else:
            resize_max = int(os.getenv("OLLAMA_DEEPSEEK_IMG_MAX_BALANCED", "1600"))
            jpeg_quality = int(os.getenv("OLLAMA_DEEPSEEK_JPEG_Q_BALANCED", "85"))
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
            print("[警告] DeepSeek OCR (PyTorch) はサポート終了のためMLX版にフォールバックします。")
            use_mlx_ocr = True

        extracted_captions = ""
        mlx_image_captions: list[dict[str, str | int | None]] = []

        if use_ollama_ocr:
            _raise_if_job_cancelled(job_id)
            _update_job_progress(job_id, "Ollama DeepSeek OCR でPDFテキストを抽出中です…")
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
                    timeout=int(os.getenv("OLLAMA_DEEPSEEK_TIMEOUT", "600")),
                    image_output_dir=IMAGES_DIR / file_id,
                    resize_max=resize_max,
                    jpeg_quality=jpeg_quality,
                )
            except Exception as e:
                warning_message += "Ollama DeepSeek OCR によるテキスト抽出に失敗したため、PyPDFベースにフォールバックします。\n"
                print(f"[警告] Ollama DeepSeek OCR失敗のためPyPDFへフォールバックします: {e}")
                use_ollama_ocr = False
                actual_ocr_engine = "pypdf"

        if not use_mlx_ocr and not use_ollama_ocr:
            _raise_if_job_cancelled(job_id)
            _update_job_progress(job_id, "PyPDF でテキストを抽出中です…")
            text = extract_pdf_text_layout(contents)
            auto_threshold = int(os.getenv("OCR_AUTO_MIN_CHARS", "200"))
            if len(text.strip()) < auto_threshold:
                print(
                    f"[警告] PyPDF抽出が短いためMLX OCRへフォールバック: "
                    f"{len(text.strip())} < {auto_threshold}"
                )
                use_mlx_ocr = True
            else:
                actual_ocr_engine = "pypdf"

        if use_mlx_ocr:
            _raise_if_job_cancelled(job_id)
            _update_job_progress(job_id, "MLX DeepSeek OCR でPDFテキストを抽出中です…")
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
                _raise_if_job_cancelled(job_id)
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
            _raise_if_job_cancelled(job_id)
            _update_job_progress(job_id, "抽出テキストをクレンジング処理中です…")
            print("[重要] クレンジング処理を実施します")
            text = cleanse_pdf_text(text)

        sample_text = text[:3000] if len(text) > 3000 else text
        print(
            f"[重要] PDF抽出完了: 合計{len(text)}文字, サンプル={sample_text[:100]}..."
        )
    except Exception as pdf_error:
        print(f"[重要] PDF処理エラー: {pdf_error}")
        raise RuntimeError(f"PDF処理エラー: {str(pdf_error)}")

    _raise_if_job_cancelled(job_id)

    # --- ここからは抽出結果のみを保存するフェーズ（QA生成は別APIで実施） ---
    _update_job_progress(job_id, "抽出結果をJSONとして保存中です…")
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

    _raise_if_job_cancelled(job_id)
    _update_job_progress(job_id, "PDFファイルをストレージに保存中です…")
    pdf_path = PDF_DIR / f"{file_id}.pdf"
    with open(pdf_path, "wb") as f_pdf:
        f_pdf.write(contents)

    _raise_if_job_cancelled(job_id)
    _update_job_progress(job_id, "データベースにPDF情報を保存中です…")
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
    print(f"[{jst_now_str()}][INFO] upload_job 抽出フェーズでの永続化処理が完了しました")

    _update_job_progress(job_id, "PDF抽出フェーズが完了しました。")
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


@app.post("/upload_job/start")
async def upload_job_start(
    file: UploadFile = File(...),
    cleanse: bool = Form(False),
    question_llm_model: str = Form("mistral"),
    answer_llm_model: str = Form("mistral"),
    generate_image_captions: bool = Form(True),
    ocr_engine: str = Form("auto"),
    ocr_quality: str = Form("balanced"),
    ocr_image_compression: str = Form("balanced"),
):
    """PDFアップロード処理をバックグラウンドジョブとして開始するAPI。"""
    contents = await file.read()
    file_name = file.filename or "uploaded.pdf"

    job_id = str(uuid.uuid4())
    job = UploadJobState(
        job_id=job_id,
        status=UploadJobStatus.PENDING,
        progress="ジョブを受け付けました。キューに登録されています。",
    )
    _set_upload_job(job)

    def worker() -> None:
        _update_job_progress(job_id, "PDF処理パイプラインを開始しました…")
        with _UPLOAD_JOBS_LOCK:
            current = _UPLOAD_JOBS.get(job_id)
            if not current:
                return
            current.status = UploadJobStatus.RUNNING
            current.updated_at = time.time()
        # DB側にもRUNNING状態を反映
        _db_update_upload_job(
            job_id,
            status=UploadJobStatus.RUNNING,
            progress="PDF処理パイプラインを開始しました…",
        )
        try:
            result = _run_pdf_upload_pipeline_sync(
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
            # DB側の状態も最終ステータスに更新
            file_id = result.get("file_id") if isinstance(result, dict) else None
            _db_update_upload_job(
                job_id,
                status=current.status,
                progress=current.progress,
                error=current.error,
                result=result,
                file_id=file_id,
            )
        except UploadJobCancelled as ce:
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
        except Exception as e:
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


@app.get("/upload_job/status/{job_id}")
def upload_job_status(job_id: str):
    """アップロードジョブの状態を返すAPI。"""
    job = _get_upload_job(job_id)
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

    # メモリ上にジョブが無い場合は、DBテーブル(upload_jobs)から状態を取得して返す
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
                status_code=404, detail="指定されたジョブIDは存在しません。"
            )

        result_obj = None
        if row["result_json"]:
            try:
                result_obj = json.loads(row["result_json"])
            except Exception:
                result_obj = None

        created_ts = (
            row["created_at"].timestamp() if row["created_at"] is not None else None
        )
        updated_ts = (
            row["updated_at"].timestamp() if row["updated_at"] is not None else None
        )

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
    except Exception as e:
        print(f"[{jst_now_str()}][ERROR] upload_job_status DB参照中にエラー: {e}")
        raise HTTPException(
            status_code=500, detail="ジョブ状態取得中にサーバエラーが発生しました。"
        )


@app.post("/upload_job/cancel/{job_id}")
def upload_job_cancel(job_id: str):
    """実行中のアップロードジョブにキャンセルフラグを立てるAPI。"""
    job = _get_upload_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="指定されたジョブIDは存在しません。")

    _mark_job_cancel_requested(job_id)
    _update_job_progress(
        job_id,
        "キャンセル要求を受け付けました。現在実行中のステップが完了し次第、中断されます。",
    )

    job = _get_upload_job(job_id)
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "cancel_requested": job.cancel_requested,
    }


@app.post("/uploadfile/")
async def uploadfile(
    file: UploadFile = File(...),
    cleanse: bool = Form(False),
    question_llm_model: str = Form("mistral"),
    answer_llm_model: str = Form("mistral"),
    generate_image_captions: bool = Form(True),
    ocr_engine: str = Form("auto"),
    ocr_image_compression: str = Form("balanced"),
):
    """
    PDFアップロード時にテキスト抽出→LLMで質問自動生成→LLMで回答自動生成まで行い、
    質問・回答セットを返すAPI。
    """
    print(f"[{jst_now_str()}][重要] uploadfile関数実行開始: ファイル名={file.filename}, サイズ={getattr(file, 'size', '不明')}")
    print(f"[{jst_now_str()}][重要] ファイル情報: {file=}, タイプ={type(file)}")
    import io
    try:
        # 1. PDFハッシュを計算（DB保存やフロント同期用に利用）
        contents = await file.read()
        file_hash = hashlib.sha256(contents).hexdigest()
        warning_message = ""

        file_id = str(uuid.uuid4())
        # 1. PDFからテキスト抽出
        print(f"[{jst_now_str()}][重要] ファイル読み込み完了: {len(contents)}バイト")
        pdf_stream = io.BytesIO(contents)
        print(f"[重要] BytesIOストリーム作成完了: {pdf_stream.getbuffer().nbytes}バイト")
        try:
            print(f"[重要] PDFテキスト抽出開始: ocr_engine={ocr_engine}")
            text = ""
            normalized_engine = (ocr_engine or "").lower()
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
                print("[警告] DeepSeek OCR (PyTorch) はサポート終了のためMLX版にフォールバックします。")
                use_mlx_ocr = True

            extracted_captions = ""
            mlx_image_captions: list[dict[str, str | int | None]] = []

            ocr_image_compression = (ocr_image_compression or "balanced").lower()
            if ocr_image_compression not in {"light", "balanced", "high"}:
                ocr_image_compression = "balanced"

            if ocr_image_compression == "light":
                resize_max = int(os.getenv("OLLAMA_DEEPSEEK_IMG_MAX_LIGHT", "1200"))
                jpeg_quality = int(os.getenv("OLLAMA_DEEPSEEK_JPEG_Q_LIGHT", "70"))
            elif ocr_image_compression == "high":
                resize_max = int(os.getenv("OLLAMA_DEEPSEEK_IMG_MAX_HIGH", "2048"))
                jpeg_quality = int(os.getenv("OLLAMA_DEEPSEEK_JPEG_Q_HIGH", "92"))
            else:
                resize_max = int(os.getenv("OLLAMA_DEEPSEEK_IMG_MAX_BALANCED", "1600"))
                jpeg_quality = int(os.getenv("OLLAMA_DEEPSEEK_JPEG_Q_BALANCED", "85"))

            if use_ollama_ocr:
                try:
                    text = run_deepseek_ocr_via_ollama_for_pdf(
                        contents,
                        model=os.getenv("OLLAMA_DEEPSEEK_OCR_MODEL"),
                        prompt=os.getenv("OLLAMA_DEEPSEEK_OCR_PROMPT", DEFAULT_OCR_PROMPT),
                        max_pages=None,
                        dpi=int(os.getenv("OLLAMA_DEEPSEEK_DPI", "300")),
                        timeout=int(os.getenv("OLLAMA_DEEPSEEK_TIMEOUT", "600")),
                        image_output_dir=IMAGES_DIR / file_id,
                        resize_max=resize_max,
                        jpeg_quality=jpeg_quality,
                    )
                except Exception as e:
                    warning_message += "Ollama DeepSeek OCR によるテキスト抽出に失敗したため、PyPDFベースにフォールバックします。\n"
                    print(f"[警告] Ollama DeepSeek OCR失敗のためPyPDFへフォールバックします: {e}")
                    use_ollama_ocr = False
                    actual_ocr_engine = "pypdf"

            if not use_mlx_ocr and not use_ollama_ocr:
                text = extract_pdf_text_layout(contents)
                AUTO_THRESHOLD = int(os.getenv("OCR_AUTO_MIN_CHARS", "200"))
                if len(text.strip()) < AUTO_THRESHOLD:
                    print(f"[警告] PyPDF抽出が短いためMLX OCRへフォールバック: {len(text.strip())} < {AUTO_THRESHOLD}")
                    use_mlx_ocr = True
                else:
                    actual_ocr_engine = "pypdf"
            if use_mlx_ocr:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                    tmp_pdf.write(contents)
                    tmp_pdf_path = Path(tmp_pdf.name)
                try:
                    ocr_results = run_deepseek_ocr(
                        input_path=tmp_pdf_path,
                        model_path=os.getenv("MLX_DEEPSEEK_MODEL", "quocnguyen/DeepSeek-OCR-bf16-mlx"),
                        prompt=os.getenv("MLX_DEEPSEEK_PROMPT", DEFAULT_OCR_PROMPT),
                        describe_photos=bool(os.getenv("MLX_DEEPSEEK_DESCRIBE_PHOTOS", "0") not in {"0", "false", "False"}),
                        photo_prompt=os.getenv("MLX_DEEPSEEK_PHOTO_PROMPT", DEFAULT_PHOTO_PROMPT),
                        max_tokens=int(os.getenv("MLX_DEEPSEEK_MAX_TOKENS", "3000")),
                        temperature=float(os.getenv("MLX_DEEPSEEK_TEMPERATURE", "0.0")),
                        max_pages=None,
                        dpi=int(os.getenv("MLX_DEEPSEEK_DPI", "300")),
                        margin=int(os.getenv("MLX_DEEPSEEK_MARGIN", "16")),
                        contrast=float(os.getenv("MLX_DEEPSEEK_CONTRAST", "1.0")),
                        color_mode=os.getenv("MLX_DEEPSEEK_COLOR_MODE", "grayscale"),
                        image_format=os.getenv("MLX_DEEPSEEK_IMAGE_FORMAT", "png"),
                        binarize_threshold=(lambda v: int(v) if v else None)(os.getenv("MLX_DEEPSEEK_BINARIZE_THRESHOLD", "")),
                        sharpen=bool(os.getenv("MLX_DEEPSEEK_SHARPEN", "0") not in {"0", "false", "False"}),
                        ocr_min_length=int(os.getenv("MLX_DEEPSEEK_OCR_MIN_LENGTH", "20")),
                        fallback_text_crop_ratio=float(os.getenv("MLX_DEEPSEEK_FALLBACK_TEXT_CROP_RATIO", "0.4")),
                        save_images_dir=None,
                        verbose=False,
                    )
                finally:
                    tmp_pdf_path.unlink(missing_ok=True)

                text_blocks: list[str] = []
                caption_lines: list[str] = []
                for page in ocr_results:
                    if page.ocr.strip():
                        text_blocks.append(f"[{page.label}]\n{page.ocr.strip()}")
                    if page.photo and page.photo.strip():
                        caption_lines.append(f"- {page.label if page.page is None else f'p{page.page}'}: {page.photo.strip()}")
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
                print("[重要] クレンジング処理を実施します")
                text = cleanse_pdf_text(text)

            sample_text = text[:3000] if len(text) > 3000 else text
            context_sentences = split_sentences(text)
            print(f"[重要] PDF抽出完了: 合計{len(text)}文字, サンプル={sample_text[:100]}...")
        except Exception as pdf_error:
            print(f"[重要] PDF処理エラー: {pdf_error}")
            return {"error": f"PDF処理エラー: {str(pdf_error)}"}
        print("[重要] LLM質問生成開始 (選択モデル)")
        llm_q_instance, resolved_question_llm = init_generation_llm(
            question_llm_model,
            purpose="question_generation",
            temperature=0.2,
            top_p=0.85,
            num_predict=320,
            max_tokens=320,
        )
        print(f"[重要] 質問生成LLM: {resolved_question_llm}")
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
            print(f"[重要] LLM質問生成レスポンス取得: {len(raw_questions_text)}文字")
            questions = [q.strip() for q in raw_questions_text.split('\n') if q.strip()]
            print(f"[重要] 質問リスト生成完了: {len(questions)}件")
        except Exception as e:
            print(f"[重要] LLM質問生成例外: {e}")
            questions = []
        answers: list[str] = []
        llm_a_instance = None
        resolved_answer_llm = None
        if not questions:
            print("[重要] 正規表現によるQA/箇条書き抽出開始")
            bullets = re.findall(r'^[\*\-\d\.]+\s*(.+)', text, re.MULTILINE)
            qas = re.findall(r'Q[\d：: ]*(.+?)\nA[\d：: ]*(.+?)(?=\nQ|\n\Z)', text, re.DOTALL)
            if qas:
                questions = [q.strip() for q, a in qas]
                answers = [a.strip() for q, a in qas]
            elif bullets:
                questions = bullets[:5]
                answers = ["該当内容を本文から要約してください。"] * len(questions)
            else:
                paras = [p.strip() for p in text.split('\n') if p.strip()]
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
            print(f"[重要] 回答生成LLM: {resolved_answer_llm}")
            answer_system_prompt = textwrap.dedent(
                """
                あなたは日本語のRAGシステムにおける回答エンジンです。以下の制約を厳密に守ってください。
                - 提供されたコンテキストに含まれる事実のみを用いて回答すること。
                - 文書内に記載が見つからない場合は「本文に該当記述がありません。」と明示すること。
                - 回答は自然な日本語で2〜3文以内にまとめること。
                - 重要な根拠がある場合はその文を要約して含めること。
                """
            ).strip()
            for i, q in enumerate(questions):
                try:
                    prompt_a = textwrap.dedent(
                        f"""
                        {answer_system_prompt}

                        ### コンテキスト
                        {sample_text}

                        ### 質問
                        {q}

                        ### 回答
                        """
                    ).strip()
                    answer_resp = llm_a_instance.invoke(prompt_a, max_tokens=640)
                    normalized_answer = _extract_answer_text(answer_resp)
                    answer = normalized_answer.strip().split('\n')[0]
                    if answer and answer[-1] not in {"。", "！", "？", ".", "!", "?"}:
                        answer = f"{answer}。"
                    print(f"[重要] LLM回答{i+1}生成完了: {len(answer)}文字")
                    answers.append(answer)
                except Exception as e:
                    import traceback
                    print(f"[重要] LLM回答{i+1}生成例外: {e}")
                    traceback.print_exc()
                    answers.append("該当内容を本文から要約してください。")

        if len(questions) < 5:
            print(f"[警告] 質問数が不足: {len(questions)}件。フォールバックで補完します。")
            fallback_needed = 5 - len(questions)
            paras = [p.strip() for p in text.split('\n') if p.strip()]
            fallback_questions = []
            for para in paras:
                candidate = f"{para[:20]}について説明してください。"
                if candidate not in questions and candidate not in fallback_questions:
                    fallback_questions.append(candidate)
                if len(fallback_questions) >= fallback_needed:
                    break
            while len(fallback_questions) < fallback_needed:
                fallback_questions.append("本文の主要な論点について説明してください。")
            print(f"[重要] フォールバック質問を追加: {fallback_questions}")
            for fallback_q in fallback_questions:
                questions.append(fallback_q)
                if llm_a_instance is not None:
                    try:
                        prompt_a = textwrap.dedent(
                            f"""
                            {answer_system_prompt}

                            ### コンテキスト
                            {sample_text}

                            ### 質問
                            {fallback_q}

                            ### 回答
                            """
                        ).strip()
                        answer_resp = llm_a_instance.invoke(prompt_a, max_tokens=640)
                        normalized_answer = _extract_answer_text(answer_resp)
                        fallback_answer = normalized_answer.strip().split('\n')[0]
                        if fallback_answer and fallback_answer[-1] not in {"。", "！", "？", ".", "!", "?"}:
                            fallback_answer = f"{fallback_answer}。"
                    except Exception as e:
                        import traceback
                        print(f"[重要] フォールバック質問の回答生成例外: {e}")
                        traceback.print_exc()
                        fallback_answer = "本文を要約してください。"
                else:
                    fallback_answer = "本文を要約してください。"
                answers.append(fallback_answer)
        if not questions or not answers:
            print("[重要] ダミーQAセットを返却（questions/answersが空）")
            questions = ["この文書の主題は何ですか？"]
            answers = ["本文を要約してください。"]
            if resolved_answer_llm is None:
                resolved_answer_llm = resolved_question_llm
        # --- qa_meta を生成し、低品質回答を自動補正 ---
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

                qa_meta.append({
                    "score": quality.get("score", 0.0),
                    "is_auto_fixed": retry_count > 0,
                    "is_dummy_answer": quality.get("is_dummy", False),
                    "quality": quality,
                    "context_snippet": context_snippet,
                    "retry_count": retry_count,
                    "candidates": [regenerated_answer],
                    "candidate_scores": [quality.get("score", 0.0)],
                })
        except Exception as e:
            print(f"[警告] qa_meta生成時に例外: {e}。全件デフォルト値を設定します")
            qa_meta = [{
                "score": 1.0,
                "is_auto_fixed": False,
                "is_dummy_answer": False,
                "quality": {"score": 1.0, "is_dummy": False, "needs_retry": False},
                "context_snippet": "",
                "retry_count": 0,
                "candidates": [a],
                "candidate_scores": [1.0],
            } for a in answers]

        # 画像キャプション生成（失敗しても処理継続）
        image_captions: list[dict] = []
        if generate_image_captions:
            try:
                image_captions = generate_image_captions_from_pdf(contents)
            except Exception as e:
                print(f"[警告] 画像キャプション生成中に例外: {e}. キャプションなしで続行します。")

        # DeepSeek MLXで得たキャプションを既存リストに統合
        if use_mlx_ocr and mlx_image_captions:
            image_captions = mlx_image_captions + image_captions

        # チャンク入力テキストを拡張（画像キャプションも検索対象に含める）
        combined_captions = []
        if extracted_captions:
            combined_captions.append(extracted_captions)
        if image_captions:
            combined_captions.append("\n".join([f"- p{c.get('page')}: {c.get('caption','')}" for c in image_captions]))

        if combined_captions:
            captions_text = "\n".join([block for block in combined_captions if block.strip()])
            combined_text = text + "\n\n【画像キャプション】\n" + captions_text
        else:
            combined_text = text

        chunks_for_storage = generate_default_chunks_for_storage(combined_text)
        print(f"[重要] API返却直前: questions={questions}, answers={answers}")
        # 4. 抽出データ保存
        extracted_path = EXTRACTED_DIR / f"{file_id}.json"
        with open(extracted_path, "w", encoding="utf-8") as f_json:
            json.dump({
                "text": sample_text,
                "questions": questions,
                "answers": answers,
                "qa_meta": qa_meta,
                "file_name": file.filename,  # ←file_nameで統一
                "settings": {
                    "cleanse_used": bool(cleanse),
                    "generate_image_captions": bool(generate_image_captions),
                    "ocr_engine_selected": normalized_engine or "auto",
                    "ocr_engine_used": actual_ocr_engine,
                },
                "image_captions": image_captions,
            }, f_json, ensure_ascii=False)
        # PDFファイル保存
        pdf_path = PDF_DIR / f"{file_id}.pdf"
        with open(pdf_path, "wb") as f_pdf:
            f_pdf.write(contents)

        persist_pdf_upload_to_db(
            file_id=file_id,
            file_name=file.filename or f"{file_id}.pdf",
            original_name=file.filename or f"{file_id}.pdf",
            file_size=len(contents),
            storage_path=str(pdf_path),
            cleanse_used=cleanse,
            question_llm_model=resolved_question_llm,
            answer_llm_model=resolved_answer_llm or resolved_question_llm,
            chunks=chunks_for_storage,
            questions=questions,
            answers=answers,
            qa_meta=qa_meta,
            file_hash=file_hash,
            ocr_engine_used=actual_ocr_engine,
            ocr_engine_selected=normalized_engine or "auto",
        )
        print(f"[{jst_now_str()}][INFO] uploadfileでの永続化処理が完了しました")
        # 5. file_id付きで返却
        return {
            "file_id": file_id,
            "text": sample_text,
            "questions": questions,
            "answers": answers,
            "qa_meta": qa_meta,
            "file_name": file.filename,  # ←file_nameで統一
            "warning": warning_message,
            "ocr_engine_used": actual_ocr_engine,
            "ocr_engine_selected": normalized_engine or "auto",
        }
    except Exception as e:
        print(f"[重要] uploadfile全体例外: {e}")
        return {"error": str(e)}

# --- PDFクレンジング関数 ---
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
    # "exam-\nple" -> "example" のようなケースを前処理で連結する。
    merged_lines: list[str] = []
    for line in lines:
        if merged_lines:
            prev = merged_lines[-1]
            # 前の行がハイフンで終わり、次の行が英字で始まる場合は連結
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
        # ごく短い行や極端に長い行はボイラープレート候補から除外
        if len(stripped) < 5 or len(stripped) > 200:
            continue
        norm = _normalize_for_boilerplate(stripped)
        freq[norm] = freq.get(norm, 0) + 1

    # 多くのページに繰り返し現れる行をボイラープレートとみなす
    boilerplate_norms: set[str] = {norm for norm, count in freq.items() if count >= 5}

    # --- 4. ページ番号などの明らかなボイラープレートを検出するパターン ---
    page_number_pattern = re.compile(
        r"^(?:"  # 例: '3', '12'
        r"\d{1,4}"
        r"|"  # 例: '3/10', '12 / 100'
        r"\d{1,4}\s*/\s*\d{1,4}"
        r"|"  # 例: 'Page 3', 'Page 3/10'
        r"page\s+\d{1,4}(?:\s*/\s*\d{1,4})?"
        r")$",
        re.IGNORECASE,
    )

    cleaned: list[str] = []
    table_buffer: list[str] = []

    table_border_pattern = re.compile(r'^[\s\-=~_`+*•·―—┄┅┈┉┌┐└┘┬┴┼┤├╴╶╷╵╸╹╺╻╾╿]+$')
    table_delimiter_pattern = re.compile(r'[\|│┃┆┊┋┇┈┉┌┐└┘┬┴┼┤├]')

    def flush_table() -> None:
        """バッファに貯めた表行をタブ区切りで整形して出力"""
        if not table_buffer:
            return
        normalized_rows: list[str] = []
        for raw in table_buffer:
            row = table_delimiter_pattern.sub('\t', raw)
            row = re.sub(r'\s{2,}', '\t', row.strip())
            row = re.sub(r'\t{2,}', '\t', row)
            normalized_rows.append(row)
        if cleaned and cleaned[-1] != "":
            cleaned.append("")
        cleaned.append('\n'.join(normalized_rows))
        cleaned.append("")
        table_buffer.clear()

    prev_blank = False
    for line in lines:
        stripped = line.strip()

        # 空行はそのまま扱う（ただし後で連続空行は圧縮）
        if not stripped:
            flush_table()
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
            continue

        # ページ番号のみの行など、明らかなボイラープレート行は除去
        if page_number_pattern.match(stripped):
            flush_table()
            continue

        # ドキュメント全体で頻出するボイラープレート行（ヘッダ/フッタ相当）は除去
        norm = _normalize_for_boilerplate(stripped)
        if norm in boilerplate_norms:
            flush_table()
            continue

        # 罫線のみの行は無視
        if table_border_pattern.match(stripped):
            flush_table()
            continue

        # パイプや縦罫線を含む行は表とみなす
        if table_delimiter_pattern.search(stripped):
            table_buffer.append(stripped)
            prev_blank = False
            continue

        # 3つ以上の連続スペースが複数回現れる場合も表候補と判断
        space_chunks = re.findall(r'\s{2,}', line)
        if len(space_chunks) >= 2:
            table_buffer.append(stripped)
            prev_blank = False
            continue

        # 通常テキスト行
        flush_table()
        cleaned.append(stripped)
        prev_blank = False

    flush_table()

    # --- 5. 末尾および連続する空行を整理（最大1行に圧縮） ---
    while len(cleaned) > 1 and cleaned[-1] == "" and cleaned[-2] == "":
        cleaned.pop()

    return '\n'.join(cleaned)


def _normalize_deepseek_ocr_html(text: str) -> str:
    s = html.unescape(str(text))
    if "<" not in s and ">" not in s:
        return s.strip()
    s = re.sub(r"</tr\\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</td\\s*>", " | ", s, flags=re.IGNORECASE)
    s = re.sub(r"<.*?>", "", s)
    lines = [line.strip(" |") for line in s.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


# --- DeepSeek OCR (Ollama) ユーティリティ ---
def run_deepseek_ocr_via_ollama_for_pdf(
    contents: bytes,
    *,
    model: str | None = None,
    prompt: str | None = None,
    max_pages: int | None = None,
    dpi: int = 150,
    timeout: int = 60,
    image_output_dir: Path | None = None,
    resize_max: int | None = None,
    jpeg_quality: int = 85,
) -> str:
    """PDFバイト列をOllama deepseek-ocrに渡してOCRテキストを取得するユーティリティ。

    各ページを画像化してから deepseek-ocr に投げ、ページラベル付きのテキストを連結して返す。
    失敗したページはスキップし、全ページ失敗した場合は例外を投げる。
    """

    try:
        pages = convert_from_bytes(contents, dpi=dpi)
    except Exception as e:
        print(f"[警告] pdf2imageによるページ画像化に失敗: {e}. Ollama DeepSeek OCR処理を中止します。")
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
    ocr_prompt = prompt or os.getenv("OLLAMA_DEEPSEEK_OCR_PROMPT", DEFAULT_OCR_PROMPT)

    text_blocks: list[str] = []

    for idx, pil_img in enumerate(pages):
        try:
            img = pil_img.convert("RGB")
            if resize_max is not None:
                try:
                    if max(img.size) > resize_max:
                        img.thumbnail((resize_max, resize_max))
                except Exception as resize_err:
                    print(f"[警告] DeepSeek OCR画像リサイズに失敗 (page={idx+1}): {resize_err}")

            buf = io.BytesIO()
            try:
                img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            except Exception as jpeg_err:
                print(f"[警告] DeepSeek OCR画像JPEGエンコードに失敗 (page={idx+1}): {jpeg_err}. PNGで再試行します。")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            if image_output_dir is not None:
                try:
                    image_output_dir.mkdir(parents=True, exist_ok=True)
                    save_path = image_output_dir / f"page_{idx + 1:03d}.jpg"
                    with save_path.open("wb") as f:
                        f.write(image_bytes)
                except Exception as save_err:
                    print(f"[警告] DeepSeek OCRページ画像の保存に失敗 (page={idx+1}): {save_err}")
            # 画像1枚に対するDeepSeek-OCR処理は、CLI互換ヘルパーを利用して実行
            # PDF経由でもCLI / /ocr/imageと同じシンプルプロンプトを使うため、promptは指定しない
            page_text = _run_deepseek_ocr_for_image_bytes(
                image_bytes,
                model=model_name,
                resize_max=None,  # ここまでの処理でリサイズ済みのためヘルパー側ではリサイズしない
                jpeg_quality=jpeg_quality,
                timeout=timeout,
            )
            label = f"PAGE {idx + 1}"
            page_text_str = str(page_text).strip()
            if page_text_str:
                text_blocks.append(f"[{label}]\n{page_text_str}")
            else:
                print(f"[警告] deepseek-ocr(Ollama)から空テキスト (page={idx+1})")
        except Exception as e:
            print(f"[警告] deepseek-ocr(Ollama)によるOCR処理に失敗 (page={idx+1}): {e}")

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
    """単一画像バイト列に対して DeepSeek-OCR (Ollama) を実行するヘルパー。

    CLIサンプル(ocr.py)と同等の挙動になるように、/api/chat + stream=True を利用する。
    """

    # 画像をPillowで開く
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"DeepSeek OCR画像の読み込みに失敗しました: {e}") from e

    # 必要に応じてリサイズ
    if resize_max is not None:
        try:
            if max(img.size) > resize_max:
                img.thumbnail((resize_max, resize_max))
        except Exception as resize_err:  # noqa: BLE001
            print(f"[警告] DeepSeek OCR画像リサイズに失敗: {resize_err}")

    # JPEGに再エンコード（CLIと同様の挙動）
    buf = io.BytesIO()
    try:
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    except Exception as jpeg_err:  # noqa: BLE001
        print(f"[警告] DeepSeek OCR画像JPEGエンコードに失敗: {jpeg_err}. PNGで再試行します。")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    encoded_bytes = buf.getvalue()

    img_b64 = base64.b64encode(encoded_bytes).decode("utf-8")

    base_url = get_ollama_base_url().rstrip("/")
    chat_url = f"{base_url}/api/chat"
    model_name = model or os.getenv("OLLAMA_DEEPSEEK_OCR_MODEL", "deepseek-ocr:latest")
    # 画像専用APIではCLIサンプル(ocr.py)と同等のシンプルなプロンプトを使用する
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


@app.post("/ocr/image")
async def ocr_image(file: UploadFile = File(...)):
    """単一画像ファイルに対して DeepSeek-OCR (Ollama) を実行するAPI。

    CLIスクリプト(ocr.py)と同等の挙動を提供する簡易エンドポイント。
    """

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="画像ファイルが空です。")

    try:
        resize_max = int(os.getenv("OLLAMA_DEEPSEEK_IMG_MAX_BALANCED", "1600"))
        jpeg_quality = int(os.getenv("OLLAMA_DEEPSEEK_JPEG_Q_BALANCED", "85"))
        timeout = int(os.getenv("OLLAMA_DEEPSEEK_TIMEOUT", "300"))
    except Exception:  # noqa: BLE001
        resize_max = 1600
        jpeg_quality = 85
        timeout = 300

    try:
        text = _run_deepseek_ocr_for_image_bytes(
            contents,
            resize_max=resize_max,
            jpeg_quality=jpeg_quality,
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[{jst_now_str()}][ERROR] /ocr/image DeepSeek OCR失敗: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"DeepSeek OCR実行中にエラーが発生しました: {e}",
        ) from e

    return {"text": text}


# --- 画像キャプション生成（LLAVA 7B / Ollama）ユーティリティ ---
def generate_image_captions_from_pdf(contents: bytes, max_pages: int | None = None, dpi: int = 150, timeout: int = 30) -> list[dict]:
    """
    PDFバイト列から各ページの画像を生成し、Ollama の llava:7b に渡して
    日本語のページキャプションを生成する。

    Returns:
        list[dict]: [{"page": 1, "caption": "..."}, ...]
    """
    captions: list[dict] = []
    try:
        pages = convert_from_bytes(contents, dpi=dpi)
    except Exception as e:
        print(f"[警告] pdf2imageによるページ画像化に失敗: {e}. 画像キャプション処理をスキップします。")
        return captions

    if max_pages is not None:
        pages = pages[:max_pages]

    base_url = get_ollama_base_url().rstrip('/')
    chat_url = f"{base_url}/api/chat"

    # 画像キャプション用モデルとプロンプトは環境変数で差し替え可能にする（デフォルトは llava:7b）
    caption_model = os.getenv("OLLAMA_IMAGE_CAPTION_MODEL", "llava:7b")
    caption_prompt = os.getenv(
        "OLLAMA_IMAGE_CAPTION_PROMPT",
        "この画像の内容を日本語で簡潔に説明してください。",
    )

    for idx, pil_img in enumerate(pages):
        try:
            # PIL画像をPNGバイトへ
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
            # Ollama /api/chat の非ストリーム応答: { message: { content: "..." }, ... }
            caption = (
                j.get("message", {}).get("content")
                or j.get("response")  # 念のため互換キーも参照
                or "(説明なし)"
            )
            captions.append({"page": idx + 1, "caption": str(caption).strip()})
        except Exception as e:
            print(f"[警告] llava:7bによる画像キャプション生成に失敗 (page={idx+1}): {e}")
            captions.append({"page": idx + 1, "caption": f"画像キャプション生成失敗: {e}"})

    return captions

# --- 新規: file_idで抽出済みデータ取得API ---
from fastapi import HTTPException


@app.get("/get_extracted/{file_id}")
def get_extracted(file_id: str):
    """指定file_idの抽出テキスト・QA・ファイル名を返すAPI。"""
    extracted_path = EXTRACTED_DIR / f"{file_id}.json"
    if not extracted_path.exists():
        raise HTTPException(status_code=404, detail=f"file_id={file_id}の抽出データが見つかりません")
    with open(extracted_path, "r", encoding="utf-8") as f_json:
        data = json.load(f_json)
    # PDF本体もbase64で必ず返す
    pdf_path = PDF_DIR / f"{file_id}.pdf"
    if pdf_path.exists():
        import base64
        with open(pdf_path, "rb") as f_pdf:
            data["pdf_bytes_base64"] = base64.b64encode(f_pdf.read()).decode("utf-8")
    # file_nameがなければfile_id.pdfをセット（後方互換）
    if "file_name" not in data:
        data["file_name"] = f"{file_id}.pdf"
    return data


def _generate_qa_for_existing_pdf(
    file_id: str,
    question_llm_model: str,
    answer_llm_model: str,
) -> dict:
    """抽出済みテキストを用いて既存PDFに対するQAを生成し、JSON/DBを更新するヘルパー。"""
    try:
        # 抽出済みJSONの読み込み
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
        warning_message = data.get("warning") or ""

        pdf_path = PDF_DIR / f"{file_id}.pdf"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"file_id={file_id}のPDF本体が見つかりません")
        with open(pdf_path, "rb") as f_pdf:
            contents = f_pdf.read()
        file_hash = hashlib.sha256(contents).hexdigest()

        # 元実装と同じサンプルテキスト・文分割
        sample_text = text[:3000] if len(text) > 3000 else text
        context_sentences = split_sentences(text)
        print(f"[{jst_now_str()}][重要] generate_qa: 抽出テキスト長={len(text)}  サンプル={sample_text[:100]}...")

        # 画像キャプション（必要に応じて生成）
        image_captions: list[dict] = []
        if generate_image_captions:
            try:
                image_captions = generate_image_captions_from_pdf(contents)
            except Exception as e:
                print(f"[警告] generate_qa: 画像キャプション生成中に例外: {e}. キャプションなしで続行します。")

        # MLX由来のキャプションはここでは再計算しない
        extracted_captions = ""
        mlx_image_captions: list[dict] = []

        # 質問生成（元uploadfileと同じプロンプト）
        print("[重要] generate_qa: LLM質問生成開始")
        llm_q_instance, resolved_question_llm = init_generation_llm(
            question_llm_model,
            purpose="question_generation",
            temperature=0.2,
            top_p=0.85,
            num_predict=320,
            max_tokens=320,
        )
        print(f"[重要] generate_qa: 質問生成LLM={resolved_question_llm}")

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
            print(f"[重要] generate_qa: LLM質問生成レスポンス長={len(raw_questions_text)}")
            questions = [q.strip() for q in raw_questions_text.split("\n") if q.strip()]
            print(f"[重要] generate_qa: 質問リスト生成完了 件数={len(questions)}")
        except Exception as e:
            print(f"[重要] generate_qa: LLM質問生成例外: {e}")
            questions = []

        answers: list[str] = []
        llm_a_instance = None
        resolved_answer_llm: Optional[str] = None

        if not questions:
            print("[重要] generate_qa: 正規表現によるQA/箇条書き抽出開始")
            bullets = re.findall(r'^[\*\-\d\.]+\s*(.+)', text, re.MULTILINE)
            qas = re.findall(r'Q[\d：: ]*(.+?)\nA[\d：: ]*(.+?)(?=\nQ|\n\Z)', text, re.DOTALL)
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
            print(f"[重要] generate_qa: 回答生成LLM={resolved_answer_llm}")
            answer_system_prompt = textwrap.dedent(
                """
                あなたは日本語のRAGシステムにおける回答エンジンです。以下の制約を厳密に守ってください。
                - 提供されたコンテキストに含まれる事実のみを用いて回答すること。
                - 文書内に記載が見つからない場合は「本文に該当記述がありません。」と明示すること。
                - 回答は自然な日本語で2〜3文以内にまとめること。
                - 重要な根拠がある場合はその文を要約して含めること。
                """
            ).strip()

            for i, q in enumerate(questions):
                try:
                    prompt_a = textwrap.dedent(
                        f"""
                        {answer_system_prompt}

                        ### コンテキスト
                        {sample_text}

                        ### 質問
                        {q}

                        ### 回答
                        """
                    ).strip()
                    answer_resp = llm_a_instance.invoke(prompt_a, max_tokens=640)
                    normalized_answer = _extract_answer_text(answer_resp)
                    answer = normalized_answer.strip().split("\n")[0]
                    if answer and answer[-1] not in {"。", "！", "？", ".", "!", "?"}:
                        answer = f"{answer}。"
                    print(f"[重要] generate_qa: LLM回答{i+1}生成完了 文字数={len(answer)}")
                    answers.append(answer)
                except Exception as e:
                    import traceback
                    print(f"[重要] generate_qa: LLM回答{i+1}生成例外: {e}")
                    traceback.print_exc()
                    answers.append("該当内容を本文から要約してください。")

        if len(questions) < 5:
            print(f"[警告] generate_qa: 質問数が不足 ({len(questions)}件)。フォールバックで補完します。")
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
            print(f"[重要] generate_qa: フォールバック質問を追加 {fallback_questions}")
            for fallback_q in fallback_questions:
                questions.append(fallback_q)
                if llm_a_instance is not None:
                    try:
                        prompt_a = textwrap.dedent(
                            f"""
                            {answer_system_prompt}

                            ### コンテキスト
                            {sample_text}

                            ### 質問
                            {fallback_q}

                            ### 回答
                            """
                        ).strip()
                        answer_resp = llm_a_instance.invoke(prompt_a, max_tokens=640)
                        normalized_answer = _extract_answer_text(answer_resp)
                        fallback_answer = normalized_answer.strip().split("\n")[0]
                        if fallback_answer and fallback_answer[-1] not in {"。", "！", "？", ".", "!", "?"}:
                            fallback_answer = f"{fallback_answer}。"
                    except Exception as e:
                        import traceback
                        print(f"[重要] generate_qa: フォールバック質問の回答生成例外: {e}")
                        traceback.print_exc()
                        fallback_answer = "本文を要約してください。"
                else:
                    fallback_answer = "本文を要約してください。"
                answers.append(fallback_answer)

        if not questions or not answers:
            print("[重要] generate_qa: ダミーQAセットを返却（questions/answersが空）")
            questions = ["この文書の主題は何ですか？"]
            answers = ["本文を要約してください。"]
            if resolved_answer_llm is None:
                resolved_answer_llm = resolved_question_llm

        # qa_meta生成（元実装と同様）
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
        except Exception as e:
            print(f"[警告] generate_qa: qa_meta生成時に例外: {e}。全件デフォルト値を設定します")
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

        # DeepSeek MLXキャプションとの統合は行わないが、llava由来キャプションはチャンクテキストに含める
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
        print(f"[重要] generate_qa: チャンク数={len(chunks_for_storage)}")

        # 抽出JSONを上書き保存
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

        # DB更新（既存レコードを上書き）
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
        print(f"[{jst_now_str()}][INFO] generate_qa: 永続化処理が完了しました (file_id={file_id})")

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
    except Exception as e:
        import traceback
        print(f"[{jst_now_str()}][重要] generate_qa全体例外: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pdf/{file_id}/generate_qa")
def generate_qa_for_pdf(
    file_id: str,
    question_llm_model: str = Form("mistral"),
    answer_llm_model: str = Form("mistral"),
):
    """既存PDF(file_id)に対してQA生成を実行するAPI。"""
    return _generate_qa_for_existing_pdf(
        file_id=file_id,
        question_llm_model=question_llm_model,
        answer_llm_model=answer_llm_model,
    )


from pydantic import BaseModel
import io
import os
import sys
print("[CRITICAL] main.pyロード開始")
from pathlib import Path

# MLX DeepSeek OCR (任意機能): コンテナ環境ではmlx_vlmが無いことがあるため、安全にインポートする
try:
    from .mlx_deepseek_ocr_check import (
        DEFAULT_OCR_PROMPT,
        DEFAULT_PHOTO_PROMPT,
        run_deepseek_ocr,
    )
    MLX_OCR_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    print(f"[警告] MLX DeepSeek OCRモジュール読み込みに失敗: {e}. MLX OCRを無効化します。")
    MLX_OCR_AVAILABLE = False
    # Ollama DeepSeek用に最低限のデフォルトプロンプトをここで定義
    DEFAULT_OCR_PROMPT = (
        "Please transcribe every visible character from this image in Japanese. "
        "Do not describe the scene. If any part is unreadable, leave it blank without guessing."
    )
    DEFAULT_PHOTO_PROMPT = (
        "Please describe the content of this image in Japanese briefly and clearly."
    )

# 画像キャプション生成・PDF画像化用の追加インポート（上部で集約）
import base64
import tempfile
import requests
from pdf2image import convert_from_bytes
from PIL import Image

# --- models.yaml, strategies.yaml 読み込み用 ---
try:
    import yaml
except ImportError:
    yaml = None  # PyYAMLが未導入の場合

# 設定ファイルのパス（Dockerコンテナ内の絶対パスを指定）
MODELS_YAML_PATH = Path("/app/models.yaml")
STRATEGIES_YAML_PATH = Path("/app/strategies.yaml")

# モデルリスト取得関数
def load_models_yaml():
    if yaml is None:
        raise RuntimeError("PyYAMLがインストールされていません。requirements.txtに 'pyyaml' を追加してください。");
    if not MODELS_YAML_PATH.exists():
        raise FileNotFoundError(f"models.yamlが見つかりません: {MODELS_YAML_PATH}")
    with open(MODELS_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 戦略リスト取得関数
def load_strategies_yaml():
    if yaml is None:
        raise RuntimeError("PyYAMLがインストールされていません。requirements.txtに 'pyyaml' を追加してください。");
    if not STRATEGIES_YAML_PATH.exists():
        raise FileNotFoundError(f"strategies.yamlが見つかりません: {STRATEGIES_YAML_PATH}")
    with open(STRATEGIES_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import os
# NLTK参照パスを明示的に複数指定
nltk.data.path = ['/usr/local/share/nltk_data', '/usr/local/lib/nltk_data'] + nltk.data.path
print('[NLTK] data search path:', nltk.data.path)
# punktを明示的にダウンロード
nltk.download('punkt', download_dir='/usr/local/share/nltk_data')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def fixed_chunk_text(text, chunk_size=1000, chunk_overlap=0):
    """
    固定長でテキストをチャンク分割
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap if chunk_overlap < chunk_size else chunk_size
    return chunks

def sentence_chunk_text(text):
    """spaCyの日本語モデルで文単位に分割する。"""
    try:
        import spacy
        try:
            nlp = spacy.load("ja_core_news_sm")
        except OSError:
            raise RuntimeError("spaCyの日本語モデル 'ja_core_news_sm' がインストールされていません。\n\n下記コマンドでインストールしてください:\n\npython -m spacy download ja_core_news_sm\n")
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        raise RuntimeError(f"spaCyによる日本語文分割時にエラー: {str(e)}")


def paragraph_chunk_text(text: str) -> list[str]:
    """空行区切りで段落単位に分割する。"""
    if not text:
        return []
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    return paragraphs if paragraphs else [text]


def semantic_chunk_text(text, chunk_size=None, chunk_overlap=None, embedding_model=None, similarity_threshold=0.7):
    """意味的なまとまりでチャンク分割を行う。"""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        import spacy
        nlp = spacy.load("ja_core_news_sm")
    except OSError:
        raise RuntimeError("spaCyの日本語モデル 'ja_core_news_sm' がインストールされていません。\n\npython -m spacy download ja_core_news_sm\n")

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return [text]

    print(f"セマンティックチャンキング: {len(sentences)}文を処理中...")
    if embedding_model is None:
        raise ValueError("embedding_modelが指定されていません")

    batch_size = 32
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)

    chunks = []
    current_chunk = []
    for i in range(len(sentences)):
        current_sentence = sentences[i]
        current_embedding = np.array(embeddings[i]).reshape(1, -1)
        if not current_chunk:
            current_chunk.append(current_sentence)
            continue
        last_embedding = np.array(embeddings[i - 1]).reshape(1, -1)
        similarity = cosine_similarity(last_embedding, current_embedding)[0][0]
        if similarity < similarity_threshold:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [current_sentence]
        else:
            current_chunk.append(current_sentence)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    print(f"セマンティックチャンキング完了: {len(chunks)}個のチャンクを生成")
    return chunks


def generate_default_chunks_for_storage(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """永続化用の固定長チャンクを生成するフォールバックユーティリティ。"""
    if not text:
        return []
    try:
        chunks = fixed_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunks if chunks else [text]
    except Exception as e:
        print(f"[警告] generate_default_chunks_for_storageで例外: {e}。全文を1チャンクとして保存します。")
        return [text]


def persist_pdf_upload_to_db(
    file_id: str,
    file_name: str,
    original_name: str,
    file_size: int,
    storage_path: str,
    cleanse_used: bool,
    question_llm_model: str,
    answer_llm_model: str,
    chunks: list[str],
    questions: list[str],
    answers: list[str],
    qa_meta: list[dict],
    *,
    file_hash: str | None = None,
    ocr_engine_used: str | None = None,
    ocr_engine_selected: str | None = None,
) -> None:
    """PDFアップロード時の情報をDBに永続化する。"""
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO pdf_files (
                        id, file_name, original_name, file_size, storage_path,
                        cleanse_used, question_llm_model, answer_llm_model, file_hash,
                        ocr_engine_used, ocr_engine_selected
                    ) VALUES (
                        :id, :file_name, :original_name, :file_size, :storage_path,
                        :cleanse_used, :question_llm_model, :answer_llm_model, :file_hash,
                        :ocr_engine_used, :ocr_engine_selected
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        file_name = EXCLUDED.file_name,
                        original_name = EXCLUDED.original_name,
                        file_size = EXCLUDED.file_size,
                        storage_path = EXCLUDED.storage_path,
                        cleanse_used = EXCLUDED.cleanse_used,
                        question_llm_model = EXCLUDED.question_llm_model,
                        answer_llm_model = EXCLUDED.answer_llm_model,
                        file_hash = EXCLUDED.file_hash,
                        ocr_engine_used = EXCLUDED.ocr_engine_used,
                        ocr_engine_selected = EXCLUDED.ocr_engine_selected,
                        uploaded_at = CURRENT_TIMESTAMP
                    """
                ),
                {
                    "id": file_id,
                    "file_name": file_name,
                    "original_name": original_name,
                    "file_size": file_size,
                    "storage_path": storage_path,
                    "cleanse_used": cleanse_used,
                    "question_llm_model": question_llm_model,
                    "answer_llm_model": answer_llm_model,
                    "file_hash": file_hash,
                    "ocr_engine_used": ocr_engine_used,
                    "ocr_engine_selected": ocr_engine_selected,
                },
            )

            # 既存のチャンク・QAを削除（同一IDで再登録された場合のクリーニング）
            conn.execute(text("DELETE FROM pdf_chunks WHERE pdf_file_id = :pdf_file_id"), {"pdf_file_id": file_id})
            conn.execute(text("DELETE FROM generated_questions WHERE pdf_file_id = :pdf_file_id"), {"pdf_file_id": file_id})

            if chunks:
                chunk_rows = [
                    {
                        "pdf_file_id": file_id,
                        "chunk_index": idx,
                        "content": chunk,
                        "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                    }
                    for idx, chunk in enumerate(chunks)
                ]
                conn.execute(
                    text(
                        """
                        INSERT INTO pdf_chunks (pdf_file_id, chunk_index, content, content_hash)
                        VALUES (:pdf_file_id, :chunk_index, :content, :content_hash)
                        """
                    ),
                    chunk_rows,
                )

            if questions:
                qa_rows = []
                for idx, question in enumerate(questions):
                    answer_val = answers[idx] if idx < len(answers) else None
                    meta_val = qa_meta[idx] if idx < len(qa_meta) else {}
                    qa_rows.append(
                        {
                            "pdf_file_id": file_id,
                            "question": question,
                            "answer": answer_val,
                            "question_model": question_llm_model,
                            "answer_model": answer_llm_model,
                            "meta_json": json.dumps(meta_val, ensure_ascii=False),
                        }
                    )
                conn.execute(
                    text(
                        """
                        INSERT INTO generated_questions (
                            pdf_file_id, question, answer, question_model, answer_model, meta_json
                        ) VALUES (
                            :pdf_file_id, :question, :answer, :question_model, :answer_model, :meta_json
                        )
                        """
                    ),
                    qa_rows,
                )
        print(f"[{jst_now_str()}] [INFO] PDFアップロード情報をDBに永続化しました (file_id={file_id})")
    except Exception as e:
        import traceback
        print(f"[{jst_now_str()}] [警告] PDFアップロード情報のDB保存に失敗: {e}")
        print(traceback.format_exc())


def persist_experiment_results(
    pdf_file_id: str | None,
    request_params: dict,
    results: list
) -> None:
    """一括評価結果を `experiments` / `experiment_results` に保存する。"""
    if not isinstance(results, list) or not results:
        return

    try:
        sanitized_params = {}
        if isinstance(request_params, dict):
            excluded_keys = {
                "text",
                "questions",
                "answers",
                "qa_meta",
                "chunks",
                "contexts",
                "chunk_texts",
            }
            sanitized_params = {
                k: v for k, v in request_params.items() if k not in excluded_keys
            }

        parameters_json = json.dumps(sanitized_params, ensure_ascii=False) if sanitized_params else None
        total = len(results)
        completed = sum(1 for r in results if isinstance(r, dict) and "error" not in r)
        status = "completed" if completed == total else ("failed" if completed == 0 else "partial")
        experiment_name = request_params.get("experiment_name") if isinstance(request_params, dict) else None
        if not experiment_name:
            experiment_name = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with engine.begin() as conn:
            result_obj = conn.execute(
                text(
                    """
                    INSERT INTO experiments (
                        pdf_file_id, experiment_name, parameters, status,
                        total_combinations, completed_combinations
                    ) VALUES (
                        :pdf_file_id, :experiment_name, :parameters, :status,
                        :total_combinations, :completed_combinations
                    )
                    RETURNING id
                    """
                ),
                {
                    "pdf_file_id": pdf_file_id,
                    "experiment_name": experiment_name,
                    "parameters": parameters_json,
                    "status": status,
                    "total_combinations": total,
                    "completed_combinations": completed,
                },
            )
            experiment_id = result_obj.scalar()
            if not experiment_id:
                return

            result_rows = []
            for res in results:
                if not isinstance(res, dict) or "error" in res:
                    continue
                metrics_list = res.get("metrics", [])
                result_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "embedding_model": res.get("embedding_model"),
                        "chunk_strategy": res.get("chunk_strategy") or res.get("chunk_method"),
                        "chunk_size": res.get("chunk_size"),
                        "chunk_overlap": res.get("chunk_overlap"),
                        "num_chunks": res.get("num_chunks"),
                        "avg_chunk_len": res.get("avg_chunk_len"),
                        "overall_score": res.get("overall_score"),
                        "faithfulness": res.get("faithfulness"),
                        "answer_relevancy": res.get("answer_relevancy"),
                        "context_recall": res.get("context_recall"),
                        "context_precision": res.get("context_precision"),
                        "answer_correctness": res.get("answer_correctness"),
                        "answer_similarity": res.get("answer_similarity"),
                        "details": json.dumps({"metrics": metrics_list}, ensure_ascii=False),
                    }
                )

            if result_rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO experiment_results (
                            experiment_id, embedding_model, chunk_strategy, chunk_size, chunk_overlap,
                            num_chunks, avg_chunk_len, overall_score, faithfulness, answer_relevancy,
                            context_recall, context_precision, answer_correctness, answer_similarity, details
                        ) VALUES (
                            :experiment_id, :embedding_model, :chunk_strategy, :chunk_size, :chunk_overlap,
                            :num_chunks, :avg_chunk_len, :overall_score, :faithfulness, :answer_relevancy,
                            :context_recall, :context_precision, :answer_correctness, :answer_similarity, :details
                        )
                        """
                    ),
                    result_rows,
                )
        print(f"[{jst_now_str()}] [INFO] 評価結果をDBに保存しました (experiment_id={experiment_id})")
    except Exception as e:
        import traceback
        print(f"[{jst_now_str()}] [警告] 評価結果のDB保存に失敗: {e}")
        print(traceback.format_exc())

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
try:
    from langchain_community.vectorstores import FAISS
    _FAISS_AVAILABLE = True
except ImportError:
    FAISS = None  # type: ignore
    _FAISS_AVAILABLE = False
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
try:
    from langchain_core.messages import BaseMessage
except ImportError:  # langchain旧バージョン互換
    from langchain.schema import BaseMessage
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,  # 追加: 回答の正確性指標
    answer_similarity,
)
from datasets import Dataset

# モデルパスの設定
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LOCAL_MODEL_PATH = Path("/app/models/BAAI_bge-small-en-v1.5")

# セキュリティ注意: 本番環境ではAPIキーの表示は避けてください
import logging
logging.basicConfig(level=logging.INFO)
logging.info(f"[起動時] OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")

# --- RAGAS互換ラッパー（set_run_config を要求する環境向け）---
def _extract_answer_text(llm_response):
    """LLM応答オブジェクトからテキスト本体のみを抽出する。"""
    try:
        if isinstance(llm_response, BaseMessage):
            return llm_response.content
        if isinstance(llm_response, dict) and "content" in llm_response:
            return llm_response["content"]
        if hasattr(llm_response, "content"):
            return getattr(llm_response, "content")
    except Exception:
        pass
    return str(llm_response)


ANSWER_DUMMY_PATTERNS = [
    "本文を要約してください",
    "該当内容を本文から要約",
    "情報が見つかりません",
    "記載がありません",
    "わかりません",
]


def split_sentences(text: str) -> list[str]:
    """文書全体を文単位に分割し、スコア計算で利用する。"""
    if not text:
        return []
    segments = re.split(r"[。！？\n]+", text)
    return [seg.strip() for seg in segments if seg.strip()]


def extract_relevant_context(question: str, sentences: list[str], max_sentences: int = 5) -> str:
    """質問と最も関連しそうな文をスコアリングして抽出する。"""
    if not sentences:
        return ""
    keywords = [kw for kw in re.split(r"[\s、,。]", question) if kw]
    scored: list[tuple[float, str]] = []
    for sent in sentences:
        match_hits = sum(1 for kw in keywords if kw and kw in sent)
        similarity = SequenceMatcher(None, question, sent).ratio()
        score = match_hits + similarity
        if score > 0:
            scored.append((score, sent))
    if not scored:
        return ""
    top_sentences = [sent for _, sent in sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]]
    return "\n".join(top_sentences)


def evaluate_answer_quality(answer: str, context_sentences: list[str]) -> dict:
    """回答の品質を判定し、スコアや低信頼フラグを返す。"""
    if not answer:
        return {
            "score": 0.0,
            "is_dummy": True,
            "needs_retry": True,
            "similarity": 0.0,
            "length_score": 0.0,
        }

    lower_answer = answer.lower()
    is_dummy = any(pat.lower() in lower_answer for pat in ANSWER_DUMMY_PATTERNS)
    length_score = min(1.0, len(answer) / 120)
    best_similarity = 0.0
    for sent in context_sentences or [answer]:
        similarity = SequenceMatcher(None, answer, sent).ratio()
        if similarity > best_similarity:
            best_similarity = similarity

    total_score = round((length_score * 0.4 + best_similarity * 0.6), 3)
    needs_retry = is_dummy or len(answer) < 20 or best_similarity < 0.25
    return {
        "score": float(total_score),
        "is_dummy": bool(is_dummy),
        "needs_retry": bool(needs_retry),
        "similarity": float(round(best_similarity, 3)),
        "length_score": float(round(length_score, 3)),
    }


def regenerate_answer_with_context(question: str, context: str, llm, *, max_tokens: int | None = None):
    """低品質と判定された回答を、より厳密な制約で再生成する。"""
    if not context:
        context = question
    refined_prompt = textwrap.dedent(
        f"""
        あなたは日本語の学術文書に基づいて質問へ回答する専門アシスタントです。
        以下の制約を必ず守って回答してください。
        - 回答は提供されたコンテキストの内容にのみ基づくこと。
        - コンテキストに存在しない情報は「本文に該当記述がありません。」と明示すること。
        - 重要な根拠は1文で引用し、そのまま要約した文を含めること。
        - 回答は3文以内で簡潔にまとめること。

        ### コンテキスト
        {context}

        ### 質問
        {question}

        ### 回答
        """
    ).strip()
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    regenerated = llm.invoke(refined_prompt, **kwargs)
    return _extract_answer_text(regenerated).strip()


class RAGASCompatibleOllamaLLM:
    """
    RAGASが set_run_config を呼んでもエラーにしないための薄いラッパー。
    それ以外の属性・メソッドは元のOllamaLLMに委譲する。
    """
    def __init__(self, model: str, base_url: str, **kwargs):
        self._llm = OllamaLLM(model=model, base_url=base_url, **kwargs)

    # RAGASが存在チェックすることがある
    def set_run_config(self, config):
        pass

    # --- ここからRAGAS互換のためのプロンプト正規化ロジックを追加 ---
    def _normalize_prompt_value(self, p):
        """
        LangChainのStringPromptValue/ChatPromptValue等が来ても
        .to_string() で文字列化する。リスト/タプルは要素を再帰的に処理。
        それ以外はstr()にフォールバック。
        """
        if p is None:
            return ""
        # StringPromptValue / ChatPromptValue を想定（duck-typing）
        try:
            to_str = getattr(p, "to_string", None)
            if callable(to_str):
                return to_str()
        except Exception:
            pass
        # リスト/タプルは各要素を文字列化して結合（単一プロンプト扱い時）
        if isinstance(p, (list, tuple)):
            return " ".join(self._normalize_prompt_value(pi) for pi in p)
        # すでに文字列ならそのまま、その他はstrにフォールバック
        return p if isinstance(p, str) else str(p)

    def _normalize_prompts(self, prompts):
        """
        RAGAS/LLMのgenerate系が期待する list[str] を必ず返す。
        - StringPromptValue/ChatPromptValue → [str]
        - str → [str]
        - list/tuple → 各要素を文字列化
        - それ以外 → [str(obj)]
        """
        # 文字列化可能なPromptValue（to_string持ち）を単一として扱う
        try:
            to_str = getattr(prompts, "to_string", None)
            if callable(to_str):
                return [to_str()]
        except Exception:
            pass
        if isinstance(prompts, str):
            return [prompts]
        if isinstance(prompts, (list, tuple)):
            return [self._normalize_prompt_value(p) for p in prompts]
        return [self._normalize_prompt_value(prompts)]

    def _sanitize_kwargs(self, kwargs: dict) -> dict:
        """OllamaLLMが受け付けないkwargを除去する。
        代表的にはOpenAI系の `n`, `best_of`, `logprobs`, `echo` に加え、
        Ollamaの低レベルclient.generateがトップレベルでは受け付けない
        生成制御系（temperature, max_tokens, top_p, top_k, num_predict, stop, seed）を除去。
        """
        if not kwargs:
            return {}
        drop_keys = {
            "n",
            "best_of",
            "logprobs",
            "top_logprobs",
            "echo",
            "presence_penalty",
            "frequency_penalty",
            # 低レベルclient.generateにトップレベルで渡すとTypeErrorになる代表例
            "temperature",
            "max_tokens",   # OpenAI系 → Ollamaはoptions.num_predict
            "top_p",
            "top_k",
            "num_predict",
            "stop",
            "seed",
        }
        return {k: v for k, v in kwargs.items() if k not in drop_keys}

    # RAGASが直接呼び出すことがあるAPIをラップ
    def generate(self, prompts, **kwargs):
        """promptsをlist[str]へ正規化してから委譲"""
        norm = self._normalize_prompts(prompts)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        return self._llm.generate(norm, **safe_kwargs)

    async def agenerate(self, prompts, **kwargs):
        """promptsをlist[str]へ正規化してから委譲（async）"""
        norm = self._normalize_prompts(prompts)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        # 一部の実装でagenerateが存在しても同期返却の場合があるため awaitable 判定を行う
        if hasattr(self._llm, "agenerate"):
            try:
                res = self._llm.agenerate(norm, **safe_kwargs)
                import inspect
                if inspect.isawaitable(res):
                    return await res
                # 同期返却（list/LLMResultなど）の場合はそのまま返す
                return res
            except TypeError:
                # 同期実装に対してawaitしてしまった等の互換性問題に備えフォールバック
                pass
        # フォールバック：スレッドでgenerateを呼ぶ
        import asyncio
        return await asyncio.to_thread(self._llm.generate, norm, **safe_kwargs)

    def invoke(self, prompt, **kwargs):
        """単一プロンプト入力のラッパー。Runnable互換向け"""
        text = self._normalize_prompt_value(prompt)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "invoke"):
            return self._llm.invoke(text, **safe_kwargs)
        # フォールバック：generateの最初のテキストを返す
        res = self._llm.generate([text], **safe_kwargs)
        try:
            # LangChain LLMResult 互換の取り出し
            return res.generations[0][0].text
        except Exception:
            return res

    async def ainvoke(self, prompt, **kwargs):
        """単一プロンプト入力の非同期ラッパー。Runnable互換向け"""
        text = self._normalize_prompt_value(prompt)
        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "ainvoke"):
            try:
                res = self._llm.ainvoke(text, **safe_kwargs)
                import inspect
                if inspect.isawaitable(res):
                    return await res
                return res
            except TypeError:
                pass
        # フォールバック：agenerate→最初のテキストを返す
        if hasattr(self._llm, "agenerate"):
            try:
                res0 = self._llm.agenerate([text], **safe_kwargs)
                import inspect
                if inspect.isawaitable(res0):
                    res = await res0
                else:
                    res = res0
            except TypeError:
                # フォールバック：スレッドでgenerate
                import asyncio
                res = await asyncio.to_thread(self._llm.generate, [text], **safe_kwargs)
        else:
            import asyncio
            res = await asyncio.to_thread(self._llm.generate, [text], **safe_kwargs)
        try:
            return res.generations[0][0].text
        except Exception:
            return res
    # --- ここまで追加 ---

    @property
    def client(self):
        """RAGASや内部実装が直接 .client.generate(**kwargs) を呼んでも
        受け付けないkwarg（例: n）を除去できるようにプロキシを返す。
        """
        try:
            base_client = getattr(self._llm, "client")
        except Exception:
            return None
        return _RAGASSafeClientProxy(base_client, self._sanitize_kwargs)

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def set_run_config(self, config):
        """RAGAS互換のためのno-op。"""
        pass


class _RAGASSafeClientProxy:
    """Ollamaの低レベルclientへの呼び出しをラップし、
    OpenAI互換kwarg（n等）を除去してから委譲する簡易プロキシ。
    """
    def __init__(self, client, sanitize_fn):
        self._client = client
        self._sanitize_fn = sanitize_fn

    async def generate(self, *args, **kwargs):
        """非同期で低レベルclient.generateを実行。
        ragas側がawaitしてもTypeErrorにならないようにするため、to_threadで包む。
        """
        safe_kwargs = self._sanitize_fn(kwargs)
        import asyncio
        return await asyncio.to_thread(self._client.generate, *args, **safe_kwargs)

    async def agenerate(self, *args, **kwargs):
        """generateのエイリアス（async）。"""
        return await self.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


class RAGASLLMAsyncAdapter:
    """
    RAGASが llm.generate(...) を await する経路に入っても例外にならないよう、
    generate/invoke を非同期対応で提供するアダプタ。
    既存の `RAGASCompatibleOllamaLLM` を内包して使うことを想定。
    """
    def __init__(self, llm):
        self._llm = llm

    def _sanitize_kwargs(self, kwargs: dict) -> dict:
        try:
            # 既存ラッパのサニタイズを流用
            return self._llm._sanitize_kwargs(kwargs)
        except Exception:
            # 最低限の防御（OpenAI系の代表キーを除去）
            drop_keys = {
                "n", "best_of", "logprobs", "top_logprobs", "echo",
                "presence_penalty", "frequency_penalty",
                "temperature", "max_tokens", "top_p", "top_k",
                "num_predict", "stop", "seed",
            }
            return {k: v for k, v in (kwargs or {}).items() if k not in drop_keys}

    async def generate(self, prompts, **kwargs):
        """await 可能な generate を提供。内部で agenerate か to_thread を使用"""
        safe_kwargs = self._sanitize_kwargs(kwargs)
        # agenerate が存在しても同期返却の場合があるため awaitable 判定を行う
        if hasattr(self._llm, "agenerate"):
            try:
                res = self._llm.agenerate(prompts, **safe_kwargs)
                import inspect
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception:
                pass
        # フォールバック：スレッドでgenerate
        import asyncio
        return await asyncio.to_thread(self._llm.generate, prompts, **safe_kwargs)

    async def agenerate(self, prompts, **kwargs):
        return await self.generate(prompts, **kwargs)

    async def invoke(self, prompt, **kwargs):
        safe_kwargs = self._sanitize_kwargs(kwargs)
        if hasattr(self._llm, "ainvoke"):
            try:
                res = self._llm.ainvoke(prompt, **safe_kwargs)
                import inspect
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception:
                pass
        # フォールバック：スレッドでinvoke
        import asyncio
        return await asyncio.to_thread(self._llm.invoke, prompt, **safe_kwargs)

    async def ainvoke(self, prompt, **kwargs):
        return await self.invoke(prompt, **kwargs)

    @property
    def client(self):
        try:
            base_client = getattr(self._llm, "client")
        except Exception:
            return None
        return _RAGASSafeClientProxy(base_client, getattr(self._llm, "_sanitize_kwargs", lambda x: x))

    def __getattr__(self, name):
        return getattr(self._llm, name)


class RAGASCompatibleOllamaEmbeddings:
    """
    RAGASが set_run_config を呼んでもエラーにしないための薄いラッパー。
    それ以外の属性・メソッドは元のOllamaEmbeddingsに委譲する。
    """
    def __init__(self, model: str, base_url: str):
        self._emb = OllamaEmbeddings(model=model, base_url=base_url)

    def set_run_config(self, config):
        pass

    async def embed_text(self, text: str):
        """RAGASがawaitしても安全な単一テキスト埋め込みAPI（async）。"""
        import asyncio
        if hasattr(self._emb, "embed_query"):
            return await asyncio.to_thread(self._emb.embed_query, text)
        # フォールバック: embed_documents に単一要素リストで委譲
        vecs = await asyncio.to_thread(self._emb.embed_documents, [text])
        return vecs[0] if vecs else []

    async def aembed_text(self, text: str):
        """非同期版。ragas 側が await しても安全。"""
        # embed_text 自体が async なのでそのまま await する
        return await self.embed_text(text)

    def embed_documents(self, texts):
        # ベクトルストア（PGVector）が同期メソッドを期待するため同期で提供
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str):
        # ベクトルストア（PGVector）が同期メソッドを期待するため同期で提供
        try:
            if isinstance(self._model_name, str) and "jina-embeddings-v4" in self._model_name:
                client = getattr(self._emb, "client", None)
                if client is not None and hasattr(client, "encode"):
                    vec = client.encode(
                        sentences=[text],
                        task="retrieval",
                        prompt_name="query",
                        normalize_embeddings=True,
                    )[0]
                    # SentenceTransformerはnumpyを返すことがあるためlist化
                    return vec.tolist() if hasattr(vec, "tolist") else vec
        except Exception:
            # 失敗時はフォールバック
            pass
        return self._emb.embed_query(text)

    async def aembed_documents(self, texts):
        import asyncio
        # 同期版 embed_documents に委譲（上のv4専用分岐を再利用）
        return await asyncio.to_thread(self.embed_documents, texts)

    def __getattr__(self, name):
        return getattr(self._emb, name)

    def __call__(self, text: str):
        """FAISSなどがコール可能オブジェクトを期待する場合に対応。"""
        return self.embed_query(text)


class RAGASCompatibleHuggingFaceEmbeddings:
    """
    RAGASが embeddings.set_run_config を呼んでもエラーにしないための薄いラッパー。
    それ以外の属性・メソッドは元のHuggingFaceEmbeddingsに委譲する。
    """
    def __init__(self, model_name: str, **kwargs):
        # 元のHuggingFaceEmbeddingsを内部に保持
        # kwargsにはdeviceやencode_kwargsなどが含まれる
        self._model_name = model_name
        self._emb = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        # 一度だけログを出すためのフラグ（スパム防止）
        self._log_once = {"doc": False, "qry": False, "doc_fallback": False, "qry_fallback": False}

    def set_run_config(self, config):
        # RAGAS互換のためのno-op
        pass

    async def embed_text(self, text: str):
        """RAGASがawaitしても安全な単一テキスト埋め込みAPI（async）。"""
        import asyncio
        if hasattr(self._emb, "embed_query"):
            return await asyncio.to_thread(self._emb.embed_query, text)
        vecs = await asyncio.to_thread(self._emb.embed_documents, [text])
        return vecs[0] if vecs else []

    async def aembed_text(self, text: str):
        # embed_text 自体が async なのでそのまま await する
        return await self.embed_text(text)

    def embed_documents(self, texts):
        # jina-embeddings-v4 は Retrieval タスクで passage/query のプロンプトを切替える必要がある
        try:
            if isinstance(self._model_name, str) and "jina-embeddings-v4" in self._model_name:
                client = getattr(self._emb, "_client", None) or getattr(self._emb, "client", None)
                if client is not None and hasattr(client, "encode"):
                    # Passage 用のプロンプトを指定（正規化も実施）
                    vecs = client.encode(
                        sentences=texts,
                        task="retrieval",
                        prompt_name="passage",
                        normalize_embeddings=True,
                    )
                    if not self._log_once["doc"]:
                        print("[emb] Jina v4 encode(passages) path used: task=retrieval, prompt_name=passage, normalize_embeddings=True")
                        self._log_once["doc"] = True
                    return vecs.tolist() if hasattr(vecs, "tolist") else vecs
        except Exception:
            # 失敗時はフォールバック
            pass
        if not self._log_once["doc_fallback"]:
            print("[emb] embed_documents fallback path used (HuggingFaceEmbeddings)")
            self._log_once["doc_fallback"] = True
        return self._emb.embed_documents(texts)

    async def aembed_documents(self, texts):
        import asyncio
        return await asyncio.to_thread(self._emb.embed_documents, texts)

    def embed_query(self, text: str):
        # ベクトルストア（PGVector）が同期メソッドを期待するため同期で提供
        # Jina v4 ではクエリ用プロンプトとタスクを明示する
        try:
            if isinstance(self._model_name, str) and "jina-embeddings-v4" in self._model_name:
                client = getattr(self._emb, "_client", None) or getattr(self._emb, "client", None)
                if client is not None and hasattr(client, "encode"):
                    vec = client.encode(
                        sentences=[text],
                        task="retrieval",
                        prompt_name="query",
                        normalize_embeddings=True,
                    )[0]
                    if not self._log_once["qry"]:
                        print("[emb] Jina v4 encode(query) path used: task=retrieval, prompt_name=query, normalize_embeddings=True")
                        self._log_once["qry"] = True
                    return vec.tolist() if hasattr(vec, "tolist") else vec
        except Exception:
            # 失敗時はフォールバック
            pass
        if not self._log_once["qry_fallback"]:
            print("[emb] embed_query fallback path used (HuggingFaceEmbeddings)")
            self._log_once["qry_fallback"] = True
        return self._emb.embed_query(text)

    async def aembed_query(self, text: str):
        import asyncio
        # 同期版 embed_query に委譲（上のv4専用分岐を再利用）
        return await asyncio.to_thread(self.embed_query, text)

    def __getattr__(self, name):
        # それ以外の属性アクセスは内部の実体に委譲
        return getattr(self._emb, name)

    def __call__(self, text: str):
        """FAISSなどがコール可能オブジェクトを期待する場合に対応。"""
        return self.embed_query(text)


class RAGASCompatibleOpenAIEmbeddings:
    """
    RAGASが set_run_config を呼んでもエラーにしないための薄いラッパー。
    OpenAIEmbeddings のインスタンスを内包し、必要メソッドを委譲する。
    """
    def __init__(self, **kwargs):
        # OpenAIEmbeddings は model / api_key などを kwargs で受け取る
        self._emb = OpenAIEmbeddings(**kwargs)

    def set_run_config(self, config):
        # RAGAS互換のためのno-op
        pass

    async def embed_text(self, text: str):
        # 非同期で単一テキストの埋め込み（RAGAS側がawaitしても安全）
        import asyncio
        if hasattr(self._emb, "embed_query"):
            return await asyncio.to_thread(self._emb.embed_query, text)
        vecs = await asyncio.to_thread(self._emb.embed_documents, [text])
        return vecs[0] if vecs else []

    async def aembed_text(self, text: str):
        return await self.embed_text(text)

    def embed_documents(self, texts):
        # PGVector 等が同期メソッドを期待するため同期で提供
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str):
        # PGVector 等が同期メソッドを期待するため同期で提供
        return self._emb.embed_query(text)

    async def aembed_documents(self, texts):
        import asyncio
        return await asyncio.to_thread(self._emb.embed_documents, texts)

    async def aembed_query(self, text: str):
        import asyncio
        return await asyncio.to_thread(self._emb.embed_query, text)

    def __getattr__(self, name):
        return getattr(self._emb, name)

    def __call__(self, text: str):
        """FAISSなどがコール可能オブジェクトを期待する場合に対応。"""
        return self.embed_query(text)


# データベース接続設定
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/rag_db")
os.environ["PGVECTOR_CONNECTION_STRING"] = DB_URL

# --- コレクション名をモデルごとに動的生成する関数 ---
def get_collection_name(model_name: str) -> str:
    """
    モデル名ごとにコレクション名を動的に決定する関数。
    例: model_name='openai' → 'rag_collection_openai'
    """
    return f"rag_collection_{model_name}"


# --- PDFベースRAG用のコレクション名を生成する関数 ---
def build_collection_name_for_pdf(embedding_model: str, scope: str, pdf_file_id: str | None = None) -> str:
    """embeddingモデルとスコープ（single/all）、PDF ID から一意なcollection_nameを生成する。

    既存の get_collection_name と互換性を保ちつつ、PDF単位のRAGに対応するための拡張。
    """
    base = get_collection_name(embedding_model)
    scope = (scope or "single").lower()
    if scope == "single":
        if not pdf_file_id:
            raise ValueError("scope='single' の場合は pdf_file_id を指定してください。")
        suffix = f"{scope}_{pdf_file_id}"
    elif scope == "all":
        suffix = scope
    else:
        # 未知のスコープはそのまま suffix として扱う
        suffix = scope

    # collection_name はテーブル名ではなく値なので長さ制約は厳しくないが、
    # 念のため英数字とアンダースコア以外は置換しておく
    safe_suffix = re.sub(r"[^0-9a-zA-Z_]+", "_", suffix)
    return f"{base}_{safe_suffix}"

# --- Model Selection ---

DEFAULT_LLM_NAME = "gpt-oss"

LLM_MODEL_CONFIG = {
    # Ollama系（ローカル／Cloud 含む）
    "gpt-oss": {"provider": "ollama", "model": "gpt-oss:20b"},
    "gpt-oss-20b-cloud": {"provider": "ollama", "model": "gpt-oss:20b-cloud"},
    "gpt-oss-120b-cloud": {"provider": "ollama", "model": "gpt-oss:120b-cloud"},
    "llama3": {"provider": "ollama", "model": "llama3"},
    "mistral": {"provider": "ollama", "model": "mistral"},
    "gemma2": {"provider": "ollama", "model": "gemma2"},
    "phi3": {"provider": "ollama", "model": "phi3"},
    # OpenAI系
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
    "gpt-3.5-turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
}


def _resolve_llm_entry(model_name: str):
    name = (model_name or DEFAULT_LLM_NAME).strip()
    entry = LLM_MODEL_CONFIG.get(name)
    if entry is None:
        print(f"[WARN] 未対応のLLM '{model_name}' が指定されました。{DEFAULT_LLM_NAME}へフォールバックします。")
        entry = LLM_MODEL_CONFIG[DEFAULT_LLM_NAME]
        name = DEFAULT_LLM_NAME
    return name, entry


def get_llm_generation(
    model_name: str,
    *,
    temperature: float = 0.3,
    top_p: float = 0.9,
    num_predict: int = 512,
    max_tokens: int | None = None,
):
    """RAG生成処理向けのLLMインスタンスを返す。"""
    resolved_name, entry = _resolve_llm_entry(model_name)
    provider = entry["provider"]
    if provider == "ollama":
        ollama_base_url = get_ollama_base_url()
        print(f"[INFO] generation LLM (Ollama) = {entry['model']} @ {ollama_base_url}")
        options = {
            "temperature": max(0.0, min(temperature, 1.0)),
            "top_p": max(0.0, min(top_p, 1.0)),
            "num_predict": num_predict,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        return RAGASCompatibleOllamaLLM(
            model=entry["model"],
            base_url=ollama_base_url,
            options=options,
        )
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAIモデルを利用するにはOPENAI_API_KEYを設定してください。")
        print(f"[INFO] generation LLM (OpenAI) = {entry['model']}")
        return ChatOpenAI(
            model=entry["model"],
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"未知のLLMプロバイダ: {provider}")


def get_llm_eval():
    """RAGAS評価専用にGPT-OSSを返す。"""
    return RAGASCompatibleOllamaLLM(
        model=LLM_MODEL_CONFIG[DEFAULT_LLM_NAME]["model"],
        base_url=get_ollama_base_url()
    )


# 後方互換のためのエイリアス（従来の呼び出しは評価用として扱う）
def get_llm(model_name: str):
    return get_llm_eval()


def init_generation_llm(
    model_name: str,
    purpose: str = "generation",
    *,
    temperature: float = 0.3,
    top_p: float = 0.9,
    num_predict: int = 512,
    max_tokens: int | None = None,
):
    """
    生成用LLMの初期化を行い、失敗時は既定モデルにフォールバックする。
    戻り値: (llm_instance, resolved_model_name)
    """
    target = model_name or DEFAULT_LLM_NAME
    try:
        llm = get_llm_generation(
            target,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            max_tokens=max_tokens,
        )
        return llm, target
    except Exception as e:
        print(f"[WARN] {purpose} LLM '{target}' 初期化失敗: {e}. {DEFAULT_LLM_NAME}へフォールバックします。")
        fallback = DEFAULT_LLM_NAME
        llm = get_llm_generation(
            fallback,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
            max_tokens=max_tokens,
        )
        return llm, fallback


# 利用可能なデバイスを自動判定（Apple Siliconならmps, NVIDIAならcuda, どちらもなければcpu）
def get_torch_device():
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"  # Apple Silicon(M1/M2/M3/M4)のMetalアクセラレーション
        elif torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"
    print(f"[INFO] get_torch_device: 利用デバイス = {device}")  # ログ出力
    return device


_EMBEDDING_CACHE: Dict[str, Any] = {}
_EMBEDDING_CACHE_LOCK = threading.Lock()
_CHUNK_CACHE: Dict[Tuple[str, str, float, str], list[str]] = {}
_CHUNK_CACHE_LOCK = threading.Lock()
_VECTORSTORE_CACHE: Dict[Tuple[str, str, str], Any] = {}
_VECTORSTORE_CACHE_LOCK = threading.Lock()


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _hash_chunks(chunks: list[str]) -> str:
    hasher = hashlib.sha1()
    for chunk in chunks:
        hasher.update(chunk.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def get_embeddings(model_name: str):
    cache_key = model_name
    with _EMBEDDING_CACHE_LOCK:
        cached = _EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        print(f"[CACHE] get_embeddings hit: {model_name}")
        return cached

    print(f"[CACHE] get_embeddings miss: {model_name} -> loading")
    device = get_torch_device()  # デバイス自動判定
    common_kwargs = {
        'model_kwargs': {
            'device': device,
            'trust_remote_code': True
        },
        'encode_kwargs': {
            'normalize_embeddings': True
        }
    }
    embedder: Any
    
    # Ollama埋め込みモデル（優先使用）
    ollama_base_url = get_ollama_base_url()
    ollama_embedding_models = {
        "nomic-embed-text": "nomic-embed-text",
        "mxbai-embed-large": "mxbai-embed-large",
        "all-minilm": "all-minilm",
        # 日本語/多言語対応のモデル（Ollama）
        # 事前に `ollama pull bge-m3` / `ollama pull qwen3-embedding` / `ollama pull snowflake-arctic-embed2` / `ollama pull jina-embeddings-v3` を実行しておくこと
        "bge-m3": "bge-m3",
        "qwen3-embedding": "qwen3-embedding",
        "snowflake-arctic-embed2": "snowflake-arctic-embed2",
        "jina-embeddings-v3": "jina-embeddings-v3",
    }
    if model_name in ollama_embedding_models:
        # RAGAS互換の薄いラッパーで包む（set_run_config 要求に対応）
        embedder = RAGASCompatibleOllamaEmbeddings(
            model=ollama_embedding_models[model_name],
            base_url=ollama_base_url,
        )
    else:
        # OpenAIモデルのマッピング
        openai_models = {
            "gpt-4o": "text-embedding-ada-002",  # 旧モデル名との互換性のため
            "text-embedding-3-small": "text-embedding-3-small",
            "text-embedding-3-large": "text-embedding-3-large",
            "text-embedding-ada-002": "text-embedding-ada-002"
        }
        
        if model_name in openai_models:
            embedder = RAGASCompatibleOpenAIEmbeddings(
                model=openai_models[model_name],
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            # HuggingFaceモデルのマッピング
            hf_models = {
                "huggingface_bge_small": "BAAI/bge-small-en-v1.5",
                "huggingface_bge_large": "BAAI/bge-large-en-v1.5",
                "huggingface_miniLM": "sentence-transformers/all-MiniLM-L6-v2",
                "huggingface_mpnet_base": "sentence-transformers/all-mpnet-base-v2",
                "huggingface_multi_qa_minilm": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                "huggingface_multi_qa_mpnet": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                "huggingface_paraphrase_multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "huggingface_distiluse_multilingual": "sentence-transformers/distiluse-base-multilingual-cased-v2",
                "huggingface_xlm_r": "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
                # --- Jina Embeddings v4 (HuggingFace) ---
                # 多言語・マルチモーダル埋め込み。text-only用途ではRetrieval(task)を使用。
                "jina-embeddings-v4": "jinaai/jina-embeddings-v4",
            }
            
            if model_name in hf_models:
                hf_id = hf_models[model_name]
                hf_local_root = os.getenv("HF_LOCAL_DIR", "/app/local_models")
                hf_local_path = os.path.join(hf_local_root, hf_id)
                # HF_LOCAL_DIR/hf_id 配下を再帰的に探索し、config.json を含むディレクトリがあればそれをローカルモデルディレクトリとみなす
                local_model_dir = None
                if os.path.isdir(hf_local_path):
                    for root, dirs, files in os.walk(hf_local_path):
                        if "config.json" in files:
                            local_model_dir = root
                            break
                if local_model_dir is not None:
                    target_model_name = local_model_dir
                else:
                    target_model_name = hf_id
                # Jina Embeddings v4 はエンコード前にタスク指定が必須
                # 参考: エラーメッセージ "Task must be specified before encoding data..."
                if model_name == "jina-embeddings-v4":
                    # 初期化時に default_task は渡さず、エンコード時に task を指定する
                    embedder = RAGASCompatibleHuggingFaceEmbeddings(
                        model_name=target_model_name,
                        model_kwargs={
                            'device': device,
                            'trust_remote_code': True,
                        },
                        encode_kwargs={
                            'normalize_embeddings': True,
                            'task': 'retrieval',
                        }
                    )
                else:
                    # それ以外は共通設定を適用
                    embedder = RAGASCompatibleHuggingFaceEmbeddings(
                        model_name=target_model_name,
                        **common_kwargs
                    )
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")

    with _EMBEDDING_CACHE_LOCK:
        _EMBEDDING_CACHE[cache_key] = embedder
    return embedder


# Default models（モデルが未ダウンロードでもサーバーが起動できるように修正）
current_llm = None
current_embeddings = None

# 内部LLMはGPT-OSS固定（評価用）
try:
    current_llm = get_llm_eval()
except Exception as e:
    import logging
    logging.warning(f"LLM初期化失敗 (gpt-oss eval): {e}")
    current_llm = None

try:
    # デフォルトでHuggingFaceの軽量モデルを使用
    current_embeddings = get_embeddings("huggingface_bge_small")
    if current_embeddings is None:
        raise ValueError("Failed to initialize huggingface_bge_small")
    logging.info("Successfully initialized HuggingFace BGE Small model")
except Exception as e:
    import logging
    logging.error(f"Embedding初期化失敗: {e}")
    try:
        # 代替モデルを試す
        current_embeddings = get_embeddings("huggingface_miniLM")
        logging.info("Falling back to HuggingFace MiniLM model")
    except Exception as e2:
        logging.error(f"代替Embeddingモデルの初期化にも失敗: {e2}")
        current_embeddings = None

# --- Pydantic Models ---
class InferenceModeResponse(BaseModel):
    """現在の推論モードを返すレスポンスモデル。"""

    mode: str


class InferenceModeUpdateRequest(BaseModel):
    """推論モードを更新するリクエストモデル。"""

    mode: str


class InferenceHealthResponse(BaseModel):
    """現在の推論モードと Ollama API の疎通状況を返すレスポンスモデル。"""

    mode: str
    base_url: str
    ok: bool
    status_code: int | None = None
    error: str | None = None


class ChunkRequest(BaseModel):
    text: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_method: str = 'recursive'  # 'recursive' or 'semantic'
    embedding_model: str = None  # Required for semantic chunking

class EmbedRequest(BaseModel):
    chunks: list[str]
    embedding_model: str  # 埋め込みモデル名
    chunk_method: str     # チャンク方式（recursive, semantic, fixed, sentence, paragraph など）

class QueryRequest(BaseModel):
    query: str
    llm_model: str = "mistral"  # デフォルト値を設定
    embedding_model: str = "huggingface_bge_small"  # デフォルト値を設定
    scope: str = "single"  # "single" または "all"
    pdf_file_id: str | None = None  # scope == "single" のときに対象PDFを指定

class BuildVectorStoreRequest(BaseModel):
    """PDFベースRAG用にベクトルストアを構築するためのリクエストモデル。"""

    scope: str  # "single" または "all"
    pdf_file_id: str | None = None  # scope == "single" のとき必須
    embedding_model: str
    chunk_method: str = "recursive"  # 'recursive', 'fixed', 'semantic', 'sentence', 'paragraph'
    chunk_size: int | None = 1000
    chunk_overlap: int | None = 200
    similarity_threshold: float | None = 0.7  # semantic 用

# 単一評価リクエストは一括評価に統合されました

class ModelSelection(BaseModel):
    llm_model: str
    embedding_model: str


def _db_get_model_selection() -> dict[str, str]:
    """DBからモデル選択状態を取得する。存在しなければデフォルト値を返す。"""
    default_llm = DEFAULT_LLM_NAME
    default_emb = "huggingface_bge_small"
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT llm_model, embedding_model FROM model_selection WHERE id = 1")
            ).fetchone()
        if row:
            llm = (row[0] or default_llm).strip()
            emb = (row[1] or default_emb).strip()
            return {"llm_model": llm, "embedding_model": emb}
    except Exception as e:
        print(f"[{jst_now_str()}][ERROR] model_selection取得失敗: {e}")
    return {"llm_model": default_llm, "embedding_model": default_emb}


def _db_update_model_selection(selection: ModelSelection) -> None:
    """DB上のモデル選択状態を更新する。"""
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO model_selection (id, llm_model, embedding_model, updated_at)
                    VALUES (1, :llm_model, :embedding_model, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO UPDATE
                    SET llm_model = EXCLUDED.llm_model,
                        embedding_model = EXCLUDED.embedding_model,
                        updated_at = CURRENT_TIMESTAMP
                    """
                ),
                {
                    "llm_model": selection.llm_model,
                    "embedding_model": selection.embedding_model,
                },
            )
    except Exception as e:
        print(f"[{jst_now_str()}][ERROR] model_selection更新失敗: {e}")


@app.get("/config/inference_mode", response_model=InferenceModeResponse)
def get_inference_mode_api():
    """現在の推論モードを取得するAPI。"""
    mode = get_inference_mode()
    return {"mode": mode}


@app.get("/config/model_selection", response_model=ModelSelection)
def get_model_selection_api():
    """現在のLLM/Embeddingモデル選択状態を取得するAPI。"""
    data = _db_get_model_selection()
    return ModelSelection(**data)


@app.post("/config/model_selection", response_model=ModelSelection)
def update_model_selection_api(req: ModelSelection):
    """LLM/Embeddingモデル選択状態を更新するAPI。"""
    # 最低限、空文字の場合はデフォルトへフォールバック
    llm = (req.llm_model or DEFAULT_LLM_NAME).strip()
    emb = (req.embedding_model or "huggingface_bge_small").strip()
    sel = ModelSelection(llm_model=llm, embedding_model=emb)
    _db_update_model_selection(sel)
    return sel


@app.get("/config/inference_health", response_model=InferenceHealthResponse)
def get_inference_health_api():
    """現在の推論モードで利用される Ollama API への疎通状況を返す。"""
    mode = get_inference_mode()
    base_url = get_ollama_base_url().rstrip("/")
    url = f"{base_url}/api/version"
    status_code: int | None = None
    error: str | None = None
    ok = False
    try:
        resp = requests.get(url, timeout=15)
        status_code = resp.status_code
        ok = resp.ok
    except Exception as e:  # noqa: BLE001
        error = str(e)
    return {
        "mode": mode,
        "base_url": base_url,
        "ok": ok,
        "status_code": status_code,
        "error": error,
    }


@app.post("/config/inference_mode", response_model=InferenceModeResponse)
def update_inference_mode_api(req: InferenceModeUpdateRequest):
    """推論モードを更新するAPI。"""
    try:
        set_inference_mode(req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    mode = get_inference_mode()
    return {"mode": mode}


# --- API Endpoints ---


@app.post("/chunk/")
def chunk_text(request: ChunkRequest):
    """
    chunk_methodに応じて適切な方法でテキストをチャンク分割
    - recursive: 再帰的にテキストを分割（デフォルト）
    - fixed: 固定長で分割
    - semantic: 意味的なまとまりで分割（embeddingモデルが必要）
    - sentence: 文単位で分割
    - paragraph: 段落単位で分割
    """
    if request.chunk_method == 'semantic':
        # embedding_modelが指定されていることを確認
        if not request.embedding_model:
            raise HTTPException(
                status_code=400,
                detail="semanticチャンキングにはembedding_modelの指定が必要です"
            )
        try:
            # モデル名から埋め込みインスタンスを生成
            embedder = get_embeddings(request.embedding_model)
            chunks = semantic_chunk_text(
                text=request.text,
                chunk_size=None,
                chunk_overlap=None,
                embedding_model=embedder  # インスタンスを渡す
            )
            return {"chunks": chunks}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"テキストのチャンキング中にエラーが発生しました: {str(e)}"
            )
    elif request.chunk_method == 'recursive':
        # 再帰的な文字数分割
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(request.text)
        return {"chunks": chunks}
    elif request.chunk_method == 'fixed':
        # 固定長で分割
        chunks = fixed_chunk_text(request.text, request.chunk_size, request.chunk_overlap)
        return {"chunks": chunks}
    elif request.chunk_method == 'sentence':
        # 文単位で分割
        chunks = sentence_chunk_text(request.text)
        return {"chunks": chunks}
    elif request.chunk_method == 'paragraph':
        # 段落単位で分割
        chunks = paragraph_chunk_text(request.text)
        return {"chunks": chunks}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"未対応のchunk_method: {request.chunk_method}。'recursive', 'fixed', 'semantic', 'sentence', 'paragraph' のいずれかを指定してください。"
        )


@app.post("/embed_and_store/")
def embed_and_store(request: EmbedRequest):
    try:
        embeddings_instance = get_embeddings(request.embedding_model)
        vectorstore = PGVector.from_documents(
            documents=[],  # 空のドキュメントで初期化
            embedding=embeddings_instance,
            collection_name=get_collection_name(request.embedding_model)  # embeddingモデルごとにコレクションを切り替え
        )
        # chunk_methodを全チャンクのmetadataに付与して保存
        chunk_method = getattr(request, 'chunk_method', None)
        # chunk_methodがEmbedRequestにない場合は、各チャンクのメタ情報としてNoneになる
        metadatas = [{"chunk_method": chunk_method} for _ in request.chunks]
        vectorstore.add_texts(texts=request.chunks, metadatas=metadatas)
        return {"message": f"Successfully embedded and stored {len(request.chunks)} chunks using {request.embedding_model} (method={chunk_method}) ."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build_vectorstore/")
def build_vectorstore(request: BuildVectorStoreRequest):
    """pdf_chunks を元に、PDFベースRAG用のベクトルストアを構築するAPI。

    scope="single" の場合は特定PDF、scope="all" の場合は全PDFを対象とする。
    """
    scope = (request.scope or "single").lower()
    if scope not in {"single", "all"}:
        raise HTTPException(status_code=400, detail=f"未対応のscopeです: {scope}。'single' または 'all' を指定してください。")

    if scope == "single" and not request.pdf_file_id:
        raise HTTPException(status_code=400, detail="scope='single' の場合は pdf_file_id を指定してください。")

    # 埋め込みモデルをロード
    embedder = get_embeddings(request.embedding_model)

    # チャンク分割用の内部ヘルパ
    def _chunk_text_for_request(text: str) -> list[str]:
        method = (request.chunk_method or "recursive").lower()
        if not text:
            return []
        if method == "semantic":
            # semantic チャンキングは埋め込みモデル必須
            sim_th = request.similarity_threshold if request.similarity_threshold is not None else 0.7
            return semantic_chunk_text(
                text=text,
                chunk_size=None,
                chunk_overlap=None,
                embedding_model=embedder,
                similarity_threshold=sim_th,
            )
        elif method == "recursive":
            size = request.chunk_size or 1000
            overlap = request.chunk_overlap or 200
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                length_function=len,
            )
            return splitter.split_text(text)
        elif method == "fixed":
            size = request.chunk_size or 1000
            overlap = request.chunk_overlap or 0
            return fixed_chunk_text(text, size, overlap)
        elif method == "sentence":
            return sentence_chunk_text(text)
        elif method == "paragraph":
            return paragraph_chunk_text(text)
        else:
            raise HTTPException(status_code=400, detail=f"未対応のchunk_methodです: {method}")

    texts: list[str] = []
    metadatas: list[dict] = []

    # pdf_chunks からテキストを収集し、チャンク化する
    try:
        with engine.begin() as conn:
            if scope == "single":
                rows = conn.execute(
                    text(
                        """
                        SELECT content FROM pdf_chunks
                        WHERE pdf_file_id = :fid
                        ORDER BY chunk_index ASC
                        """
                    ),
                    {"fid": request.pdf_file_id},
                ).fetchall()
                if not rows:
                    fallback_text = ""
                    try:
                        extracted_path = EXTRACTED_DIR / f"{request.pdf_file_id}.json"
                        if extracted_path.exists():
                            with extracted_path.open("r", encoding="utf-8") as f:
                                data = json.load(f)
                            raw_text = data.get("text") or ""
                            if raw_text and isinstance(raw_text, str):
                                fallback_text = raw_text
                    except Exception as e:
                        print(f"[{jst_now_str()}] [WARN] pdf_chunks fallback from extracted JSON failed: {e}")
                    if not fallback_text.strip():
                        raise HTTPException(status_code=404, detail="指定されたPDFのチャンクが見つかりません。")
                    sample_text = fallback_text
                    chunks = _chunk_text_for_request(sample_text)
                    if not chunks:
                        raise HTTPException(status_code=400, detail="抽出テキストからのチャンク化結果が空です。PDF内容を確認してください。")
                    try:
                        chunk_rows = [
                            {
                                "pdf_file_id": request.pdf_file_id,
                                "chunk_index": idx,
                                "content": ch,
                                "content_hash": hashlib.sha256(ch.encode("utf-8")).hexdigest(),
                            }
                            for idx, ch in enumerate(chunks)
                        ]
                        conn.execute(
                            text(
                                """
                                INSERT INTO pdf_chunks (pdf_file_id, chunk_index, content, content_hash)
                                VALUES (:pdf_file_id, :chunk_index, :content, :content_hash)
                                """
                            ),
                            chunk_rows,
                        )
                    except Exception as e:
                        print(f"[{jst_now_str()}] [WARN] failed to repopulate pdf_chunks from extracted JSON: {e}")
                else:
                    sample_text = "\n".join(r[0] for r in rows if r[0])
                    chunks = _chunk_text_for_request(sample_text)
                for idx, ch in enumerate(chunks):
                    texts.append(ch)
                    metadatas.append(
                        {
                            "pdf_file_id": request.pdf_file_id,
                            "chunk_index": idx,
                            "chunk_method": request.chunk_method,
                            "scope": scope,
                        }
                    )
            else:  # scope == "all"
                pdf_rows = conn.execute(
                    text(
                        """
                        SELECT DISTINCT pdf_file_id
                        FROM pdf_chunks
                        WHERE pdf_file_id IS NOT NULL
                        ORDER BY pdf_file_id
                        """
                    )
                ).fetchall()
                pdf_ids = [r[0] for r in pdf_rows if r[0]]
                if not pdf_ids:
                    raise HTTPException(status_code=404, detail="pdf_chunks に有効なデータが存在しません。")

                for fid in pdf_ids:
                    rows = conn.execute(
                        text(
                            """
                            SELECT content FROM pdf_chunks
                            WHERE pdf_file_id = :fid
                            ORDER BY chunk_index ASC
                            """
                        ),
                        {"fid": fid},
                    ).fetchall()
                    if not rows:
                        continue
                    sample_text = "\n".join(r[0] for r in rows if r[0])
                    if not sample_text.strip():
                        continue
                    chunks = _chunk_text_for_request(sample_text)
                    for idx, ch in enumerate(chunks):
                        texts.append(ch)
                        metadatas.append(
                            {
                                "pdf_file_id": fid,
                                "chunk_index": idx,
                                "chunk_method": request.chunk_method,
                                "scope": scope,
                            }
                        )
    except HTTPException:
        # そのまま再送出
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pdf_chunks からのテキスト収集中にエラーが発生しました: {str(e)}")

    if not texts:
        raise HTTPException(status_code=400, detail="チャンク化後のテキストが空です。PDFの内容を確認してください。")

    # コレクション名を決定し、既存コレクションがあれば削除して再構築
    try:
        collection_name = build_collection_name_for_pdf(
            request.embedding_model,
            scope,
            request.pdf_file_id if scope == "single" else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 既存コレクションを削除（存在しない場合は無視）
    try:
        prev_vs = PGVector.from_documents(
            documents=[],
            embedding=embedder,
            collection_name=collection_name,
        )
        prev_vs.delete_collection()
    except Exception:
        # コレクションがまだ存在しない場合などはエラーにしない
        pass

    # 新しいベクトルストアを構築
    try:
        vectorstore = PGVector.from_documents(
            documents=[],
            embedding=embedder,
            collection_name=collection_name,
        )
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ベクトルストア構築中にエラーが発生しました: {str(e)}")

    return {
        "status": "success",
        "collection_name": collection_name,
        "num_chunks": len(texts),
        "scope": scope,
    }


@app.post("/query/")
def query_rag(request: QueryRequest):
    """事前に構築されたPDFベースのベクトルストアを用いてRAG応答を生成する。"""
    try:
        llm_instance, resolved_llm = init_generation_llm(request.llm_model, purpose="/query")
        print(f"[INFO] /query 生成LLM={resolved_llm}")

        scope = (request.scope or "single").lower()
        if scope not in {"single", "all"}:
            raise HTTPException(status_code=400, detail=f"未対応のscopeです: {scope}。'single' または 'all' を指定してください。")
        if scope == "single" and not request.pdf_file_id:
            raise HTTPException(status_code=400, detail="scope='single' の場合は pdf_file_id を指定してください。")

        embeddings_instance = get_embeddings(request.embedding_model)

        # 対象コレクション名を決定
        try:
            collection_name = build_collection_name_for_pdf(
                request.embedding_model,
                scope,
                request.pdf_file_id if scope == "single" else None,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # 既存のベクトルストアを利用（PGVector内でコレクション名を解決）
        try:
            vectorstore = PGVector(
                embedding_function=embeddings_instance,
                collection_name=collection_name,
                connection_string=DB_URL,
                use_jsonb=True,
            )
        except Exception as e:
            # コレクション未構築などの場合は404として扱う
            raise HTTPException(
                status_code=404,
                detail=(
                    "指定されたベクトルストアが存在しません。先に /build_vectorstore/ を実行してください。"
                    f" collection_name={collection_name}, error={str(e)}"
                ),
            )

        retriever = vectorstore.as_retriever()

        # プロンプトテンプレート
        template = """以下の文脈に基づいて質問に答えてください。

文脈:
{context}

質問: {question}"""
        prompt = ChatPromptTemplate.from_template(template)

        # 関連するドキュメントを取得し、コンテキスト文字列を構築
        retrieved_docs = retriever.get_relevant_documents(request.query)
        contexts = [doc.page_content for doc in retrieved_docs]
        context_text = "\n\n".join(contexts) if contexts else "(関連文脈が見つかりませんでした)"

        # LLMへ渡すプロンプト文字列を生成し、RAGAS互換ラッパを直接呼び出す
        prompt_text = template.format(context=context_text, question=request.query)
        raw_answer = llm_instance.invoke(prompt_text)
        answer = _extract_answer_text(raw_answer)

        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO chat_logs (
                            pdf_file_id,
                            user_message,
                            assistant_message,
                            llm_model_used,
                            embedding_model,
                            scope
                        )
                        VALUES (
                            :pdf_file_id,
                            :user_message,
                            :assistant_message,
                            :llm_model_used,
                            :embedding_model,
                            :scope
                        )
                        """
                    ),
                    {
                        "pdf_file_id": request.pdf_file_id if scope == "single" else None,
                        "user_message": request.query,
                        "assistant_message": answer,
                        "llm_model_used": resolved_llm,
                        "embedding_model": request.embedding_model,
                        "scope": scope,
                    },
                )
        except Exception as e:
            print(f"[WARN] chat_logs insert failed: {e}")

        return {
            "answer": answer,
            "contexts": contexts,
            "source_documents": [{"page_content": doc.page_content} for doc in retrieved_docs],
            "llm_model_used": resolved_llm,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in query_rag: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"エラーが発生しました: {str(e)}\n{error_trace}",
        )

def calculate_overlap_metrics(contexts: list[list[str]], embedder=None) -> dict:
    """複数のオーバーラップメトリクスを計算する
    
    Args:
        contexts: コンテキストのリスト
        embedder: オプションの埋め込みモデル（セマンティックオーバーラップ用）
    
    Returns:
        dict: 各種オーバーラップメトリクスを含む辞書
    """
    if not contexts or len(contexts) < 2:
        return {
            "overlap_ratio": 0.0,
            "adjacent_overlap": [0.0],
            "semantic_overlap": 0.0
        }
    
    # 1. 元のオーバーラップ計算（後方互換性のため保持）
    all_tokens = []
    for ctx in contexts:
        if isinstance(ctx, str):
            all_tokens.extend(ctx.split())
        else:
            for text in ctx:
                all_tokens.extend(text.split())
    
    unique_tokens = set(all_tokens)
    total_tokens = len(all_tokens)
    unique_count = len(unique_tokens)
    
    overlap_ratio = 1.0 - (unique_count / total_tokens) if total_tokens > 0 else 0.0
    
    # 2. 隣接チャンク間のオーバーラップ
    adjacent_overlaps = []
    for i in range(len(contexts) - 1):
        # 現在のチャンクと次のチャンクのトークンを取得
        current_ctx = contexts[i] if isinstance(contexts[i], list) else [contexts[i]]
        next_ctx = contexts[i+1] if isinstance(contexts[i+1], list) else [contexts[i+1]]
        
        current_tokens = set(' '.join(current_ctx).split())
        next_tokens = set(' '.join(next_ctx).split())
        
        # 共通トークン数を計算
        common_tokens = current_tokens.intersection(next_tokens)
        min_len = min(len(current_tokens), len(next_tokens))
        
        # オーバーラップ率を計算
        overlap = len(common_tokens) / min_len if min_len > 0 else 0.0
        adjacent_overlaps.append(overlap)
    
    # 3. セマンティックオーバーラップ（埋め込みモデルが利用可能な場合）
    semantic_overlap = 0.0
    if embedder and len(contexts) > 1:
        try:
            # 各チャンクを1つの文字列に結合
            chunk_texts = [' '.join(ctx) if isinstance(ctx, list) else ctx for ctx in contexts]
            
            # 埋め込みを取得
            embeddings = embedder.embed_documents(chunk_texts)
            
            # 隣接チャンク間の類似度を計算
            similarities = []
            for i in range(len(embeddings) - 1):
                # コサイン類似度を計算
                sim = cosine_similarity(
                    [embeddings[i]], 
                    [embeddings[i+1]]
                )[0][0]
                similarities.append(sim)
            
            semantic_overlap = sum(similarities) / len(similarities) if similarities else 0.0
        except Exception as e:
            print(f"セマンティックオーバーラップの計算中にエラーが発生しました: {str(e)}")
            semantic_overlap = 0.0
    
    return {
        "overlap_ratio": overlap_ratio,
        "adjacent_overlap": adjacent_overlaps,
        "avg_adjacent_overlap": sum(adjacent_overlaps) / len(adjacent_overlaps) if adjacent_overlaps else 0.0,
        "semantic_overlap": semantic_overlap
    }

# /evaluate/エンドポイントは一括評価に統合されました
# 代わりに/bulk_evaluate/エンドポイントを使用してください

@app.post("/clear_db/")
def clear_db():
    """
    すべてのembeddingモデルのコレクション（DBデータ）を完全削除するAPI。
    主要embeddingモデル（huggingface_bge_small, openai等）すべてをループで削除。
    """
    try:
        if not LOCAL_MODEL_PATH.exists():
            return {
                "status": "error",
                "message": f"モデルが見つかりません: {LOCAL_MODEL_PATH}。DBリセット不可。",
                "model_exists": False
            }
        # 削除対象embeddingモデルリスト
        embedding_models = ["huggingface_bge_small", "gpt-4o"]
        results = []
        for emb_model in embedding_models:
            try:
                if emb_model == "huggingface_bge_small":
                    dummy_embeddings = HuggingFaceEmbeddings(
                        model_name=str(LOCAL_MODEL_PATH),
                        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                elif emb_model == "gpt-4o":
                    from langchain_openai import OpenAIEmbeddings
                    dummy_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
                else:
                    continue
                vectorstore = PGVector.from_documents(
                    documents=[],  # 空のドキュメントで初期化
                    embedding=dummy_embeddings,
                    collection_name=get_collection_name(emb_model)
                )
                vectorstore.delete_collection()
                results.append(f"{emb_model}: 削除成功")
            except Exception as e:
                results.append(f"{emb_model}: 削除失敗 ({str(e)})")
        return {
            "status": "success",
            "message": "全embeddingモデルのコレクションを削除しました。",
            "details": results,
            "model_exists": True
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"DB全体削除時エラー: {str(e)}",
            "model_exists": LOCAL_MODEL_PATH.exists()
        }


@app.post("/admin/reset_vectors_and_chunks/")
def admin_reset_vectors_and_chunks():
    try:
        print(f"[{jst_now_str()}] [WARN] admin_reset_vectors_and_chunks called")
        with engine.begin() as conn:
            try:
                conn.execute(text("DELETE FROM langchain_pg_embedding;"))
            except Exception as e:
                print(f"[{jst_now_str()}] [WARN] failed to delete from langchain_pg_embedding: {e}")
            try:
                conn.execute(text("DELETE FROM langchain_pg_collection;"))
            except Exception as e:
                print(f"[{jst_now_str()}] [WARN] failed to delete from langchain_pg_collection: {e}")
            try:
                conn.execute(text("DELETE FROM pdf_chunks;"))
            except Exception as e:
                print(f"[{jst_now_str()}] [WARN] failed to delete from pdf_chunks: {e}")

        return {
            "status": "success",
            "message": "langchain_pg_embedding, langchain_pg_collection, pdf_chunks の全レコードを削除しました（pdf_files とPDF本体は維持されます）。",
        }
    except Exception as e:
        print(f"[{jst_now_str()}] [ERROR] admin_reset_vectors_and_chunks failed: {e}")
        return {
            "status": "error",
            "message": f"admin_reset_vectors_and_chunks でエラーが発生しました: {str(e)}",
        }


@app.post("/admin/reset_pdfs_and_vectors/")
def admin_reset_pdfs_and_vectors():
    try:
        print(f"[{jst_now_str()}] [WARN] admin_reset_pdfs_and_vectors called")
        pdf_rows = []
        with engine.begin() as conn:
            try:
                pdf_rows = conn.execute(
                    text("SELECT id, storage_path FROM pdf_files;")
                ).fetchall()
            except Exception as e:
                print(f"[{jst_now_str()}] [WARN] failed to select from pdf_files: {e}")
                pdf_rows = []

            for table_name in [
                "pdf_chunks",
                "generated_questions",
                "experiment_results",
                "experiments",
            ]:
                try:
                    conn.execute(text(f"DELETE FROM {table_name};"))
                except Exception as e:
                    print(f"[{jst_now_str()}] [WARN] failed to delete from {table_name}: {e}")

            for table_name in [
                "langchain_pg_embedding",
                "langchain_pg_collection",
            ]:
                try:
                    conn.execute(text(f"DELETE FROM {table_name};"))
                except Exception as e:
                    print(f"[{jst_now_str()}] [WARN] failed to delete from {table_name}: {e}")

            try:
                conn.execute(text("DELETE FROM pdf_files;"))
            except Exception as e:
                print(f"[{jst_now_str()}] [WARN] failed to delete from pdf_files: {e}")

        deleted_files = 0
        deleted_extracted = 0
        for row in pdf_rows:
            try:
                file_id = row[0]
                storage_path = row[1]
            except Exception:
                try:
                    file_id = getattr(row, "id", None)
                    storage_path = getattr(row, "storage_path", None)
                except Exception:
                    file_id = None
                    storage_path = None

            if storage_path:
                try:
                    p = Path(storage_path)
                    if p.exists():
                        p.unlink()
                        deleted_files += 1
                except Exception as e:
                    print(f"[{jst_now_str()}] [WARN] failed to delete storage file {storage_path}: {e}")

            if file_id:
                try:
                    extracted_path = EXTRACTED_DIR / f"{file_id}.json"
                    if extracted_path.exists():
                        extracted_path.unlink()
                        deleted_extracted += 1
                except Exception as e:
                    print(f"[{jst_now_str()}] [WARN] failed to delete extracted json {file_id}: {e}")

        return {
            "status": "success",
            "message": "PDF関連レコード（pdf_files, pdf_chunks など）および PGVector ベクトルストアを削除し、PDFファイルと抽出JSONも削除しました。",
            "deleted_pdf_rows": len(pdf_rows),
            "deleted_storage_files": deleted_files,
            "deleted_extracted_json": deleted_extracted,
        }
    except Exception as e:
        print(f"[{jst_now_str()}] [ERROR] admin_reset_pdfs_and_vectors failed: {e}")
        return {
            "status": "error",
            "message": f"admin_reset_pdfs_and_vectors でエラーが発生しました: {str(e)}",
        }


@app.get("/models/")
def get_available_models():
    """
    利用可能なモデルと現在のモデル状態を返します。
    """
    model_exists = LOCAL_MODEL_PATH.exists()
    model_info = {
        "model_name": str(MODEL_NAME),
        "local_path": str(LOCAL_MODEL_PATH),
        "exists": model_exists,
        "size_mb": (
            sum(f.stat().st_size for f in LOCAL_MODEL_PATH.glob('**/*') if f.is_file()) / (1024 * 1024)
        ) if model_exists else 0
    }
    
    return {
        "llm_models": ["ollama_llama2", "gpt-4o"],
        "embedding_models": ["huggingface_bge_small", "gpt-4o"],
        "current_embedding_model": {
            "name": "huggingface_bge_small",
            "type": "local" if model_exists else "remote",
            "info": model_info
        },
        "environment": {
            "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", "Not set"),
            "hf_home": os.environ.get("HF_HOME", "Not set")
        }
    }

# --- 一括評価API（ダミー実装） ---
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import Request
import asyncio

@app.post("/bulk_evaluate/")
async def bulk_evaluate(request: Request):
    """
    embeddingモデル・チャンク分割パラメータを受けてRAG自動評価を行うAPI。
    Eval.mdの方針に従い、faithfulness等の指標でスコア返却。
    """
    try:
        # --- 数値のNaN/infガード用ユーティリティ ---
        import math
        def safe_val(x):
            try:
                if math.isnan(x) or math.isinf(x):
                    return 0.0
                return float(x)
            except Exception:
                return 0.0

        data = await request.json()
        # --- dataがリスト型なら各要素ごとに個別評価 ---
        def find_first_dict(obj):
            if isinstance(obj, dict):
                return obj
            elif isinstance(obj, list):
                for item in obj:
                    found = find_first_dict(item)
                    if isinstance(found, dict):
                        return found
                return {}

        # 並列処理の最大数を制限するセマフォを作成
        cpu_count = os.cpu_count() or 4
        # CPUコア数に応じてデフォルト並列数を自動調整（最小2、最大8）
        default_parallel = max(2, min(8, max(1, cpu_count // 2)))
        MAX_PARALLEL_TASKS = int(os.getenv("EVAL_MAX_PARALLEL_TASKS", str(default_parallel)))
        semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
        # 計測ログの有効化（環境変数でON/OFF）
        TIMING_LOG = os.getenv("EVAL_TIMING_LOG", "1").lower() in {"1", "true", "yes"}
        import time
        def _tnow():
            return time.monotonic()
        def _tlog(label: str, start: float = None):
            # EVAL_TIMING_LOG が有効なときのみ出力
            if not TIMING_LOG:
                return
            try:
                if start is None:
                    print(f"[timing] {label} at {jst_now_str()}")
                else:
                    dur = _tnow() - start
                    print(f"[timing] {label} took {dur:.3f}s")
            except Exception:
                pass

        async def evaluate_one_bulk(data):
            try:
                print("[進捗] 評価データを処理中...")
                # タイムアウト設定（環境変数で調整可能）
                # 既定値: LLM呼び出し=45秒, 評価全体は質問数に応じて自動調整
                def _parse_timeout_env(key: str, default_seconds: int):
                    val = os.getenv(key, str(default_seconds))
                    if val is None:
                        return default_seconds
                    v = str(val).strip().lower()
                    if v in ("none", "no", "off", "false", "0", "-1"):
                        return None  # 無制限
                    try:
                        return int(v)
                    except Exception:
                        return default_seconds
                LLM_TIMEOUT = _parse_timeout_env("EVAL_LLM_TIMEOUT_SECONDS", 45)
                EVAL_TIMEOUT = None  # 後段で質問数を基に算出
                embedding_model = data.get("embedding_model")
                chunk_methods = data.get("chunk_methods", [data.get("chunk_method", "recursive")])
                chunk_sizes = data.get("chunk_sizes", [data.get("chunk_size", 1000)])
                chunk_overlaps = data.get("chunk_overlaps", [data.get("chunk_overlap", 0)])
                request_llm_model = data.get("llm_model", DEFAULT_LLM_NAME)
                llm_instance_generation, resolved_llm_model = init_generation_llm(request_llm_model, purpose="/bulk_evaluate generation")
                print(f"[設定] 生成LLM={resolved_llm_model}（評価用LLMはGPT-OSS固定）")
                
                # セマンティックチャンキングが選択されている場合の情報メッセージ
                if "semantic" in chunk_methods:
                    if len(chunk_methods) == 1:
                        print("情報: セマンティックチャンキングが選択されました。チャンクサイズとオーバーラップは使用されません。")
                    else:
                        print(f"情報: セマンティックチャンキングとその他のチャンキング方式が同時に選択されています。")
                        print(f"      セマンティックチャンキング: デフォルトパラメータを使用")
                        print(f"      その他の方式: 指定されたチャンクサイズとオーバーラップを使用")

                # 必須パラメータチェック
                sample_text = data.get("text")
                if not sample_text:
                    raise ValueError("textが指定されていません")
                    
                # サポートされているモデルかチェック
                supported_models = {
                    # OpenAIモデル
                    'openai', 'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002',
                    # HuggingFaceモデル
                    'huggingface_bge_small', 'huggingface_bge_large', 'huggingface_miniLM', 'huggingface_mpnet_base',
                    'huggingface_multi_qa_minilm', 'huggingface_multi_qa_mpnet',
                    'huggingface_paraphrase_multilingual', 'huggingface_distiluse_multilingual',
                    'huggingface_xlm_r', 'jina-embeddings-v4',
                    # Ollama埋め込みモデル
                    'nomic-embed-text', 'mxbai-embed-large', 'all-minilm', 'bge-m3', 'qwen3-embedding', 'snowflake-arctic-embed2', 'jina-embeddings-v3',
                }
                
                if embedding_model not in supported_models:
                    raise ValueError(f"未サポートの埋め込みモデルが指定されました: {embedding_model}")
                
                # OpenAI埋め込みの旧指定に対する注意喚起
                if embedding_model == "openai":
                    print("警告: 'openai' は包括的な指定です。具体的な 'text-embedding-3-small' または 'text-embedding-3-large' を選択してください。")

                questions = data.get("questions")
                # ground_truthキーまたはanswersキーのどちらかを使用（互換性のため）
                answers = data.get("ground_truth", data.get("answers"))
                if not questions or not answers:
                    raise ValueError("questions/answersが指定されていません。PDFアップロード時の自動生成結果をそのまま送信してください。")
                if not (sample_text and questions and answers):
                    raise ValueError("PDFアップロードとQA自動生成を先に実施してください（text, questions, answers必須）。")

                def _bool_env(val, default=True):
                    if val is None:
                        return default
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, (int, float)):
                        return bool(val)
                    if isinstance(val, str):
                        v = val.strip().lower()
                        if v in {"1", "true", "yes", "on"}:
                            return True
                        if v in {"0", "false", "no", "off"}:
                            return False
                    return default

                include_answer_similarity = _bool_env(data.get("include_answer_similarity"), True)

                # 質問数に応じて評価タイムアウトのデフォルト値を自動調整
                question_count = len(questions)
                # LLMタイムアウトが無制限の場合は基準を60秒とする
                base_llm_timeout = 60 if LLM_TIMEOUT is None else max(LLM_TIMEOUT, 30)
                # 質問1件あたり約30秒を目安としつつ、全体は最大480秒で抑制
                dynamic_eval_default = max(180, min(480, question_count * 30))
                # LLM呼び出し時間が長い場合を考慮して上限を引き上げ
                dynamic_eval_default = max(dynamic_eval_default, min(600, question_count * base_llm_timeout))
                EVAL_TIMEOUT = _parse_timeout_env("RAGAS_EVAL_TIMEOUT_SECONDS", dynamic_eval_default)

                def _fmt(t):
                    return "no-timeout" if t is None else f"{t}s"
                print(f"[設定] TIMEOUT: LLM_TIMEOUT={_fmt(LLM_TIMEOUT)}, EVAL_TIMEOUT={_fmt(EVAL_TIMEOUT)}, MAX_PARALLEL_TASKS={MAX_PARALLEL_TASKS}")

                results = []
                # embedding_modelのインスタンスを一度だけロードし再利用
                print(f"[進捗] 埋め込みモデル '{embedding_model}' をロード中...")
                _t0_embed = _tnow()
                embedder = get_embeddings(embedding_model)
                _tlog("embedder.load", _t0_embed)
                
                # chunk_method/chunk_size/chunk_overlapごとに完全に独立してチャンク分割→ベクトルストア→retriever→RAG回答生成→評価→スコア集計を実行
                for i in range(len(chunk_methods)):
                    try:
                        chunk_method = chunk_methods[i]
                        print(f"[進捗] チャンク方法 '{chunk_method}' の処理を開始...")
                        
                        # セマンティックチャンキングの場合、チャンクサイズとオーバーラップは無視する
                        if chunk_method == "semantic":
                            if not embedding_model:
                                results.append({
                                    "error": "セマンティックチャンキングにはembedding_modelの指定が必須です", 
                                    "chunk_method": chunk_method
                                })
                                continue
                                
                            print(f"[進捗] セマンティックチャンキングを開始します（chunk_sizeとchunk_overlapは無視されます）...")
                            _t0_chunk = _tnow()
                            
                            # セマンティックチャンキングのパラメータを取得
                            semantic_params = data.get("semantic_params", {})
                            similarity_threshold = float(semantic_params.get("similarity_threshold", 0.7))
                            
                            print(f"[進捗] セマンティックチャンキングを実行: similarity_threshold={similarity_threshold}")
                            text_hash = _hash_text(sample_text)
                            chunk_cache_key = (text_hash, embedding_model or "", similarity_threshold, chunk_method)
                            with _CHUNK_CACHE_LOCK:
                                cached_chunks = _CHUNK_CACHE.get(chunk_cache_key)
                            if cached_chunks is not None:
                                print(f"[CACHE] semantic chunks hit: key={chunk_cache_key} len={len(cached_chunks)}")
                                chunks = cached_chunks
                            else:
                                chunks = await asyncio.to_thread(
                                    semantic_chunk_text,
                                    text=sample_text,
                                    chunk_size=None,  # 無視される
                                    chunk_overlap=None,  # 無視される
                                    embedding_model=embedder,
                                    similarity_threshold=similarity_threshold
                                )
                                with _CHUNK_CACHE_LOCK:
                                    _CHUNK_CACHE[chunk_cache_key] = chunks
                                print(f"[CACHE] semantic chunks store: key={chunk_cache_key} len={len(chunks)}")
                            _tlog(f"chunking.semantic", _t0_chunk)
                            
                            # セマンティックチャンキングの場合はchunk_sizeとchunk_overlapをNoneに設定
                            chunk_size_val = None
                            chunk_overlap_val = None
                            chunk_strategy = "semantic"
                        else:
                            # 通常のチャンキング方法の場合
                            chunk_size = chunk_sizes[i] if i < len(chunk_sizes) else 1000
                            chunk_overlap = chunk_overlaps[i] if i < len(chunk_overlaps) else 200
                            # チャンク分割
                            print(f"[進捗] チャンク分割を実行: 方式={chunk_method}, サイズ={chunk_size}, オーバーラップ={chunk_overlap}")
                            _t0_chunk = _tnow()
                            
                            # 非同期でチャンク分割を実行
                            if chunk_method == "recursive":
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    length_function=len,
                                )
                                chunks = await asyncio.to_thread(text_splitter.split_text, sample_text)
                            elif chunk_method == "fixed":
                                chunks = await asyncio.to_thread(
                                    fixed_chunk_text,
                                    sample_text, 
                                    chunk_size=chunk_size, 
                                    chunk_overlap=chunk_overlap
                                )
                            elif chunk_method == "sentence":
                                chunks = await asyncio.to_thread(sentence_chunk_text, sample_text)
                            elif chunk_method == "paragraph":
                                chunk_func = globals().get("paragraph_chunk_text")
                                if not callable(chunk_func):
                                    print("[警告] paragraph_chunk_text が未定義のため簡易段落分割ロジックを使用します。")

                                    def _paragraph_fallback(text):
                                        if not text:
                                            return []
                                        paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
                                        return paragraphs if paragraphs else [text]

                                    chunk_func = _paragraph_fallback
                                chunks = await asyncio.to_thread(chunk_func, sample_text)
                            # semanticチャンキングは上記のif文で既に処理済み
                            else:
                                raise ValueError(f"未対応のchunk_method: {chunk_method}")
                            _tlog(f"chunking.{chunk_method}", _t0_chunk)
                            
                            # チャンク戦略を設定
                            chunk_size_val = chunk_sizes[i] if i < len(chunk_sizes) else chunk_sizes[0]
                            chunk_overlap_val = chunk_overlaps[i] if i < len(chunk_overlaps) else chunk_overlaps[0]
                            chunk_strategies = data.get("chunk_strategies", []) if isinstance(data, dict) else []
                            if chunk_strategies and i < len(chunk_strategies):
                                chunk_strategy = chunk_strategies[i]
                            else:
                                chunk_strategy = f"{chunk_method}-{chunk_size_val}-{chunk_overlap_val}"

                        print(f"[進捗] {len(chunks)}個のチャンクを作成しました。平均長さ: {sum(len(c) for c in chunks) / max(len(chunks), 1):.1f}文字")
                        print(f"[進捗] ベクトルストアを構築中 ({'FAISS' if _FAISS_AVAILABLE else 'PGVector'})...")
                        _t0_vs = _tnow()
                        chunk_hash = _hash_chunks(chunks)
                        vector_cache_key = (embedding_model or "", chunk_hash, chunk_method)
                        with _VECTORSTORE_CACHE_LOCK:
                            cached_vs = _VECTORSTORE_CACHE.get(vector_cache_key)
                        if cached_vs is not None:
                            print(f"[CACHE] vectorstore hit: key={vector_cache_key}")
                            vectorstore = cached_vs
                        else:
                            if _FAISS_AVAILABLE:
                                vectorstore = await asyncio.to_thread(FAISS.from_texts, texts=chunks, embedding=embedder)
                                _tlog("vectorstore.faiss.build", _t0_vs)
                            else:
                                # PGVectorへのフォールバック
                                print("[警告] FAISSが未インストールのためPGVectorにフォールバックします")
                                vectorstore = PGVector.from_documents(
                                    documents=[],
                                    embedding=embedder,
                                    collection_name=get_collection_name(embedding_model)
                                )
                                await asyncio.to_thread(vectorstore.add_texts, texts=chunks)
                                _tlog("vectorstore.pgvector.build", _t0_vs)
                            with _VECTORSTORE_CACHE_LOCK:
                                _VECTORSTORE_CACHE[vector_cache_key] = vectorstore
                        # 検索パラメータの受け口（既定は従来と互換）
                        top_k = int(data.get("top_k", 5))
                        use_mmr = bool(data.get("use_mmr", False))
                        fetch_k = int(data.get("fetch_k", max(top_k * 2, 20)))
                        try:
                            lambda_mult = float(data.get("lambda_mult", 0.5))
                        except Exception:
                            lambda_mult = 0.5
                        if use_mmr:
                            print(f"[設定] retriever=MMR k={top_k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
                            retriever = vectorstore.as_retriever(
                                search_type="mmr",
                                search_kwargs={"k": top_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
                            )
                        else:
                            print(f"[設定] retriever=similarity k={top_k}")
                            retriever = vectorstore.as_retriever(
                                search_kwargs={"k": top_k},
                            )

                        # RAG回答生成＆コンテキスト取得
                        contexts = []
                        pred_answers = []
                        
                        # PDFアップロード時の回答が揃っていれば使い回し（高速化）
                        if answers and len(answers) == len(questions):
                            print(f"[進捗] PDFアップロード時の回答を使用（{len(answers)}個の回答）")
                            pred_answers = answers  # 回答は使い回し
                            
                            async def get_context_only(q):
                                async with semaphore:
                                    retrieved_docs = await asyncio.to_thread(retriever.get_relevant_documents, q)
                                    return [doc.page_content for doc in retrieved_docs]
                            # 全質問のコンテキストのみ取得
                            _t0_ctx = _tnow()
                            contexts = await asyncio.gather(*[get_context_only(q) for q in questions])
                            _tlog("retrieval.contexts_only", _t0_ctx)
                            print(f"[進捗] コンテキスト取得完了。評価処理を開始...")
                        else:
                            print(f"[進捗] 新しいRAG回答を生成（{len(questions)}個の質問）...")
                            
                            # 各質問に対して非同期でコンテキスト取得と回答生成を行う
                            async def get_context_and_answer(q):
                                async with semaphore:  # セマフォで並列処理数を制限
                                    # 各質問ごとにリトリーバーで文脈取得（非同期化）
                                    retrieved_docs = await asyncio.to_thread(retriever.get_relevant_documents, q)
                                    context_texts = [doc.page_content for doc in retrieved_docs]
                                    # LLMインスタンス・プロンプト生成（GPT-OSS固定）
                                    llm_instance = llm_instance_generation
                                    prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:\n{context}\n\nQuestion: {question}""")
                                    # Ollama ラッパーはLCELのRunnableではないため、RunnableLambdaで委譲して対応
                                    def _to_text(x):
                                        try:
                                            return x.to_string()
                                        except Exception:
                                            return x
                                    # ラッパーのinvokeを確実に通す（型正規化のため）
                                    llm_runnable = RunnableLambda(lambda x: llm_instance.invoke(_to_text(x)))
                                    chain = (
                                        {"context": lambda _: context_texts, "question": lambda _: q}
                                        | prompt
                                        | llm_runnable
                                        | StrOutputParser()
                                    )
                                    # 非同期で回答生成（タイムアウト付与）
                                    try:
                                        # 最初のマッピングでcontext/questionを供給するため空dictで十分
                                        if LLM_TIMEOUT is None:
                                            answer = await chain.ainvoke({})
                                        else:
                                            answer = await asyncio.wait_for(chain.ainvoke({}), timeout=LLM_TIMEOUT)
                                    except asyncio.TimeoutError:
                                        print(f"[警告] LLM回答生成がタイムアウト: model={resolved_llm_model}, timeout={LLM_TIMEOUT}s, question={q[:30]}...")
                                        answer = "[LLMタイムアウト]"
                                    except Exception as e:
                                        print(f"[警告] LLM回答生成失敗: {e}")
                                        answer = "[LLMエラー]"
                                    return context_texts, answer
                            
                            # 非同期で全質問の回答を生成
                            _t0_rag = _tnow()
                            results_list = await asyncio.gather(*[get_context_and_answer(q) for q in questions])
                            _tlog("retrieval+llm_answers", _t0_rag)
                            for context_texts, answer in results_list:
                                contexts.append(context_texts)
                                pred_answers.append(answer)
                            print(f"[進捗] RAG回答生成完了。評価処理を開始...")
                        # --- ここまで並列化 ---

                        # RAGAS等で自動評価
                        print(f"[進捗] 評価メトリクスの計算を開始...")
                        
                        dataset_dict = {
                            "question": questions,
                            "answer": pred_answers,
                            "contexts": contexts,
                            "ground_truth": answers
                        }
                        # 必須カラム 'reference' を追加（answer_correctness 用）
                        dataset_dict_with_ref = dict(dataset_dict)
                        dataset_dict_with_ref["reference"] = answers
                        dataset = Dataset.from_dict(dataset_dict_with_ref)
                        # 評価用LLMもGPT-OSS固定
                        llm_instance_eval = get_llm("gpt-oss")
                        # RAGAS が await するケースに対応する非同期アダプタ
                        ragas_llm = RAGASLLMAsyncAdapter(llm_instance_eval)
                        
                        # --- 全質問を1つのDatasetにまとめて一括評価（ragas側でmax_workers並列化） ---
                        import copy as _copy
                        metric_defs = [
                            ("faithfulness", faithfulness),
                            ("answer_relevancy", answer_relevancy),
                            ("context_recall", context_recall),
                            ("context_precision", context_precision),
                            ("answer_correctness", answer_correctness),
                            ("answer_similarity", answer_similarity),
                        ]
                        selected_metric_defs = []
                        for name, metric in metric_defs:
                            if name == "answer_similarity" and not include_answer_similarity:
                                continue
                            selected_metric_defs.append((name, metric))
                        metrics_local = [_copy.deepcopy(m) for _, m in selected_metric_defs]
                        for m in metrics_local:
                            if hasattr(m, "llm"):
                                m.llm = ragas_llm
                            if hasattr(m, "embeddings"):
                                m.embeddings = embedder
                        # ragas.evaluate は同期関数のため、スレッド実行＋必要に応じてタイムアウトを適用
                        eval_df = None
                        try:
                            _t0_eval = _tnow()
                            if EVAL_TIMEOUT is None:
                                eval_res_all = await asyncio.to_thread(
                                    evaluate,
                                    dataset=dataset,
                                    metrics=metrics_local,
                                    llm=ragas_llm,
                                    embeddings=embedder,
                                    run_config=RunConfig(timeout=EVAL_TIMEOUT, max_workers=MAX_PARALLEL_TASKS),
                                )
                            else:
                                eval_res_all = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        evaluate,
                                        dataset=dataset,
                                        metrics=metrics_local,
                                        llm=ragas_llm,
                                        embeddings=embedder,
                                        run_config=RunConfig(timeout=EVAL_TIMEOUT, max_workers=MAX_PARALLEL_TASKS),
                                    ),
                                    timeout=EVAL_TIMEOUT,
                                )
                            _tlog("ragas.evaluate", _t0_eval)
                            # 代表的な戻り: Resultオブジェクト。to_pandas() があればDataFrame化
                            try:
                                if hasattr(eval_res_all, "to_pandas"):
                                    eval_df = eval_res_all.to_pandas()
                                elif hasattr(eval_res_all, "to_dict") and hasattr(eval_res_all, "columns"):
                                    eval_df = eval_res_all  # 既にDataFrame互換
                                else:
                                    eval_df = None
                            except Exception:
                                eval_df = None
                        except asyncio.TimeoutError:
                            print(f"[警告] ragas.evaluate 一括評価がタイムアウト: timeout={EVAL_TIMEOUT}s")
                            eval_df = None
                        except TypeError:
                            # 互換性問題フォールバック: embeddings を外して実行
                            try:
                                if EVAL_TIMEOUT is None:
                                    eval_res_all = await asyncio.to_thread(
                                        evaluate,
                                        dataset=dataset,
                                        metrics=metrics_local,
                                        llm=ragas_llm,
                                        run_config=RunConfig(timeout=EVAL_TIMEOUT, max_workers=MAX_PARALLEL_TASKS),
                                    )
                                else:
                                    eval_res_all = await asyncio.wait_for(
                                        asyncio.to_thread(
                                            evaluate,
                                            dataset=dataset,
                                            metrics=metrics_local,
                                            llm=ragas_llm,
                                            run_config=RunConfig(timeout=EVAL_TIMEOUT, max_workers=MAX_PARALLEL_TASKS),
                                        ),
                                        timeout=EVAL_TIMEOUT,
                                    )
                                if hasattr(eval_res_all, "to_pandas"):
                                    eval_df = eval_res_all.to_pandas()
                                elif hasattr(eval_res_all, "to_dict") and hasattr(eval_res_all, "columns"):
                                    eval_df = eval_res_all
                                else:
                                    eval_df = None
                            except TypeError:
                                # 最終フォールバック: run_config も外す
                                try:
                                    if EVAL_TIMEOUT is None:
                                        eval_res_all = await asyncio.to_thread(
                                            evaluate,
                                            dataset=dataset,
                                            metrics=metrics_local,
                                            llm=ragas_llm,
                                        )
                                    else:
                                        eval_res_all = await asyncio.wait_for(
                                            asyncio.to_thread(
                                                evaluate,
                                                dataset=dataset,
                                                metrics=metrics_local,
                                                llm=ragas_llm,
                                            ),
                                            timeout=EVAL_TIMEOUT,
                                        )
                                    if hasattr(eval_res_all, "to_pandas"):
                                        eval_df = eval_res_all.to_pandas()
                                    elif hasattr(eval_res_all, "to_dict") and hasattr(eval_res_all, "columns"):
                                        eval_df = eval_res_all
                                    else:
                                        eval_df = None
                                except Exception:
                                    print("[警告] ragas.evaluate 一括評価フォールバック失敗")
                                    eval_df = None

                        # 評価メトリクスの定義と結果整形（answer_similarity を含む）
                        metrics_keys = [name for name, _ in selected_metric_defs]
                        metrics_per_qa = []
                        metrics_avg = {k: 0.0 for k in metrics_keys}
                        try:
                            if eval_df is not None:
                                try:
                                    rows = eval_df.to_dict(orient="records")
                                except Exception:
                                    rows = []
                                for idx_row, r in enumerate(rows):
                                    metric_values = {k: safe_val(r.get(k, 0.0)) for k in metrics_keys}
                                    metrics_per_qa.append({
                                        "question": questions[idx_row] if idx_row < len(questions) else "",
                                        "pred_answer": pred_answers[idx_row] if idx_row < len(pred_answers) else "",
                                        "ground_truth": answers[idx_row] if idx_row < len(answers) else "",
                                        "metrics": metric_values,
                                    })
                                # 平均値を計算（列が無い場合は0）
                                for k in metrics_keys:
                                    try:
                                        if hasattr(eval_df, "columns") and k in list(eval_df.columns):
                                            metrics_avg[k] = safe_val(float(eval_df[k].mean()))
                                        else:
                                            metrics_avg[k] = 0.0
                                    except Exception:
                                        metrics_avg[k] = 0.0
                            else:
                                # タイムアウトや失敗時のフォールバック（質問数ぶんの0レコード）
                                metrics_per_qa = [{
                                    "question": questions[idx] if idx < len(questions) else "",
                                    "pred_answer": pred_answers[idx] if idx < len(pred_answers) else "",
                                    "ground_truth": answers[idx] if idx < len(answers) else "",
                                    "metrics": {k: 0.0 for k in metrics_keys},
                                } for idx in range(len(questions))]
                                metrics_avg = {k: 0.0 for k in metrics_keys}
                        except Exception:
                            # 万一の整形失敗時も0で埋める
                            metrics_per_qa = [{
                                "question": questions[idx] if idx < len(questions) else "",
                                "pred_answer": pred_answers[idx] if idx < len(pred_answers) else "",
                                "ground_truth": answers[idx] if idx < len(answers) else "",
                                "metrics": {k: 0.0 for k in metrics_keys},
                            } for idx in range(len(questions))]
                            metrics_avg = {k: 0.0 for k in metrics_keys}
                        # 総合スコアの計算（重み付けは従来比率を踏襲）
                        overall_score = (
                            metrics_avg["answer_relevancy"] * 0.25 +
                            metrics_avg["faithfulness"] * 0.25 +
                            metrics_avg["context_precision"] * 0.2 +
                            metrics_avg["context_recall"] * 0.2 +
                            metrics_avg["answer_correctness"] * 0.1
                        )
                        overall_score = safe_val(overall_score)
                        
                        # チャンク関連の統計情報
                        num_chunks = len(chunks)
                        avg_chunk_len = int(sum(len(c) for c in chunks) / num_chunks) if num_chunks > 0 else 0
                        
                        # 必須キーのリスト（動的メトリクスを考慮）
                        required_keys = {
                            "overall_score",
                            "avg_chunk_len",
                            "num_chunks",
                        }
                        required_keys.update(metrics_keys)
                        if include_answer_similarity:
                            required_keys.add("answer_similarity")
                        
                        print(f"[進捗] 評価メトリクスの計算が完了しました。総合スコア: {overall_score:.4f}")
                        # 評価結果を格納する辞書を作成
                        response_dict = {
                            "embedding_model": embedding_model,
                            "chunk_size": chunk_size_val if chunk_method != "semantic" else None,
                            "chunk_overlap": chunk_overlap_val if chunk_method != "semantic" else None,
                            "chunk_method": chunk_method,
                            "overall_score": overall_score,
                            "chunk_strategy": chunk_strategy,
                            "num_chunks": num_chunks,
                            "avg_chunk_len": avg_chunk_len,
                            "metrics": metrics_per_qa
                        }

                        for metric_name, metric_value in metrics_avg.items():
                            response_dict[metric_name] = metric_value

                        if include_answer_similarity and "answer_similarity" not in response_dict:
                            response_dict["answer_similarity"] = None
                        
                        # セマンティックチャンキングの場合は類似度閾値を追加
                        if chunk_method == "semantic":
                            response_dict["similarity_threshold"] = similarity_threshold
                        
                        # 必須キーが含まれているか確認、なければデフォルト値を設定
                        for k in required_keys:
                            if k not in response_dict:
                                response_dict[k] = 0.0
                                
                        print(f"[進捗] チャンク方法 '{chunk_method}' の処理が完了しました。スコア: {overall_score:.4f}")
                        results.append(response_dict)
                    except Exception as e:
                        # エラー時も必ずエラー内容を返す
                        import traceback
                        error_detail = traceback.format_exc()
                        print(f"[エラー] チャンク方法 '{chunk_method}' の処理中にエラーが発生しました: {str(e)}")
                        traceback.print_exc()
                        results.append({
                            "error": str(e), 
                            "chunk_method": chunk_method,
                            "error_detail": error_detail,
                            "input_data": data
                        })
                
                print(f"[進捗] すべてのチャンク方法の評価が完了しました。結果数: {len(results)}")
                return results
            except Exception as e:
                # エラー時も必ずエラー内容を返す
                import traceback
                error_detail = traceback.format_exc()
                print(f"[重要エラー] evaluate_one_bulk処理全体で例外が発生: {str(e)}")
                traceback.print_exc()
                return {
                    "error": str(e), 
                    "error_detail": error_detail,
                    "input_data": data
                }

        # --- 本体分岐 ---
        print(f"[進捗] bulk_evaluate APIが呼び出されました")
        if isinstance(data, list):
            print(f"[進捗] リストデータを処理します。データ数: {len(data)}")
            results = []
            for i, d in enumerate(data):
                try:
                    print(f"[進捗] データ {i+1}/{len(data)} を処理中...")
                    if not isinstance(d, dict):
                        d = find_first_dict(d)
                    res = await evaluate_one_bulk(d)
                    results.append(res)
                    print(f"[進捗] データ {i+1}/{len(data)} の処理が完了しました")
                except Exception as e:
                    # 個別データでエラーが発生しても全体を止めず、エラー内容を追加
                    import traceback
                    error_detail = traceback.format_exc()
                    print(f"[エラー] データ {i+1}/{len(data)} の処理中にエラーが発生: {str(e)}")
                    traceback.print_exc()
                    results.append({
                        "error": str(e), 
                        "error_detail": error_detail,
                        "input_data": d
                    })
            print(f"[進捗] すべてのデータ処理が完了しました。結果数: {len(results)}")
            persist_experiment_results(
                pdf_file_id=data[0].get("file_id") if data and isinstance(data[0], dict) else None,
                request_params=data[0] if data and isinstance(data[0], dict) else {},
                results=results,
            )
            return results
        else:
            print(f"[進捗] 単一データを処理します")
            result = await evaluate_one_bulk(data)
            print(f"[進捗] 処理が完了しました")
            persist_experiment_results(
                pdf_file_id=data.get("file_id") if isinstance(data, dict) else None,
                request_params=data if isinstance(data, dict) else {},
                results=result if isinstance(result, list) else [result],
            )
            return result
    except Exception as e:
        # 異常時も辞書を直接返す（JSONResponse不使用）
        import traceback
        error_detail = traceback.format_exc()
        print(f"[重要エラー] bulk_evaluate全体例外: {str(e)}")
        traceback.print_exc()
        return {
            "error": str(e),
            "error_detail": error_detail
        }

# （重複していた下側の uploadfile 実装は削除し、上側の永続化付き実装に一本化しました）

# --- モデル・戦略リスト取得API（YAMLファイルを返す） ---
from fastapi.responses import JSONResponse

@app.get("/list_models")
def list_models():
    """
    models.yamlの内容を {"models": [...]} 形式で返すAPI。エラー時はprintログも出し、説明付きで返却。
    """
    import os
    try:
        # デバッグ用: カレントディレクトリとファイル一覧を表示
        print(f"[DEBUG] os.getcwd() = {os.getcwd()}")
        print(f"[DEBUG] os.listdir('.') = {os.listdir('.')}")
        abs_path = os.path.abspath("models.yaml")
        print(f"[DEBUG] models.yaml abs path = {abs_path}")
        print(f"[DEBUG] models.yaml exists = {os.path.exists(abs_path)}")
        # 読み込み前
        models_dict = load_models_yaml()
        print(f"[DEBUG] models_dict loaded: {models_dict}")
        if not models_dict or "models" not in models_dict:
            print("[list_models ERROR] models.yamlに'models'キーがありません")
            return JSONResponse(status_code=404, content={"error": "models.yamlに'models'キーがありません"})
        
        # モデルをカテゴリー別に分類
        categorized_models = {
            "LLM": [m for m in models_dict["models"] if m.get("category") == "LLM"],
            "Embedding": [m for m in models_dict["models"] if m.get("category") == "Embedding"]
        }
        
        print(f"[DEBUG] categorized_models: {categorized_models}")
        return JSONResponse(content=categorized_models)
    except Exception as e:
        print(f"[list_models ERROR] {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/list_strategies")
def list_strategies():
    """
    strategies.yamlの内容を返すAPI。エラー時は説明付きで返却。
    """
    try:
        strategies = load_strategies_yaml()
        return JSONResponse(content=strategies)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- 管理用: DB・保存ファイルを丸ごと消去（事前バックアップ付き） ---
@app.post("/admin/wipe_all")
def admin_wipe_all(payload: dict):
    """
    破壊的操作: データベースの主要テーブルと /app/data 配下のPDF・抽出JSONを全消去します。
    安全のため、実行には payload {"confirm": true} が必須。既定でバックアップを作成します。

    payload 例:
    {
      "confirm": true,
      "backup": true,
      "delete_files": true
    }
    """
    try:
        confirm = bool(payload.get("confirm"))
        make_backup = bool(payload.get("backup", True))
        delete_files = bool(payload.get("delete_files", True))
        if not confirm:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "confirm=true が必須です（破壊的操作の安全装置）"
            })

        # --- バックアップの作成 ---
        backup_info = {}
        if make_backup:
            import shutil, zipfile, datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backups_dir = DATA_DIR / "backups" / ts
            backups_dir.mkdir(parents=True, exist_ok=True)

            # テーブルダンプ（JSON）
            tables = [
                "embeddings",
                "pdf_files",
                "pdf_chunks",
                "generated_questions",
                "experiments",
                "experiment_results",
            ]
            table_dumps = []
            with engine.begin() as conn:
                for t in tables:
                    try:
                        rows = conn.execute(text(f"SELECT * FROM {t}"))
                        items = [dict(r._mapping) for r in rows]
                        out_path = backups_dir / f"{t}.json"
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(items, f, ensure_ascii=False, default=str)
                        table_dumps.append(str(out_path))
                    except Exception as te:
                        table_dumps.append(f"{t}: dump失敗 ({te})")

            # /app/data をZIPバックアップ（PDF/抽出JSON/画像）
            data_zip = backups_dir / "data_dir.zip"
            try:
                with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                    for subdir in (PDF_DIR, EXTRACTED_DIR, IMAGES_DIR):
                        if subdir.exists():
                            for p in subdir.rglob("*"):
                                if p.is_file():
                                    arcname = p.relative_to(DATA_DIR)
                                    zf.write(p, arcname)
                backup_info = {
                    "tables": table_dumps,
                    "data_zip": str(data_zip),
                    "backup_dir": str(backups_dir),
                }
            except Exception as ze:
                backup_info = {
                    "tables": table_dumps,
                    "data_zip_error": str(ze),
                    "backup_dir": str(backups_dir),
                }

        # --- DB削除（外部キー制約に配慮し順序を決定） ---
        with engine.begin() as conn:
            # 依存の深い順に消す
            conn.execute(text("DELETE FROM experiment_results"))
            conn.execute(text("DELETE FROM experiments"))
            conn.execute(text("DELETE FROM generated_questions"))
            conn.execute(text("DELETE FROM pdf_chunks"))
            conn.execute(text("DELETE FROM pdf_files"))
            try:
                conn.execute(text("DELETE FROM embeddings"))
            except Exception:
                # 旧環境など embeddings が無い場合を許容
                pass

        # --- ファイル削除 ---
        file_details = []
        if delete_files:
            import itertools
            for target_dir in (PDF_DIR, EXTRACTED_DIR):
                if target_dir.exists():
                    for p in list(target_dir.glob("*")):
                        try:
                            if p.is_file():
                                p.unlink()
                                file_details.append(f"delete: {p}")
                            elif p.is_dir():
                                import shutil
                                shutil.rmtree(p)
                                file_details.append(f"rmdir: {p}")
                        except Exception as fe:
                            file_details.append(f"error: {p} ({fe})")

        return JSONResponse(content={
            "status": "success",
            "message": "DBテーブルと保存ファイルを消去しました（必要に応じてバックアップ済み）",
            "backup": backup_info if make_backup else None,
            "file_ops": file_details if delete_files else [],
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
        })

@app.post("/history/import-experiment")
def import_experiment(payload: dict):
    """
    既存の一括評価結果を履歴DB（experiments / experiment_results）に保存するAPI。
    期待するpayload:
      {
        "pdf_file_id": "<optional>",
        "experiment_name": "<optional>",
        "parameters": { ... },
        "results": [
           {
             "embedding_model": str,
             "chunk_strategy": str,
             "chunk_size": int,
             "chunk_overlap": int,
             "num_chunks": int,
             "avg_chunk_len": int,
             "overall_score": float,
             "faithfulness": float,
             "answer_relevancy": float,
             "context_recall": float,
             "context_precision": float,
             "answer_correctness": float,
             "answer_similarity": float,
             "details": dict or str
           }, ...
        ]
      }
    """
    try:
        if not isinstance(payload, dict):
            return JSONResponse(status_code=400, content={"error": "payload must be a JSON object"})

        pdf_file_id = payload.get("pdf_file_id")
        experiment_name = payload.get("experiment_name") or "manual-import"
        parameters = payload.get("parameters") or {}
        results = payload.get("results") or []

        # --- results 正規化: 文字列で配列JSONが来ても自動解釈 ---
        try:
            # case 1: results がそのままJSON文字列
            if isinstance(results, str):
                results = json.loads(results)
            # case 2: [ "[ {...}, {...} ]" ] のような一要素文字列
            elif isinstance(results, list) and len(results) == 1 and isinstance(results[0], str):
                # 先頭要素がJSON配列文字列なら展開
                s0 = results[0].strip()
                if (s0.startswith("[") and s0.endswith("]")) or (s0.startswith("{") and s0.endswith("}")):
                    parsed = json.loads(s0)
                    # 配列でも単体オブジェクトでも配列化
                    results = parsed if isinstance(parsed, list) else [parsed]
            # case 3: dict 単体 -> 配列化
            elif isinstance(results, dict):
                results = [results]
        except Exception as e:
            print(f"[{jst_now_str()}] [警告] results 正規化に失敗: {e}. 生データを使用します。")

        # 型・不足キーの補完
        def _safe_float(x, default=None):
            try:
                if x is None:
                    return default
                return float(x)
            except Exception:
                return default

        def _safe_int(x, default=None):
            try:
                if x is None:
                    return default
                return int(float(x))
            except Exception:
                return default

        norm_results = []
        for r in results if isinstance(results, list) else []:
            if not isinstance(r, dict):
                # 不正形式はスキップ
                continue
            rr = dict(r)
            # chunk_strategy が無ければ chunk_method/size/overlap から補完
            if not rr.get("chunk_strategy"):
                method = rr.get("chunk_method")
                size = rr.get("chunk_size")
                overlap = rr.get("chunk_overlap")
                if isinstance(method, str):
                    if method == "semantic":
                        rr["chunk_strategy"] = "semantic"
                    else:
                        rr["chunk_strategy"] = f"{method}-{_safe_int(size, 0)}-{_safe_int(overlap, 0)}"
            # 数値系の安全変換
            for k in [
                "overall_score","faithfulness","answer_relevancy","context_recall",
                "context_precision","answer_correctness","answer_similarity"
            ]:
                if k in rr:
                    rr[k] = _safe_float(rr.get(k))
            for k in ["chunk_size","chunk_overlap","num_chunks","avg_chunk_len", "overlap"]:
                if k in rr:
                    rr[k] = _safe_int(rr.get(k))
            norm_results.append(rr)

        results = norm_results if norm_results else results

        # persist_experiment_results 内で experiments 行を作成するため、parameters に experiment_name を含めておく
        if isinstance(parameters, dict):
            parameters.setdefault("experiment_name", experiment_name)

        # 保存実行
        persist_experiment_results(pdf_file_id=pdf_file_id, request_params=parameters, results=results)

        # 直近の experiment_id を返却（保存が成功していれば最大IDが対象）
        with engine.begin() as conn:
            row = conn.execute(text("SELECT id FROM experiments ORDER BY id DESC LIMIT 1")).fetchone()
            exp_id = row[0] if row else None

        return JSONResponse(content=jsonable_encoder({
            "status": "ok",
            "experiment_id": exp_id
        }))
    except Exception as e:
        import traceback
        print(f"[{jst_now_str()}] [警告] import_experiment 失敗: {e}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/history/backfill")
def history_backfill(file_id: str | None = None, dry_run: bool = False):
    """
    既存の抽出JSON(EXTRACTED_DIR)とPDF(PDF_DIR)からDBへ再登録（バックフィル）するAPI。
    - file_id を指定しない場合はディレクトリ内の全件を対象
    - dry_run=True の場合は保存せずにプレビューのみ返却
    """
    try:
        targets = []
        if file_id:
            json_path = EXTRACTED_DIR / f"{file_id}.json"
            if json_path.exists():
                targets.append((file_id, json_path))
        else:
            for p in EXTRACTED_DIR.glob("*.json"):
                fid = p.stem
                targets.append((fid, p))

        if not targets:
            return JSONResponse(content={
                "status": "ok",
                "message": "対象データが見つかりません",
                "processed": 0,
                "dry_run": dry_run,
            })

        processed = 0
        details = []
        for fid, jp in targets:
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text = data.get("text") or ""
                questions = data.get("questions") or []
                answers = data.get("answers") or []
                qa_meta = data.get("qa_meta") or []
                file_name = data.get("file_name") or f"{fid}.pdf"
                file_hash = data.get("file_hash")

                pdf_path = PDF_DIR / f"{fid}.pdf"
                file_size = pdf_path.stat().st_size if pdf_path.exists() else 0
                storage_path = str(pdf_path) if pdf_path.exists() else None
                cleanse_used = bool(data.get("cleanse_used", False))
                # 可能ならJSON側の使用モデルを利用、それ以外は既定値
                question_llm_model = data.get("question_llm_used") or data.get("question_llm_model") or "gpt-oss"
                answer_llm_model = data.get("answer_llm_used") or data.get("answer_llm_model") or question_llm_model

                # チャンクが保存されていない場合もあるため、固定長で生成
                chunks = generate_default_chunks_for_storage(text)

                details.append({
                    "file_id": fid,
                    "file_name": file_name,
                    "has_pdf": pdf_path.exists(),
                    "questions": len(questions),
                    "answers": len(answers),
                    "chunks": len(chunks),
                })

                if not dry_run:
                    persist_pdf_upload_to_db(
                        file_id=fid,
                        file_name=file_name,
                        original_name=file_name,
                        file_size=file_size,
                        storage_path=storage_path or "",
                        cleanse_used=cleanse_used,
                        question_llm_model=question_llm_model,
                        answer_llm_model=answer_llm_model,
                        chunks=chunks,
                        questions=questions,
                        answers=answers,
                        qa_meta=qa_meta,
                        file_hash=file_hash,
                    )
                processed += 1
            except Exception as ie:
                details.append({
                    "file_id": fid,
                    "error": str(ie),
                })

        return JSONResponse(content={
            "status": "ok",
            "processed": processed,
            "total_targets": len(targets),
            "dry_run": dry_run,
            "details": details,
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/history/pdf-files")
def history_pdf_files():
    """PDF一覧を返す。"""
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(
                """
                SELECT id, file_name, original_name, file_size, storage_path,
                       cleanse_used, question_llm_model, answer_llm_model, uploaded_at
                FROM pdf_files
                ORDER BY uploaded_at DESC
                """
            )).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/history/pdf-files/{file_id}/questions")
def history_pdf_questions(file_id: str):
    """指定PDFの生成QA一覧を返す。"""
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(
                """
                SELECT id, question, answer, question_model, answer_model, meta_json, created_at
                FROM generated_questions
                WHERE pdf_file_id = :fid
                ORDER BY id ASC
                """
            ), {"fid": file_id}).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/history/pdf-files/{file_id}/chunks")
def history_pdf_chunks(file_id: str):
    """指定PDFのチャンク一覧を返す。"""
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(
                """
                SELECT id, chunk_index, content, content_hash, created_at
                FROM pdf_chunks
                WHERE pdf_file_id = :fid
                ORDER BY chunk_index ASC
                """
            ), {"fid": file_id}).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/history/chat-logs")
def history_chat_logs(pdf_file_id: str | None = None, limit: int = 200):
    try:
        with engine.begin() as conn:
            if pdf_file_id:
                rows = conn.execute(
                    text(
                        """
                        SELECT
                            id,
                            pdf_file_id,
                            user_message,
                            assistant_message,
                            llm_model_used,
                            embedding_model,
                            scope,
                            created_at
                        FROM chat_logs
                        WHERE pdf_file_id = :fid
                        ORDER BY created_at ASC, id ASC
                        LIMIT :limit
                        """
                    ),
                    {"fid": pdf_file_id, "limit": limit},
                ).fetchall()
            else:
                rows = conn.execute(
                    text(
                        """
                        SELECT
                            id,
                            pdf_file_id,
                            user_message,
                            assistant_message,
                            llm_model_used,
                            embedding_model,
                            scope,
                            created_at
                        FROM chat_logs
                        ORDER BY created_at DESC, id DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                ).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/history/experiments")
def history_experiments():
    """実験一覧を返す。"""
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(
                """
                SELECT id, pdf_file_id, experiment_name, parameters, status,
                       total_combinations, completed_combinations,
                       created_at, updated_at
                FROM experiments
                ORDER BY created_at DESC
                """
            )).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/history/experiments/{experiment_id}/results")
def history_experiment_results(experiment_id: int):
    """指定実験の評価結果を返す。"""
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(
                """
                SELECT id, embedding_model, chunk_strategy, chunk_size, chunk_overlap,
                       num_chunks, avg_chunk_len, overall_score, faithfulness,
                       answer_relevancy, context_recall, context_precision,
                       answer_correctness, answer_similarity, details, created_at
                FROM experiment_results
                WHERE experiment_id = :eid
                ORDER BY id ASC
                """
            ), {"eid": experiment_id}).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
