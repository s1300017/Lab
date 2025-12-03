from datetime import datetime
from pytz import timezone
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import hashlib
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from .ollama_config import (
    get_inference_mode,
    set_inference_mode,
    get_ollama_base_url,
)
from .persistence_utils import (
    persist_pdf_upload_to_db,
    persist_experiment_results,
)
from .llm_ragas_utils import (
    _extract_answer_text,
    build_rag_answer_prompt,
    split_sentences,
    extract_relevant_context,
    evaluate_answer_quality,
    regenerate_answer_with_context,
    RAGASCompatibleOllamaLLM,
    RAGASLLMAsyncAdapter,
    RAGASCompatibleOllamaEmbeddings,
    RAGASCompatibleHuggingFaceEmbeddings,
    RAGASCompatibleOpenAIEmbeddings,
)
from .ocr_utils import (
    extract_pdf_text_layout,
    run_deepseek_ocr_via_ollama_for_pdf,
    generate_image_captions_from_pdf,
    _run_deepseek_ocr_for_image_bytes,
    cleanse_pdf_text,
)
from .chunk_utils import (
    fixed_chunk_text,
    sentence_chunk_text,
    paragraph_chunk_text,
    semantic_chunk_text,
    generate_default_chunks_for_storage,
)
from .settings import (
    DB_URL,
    DATA_DIR,
    PDF_DIR,
    EXTRACTED_DIR,
    IMAGES_DIR,
    DEFAULT_LLM_NAME,
    SUPPORTED_EMBEDDING_MODELS,
    resolve_ocr_image_compression,
    get_ollama_deepseek_timeout,
    get_ollama_image_caption_env_defaults,
    REQUEST_ID_CTX,
)


logger = logging.getLogger(__name__)


def jst_now_str():
    return datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')


from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from contextlib import asynccontextmanager
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
logger.info("[%s] === FastAPI main.py 起動開始 [テスト用] ===", jst_now_str())

"""FastAPI アプリケーションのエントリポイント。"""

# データベース接続設定（settings.DB_URL を利用）
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフスパン管理。起動時にDB初期化とモデルウォームアップを実施する。"""
    logger.debug("[%s] [DEBUG] startup_event呼び出し", jst_now_str())
    logger.debug("[%s] [DEBUG] DB_URL = %s", jst_now_str(), os.getenv("DATABASE_URL"))

    # データベース接続をテスト
    max_retries = 5
    retry_delay = 5  # 秒

    for attempt in range(max_retries):
        try:
            logger.debug(
                "[%s] [DEBUG] データベース接続を試行中... (試行 %d/%d)",
                jst_now_str(),
                attempt + 1,
                max_retries,
            )
            init_db()
            logger.debug("[%s] [DEBUG] データベース初期化に成功しました", jst_now_str())

            # 一括評価ジョブの中途半端な状態を起動時にクリーンアップ
            try:
                from . import evaluation_job_service

                evaluation_job_service.cleanup_stale_bulk_jobs_on_startup()
            except Exception as ce:  # noqa: BLE001
                logger.error(
                    "[%s][ERROR] 起動時クリーンアップ処理中にエラーが発生しました: %s",
                    jst_now_str(),
                    ce,
                )
            break
        except Exception as e:
            logger.error(
                "[%s] [ERROR] データベース初期化エラー (試行 %d/%d): %s",
                jst_now_str(),
                attempt + 1,
                max_retries,
                e,
            )
            if attempt == max_retries - 1:
                logger.critical(
                    "[%s] [CRITICAL] データベース初期化に失敗しました。最大試行回数に達しました。",
                    jst_now_str(),
                )
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
                        logger.info("[ウォームアップ] %s 成功", body["model"])
                    except Exception as we:
                        logger.warning("[ウォームアップ警告] %s 失敗: %s", body.get("model"), we)
            except Exception as e:
                logger.error("[ウォームアップ初期化失敗] %s", e)

        threading.Thread(target=_warmup, daemon=True).start()
    except Exception as e:
        logger.error("[ウォームアップ起動失敗] %s", e)

    # lifespan の終了時に特別なクリーンアップは不要
    try:
        yield
    finally:
        pass


# FastAPIアプリケーションの初期化
app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """各HTTPリクエストに一意の request_id を割り当てて ContextVar に保存するミドルウェア。"""

    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = REQUEST_ID_CTX.set(request_id)
    try:
        response = await call_next(request)
    finally:
        REQUEST_ID_CTX.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response

# サブルーターの読み込み
try:
    from .admin_api import router as admin_router
    from .history_api import router as history_router
    from .config_api import router as config_router
    from .chat_api import router as chat_router
    from .upload_api import router as upload_router
    from .evaluation_job_api import router as evaluation_job_router

    app.include_router(admin_router)
    app.include_router(history_router)
    app.include_router(config_router)
    app.include_router(chat_router)
    app.include_router(upload_router)
    app.include_router(evaluation_job_router)
except Exception as e:  # noqa: BLE001
    logger.warning("[%s] [WARN] ルーター読み込みに失敗しました: %s", jst_now_str(), e)


# --- Dockerヘルスチェック用エンドポイント ---
@app.get("/health")
def health_check():
    """Docker用のシンプルなヘルスチェックAPI"""
    return {"status": "ok"}

def init_db():
    logger.debug("[%s] [DEBUG] init_db呼び出し", jst_now_str())
    try:
        logger.debug("[%s] [DEBUG] データベース接続テスト開始", jst_now_str())
        with engine.connect() as conn:
            logger.debug("[%s] [DEBUG] データベース接続成功", jst_now_str())
            
            # トランザクションを開始
            with conn.begin():
                # テーブルが存在するか確認
                result = conn.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'embeddings');"
                ))
                table_exists = result.scalar()
                
                if not table_exists:
                    logger.info("[%s] [INFO] embeddingsテーブルを作成します", jst_now_str())
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
                    logger.info("[%s] [INFO] embeddingsテーブルを作成しました", jst_now_str())
                else:
                    logger.info("[%s] [INFO] embeddingsテーブルは既に存在します", jst_now_str())

                # --- PDF履歴用テーブル作成（存在しない場合のみ実行） ---
                logger.info("[%s] [INFO] PDF・評価履歴用テーブルを確認します", jst_now_str())
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

                # RAGAS一括評価ジョブ状態管理テーブル（evaluation_jobs）を作成
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS evaluation_jobs (
                        job_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        progress TEXT,
                        cancel_requested BOOLEAN DEFAULT FALSE,
                        error TEXT,
                        result_json TEXT,
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
                        request_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    ALTER TABLE chat_logs
                    ADD COLUMN IF NOT EXISTS request_id TEXT
                """))

                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_contexts (
                        id SERIAL PRIMARY KEY,
                        chat_log_id INTEGER REFERENCES chat_logs(id) ON DELETE CASCADE,
                        context_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
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
        logger.error("[%s] [ERROR] データベース初期化エラー: %s", jst_now_str(), e)
        # エラーの詳細をログに出力
        import traceback
        logger.error("[%s] [ERROR] スタックトレース:\n%s", jst_now_str(), traceback.format_exc())
        raise
    for route in app.routes:
        logger.info("[%s] [ROUTE] %s %s", jst_now_str(), route.path, route.methods)

# サーバ起動時にルート一覧を出力
import threading

def print_routes():
    import time
    time.sleep(2)  # サーバ起動待ち
    logger.info("[%s] === FastAPI登録ルート一覧 ===", jst_now_str())
    for route in app.routes:
        logger.info("[%s] [ROUTE] %s %s", jst_now_str(), route.path, route.methods)
threading.Thread(target=print_routes, daemon=True).start()

# --- PDFアップロード＆QA自動生成API ---
from fastapi import UploadFile, File


# --- PDF削除API ---
@app.delete("/pdf/{file_id}")
def delete_pdf(file_id: str):
    """指定されたPDFのDBレコードとファイルを削除する。"""
    logger.info("[%s][INFO] delete_pdf呼び出し: file_id=%s", jst_now_str(), file_id)
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
                logger.info("[%s][INFO] ストレージPDF削除完了: %s", jst_now_str(), storage_path)
        except Exception as file_err:
            logger.warning("[%s][警告] ストレージPDF削除失敗: %s", jst_now_str(), file_err)

        extracted_path = EXTRACTED_DIR / f"{file_id}.json"
        if extracted_path.exists():
            try:
                extracted_path.unlink()
                logger.info("[%s][INFO] 抽出データ削除完了: %s", jst_now_str(), extracted_path)
            except Exception as json_err:
                logger.warning("[%s][警告] 抽出データ削除失敗: %s", jst_now_str(), json_err)

        # 画像化したPDFページのディレクトリも削除
        images_dir = IMAGES_DIR / file_id
        if images_dir.exists():
            try:
                import shutil
                shutil.rmtree(images_dir)
                logger.info("[%s][INFO] 画像ディレクトリ削除完了: %s", jst_now_str(), images_dir)
            except Exception as img_err:
                logger.warning("[%s][警告] 画像ディレクトリ削除失敗: %s", jst_now_str(), img_err)

        return {"status": "deleted", "file_id": file_id}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.warning("[%s][警告] delete_pdfで例外: %s", jst_now_str(), e)
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail="PDF削除中にエラーが発生しました。")

@app.post("/ocr/image")
async def ocr_image(file: UploadFile = File(...)):
    """単一画像ファイルに対して DeepSeek-OCR (Ollama) を実行するAPI。

    CLIスクリプト(ocr.py)と同等の挙動を提供する簡易エンドポイント。
    """

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="画像ファイルが空です。")

    # 画像圧縮とタイムアウトは settings 経由で環境変数込みで解決
    _, resize_max, jpeg_quality = resolve_ocr_image_compression("balanced")
    timeout = get_ollama_deepseek_timeout(300)

    try:
        text = _run_deepseek_ocr_for_image_bytes(
            contents,
            resize_max=resize_max,
            jpeg_quality=jpeg_quality,
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("[%s][ERROR] /ocr/image DeepSeek OCR失敗: %s", jst_now_str(), e)
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

    # 環境変数からページ数とタイムアウト、モデル名・プロンプトのデフォルト値を解決
    max_pages, timeout, caption_model, caption_prompt = get_ollama_image_caption_env_defaults(
        default_max_pages=max_pages,
        default_timeout=timeout,
    )

    try:
        pages = convert_from_bytes(contents, dpi=dpi)
    except Exception as e:
        logger.warning("[警告] pdf2imageによるページ画像化に失敗: %s. 画像キャプション処理をスキップします。", e)
        return captions

    if max_pages is not None:
        pages = pages[:max_pages]

    base_url = get_ollama_base_url().rstrip('/')
    chat_url = f"{base_url}/api/chat"

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
            logger.warning(
                "[警告] llava:7bによる画像キャプション生成に失敗 (page=%d): %s", idx + 1, e)

    return captions

# --- 新規: file_idで抽出済みデータ取得API ---
from fastapi import HTTPException


@app.get("/get_extracted/{file_id}")
def get_extracted(file_id: str):  # ルーター移行済みダミー
    raise HTTPException(status_code=500, detail="/get_extracted は現在利用できません。upload_api 経由でアクセスしてください。")


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
        logger.info(
            "[%s][重要] generate_qa: 抽出テキスト長=%d  サンプル=%s...",
            jst_now_str(),
            len(text),
            sample_text[:100],
            extra={"file_id": file_id},
        )

        # 画像キャプション（必要に応じて生成）
        image_captions: list[dict] = []
        if generate_image_captions:
            try:
                image_captions = generate_image_captions_from_pdf(contents)
            except Exception as e:
                logger.warning(
                    "[警告] generate_qa: 画像キャプション生成中に例外: %s. キャプションなしで続行します。",
                    e,
                    extra={"file_id": file_id},
                )

        # MLX由来のキャプションはここでは再計算しない
        extracted_captions = ""
        mlx_image_captions: list[dict] = []

        # 質問生成（元uploadfileと同じプロンプト）
        logger.info("[重要] generate_qa: LLM質問生成開始", extra={"file_id": file_id})
        llm_q_instance, resolved_question_llm = init_generation_llm(
            question_llm_model,
            purpose="question_generation",
            temperature=0.2,
            top_p=0.85,
            num_predict=320,
            max_tokens=320,
        )
        logger.info("[重要] generate_qa: 質問生成LLM=%s", resolved_question_llm, extra={"file_id": file_id})

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
            logger.info(
                "[重要] generate_qa: LLM質問生成レスポンス長=%d",
                len(raw_questions_text),
                extra={"file_id": file_id},
            )
            questions = [q.strip() for q in raw_questions_text.split("\n") if q.strip()]
            logger.info(
                "[重要] generate_qa: 質問リスト生成完了 件数=%d",
                len(questions),
                extra={"file_id": file_id},
            )
        except Exception as e:
            logger.error("[重要] generate_qa: LLM質問生成例外: %s", e, extra={"file_id": file_id})
            questions = []

        answers: list[str] = []
        llm_a_instance = None
        resolved_answer_llm: Optional[str] = None

        if not questions:
            logger.info("[重要] generate_qa: 正規表現によるQA/箇条書き抽出開始", extra={"file_id": file_id})
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
            logger.info("[重要] generate_qa: 回答生成LLM=%s", resolved_answer_llm, extra={"file_id": file_id})

            for i, q in enumerate(questions):
                try:
                    prompt_a = build_rag_answer_prompt(sample_text, q)
                    answer_resp = llm_a_instance.invoke(prompt_a, max_tokens=640)
                    normalized_answer = _extract_answer_text(answer_resp)
                    answer = normalized_answer.strip().split("\n")[0]
                    if answer and answer[-1] not in {"。", "！", "？", ".", "!", "?"}:
                        answer = f"{answer}。"
                    logger.info(
                        "[重要] generate_qa: LLM回答%d生成完了 文字数=%d",
                        i + 1,
                        len(answer),
                        extra={"file_id": file_id},
                    )
                    answers.append(answer)
                except Exception as e:
                    import traceback
                    logger.error("[重要] generate_qa: LLM回答%d生成例外: %s", i + 1, e, extra={"file_id": file_id})
                    traceback.print_exc()
                    answers.append("該当内容を本文から要約してください。")

        if len(questions) < 5:
            logger.warning(
                "[警告] generate_qa: 質問数が不足 (%d件)。フォールバックで補完します。",
                len(questions),
                extra={"file_id": file_id},
            )
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
            logger.info("[重要] generate_qa: フォールバック質問を追加 %s", fallback_questions, extra={"file_id": file_id})
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
                        logger.error("[重要] generate_qa: フォールバック質問の回答生成例外: %s", e, extra={"file_id": file_id})
                        traceback.print_exc()
                        fallback_answer = "本文を要約してください。"
                else:
                    fallback_answer = "本文を要約してください。"
                answers.append(fallback_answer)

        if not questions or not answers:
            logger.warning("[重要] generate_qa: ダミーQAセットを返却（questions/answersが空）", extra={"file_id": file_id})
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
            logger.warning("[警告] generate_qa: qa_meta生成時に例外: %s。全件デフォルト値を設定します", e, extra={"file_id": file_id})
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
        logger.info("[重要] generate_qa: チャンク数=%d", len(chunks_for_storage), extra={"file_id": file_id})

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
    except Exception as e:
        import traceback
        logger.error("[%s][重要] generate_qa全体例外: %s", jst_now_str(), e, extra={"file_id": file_id})
        logger.debug(traceback.format_exc(), extra={"file_id": file_id})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pdf/{file_id}/generate_qa")
def generate_qa_for_pdf(  # ルーター移行済みダミー
    file_id: str,
    question_llm_model: str = Form("mistral"),
    answer_llm_model: str = Form("mistral"),
):
    raise HTTPException(status_code=500, detail="/pdf/{file_id}/generate_qa は現在利用できません。upload_api 経由でアクセスしてください。")


from pydantic import BaseModel
import io
import os
import sys
logger.critical("[CRITICAL] main.pyロード開始")
from pathlib import Path

# 画像キャプション生成・PDF画像化用の追加インポート（上部で集約）
import base64
import tempfile
import requests
from pdf2image import convert_from_bytes
from PIL import Image

from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import os
# NLTK参照パスを明示的に複数指定
nltk.data.path = ['/usr/local/share/nltk_data', '/usr/local/lib/nltk_data'] + nltk.data.path
logger.debug("[NLTK] data search path: %s", nltk.data.path)
# punktを明示的にダウンロード
if os.getenv("NLTK_AUTO_DOWNLOAD", "0").lower() in {"1", "true", "yes"}:
    nltk.download('punkt', download_dir='/usr/local/share/nltk_data')

from langchain_openai import ChatOpenAI
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
# モデルパスの設定
MODEL_NAME = "BAAI/bge-small-en-v1.5"
LOCAL_MODEL_PATH = Path("/app/models/BAAI_bge-small-en-v1.5")

# セキュリティ注意: 本番環境ではAPIキーの表示は避けてください
_openai_key = os.getenv('OPENAI_API_KEY')
logger.info("[起動時] OPENAI_API_KEY: 設定あり" if _openai_key else "[起動時] OPENAI_API_KEY: 未設定")


# データベース接続設定（settings.DB_URL を既に利用しているため、PGVECTOR 用の環境変数設定のみ維持）
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
        logger.warning("[WARN] 未対応のLLM '%s' が指定されました。%sへフォールバックします。", model_name, DEFAULT_LLM_NAME)
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
        logger.info("[INFO] generation LLM (Ollama) = %s @ %s", entry["model"], ollama_base_url)
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
        logger.info("[INFO] generation LLM (OpenAI) = %s", entry["model"])
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
        logger.warning("[WARN] %s LLM '%s' 初期化失敗: %s. %sへフォールバックします。", purpose, target, e, DEFAULT_LLM_NAME)
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
    logger.info("[INFO] get_torch_device: 利用デバイス = %s", device)
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
        logger.info("[CACHE] get_embeddings hit: %s", model_name)
        return cached

    logger.info("[CACHE] get_embeddings miss: %s -> loading", model_name)
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


# --- API Endpoints ---


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


# --- 一括評価API（ダミー実装） ---
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import Request
import asyncio


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


async def bulk_evaluate(request: Request):
    """
    embeddingモデル・チャンク分割パラメータを受けてRAG自動評価を行うAPI。
    Eval.mdの方針に従い、faithfulness等の指標でスコア返却。
    """
    from .evaluation_service import bulk_evaluate as eval_bulk

    data = await request.json()
    return await eval_bulk(data)

# （重複していた下側の uploadfile 実装は削除し、上側の永続化付き実装に一本化しました）
