from __future__ import annotations

from datetime import datetime
from typing import Any

import logging
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pytz import timezone
from sqlalchemy import create_engine, text
from pathlib import Path
import json

from .chunk_utils import generate_default_chunks_for_storage
from .persistence_utils import (
    persist_pdf_upload_to_db,
    persist_experiment_results,
)
from .settings import DB_URL


def jst_now_str() -> str:
    return datetime.now(timezone("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S JST")


logger = logging.getLogger(__name__)


# データベース接続設定（main.py と同等の環境変数を利用）
engine = create_engine(DB_URL)

# PDF・抽出データ保存用ディレクトリ（main.py と同等のルール）
DATA_DIR = Path(__file__).parent.parent / "data"
PDF_DIR = DATA_DIR / "pdf"
EXTRACTED_DIR = DATA_DIR / "extracted"


router = APIRouter()


@router.get("/history/pdf-files")
def history_pdf_files() -> JSONResponse:
    """PDF一覧を返す。"""
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, original_name, file_name, file_size, uploaded_at
                    FROM pdf_files
                    ORDER BY uploaded_at DESC, id DESC
                    LIMIT 1000
                    """
                )
            ).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/history/import-experiment")
def import_experiment(payload: dict) -> JSONResponse:
    """既存の一括評価結果を履歴DB（experiments / experiment_results）に保存するAPI。"""

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
                s0 = results[0].strip()
                if (s0.startswith("[") and s0.endswith("]")) or (s0.startswith("{") and s0.endswith("}")):
                    parsed = json.loads(s0)
                    results = parsed if isinstance(parsed, list) else [parsed]
            # case 3: dict 単体 -> 配列化
            elif isinstance(results, dict):
                results = [results]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[%s] [警告] results 正規化に失敗: %s. 生データを使用します。",
                jst_now_str(),
                e,
            )

        # 型・不足キーの補完
        def _safe_float(x, default=None):
            try:
                if x is None:
                    return default
                return float(x)
            except Exception:  # noqa: BLE001
                return default

        def _safe_int(x, default=None):
            try:
                if x is None:
                    return default
                return int(float(x))
            except Exception:  # noqa: BLE001
                return default

        norm_results = []
        for r in results if isinstance(results, list) else []:
            if not isinstance(r, dict):
                continue
            rr = dict(r)
            if not rr.get("chunk_strategy"):
                method = rr.get("chunk_method")
                size = rr.get("chunk_size")
                overlap = rr.get("chunk_overlap")
                if isinstance(method, str):
                    if method == "semantic":
                        rr["chunk_strategy"] = "semantic"
                    else:
                        rr["chunk_strategy"] = f"{method}-{_safe_int(size, 0)}-{_safe_int(overlap, 0)}"
            for k in [
                "overall_score",
                "faithfulness",
                "answer_relevancy",
                "context_recall",
                "context_precision",
                "answer_correctness",
                "answer_similarity",
            ]:
                if k in rr:
                    rr[k] = _safe_float(rr.get(k))
            for k in ["chunk_size", "chunk_overlap", "num_chunks", "avg_chunk_len", "overlap"]:
                if k in rr:
                    rr[k] = _safe_int(rr.get(k))
            norm_results.append(rr)

        results = norm_results if norm_results else results

        if isinstance(parameters, dict):
            parameters.setdefault("experiment_name", experiment_name)

        persist_experiment_results(pdf_file_id=pdf_file_id, request_params=parameters, results=results)

        with engine.begin() as conn:
            row = conn.execute(text("SELECT id FROM experiments ORDER BY id DESC LIMIT 1")).fetchone()
            exp_id = row[0] if row else None

        return JSONResponse(
            content=jsonable_encoder(
                {
                    "status": "ok",
                    "experiment_id": exp_id,
                }
            )
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        logger.warning("[%s] [警告] import_experiment 失敗: %s", jst_now_str(), e)
        logger.debug(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/history/backfill")
def history_backfill(file_id: str | None = None, dry_run: bool = False) -> JSONResponse:
    """既存の抽出JSON(EXTRACTED_DIR)とPDF(PDF_DIR)からDBへ再登録（バックフィル）するAPI。"""

    try:
        targets: list[tuple[str, Path]] = []
        if file_id:
            json_path = EXTRACTED_DIR / f"{file_id}.json"
            if json_path.exists():
                targets.append((file_id, json_path))
        else:
            for p in EXTRACTED_DIR.glob("*.json"):
                fid = p.stem
                targets.append((fid, p))

        if not targets:
            return JSONResponse(
                content={
                    "status": "ok",
                    "message": "対象データが見つかりません",
                    "processed": 0,
                    "dry_run": dry_run,
                }
            )

        processed = 0
        details: list[dict[str, Any]] = []
        for fid, jp in targets:
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text_data = data.get("text") or ""
                questions = data.get("questions") or []
                answers = data.get("answers") or []
                qa_meta = data.get("qa_meta") or []
                file_name = data.get("file_name") or f"{fid}.pdf"
                file_hash = data.get("file_hash")

                pdf_path = PDF_DIR / f"{fid}.pdf"
                file_size = pdf_path.stat().st_size if pdf_path.exists() else 0
                storage_path = str(pdf_path) if pdf_path.exists() else None
                cleanse_used = bool(data.get("cleanse_used", False))
                question_llm_model = (
                    data.get("question_llm_used")
                    or data.get("question_llm_model")
                    or "gpt-oss"
                )
                answer_llm_model = (
                    data.get("answer_llm_used")
                    or data.get("answer_llm_model")
                    or question_llm_model
                )

                chunks = generate_default_chunks_for_storage(text_data)

                details.append(
                    {
                        "file_id": fid,
                        "file_name": file_name,
                        "has_pdf": pdf_path.exists(),
                        "questions": len(questions),
                        "answers": len(answers),
                        "chunks": len(chunks),
                    }
                )

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
            except Exception as ie:  # noqa: BLE001
                details.append(
                    {
                        "file_id": fid,
                        "error": str(ie),
                    }
                )

        return JSONResponse(
            content={
                "status": "ok",
                "processed": processed,
                "total_targets": len(targets),
                "dry_run": dry_run,
                "details": details,
            }
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        logger.error("[%s] [ERROR] history_backfill 失敗: %s", jst_now_str(), e)
        logger.debug(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/history/pdf-files/{file_id}/questions")
def history_pdf_questions(file_id: str) -> JSONResponse:
    """指定PDFの生成QA一覧を返す。"""
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, question, answer, question_model, answer_model, meta_json, created_at
                    FROM generated_questions
                    WHERE pdf_file_id = :fid
                    ORDER BY id ASC
                    """
                ),
                {"fid": file_id},
            ).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/history/pdf-files/{file_id}/chunks")
def history_pdf_chunks(file_id: str) -> JSONResponse:
    """指定PDFのチャンク一覧を返す。"""
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, chunk_index, content, content_hash, created_at
                    FROM pdf_chunks
                    WHERE pdf_file_id = :fid
                    ORDER BY chunk_index ASC
                    """
                ),
                {"fid": file_id},
            ).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/history/chat-logs")
def history_chat_logs(pdf_file_id: str | None = None, limit: int = 200) -> JSONResponse:
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
                            request_id,
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
                            request_id,
                            created_at
                        FROM chat_logs
                        ORDER BY created_at DESC, id DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                ).fetchall()

            items: list[dict[str, Any]] = []
            for r in rows:
                base = dict(r._mapping)
                contexts_list: list[str] = []
                try:
                    ctx_rows = conn.execute(
                        text(
                            """
                            SELECT context_index, content
                            FROM chat_contexts
                            WHERE chat_log_id = :cid
                            ORDER BY context_index ASC
                            """
                        ),
                        {"cid": base.get("id")},
                    ).fetchall()
                    for cr in ctx_rows:
                        m = cr._mapping
                        content = m.get("content")
                        if isinstance(content, str):
                            contexts_list.append(content)
                except Exception:  # noqa: BLE001
                    contexts_list = []
                base["contexts"] = contexts_list
                items.append(base)

            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/history/experiments")
def history_experiments() -> JSONResponse:
    """実験一覧を返す。"""
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, pdf_file_id, experiment_name, parameters, status,
                           total_combinations, completed_combinations,
                           created_at, updated_at
                    FROM experiments
                    ORDER BY created_at DESC
                    """
                )
            ).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/history/experiments/{experiment_id}/results")
def history_experiment_results(experiment_id: int) -> JSONResponse:
    """指定実験の評価結果を返す。"""
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, embedding_model, chunk_strategy, chunk_size, chunk_overlap,
                           num_chunks, avg_chunk_len, overall_score, faithfulness,
                           answer_relevancy, context_recall, context_precision,
                           answer_correctness, answer_similarity, details, created_at
                    FROM experiment_results
                    WHERE experiment_id = :eid
                    ORDER BY id ASC
                    """
                ),
                {"eid": experiment_id},
            ).fetchall()
            items = [dict(r._mapping) for r in rows]
            return JSONResponse(content=jsonable_encoder({"items": items}))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e)})
