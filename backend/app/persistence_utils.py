from __future__ import annotations

from datetime import datetime
from typing import Any

import hashlib
import json
import os
import logging

from pytz import timezone
from sqlalchemy import create_engine, text

from .settings import DB_URL


def jst_now_str() -> str:
    """JST 現在時刻を文字列で返すユーティリティ。"""
    return datetime.now(timezone("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S JST")


logger = logging.getLogger(__name__)


# データベース接続設定（main.py / history_api.py と同等の環境変数を利用）
engine = create_engine(DB_URL)


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
                qa_rows: list[dict[str, Any]] = []
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
        logger.info(
            "[%s] [INFO] PDFアップロード情報をDBに永続化しました (file_id=%s)",
            jst_now_str(),
            file_id,
            extra={"file_id": file_id},
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        logger.warning(
            "[%s] [警告] PDFアップロード情報のDB保存に失敗: %s",
            jst_now_str(),
            e,
            extra={"file_id": file_id},
        )
        logger.debug(traceback.format_exc(), extra={"file_id": file_id})


def persist_experiment_results(
    pdf_file_id: str | None,
    request_params: dict,
    results: list,
) -> None:
    """一括評価結果を `experiments` / `experiment_results` に保存する。"""
    if not isinstance(results, list) or not results:
        return

    try:
        sanitized_params: dict[str, Any] = {}
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
            sanitized_params = {k: v for k, v in request_params.items() if k not in excluded_keys}

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

            result_rows: list[dict[str, Any]] = []
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
        logger.info(
            "[%s] [INFO] 評価結果をDBに保存しました (experiment_id=%s)",
            jst_now_str(),
            experiment_id,
            extra={"experiment_id": experiment_id, "pdf_file_id": pdf_file_id},
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        logger.warning(
            "[%s] [警告] 評価結果のDB保存に失敗: %s",
            jst_now_str(),
            e,
            extra={"pdf_file_id": pdf_file_id},
        )
        logger.debug(traceback.format_exc(), extra={"pdf_file_id": pdf_file_id})
