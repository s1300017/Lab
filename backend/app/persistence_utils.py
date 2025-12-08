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


def _safe_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


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

        flat_results: list[dict[str, Any]] = []
        for item in results:
            if isinstance(item, dict):
                flat_results.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        flat_results.append(sub)

        if not flat_results:
            return

        total = len(flat_results)
        completed = sum(1 for r in flat_results if "error" not in r)
        status = "completed" if completed == total else ("failed" if completed == 0 else "partial")
        experiment_name = request_params.get("experiment_name") if isinstance(request_params, dict) else None
        if not experiment_name:
            experiment_name = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        total_duration = 0.0
        duration_count = 0
        duration_summary: dict[str, dict[str, float]] = {
            "llm_models": {},
            "embedding_models": {},
            "chunk_methods": {},
            "chunk_strategies": {},
        }

        def _add_duration(bucket: dict[str, float], key: Any, duration: float) -> None:
            if not key:
                return
            bucket[str(key)] = bucket.get(str(key), 0.0) + duration

        duration_values: list[tuple[float, dict[str, Any]]] = []
        for res in flat_results:
            duration_val = _safe_float(res.get("duration_seconds"))
            if duration_val is None or duration_val < 0:
                continue
            duration_values.append((duration_val, res))
            total_duration += duration_val
            duration_count += 1

            _add_duration(duration_summary["llm_models"], res.get("llm_model"), duration_val)
            _add_duration(duration_summary["embedding_models"], res.get("embedding_model"), duration_val)
            _add_duration(duration_summary["chunk_methods"], res.get("chunk_method"), duration_val)
            _add_duration(duration_summary["chunk_strategies"], res.get("chunk_strategy"), duration_val)

        avg_job_duration = (total_duration / duration_count) if duration_count > 0 else None
        duration_summary_json = json.dumps(duration_summary, ensure_ascii=False) if duration_count > 0 else None

        with engine.begin() as conn:
            result_obj = conn.execute(
                text(
                    """
                    INSERT INTO experiments (
                        pdf_file_id, experiment_name, parameters, status,
                        total_combinations, completed_combinations,
                        total_elapsed_seconds, avg_job_duration_seconds, duration_summary
                    ) VALUES (
                        :pdf_file_id, :experiment_name, :parameters, :status,
                        :total_combinations, :completed_combinations,
                        :total_elapsed_seconds, :avg_job_duration_seconds, :duration_summary
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
                    "total_elapsed_seconds": total_duration if duration_count > 0 else None,
                    "avg_job_duration_seconds": avg_job_duration,
                    "duration_summary": duration_summary_json,
                },
            )
            experiment_id = result_obj.scalar()
            if not experiment_id:
                return

            result_rows: list[dict[str, Any]] = []
            for res in flat_results:
                if not isinstance(res, dict):
                    continue

                is_error = "error" in res
                metrics_list = res.get("metrics", [])
                details_payload: dict[str, Any] = {}
                if isinstance(metrics_list, list) and metrics_list:
                    details_payload["metrics"] = metrics_list

                details_payload["status"] = "error" if is_error else "ok"
                if is_error:
                    if res.get("error") is not None:
                        details_payload["error"] = res.get("error")
                    if res.get("error_detail") is not None:
                        details_payload["error_detail"] = res.get("error_detail")
                    if res.get("input_data") is not None:
                        details_payload["input_data"] = res.get("input_data")

                row: dict[str, Any] = {
                    "experiment_id": experiment_id,
                    "embedding_model": res.get("embedding_model"),
                    "llm_model": res.get("llm_model"),
                    "evaluation_llm_model": res.get("evaluation_llm_model"),
                    "chunk_strategy": res.get("chunk_strategy") or res.get("chunk_method"),
                    "chunk_size": res.get("chunk_size"),
                    "chunk_overlap": res.get("chunk_overlap"),
                    "num_chunks": res.get("num_chunks"),
                    "avg_chunk_len": res.get("avg_chunk_len"),
                    "overall_score": None if is_error else res.get("overall_score"),
                    "faithfulness": None if is_error else res.get("faithfulness"),
                    "answer_relevancy": None if is_error else res.get("answer_relevancy"),
                    "context_recall": None if is_error else res.get("context_recall"),
                    "context_precision": None if is_error else res.get("context_precision"),
                    "answer_correctness": None if is_error else res.get("answer_correctness"),
                    "answer_similarity": None if is_error else res.get("answer_similarity"),
                    "duration_seconds": _safe_float(res.get("duration_seconds")),
                    "details": json.dumps(details_payload, ensure_ascii=False) if details_payload else None,
                }
                result_rows.append(row)

            if result_rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO experiment_results (
                            experiment_id, embedding_model, llm_model, evaluation_llm_model, chunk_strategy, chunk_size, chunk_overlap,
                            num_chunks, avg_chunk_len, overall_score, faithfulness, answer_relevancy,
                            context_recall, context_precision, answer_correctness, answer_similarity, duration_seconds, details
                        ) VALUES (
                            :experiment_id, :embedding_model, :llm_model, :evaluation_llm_model, :chunk_strategy, :chunk_size, :chunk_overlap,
                            :num_chunks, :avg_chunk_len, :overall_score, :faithfulness, :answer_relevancy,
                            :context_recall, :context_precision, :answer_correctness, :answer_similarity, :duration_seconds, :details
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
