from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import asyncio
import json
import logging
import threading
import time
import uuid

from fastapi import HTTPException
from sqlalchemy import create_engine, text

from .settings import DB_URL
from . import evaluation_service


logger = logging.getLogger(__name__)

# 評価ジョブ状態管理用のDBエンジン
engine = create_engine(DB_URL)


class BulkJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class BulkJobCancelled(Exception):
    """一括評価ジョブのユーザーキャンセルを表す内部例外。"""


@dataclass
class BulkJobState:
    job_id: str
    status: BulkJobStatus = BulkJobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_requested: bool = False
    progress: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None


_BULK_JOBS: Dict[str, BulkJobState] = {}
_BULK_JOBS_LOCK = threading.Lock()


def cleanup_stale_bulk_jobs_on_startup() -> None:
    """サーバ再起動時に、中途半端な一括評価ジョブをERROR状態に遷移させる。

    - コンテナ/プロセスの再起動後は、メモリ上のジョブワーカーは必ず失われているため、
      DB上で status が pending / running のジョブはすべて「実行中断」とみなす。
    - これらのジョブを一括で status=error に更新し、エラーメッセージを付与する。
    """

    from .main import jst_now_str  # 遅延インポート

    sql = text(
        """
        UPDATE evaluation_jobs
        SET
            status = 'error',
            error = COALESCE(error, 'サーバ再起動によりジョブが中断されました。'),
            updated_at = CURRENT_TIMESTAMP
        WHERE status IN ('pending', 'running')
        """
    )

    try:
        with engine.begin() as conn:
            result = conn.execute(sql)
            affected = result.rowcount if hasattr(result, "rowcount") else None
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[%s][ERROR] 起動時の一括評価ジョブクリーンアップ中にエラー: %s",
            jst_now_str(),
            e,
        )
    else:
        logger.info(
            "[%s][INFO] 起動時クリーンアップ: pending/running ジョブを error に更新しました (件数=%s)",
            jst_now_str(),
            affected,
        )


def _db_create_bulk_job(job_id: str, status: BulkJobStatus, progress: str) -> None:
    """evaluation_jobs にジョブを登録する。"""
    from .main import jst_now_str  # 遅延インポート

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO evaluation_jobs (
                        job_id, status, progress, cancel_requested, created_at, updated_at
                    ) VALUES (
                        :job_id, :status, :progress, FALSE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                    )
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
            "[%s][ERROR] evaluation_jobs insert/update失敗: %s",
            jst_now_str(),
            e,
            extra={"job_id": job_id},
        )


def _db_update_bulk_job(
    job_id: str,
    *,
    status: Optional[BulkJobStatus] = None,
    progress: Optional[str] = None,
    cancel_requested: Optional[bool] = None,
    error: Optional[str] = None,
    result: Optional[Any] = None,
) -> None:
    """evaluation_jobs の状態を更新する。"""
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

    if not fields:
        return

    set_clause = ", ".join(fields) + ", updated_at = CURRENT_TIMESTAMP"
    sql = f"UPDATE evaluation_jobs SET {set_clause} WHERE job_id = :job_id"
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), params)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "[%s][ERROR] evaluation_jobs更新失敗: %s",
            jst_now_str(),
            e,
            extra={"job_id": job_id},
        )


def get_bulk_job(job_id: str) -> Optional[BulkJobState]:
    with _BULK_JOBS_LOCK:
        return _BULK_JOBS.get(job_id)


def request_cancel_bulk_job(job_id: str) -> dict[str, Any]:
    """指定された一括評価ジョブにキャンセルフラグを立てる。"""
    from .main import jst_now_str  # 遅延インポート

    # まずメモリ上のジョブ状態を確認
    with _BULK_JOBS_LOCK:
        job = _BULK_JOBS.get(job_id)
        if not job:
            # メモリ上にジョブが無い場合は、DB上の状態のみ確認する
            try:
                with engine.connect() as conn:
                    row = (
                        conn.execute(
                            text(
                                """
                                SELECT status
                                FROM evaluation_jobs
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
                        status_code=404,
                        detail="指定されたジョブIDは存在しません。",
                    )
                status_str = str(row["status"]).lower()
                if status_str in {"completed", "error", "cancelled"}:
                    raise HTTPException(
                        status_code=400,
                        detail="このジョブは既に完了または終了しています。",
                    )
            except HTTPException:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "[%s][ERROR] request_cancel_bulk_job DB確認中にエラー: %s",
                    jst_now_str(),
                    e,
                    extra={"job_id": job_id},
                )
                raise HTTPException(
                    status_code=500,
                    detail="ジョブキャンセル処理中にサーバエラーが発生しました。",
                ) from e

            # メモリ上にワーカーが存在しない（=すでに実行終了 or プロセス再起動）のため、実質キャンセル不可
            raise HTTPException(
                status_code=400,
                detail="このジョブは現在実行中ではないためキャンセルできません。",
            )

        if job.status in {BulkJobStatus.COMPLETED, BulkJobStatus.ERROR}:
            raise HTTPException(
                status_code=400,
                detail="このジョブは既に完了または終了しています。",
            )

        job.cancel_requested = True
        job.updated_at = time.time()

    _db_update_bulk_job(
        job_id,
        cancel_requested=True,
        progress="キャンセル要求を受け付けました。安全な停止を待機しています…",
    )

    logger.info(
        "[%s][INFO] bulk_job にキャンセル要求を登録しました: job_id=%s",
        jst_now_str(),
        job_id,
        extra={"job_id": job_id, "component": "evaluation", "endpoint": "bulk_job"},
    )
    return {
        "job_id": job_id,
        "status": job.status.value,
        "cancel_requested": True,
    }


def set_bulk_job(job: BulkJobState) -> None:
    with _BULK_JOBS_LOCK:
        _BULK_JOBS[job.job_id] = job
    _db_create_bulk_job(job.job_id, job.status, job.progress)


def update_job_progress(job_id: Optional[str], message: str) -> None:
    if not job_id:
        return
    with _BULK_JOBS_LOCK:
        job = _BULK_JOBS.get(job_id)
        if not job:
            return
        job.progress = message
        job.updated_at = time.time()
    _db_update_bulk_job(job_id, progress=message)


def start_bulk_job(*, payload: Any) -> dict:
    """一括評価ジョブをバックグラウンドで開始する。"""
    job_id = str(uuid.uuid4())
    job = BulkJobState(
        job_id=job_id,
        status=BulkJobStatus.PENDING,
        progress="ジョブを受け付けました。キューに登録されています。",
    )
    set_bulk_job(job)

    def worker() -> None:
        from .main import jst_now_str  # 遅延インポート

        update_job_progress(job_id, "RAGAS一括評価を開始しました…")
        with _BULK_JOBS_LOCK:
            current = _BULK_JOBS.get(job_id)
            if not current:
                return
            current.status = BulkJobStatus.RUNNING
            current.updated_at = time.time()
        _db_update_bulk_job(
            job_id,
            status=BulkJobStatus.RUNNING,
            progress="RAGAS一括評価を開始しました…",
        )
        try:
            # evaluation_service.bulk_evaluate は async 関数のため、このスレッド内でイベントループを作成して実行
            result = asyncio.run(evaluation_service.bulk_evaluate(payload, job_id=job_id))
            with _BULK_JOBS_LOCK:
                current = _BULK_JOBS.get(job_id)
                if not current:
                    return
                current.status = BulkJobStatus.COMPLETED
                current.result = result
                current.error = None
                current.updated_at = time.time()
            _db_update_bulk_job(
                job_id,
                status=BulkJobStatus.COMPLETED,
                error=None,
                result=result,
            )
            logger.info(
                "[%s][INFO] bulk_job 完了: job_id=%s",
                jst_now_str(),
                job_id,
                extra={"job_id": job_id, "component": "evaluation", "endpoint": "bulk_job"},
            )
        except BulkJobCancelled:
            # ユーザーによるキャンセル
            with _BULK_JOBS_LOCK:
                current = _BULK_JOBS.get(job_id)
                if not current:
                    return
                current.status = BulkJobStatus.CANCELLED
                current.error = "ユーザーによりキャンセルされました。"
                current.updated_at = time.time()
            _db_update_bulk_job(
                job_id,
                status=BulkJobStatus.CANCELLED,
                error="ユーザーによりキャンセルされました。",
            )
            logger.info(
                "[%s][INFO] bulk_job はユーザーによりキャンセルされました: job_id=%s",
                jst_now_str(),
                job_id,
                extra={"job_id": job_id, "component": "evaluation", "endpoint": "bulk_job"},
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            error_detail = traceback.format_exc()
            with _BULK_JOBS_LOCK:
                current = _BULK_JOBS.get(job_id)
                if current:
                    current.status = BulkJobStatus.ERROR
                    current.error = str(e)
                    current.updated_at = time.time()
            _db_update_bulk_job(
                job_id,
                status=BulkJobStatus.ERROR,
                error=str(e),
            )
            logger.error(
                "[ERROR] bulk_job 実行中にエラーが発生しました: %s",
                e,
                extra={"job_id": job_id, "component": "evaluation", "endpoint": "bulk_job"},
            )
            logger.debug(error_detail)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


def get_bulk_job_status(job_id: str) -> dict:
    """メモリまたはDBから一括評価ジョブの状態を取得する。"""
    job = get_bulk_job(job_id)
    if job:
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    # メモリ上にジョブが無い場合は、DBテーブル(evaluation_jobs)から状態を取得
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT job_id, status, progress, cancel_requested, error,
                               result_json, created_at, updated_at
                        FROM evaluation_jobs
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
            "result": result_obj,
            "error": row["error"],
            "created_at": created_ts,
            "updated_at": updated_ts,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        from .main import jst_now_str  # 遅延インポート

        logger.error("[%s][ERROR] bulk_job_status DB参照中にエラー: %s", jst_now_str(), e)
        raise HTTPException(
            status_code=500, detail="ジョブ状態取得中にサーバエラーが発生しました。",
        ) from e
