from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from .evaluation_job_service import (
    start_bulk_job,
    get_bulk_job_status,
    request_cancel_bulk_job,
)


router = APIRouter()


@router.post("/bulk_job/start")
async def bulk_job_start(request: Request) -> dict[str, Any]:
    """RAGAS一括評価ジョブをバックグラウンドで開始するAPI。"""

    payload = await request.json()
    return start_bulk_job(payload=payload)


@router.get("/bulk_job/status/{job_id}")
def bulk_job_status(job_id: str) -> dict[str, Any]:
    """一括評価ジョブの状態を取得するAPI。"""

    return get_bulk_job_status(job_id)


@router.post("/bulk_job/cancel/{job_id}")
def bulk_job_cancel(job_id: str) -> dict[str, Any]:
    """一括評価ジョブのキャンセルを要求するAPI。"""

    return request_cancel_bulk_job(job_id)
