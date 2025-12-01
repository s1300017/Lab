from __future__ import annotations

from datetime import datetime
from typing import Any

import os
import logging

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pytz import timezone
from sqlalchemy import create_engine, text

from .ollama_config import (
    get_inference_mode,
    set_inference_mode,
    get_ollama_base_url,
)
from .settings import DB_URL, DEFAULT_LLM_NAME, DEFAULT_EMBEDDING_MODEL


def jst_now_str() -> str:
    """JST 現在時刻を文字列で返すユーティリティ。"""
    return datetime.now(timezone("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S JST")


logger = logging.getLogger(__name__)


# データベース接続設定（main.py と同等の環境変数を利用）
engine = create_engine(DB_URL)

# モデル選択のデフォルト値（main.py と同一設定）


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


class ModelSelection(BaseModel):
    """LLM / Embedding モデル選択状態を表すモデル。"""

    llm_model: str
    embedding_model: str


def _db_get_model_selection() -> dict[str, str]:
    """DBからモデル選択状態を取得する。存在しなければデフォルト値を返す。"""

    default_llm = DEFAULT_LLM_NAME
    default_emb = DEFAULT_EMBEDDING_MODEL
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT llm_model, embedding_model FROM model_selection WHERE id = 1")
            ).fetchone()
        if row:
            llm = (row[0] or default_llm).strip()
            emb = (row[1] or default_emb).strip()
            return {"llm_model": llm, "embedding_model": emb}
    except Exception as e:  # noqa: BLE001
        logger.error("[%s][ERROR] model_selection取得失敗: %s", jst_now_str(), e)
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
    except Exception as e:  # noqa: BLE001
        logger.error("[%s][ERROR] model_selection更新失敗: %s", jst_now_str(), e)


router = APIRouter()


@router.get("/config/inference_mode", response_model=InferenceModeResponse)
def get_inference_mode_api() -> InferenceModeResponse:
    """現在の推論モードを取得するAPI。"""

    mode = get_inference_mode()
    return InferenceModeResponse(mode=mode)


@router.post("/config/inference_mode", response_model=InferenceModeResponse)
def update_inference_mode_api(req: InferenceModeUpdateRequest) -> InferenceModeResponse:
    """推論モードを更新するAPI。"""

    try:
        set_inference_mode(req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    mode = get_inference_mode()
    return InferenceModeResponse(mode=mode)


@router.get("/config/model_selection", response_model=ModelSelection)
def get_model_selection_api() -> ModelSelection:
    """現在のLLM/Embeddingモデル選択状態を取得するAPI。"""

    data = _db_get_model_selection()
    return ModelSelection(**data)


@router.post("/config/model_selection", response_model=ModelSelection)
def update_model_selection_api(req: ModelSelection) -> ModelSelection:
    """LLM/Embeddingモデル選択状態を更新するAPI。"""

    llm = (req.llm_model or DEFAULT_LLM_NAME).strip()
    emb = (req.embedding_model or DEFAULT_EMBEDDING_MODEL).strip()
    sel = ModelSelection(llm_model=llm, embedding_model=emb)
    _db_update_model_selection(sel)
    return sel


@router.get("/config/inference_health", response_model=InferenceHealthResponse)
def get_inference_health_api() -> InferenceHealthResponse:
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
    return InferenceHealthResponse(
        mode=mode,
        base_url=base_url,
        ok=ok,
        status_code=status_code,
        error=error,
    )
