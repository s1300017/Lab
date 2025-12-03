from __future__ import annotations

from typing import Any, Dict, List, Tuple

import time
import streamlit as st

from http_client import http_get


# /list_models の結果を簡易キャッシュするためのモジュール内キャッシュ
# 非同期性は考慮せず、単一プロセス内での再利用のみを想定
__MODEL_CACHE: Dict[str, Any] = {
    "ts": 0.0,
    "llm": [],
    "embedding": [],
}


def fetch_model_lists(BACKEND_URL: str, ttl_sec: float = 60.0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """バックエンドの /list_models からLLM/Embeddingモデル一覧を取得する共通ヘルパー。

    - 一度取得した結果は ttl_sec 秒間キャッシュして再利用する。
    - 取得に失敗した場合は警告を出し、直近のキャッシュ（あれば）または空リストを返す。
    """
    global __MODEL_CACHE

    now = time.time()
    cached_ts = float(__MODEL_CACHE.get("ts") or 0.0)
    cached_llm = __MODEL_CACHE.get("llm") or []
    cached_emb = __MODEL_CACHE.get("embedding") or []

    if cached_llm or cached_emb:
        if now - cached_ts < ttl_sec:
            return list(cached_llm), list(cached_emb)

    try:
        resp = http_get(f"{BACKEND_URL}/list_models")
        resp.raise_for_status()
        data = (
            resp.json()
            if resp.headers.get("Content-Type", "").startswith("application/json")
            else {}
        )
        llm_models = data.get("LLM", []) or []
        emb_models = data.get("Embedding", []) or []
        if not isinstance(llm_models, list):
            llm_models = []
        if not isinstance(emb_models, list):
            emb_models = []

        __MODEL_CACHE = {
            "ts": now,
            "llm": llm_models,
            "embedding": emb_models,
        }
        return llm_models, emb_models
    except Exception as e:  # noqa: BLE001
        # 取得に失敗した場合はキャッシュを優先し、それも無ければ空リストを返す
        st.warning(f"モデル一覧の取得に失敗しました: {e}")
        return list(cached_llm), list(cached_emb)


def fetch_llm_models(BACKEND_URL: str, ttl_sec: float = 60.0) -> List[Dict[str, Any]]:
    """LLMモデル一覧のみを返すヘルパー。"""
    llm_models, _ = fetch_model_lists(BACKEND_URL, ttl_sec=ttl_sec)
    return llm_models


def fetch_embedding_models(BACKEND_URL: str, ttl_sec: float = 60.0) -> List[Dict[str, Any]]:
    """Embeddingモデル一覧のみを返すヘルパー。"""
    _, emb_models = fetch_model_lists(BACKEND_URL, ttl_sec=ttl_sec)
    return emb_models


def fetch_history_pdfs(BACKEND_URL: str, ttl_sec: float = 30.0) -> List[Dict[str, Any]]:
    """/history/pdf-files からPDF一覧を取得する共通ヘルパー（簡易キャッシュ付き）。"""

    now = time.time()
    cache_key = "_history_pdfs_cache"
    ts_key = "_history_pdfs_cache_ts"
    cached_items = st.session_state.get(cache_key)
    cached_ts = st.session_state.get(ts_key)

    if cached_items is not None and isinstance(cached_ts, (int, float)) and now - cached_ts < ttl_sec:
        try:
            return list(cached_items)
        except TypeError:
            pass

    # 一括評価ジョブ実行中は新規の /history/pdf-files 呼び出しを抑制し、
    # 既存キャッシュがあればそれのみを返す。
    bulk_job_id = st.session_state.get("bulk_eval_job_id")
    bulk_job_status = (st.session_state.get("bulk_eval_job_status") or "").lower()
    bulk_job_active = bool(bulk_job_id) and bulk_job_status in ("pending", "running")
    if bulk_job_active:
        return list(cached_items) if cached_items is not None else []

    try:
        resp = http_get(f"{BACKEND_URL}/history/pdf-files")
        if resp.status_code != 200:
            return []
        data = (
            resp.json()
            if resp.headers.get("Content-Type", "").startswith("application/json")
            else {}
        )
        items = data.get("items", []) or []
        st.session_state[cache_key] = list(items)
        st.session_state[ts_key] = now
        return items
    except Exception as e:  # noqa: BLE001
        st.warning(f"PDF履歴の取得に失敗しました: {e}")
        return list(cached_items) if cached_items is not None else []
