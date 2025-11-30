from datetime import datetime, date
from pytz import timezone
import os
import json
import time
import html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import re
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv
import base64
import io
import hashlib
import zipfile
import textwrap
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval  # localStorage操作用
from http_client import http_get, http_post, http_delete, format_http_error
from evaluation_history_ui import show_evaluation_history
from graph_utils import (
    japanese_font,
    plot_overlap_comparison,
    save_plotly_figure,
    create_zip_with_graphs,
)
from sidebar_pdf_upload import render_pdf_upload_sidebar
from tab_bulk_evaluation import render_bulk_evaluation_tab
from tab_chunking import render_chunking_tab
from tab_chatbot import render_chatbot_tab
from tab_thesis import render_thesis_tab
from tab_overview import render_overview_tab
from tab_history import render_history_tab

# http_get/http_post は共通モジュール http_client から使用する

# --- experiment_idによるQA・スコア復元関数 ---
def restore_qa_from_backend():
    """
    experiment_idがsession_stateまたはlocalStorageにあれば、APIからQA・スコアを復元してsession_stateに格納
    """
    # 共通セッション使用のため、ローカルrequestsインポートは不要
    import os
    experiment_id = st.session_state.get("experiment_id")
    if not experiment_id:
        # localStorageから取得（非同期→即時反映されないので注意）
        experiment_id = streamlit_js_eval(
            js_expressions="localStorage.getItem('rag_experiment_id')",
            key="get_exp_id"
        )
        if experiment_id:
            st.session_state["experiment_id"] = experiment_id
            st.rerun()  # experiment_idセット後は即再読み込みしてAPI復元を確実に実行
            return  # 2回目のロードで以降の処理が実行される
    if experiment_id:
        BACKEND_URL = os.environ.get('BACKEND_URL', st.secrets.get('BACKEND_URL', 'http://backend:8000'))
        try:
            response = http_get(f"{BACKEND_URL}/api/v1/experiments/{experiment_id}/detailed_results/")
            if response.status_code == 200:
                data = response.json()
                st.session_state["qa_questions"] = data.get("questions", [])
                st.session_state["qa_answers"] = data.get("answers", [])
                # --- スコア情報（qa_metaまたはscores）も復元 ---
                scores = data.get("qa_meta") or data.get("scores") or []
                st.session_state["qa_meta"] = scores  # どちらでもOKなように対応
        except Exception as e:
            st.warning(f"experiment_idからQA復元失敗: {e}")

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import matplotlib.font_manager as fm
import tempfile
import shutil
import matplotlib.pyplot as plt
from openai import OpenAI

def jst_now_str():
    return datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')

# 環境変数を読み込み
load_dotenv()

st.set_page_config(layout="wide")
st.title("RAG評価システム")

# --- Session State Initialization ---
def init_session_state():
    if 'text' not in st.session_state:
        st.session_state.text = ""
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'bulk_evaluation_results' not in st.session_state:
        st.session_state.bulk_evaluation_results = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = "gpt-oss"  # Default to GPT-OSS
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = "huggingface_bge_small" # Default to HuggingFace
    # --- chat_modelもllm_modelと同期して初期化 ---
    if 'chat_model' not in st.session_state:
        st.session_state.chat_model = st.session_state.llm_model
    # --- experiment_idがあればQA・スコアをAPIから復元 ---
    restore_qa_from_backend()  # 永続化されたQAデータを復元


init_session_state()

# --- Backend API Calls ---
# --- バックエンドAPIのURL設定（ローカル開発時はlocalhostを推奨） ---
# secrets.tomlが存在しなくてもエラーにならないよう例外処理を追加
# Docker環境ではbackendサービス名でAPIに接続するのが推奨
import os
try:
    BACKEND_URL = st.secrets.get('BACKEND_URL', os.environ.get('BACKEND_URL', 'http://backend:8000'))  # Docker時はbackend:8000、ローカル時は環境変数orlocalhost
except Exception as e:
    print(f"[WARNING] st.secrets読み込み失敗: {e}")
    BACKEND_URL = os.environ.get('BACKEND_URL', 'http://backend:8000')


def sync_model_selection_from_backend(ttl_sec: float = 60.0) -> None:
    """バックエンドの /config/model_selection からLLM/Embedding選択を復元し、
    session_state に反映する。ttl_sec 以内に同期済みなら再取得しない。"""
    now = time.time()
    cached_ts = st.session_state.get("_model_selection_synced_ts")
    if isinstance(cached_ts, (int, float)) and now - cached_ts < ttl_sec:
        return

    try:
        resp = http_get(f"{BACKEND_URL}/config/model_selection")
    except Exception as e:  # noqa: BLE001
        # バックエンド未起動などの場合はデフォルトのまま進める
        print(f"[WARN] model_selection取得エラー: {e}")
        return

    if resp.status_code != 200 or not resp.headers.get("Content-Type", "").startswith("application/json"):
        return

    try:
        data = resp.json() or {}
    except Exception:
        return

    llm = (data.get("llm_model") or st.session_state.get("llm_model") or "gpt-oss").strip()
    emb = (data.get("embedding_model") or st.session_state.get("embedding_model") or "huggingface_bge_small").strip()
    # DBに保存されているモデル選択をグローバル状態に反映
    st.session_state.llm_model = llm
    st.session_state.embedding_model = emb
    # チャットタブで使用する chat_model も常に最新の llm_model に合わせる
    # （以前は初期化時のみ同期していたため、リロード直後に旧デフォルトで上書きされてしまっていた）
    st.session_state.chat_model = llm
    st.session_state["_model_selection_synced_ts"] = now


# 起動時に一度だけモデル選択状態を同期
sync_model_selection_from_backend()


def fetch_inference_mode(ttl_sec: float = 30.0) -> str:
    """バックエンドから現在の推論モードを取得する。失敗時はmac_localを返す。

    ttl_sec 秒以内に取得済みの値があれば、そのキャッシュを優先する。
    """
    now = time.time()
    cached_mode = st.session_state.get("_inference_mode_cache")
    cached_ts = st.session_state.get("_inference_mode_cache_ts")
    if cached_mode is not None and isinstance(cached_ts, (int, float)) and now - cached_ts < ttl_sec:
        return cached_mode

    new_mode: str | None = None
    try:
        resp = http_get(f"{BACKEND_URL}/config/inference_mode")
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("application/json"):
            data = resp.json() or {}
            mode_candidate = data.get("mode") or "mac_local"
            if mode_candidate in ("mac_local", "windows_gpu"):
                new_mode = mode_candidate
    except Exception as e:  # noqa: BLE001
        st.warning(f"推論モード取得中にエラーが発生しました: {e}")

    if new_mode is not None:
        st.session_state["_inference_mode_cache"] = new_mode
        st.session_state["_inference_mode_cache_ts"] = now
        return new_mode

    # HTTP取得に失敗した場合は既存キャッシュを優先し、それもなければmac_localを返す
    if cached_mode is not None:
        return cached_mode

    st.session_state["_inference_mode_cache"] = "mac_local"
    st.session_state["_inference_mode_cache_ts"] = now
    return "mac_local"


def fetch_inference_health(ttl_sec: float = 30.0) -> dict | None:
    """現在の推論モードに基づき、バックエンド経由で Ollama API への疎通状況を取得する。

    ttl_sec 秒以内に取得済みの値があれば、そのキャッシュを優先する。
    """
    now = time.time()
    cached_health = st.session_state.get("_inference_health_cache")
    cached_ts = st.session_state.get("_inference_health_cache_ts")
    if cached_health is not None and isinstance(cached_ts, (int, float)) and now - cached_ts < ttl_sec:
        return cached_health

    try:
        resp = http_get(f"{BACKEND_URL}/config/inference_health")
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("application/json"):
            health = resp.json() or {}
            st.session_state["_inference_health_cache"] = health
            st.session_state["_inference_health_cache_ts"] = now
            return health
        st.error(f"推論先疎通テストに失敗しました: {resp.status_code} {resp.text}")
    except Exception as e:  # noqa: BLE001
        st.error(f"推論先疎通テスト中にエラーが発生しました: {e}")

    # HTTP取得に失敗した場合は既存キャッシュを優先し、それもなければNone
    if cached_health is not None:
        return cached_health
    return None


# --- 推論先（Mac / Windows）の切り替えUI ---
mode_labels = {
    "mac_local": "Macローカル（このMacで実行）",
    "windows_gpu": "Windows GPU（RTX 3080 Tiで実行）",
}
current_mode = fetch_inference_mode()
current_label = mode_labels.get(current_mode, mode_labels["mac_local"])
label_list = list(mode_labels.values())
try:
    current_index = label_list.index(current_label)
except ValueError:
    current_index = 0
selected_label = st.radio(
    "推論先（LLM / OCR の実行環境）",
    options=label_list,
    index=current_index,
    horizontal=True,
)
label_to_mode = {v: k for k, v in mode_labels.items()}
selected_mode = label_to_mode.get(selected_label, "mac_local")
if selected_mode != current_mode:
    try:
        resp = http_post(f"{BACKEND_URL}/config/inference_mode", json={"mode": selected_mode})
        if resp.status_code != 200:
            st.error(f"推論モード更新に失敗しました: {resp.status_code} {resp.text}")
        else:
            # モード更新に成功した場合はキャッシュを即時反映し、ヘルスチェックキャッシュを無効化
            st.session_state["_inference_mode_cache"] = selected_mode
            st.session_state["_inference_mode_cache_ts"] = time.time()
            st.session_state.pop("_inference_health_cache", None)
            st.session_state.pop("_inference_health_cache_ts", None)
            try:
                st.toast(f"推論先を「{selected_label}」に切り替えました。")
            except Exception:
                st.info(f"推論先を「{selected_label}」に切り替えました。")
    except Exception as e:  # noqa: BLE001
        st.error(f"推論モード更新中にエラーが発生しました: {e}")

# 常に現在の疎通状況を確認して表示
health = fetch_inference_health()
if health is not None:
    mode = health.get("mode", "mac_local")
    base_url = health.get("base_url", "-")
    ok = bool(health.get("ok", False))
    status_code = health.get("status_code")
    error = health.get("error")
    mode_label = mode_labels.get(mode, mode)
    if ok:
        msg = f"現在の推論先: {mode_label} @ {base_url}（Ollama API 応答OK: status={status_code}）"
        st.success(msg)
    else:
        msg = f"現在の推論先: {mode_label} @ {base_url} への疎通に失敗しました。status={status_code if status_code is not None else '-'}"
        if error:
            msg += f" / error={error}"
        st.error(msg)


def clear_database():
    try:
        response = http_post(f"{BACKEND_URL}/clear_db/")
        if response.status_code == 200:
            result = response.json()
            st.success("🗑️ 全DBデータを正常に削除しました！")
            
            # 削除結果の詳細を表示
            if "details" in result and result["details"]:
                st.write("**削除結果詳細:**")
                for detail in result["details"]:
                    if "エラー" in detail or "失敗" in detail:
                        st.warning(f"⚠️ {detail}")
                    else:
                        st.info(f"✅ {detail}")
            
            # セッション状態をクリア
            st.session_state.text = ""
            st.session_state.chunks = []
            st.session_state.evaluation_results = None
            st.session_state.bulk_evaluation_results = None
            st.session_state.chat_history = []
            
            # リロードを促す
            st.info("🔄 ページをリロードして変更を反映してください")
        else:
            st.error(f"データベースのクリアに失敗しました: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"バックエンドに接続できませんでした: {e}")
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")

# --- localStorageユーティリティ ---
def save_state_to_localstorage():
    # 状態永続化はDB側の履歴テーブルに委ねる方針とし、localStorageへの保存は行わない
    return

# --- session_stateの初期化 ---
def init_session_state():
    default_state = {
        "file_id": None,
        "text": "",
        "qa_questions": [],
        "qa_answers": [],
        "uploaded_file_name": "",
        "uploaded_file_bytes": None,
        "uploaded_file_size": None,
        "uploaded_at": None,
        "evaluation_results": {},  # 評価結果
        "bulk_evaluation_results": {},  # バルク評価結果
        "chunks": [],  # チャンクデータ
        "chat_history": [],  # チャット履歴
        "current_evaluation": None,  # 現在の評価セッション
        "evaluation_history": [],  # 評価履歴
        "active_tab": "tab1",  # 現在のアクティブタブ
        "tab1_content": {"chat_history": []},  # タブ1の表示内容
        "tab2_content": {},  # タブ2の表示内容
        "tab3_content": {},  # タブ3の表示内容
        "tab4_content": {},  # タブ4の表示内容
        "_localstorage_loaded": False,
        "upload_processed_once": False,
        "upload_error_message": None,
        "upload_warning_message": "",
        "upload_cleanse_flag": False,
        "upload_processing": False,
        "upload_cancel_requested": False,
        "upload_cancel_message": "",
        "upload_job_id": None,
        "upload_job_status": None,
        "upload_job_progress": "",
        "upload_job_started_at": None,
        "current_upload_file_id": None,
    }
    for k, v in default_state.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_document_session_state():
    """現在読み込み中のPDFに紐づくセッション状態を初期化し、連続アップロードを可能にする。"""
    doc_state_defaults = {
        "file_id": None,
        "text": "",
        "qa_questions": [],
        "qa_answers": [],
        "qa_meta": [],
        "uploaded_file_name": "",
        "uploaded_file_bytes": None,
        "uploaded_file_size": None,
        "uploaded_at": None,
        "cleanse_used": False,
        "experiment_id": None,
        "chunks": [],
        "chat_history": [],
        "evaluation_results": {},
        "bulk_evaluation_results": {},
        "current_evaluation": None,
        "active_tab": "tab1",
        "tab1_content": {"chat_history": []},
        "tab2_content": {},
        "tab3_content": {},
        "tab4_content": {},
        "upload_processed_once": False,
        "upload_error_message": None,
        "upload_warning_message": "",
        "upload_cleanse_flag": False,
        "upload_processing": False,
        "upload_cancel_requested": False,
        "upload_cancel_message": "",
        "upload_job_id": None,
        "upload_job_status": None,
        "upload_job_progress": "",
        "upload_job_started_at": None,
        "current_upload_file_id": None,
    }
    for key, value in doc_state_defaults.items():
        st.session_state[key] = value


def format_file_size(num_bytes: Optional[int]) -> str:
    """ファイルサイズを人間が読みやすい形式に変換するユーティリティ。"""
    if not isinstance(num_bytes, (int, float)) or num_bytes < 0:
        return "-"
    step_unit = 1024.0
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < step_unit:
            return f"{size:.1f} {unit}"
        size /= step_unit
    return f"{size:.1f} PB"

# --- UI Layout ---
render_pdf_upload_sidebar(
    BACKEND_URL=BACKEND_URL,
    init_session_state=init_session_state,
    reset_document_session_state=reset_document_session_state,
    save_state_to_localstorage=save_state_to_localstorage,
    jst_now_str=jst_now_str,
)

# メインコンテンツのタブ定義
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["チャンキング設定", "一括評価", "チャットボット", "卒論向け分析", "履歴", "システム説明"])
tab_chatbot = tab3  # チャットボットタブ
tab_thesis = tab4   # 卒論向け分析タブ
tab_history = tab5  # 履歴タブ
tab_overview = tab6  # システム説明タブ

# システム説明タブ
render_overview_tab(tab_overview)

# タブ1: チャンキング設定
render_chunking_tab(tab1, BACKEND_URL, save_state_to_localstorage)
# メインコンテンツ
upload_processing = bool(st.session_state.get("upload_processing", False))
if (not st.session_state.text) and (not upload_processing):
    # 歴史データの有無を確認し、全画面アクセス可否を案内（停止はしない）
    try:
        resp_pdf = http_get(f"{BACKEND_URL}/history/pdf-files")
        if resp_pdf.status_code == 200:
            data_pdf = resp_pdf.json() if resp_pdf.headers.get("Content-Type", "").startswith("application/json") else {}
            has_pdf_history = isinstance(data_pdf, dict) and len(data_pdf.get("items", [])) > 0
        else:
            has_pdf_history = False
    except Exception:
        has_pdf_history = False
    try:
        resp_exp = http_get(f"{BACKEND_URL}/history/experiments")
        if resp_exp.status_code == 200:
            data_exp = resp_exp.json() if resp_exp.headers.get("Content-Type", "").startswith("application/json") else {}
            has_experiment_history = isinstance(data_exp, dict) and len(data_exp.get("items", [])) > 0
        else:
            has_experiment_history = False
    except Exception:
        has_experiment_history = False
    st.session_state["has_pdf_history"] = has_pdf_history
    st.session_state["has_experiment_history"] = has_experiment_history

    if has_pdf_history:
        st.info("PDFのアップロード履歴が見つかりました。履歴から復元すると、全ての画面にアクセスできます。")
    elif has_experiment_history:
        st.info("評価履歴のみが存在します。履歴タブから閲覧できます。その他のタブはPDF復元または新規アップロード後に利用できます。")
    else:
        st.info("サイドバーでPDFファイルをアップロードし、設定を行ってください。")

# タブ2: 一括評価
render_bulk_evaluation_tab(tab2, BACKEND_URL)
# チャットボットタブ
render_chatbot_tab(tab_chatbot, BACKEND_URL, save_state_to_localstorage)

# 卒論向け分析タブ
render_thesis_tab(tab_thesis, BACKEND_URL)

# 履歴タブ
with tab_history:
    st.header("履歴")
    st.caption("アップロードしたPDF・生成QA・チャンク・一括評価結果を一覧・参照できます。")

    # --- 管理: DB全消去（バックアップ付き・危険操作） ---
    with st.expander("管理（危険）: DBと保存ファイルの全消去", expanded=False):
        st.warning("この操作は元に戻せません。実行前にバックアップを作成します。")

        # RAGベクトル・チャンクのみリセット（PDFは残す）
        st.markdown("#### RAGベクトル・チャンクのみリセット（PDFは残す）")
        confirm_reset_vectors = st.checkbox(
            "langchain_pg_embedding / langchain_pg_collection / pdf_chunks を削除（pdf_files とPDF本体は維持）",
            value=False,
            key="confirm_reset_vectors_and_chunks",
        )
        if st.button("ベクトル＆チャンクのみリセット", disabled=not confirm_reset_vectors, key="btn_reset_vectors_and_chunks"):
            with st.spinner("ベクトルとチャンクをリセット中..."):
                try:
                    resp = http_post(f"{BACKEND_URL}/admin/reset_vectors_and_chunks/")
                    if resp.status_code == 200:
                        data = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
                        msg = data.get("message", "ベクトルとチャンクをリセットしました。")
                        st.success(msg)
                    else:
                        st.error(f"リセットに失敗しました: {resp.status_code} {resp.text}")
                except Exception as e:
                    st.error(f"リセット中にエラーが発生しました: {e}")

        st.markdown("---")

        # PDFも含めてRAG関連を全リセット
        st.markdown("#### PDFも含めてRAG関連を全リセット")
        confirm_reset_all = st.text_input(
            "PDFファイルと抽出JSONも削除します。確認のため 'RESET ALL' と入力してください",
            value="",
            key="confirm_reset_pdfs_and_vectors",
        )
        disabled_reset_all = (confirm_reset_all.strip().upper() != "RESET ALL")
        if st.button("PDFとRAGベクトルを全リセット", type="secondary", disabled=disabled_reset_all, key="btn_reset_pdfs_and_vectors"):
            with st.spinner("PDFおよびRAGベクトルをリセット中..."):
                try:
                    resp = http_post(f"{BACKEND_URL}/admin/reset_pdfs_and_vectors/")
                    if resp.status_code == 200:
                        data = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
                        msg = data.get("message", "PDFおよびRAGベクトルをリセットしました。")
                        st.success(msg)
                        st.json(data)
                        st.info("必要に応じてページを再読み込みしてください。")
                    else:
                        st.error(f"リセットに失敗しました: {resp.status_code} {resp.text}")
                except Exception as e:
                    st.error(f"リセット中にエラーが発生しました: {e}")

        st.markdown("---")

        # 既存のDB全消去（バックアップ付き）
        col1, col2, col3 = st.columns(3)
        with col1:
            make_backup = st.checkbox("バックアップを作成する", value=True)
        with col2:
            delete_files = st.checkbox("PDF/抽出ファイルも削除", value=True)
        with col3:
            st.write("")
            st.write("")
            st.write("")
        confirm_text = st.text_input("確認のため 'CONFIRM' と入力してください", value="")
        disabled = (confirm_text.strip().upper() != "CONFIRM")
        if st.button("DBを全消去（バックアップ付き）", type="primary", disabled=disabled):
            with st.spinner("全消去を実行中..."):
                payload = {
                    "confirm": True,
                    "backup": bool(make_backup),
                    "delete_files": bool(delete_files),
                }
                try:
                    resp = http_post(f"{BACKEND_URL}/admin/wipe_all", json=payload)
                    if resp.status_code == 200:
                        res = resp.json()
                        st.success("全消去が完了しました。アプリを再読み込みします。")
                        st.json(res)
                        st.rerun()
                    else:
                        st.error(f"全消去に失敗しました: {resp.status_code} {resp.text}")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

    # ユーティリティ: GETリクエスト（簡易ラッパー）
    def history_api_get(url: str, timeout: float | None = None):
        import requests
        try:
            resp = requests.get(url, timeout=timeout)
            return resp.status_code, (resp.json() if resp.headers.get('content-type','').startswith('application/json') else resp.text)
        except Exception as e:
            return 599, {"error": str(e)}

    # PDF一覧の取得
    with st.expander("PDF一覧", expanded=True):
        with st.spinner("PDF一覧を取得中..."):
            code, data = history_api_get(f"{BACKEND_URL}/history/pdf-files")
        if code == 200 and isinstance(data, dict):
            pdf_items = data.get("items", [])
            if not pdf_items:
                st.info("登録済みのPDFがありません。先にPDFをアップロードしてください。")
            else:
                # セレクトボックス用にラベル整形（ファイル名 + 追加日時）
                labels = []
                for it in pdf_items:
                    base_name = it.get('original_name') or it.get('file_name') or str(it.get('id'))
                    uploaded_at = it.get('uploaded_at') or ""
                    label = base_name
                    if uploaded_at:
                        label = f"{base_name}（追加日時: {uploaded_at}）"
                    labels.append(label)
                idx = st.selectbox("PDFを選択", options=list(range(len(pdf_items))), format_func=lambda i: labels[i])
                selected_pdf = pdf_items[idx]
                st.markdown("---")
                st.subheader("選択中のPDF")
                st.json(selected_pdf)

                # PDF削除操作を提供
                st.markdown("#### PDF削除")
                delete_confirm = st.text_input(
                    "削除確認のため 'DELETE' と入力してください",
                    value="",
                    key=f"delete_confirm_{selected_pdf['id']}"
                ).strip().upper()
                delete_disabled = delete_confirm != "DELETE"
                if st.button(
                    "このPDFを削除",
                    key=f"delete_pdf_{selected_pdf['id']}",
                    disabled=delete_disabled,
                ):
                    with st.spinner("PDFを削除中です..."):
                        try:
                            delete_resp = http_delete(f"{BACKEND_URL}/pdf/{selected_pdf['id']}")
                            if delete_resp.status_code == 200:
                                st.success("PDFを削除しました。リストを更新します。")
                                st.rerun()
                            else:
                                st.error(f"削除に失敗しました: {delete_resp.status_code} {delete_resp.text}")
                        except Exception as delete_error:
                            st.error(f"削除処理中にエラーが発生しました: {delete_error}")

                # 履歴から抽出データを復元して全タブを有効化
                if st.button("このPDFを開く（抽出テキスト・QAを復元）", key=f"restore_pdf_{selected_pdf['id']}"):
                    with st.spinner("PDF抽出データを復元中..."):
                        code_x, data_x = history_api_get(f"{BACKEND_URL}/get_extracted/{selected_pdf['id']}")
                        if code_x == 200 and isinstance(data_x, dict):
                            st.session_state["file_id"] = selected_pdf['id']
                            st.session_state.text = data_x.get("text", "")
                            st.session_state.qa_questions = data_x.get("questions", [])
                            st.session_state.qa_answers = data_x.get("answers", [])
                            st.session_state.qa_meta = data_x.get("qa_meta", [])
                            st.session_state.image_captions = data_x.get("image_captions", [])
                            save_state_to_localstorage()
                            st.success("PDFを復元しました。各タブから操作できます。")
                            st.rerun()

                col_generate_qa, _ = st.columns([1, 3])
                with col_generate_qa:
                    if st.button("このPDFから質問を生成する", key=f"generate_qa_for_history_{selected_pdf['id']}"):
                        with st.spinner("このPDFから質問を生成中です..."):
                            try:
                                question_model = selected_pdf.get("question_llm_model") or "mistral"
                                answer_model = selected_pdf.get("answer_llm_model") or "mistral"
                                resp_qa = http_post(
                                    f"{BACKEND_URL}/pdf/{selected_pdf['id']}/generate_qa",
                                    data={
                                        "question_llm_model": question_model,
                                        "answer_llm_model": answer_model,
                                    },
                                )
                                if resp_qa.status_code == 200:
                                    qa_data = (
                                        resp_qa.json()
                                        if resp_qa.headers.get("Content-Type", "").startswith("application/json")
                                        else {}
                                    )
                                    st.success("質問・回答の自動生成が完了しました。")
                                    st.rerun()
                                else:
                                    st.error(f"質問生成APIエラー: {format_http_error(resp_qa)}")
                            except Exception as e:
                                st.error(f"質問生成API呼び出し中にエラーが発生しました: {e}")

                # 生成QA一覧
                st.markdown("### 生成QA一覧")
                with st.spinner("生成QAを取得中..."):
                    code_q, data_q = history_api_get(f"{BACKEND_URL}/history/pdf-files/{selected_pdf['id']}/questions")
                if code_q == 200 and isinstance(data_q, dict):
                    qa_items = data_q.get("items", [])
                    if not qa_items:
                        code_x, data_x = history_api_get(f"{BACKEND_URL}/get_extracted/{selected_pdf['id']}")
                        if code_x == 200 and isinstance(data_x, dict):
                            extracted_questions = data_x.get("questions") or []
                            extracted_answers = data_x.get("answers") or []
                            extracted_meta = data_x.get("qa_meta") or []
                            for idx_q, q in enumerate(extracted_questions):
                                a = extracted_answers[idx_q] if idx_q < len(extracted_answers) else ""
                                meta_val = extracted_meta[idx_q] if idx_q < len(extracted_meta) else {}
                                qa_items.append(
                                    {
                                        "question": q,
                                        "answer": a,
                                        "meta_json": meta_val,
                                    }
                                )

                    if qa_items:
                        import pandas as pd

                        # --- メタ情報をパースしてスコア順に整列 ---
                        qa_tuples: list[tuple[str, str, dict]] = []
                        for item in qa_items:
                            question = item.get("question", "")
                            answer = item.get("answer", "")
                            meta_raw = item.get("meta_json")
                            meta: dict = {}
                            if meta_raw:
                                if isinstance(meta_raw, str):
                                    try:
                                        meta = json.loads(meta_raw)
                                    except Exception:
                                        meta = {"raw": meta_raw}
                                elif isinstance(meta_raw, dict):
                                    meta = meta_raw
                            score = meta.get("score") or meta.get("total_score")
                            if score is None and answer:
                                score = float(len(answer)) / 1000.0
                                meta["score"] = score
                            qa_tuples.append((question, answer, meta))

                        qa_tuples_sorted = sorted(qa_tuples, key=lambda x: x[2].get("score", 0), reverse=True)

                        # --- 仕組みとスコア計算補足 ---
                        with st.expander("🤖 自動質問生成の仕組みとスコア計算", expanded=False):
                            question_model = selected_pdf.get("question_llm_model", "不明")
                            answer_model = selected_pdf.get("answer_llm_model", "不明")
                            st.markdown(f"""
                            ### 質問生成プロセス

                            - **主要手法**: LLM（`{question_model}`）がPDF本文の冒頭およそ1,500文字から代表的な質問を5件生成します。
                            - **フォールバック**: LLMが失敗した場合、QA抽出・箇条書き抽出・段落要約の3段構えで質問を補完します。
                            - **回答生成**: ` {answer_model} ` が本文（約3,000文字）をコンテキストとして各質問への回答を生成します。

                            ### 信頼性スコアの考え方

                            - 同一の質問/回答ペアが複数回候補に現れた頻度（出現回数スコア）
                            - 回答文の長さを全候補で正規化した値（回答長スコア）
                            - 上記を合算した **`score` = 出現回数スコア + 回答長スコア** を信頼性の指標としています。

                            ```python
                            qa_df["count_score"] = qa_df.groupby(["question", "answer"])['answer'].transform('count')
                            qa_df["len_score"] = qa_df["answer"].apply(len)
                            qa_df["len_score"] = (qa_df["len_score"] - qa_df["len_score"].min()) / (qa_df["len_score"].max() - qa_df["len_score"].min() + 1e-6)
                            qa_df["score"] = qa_df["count_score"] + qa_df["len_score"]
                            ```

                            - スコアが高いほど「頻出かつ情報量の多い回答」とみなし、上位から順に表示します。
                            - ダミー回答や自動修正の有無はメタ情報に記録され、バッジ表示に反映されます。
                            """)

                        st.markdown("#### 自動生成QAセット（信頼性スコア順）")
                        for idx, (question, answer, meta) in enumerate(qa_tuples_sorted, start=1):
                            with st.expander(f"Q{idx}: {question}"):
                                score = meta.get("score")
                                is_auto_fixed = meta.get("is_auto_fixed")
                                is_dummy_answer = meta.get("is_dummy_answer")

                                if is_dummy_answer:
                                    badge_text = "🟠 ダミー回答"
                                    badge_color = "orange"
                                elif is_auto_fixed:
                                    badge_text = "🔴 自動修正済み"
                                    badge_color = "red"
                                else:
                                    badge_text = "🔵 一意回答"
                                    badge_color = "blue"

                                st.markdown(f"**A:** {answer}")
                                col_q1, col_q2 = st.columns([1, 3])
                                with col_q1:
                                    st.markdown(f":{badge_color}[{badge_text}]")
                                with col_q2:
                                    st.markdown(
                                        f"信頼性スコア: {score:.3f}" if isinstance(score, (int, float)) else "信頼性スコア: -"
                                    )

                                candidates = meta.get("candidates", [])
                                candidate_scores = meta.get("candidate_scores", [])
                                if candidates and len(candidates) > 1:
                                    with st.expander("候補回答リスト（スコア付き"):
                                        for cand, cand_score in zip(candidates, candidate_scores):
                                            if isinstance(cand_score, (int, float)):
                                                st.markdown(f"- {cand}（スコア: {cand_score:.3f}）")
                                            else:
                                                st.markdown(f"- {cand}")
                    else:
                        st.info("生成QAはまだ保存されていません。")
                else:
                    st.error(f"生成QA取得エラー: {code_q} {data_q}")
                st.markdown("### チャンク一覧")
                with st.spinner("チャンクを取得中..."):
                    code_c, data_c = history_api_get(f"{BACKEND_URL}/history/pdf-files/{selected_pdf['id']}/chunks")
                if code_c == 200 and isinstance(data_c, dict):
                    chunk_items = data_c.get("items", [])
                    if chunk_items:
                        import pandas as pd
                        for it in chunk_items:
                            content = it.get("content", "")
                            it["preview"] = content[:100] + ("..." if len(content) > 100 else "")
                        chunk_df = pd.DataFrame(chunk_items)
                        st.dataframe(chunk_df[[c for c in chunk_df.columns if c != 'content']], use_container_width=True)
                        if st.button("このPDFの抽出データを復元"):
                            with st.spinner("PDF抽出データを復元中..."):
                                code_x, data_x = history_api_get(f"{BACKEND_URL}/get_extracted/{selected_pdf['id']}")
                                if code_x == 200 and isinstance(data_x, dict):
                                    st.session_state["file_id"] = selected_pdf['id']
                                    st.session_state.text = data_x.get("text", "")
                                    st.session_state.qa_questions = data_x.get("questions", [])
                                    st.session_state.qa_answers = data_x.get("answers", [])
                                    st.session_state.qa_meta = data_x.get("qa_meta", [])
                                    st.session_state.image_captions = data_x.get("image_captions", [])
                                    save_state_to_localstorage()
                                    st.success("PDFを復元しました。各タブから操作できます。")
                                    st.rerun()
                else:
                    st.error(f"チャンク取得エラー: {code_c} {data_c}")

    st.markdown("---")
    st.subheader("一括評価結果一覧")
    st.caption("実行済みの実測ごとの評価メトリクスと詳細を参照できます。")
    try:
        with st.expander("評価履歴", expanded=True):
            show_evaluation_history(BACKEND_URL.rstrip("/"))
    except Exception as e:
        st.error(f"評価履歴の表示に失敗しました: {e}")