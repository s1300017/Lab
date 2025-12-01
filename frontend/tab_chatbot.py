from __future__ import annotations

from datetime import datetime, date
from typing import List, Dict, Any, Optional, Callable

import html
import pandas as pd
from pytz import timezone
import streamlit as st

from http_client import http_get, http_post, format_http_error
from model_utils import fetch_model_lists


def _fetch_models(BACKEND_URL: str) -> tuple[list[dict], list[dict]]:
    """バックエンドの /list_models からLLM/Embeddingモデル一覧を取得する。

    共通ヘルパー fetch_model_lists を経由して取得し、結果をタブ内で利用しやすい
    (LLM, Embedding) タプルとして返す。
    """
    llm_models, embedding_models = fetch_model_lists(BACKEND_URL.rstrip("/"))
    return llm_models, embedding_models


def _persist_model_selection(BACKEND_URL: str, llm_model: str, embedding_model: str) -> None:
    """選択されたLLM/Embeddingモデルをバックエンドに保存する。"""
    payload = {
        "llm_model": llm_model,
        "embedding_model": embedding_model,
    }
    try:
        http_post(f"{BACKEND_URL}/config/model_selection", json=payload)
    except Exception as e:  # noqa: BLE001
        # 永続化に失敗しても致命的ではないため警告のみにとどめる
        st.warning(f"モデル選択の保存に失敗しました: {e}")


def render_chatbot_tab(
    tab, BACKEND_URL: str, save_state_to_localstorage: Callable[[], None]
) -> None:
    """チャットボットタブのUIとロジックを描画する。"""
    with tab:
        st.header("チャットボット")

        # --- チャット用LLM / Embeddingモデルの選択 ---
        llm_models, embedding_models = _fetch_models(BACKEND_URL.rstrip("/"))

        # LLMモデル選択
        selected_llm_name = (
            st.session_state.get("chat_model")
            or st.session_state.get("llm_model")
            or "gpt-oss"
        )
        if llm_models:
            llm_names = [m.get("name", "") for m in llm_models]
            llm_labels: list[str] = []
            for m in llm_models:
                provider = m.get("type") or "unknown"
                display = m.get("display_name") or m.get("name") or "unknown"
                llm_labels.append(f"[{provider}] {display}")
            if selected_llm_name in llm_names:
                default_llm_idx = llm_names.index(selected_llm_name)
            else:
                default_llm_idx = 0
                selected_llm_name = llm_names[0]
            idx_llm = st.selectbox(
                "チャット用LLMモデル",
                options=list(range(len(llm_names))),
                format_func=lambda i: llm_labels[i],
                index=default_llm_idx,
                key="chat_llm_model_select",
            )
            selected_llm_name = llm_names[idx_llm]
        st.session_state.chat_model = selected_llm_name
        st.session_state.llm_model = selected_llm_name

        # Embeddingモデル選択
        selected_emb_name = st.session_state.get("embedding_model", "huggingface_bge_small")
        if embedding_models:
            emb_names = [m.get("name", "") for m in embedding_models]
            emb_labels: list[str] = []
            for m in embedding_models:
                provider = m.get("type") or "unknown"
                display = m.get("display_name") or m.get("name") or "unknown"
                emb_labels.append(f"[{provider}] {display}")
            if selected_emb_name in emb_names:
                default_emb_idx = emb_names.index(selected_emb_name)
            else:
                default_emb_idx = 0
                selected_emb_name = emb_names[0]
            idx_emb = st.selectbox(
                "チャット用Embeddingモデル",
                options=list(range(len(emb_names))),
                format_func=lambda i: emb_labels[i],
                index=default_emb_idx,
                key="chat_embedding_model_select",
            )
            selected_emb_name = emb_names[idx_emb]
        st.session_state.embedding_model = selected_emb_name

        # 選択内容をバックエンドに永続化
        _persist_model_selection(BACKEND_URL.rstrip("/"), selected_llm_name, selected_emb_name)

        # チャット送信後に入力欄が画面下に来るように自動スクロール
        if st.session_state.get("chat_scroll_to_bottom"):
            st.markdown(
                """
                <script>
                const main = window.parent.document.querySelector('section.main');
                if (main) { main.scrollTop = main.scrollHeight; }
                </script>
                """,
                unsafe_allow_html=True,
            )
            st.session_state["chat_scroll_to_bottom"] = False

        # RAG対象スコープとPDF選択（チャット用）
        scope = st.session_state.get("rag_scope", "single")
        scope_label = st.radio(
            "検索対象",
            ["単一PDFのみ", "すべてのPDF"],
            index=0 if scope == "single" else 1,
            key="chat_scope_radio",
            horizontal=True,
        )
        scope = "single" if scope_label == "単一PDFのみ" else "all"
        st.session_state["rag_scope"] = scope

        # 現在のRAG対象スコープを簡易表示
        if scope == "all":
            st.caption("現在のRAG対象: すべてのPDF")
        else:
            st.caption("現在のRAG対象: 単一PDF（下の『対象PDF』から選択）")

        selected_pdf_id = st.session_state.get("rag_pdf_file_id") or st.session_state.get(
            "file_id"
        )
        if scope == "single":
            history_pdf_items: list[dict] = []
            history_pdf_mapping: dict[str, dict] = {}
            history_pdf_labels: list[str] = []
            default_idx = 0
            try:
                resp_hist = http_get(f"{BACKEND_URL}/history/pdf-files")
                if resp_hist.status_code == 200:
                    data_hist = (
                        resp_hist.json()
                        if resp_hist.headers.get("Content-Type", "").startswith(
                            "application/json"
                        )
                        else {}
                    )
                    if isinstance(data_hist, dict):
                        history_pdf_items = data_hist.get("items", []) or []
                        for idx, item in enumerate(history_pdf_items):
                            label = f"{item.get('original_name', item.get('file_name', '不明なPDF'))} ({item.get('uploaded_at', '日時不明')})"
                            history_pdf_labels.append(label)
                            history_pdf_mapping[label] = item
                            if item.get("id") == selected_pdf_id:
                                default_idx = idx
            except Exception as e:  # noqa: BLE001
                st.warning(f"履歴PDFの取得に失敗しました: {e}")

            if history_pdf_labels:
                sel_label = st.selectbox(
                    "対象PDF",
                    options=history_pdf_labels,
                    index=default_idx,
                    help="RAGの対象とするPDFを選択してください。",
                    key="chat_pdf_select",
                )
                selected_item = history_pdf_mapping.get(sel_label)
                selected_pdf_id = selected_item.get("id") if selected_item else None
                st.session_state["rag_pdf_file_id"] = selected_pdf_id
                if selected_pdf_id is not None:
                    st.caption(f"現在のRAG対象PDF ID: {selected_pdf_id}")

                # 履歴から抽出テキスト・QAを復元（他タブも有効化）
                if st.button(
                    "このPDFを開く（抽出テキスト・QAを復元）",
                    key="chat_restore_pdf",
                ):
                    with st.spinner("PDF抽出データを復元中..."):
                        resp_x = http_get(
                            f"{BACKEND_URL}/get_extracted/{selected_pdf_id}"
                        )
                        if resp_x.status_code == 200:
                            data_x = (
                                resp_x.json()
                                if resp_x.headers.get("Content-Type", "").startswith(
                                    "application/json"
                                )
                                else {}
                            )
                            if isinstance(data_x, dict):
                                st.session_state["file_id"] = selected_pdf_id
                                st.session_state.text = data_x.get("text", "")
                                st.session_state.qa_questions = data_x.get(
                                    "questions", []
                                )
                                st.session_state.qa_answers = data_x.get(
                                    "answers", []
                                )
                                st.session_state.qa_meta = data_x.get("qa_meta", [])
                                st.session_state.image_captions = data_x.get(
                                    "image_captions", []
                                )
                                save_state_to_localstorage()
                                st.success(
                                    "PDFを復元しました。各タブから操作できます。"
                                )
                                st.rerun()
                            else:
                                st.error("抽出データの形式が不正です。")
                        else:
                            st.error(
                                f"抽出データ取得に失敗しました: {resp_x.status_code} {resp_x.text}"
                            )
            else:
                st.info(
                    "履歴に保存されたPDFがありません。PDFをアップロードしてから実行してください。"
                )

        # チャット履歴の初期化 & DBからの復元（ページリロード対応）
        if "chat_messages" not in st.session_state or not st.session_state.chat_messages:
            st.session_state.chat_messages = []
            if scope == "single" and selected_pdf_id:
                try:
                    resp_logs = http_get(
                        f"{BACKEND_URL}/history/chat-logs?pdf_file_id={selected_pdf_id}"
                    )
                    if resp_logs.status_code == 200:
                        data_logs = (
                            resp_logs.json()
                            if resp_logs.headers.get("Content-Type", "").startswith(
                                "application/json"
                            )
                            else {}
                        )
                        if isinstance(data_logs, dict):
                            items = data_logs.get("items", []) or []
                            for row in items:
                                ts = row.get("created_at")
                                user_msg = row.get("user_message", "")
                                assistant_msg = row.get("assistant_message", "")
                                model_used = row.get("llm_model_used")
                                ctx_list = row.get("contexts") or []
                                req_id = row.get("request_id")
                                st.session_state.chat_messages.append(
                                    {
                                        "role": "user",
                                        "content": user_msg,
                                        "model": None,
                                        "timestamp": ts,
                                    }
                                )
                                st.session_state.chat_messages.append(
                                    {
                                        "role": "assistant",
                                        "content": assistant_msg,
                                        "model": model_used,
                                        "timestamp": ts,
                                        "contexts": ctx_list,
                                        "request_id": req_id,
                                    }
                                )
                except Exception as e:  # noqa: BLE001
                    st.warning(f"チャット履歴の取得に失敗しました: {e}")

        reset_filter = st.session_state.pop("chat_reset_history_filter", False)
        if reset_filter and "chat_history_date_selectbox" in st.session_state:
            del st.session_state["chat_history_date_selectbox"]

        # チャットメッセージの整形と日付フィルタ
        tz_jst = timezone("Asia/Tokyo")

        def _ensure_message_timestamp(msg: Dict[str, Any]) -> str:
            if not msg.get("timestamp"):
                msg["timestamp"] = datetime.now(tz_jst).isoformat()
            return msg["timestamp"]

        def _parse_message_timestamp(ts: Optional[str]) -> Optional[datetime]:
            if not ts:
                return None
            try:
                parsed = datetime.fromisoformat(ts)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return tz_jst.localize(parsed)
            return parsed.astimezone(tz_jst)

        parsed_messages: List[Dict[str, Any]] = []
        for stored in st.session_state.chat_messages:
            ts_str = _ensure_message_timestamp(stored)
            ts_dt = _parse_message_timestamp(ts_str)
            parsed_messages.append(
                {
                    "message": stored,
                    "timestamp": ts_dt,
                    "date": ts_dt.date() if ts_dt else None,
                }
            )

        messages_to_display = parsed_messages
        selected_chat_date: Optional[date] = None
        filter_applied = False
        keyword: str = ""
        selected_model: Optional[str] = None

        if parsed_messages:
            date_candidates = sorted(
                {
                    entry["date"]
                    for entry in parsed_messages
                    if entry["date"] is not None
                },
                reverse=True,
            )
            if date_candidates:
                selected_chat_date = st.selectbox(
                    "表示する日付",
                    options=[None] + date_candidates,
                    format_func=lambda v: "すべて" if v is None else v.isoformat(),
                    key="chat_history_date_selectbox",
                )
                if selected_chat_date is not None:
                    filter_applied = True
                    messages_to_display = [
                        entry
                        for entry in messages_to_display
                        if entry["date"] == selected_chat_date
                    ]

            # キーワードフィルタ（ユーザー発話・アシスタント応答の両方を対象）
            keyword = st.text_input(
                "キーワードで絞り込み",
                value="",
                key="chat_history_keyword_filter",
                placeholder="メッセージ内容に含まれる文字列で検索します",
            ).strip()
            if keyword:
                kw_lower = keyword.lower()
                filter_applied = True
                messages_to_display = [
                    entry
                    for entry in messages_to_display
                    if kw_lower in str(entry["message"].get("content", "")).lower()
                ]

            # モデル名でのフィルタ（アシスタントメッセージの model フィールド）
            model_candidates = sorted(
                {
                    str(msg.get("model"))
                    for msg in (e["message"] for e in parsed_messages)
                    if msg.get("model")
                }
            )
            if model_candidates:
                selected_model = st.selectbox(
                    "モデルで絞り込み",
                    options=[None] + model_candidates,
                    format_func=lambda v: "すべて" if v is None else str(v),
                    key="chat_history_model_filter",
                )
                if selected_model is not None:
                    filter_applied = True
                    messages_to_display = [
                        entry
                        for entry in messages_to_display
                        if entry["message"].get("model") == selected_model
                    ]
            st.caption(f"チャット履歴件数: {len(parsed_messages)}")
        else:
            st.info("チャット履歴はまだありません。")

        if messages_to_display:
            for entry in messages_to_display:
                message = entry["message"]
                role = message.get("role", "assistant")
                ts_dt = entry["timestamp"]
                timestamp_label = (
                    ts_dt.strftime("%Y-%m-%d %H:%M") if ts_dt else None
                )
                with st.chat_message(role):
                    if timestamp_label:
                        st.caption(timestamp_label)
                    st.markdown(message.get("content", ""))
                    if role == "assistant" and message.get("model"):
                        tooltip_text = html.escape(message["model"])
                        st.markdown(
                            f'<span style="font-size:0.75rem; color:#888;" '
                            f'title="回答生成に使用したモデル">🛈 {tooltip_text}</span>',
                            unsafe_allow_html=True,
                        )
                        contexts_hist = message.get("contexts") or []
                        if contexts_hist:
                            with st.expander(
                                "この応答で使用したコンテキストを表示",
                                expanded=False,
                            ):
                                for i, ctx in enumerate(contexts_hist, start=1):
                                    st.markdown(f"**コンテキスト {i}**")
                                    st.text_area(
                                        f"hist_context_{timestamp_label}_{i}",
                                        value=str(ctx),
                                        height=120,
                                        key=f"chat_hist_context_{timestamp_label}_{i}",
                                    )
        elif filter_applied:
            st.info("選択された条件に一致するチャット履歴はありません。")

        if parsed_messages:
            st.divider()
            st.write("**日付別チャット履歴**")
            for date_value in sorted(
                {
                    entry["date"]
                    for entry in parsed_messages
                    if entry["date"] is not None
                },
                reverse=True,
            ):
                daily_entries = [
                    entry
                    for entry in parsed_messages
                    if entry["date"] == date_value
                ]
                daily_records = []
                for entry in daily_entries:
                    ts_dt = entry["timestamp"]
                    time_str = ts_dt.strftime("%H:%M:%S") if ts_dt else "-"
                    msg = entry["message"]
                    role_label = (
                        "ユーザー"
                        if msg.get("role") == "user"
                        else "アシスタント"
                    )
                    daily_records.append(
                        {
                            "時刻": time_str,
                            "役割": role_label,
                            "内容": msg.get("content", ""),
                            "モデル": msg.get("model") or "",
                            "Request ID": msg.get("request_id") or "",
                        }
                    )
                daily_df = (
                    pd.DataFrame(daily_records)
                    if daily_records
                    else pd.DataFrame(columns=["時刻", "役割", "内容"])
                )
                with st.expander(
                    f"{date_value} のチャット ({len(daily_records)} 件)",
                    expanded=(selected_chat_date == date_value),
                ):
                    st.dataframe(daily_df, use_container_width=True)
        
        # チャット入力
        can_send_message = not (scope == "single" and not selected_pdf_id)
        if not can_send_message:
            st.info(
                "検索対象が『単一PDFのみ』の場合は、上の『対象PDF』からPDFを1つ選択してからメッセージを送信してください。"
            )

        if can_send_message and (prompt := st.chat_input("メッセージを入力...")):
            user_timestamp = datetime.now(tz_jst).isoformat()
            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": prompt,
                    "model": None,
                    "timestamp": user_timestamp,
                }
            )
            with st.chat_message("user"):
                st.caption(
                    datetime.fromisoformat(user_timestamp).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                )
                st.markdown(prompt)
            
            # 選択されたモデルで応答を生成
            response_text = ""
            # llm_modelは常にst.session_state["chat_model"]を利用
            chat_model = st.session_state.get("chat_model")
            request_id: Optional[str] = None
            if not chat_model:
                response_text = (
                    "エラー: チャットボットモデルが未選択です。設定タブでモデルを選択してください。"
                )
                model_used: Optional[str] = None
                contexts: list[str] | None = None
            else:
                # --- RAGバックエンドAPIを呼び出して実際の応答を取得 ---
                contexts = None
                try:
                    current_scope = st.session_state.get("rag_scope", "single")
                    current_pdf_id = (
                        st.session_state.get("rag_pdf_file_id")
                        if current_scope == "single"
                        else None
                    )
                    query_payload: Dict[str, Any] = {
                        "query": prompt,
                        "llm_model": chat_model,
                        "embedding_model": st.session_state.get(
                            "embedding_model", "huggingface_bge_small"
                        ),
                        "scope": current_scope,
                    }
                    if current_scope == "single" and current_pdf_id:
                        query_payload["pdf_file_id"] = current_pdf_id
                    # タイムアウトを無効化（timeout指定なし）
                    with st.spinner("AIエージェントが考え中です… 少しだけお待ちください。"):
                        response = http_post(
                            f"{BACKEND_URL}/query/", json=query_payload
                        )
                    request_id = response.headers.get("X-Request-ID")
                    model_used = None
                    if response.status_code == 200:
                        data = response.json()
                        response_text = data.get("answer", "（応答がありません）")
                        model_used = data.get("llm_model_used", chat_model)
                        contexts = data.get("contexts") or []
                    else:
                        response_text = (
                            f"APIエラー: {format_http_error(response)}"
                        )
                        contexts = None
                except Exception as e:  # noqa: BLE001
                    response_text = f"リクエストエラー: {str(e)}"
                    model_used = None
                    contexts = None

            # --- バックエンドAPIの応答のみを表示・履歴追加 ---
            with st.chat_message("assistant"):
                response_timestamp = datetime.now(tz_jst).isoformat()
                st.caption(
                    datetime.fromisoformat(response_timestamp).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                )
                st.markdown(response_text)
                if model_used:
                    st.markdown(
                        f'<span style="font-size:0.75rem; color:#888;" '
                        f'title="回答生成に使用したモデル">🛈 {html.escape(model_used)}</span>',
                        unsafe_allow_html=True,
                    )
                if request_id:
                    st.markdown(
                        f'<span style="font-size:0.7rem; color:#aaa;" '
                        f'title="この応答に対応するバックエンドのリクエストID">Request ID: {html.escape(request_id)}</span>',
                        unsafe_allow_html=True,
                    )
                if contexts:
                    with st.expander(
                        "この応答で使用したコンテキストを表示",
                        expanded=False,
                    ):
                        for i, ctx in enumerate(contexts, start=1):
                            st.markdown(f"**コンテキスト {i}**")
                            st.text_area(
                                f"live_context_{response_timestamp}_{i}",
                                value=str(ctx),
                                height=120,
                                key=f"chat_live_context_{response_timestamp}_{i}",
                            )
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "model": model_used,
                        "timestamp": response_timestamp,
                        "request_id": request_id,
                    }
                )

            # 画面を更新
            st.session_state["chat_scroll_to_bottom"] = True
            # 新しいメッセージ送信時は日付フィルタをリセットして全履歴を表示する
            st.session_state["chat_reset_history_filter"] = True
            st.rerun()
