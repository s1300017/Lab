from __future__ import annotations

from datetime import datetime, date
from typing import List, Dict, Any, Optional, Callable

import html
import pandas as pd
from pytz import timezone
import streamlit as st

from http_client import http_get, http_post


def render_chatbot_tab(
    tab, BACKEND_URL: str, save_state_to_localstorage: Callable[[], None]
) -> None:
    """チャットボットタブのUIとロジックを描画する。"""
    with tab:
        st.header("チャットボット")

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
                                    }
                                )
                except Exception as e:  # noqa: BLE001
                    st.warning(f"チャット履歴の取得に失敗しました: {e}")

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
                filter_applied = selected_chat_date is not None
                if filter_applied:
                    messages_to_display = [
                        entry
                        for entry in parsed_messages
                        if entry["date"] == selected_chat_date
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
        elif filter_applied:
            st.info("選択された日付のチャット履歴はありません。")

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
        if prompt := st.chat_input("メッセージを入力..."):
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
            if not chat_model:
                response_text = (
                    "エラー: チャットボットモデルが未選択です。設定タブでモデルを選択してください。"
                )
            else:
                # --- RAGバックエンドAPIを呼び出して実際の応答を取得 ---
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
                    model_used: Optional[str] = None
                    if response.status_code == 200:
                        data = response.json()
                        response_text = data.get("answer", "（応答がありません）")
                        model_used = data.get("llm_model_used", chat_model)
                    else:
                        response_text = (
                            f"APIエラー: {response.status_code} - {response.text}"
                        )
                except Exception as e:  # noqa: BLE001
                    response_text = f"リクエストエラー: {str(e)}"
                    model_used = None
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
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "model": model_used,
                        "timestamp": response_timestamp,
                    }
                )

            # 画面を更新
            st.session_state["chat_scroll_to_bottom"] = True
            st.rerun()
