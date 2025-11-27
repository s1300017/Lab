from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd
import streamlit as st

from http_client import http_post, http_delete
from evaluation_history_ui import show_evaluation_history


def render_history_tab(
    tab_history: Any,
    BACKEND_URL: str,
    save_state_to_localstorage: Callable[[], None],
) -> None:
    """履歴タブのUIとロジックを描画する。"""
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
            if st.button(
                "ベクトル＆チャンクのみリセット",
                disabled=not confirm_reset_vectors,
                key="btn_reset_vectors_and_chunks",
            ):
                with st.spinner("ベクトルとチャンクをリセット中..."):
                    try:
                        resp = http_post(
                            f"{BACKEND_URL}/admin/reset_vectors_and_chunks/"
                        )
                        if resp.status_code == 200:
                            data = (
                                resp.json()
                                if resp.headers.get("Content-Type", "").startswith(
                                    "application/json"
                                )
                                else {}
                            )
                            msg = data.get(
                                "message", "ベクトルとチャンクをリセットしました。"
                            )
                            st.success(msg)
                        else:
                            st.error(
                                f"リセットに失敗しました: {resp.status_code} {resp.text}"
                            )
                    except Exception as e:  # noqa: BLE001
                        st.error(f"リセット中にエラーが発生しました: {e}")

            st.markdown("---")

            # PDFも含めてRAG関連を全リセット
            st.markdown("#### PDFも含めてRAG関連を全リセット")
            confirm_reset_all = st.text_input(
                "PDFファイルと抽出JSONも削除します。確認のため 'RESET ALL' と入力してください",
                value="",
                key="confirm_reset_pdfs_and_vectors",
            )
            disabled_reset_all = confirm_reset_all.strip().upper() != "RESET ALL"
            if st.button(
                "PDFとRAGベクトルを全リセット",
                type="secondary",
                disabled=disabled_reset_all,
                key="btn_reset_pdfs_and_vectors",
            ):
                with st.spinner("PDFおよびRAGベクトルをリセット中..."):
                    try:
                        resp = http_post(
                            f"{BACKEND_URL}/admin/reset_pdfs_and_vectors/"
                        )
                        if resp.status_code == 200:
                            data = (
                                resp.json()
                                if resp.headers.get("Content-Type", "").startswith(
                                    "application/json"
                                )
                                else {}
                            )
                            msg = data.get(
                                "message",
                                "PDFおよびRAGベクトルをリセットしました。",
                            )
                            st.success(msg)
                            st.json(data)
                            st.info("必要に応じてページを再読み込みしてください。")
                        else:
                            st.error(
                                f"リセットに失敗しました: {resp.status_code} {resp.text}"
                            )
                    except Exception as e:  # noqa: BLE001
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
            confirm_text = st.text_input(
                "確認のため 'CONFIRM' と入力してください", value=""
            )
            disabled = confirm_text.strip().upper() != "CONFIRM"
            if st.button(
                "DBを全消去（バックアップ付き）",
                type="primary",
                disabled=disabled,
            ):
                with st.spinner("全消去を実行中..."):
                    payload = {
                        "confirm": True,
                        "backup": bool(make_backup),
                        "delete_files": bool(delete_files),
                    }
                    try:
                        resp = http_post(
                            f"{BACKEND_URL}/admin/wipe_all", json=payload
                        )
                        if resp.status_code == 200:
                            res = resp.json()
                            st.success(
                                "全消去が完了しました。アプリを再読み込みします。"
                            )
                            st.json(res)
                            st.rerun()
                        else:
                            st.error(
                                f"全消去に失敗しました: {resp.status_code} {resp.text}"
                            )
                    except Exception as e:  # noqa: BLE001
                        st.error(f"エラーが発生しました: {e}")

        # ユーティリティ: GETリクエスト（簡易ラッパー）
        def history_api_get(url: str, timeout: float | None = None):
            import requests

            try:
                resp = requests.get(url, timeout=timeout)
                return resp.status_code, (
                    resp.json()
                    if resp.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else resp.text
                )
            except Exception as e:  # noqa: BLE001
                return 599, {"error": str(e)}

        # PDF一覧の取得
        with st.expander("PDF一覧", expanded=True):
            with st.spinner("PDF一覧を取得中..."):
                code, data = history_api_get(f"{BACKEND_URL}/history/pdf-files")
            if code == 200 and isinstance(data, dict):
                pdf_items = data.get("items", [])
                if not pdf_items:
                    st.info(
                        "登録済みのPDFがありません。先にPDFをアップロードしてください。"
                    )
                else:
                    # セレクトボックス用にラベル整形（ファイル名 + 追加日時）
                    labels: list[str] = []
                    pdf_label_map: dict[str, str] = {}
                    for it in pdf_items:
                        base_name = (
                            it.get("original_name")
                            or it.get("file_name")
                            or str(it.get("id"))
                        )
                        uploaded_at = it.get("uploaded_at") or ""
                        label = base_name
                        if uploaded_at:
                            label = f"{base_name}（追加日時: {uploaded_at}）"
                        labels.append(label)
                        pdf_label_map[it.get("id")] = label
                    idx = st.selectbox(
                        "PDFを選択",
                        options=list(range(len(pdf_items))),
                        format_func=lambda i: labels[i],
                    )
                    selected_pdf = pdf_items[idx]
                    st.markdown("---")
                    st.subheader("選択中のPDF")
                    st.json(selected_pdf)

                    # PDF削除操作を提供
                    st.markdown("#### PDF削除")
                    delete_confirm = (
                        st.text_input(
                            "削除確認のため 'DELETE' と入力してください",
                            value="",
                            key=f"delete_confirm_{selected_pdf['id']}",
                        )
                        .strip()
                        .upper()
                    )
                    delete_disabled = delete_confirm != "DELETE"
                    if st.button(
                        "このPDFを削除",
                        key=f"delete_pdf_{selected_pdf['id']}",
                        disabled=delete_disabled,
                    ):
                        with st.spinner("PDFを削除中です..."):
                            try:
                                delete_resp = http_delete(
                                    f"{BACKEND_URL}/pdf/{selected_pdf['id']}"
                                )
                                if delete_resp.status_code == 200:
                                    st.success(
                                        "PDFを削除しました。リストを更新します。"
                                    )
                                    st.rerun()
                                else:
                                    st.error(
                                        f"削除に失敗しました: {delete_resp.status_code} {delete_resp.text}"
                                    )
                            except Exception as delete_error:  # noqa: BLE001
                                st.error(
                                    f"削除処理中にエラーが発生しました: {delete_error}"
                                )

                    # 履歴から抽出データを復元して全タブを有効化
                    if st.button(
                        "このPDFを開く（抽出テキスト・QAを復元）",
                        key=f"restore_pdf_{selected_pdf['id']}",
                    ):
                        with st.spinner("PDF抽出データを復元中..."):
                            code_x, data_x = history_api_get(
                                f"{BACKEND_URL}/get_extracted/{selected_pdf['id']}"
                            )
                            if code_x == 200 and isinstance(data_x, dict):
                                st.session_state["file_id"] = selected_pdf["id"]
                                st.session_state.text = data_x.get("text", "")
                                st.session_state.qa_questions = data_x.get(
                                    "questions", []
                                )
                                st.session_state.qa_answers = data_x.get(
                                    "answers", []
                                )
                                st.session_state.qa_meta = data_x.get(
                                    "qa_meta", []
                                )
                                st.session_state.image_captions = data_x.get(
                                    "image_captions", []
                                )
                                save_state_to_localstorage()
                                st.success(
                                    "PDFを復元しました。各タブから操作できます。"
                                )
                                st.rerun()

                    # 生成QA一覧
                    st.markdown("### 生成QA一覧")
                    with st.spinner("生成QAを取得中..."):
                        code_q, data_q = history_api_get(
                            f"{BACKEND_URL}/history/pdf-files/{selected_pdf['id']}/questions"
                        )
                    if code_q == 200 and isinstance(data_q, dict):
                        qa_items = data_q.get("items", [])
                        if qa_items:
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
                                        except Exception:  # noqa: BLE001
                                            meta = {"raw": meta_raw}
                                    elif isinstance(meta_raw, dict):
                                        meta = meta_raw
                                score = meta.get("score") or meta.get(
                                    "total_score"
                                )
                                if score is None and answer:
                                    score = float(len(answer)) / 1000.0
                                    meta["score"] = score
                                qa_tuples.append((question, answer, meta))

                            qa_tuples_sorted = sorted(
                                qa_tuples,
                                key=lambda x: x[2].get("score", 0),
                                reverse=True,
                            )

                            st.markdown("#### 自動生成QAセット（信頼性スコア順）")
                            for idx, (question, answer, meta) in enumerate(
                                qa_tuples_sorted, start=1
                            ):
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
                                            f"信頼性スコア: {score:.3f}"
                                            if isinstance(score, (int, float))
                                            else "信頼性スコア: -"
                                        )

                                    candidates = meta.get("candidates", [])
                                    candidate_scores = meta.get(
                                        "candidate_scores", []
                                    )
                                    if candidates and len(candidates) > 1:
                                        with st.expander(
                                            "候補回答リスト（スコア付き）"
                                        ):
                                            for cand, cand_score in zip(
                                                candidates, candidate_scores
                                            ):
                                                if isinstance(cand_score, (int, float)):
                                                    st.markdown(
                                                        f"- {cand}（スコア: {cand_score:.3f}）"
                                                    )
                                                else:
                                                    st.markdown(f"- {cand}")
                        else:
                            st.info("生成QAはまだ保存されていません。")
                    else:
                        st.error(f"生成QA取得エラー: {code_q} {data_q}")

                    st.markdown("### チャンク一覧")
                    with st.spinner("チャンクを取得中..."):
                        code_c, data_c = history_api_get(
                            f"{BACKEND_URL}/history/pdf-files/{selected_pdf['id']}/chunks"
                        )
                    if code_c == 200 and isinstance(data_c, dict):
                        chunk_items = data_c.get("items", [])
                        if chunk_items:
                            for it in chunk_items:
                                content = it.get("content", "")
                                it["preview"] = content[:100] + (
                                    "..." if len(content) > 100 else ""
                                )
                            chunk_df = pd.DataFrame(chunk_items)
                            st.dataframe(
                                chunk_df[
                                    [c for c in chunk_df.columns if c != "content"]
                                ],
                                use_container_width=True,
                            )
                            if st.button("このPDFの抽出データを復元"):
                                with st.spinner("PDF抽出データを復元中..."):
                                    code_x, data_x = history_api_get(
                                        f"{BACKEND_URL}/get_extracted/{selected_pdf['id']}"
                                    )
                                    if code_x == 200 and isinstance(data_x, dict):
                                        st.session_state["file_id"] = selected_pdf["id"]
                                        st.session_state.text = data_x.get("text", "")
                                        st.session_state.qa_questions = data_x.get(
                                            "questions", []
                                        )
                                        st.session_state.qa_answers = data_x.get(
                                            "answers", []
                                        )
                                        st.session_state.qa_meta = data_x.get(
                                            "qa_meta", []
                                        )
                                        st.session_state.image_captions = data_x.get(
                                            "image_captions", []
                                        )
                                        save_state_to_localstorage()
                                        st.success(
                                            "PDFを復元しました。各タブから操作できます。"
                                        )
                                        st.rerun()
                    else:
                        st.error(f"チャンク取得エラー: {code_c} {data_c}")

                    # チャット履歴（PDF単位／全PDF横断）
                    st.markdown("### チャット履歴")
                    chat_tab_pdf, chat_tab_all = st.tabs([
                        "このPDFのチャット履歴",
                        "すべてのPDF",
                    ])

                    # このPDFに紐づくチャット履歴（縦積みチャット風表示）
                    with chat_tab_pdf:
                        with st.spinner("チャット履歴を取得中..."):
                            code_logs, data_logs = history_api_get(
                                f"{BACKEND_URL}/history/chat-logs?pdf_file_id={selected_pdf['id']}"
                            )
                        if code_logs == 200 and isinstance(data_logs, dict):
                            log_items = data_logs.get("items", []) or []
                            if not log_items:
                                st.info("このPDFのチャット履歴はまだありません。")
                            else:
                                date_set: set[str] = set()
                                for row in log_items:
                                    created_at = row.get("created_at")
                                    if (
                                        isinstance(created_at, str)
                                        and len(created_at) >= 10
                                    ):
                                        date_set.add(created_at[:10])
                                unique_dates = sorted(list(date_set), reverse=True)
                                selected_date = st.selectbox(
                                    "表示する日付",
                                    options=["すべて"] + unique_dates,
                                    format_func=lambda v: "すべて"
                                    if v == "すべて"
                                    else str(v),
                                    key=f"chat_logs_date_{selected_pdf['id']}",
                                )
                                if selected_date != "すべて":
                                    display_items = [
                                        row
                                        for row in log_items
                                        if isinstance(row.get("created_at"), str)
                                        and row["created_at"].startswith(selected_date)
                                    ]
                                else:
                                    display_items = list(log_items)
                                display_items_sorted = sorted(
                                    display_items,
                                    key=lambda r: r.get("created_at") or "",
                                )
                                current_date_label: str | None = None
                                for row in display_items_sorted:
                                    created_at = row.get("created_at")
                                    if isinstance(created_at, str) and created_at:
                                        ts_label = created_at.replace(
                                            "T", " "
                                        ).split("+")[0]
                                        date_label = ts_label[:10]
                                    else:
                                        ts_label = ""
                                        date_label = None

                                    if date_label and date_label != current_date_label:
                                        if current_date_label is not None:
                                            st.markdown("---")
                                        current_date_label = date_label
                                        st.markdown(
                                            f"#### {date_label} の会話"
                                        )

                                    user_msg = row.get("user_message") or ""
                                    assistant_msg = (
                                        row.get("assistant_message") or ""
                                    )
                                    model_used = row.get("llm_model_used")
                                    embedding_model = row.get("embedding_model")
                                    with st.chat_message("user"):
                                        if ts_label:
                                            st.caption(ts_label)
                                        st.markdown(
                                            user_msg or "（メッセージなし）"
                                        )
                                    with st.chat_message("assistant"):
                                        if ts_label:
                                            st.caption(ts_label)
                                        st.markdown(
                                            assistant_msg or "（応答なし）"
                                        )
                                        meta_parts: list[str] = []
                                        if model_used:
                                            meta_parts.append(
                                                f"LLM: {model_used}"
                                            )
                                        if embedding_model:
                                            meta_parts.append(
                                                f"Embedding: {embedding_model}"
                                            )
                                        if meta_parts:
                                            st.caption(" / ".join(meta_parts))

                                processed_rows: list[dict[str, Any]] = []
                                for row in log_items:
                                    created_at = row.get("created_at")
                                    if isinstance(created_at, str):
                                        date_str = created_at[:10]
                                        time_str = (
                                            created_at[11:19]
                                            if len(created_at) >= 19
                                            else ""
                                        )
                                    else:
                                        date_str = None
                                        time_str = ""
                                    processed_rows.append(
                                        {
                                            "日付": date_str,
                                            "時刻": time_str,
                                            "ユーザーメッセージ": row.get(
                                                "user_message", ""
                                            ),
                                            "アシスタント応答": row.get(
                                                "assistant_message", ""
                                            ),
                                            "LLMモデル": row.get("llm_model_used"),
                                            "Embeddingモデル": row.get(
                                                "embedding_model"
                                            ),
                                            "scope": row.get("scope"),
                                        }
                                    )
                                log_df = pd.DataFrame(processed_rows)
                                if not log_df.empty:
                                    log_df = log_df.sort_values(
                                        ["日付", "時刻"], ascending=[False, False]
                                    )
                                    with st.expander("表形式で表示"):
                                        st.dataframe(
                                            log_df, use_container_width=True
                                        )
                        else:
                            st.error(
                                f"チャット履歴取得エラー: {code_logs} {data_logs}"
                            )

                    # 全PDF横断のチャット履歴
                    with chat_tab_all:
                        with st.spinner("全PDFのチャット履歴を取得中..."):
                            code_all, data_all = history_api_get(
                                f"{BACKEND_URL}/history/chat-logs?limit=500"
                            )
                        if code_all == 200 and isinstance(data_all, dict):
                            all_items = data_all.get("items", []) or []
                            if not all_items:
                                st.info("チャット履歴はまだありません。")
                            else:
                                processed_rows_all: list[dict[str, Any]] = []
                                for row in all_items:
                                    created_at = row.get("created_at")
                                    if isinstance(created_at, str):
                                        date_str = created_at[:10]
                                        time_str = (
                                            created_at[11:19]
                                            if len(created_at) >= 19
                                            else ""
                                        )
                                    else:
                                        date_str = None
                                        time_str = ""
                                    fid = row.get("pdf_file_id")
                                    pdf_label = pdf_label_map.get(
                                        fid,
                                        str(fid) if fid else "（PDF未指定）",
                                    )
                                    processed_rows_all.append(
                                        {
                                            "PDF": pdf_label,
                                            "日付": date_str,
                                            "時刻": time_str,
                                            "ユーザーメッセージ": row.get(
                                                "user_message", ""
                                            ),
                                            "アシスタント応答": row.get(
                                                "assistant_message", ""
                                            ),
                                            "LLMモデル": row.get("llm_model_used"),
                                            "Embeddingモデル": row.get(
                                                "embedding_model"
                                            ),
                                            "scope": row.get("scope"),
                                        }
                                    )
                                all_df = pd.DataFrame(processed_rows_all)
                                if not all_df.empty:
                                    all_df = all_df.sort_values(
                                        ["日付", "時刻"], ascending=[False, False]
                                    )
                                    unique_pdfs = sorted(
                                        {
                                            p
                                            for p in all_df["PDF"].dropna().unique()
                                        }
                                    )
                                    pdf_option = st.selectbox(
                                        "PDFを絞り込み",
                                        options=["すべて"] + unique_pdfs,
                                        key="chat_logs_all_pdf_filter",
                                    )
                                    filtered_df = all_df
                                    if pdf_option != "すべて":
                                        filtered_df = filtered_df[
                                            filtered_df["PDF"] == pdf_option
                                        ]
                                    unique_dates_all = sorted(
                                        {
                                            d
                                            for d in filtered_df["日付"]
                                            .dropna()
                                            .unique()
                                        },
                                        reverse=True,
                                    )
                                    date_option = st.selectbox(
                                        "日付を絞り込み",
                                        options=["すべて"] + unique_dates_all,
                                        key="chat_logs_all_date_filter",
                                    )
                                    if date_option != "すべて":
                                        filtered_df = filtered_df[
                                            filtered_df["日付"] == date_option
                                        ]
                                    filtered_df = filtered_df.sort_values(
                                        ["日付", "時刻"], ascending=[False, False]
                                    )
                                    st.dataframe(
                                        filtered_df, use_container_width=True
                                    )
                                else:
                                    st.info("チャット履歴はまだありません。")
                        else:
                            st.error(
                                f"チャット履歴取得エラー: {code_all} {data_all}"
                            )

        st.markdown("---")
        st.subheader("一括評価結果一覧")
        st.caption("実行済みの実測ごとの評価メトリクスと詳細を参照できます。")
        try:
            with st.expander("評価履歴", expanded=True):
                show_evaluation_history(BACKEND_URL.rstrip("/"))
        except Exception as e:  # noqa: BLE001
            st.error(f"評価履歴の表示に失敗しました: {e}")
