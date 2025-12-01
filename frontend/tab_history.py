from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd
import streamlit as st
import plotly.express as px

from http_client import http_get, http_post, http_delete
from evaluation_history_ui import show_evaluation_history, _render_bulk_style_charts


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
            """履歴系API向けの簡易GETラッパー。

            フロントエンド共通の http_client.http_get を利用し、JSONレスポンスを優先的に返す。
            エラー時は 599 とエラーメッセージを含む辞書を返す。
            """
            try:
                kwargs = {}
                if timeout is not None:
                    kwargs["timeout"] = timeout
                resp = http_get(url, **kwargs)
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

                    # このPDFに紐づく一括評価結果の再表示
                    st.markdown("### このPDFの一括評価結果")
                    with st.spinner("評価結果を取得中..."):
                        code_e, data_e = history_api_get(
                            f"{BACKEND_URL}/history/experiments"
                        )
                    if code_e == 200 and isinstance(data_e, dict):
                        exp_items = data_e.get("items", data_e.get("experiments", [])) or []
                        pdf_id = selected_pdf.get("id")
                        # experimentsテーブルのpdf_file_idと紐づけて、このPDFに対応する実験のみ抽出
                        exp_for_pdf = [
                            it for it in exp_items if str(it.get("pdf_file_id")) == str(pdf_id)
                        ]
                        if not exp_for_pdf:
                            st.info("このPDFに対する一括評価結果はまだありません。")
                        else:
                            # 実験ごとのラベル（実験名＋作成日時）を整形
                            exp_labels: list[str] = []
                            for it in exp_for_pdf:
                                name = it.get("experiment_name") or f"ID:{it.get('id')}"
                                created_at = it.get("created_at") or ""
                                label = name
                                if created_at:
                                    label = f"{name}（{created_at}）"
                                exp_labels.append(label)

                            selected_exp_idx = st.selectbox(
                                "評価結果を表示する実験",
                                options=list(range(len(exp_for_pdf))),
                                format_func=lambda i: exp_labels[i],
                                key=f"pdf_eval_experiment_select_{pdf_id}",
                            )
                            selected_exp = exp_for_pdf[selected_exp_idx]
                            exp_id = selected_exp.get("id")

                            if exp_id is not None:
                                with st.spinner("実験結果を取得中..."):
                                    code_r, data_r = history_api_get(
                                        f"{BACKEND_URL}/history/experiments/{exp_id}/results"
                                    )
                                if code_r == 200 and isinstance(data_r, dict):
                                    results = data_r.get("items", data_r.get("results", [])) or []
                                    if results:
                                        result_df = pd.DataFrame(results)

                                        # 一覧用の代表的なカラムのみ抜粋して表示
                                        result_columns = [
                                            "embedding_model",
                                            "chunk_strategy",
                                            "chunk_size",
                                            "chunk_overlap",
                                            "num_chunks",
                                            "avg_chunk_len",
                                            "overall_score",
                                            "faithfulness",
                                            "answer_relevancy",
                                            "context_recall",
                                            "context_precision",
                                            "answer_correctness",
                                            "answer_similarity",
                                        ]
                                        available_result_columns = [
                                            col
                                            for col in result_columns
                                            if col in result_df.columns
                                        ]
                                        if available_result_columns:
                                            st.write("**評価結果一覧（このPDF）**")
                                            st.dataframe(
                                                result_df[available_result_columns],
                                                use_container_width=True,
                                            )

                                        # evaluation_history_uiと同じスタイルのグラフ群を再利用
                                        st.write("**このPDFの一括評価グラフ**")
                                        key_prefix = f"pdf_{pdf_id}_exp_{exp_id}_"
                                        _render_bulk_style_charts(
                                            result_df,
                                            key_prefix=key_prefix,
                                        )
                                    else:
                                        st.info("この実験の結果データがありません。")
                                else:
                                    st.error(f"評価結果取得エラー: {code_r} {data_r}")
                    else:
                        st.error(f"実験履歴取得エラー: {code_e} {data_e}")

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

                                        contexts = row.get("contexts") or []
                                        if contexts:
                                            with st.expander(
                                                "この応答で使用したコンテキストを表示",
                                                expanded=False,
                                            ):
                                                for i, ctx in enumerate(contexts, start=1):
                                                    st.markdown(f"**コンテキスト {i}**")
                                                    st.text_area(
                                                        f"context_{row.get('id')}_{i}",
                                                        value=str(ctx),
                                                        height=120,
                                                        key=f"chatlog_context_{row.get('id')}_{i}",
                                                    )

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

                                    ctx_list = row.get("contexts") or []
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
                                            "コンテキスト数": len(ctx_list),
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
                                    ctx_list = row.get("contexts") or []
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

        # --- チャットログダッシュボード ---
        st.markdown("---")
        st.subheader("チャットログダッシュボード")
        st.caption("チャットログを集計し、期間やPDF IDでフィルタして傾向を可視化します。")

        try:
            with st.spinner("チャットログを集計中..."):
                code_dash, data_dash = history_api_get(
                    f"{BACKEND_URL}/history/chat-logs?limit=1000"
                )
            if code_dash == 200 and isinstance(data_dash, dict):
                log_items_all = data_dash.get("items", []) or []
                if not log_items_all:
                    st.info("チャットログがまだありません。")
                else:
                    df = pd.DataFrame(log_items_all)

                    # 日付列を生成
                    if "created_at" in df.columns:
                        try:
                            created_ts = pd.to_datetime(
                                df["created_at"], errors="coerce"
                            )
                            df["date"] = created_ts.dt.date
                        except Exception:
                            df["date"] = pd.NaT
                    else:
                        df["date"] = pd.NaT

                    col_f1, col_f2 = st.columns(2)

                    # 日付範囲フィルタ
                    with col_f1:
                        if df["date"].notna().any():
                            min_date = df["date"].dropna().min()
                            max_date = df["date"].dropna().max()
                            from_date = st.date_input(
                                "日付 From",
                                value=min_date,
                                key="chat_dash_date_from",
                            )
                            to_date = st.date_input(
                                "日付 To",
                                value=max_date,
                                key="chat_dash_date_to",
                            )
                            if from_date:
                                df = df[df["date"] >= from_date]
                            if to_date:
                                df = df[df["date"] <= to_date]

                    # PDF ID フィルタ
                    with col_f2:
                        if "pdf_file_id" in df.columns:
                            pdf_ids = (
                                df["pdf_file_id"]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )
                            pdf_ids = sorted(pdf_ids)
                            if pdf_ids:
                                selected_pdf = st.selectbox(
                                    "PDF IDで絞り込み",
                                    options=[None] + pdf_ids,
                                    format_func=lambda v: "すべて" if v is None else str(v),
                                    key="chat_dash_pdf_filter",
                                )
                                if selected_pdf is not None:
                                    df = df[
                                        df["pdf_file_id"].astype(str)
                                        == str(selected_pdf)
                                    ]

                    if df.empty:
                        st.info(
                            "選択された条件に一致するチャットログがありません。フィルタ条件を見直してください。"
                        )
                    else:
                        total_msgs = len(df)
                        unique_pdfs = (
                            df["pdf_file_id"].dropna().nunique()
                            if "pdf_file_id" in df.columns
                            else 0
                        )
                        unique_days = df["date"].dropna().nunique()

                        col_k1, col_k2, col_k3 = st.columns(3)
                        with col_k1:
                            st.metric("メッセージ数", total_msgs)
                        with col_k2:
                            st.metric("PDF数", unique_pdfs)
                        with col_k3:
                            st.metric("日数", unique_days)

                        # 日別メッセージ数
                        if df["date"].notna().any():
                            st.write("**日別メッセージ数**")
                            daily = (
                                df.groupby("date").size().reset_index(name="count")
                            )
                            daily = daily.sort_values("date")
                            daily = daily.set_index("date")
                            st.bar_chart(daily["count"])

                        # PDF別メッセージ数 上位5件
                        if "pdf_file_id" in df.columns:
                            st.write("**PDF別メッセージ数 上位5件**")
                            pdf_counts = (
                                df["pdf_file_id"]
                                .astype(str)
                                .value_counts()
                                .head(5)
                                .reset_index()
                            )
                            pdf_counts.columns = ["pdf_file_id", "count"]
                            st.dataframe(pdf_counts, use_container_width=True)

                        # LLMモデル別メッセージ数
                        if "llm_model_used" in df.columns:
                            st.write("**LLMモデル別メッセージ数**")
                            model_counts = (
                                df["llm_model_used"]
                                .fillna("(不明)")
                                .astype(str)
                                .value_counts()
                                .reset_index()
                            )
                            model_counts.columns = ["llm_model_used", "count"]
                            st.dataframe(model_counts, use_container_width=True)

                        # scope 別メッセージ数
                        if "scope" in df.columns:
                            st.write("**scope 別メッセージ数**")
                            scope_counts = (
                                df["scope"]
                                .fillna("(不明)")
                                .astype(str)
                                .value_counts()
                                .reset_index()
                            )
                            scope_counts.columns = ["scope", "count"]
                            col_sc1, col_sc2 = st.columns([1, 1])
                            with col_sc1:
                                st.dataframe(scope_counts, use_container_width=True)
                            with col_sc2:
                                try:
                                    fig_scope = px.bar(
                                        scope_counts,
                                        x="scope",
                                        y="count",
                                        title="scope 別メッセージ数",
                                    )
                                    st.plotly_chart(fig_scope, use_container_width=True)
                                except Exception:
                                    pass

                        # モデル×PDF のヒートマップ（メッセージ数）
                        if "pdf_file_id" in df.columns and "llm_model_used" in df.columns:
                            st.write("**モデル×PDF のメッセージ数ヒートマップ**")
                            cross_df = (
                                df.assign(
                                    pdf_file_id_str=df["pdf_file_id"].astype(str),
                                    llm_model_str=df["llm_model_used"].fillna("(不明)").astype(str),
                                )
                                .groupby(["pdf_file_id_str", "llm_model_str"])
                                .size()
                                .reset_index(name="count")
                            )
                            if not cross_df.empty:
                                try:
                                    fig_heat = px.density_heatmap(
                                        cross_df,
                                        x="llm_model_str",
                                        y="pdf_file_id_str",
                                        z="count",
                                        color_continuous_scale="Blues",
                                        labels={
                                            "llm_model_str": "LLMモデル",
                                            "pdf_file_id_str": "PDF ID",
                                            "count": "メッセージ数",
                                        },
                                        title="LLMモデル×PDF のメッセージ数ヒートマップ",
                                    )
                                    fig_heat.update_layout(height=400)
                                    st.plotly_chart(fig_heat, use_container_width=True)
                                except Exception:
                                    st.dataframe(cross_df, use_container_width=True)

                        # プロンプト・応答長の統計
                        st.write("**プロンプト・応答長の統計**")
                        df_lengths = df.copy()
                        df_lengths["prompt_len"] = df_lengths.get("user_message", "").astype(str).str.len()
                        df_lengths["answer_len"] = df_lengths.get("assistant_message", "").astype(str).str.len()

                        len_stats = df_lengths[["prompt_len", "answer_len"]].agg(
                            ["mean", "median", "max"]
                        )
                        len_stats = len_stats.round(1)
                        st.dataframe(len_stats, use_container_width=True)

                        col_len1, col_len2 = st.columns(2)
                        with col_len1:
                            try:
                                fig_prompt = px.histogram(
                                    df_lengths,
                                    x="prompt_len",
                                    nbins=30,
                                    title="プロンプト長の分布",
                                    labels={"prompt_len": "文字数"},
                                )
                                st.plotly_chart(fig_prompt, use_container_width=True)
                            except Exception:
                                pass
                        with col_len2:
                            try:
                                fig_answer = px.histogram(
                                    df_lengths,
                                    x="answer_len",
                                    nbins=30,
                                    title="応答長の分布",
                                    labels={"answer_len": "文字数"},
                                )
                                st.plotly_chart(fig_answer, use_container_width=True)
                            except Exception:
                                pass
            else:
                st.error(f"チャットログ取得エラー: {code_dash} {data_dash}")
        except Exception as e:  # noqa: BLE001
            st.error(f"チャットログダッシュボードの集計中にエラーが発生しました: {e}")

        st.markdown("---")
        st.subheader("一括評価結果一覧")
        st.caption("実行済みの実測ごとの評価メトリクスと詳細を参照できます。")
        try:
            with st.expander("評価履歴", expanded=True):
                show_evaluation_history(BACKEND_URL.rstrip("/"))
        except Exception as e:  # noqa: BLE001
            st.error(f"評価履歴の表示に失敗しました: {e}")

        st.markdown("---")
        st.subheader("最近の失敗ジョブ")
        st.caption("upload_jobs / evaluation_jobs のうち、直近でエラーになったジョブを一覧表示します。")

        try:
            with st.spinner("失敗ジョブ一覧を取得中..."):
                code_jobs, data_jobs = history_api_get(f"{BACKEND_URL}/history/jobs/errors?limit=50")
            if code_jobs == 200 and isinstance(data_jobs, dict):
                upload_jobs = data_jobs.get("upload_jobs", []) or []
                eval_jobs = data_jobs.get("evaluation_jobs", []) or []

                if not upload_jobs and not eval_jobs:
                    st.info("現在、失敗状態のジョブは記録されていません。")
                else:
                    col_u, col_e = st.columns(2)

                    with col_u:
                        st.markdown("#### PDFアップロードジョブの失敗履歴")
                        if upload_jobs:
                            u_rows = []
                            for it in upload_jobs:
                                u_rows.append(
                                    {
                                        "job_id": it.get("job_id"),
                                        "file_id": it.get("file_id"),
                                        "status": it.get("status"),
                                        "updated_at": it.get("updated_at"),
                                        "error": it.get("error"),
                                        "progress": it.get("progress"),
                                    }
                                )
                            u_df = pd.DataFrame(u_rows)
                            if not u_df.empty:
                                st.dataframe(u_df, use_container_width=True)
                        else:
                            st.caption("失敗したアップロードジョブはありません。")

                    with col_e:
                        st.markdown("#### 一括評価ジョブの失敗履歴")
                        if eval_jobs:
                            e_rows = []
                            for it in eval_jobs:
                                e_rows.append(
                                    {
                                        "job_id": it.get("job_id"),
                                        "status": it.get("status"),
                                        "updated_at": it.get("updated_at"),
                                        "error": it.get("error"),
                                        "progress": it.get("progress"),
                                    }
                                )
                            e_df = pd.DataFrame(e_rows)
                            if not e_df.empty:
                                st.dataframe(e_df, use_container_width=True)
                        else:
                            st.caption("失敗した一括評価ジョブはありません。")
            else:
                st.error(f"失敗ジョブ取得エラー: {code_jobs} {data_jobs}")
        except Exception as e:  # noqa: BLE001
            st.error(f"失敗ジョブ一覧取得中にエラーが発生しました: {e}")
