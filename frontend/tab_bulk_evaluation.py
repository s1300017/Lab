from __future__ import annotations

from typing import Any, Dict, List

import base64
import time
import re
import pandas as pd
import streamlit as st

from http_client import http_get, http_post, format_http_error
from evaluation_history_ui import _render_bulk_style_charts, apply_bulk_chunk_settings_from_history
from graph_utils import create_zip_with_graphs
from model_utils import (
    fetch_embedding_models as _fetch_embedding_models_common,
    fetch_llm_models as _fetch_llm_models_common,
    fetch_history_pdfs as _fetch_history_pdfs_common,
)


def _fetch_embedding_models(BACKEND_URL: str) -> List[Dict[str, str]]:
    """バックエンドの /list_models からEmbeddingモデル一覧を取得するヘルパー。"""
    return _fetch_embedding_models_common(BACKEND_URL.rstrip("/"))


def _fetch_llm_models(BACKEND_URL: str) -> List[Dict[str, str]]:
    return _fetch_llm_models_common(BACKEND_URL.rstrip("/"))


def _persist_model_selection(BACKEND_URL: str, llm_model: str, embedding_model: str) -> None:
    """選択されたLLM/Embeddingモデルをバックエンドに保存する。"""
    payload = {
        "llm_model": llm_model,
        "embedding_model": embedding_model,
    }
    key_llm = "_last_persisted_llm_model"
    key_emb = "_last_persisted_embedding_model"

    if (
        st.session_state.get(key_llm) == llm_model
        and st.session_state.get(key_emb) == embedding_model
    ):
        return

    bulk_job_id = st.session_state.get("bulk_eval_job_id")
    bulk_job_status = (st.session_state.get("bulk_eval_job_status") or "").lower()
    bulk_job_active = bool(bulk_job_id) and bulk_job_status in ("pending", "running")
    if bulk_job_active:
        return

    try:
        http_post(f"{BACKEND_URL}/config/model_selection", json=payload)
    except Exception as e:  # noqa: BLE001
        st.warning(f"モデル選択の保存に失敗しました: {e}")
    else:
        st.session_state[key_llm] = llm_model
        st.session_state[key_emb] = embedding_model


def _fetch_history_pdfs(BACKEND_URL: str) -> List[Dict[str, Any]]:
    """/history/pdf-files からPDF一覧を取得するヘルパー（共通キャッシュ利用）。"""
    return _fetch_history_pdfs_common(BACKEND_URL.rstrip("/"))


def _load_qa_from_history(
    BACKEND_URL: str, pdf_id: str
) -> tuple[str, List[str], List[str]]:
    """指定PDF IDに紐づくテキスト・QAセットを履歴APIから取得する。

    優先度:
    1. /get_extracted/{file_id} の JSON（text/questions/answers）
    2. /history/pdf-files/{file_id}/questions から Q/A を取得し、
       /history/pdf-files/{file_id}/chunks を連結してテキストを組み立て
    """
    text = ""
    questions: List[str] = []
    answers: List[str] = []

    # 1. 抽出JSONからの復元を試す
    try:
        resp_x = http_get(f"{BACKEND_URL}/get_extracted/{pdf_id}")
        if resp_x.status_code == 200 and resp_x.headers.get("Content-Type", "").startswith(
            "application/json"
        ):
            data_x = resp_x.json() or {}
            text = data_x.get("text", "") or ""
            questions = data_x.get("questions", []) or []
            answers = data_x.get("answers", []) or []
            if text and questions and answers:
                return text, questions, answers
    except Exception as e:  # noqa: BLE001
        st.warning(f"抽出データ復元中にエラーが発生しました: {e}")

    # 2. 生成QAテーブルとチャンクからのフォールバック
    try:
        resp_q = http_get(
            f"{BACKEND_URL}/history/pdf-files/{pdf_id}/questions"
        )
        if resp_q.status_code == 200 and resp_q.headers.get("Content-Type", "").startswith(
            "application/json"
        ):
            data_q = resp_q.json() or {}
            items = data_q.get("items", []) or []
            for row in items:
                q = row.get("question") or ""
                a = row.get("answer") or ""
                if q and a:
                    questions.append(q)
                    answers.append(a)
    except Exception as e:  # noqa: BLE001
        st.warning(f"生成QA履歴の取得中にエラーが発生しました: {e}")

    # テキストはチャンクを連結して近似的に再構成
    try:
        resp_c = http_get(
            f"{BACKEND_URL}/history/pdf-files/{pdf_id}/chunks"
        )
        if resp_c.status_code == 200 and resp_c.headers.get("Content-Type", "").startswith(
            "application/json"
        ):
            data_c = resp_c.json() or {}
            chunk_items = data_c.get("items", []) or []
            contents = [it.get("content", "") for it in chunk_items]
            text = "\n\n".join([c for c in contents if c])
    except Exception as e:  # noqa: BLE001
        st.warning(f"チャンク履歴からのテキスト再構成中にエラーが発生しました: {e}")

    return text, questions, answers


def render_bulk_evaluation_tab(tab_bulk: Any, BACKEND_URL: str) -> None:
    """一括評価タブのUIとロジックを描画する。

    - 評価用LLMはバックエンド側で GPT-OSS 固定。
    - フロントでは Embedding モデルとチャンク設定のみを指定し、/bulk_evaluate/ を1回呼び出す。
    - 結果は DataFrame とグラフ（evaluation_history_ui の一括評価スタイル）で表示し、ZIP出力も提供する。
    """
    with tab_bulk:
        st.header("RAGAS一括評価")
        st.caption(
            "履歴に保存されたPDFごとのQAセットを用いて、\n"
            "埋め込みモデルとチャンク設定ごとのRAG性能を一括で評価します。\n"
            "評価用LLMはバックエンド側で GPT-OSS 固定です。"
        )

        # すでに一括評価ジョブが動いている場合は、まずステータスのみを確認して表示する
        bulk_job_id = st.session_state.get("bulk_eval_job_id")
        if bulk_job_id:
            try:
                resp_status = http_get(f"{BACKEND_URL}/bulk_job/status/{bulk_job_id}")
            except Exception as e:  # noqa: BLE001
                st.error(f"一括評価ジョブ状態取得中にエラーが発生しました: {e}")
                return
            else:
                if resp_status.status_code != 200:
                    st.error(
                        f"一括評価ジョブ状態取得エラー: {format_http_error(resp_status)}"
                    )
                    return

                data_status = (
                    resp_status.json()
                    if resp_status.headers.get("Content-Type", "").startswith(
                        "application/json"
                    )
                    else {}
                )
                status = str(data_status.get("status", "unknown"))
                progress_msg = data_status.get("progress") or ""
                err_msg = data_status.get("error") or ""
                st.session_state["bulk_eval_job_status"] = status
                st.session_state["bulk_eval_job_progress"] = progress_msg
                st.session_state["bulk_eval_job_error"] = err_msg or None

                # 進捗バーと簡易アニメーション付きステータス表示
                status_lower = status.lower()

                done_count: int | None = None
                total_count: int | None = None
                if progress_msg:
                    try:
                        m = re.search(r"(\d+)\s*/\s*(\d+)", progress_msg)
                        if m:
                            done_val = int(m.group(1))
                            total_val = int(m.group(2))
                            if total_val > 0 and 0 <= done_val <= total_val:
                                done_count = done_val
                                total_count = total_val
                    except Exception:  # noqa: BLE001
                        done_count = None
                        total_count = None

                if total_count is not None and total_count > 0 and done_count is not None:
                    progress_ratio = max(0.0, min(1.0, done_count / total_count))
                else:
                    if status_lower in {"pending"}:
                        progress_ratio = 0.1
                    elif status_lower in {"running"}:
                        progress_ratio = 0.6
                    elif status_lower in {"completed"}:
                        progress_ratio = 1.0
                    elif status_lower in {"error"}:
                        progress_ratio = 0.0
                    else:
                        progress_ratio = 0.0

                tick_key = "bulk_eval_progress_tick"
                tick = int(st.session_state.get(tick_key, 0) or 0)
                st.session_state[tick_key] = tick + 1
                spinner_frames = ["-", "\\", "|", "/"]
                spinner = spinner_frames[tick % len(spinner_frames)]

                st.progress(progress_ratio)
                if total_count is not None and done_count is not None and total_count > 0:
                    st.caption(f"進捗: {done_count} / {total_count} 件完了")
                st.info(
                    f"[{spinner}] 一括評価ジョブ状態: {status}（ID: {bulk_job_id}）"
                )
                if progress_msg:
                    st.caption(f"進捗メッセージ: {progress_msg}")
                else:
                    if status_lower in {"pending"}:
                        st.caption(
                            "ジョブはキューに登録されています。しばらくお待ちください。"
                        )
                    elif status_lower in {"running"}:
                        st.caption("一括評価を実行中です。しばらくお待ちください。")

                # pending / running 中は他の重いHTTP呼び出しを避け、状態だけをポーリングする
                if status_lower in {"pending", "running"}:
                    if "bulk_eval_auto_refresh" not in st.session_state:
                        st.session_state["bulk_eval_auto_refresh"] = True

                    confirm_open = bool(
                        st.session_state.get("bulk_eval_cancel_confirm_open")
                    )

                    col_auto, col_manual, col_cancel = st.columns(3)
                    with col_auto:
                        auto_refresh = st.checkbox(
                            "2秒ごとに自動で進捗を更新する",
                            key="bulk_eval_auto_refresh",
                        )
                    with col_manual:
                        if st.button("進捗を手動更新", key="bulk_eval_manual_refresh"):
                            st.rerun()
                    with col_cancel:
                        if not confirm_open:
                            if st.button(
                                "この一括評価をキャンセル", key="bulk_eval_cancel"
                            ):
                                st.session_state["bulk_eval_cancel_confirm_open"] = True
                                confirm_open = True

                    # 協調キャンセルの説明と確認UI
                    if confirm_open:
                        st.warning(
                            "このキャンセルは\"協調キャンセル\"です。現在実行中の評価セットが安全に終了した後、残りの設定・残りのジョブを停止します。"\
                            " いま走っている処理自体を途中で強制終了することはできません。"
                        )
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button(
                                "はい、キャンセルします", key="bulk_eval_cancel_yes"
                            ):
                                try:
                                    resp_cancel = http_post(
                                        f"{BACKEND_URL}/bulk_job/cancel/{bulk_job_id}",
                                        json={},
                                    )
                                except Exception as e:  # noqa: BLE001
                                    st.error(
                                        f"キャンセル要求送信中にエラーが発生しました: {e}"
                                    )
                                else:
                                    if resp_cancel.status_code != 200:
                                        st.error(
                                            "キャンセル要求エラー: "
                                            f"{format_http_error(resp_cancel)}"
                                        )
                                    else:
                                        st.info(
                                            "キャンセル要求を送信しました。現在の評価セット終了後に停止されます。数秒後に状態が反映されます。"
                                        )
                                        st.session_state[
                                            "bulk_eval_cancel_confirm_open"
                                        ] = False
                                        time.sleep(0.5)
                                        st.rerun()
                        with col_no:
                            if st.button(
                                "いいえ、キャンセルしません", key="bulk_eval_cancel_no"
                            ):
                                st.session_state["bulk_eval_cancel_confirm_open"] = False
                                st.info("キャンセルは実行されませんでした。")

                    # 確認ダイアログが開いている間は自動更新を止める
                    if auto_refresh and not confirm_open:
                        time.sleep(2.0)
                        st.rerun()
                    return

                # completed / error / cancelled の場合はこのまま下の通常UIに進み、結果表示などを行う
                result_obj = data_status.get("result")
                if status in {"completed", "COMPLETED"} and result_obj is not None:
                    # 結果をフラット化してセッションに保存（既存ロジックと同様）
                    flat_results: List[Dict[str, Any]] = []
                    data = result_obj
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, list):
                                flat_results.extend(
                                    x for x in item if isinstance(x, dict)
                                )
                            elif isinstance(item, dict):
                                flat_results.append(item)
                    elif isinstance(data, dict):
                        flat_results.append(data)

                    valid_results: List[Dict[str, Any]] = [
                        r
                        for r in flat_results
                        if isinstance(r, dict) and not r.get("error")
                    ]
                    error_results: List[Dict[str, Any]] = [
                        r
                        for r in flat_results
                        if isinstance(r, dict) and r.get("error")
                    ]

                    if not valid_results:
                        st.error(
                            "一括評価がすべてエラーで終了しました。設定とバックエンドログを確認してください。"
                        )
                        if error_results:
                            with st.expander("エラー詳細", expanded=False):
                                st.json(error_results[:3])
                    else:
                        st.session_state.bulk_evaluation_results = valid_results
                        msg = (
                            f"一括評価が完了しました。有効な設定: {len(valid_results)} 件"
                        )
                        if error_results:
                            msg += f"（エラー: {len(error_results)} 件は除外）"
                        st.success(msg)
                    # ジョブIDをクリアして再実行できるようにする
                    st.session_state.pop("bulk_eval_job_id", None)
                elif status_lower in {"error"} and err_msg:
                    st.error(f"一括評価ジョブがエラーで終了しました: {err_msg}")
                    # エラー時もここで一旦ジョブIDをクリアしておく
                    st.session_state.pop("bulk_eval_job_id", None)
                elif status_lower in {"cancelled"}:
                    st.info("一括評価ジョブはユーザーによりキャンセルされました。")
                    st.session_state.pop("bulk_eval_job_id", None)

        # --- 評価対象PDFの選択（履歴から） ---
        st.markdown("### 評価対象PDFの選択")
        history_pdfs = _fetch_history_pdfs(BACKEND_URL.rstrip("/"))
        selected_pdf_id: str | None = None
        if not history_pdfs:
            st.info(
                "履歴にPDFがありません。PDFをアップロードしてから一括評価を実行してください。"
            )
        else:
            labels: List[str] = []
            id_list: List[str] = []
            current_file_id = (
                st.session_state.get("file_id")
                or st.session_state.get("rag_pdf_file_id")
            )
            default_idx = 0
            for idx, it in enumerate(history_pdfs):
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
                id_list.append(str(it.get("id")))
                if current_file_id and str(it.get("id")) == str(current_file_id):
                    default_idx = idx

            sel_idx = st.selectbox(
                "評価対象のPDF",
                options=list(range(len(labels))),
                format_func=lambda i: labels[i],
                index=default_idx,
                key="bulk_target_pdf_select",
            )
            selected_pdf_id = id_list[sel_idx]
            st.session_state["file_id"] = selected_pdf_id
            st.caption(f"選択中のPDF ID: {selected_pdf_id}")

        # 評価に使うテキスト・QAセットを決定（優先: 履歴で選択したPDF）
        eval_text = ""
        eval_questions: List[str] = []
        eval_answers: List[str] = []

        if selected_pdf_id:
            t, q_list, a_list = _load_qa_from_history(
                BACKEND_URL.rstrip("/"), selected_pdf_id
            )
            eval_text = t or ""
            eval_questions = q_list or []
            eval_answers = a_list or []

        # 履歴から取得できなかった場合は、従来どおり session_state の値にフォールバック
        if not eval_text or not eval_questions or not eval_answers:
            text_fallback = st.session_state.get("text") or ""
            questions_fallback: List[str] = (
                st.session_state.get("qa_questions") or []
            )
            answers_fallback: List[str] = (
                st.session_state.get("qa_answers") or []
            )
            if text_fallback and questions_fallback and answers_fallback:
                eval_text = eval_text or text_fallback
                eval_questions = eval_questions or questions_fallback
                eval_answers = eval_answers or answers_fallback

        # 質問・回答ペア数（max_pairs）はこの時点で一度計算しておき、
        # 後続のプリセットボタンやスライダー設定で利用する。
        total_q = len(eval_questions)
        total_a = len(eval_answers)
        max_pairs = min(total_q, total_a) if total_q and total_a else 0

        if not eval_text or not eval_questions or not eval_answers:
            st.info(
                "評価に使用するテキストまたはQAセットが見つかりません。\n"
                "対象PDFに抽出結果や生成QAがない可能性があります。PDFをアップロードし直すか、履歴タブで状態を確認してください。"
            )
        else:
            st.success(
                f"評価対象PDF ID: {selected_pdf_id or '（セッションのPDF）'} / "
                f"質問 {len(eval_questions)} 件 / 回答 {len(eval_answers)} 件"
            )

            history_experiments_for_pdf: List[Dict[str, Any]] = []
            if selected_pdf_id:
                try:
                    resp_hist = http_get(f"{BACKEND_URL.rstrip('/')}/history/experiments")
                    if resp_hist.status_code == 200 and resp_hist.headers.get("Content-Type", "").startswith(
                        "application/json"
                    ):
                        data_hist = resp_hist.json() or {}
                        experiments_all = data_hist.get("items", data_hist.get("experiments", [])) or []
                        for exp in experiments_all:
                            if str(exp.get("pdf_file_id")) == str(selected_pdf_id):
                                history_experiments_for_pdf.append(exp)
                except Exception as e:  # noqa: BLE001
                    st.warning(f"このPDFに紐づく評価履歴の取得中にエラーが発生しました: {e}")

            if history_experiments_for_pdf:
                st.markdown("### 過去の実験からチャンク設定を復元")
                exp_indices = list(range(len(history_experiments_for_pdf)))

                def _format_exp_for_chunk(i: int) -> str:
                    item = history_experiments_for_pdf[i]
                    exp_id = item.get("id")
                    name = item.get("experiment_name") or ""
                    status = item.get("status") or ""
                    created_at = item.get("created_at") or ""
                    return f"ID:{exp_id} {name} [{status}] {created_at}"

                selected_exp_idx = st.selectbox(
                    "このPDFに対する過去の実験",
                    options=exp_indices,
                    format_func=_format_exp_for_chunk,
                    key="bulk_history_experiment_for_chunk",
                )
                selected_exp_item = history_experiments_for_pdf[selected_exp_idx]
                selected_exp_id_for_chunk = selected_exp_item.get("id")
                history_results_for_pdf: List[Dict[str, Any]] = []

                if selected_exp_id_for_chunk is not None:
                    try:
                        resp_res = http_get(
                            f"{BACKEND_URL.rstrip('/')}/history/experiments/{selected_exp_id_for_chunk}/results"
                        )
                        if resp_res.status_code == 200 and resp_res.headers.get("Content-Type", "").startswith(
                            "application/json"
                        ):
                            data_res = resp_res.json() or {}
                            history_results_for_pdf = data_res.get("items", data_res.get("results", [])) or []
                    except Exception as e:  # noqa: BLE001
                        st.warning(f"実験結果の取得中にエラーが発生しました: {e}")

                if history_results_for_pdf:
                    result_indices = list(range(len(history_results_for_pdf)))
                    result_labels: List[str] = []
                    for idx_r in result_indices:
                        r = history_results_for_pdf[idx_r]
                        res_id = r.get("id")
                        emb_name = r.get("embedding_model", "unknown")
                        chunk_name = r.get("chunk_strategy", r.get("chunk_method", "unknown"))
                        label = (
                            f"ID:{res_id} {emb_name} / "
                            f"{chunk_name} / "
                            f"size={r.get('chunk_size', '-')}, overlap={r.get('chunk_overlap', '-')}"
                        )
                        result_labels.append(label)

                    selected_result_idx = st.selectbox(
                        "復元するEmbedding+チャンク設定",
                        options=result_indices,
                        format_func=lambda i: result_labels[i],
                        key="bulk_history_result_for_chunk",
                    )
                    if st.button(
                        "この履歴の設定をチャンク設定に反映",
                        key="bulk_apply_chunk_from_history",
                    ):
                        apply_bulk_chunk_settings_from_history(history_results_for_pdf[selected_result_idx])
                        st.success("履歴の設定をチャンク設定に反映しました。下の『チャンク設定』セクションを確認してください。")

        st.markdown("---")
        st.subheader("評価パラメータの設定")
        st.caption(
            "※ 質問数や組み合わせ（Embedding×チャンク設定）が多いほど処理時間が長くなります。"
            " まずは少ない質問・少ない組み合わせで試すことをおすすめします。"
        )
        st.caption(
            "※ RAGAS は1つの質問に対して複数の評価指標を同時に計算するため、バックエンドの進捗表示では"
            "『質問数×指標数』件分の処理としてカウントされます（例: 質問5件・指標5種類 → 25件）。"
        )

        # よく使う設定をワンクリックで適用できるプリセット
        col_preset_light, col_preset_full = st.columns(2)
        with col_preset_light:
            if st.button("🔧 軽量プリセットを適用", key="bulk_preset_light"):
                # 少数の質問・1つのEmbedding・1パターンのチャンク設定で素早く傾向を確認したい場合の推奨設定
                st.session_state["bulk_chunk_methods"] = ["recursive"]
                st.session_state["bulk_chunk_sizes_select"] = [1000]
                st.session_state["bulk_chunk_overlaps_select"] = [0]
                st.session_state["bulk_eval_question_mode"] = "head_n"
                # 質問数が取得できていれば5件、なければデフォルト10件
                questions_available = max_pairs if max_pairs > 0 else 10
                st.session_state["bulk_eval_question_count"] = min(5, questions_available)
                st.session_state["bulk_include_answer_similarity"] = False
                st.success("軽量プリセットを適用しました。下の設定内容を確認してください。")
        with col_preset_full:
            if st.button("📊 本番向けプリセットを適用", key="bulk_preset_full"):
                # 本番比較向け: 複数チャンク戦略・サイズ／オーバーラップ＋全件評価
                st.session_state["bulk_chunk_methods"] = ["recursive", "sentence"]
                st.session_state["bulk_chunk_sizes_select"] = [512, 1000]
                st.session_state["bulk_chunk_overlaps_select"] = [0, 200]
                st.session_state["bulk_eval_question_mode"] = "all"
                st.session_state["bulk_include_answer_similarity"] = True
                st.success("本番向けプリセットを適用しました。下の設定内容を確認してください。")

        total_q = len(eval_questions)
        total_a = len(eval_answers)
        max_pairs = min(total_q, total_a) if total_q and total_a else 0

        question_mode = st.radio(
            "評価に使用する質問の範囲",
            options=["all", "head_n"],
            format_func=lambda v: "すべての質問を使用" if v == "all" else "先頭 N 問のみ使用",
            index=0,
            key="bulk_eval_question_mode",
            help="全件評価は精度が高いですが時間がかかります。まずは先頭N問のみでお試しすることをおすすめします。",
        )

        if question_mode == "head_n" and max_pairs > 0:
            default_n = min(10, max_pairs)
            st.slider(
                "使用する質問数",
                min_value=1,
                max_value=max_pairs,
                value=default_n,
                step=1,
                key="bulk_eval_question_count",
                help="大きい値にするほど評価時間が長くなります。",
            )

        # RAG回答生成および評価に使用するLLMの選択
        llm_models = _fetch_llm_models(BACKEND_URL.rstrip("/"))
        selected_llm_models: List[str] = []
        selected_eval_llm: str | None = None
        force_llm_generation = st.session_state.get("bulk_force_llm_generation", False)
        if llm_models:
            provider_labels = {"huggingface": "HuggingFace", "ollama": "Ollama", "openai": "OpenAI"}
            llm_names = [m.get("name", "") for m in llm_models if m.get("name")]
            llm_labels = {}
            for m in llm_models:
                name = m.get("name")
                if not name:
                    continue
                t = m.get("type") or "unknown"
                prefix = provider_labels.get(t, t)
                base_label = m.get("display_name") or name
                llm_labels[name] = f"[{prefix}] {base_label}"
            default_llm_name = st.session_state.get("llm_model", "")
            default_llm_choices: list[str] = []
            prev_selected_llms = [
                name for name in st.session_state.get("bulk_selected_llm_models", []) if name in llm_names
            ]
            if prev_selected_llms:
                default_llm_choices = prev_selected_llms
            elif default_llm_name in llm_names:
                default_llm_choices = [default_llm_name]
            elif llm_names:
                default_llm_choices = [llm_names[0]]

            # Streamlitのマルチセレクト値を保持するため、ウィジェット作成前にstateを初期化
            if "bulk_llm_model_select" not in st.session_state:
                st.session_state["bulk_llm_model_select"] = list(default_llm_choices)
            else:
                current_selection = [
                    name for name in st.session_state.get("bulk_llm_model_select", []) if name in llm_names
                ]
                if not current_selection and default_llm_choices:
                    current_selection = list(default_llm_choices)
                st.session_state["bulk_llm_model_select"] = current_selection

            selected_llm_models = st.multiselect(
                "RAG回答生成に使用するLLM（複数選択可）",
                options=llm_names,
                format_func=lambda name: llm_labels.get(name, name),
                key="bulk_llm_model_select",
                help="選択したLLMすべてでRAG回答生成を行い、埋め込み・チャンク設定と組み合わせて評価します。",
            )
            if selected_llm_models:
                st.session_state["bulk_selected_llm_models"] = selected_llm_models
                # グローバルなLLMモデルとして反映（先頭を代表として利用）
                st.session_state.llm_model = selected_llm_models[0]
                # chat_modelが未設定なら同期
                if not st.session_state.get("chat_model"):
                    st.session_state.chat_model = selected_llm_models[0]

            default_eval_llm = (
                st.session_state.get("bulk_selected_evaluation_llm")
                or st.session_state.get("llm_model")
                or (selected_llm_models[0] if selected_llm_models else (llm_names[0] if llm_names else ""))
            )
            if default_eval_llm not in llm_names and llm_names:
                default_eval_llm = llm_names[0]
            if "bulk_eval_llm_select" not in st.session_state or st.session_state["bulk_eval_llm_select"] not in llm_names:
                st.session_state["bulk_eval_llm_select"] = default_eval_llm
            eval_select_idx = llm_names.index(st.session_state["bulk_eval_llm_select"]) if llm_names else 0
            selected_eval_llm = None
            if llm_names:
                selected_eval_llm = st.selectbox(
                    "RAGAS評価LLM（採点者）",
                    options=llm_names,
                    format_func=lambda name: llm_labels.get(name, name),
                    index=eval_select_idx,
                    key="bulk_eval_llm_select",
                    help="RAGASの採点で使用するLLMです。評価結果に一貫性を持たせたい場合は固定してください。",
                )
                if selected_eval_llm:
                    st.session_state["bulk_selected_evaluation_llm"] = selected_eval_llm

            force_llm_generation = st.checkbox(
                "LLMごとに回答を再生成する（既存回答を使い回さない）",
                value=force_llm_generation,
                key="bulk_force_llm_generation",
                help="オンにすると、LLMごとに回答を再生成して評価します。処理時間は長くなりますが、LLM間の違いを比較できます。",
            )

        # Embeddingモデルの選択（複数選択）
        embedding_models = _fetch_embedding_models(BACKEND_URL.rstrip("/"))
        if embedding_models:
            provider_labels = {"huggingface": "HuggingFace", "ollama": "Ollama", "openai": "OpenAI"}
            providers = [m.get("type", "unknown") for m in embedding_models]
            unique_providers = []
            for p in providers:
                if p not in unique_providers:
                    unique_providers.append(p)
            provider_options = ["すべて"] + [
                provider_labels.get(p, p) for p in unique_providers if p
            ]
            selected_provider_label = st.selectbox(
                "Embeddingプロバイダ",
                provider_options,
                index=0,
                key="bulk_embedding_provider_filter",
            )
            selected_type = None
            if selected_provider_label != "すべて":
                for k, v in provider_labels.items():
                    if v == selected_provider_label:
                        selected_type = k
                        break
            if selected_type:
                filtered_models = [
                    m for m in embedding_models if m.get("type") == selected_type
                ]
            else:
                filtered_models = embedding_models
            if not filtered_models:
                st.info("選択されたプロバイダに対応するEmbeddingモデルがありません。")
                filtered_models = embedding_models
            embedding_names = [m.get("name", "") for m in filtered_models]
            embedding_labels = []
            for m in filtered_models:
                t = m.get("type") or "unknown"
                prefix = provider_labels.get(t, t)
                base_label = m.get("display_name") or m.get("name") or "unknown"
                embedding_labels.append(f"[{prefix}] {base_label}")
            default_name = st.session_state.get("embedding_model")
            default_indices = []
            if default_name in embedding_names:
                default_indices = [embedding_names.index(default_name)]
            selected_indices = st.multiselect(
                "埋め込みモデル（複数選択可）",
                options=list(range(len(embedding_names))),
                format_func=lambda i: embedding_labels[i],
                default=default_indices or [0],
                key="bulk_embedding_models_select",
            )
            selected_embeddings = [
                embedding_names[i] for i in selected_indices if 0 <= i < len(embedding_names)
            ]
            # 先頭のEmbeddingを「共通Embedding」として扱い、グローバルに反映
            if selected_embeddings:
                primary_emb = selected_embeddings[0]
                st.session_state.embedding_model = primary_emb
                # LLMモデルが決まっていればDBにも保存
                base_llm_for_persist = (
                    selected_llm_models[0] if selected_llm_models else st.session_state.get("llm_model", "gpt-oss")
                )
                if base_llm_for_persist:
                    _persist_model_selection(BACKEND_URL.rstrip("/"), base_llm_for_persist, primary_emb)
        else:
            # バックエンドから取得できなかった場合のフォールバック
            options = [
                "huggingface_bge_small",
                "huggingface_bge_large",
                "text-embedding-3-small",
                "text-embedding-3-large",
                "mxbai-embed-large",
            ]
            selected_embeddings = st.multiselect(
                "埋め込みモデル（フォールバック・複数選択可）",
                options=options,
                default=[options[0]],
                key="bulk_embedding_models_fallback",
            )

        # チャンク方式・サイズ・オーバーラップの設定（複数組み合わせ）
        st.markdown("### チャンク設定（複数組み合わせ）")
        col_method, col_size, col_overlap = st.columns([1, 1, 1])
        with col_method:
            selected_chunk_methods = st.multiselect(
                "チャンク方式（複数選択可）",
                options=["recursive", "fixed", "sentence", "paragraph", "semantic"],
                default=["recursive"],
                key="bulk_chunk_methods",
                help=(
                    "semantic: 意味的まとまりで分割（サイズ・オーバーラップは無視） / "
                    "recursive: 再帰的文字分割 / sentence: 文単位 / paragraph: 段落単位 / fixed: 固定長"
                ),
            )

        methods_require_size = {"recursive", "fixed"}
        uses_size_overlap = any(m in methods_require_size for m in selected_chunk_methods)
        only_non_size_methods = bool(selected_chunk_methods) and all(
            m not in methods_require_size for m in selected_chunk_methods
        )

        # よく使うチャンクサイズ候補
        default_size_candidates = [256, 512, 768, 1000, 1500, 2000]
        with col_size:
            chunk_sizes = st.multiselect(
                "チャンクサイズ候補（複数選択可）",
                options=default_size_candidates,
                default=[1000],
                key="bulk_chunk_sizes_select",
                help="recursive / fixed にのみ適用されます。",
                disabled=not uses_size_overlap,
            )
        # よく使うオーバーラップ候補
        default_overlap_candidates = [0, 100, 200, 300, 400]
        with col_overlap:
            chunk_overlaps = st.multiselect(
                "オーバーラップ候補（複数選択可）",
                options=default_overlap_candidates,
                default=[0, 200],
                key="bulk_chunk_overlaps_select",
                help="recursive / fixed にのみ適用されます。",
                disabled=not uses_size_overlap,
            )

        if uses_size_overlap:
            if not chunk_sizes:
                chunk_sizes = [1000]
            if not chunk_overlaps:
                chunk_overlaps = [200]

        if only_non_size_methods:
            st.info(
                "選択されているチャンク方式（semantic / sentence / paragraph）では、チャンクサイズとオーバーラップは使用されません。"
            )
        elif "semantic" in selected_chunk_methods and uses_size_overlap:
            st.caption(
                "semantic にはチャンクサイズ・オーバーラップは適用されません（recursive / fixed のみに適用）。"
            )

        similarity_threshold = 0.7
        if "semantic" in selected_chunk_methods:
            similarity_threshold = st.slider(
                "semantic 用 類似度しきい値",
                min_value=0.1,
                max_value=0.95,
                value=0.7,
                step=0.05,
                key="bulk_semantic_similarity_threshold",
                help="値が高いほど意味的に近い文同士を強くまとめます。",
            )

        include_answer_similarity = st.checkbox(
            "answer_similarity 指標も計算する (重め)",
            value=False,
            key="bulk_include_answer_similarity",
            help="オンにすると質問ごとの生成回答と正解の類似度を追加で計算します。精度向上に役立ちますが計算コストが高くなります。",
        )

        job_limit = int(
            st.number_input(
                "一括評価で許可する最大ジョブ数の上限",
                min_value=1,
                max_value=100,
                value=int(st.session_state.get("bulk_eval_job_limit", 10) or 10),
                step=1,
                key="bulk_eval_job_limit",
                help=(
                    "一度に実行する評価ジョブ数の上限です。ジョブ数が多いほど処理時間が長くなり固まりやすくなります。"
                    "例: 10 件程度までに抑えることを推奨します。"
                ),
            )
        )

        # 現在の設定から、おおよそのジョブ数を算出してユーザーに提示
        estimated_jobs = 0
        if selected_llm_models and selected_embeddings and selected_chunk_methods:
            num_llm = len(selected_llm_models)
            num_embeddings = len(selected_embeddings)
            num_sizes = max(len(chunk_sizes), 1)
            num_overlaps = max(len(chunk_overlaps), 1)

            for method in selected_chunk_methods:
                if method == "semantic":
                    # semantic はサイズ・オーバーラップを持たない1組み合わせ扱い
                    estimated_jobs += num_llm * num_embeddings
                elif method in ("sentence", "paragraph"):
                    estimated_jobs += num_llm * num_embeddings
                elif method in ("recursive", "fixed"):
                    estimated_jobs += num_llm * num_embeddings * num_sizes * num_overlaps
                else:
                    estimated_jobs += num_llm * num_embeddings

        if estimated_jobs > 0:
            st.caption(
                f"現在の設定での評価ジョブ数（概算）: {estimated_jobs} 件（上限: {job_limit} 件）"
            )

            # ごく大まかな処理時間の目安を表示（1ジョブあたり 10〜30 秒程度と仮定）
            approx_min_sec = estimated_jobs * 10
            approx_max_sec = estimated_jobs * 30
            if approx_max_sec < 60:
                time_hint = f"約 {approx_min_sec}〜{approx_max_sec} 秒程度"
            else:
                min_min = approx_min_sec // 60
                min_max = max(approx_max_sec // 60, min_min + 1)
                time_hint = f"約 {min_min}〜{min_max} 分程度"
            st.caption(
                f"処理時間の目安: {time_hint}（モデルや環境により前後します。まずは軽量プリセットでお試しください。）"
            )

            if estimated_jobs > job_limit:
                st.warning(
                    "現在の設定では評価ジョブ数が上限を超えています。このままでは実行できません。"
                    "Embeddingモデルやチャンク設定を減らすか、上記の『一括評価で許可する最大ジョブ数の上限』を引き上げてください。"
                )

        st.markdown("---")
        st.subheader("一括評価の実行")

        # 既存ジョブがある場合は古い確認状態をクリア
        if st.session_state.get("bulk_eval_job_id"):
            st.session_state.pop("bulk_eval_confirm_payload", None)
            st.session_state.pop("bulk_eval_confirm_summary", None)

        if st.button("この設定で一括評価を実行", key="run_bulk_evaluate"):
            # まずは現在の設定からジョブとサマリー情報だけを生成し、実際の実行は確認後に行う
            if not eval_text or not eval_questions or not eval_answers:
                st.error(
                    "評価対象のテキストまたはQ&Aが見つかりません。PDFアップロードとQA自動生成、"
                    "もしくは『履歴』タブからの復元を行ってから再試行してください。"
                )
            else:
                # 質問数と回答数が揃っていない場合は短い方に揃える
                total_q = len(eval_questions)
                total_a = len(eval_answers)
                max_pairs = min(total_q, total_a)
                if max_pairs == 0:
                    st.error(
                        "評価対象のテキストまたはQ&Aが見つかりません。PDFアップロードとQA自動生成、"
                        "もしくは『履歴』タブからの復元を行ってから再試行してください。"
                    )
                    return
                if total_q != total_a:
                    st.warning(
                        f"質問数({total_q})と回答数({total_a})が一致しません。"
                        f"評価には先頭 {max_pairs} 件のみを使用します。"
                    )

                question_mode = st.session_state.get("bulk_eval_question_mode", "all")
                n_limit = st.session_state.get("bulk_eval_question_count")

                if question_mode == "head_n" and isinstance(n_limit, int):
                    use_n = min(max(n_limit, 1), max_pairs)
                else:
                    use_n = max_pairs

                questions_eval = eval_questions[:use_n]
                answers_eval = eval_answers[:use_n]

                if not selected_embeddings:
                    st.error("少なくとも1つの埋め込みモデルを選択してください。")
                    return
                if not selected_llm_models:
                    st.error("少なくとも1つのLLMモデルを選択してください。")
                    return
                if not selected_chunk_methods:
                    st.error("少なくとも1つのチャンク方式を選択してください。")
                    return

                # Embedding × チャンク方式 × サイズ × オーバーラップ の組み合わせでジョブを生成
                jobs: List[Dict[str, Any]] = []
                file_id = st.session_state.get("file_id")

                eval_llm_for_job = selected_eval_llm or (
                    selected_llm_models[0] if selected_llm_models else None
                )
                for llm_model in selected_llm_models:
                    for emb in selected_embeddings:
                        for method in selected_chunk_methods:
                            if method == "semantic":
                                job: Dict[str, Any] = {
                                    "llm_model": llm_model,
                                    "evaluation_llm_model": eval_llm_for_job,
                                    "embedding_model": emb,
                                    "chunk_methods": [method],
                                    "text": eval_text,
                                    "questions": questions_eval,
                                    "answers": answers_eval,
                                    "include_answer_similarity": include_answer_similarity,
                                    "force_llm_generation": force_llm_generation,
                                    "semantic_params": {
                                        "similarity_threshold": float(similarity_threshold)
                                    },
                                }
                                if file_id:
                                    job["file_id"] = file_id
                                jobs.append(job)
                            elif method in ("sentence", "paragraph"):
                                job = {
                                    "llm_model": llm_model,
                                    "evaluation_llm_model": eval_llm_for_job,
                                    "embedding_model": emb,
                                    "chunk_methods": [method],
                                    "text": eval_text,
                                    "questions": questions_eval,
                                    "answers": answers_eval,
                                    "include_answer_similarity": include_answer_similarity,
                                    "force_llm_generation": force_llm_generation,
                                }
                                if file_id:
                                    job["file_id"] = file_id
                                jobs.append(job)
                            else:
                                for size in chunk_sizes:
                                    for ov in chunk_overlaps:
                                        job = {
                                            "llm_model": llm_model,
                                            "evaluation_llm_model": eval_llm_for_job,
                                            "embedding_model": emb,
                                            "chunk_methods": [method],
                                            "chunk_sizes": [int(size)],
                                            "chunk_overlaps": [int(ov)],
                                            "text": eval_text,
                                            "questions": questions_eval,
                                            "answers": answers_eval,
                                            "include_answer_similarity": include_answer_similarity,
                                            "force_llm_generation": force_llm_generation,
                                        }
                                        if file_id:
                                            job["file_id"] = file_id
                                        jobs.append(job)

                if not jobs:
                    st.error("有効な評価ジョブが生成できませんでした。設定を見直してください。")
                    return
                # ジョブ数が多すぎる場合は、設定された上限に基づいて実行をブロック
                job_count = len(jobs)
                job_limit = int(st.session_state.get("bulk_eval_job_limit", 10) or 10)
                if job_count > job_limit:
                    st.error(
                        f"評価ジョブが {job_count} 件あります。現在の上限 {job_limit} 件を超えているため実行できません。"
                        "チャンク設定やEmbeddingモデルの数を減らすか、『一括評価で許可する最大ジョブ数の上限』を引き上げてください。"
                    )
                    return

                payload: Any
                if job_count == 1:
                    payload = jobs[0]
                else:
                    payload = jobs

                # 実行前確認用にサマリー情報をセッションに保存しておき、別ボタンで実際のジョブ起動を行う
                summary_llm_models = selected_llm_models or [st.session_state.get("llm_model", "gpt-oss")]
                summary = {
                    "pdf_file_id": selected_pdf_id or file_id,
                    "question_mode": question_mode,
                    "use_n": use_n,
                    "total_pairs": max_pairs,
                    "llm_models": summary_llm_models,
                    "evaluation_llm_model": selected_eval_llm or summary_llm_models[0],
                    "force_llm_generation": bool(force_llm_generation),
                    "embedding_models": list(selected_embeddings),
                    "chunk_methods": list(selected_chunk_methods),
                    "chunk_sizes": list(chunk_sizes),
                    "chunk_overlaps": list(chunk_overlaps),
                    "include_answer_similarity": bool(include_answer_similarity),
                    "similarity_threshold": float(similarity_threshold) if "semantic" in selected_chunk_methods else None,
                    "estimated_jobs": job_count,
                }
                st.session_state["bulk_eval_confirm_payload"] = payload
                st.session_state["bulk_eval_confirm_summary"] = summary

        # 実行前の確認セクション（簡易モーダル的なUI）
        confirm_payload: Any = st.session_state.get("bulk_eval_confirm_payload")
        confirm_summary: Dict[str, Any] | None = st.session_state.get("bulk_eval_confirm_summary")
        if confirm_payload is not None and isinstance(confirm_summary, dict):
            st.markdown("#### 実行前の設定確認")
            st.info("以下の設定で一括評価を実行します。内容を確認し、問題なければ『はい、この設定で実行』を押してください。")

            # 設定内容の一覧表示
            lines: List[str] = []
            pdf_label = confirm_summary.get("pdf_file_id") or "-"
            lines.append(f"- 対象PDF ID: **{pdf_label}**")
            q_mode_label = "すべての質問" if confirm_summary.get("question_mode") == "all" else "先頭N問のみ"
            lines.append(
                f"- 質問の使用範囲: **{q_mode_label}** / 使用質問数: **{confirm_summary.get('use_n')}** / 総ペア数: {confirm_summary.get('total_pairs')}"
            )
            llm_list = confirm_summary.get("llm_models") or []
            lines.append(f"- 使用LLMモデル: **{', '.join(map(str, llm_list)) or '-'}**")
            emb_list = confirm_summary.get("embedding_models") or []
            lines.append(f"- 使用Embeddingモデル: **{', '.join(map(str, emb_list)) or '-'}**")
            methods_list = confirm_summary.get("chunk_methods") or []
            lines.append(f"- チャンク方式: **{', '.join(map(str, methods_list)) or '-'}**")
            size_list = confirm_summary.get("chunk_sizes") or []
            overlap_list = confirm_summary.get("chunk_overlaps") or []
            if size_list:
                lines.append(f"- チャンクサイズ候補: {', '.join(map(str, size_list))}")
            if overlap_list:
                lines.append(f"- オーバーラップ候補: {', '.join(map(str, overlap_list))}")
            if "semantic" in (confirm_summary.get("chunk_methods") or []):
                sim_th = confirm_summary.get("similarity_threshold")
                if sim_th is not None:
                    lines.append(f"- semantic 類似度しきい値: {sim_th}")
            inc_sim = bool(confirm_summary.get("include_answer_similarity"))
            lines.append(f"- answer_similarity の計算: {'有効' if inc_sim else '無効'}")
            est_jobs = confirm_summary.get("estimated_jobs")
            if est_jobs is not None:
                lines.append(f"- 推定ジョブ数: **{est_jobs} 件**")

            st.markdown("\n".join(lines))

            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.button("はい、この設定で実行", key="bulk_eval_confirm_yes"):
                    # 一括評価はバックエンドのジョブAPI経由で非同期実行する
                    with st.spinner("一括評価ジョブを起動中です…"):
                        try:
                            resp = http_post(f"{BACKEND_URL}/bulk_job/start", json=confirm_payload)
                        except Exception as e:  # noqa: BLE001
                            st.error(f"一括評価ジョブ起動時にエラーが発生しました: {e}")
                        else:
                            if resp.status_code != 200:
                                st.error(
                                    f"一括評価ジョブ起動エラー: {format_http_error(resp)}"
                                )
                            else:
                                data = (
                                    resp.json()
                                    if resp.headers.get("Content-Type", "").startswith("application/json")
                                    else {}
                                )
                                job_id = data.get("job_id")
                                if not job_id:
                                    st.error("一括評価ジョブIDの取得に失敗しました。")
                                else:
                                    # 確認用の一時状態はここでクリアする
                                    st.session_state.pop("bulk_eval_confirm_payload", None)
                                    st.session_state.pop("bulk_eval_confirm_summary", None)
                                    st.session_state["bulk_eval_job_id"] = job_id
                                    st.session_state["bulk_eval_job_status"] = "pending"
                                    st.session_state["bulk_eval_job_progress"] = "ジョブを受け付けました。"
                                    st.session_state["bulk_eval_job_error"] = None
                                    st.info(f"一括評価ジョブを起動しました。ジョブID: {job_id}")
                                    # ジョブ起動直後に進捗ポーリング用のステータス表示に切り替えるため、即座に再実行する
                                    st.rerun()

            with col_cancel:
                if st.button("キャンセル", key="bulk_eval_confirm_cancel"):
                    st.session_state.pop("bulk_eval_confirm_payload", None)
                    st.session_state.pop("bulk_eval_confirm_summary", None)
                    st.info("一括評価の実行をキャンセルしました。")

        # --- 結果の表示（セッションに保存されたものを使用） ---
        results: List[Dict[str, Any]] = (
            st.session_state.get("bulk_evaluation_results") or []
        )
        if results:
            st.markdown("---")
            st.subheader("一括評価結果")

            results_df = pd.DataFrame(results)
            with st.expander("評価結果テーブル（デバッグ用）", expanded=False):
                st.dataframe(results_df, use_container_width=True)

            st.markdown("### 評価グラフ")
            _render_bulk_style_charts(results_df, key_prefix="bulk_eval")

            st.markdown("---")
            st.subheader("グラフのエクスポート")
            if st.button(
                "📥 すべてのグラフをZIPファイルでダウンロード",
                key="download_bulk_eval_graphs",
            ):
                with st.spinner("グラフを生成してZIPファイルを作成中です..."):
                    zip_bytes = create_zip_with_graphs(
                        results, filename="rag_bulk_evaluation_graphs"
                    )
                if zip_bytes:
                    b64 = base64.b64encode(zip_bytes).decode("utf-8")
                    href = (
                        f'<a href="data:application/zip;base64,{b64}" '
                        f'download="rag_bulk_evaluation_graphs.zip">'
                        f"ZIPファイルをダウンロード</a>"
                    )
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("グラフのZIPファイルを生成しました。")
                else:
                    st.error("グラフZIPの生成に失敗しました。もう一度お試しください。")
