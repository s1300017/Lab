from __future__ import annotations

from typing import Any, Dict, List

import base64
import pandas as pd
import streamlit as st

from http_client import http_get, http_post, format_http_error
from evaluation_history_ui import _render_bulk_style_charts, apply_bulk_chunk_settings_from_history
from graph_utils import create_zip_with_graphs


def _fetch_embedding_models(BACKEND_URL: str) -> List[Dict[str, str]]:
    """バックエンドの /list_models からEmbeddingモデル一覧を取得するヘルパー。"""
    try:
        resp = http_get(f"{BACKEND_URL}/list_models")
        resp.raise_for_status()
        data = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
        return data.get("Embedding", []) or []
    except Exception as e:  # noqa: BLE001
        st.warning(f"Embeddingモデル一覧の取得に失敗しました: {e}")
        return []


def _fetch_llm_models(BACKEND_URL: str) -> List[Dict[str, str]]:
    try:
        resp = http_get(f"{BACKEND_URL}/list_models")
        resp.raise_for_status()
        data = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
        return data.get("LLM", []) or []
    except Exception as e:  # noqa: BLE001
        st.warning(f"LLMモデル一覧の取得に失敗しました: {e}")
        return []


def _persist_model_selection(BACKEND_URL: str, llm_model: str, embedding_model: str) -> None:
    """選択されたLLM/Embeddingモデルをバックエンドに保存する。"""
    payload = {
        "llm_model": llm_model,
        "embedding_model": embedding_model,
    }
    try:
        http_post(f"{BACKEND_URL}/config/model_selection", json=payload)
    except Exception as e:  # noqa: BLE001
        st.warning(f"モデル選択の保存に失敗しました: {e}")


def _fetch_history_pdfs(BACKEND_URL: str) -> List[Dict[str, Any]]:
    """/history/pdf-files からPDF一覧を取得するヘルパー。"""
    try:
        resp = http_get(f"{BACKEND_URL}/history/pdf-files")
        if resp.status_code != 200:
            st.warning(
                f"PDF履歴の取得に失敗しました: {resp.status_code} {resp.text}"
            )
            return []
        data = (
            resp.json()
            if resp.headers.get("Content-Type", "").startswith("application/json")
            else {}
        )
        return data.get("items", []) or []
    except Exception as e:  # noqa: BLE001
        st.warning(f"PDF履歴の取得中にエラーが発生しました: {e}")
        return []


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
                        label = (
                            f"{r.get('embedding_model', 'unknown')} / "
                            f"{r.get('chunk_strategy', r.get('chunk_method', 'unknown'))} / "
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

        total_q = len(eval_questions)
        total_a = len(eval_answers)
        max_pairs = min(total_q, total_a) if total_q and total_a else 0

        question_mode = st.radio(
            "評価に使用する質問の範囲",
            options=["all", "head_n"],
            format_func=lambda v: "すべての質問を使用" if v == "all" else "先頭 N 問のみ使用",
            index=0,
            key="bulk_eval_question_mode",
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
            )

        # RAG回答生成に使用するLLMの選択
        llm_models = _fetch_llm_models(BACKEND_URL.rstrip("/"))
        selected_llm_model: str | None = None
        if llm_models:
            provider_labels = {"huggingface": "HuggingFace", "ollama": "Ollama", "openai": "OpenAI"}
            llm_names = [m.get("name", "") for m in llm_models]
            llm_labels: List[str] = []
            for m in llm_models:
                t = m.get("type") or "unknown"
                prefix = provider_labels.get(t, t)
                base_label = m.get("display_name") or m.get("name") or "unknown"
                llm_labels.append(f"[{prefix}] {base_label}")
            default_llm_name = st.session_state.get("llm_model") or "gpt-oss"
            if default_llm_name in llm_names:
                default_llm_index = llm_names.index(default_llm_name)
            else:
                default_llm_index = 0
            selected_llm_index = st.selectbox(
                "RAG回答生成に使用するLLM",
                options=list(range(len(llm_names))),
                format_func=lambda i: llm_labels[i],
                index=default_llm_index,
                key="bulk_llm_model_select",
            )
            if 0 <= selected_llm_index < len(llm_names):
                selected_llm_model = llm_names[selected_llm_index]
                # グローバルなLLMモデルとして反映
                st.session_state.llm_model = selected_llm_model
                # chat_modelが未設定なら同期
                if not st.session_state.get("chat_model"):
                    st.session_state.chat_model = selected_llm_model

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
                base_llm_for_persist = selected_llm_model or st.session_state.get("llm_model", "gpt-oss")
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
        # よく使うチャンクサイズ候補
        default_size_candidates = [256, 512, 768, 1000, 1500, 2000]
        with col_size:
            chunk_sizes = st.multiselect(
                "チャンクサイズ候補（複数選択可）",
                options=default_size_candidates,
                default=[1000],
                key="bulk_chunk_sizes_select",
                help="評価したいチャンクサイズを複数選択できます。",
            )
        # よく使うオーバーラップ候補
        default_overlap_candidates = [0, 100, 200, 300, 400]
        with col_overlap:
            chunk_overlaps = st.multiselect(
                "オーバーラップ候補（複数選択可）",
                options=default_overlap_candidates,
                default=[0, 200],
                key="bulk_chunk_overlaps_select",
                help="評価したいオーバーラップサイズを複数選択できます。",
            )

        # どちらかが空の場合はデフォルト値を補完
        if not chunk_sizes:
            chunk_sizes = [1000]
        if not chunk_overlaps:
            chunk_overlaps = [200]

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
            value=True,
            key="bulk_include_answer_similarity",
        )

        st.markdown("---")
        st.subheader("一括評価の実行")

        if st.button("この設定で一括評価を実行", key="run_bulk_evaluate"):
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
                if not selected_chunk_methods:
                    st.error("少なくとも1つのチャンク方式を選択してください。")
                    return

                # Embedding × チャンク方式 × サイズ × オーバーラップ の組み合わせでジョブを生成
                jobs: List[Dict[str, Any]] = []
                file_id = st.session_state.get("file_id")

                for emb in selected_embeddings:
                    for method in selected_chunk_methods:
                        if method == "semantic":
                            job: Dict[str, Any] = {
                                "embedding_model": emb,
                                "chunk_methods": [method],
                                "text": eval_text,
                                "questions": questions_eval,
                                "answers": answers_eval,
                                "include_answer_similarity": include_answer_similarity,
                                "semantic_params": {
                                    "similarity_threshold": float(similarity_threshold)
                                },
                            }
                            if file_id:
                                job["file_id"] = file_id
                            if selected_llm_model:
                                job["llm_model"] = selected_llm_model
                            jobs.append(job)
                        else:
                            for size in chunk_sizes:
                                for ov in chunk_overlaps:
                                    job = {
                                        "embedding_model": emb,
                                        "chunk_methods": [method],
                                        "chunk_sizes": [int(size)],
                                        "chunk_overlaps": [int(ov)],
                                        "text": eval_text,
                                        "questions": questions_eval,
                                        "answers": answers_eval,
                                        "include_answer_similarity": include_answer_similarity,
                                    }
                                    if file_id:
                                        job["file_id"] = file_id
                                    if selected_llm_model:
                                        job["llm_model"] = selected_llm_model
                                    jobs.append(job)

                if not jobs:
                    st.error("有効な評価ジョブが生成できませんでした。設定を見直してください。")
                    return

                # ジョブ数が多すぎる場合は警告
                if len(jobs) > 20:
                    st.warning(
                        f"評価ジョブが {len(jobs)} 件あります。処理に時間がかかる可能性があります。"
                    )

                payload: Any
                if len(jobs) == 1:
                    payload = jobs[0]
                else:
                    payload = jobs

                with st.spinner("一括評価を実行中です…（数分かかる場合があります）"):
                    try:
                        resp = http_post(f"{BACKEND_URL}/bulk_evaluate/", json=payload)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"一括評価API呼び出し時にエラーが発生しました: {e}")
                    else:
                        if resp.status_code != 200:
                            st.error(
                                f"一括評価APIエラー: {format_http_error(resp)}"
                            )
                        else:
                            data = resp.json()
                            flat_results: List[Dict[str, Any]] = []
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

                            # エントリを分類: error を含むものはエラー結果として扱い、それ以外を有効結果とする
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
                                # すべてエラーだった場合は成功扱いにせず、代表的なエラーを表示して終了
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
                                # 一括評価完了後に画面を即時更新して結果セクションを表示
                                st.rerun()

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
