from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from http_client import http_get, http_post, format_http_error
from model_utils import fetch_embedding_models as _fetch_embedding_models_common


def _fetch_history_pdfs(BACKEND_URL: str) -> List[Dict[str, Any]]:
    """履歴APIからPDF一覧を取得するヘルパー。"""
    try:
        resp = http_get(f"{BACKEND_URL}/history/pdf-files")
        if resp.status_code != 200:
            st.warning(f"PDF履歴の取得に失敗しました: {resp.status_code} {resp.text}")
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


def _load_sample_text(BACKEND_URL: str, pdf_id: str, max_chars: int = 5000) -> str:
    """指定PDFの抽出テキストからプレビュー用の先頭数千文字を取得する。"""
    text = ""
    try:
        resp = http_get(f"{BACKEND_URL}/get_extracted/{pdf_id}")
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith(
            "application/json"
        ):
            data = resp.json() or {}
            text = data.get("text", "") or ""
    except Exception as e:  # noqa: BLE001
        st.warning(f"抽出テキスト取得中にエラーが発生しました: {e}")
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _fetch_embedding_models(BACKEND_URL: str) -> List[Dict[str, str]]:
    """バックエンドの /list_models からEmbeddingモデル一覧を取得するヘルパー。"""
    return _fetch_embedding_models_common(BACKEND_URL.rstrip("/"))


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


def render_chunking_tab(
    tab_chunking: Any,
    BACKEND_URL: str,
    save_state_to_localstorage: Callable[[], None],
) -> None:
    """チャンキング設定タブの本来のUI。

    - 履歴または現在のPDFからテキストを取得
    - チャンク方式・サイズ・オーバーラップ・（必要なら）Embeddingモデルを指定
    - /chunk API を呼び出してチャンキング結果をプレビュー
    - 任意で /build_vector_store 相当の処理をトリガーしてRAG用ベクトルストアを再構築
    """
    BACKEND_URL = BACKEND_URL.rstrip("/")

    with tab_chunking:
        st.header("チャンキング設定")
        st.caption(
            "PDFから抽出したテキストをどのようにチャンク分割するかを試行できます。\n"
            "一括評価タブと同じパラメータ体系で、サンプルテキストに対するチャンキング結果を確認できます。"
        )

        # --- 共通Embeddingモデルの選択（RAG/チャンキング用） ---
        embedding_models_all = _fetch_embedding_models(BACKEND_URL)
        if embedding_models_all:
            embedding_names_all = [m.get("name", "") for m in embedding_models_all]
            embedding_labels_all = [
                m.get("display_name") or m.get("name") or "unknown" for m in embedding_models_all
            ]
            current_emb = st.session_state.get("embedding_model", "huggingface_bge_small")
            default_emb_idx = (
                embedding_names_all.index(current_emb)
                if current_emb in embedding_names_all
                else 0
            )
            idx_global_emb = st.selectbox(
                "RAG/チャンキング用の共通Embeddingモデル",
                options=list(range(len(embedding_names_all))),
                format_func=lambda i: embedding_labels_all[i],
                index=default_emb_idx,
                key="chunking_global_embedding_select",
            )
            selected_global_emb = embedding_names_all[idx_global_emb]
            st.session_state["embedding_model"] = selected_global_emb
            # 現在のLLMモデルとあわせて永続化（LLMは既存の選択をそのまま利用）
            current_llm = st.session_state.get("llm_model", "gpt-oss")
            _persist_model_selection(BACKEND_URL, current_llm, selected_global_emb)

        # --- 対象PDFの選択 ---
        history_pdfs = _fetch_history_pdfs(BACKEND_URL)
        selected_pdf_id: Optional[str] = None
        if not history_pdfs:
            st.info("履歴にPDFがありません。サイドバーからPDFをアップロードしてください。")
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
                "チャンキング対象のPDF",
                options=list(range(len(labels))),
                format_func=lambda i: labels[i],
                index=default_idx,
                key="chunking_target_pdf_select",
            )
            selected_pdf_id = id_list[sel_idx]
            st.session_state["file_id"] = selected_pdf_id
            st.caption(f"選択中のPDF ID: {selected_pdf_id}")

        st.markdown("---")

        # --- サンプルテキスト取得 ---
        sample_text = ""
        if selected_pdf_id:
            sample_text = _load_sample_text(BACKEND_URL, selected_pdf_id)
        if not sample_text:
            st.info(
                "このPDFから抽出されたテキストが見つかりません。PDFアップロードと抽出処理を確認してください。"
            )
            return

        with st.expander("サンプルテキスト（先頭数千文字）", expanded=False):
            st.text_area("テキスト", value=sample_text, height=200, key="chunking_sample_text", disabled=True)

        st.markdown("---")
        st.subheader("チャンクパラメータの設定")

        col_method, col_size, col_overlap = st.columns([1, 1, 1])
        with col_method:
            chunk_method = st.selectbox(
                "チャンク方式",
                options=["recursive", "fixed", "sentence", "paragraph", "semantic"],
                index=["recursive", "fixed", "sentence", "paragraph", "semantic"].index(
                    st.session_state.get("chunk_method", "recursive")
                ),
                help=(
                    "semantic: 意味的まとまりで分割（サイズ・オーバーラップは無視） / "
                    "recursive: 再帰的文字分割 / sentence: 文単位 / paragraph: 段落単位 / fixed: 固定長"
                ),
                key="chunking_method_select",
            )
            st.session_state["chunk_method"] = chunk_method

        default_size = int(st.session_state.get("chunk_size", 1000) or 1000)
        default_overlap = int(st.session_state.get("chunk_overlap", 200) or 200)
        with col_size:
            chunk_size = st.number_input(
                "チャンクサイズ",
                min_value=100,
                max_value=4000,
                value=default_size,
                step=100,
                key="chunking_size_input",
            )
        with col_overlap:
            chunk_overlap = st.number_input(
                "オーバーラップ",
                min_value=0,
                max_value=2000,
                value=default_overlap,
                step=50,
                key="chunking_overlap_input",
            )

        st.session_state["chunk_size"] = int(chunk_size)
        st.session_state["chunk_overlap"] = int(chunk_overlap)

        # semantic 用パラメータ
        embedding_model_for_semantic: Optional[str] = None
        similarity_threshold = 0.7
        if chunk_method == "semantic":
            st.markdown("### セマンティックチャンキング設定")
            embedding_models = embedding_models_all or _fetch_embedding_models(BACKEND_URL)
            if embedding_models:
                embedding_names = [m.get("name", "") for m in embedding_models]
                embedding_labels = [
                    m.get("display_name") or m.get("name") or "unknown" for m in embedding_models
                ]
                default_name = st.session_state.get("embedding_model")
                default_index = embedding_names.index(default_name) if default_name in embedding_names else 0
                idx_emb = st.selectbox(
                    "セマンティック用Embeddingモデル",
                    options=list(range(len(embedding_names))),
                    format_func=lambda i: embedding_labels[i],
                    index=default_index,
                    key="chunking_semantic_embedding_select",
                )
                embedding_model_for_semantic = embedding_names[idx_emb]
                st.session_state["embedding_model"] = embedding_model_for_semantic
            else:
                st.warning("Embeddingモデル一覧が取得できなかったため、セマンティックチャンキングは利用できません。")

            similarity_threshold = st.slider(
                "semantic 用 類似度しきい値",
                min_value=0.1,
                max_value=0.95,
                value=float(st.session_state.get("semantic_similarity_threshold", 0.7) or 0.7),
                step=0.05,
                key="chunking_semantic_similarity_threshold",
                help="値が高いほど意味的に近い文同士を強くまとめます。",
            )
            st.session_state["semantic_similarity_threshold"] = float(similarity_threshold)

        st.markdown("---")
        st.subheader("チャンキング結果プレビュー")

        if st.button("この設定でチャンキングを実行", key="run_chunk_preview"):
            payload: Dict[str, Any] = {
                "text": sample_text,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "chunk_method": chunk_method,
            }
            if chunk_method == "semantic" and embedding_model_for_semantic:
                payload["chunk_method"] = "semantic"
                payload["embedding_model"] = embedding_model_for_semantic
            try:
                resp = http_post(f"{BACKEND_URL}/chunk/", json=payload)
            except Exception as e:  # noqa: BLE001
                st.error(f"/chunk API呼び出し中にエラーが発生しました: {e}")
            else:
                if resp.status_code != 200:
                    st.error(f"/chunk APIエラー: {format_http_error(resp)}")
                else:
                    data = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
                    chunks = data.get("chunks", []) or []
                    st.session_state["chunk_preview_chunks"] = chunks
                    st.success(f"{len(chunks)} 個のチャンクを生成しました。下で内容を確認できます。")
                    # チャンキング結果を反映した後に画面を再実行して統計・プレビューを更新
                    st.rerun()

        chunks_preview: List[str] = st.session_state.get("chunk_preview_chunks", []) or []
        if chunks_preview:
            lengths = [len(c) for c in chunks_preview]
            st.write(f"チャンク数: {len(chunks_preview)} / 平均長: {sum(lengths) / max(len(lengths), 1):.1f} 文字")
            with st.expander("先頭いくつかのチャンクを表示", expanded=True):
                max_show = min(20, len(chunks_preview))
                for i in range(max_show):
                    st.markdown(f"#### チャンク {i+1}")
                    st.text_area(
                        f"chunk_{i+1}",
                        value=chunks_preview[i],
                        height=120,
                        key=f"chunk_preview_{i}",
                    )

        st.markdown("---")
        st.subheader("RAG用ベクトルストアの再構築（任意）")
        st.caption("現在のPDFとチャンク設定でベクトルストアを再構築します。時間がかかる場合があります。")

        if st.button("この設定でベクトルストアを再構築", key="rebuild_vector_store"):
            if not selected_pdf_id:
                st.error("PDFが選択されていません。")
            else:
                if chunk_method == "semantic" and not embedding_model_for_semantic:
                    st.error("セマンティックチャンキングにはEmbeddingモデルの選択が必要です。")
                else:
                    payload_vs: Dict[str, Any] = {
                        "scope": "single",
                        "pdf_file_id": selected_pdf_id,
                        "embedding_model": embedding_model_for_semantic or st.session_state.get("embedding_model") or "huggingface_bge_small",
                        "chunk_method": chunk_method,
                        "chunk_size": int(chunk_size) if chunk_method != "semantic" else None,
                        "chunk_overlap": int(chunk_overlap) if chunk_method != "semantic" else None,
                        "similarity_threshold": float(similarity_threshold) if chunk_method == "semantic" else None,
                    }
                    progress_placeholder = st.empty()
                    with st.spinner("ベクトルストアを再構築中です…（数分かかる場合があります）"):
                        progress_placeholder.info(
                            "ベクトルストア再構築を開始しました。バックエンドでチャンク化とPGVector構築を実行しています…"
                        )
                        try:
                            resp_vs = http_post(f"{BACKEND_URL}/build_vectorstore/", json=payload_vs)
                        except Exception as e:  # noqa: BLE001
                            progress_placeholder.empty()
                            st.error(f"/build_vector_store API呼び出し中にエラーが発生しました: {e}")
                        else:
                            progress_placeholder.empty()
                            if resp_vs.status_code != 200:
                                st.error(f"ベクトルストア再構築APIエラー: {format_http_error(resp_vs)}")
                            else:
                                data_vs = (
                                    resp_vs.json()
                                    if resp_vs.headers.get("Content-Type", "").startswith("application/json")
                                    else {}
                                )
                                num_chunks = data_vs.get("num_chunks")
                                collection_name = data_vs.get("collection_name")

                                lines: list[str] = []
                                if isinstance(num_chunks, int):
                                    lines.append(
                                        f"ベクトルストアの再構築が完了しました。（チャンク数: {num_chunks}）"
                                    )
                                else:
                                    lines.append("ベクトルストアの再構築が完了しました。")

                                # 画面上で指定されたチャンキング設定をまとめて表示
                                if chunk_method == "semantic":
                                    detail = (
                                        f"方式: semantic / Embedding: {payload_vs.get('embedding_model')} / "
                                        f"類似度しきい値: {payload_vs.get('similarity_threshold')}"
                                    )
                                else:
                                    detail = (
                                        f"方式: {chunk_method} / サイズ: {payload_vs.get('chunk_size')} / "
                                        f"オーバーラップ: {payload_vs.get('chunk_overlap')}"
                                    )
                                lines.append(f"チャンキング設定: {detail}")

                                if collection_name:
                                    lines.append(f"コレクション名: {collection_name}")

                                st.success("\n".join(lines))

