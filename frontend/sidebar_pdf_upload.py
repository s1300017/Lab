import os
import json
import base64
import io
import hashlib
import time
from typing import Callable

import streamlit as st
import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval

from http_client import http_get, http_post


def render_pdf_upload_sidebar(
    BACKEND_URL: str,
    *,
    init_session_state: Callable[[], None],
    reset_document_session_state: Callable[[], None],
    save_state_to_localstorage: Callable[[], None],
    jst_now_str: Callable[[], str],
) -> None:
    """サイドバーにPDFアップロードと非同期ジョブUIを描画する。"""

    with st.sidebar:
        st.header("設定")

        # 直前の質問生成完了メッセージがあれば一度だけ表示する
        last_qa_msg = st.session_state.get("last_qa_generation_success")
        if last_qa_msg:
            st.success(last_qa_msg)
            # 再表示を防ぐためクリア
            st.session_state["last_qa_generation_success"] = ""

        # PDF処理ジョブ実行中かどうかのフラグ
        upload_processing_flag = bool(st.session_state.get("upload_processing", False))
        if upload_processing_flag and not st.session_state.get("upload_job_id"):
            # ジョブIDが無いのに処理中フラグだけ立っている場合は異常状態とみなしリセットする
            st.session_state["upload_processing"] = False
            upload_processing_flag = False

        # モデル・エンベディングモデルリストをAPI経由で取得
        def fetch_models():
            try:
                resp = http_get(f"{BACKEND_URL}/list_models")
                resp.raise_for_status()
                data = resp.json()
                # カテゴライズされたモデルを別々のリストで返す
                llm_models = data.get("LLM", [])
                embedding_models = data.get("Embedding", [])
                return {
                    "llm": llm_models,
                    "embedding": embedding_models,
                }
            except Exception as e:
                st.error(f"モデルリスト取得エラー: {e}")
                return {"llm": [], "embedding": []}

        # グローバルにモデルリストを保存
        if "models" not in st.session_state:
            st.session_state.models = fetch_models()

        models = st.session_state.models

        # ここではグローバルなLLM/Embedding/チャットモデルは扱わず、
        # 後続の「PDFアップロード用 LLMモデル選択」セクションのみで
        # PDF処理パイプライン専用のモデル設定を行う。

        # モデル設定（PDFアップロード用の情報表示）
        st.subheader("PDF用モデル設定")

        # 環境変数の読み込みを確認
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("警告: OPENAI_API_KEY が設定されていません。")
        else:
            st.sidebar.success("APIキーが設定されています")

        # 以前ここにあった汎用LLM/チャットボット/Embeddingモデル選択UIは、
        # チャットタブ・チャンキングタブ・評価タブ側で行う方針に変更しました。
        # そのため、このセクションでは実際の選択UIは表示しません。

        # --- localStorageからの復元は廃止 ---
        def load_state_from_localstorage():
            """以前はブラウザlocalStorageから状態を復元していたが、

            状態管理はDB（履歴API）とsession_stateに一本化する方針としたため、
            ここでは何も行わない。
            """
            return

        load_state_from_localstorage()

        # --- LLMモデル選択（常に表示） ---
        st.subheader("🤖 PDFアップロード用 LLMモデル選択")
        col1, col2 = st.columns(2)

        # バックエンドからモデル一覧を取得（初回取得済みのキャッシュを利用）
        models = st.session_state.get("models") or fetch_models()
        st.session_state.models = models
        llm_models = [model["name"] for model in models.get("llm", [])]

        gpt_oss_candidates = ["gpt-oss", "gpt_oss", "gptoss"]

        def find_gpt_oss_index(models: list[str]) -> int:
            """gpt-oss系モデルを優先的に選択するためのインデックスを返す。"""
            if not models:
                return 0
            for idx, model_name in enumerate(models):
                lower_name = model_name.lower()
                if any(candidate in lower_name for candidate in gpt_oss_candidates):
                    return idx
            return 0

        default_pdf_llm_idx = find_gpt_oss_index(llm_models)

        with col1:
            question_llm_model = st.selectbox(
                "質問生成用LLMモデル（PDF用）",
                llm_models,
                index=default_pdf_llm_idx,
                help="PDFから質問を自動生成するためのLLMモデルを選択。",
                disabled=upload_processing_flag,
            )

        with col2:
            answer_llm_model = st.selectbox(
                "回答生成用LLMモデル（PDF用）",
                llm_models,
                index=default_pdf_llm_idx,
                help="質問に対する回答を自動で生成するためのLLMモデルを選択。",
                disabled=upload_processing_flag,
            )

        # --- クレンジング設定とアップローダー（常時表示） ---
        st.subheader("1. PDFアップロードとOCR設定")
        cleanse_default = st.session_state.get("upload_cleanse_flag", False)
        cleanse = st.checkbox(
            "表・ノイズ除去クレンジング処理を行う",
            value=cleanse_default,
            help="PDF内の表やノイズを自動で除去します",
            key="pdf_cleanse_checkbox",
            disabled=upload_processing_flag,
        )
        st.session_state["upload_cleanse_flag"] = cleanse

        ocr_default = st.session_state.get("ocr_engine", "auto")
        ocr_options = ["auto", "pypdf", "deepseek", "ollama_deepseek"]
        try:
            default_ocr_idx = (
                ocr_options.index(ocr_default) if ocr_default in ocr_options else 0
            )
        except Exception:
            default_ocr_idx = 0
        ocr_engine = st.selectbox(
            "OCRエンジン",
            ocr_options,
            index=default_ocr_idx,
            help=(
                "auto: まずPyPDFで抽出し不足時にOCRへフォールバック / "
                "pypdf: テキスト抽出のみ / "
                "deepseek: DeepSeek OCR (MLX版) を使用 / "
                "ollama_deepseek: DeepSeek-OCR (Ollama版) を使用"
            ),
            key="ocr_engine_select",
            disabled=upload_processing_flag,
        )
        st.session_state["ocr_engine"] = ocr_engine

        quality_default = st.session_state.get("ocr_quality", "balanced")
        quality_keys = ["fast", "balanced", "high"]
        quality_labels = {
            "fast": "高速（先頭少数ページ・高速処理）",
            "balanced": "標準（中程度のページ数・標準処理）",
            "high": "高品質（全ページ対象・時間長め）",
        }
        try:
            default_quality_idx = (
                quality_keys.index(quality_default)
                if quality_default in quality_keys
                else 1
            )
        except Exception:
            default_quality_idx = 1
        quality_label_list = [quality_labels[k] for k in quality_keys]
        selected_quality_label = st.selectbox(
            "OCR品質（DeepSeek-OCR 用）",
            quality_label_list,
            index=default_quality_idx,
            key="ocr_quality_select",
            disabled=upload_processing_flag,
        )
        selected_quality = quality_keys[quality_label_list.index(selected_quality_label)]
        st.session_state["ocr_quality"] = selected_quality

        compression_default = st.session_state.get("ocr_image_compression", "balanced")
        compression_keys = ["light", "balanced", "high"]
        compression_labels = {
            "light": "軽量（高圧縮・高速・小さめ画像）",
            "balanced": "標準（バランス重視）",
            "high": "高画質（低圧縮・時間長め）",
        }
        try:
            default_compression_idx = (
                compression_keys.index(compression_default)
                if compression_default in compression_keys
                else 1
            )
        except Exception:
            default_compression_idx = 1
        compression_label_list = [compression_labels[k] for k in compression_keys]
        selected_compression_label = st.selectbox(
            "OCR画像圧縮（DeepSeek-OCR 用）",
            compression_label_list,
            index=default_compression_idx,
            key="ocr_image_compression_select",
            help="DeepSeek-OCR (Ollama版) 利用時に有効です。圧縮を強くすると処理は速くなりますが画質は下がります。",
            disabled=upload_processing_flag,
        )
        selected_compression = compression_keys[
            compression_label_list.index(selected_compression_label)
        ]
        st.session_state["ocr_image_compression"] = selected_compression

        uploaded_file_input = st.file_uploader(
            "PDFをアップロード",
            type=["pdf"],
            key="pdf_uploader",
            disabled=upload_processing_flag,
        )
        if uploaded_file_input is not None:
            file_bytes = uploaded_file_input.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            previous_hash = st.session_state.get("current_upload_file_id")

            if previous_hash != file_hash:
                # 新しいPDFが選択された場合のみ状態をリセット
                reset_document_session_state()
                st.session_state["upload_cleanse_flag"] = cleanse
                st.session_state["uploaded_file_bytes"] = file_bytes
                st.session_state["uploaded_file_name"] = uploaded_file_input.name
                st.session_state["uploaded_file_size"] = uploaded_file_input.size
                st.session_state["uploaded_at"] = jst_now_str()
                st.session_state["cleanse_used"] = cleanse
                # LLMモデル情報も保存
                st.session_state["question_llm_model"] = question_llm_model
                st.session_state["answer_llm_model"] = answer_llm_model
                st.session_state["upload_warning_message"] = ""
                st.session_state["current_upload_file_id"] = file_hash
                save_state_to_localstorage()
                st.rerun()
            else:
                # 同一PDFでも再処理できるよう状態を更新し直す
                st.session_state["upload_cleanse_flag"] = cleanse
                st.session_state["uploaded_file_bytes"] = file_bytes
                st.session_state["uploaded_file_name"] = uploaded_file_input.name
                st.session_state["uploaded_file_size"] = uploaded_file_input.size
                st.session_state["uploaded_at"] = jst_now_str()
                st.session_state["cleanse_used"] = cleanse
                st.session_state["question_llm_model"] = question_llm_model
                st.session_state["answer_llm_model"] = answer_llm_model
                st.session_state["upload_error_message"] = None
                st.session_state["upload_warning_message"] = ""
                st.session_state["text"] = ""
                st.session_state["qa_questions"] = []
                st.session_state["qa_answers"] = []
                st.session_state["qa_meta"] = []
                st.session_state["upload_processed_once"] = False
                st.session_state["upload_processing"] = False
                st.session_state["upload_cancel_requested"] = False
                st.session_state["upload_cancel_message"] = ""
                st.session_state["current_upload_file_id"] = file_hash
                save_state_to_localstorage()

        # すでにアップロード済みかどうかでUIを分岐
        has_uploaded_file = (
            "uploaded_file_bytes" in st.session_state
            and st.session_state.get("uploaded_file_bytes") is not None
            and "uploaded_file_name" in st.session_state
            and st.session_state.get("uploaded_file_name")
        )
        if has_uploaded_file:
            upload_processed_once = st.session_state.get(
                "upload_processed_once", False
            )

            # まだ未処理フラグだが、履歴側には同名PDFが存在して抽出結果がある場合は自動復元する
            if (
                not upload_processed_once
                and not st.session_state.get("upload_processing", False)
                and not st.session_state.get("upload_error_message")
            ):
                try:
                    resp_hist = http_get(f"{BACKEND_URL}/history/pdf-files")
                    if resp_hist.status_code == 200 and resp_hist.headers.get(
                        "Content-Type", ""
                    ).startswith("application/json"):
                        data_hist = resp_hist.json() or {}
                        items = data_hist.get("items", []) or []
                        current_name = st.session_state.get("uploaded_file_name")
                        candidates = [
                            it
                            for it in items
                            if (it.get("original_name") or it.get("file_name"))
                            == current_name
                        ]
                        if candidates:
                            def _uploaded_at_key(it: dict) -> str:
                                val = it.get("uploaded_at")
                                return str(val) if val is not None else ""

                            candidates_sorted = sorted(
                                candidates,
                                key=_uploaded_at_key,
                                reverse=True,
                            )
                            target = candidates_sorted[0]
                            fid = target.get("id")
                            if fid is not None:
                                try:
                                    resp_ext = http_get(
                                        f"{BACKEND_URL}/get_extracted/{fid}"
                                    )
                                    if resp_ext.status_code == 200 and resp_ext.headers.get(
                                        "Content-Type", ""
                                    ).startswith("application/json"):
                                        data_x = resp_ext.json() or {}
                                        st.session_state["file_id"] = fid
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
                                        st.session_state["upload_processed_once"] = True
                                        save_state_to_localstorage()
                                except Exception:
                                    pass
                except Exception:
                    pass

            upload_processed_once = st.session_state.get(
                "upload_processed_once", False
            )
            # QAテキストが空のままの場合は再処理を強制する（キャンセル済みは除外）
            # ただし、upload_job_id が存在する（直近のジョブ実行コンテキスト）場合に限定する
            if (
                upload_processed_once
                and not st.session_state.get("text")
                and not st.session_state.get("upload_error_message")
                and not st.session_state.get("upload_cancel_requested")
                and st.session_state.get("upload_job_id")
            ):
                st.session_state["upload_processed_once"] = False
                st.session_state["upload_processing"] = False
                st.session_state["upload_cancel_requested"] = False
                st.session_state["upload_cancel_message"] = ""
                st.session_state.pop("qa_questions", None)
                st.session_state.pop("qa_answers", None)
                st.session_state.pop("qa_meta", None)
                save_state_to_localstorage()
                st.rerun()

            show_upload_banner = not st.session_state.get(
                "upload_processed_once", False
            )
            if show_upload_banner:
                cleanse_used = st.session_state.get("cleanse_used", False)
                st.success(
                    f"アップロード済みPDF: {st.session_state['uploaded_file_name']}（クレンジング処理設定: {'あり' if cleanse_used else 'なし'}）"
                )

            upload_error_message = st.session_state.get("upload_error_message")
            upload_processing = st.session_state.get("upload_processing", False)
            upload_cancel_requested = st.session_state.get(
                "upload_cancel_requested", False
            )
            upload_cancel_message = (
                st.session_state.get("upload_cancel_message") or ""
            )
            upload_warning_message = (
                st.session_state.get("upload_warning_message") or ""
            )
            job_start_requested = st.session_state.get("upload_job_start_requested", False)

            st.subheader("2. PDF処理の実行")

            with st.expander("デバッグ: PDF処理ジョブ状態", expanded=False):
                st.write(
                    {
                        "upload_processed_once": upload_processed_once,
                        "upload_processing": upload_processing,
                        "upload_error_message": upload_error_message,
                        "upload_cancel_requested": upload_cancel_requested,
                        "upload_job_id": st.session_state.get("upload_job_id"),
                        "upload_job_status": st.session_state.get("upload_job_status"),
                        "upload_job_progress": st.session_state.get("upload_job_progress"),
                        "upload_job_started_at": st.session_state.get("upload_job_started_at"),
                        "upload_job_start_requested": st.session_state.get("upload_job_start_requested"),
                    }
                )

            if (not upload_processed_once) and not upload_error_message and not upload_processing:
                if not job_start_requested:
                    if st.button("この設定でPDF処理を開始", key="start_pdf_processing_button"):
                        st.session_state["upload_job_start_requested"] = True
                        save_state_to_localstorage()
                        st.rerun()

            # ファイル内容をBytesIOで復元
            uploaded_file = io.BytesIO(st.session_state["uploaded_file_bytes"])
            uploaded_file.name = st.session_state["uploaded_file_name"]

            if upload_warning_message:
                st.warning(upload_warning_message)

            # エラー終了時のUI（リトライ・別PDF選択）
            if upload_error_message:
                st.error(f"PDF処理中にエラーが発生しました: {upload_error_message}")
                st.info(
                    "設定を見直すか、再処理または別ファイルのアップロードをお試しください。"
                )
                col_retry, col_reset = st.columns(2)
                with col_retry:
                    if st.button("同じPDFで再処理", key="retry_pdf_processing"):
                        st.session_state.text = ""
                        st.session_state.qa_questions = []
                        st.session_state.qa_answers = []
                        st.session_state.qa_meta = []
                        st.session_state.upload_error_message = None
                        st.session_state.upload_processed_once = False
                        st.session_state.upload_cancel_requested = False
                        st.session_state.upload_cancel_message = ""
                        st.session_state.upload_processing = False
                        st.session_state.upload_job_id = None
                        st.session_state.upload_job_status = None
                        st.session_state.upload_job_progress = ""
                        save_state_to_localstorage()
                        st.rerun()
                with col_reset:
                    if st.button("別のPDFを選び直す", key="choose_another_pdf"):
                        reset_document_session_state()
                        save_state_to_localstorage()
                        st.rerun()

            # まだテキストやQAがセッションに無く、エラー状態でなければPDF処理ジョブを起動または状態確認
            if (not upload_processed_once) and not st.session_state.get(
                "upload_error_message"
            ):
                uploaded_name = st.session_state.get("uploaded_file_name") or "-"
                upload_job_id = st.session_state.get("upload_job_id")
                job_start_requested = st.session_state.get("upload_job_start_requested", False)

                # ジョブ未開始かつ未キャンセルなら開始
                if job_start_requested and (upload_job_id is None) and not upload_cancel_requested:
                    st.session_state["upload_job_start_requested"] = False
                    st.session_state["upload_processing"] = True
                    st.session_state["upload_cancel_requested"] = False
                    st.session_state["upload_cancel_message"] = ""
                    save_state_to_localstorage()

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file,
                            "application/pdf",
                        )
                    }
                    payload = {
                        "cleanse": str(cleanse),
                        "question_llm_model": question_llm_model,
                        "answer_llm_model": answer_llm_model,
                        "ocr_engine": ocr_engine,
                        "ocr_quality": selected_quality,
                         "ocr_image_compression": selected_compression,
                    }
                    try:
                        resp_start = http_post(
                            f"{BACKEND_URL}/upload_job/start",
                            files=files,
                            data=payload,
                        )
                        if resp_start.status_code == 200:
                            data_start = (
                                resp_start.json()
                                if resp_start.headers.get(
                                    "Content-Type", ""
                                ).startswith("application/json")
                                else {}
                            )
                            job_id = data_start.get("job_id")
                            if job_id:
                                st.session_state["upload_job_id"] = job_id
                                # ジョブ開始時刻を記録（タイムアウト検知用）
                                st.session_state["upload_job_started_at"] = time.time()
                            else:
                                st.session_state[
                                    "upload_error_message"
                                ] = "アップロードジョブIDの取得に失敗しました。"
                                st.session_state["upload_processing"] = False
                        else:
                            st.session_state[
                                "upload_error_message"
                            ] = (
                                f"PDF処理ジョブの開始に失敗しました: "
                                f"{resp_start.status_code} {resp_start.text}"
                            )
                            st.session_state["upload_processing"] = False
                    except Exception as e:
                        st.session_state[
                            "upload_error_message"
                        ] = f"PDF処理ジョブ開始時にエラーが発生しました: {e}"
                        st.session_state["upload_processing"] = False
                    save_state_to_localstorage()

                # ジョブIDがあれば状態を取得して進捗を表示
                upload_job_id = st.session_state.get("upload_job_id")
                if upload_job_id:
                    # ローディング表示用プレースホルダ（状態に応じて表示/非表示を切り替える）
                    loader_placeholder = st.empty()

                    try:
                        resp_status = http_get(
                            f"{BACKEND_URL}/upload_job/status/{upload_job_id}"
                        )
                        if resp_status.status_code == 200:
                            status_data = (
                                resp_status.json()
                                if resp_status.headers.get(
                                    "Content-Type", ""
                                ).startswith("application/json")
                                else {}
                            )
                            job_status = status_data.get("status")
                            job_progress = status_data.get("progress") or ""
                            job_error = status_data.get("error") or ""
                            job_result = status_data.get("result") or {}
                            job_cancel_requested = bool(
                                status_data.get("cancel_requested")
                            )

                            st.session_state["upload_job_status"] = job_status
                            st.session_state["upload_job_progress"] = job_progress

                            status_label_map = {
                                "pending": "待機中",
                                "running": "実行中",
                                "completed": "完了",
                                "cancelled": "キャンセル済み",
                                "error": "エラー",
                            }
                            label = status_label_map.get(
                                job_status, job_status or "不明"
                            )

                            # 実行中・待機中は経過時間＋アニメーションのみ表示
                            if job_status in ("pending", "running"):
                                st.session_state["upload_processing"] = True

                                started_at = st.session_state.get("upload_job_started_at")
                                now_ts = time.time()
                                if started_at is None:
                                    st.session_state["upload_job_started_at"] = now_ts
                                    started_at = now_ts

                                # JS側でカウントアップするための開始時刻（ミリ秒）
                                start_ms = int(started_at * 1000)

                                loader_placeholder.markdown(
                                    f"""
                                    <div class="pdf-upload-loader">
                                      <div class="pdf-loader-indicator">
                                        <div class="pdf-loader-dot"></div>
                                        <div class="pdf-loader-dot"></div>
                                        <div class="pdf-loader-dot"></div>
                                      </div>
                                      <div class="pdf-loader-text">
                                        PDFを処理中です… 少しお待ちください。
                                      </div>
                                    </div>
                                    <div class="pdf-loader-elapsed">
                                      経過時間: <span id="pdf-upload-elapsed">0</span> 秒
                                    </div>
                                    <script>
                                    (function() {{
                                      const startMs = {start_ms};
                                      function updateElapsed() {{
                                        const now = Date.now();
                                        const elapsedSec = Math.max(0, Math.floor((now - startMs) / 1000));
                                        const el = document.getElementById("pdf-upload-elapsed");
                                        if (el) {{
                                          el.textContent = String(elapsedSec);
                                        }}
                                      }}
                                      updateElapsed();
                                      setInterval(updateElapsed, 1000);
                                    }})();
                                    </script>
                                    <style>
                                    .pdf-upload-loader {{
                                      display: flex;
                                      align-items: center;
                                      gap: 0.6rem;
                                      padding: 0.4rem 0.2rem 0.2rem 0.2rem;
                                    }}
                                    .pdf-loader-indicator {{
                                      display: inline-flex;
                                      gap: 0.3rem;
                                    }}
                                    .pdf-loader-dot {{
                                      width: 0.45rem;
                                      height: 0.45rem;
                                      border-radius: 999px;
                                      background: #4f8bf9;
                                      opacity: 0.5;
                                      animation: pdf-loader-bounce 1.2s infinite ease-in-out;
                                    }}
                                    .pdf-loader-dot:nth-child(2) {{
                                      animation-delay: 0.2s;
                                    }}
                                    .pdf-loader-dot:nth-child(3) {{
                                      animation-delay: 0.4s;
                                    }}
                                    .pdf-loader-text {{
                                      font-size: 0.85rem;
                                      color: #444;
                                    }}
                                    .pdf-loader-elapsed {{
                                      font-size: 0.8rem;
                                      color: #666;
                                      padding: 0 0.2rem 0.4rem 1.1rem;
                                    }}
                                    @keyframes pdf-loader-bounce {{
                                      0%, 80%, 100% {{
                                        transform: scale(0.4);
                                        opacity: 0.3;
                                      }}
                                      40% {{
                                        transform: scale(1.0);
                                        opacity: 1.0;
                                      }}
                                    }}
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                if not job_cancel_requested:
                                    col_cancel, col_spacer = st.columns([1, 1])
                                    with col_cancel:
                                        if st.button(
                                            "このPDF処理をキャンセルする",
                                            key="cancel_pdf_processing_button_job",
                                        ):
                                            try:
                                                resp_cancel = http_post(
                                                    f"{BACKEND_URL}/upload_job/cancel/{upload_job_id}"
                                                )
                                                if resp_cancel.status_code == 200:
                                                    st.session_state[
                                                        "upload_cancel_requested"
                                                    ] = True
                                                    st.session_state[
                                                        "upload_cancel_message"
                                                    ] = "キャンセル要求を送信しました。"
                                                else:
                                                    st.error(
                                                        "キャンセル要求に失敗しました: "
                                                        f"{resp_cancel.status_code} {resp_cancel.text}"
                                                    )
                                            except Exception as e:
                                                st.error(
                                                    f"キャンセル要求送信中にエラーが発生しました: {e}"
                                                )
                                            save_state_to_localstorage()

                                time.sleep(2.0)
                                st.rerun()

                            # 完了・キャンセル・エラー時は、状態ラベルと詳細メッセージを表示
                            else:
                                st.info(f"PDF自動QA生成の状態: {label}")
                                if job_progress:
                                    st.write(job_progress)

                                # 正常完了: ローディングを消して結果をセッションに反映
                                if job_status == "completed":
                                    loader_placeholder.empty()
                                    st.session_state["upload_processing"] = False
                                    st.session_state["upload_processed_once"] = True
                                    st.session_state["upload_cancel_requested"] = False
                                    st.session_state["upload_cancel_message"] = ""
                                    st.session_state["upload_job_id"] = None
                                    st.session_state["upload_job_started_at"] = None

                                    response_data = job_result or {}
                                    error_message = None
                                    warning_message = ""

                                    if response_data.get("file_id"):
                                        st.session_state["file_id"] = response_data["file_id"]
                                    if response_data.get("experiment_id"):
                                        st.session_state["experiment_id"] = response_data["experiment_id"]
                                        components.html(
                                            f"""
                                            <script>
                                            localStorage.setItem('rag_experiment_id', '{response_data['experiment_id']}');
                                            </script>
                                            """,
                                            height=0,
                                        )

                                    st.session_state.text = response_data.get(
                                        "text", st.session_state.get("text", "")
                                    )
                                    questions = response_data.get("questions") or []
                                    answers = response_data.get("answers") or []
                                    qa_meta = response_data.get("qa_meta") or []
                                    warning_message = response_data.get("warning") or ""

                                    st.session_state.qa_questions = questions
                                    st.session_state.qa_answers = answers
                                    st.session_state.qa_meta = qa_meta

                                    if (
                                        "current_upload_file_id" not in st.session_state
                                        or st.session_state["current_upload_file_id"] is None
                                    ):
                                        st.session_state["current_upload_file_id"] = hashlib.md5(
                                            st.session_state["uploaded_file_bytes"]
                                        ).hexdigest()

                                    if response_data.get("error"):
                                        error_message = f"PDF処理APIエラー: {response_data['error']}"
                                    elif questions and answers:
                                        st.success(
                                            "PDFからテキスト・質問・回答セットを自動生成しました。履歴タブから確認・編集が可能です。"
                                        )
                                        if warning_message:
                                            st.warning(warning_message)
                                    else:
                                        st.success(
                                            "PDFからテキストの抽出が完了しました。下のボタンから質問生成を実行できます。"
                                        )
                                        if warning_message:
                                            st.warning(warning_message)

                                    # qa_meta が質問数より少ない場合はダミーで補完
                                    while len(qa_meta) < len(questions):
                                        idx = len(qa_meta)
                                        fallback_answer = (
                                            answers[idx] if idx < len(answers) else "データ不足"
                                        )
                                        qa_meta.append(
                                            {
                                                "score": 1.0,
                                                "is_auto_fixed": False,
                                                "is_dummy_answer": True,
                                                "candidates": [fallback_answer],
                                                "candidate_scores": [1.0],
                                            }
                                        )
                                    st.session_state["qa_meta"] = qa_meta

                                    if error_message:
                                        st.session_state["upload_error_message"] = error_message
                                        st.error(error_message)
                                    # 正常完了またはエラー状態をlocalStorageに永続化しておく
                                    save_state_to_localstorage()
                                    # PDF処理完了後にメインエリアを即時更新して新しい状態を反映
                                    st.rerun()
                        else:
                            # ステータス取得が非200の場合もエラー扱いとしてジョブを終了させる
                            loader_placeholder.empty()
                            st.session_state["upload_processing"] = False
                            st.session_state["upload_job_id"] = None
                            st.session_state["upload_job_started_at"] = None
                            st.session_state["upload_job_status"] = None
                            st.session_state["upload_error_message"] = (
                                f"PDF処理ジョブ状態取得に失敗しました: {resp_status.status_code} {resp_status.text}"
                            )
                            save_state_to_localstorage()
                    except Exception as e:
                        loader_placeholder.empty()
                        st.error(
                            f"PDF処理ジョブ状態取得中にエラーが発生しました: {e}"
                        )
                        st.session_state["upload_error_message"] = (
                            f"PDF処理ジョブ状態取得中にエラーが発生しました: {e}"
                        )
                        st.session_state["upload_processing"] = False
                        # エラー状態を永続化し、次回リロード時にジョブIDをクリアできるようにする
                        save_state_to_localstorage()

            file_id_value = st.session_state.get("file_id")
            extracted_text = st.session_state.get("text") or ""
            if file_id_value and extracted_text and not upload_processing:
                with st.expander("抽出テキスト（先頭部分）", expanded=False):
                    st.text_area(
                        "抽出テキスト",
                        value=extracted_text,
                        height=200,
                        key="pdf_extracted_text_preview",
                        disabled=True,
                    )

                # すでにQAが存在する場合は再度ボタンを出さない（履歴からの再生成は履歴タブで行う想定）
                has_qa = bool(st.session_state.get("qa_questions"))
                if not has_qa:
                    if st.button("このPDFから質問を生成する", key="generate_qa_for_pdf"):
                        try:
                            resp_qa = http_post(
                                f"{BACKEND_URL}/pdf/{file_id_value}/generate_qa",
                                data={
                                    "question_llm_model": question_llm_model,
                                    "answer_llm_model": answer_llm_model,
                                },
                            )
                        except Exception as e:
                            st.error(f"質問生成API呼び出し中にエラーが発生しました: {e}")
                        else:
                            if resp_qa.status_code != 200:
                                st.error(
                                    f"質問生成APIエラー: {resp_qa.status_code} {resp_qa.text}"
                                )
                            else:
                                qa_data = (
                                    resp_qa.json()
                                    if resp_qa.headers.get("Content-Type", "").startswith(
                                        "application/json"
                                    )
                                    else {}
                                )
                                st.session_state.text = qa_data.get("text", extracted_text)
                                st.session_state.qa_questions = qa_data.get("questions") or []
                                st.session_state.qa_answers = qa_data.get("answers") or []
                                st.session_state.qa_meta = qa_data.get("qa_meta") or []
                                save_state_to_localstorage()
                                # 次回リロード時に完了メッセージを表示する
                                st.session_state[
                                    "last_qa_generation_success"
                                ] = "質問・回答の自動生成が完了しました。履歴タブから確認・編集が可能です。"
                                st.rerun()
        # has_uploaded_file の場合の処理ここまで
