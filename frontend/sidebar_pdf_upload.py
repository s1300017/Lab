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

        # PDF処理ジョブ実行中かどうかのフラグ
        upload_processing_flag = bool(st.session_state.get("upload_processing", False))

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

        # LLMモデルの選択肢
        llm_models = models.get("llm", [])
        llm_options = [m["display_name"] for m in llm_models] if llm_models else [
            "ollama_llama2"
        ]
        llm_names = [m["name"] for m in llm_models] if llm_models else [
            "ollama_llama2"
        ]

        # Embeddingモデルの選択肢
        embedding_models = models.get("embedding", [])
        embedding_options = [
            m["display_name"] for m in embedding_models
        ] if embedding_models else ["openai"]
        embedding_names = [
            m["name"] for m in embedding_models
        ] if embedding_models else ["openai"]

        # デフォルト選択ロジック（初回はGPT-OSSを優先）
        default_llm_idx = 0
        gpt_oss_display_candidates = ["gpt-oss", "gpt oss", "gptoss"]
        gpt_oss_name_candidates = ["gpt-oss", "gpt_oss", "gptoss"]

        if llm_models:
            if (
                "llm_model" in st.session_state
                and st.session_state.llm_model in llm_names
            ):
                default_llm_idx = llm_names.index(st.session_state.llm_model)
            else:
                # display_name優先でGPT-OSSを検索（部分一致・ケース無視）
                for idx, option in enumerate(llm_options):
                    option_lower = option.lower()
                    if any(
                        candidate in option_lower
                        for candidate in gpt_oss_display_candidates
                    ):
                        default_llm_idx = idx
                        break
                else:
                    # nameから候補を検索（部分一致・ケース無視）
                    for idx, llm_name in enumerate(llm_names):
                        llm_name_lower = llm_name.lower()
                        if any(
                            candidate in llm_name_lower
                            for candidate in gpt_oss_name_candidates
                        ):
                            default_llm_idx = idx
                            break
                # セッションの初期値を設定
                if llm_names:
                    st.session_state.llm_model = llm_names[default_llm_idx]
        else:
            st.session_state.llm_model = None

        # モデル設定
        st.subheader("PDF用モデル設定")

        # 環境変数の読み込みを確認
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("警告: OPENAI_API_KEY が設定されていません。")
        else:
            st.sidebar.success("APIキーが設定されています")

        # LLMモデル選択UI
        if llm_models:
            llm_model = st.selectbox(
                "LLMモデルを選択",
                llm_options,
                index=default_llm_idx,
                key="llm_model_select",
                disabled=upload_processing_flag,
            )
            st.session_state.llm_model = llm_names[llm_options.index(llm_model)]
            # --- チャット用モデルも必ず同期（未定義エラー防止） ---
            if (
                "chat_model" not in st.session_state
                or st.session_state.chat_model not in llm_names
            ):
                st.session_state.chat_model = st.session_state.llm_model
        else:
            st.warning("利用可能なLLMモデルが見つかりません")
            st.session_state.llm_model = None
            llm_model = None

        # チャットボットモデル選択肢をllm_modelsから自動生成
        chat_model_options = [
            m["display_name"] if "display_name" in m else m["name"]
            for m in llm_models
        ] if llm_models else ["ollama_llama2"]
        chat_model_names = [m["name"] for m in llm_models] if llm_models else [
            "ollama_llama2"
        ]
        default_chat_idx = 0
        if (
            "chat_model" in st.session_state
            and st.session_state.chat_model in chat_model_names
        ):
            default_chat_idx = chat_model_names.index(st.session_state.chat_model)
        chat_model = st.selectbox(
            "チャットボットモデル",
            options=chat_model_options,
            index=default_chat_idx,
            key="chat_model_select",
            disabled=upload_processing_flag,
        )
        st.session_state.chat_model = chat_model_names[
            chat_model_options.index(chat_model)
        ]

        # Embeddingモデル選択
        if embedding_models:
            default_emb_idx = 0
            if (
                "embedding_model" in st.session_state
                and st.session_state.embedding_model in embedding_names
            ):
                default_emb_idx = embedding_names.index(
                    st.session_state.embedding_model
                )

            selected_embedding = st.selectbox(
                "Embeddingモデル",
                embedding_options,
                index=default_emb_idx,
                key="embedding_model_select",
                disabled=upload_processing_flag,
            )
            st.session_state.embedding_model = embedding_names[
                embedding_options.index(selected_embedding)
            ]
        else:
            st.warning("利用可能なEmbeddingモデルが見つかりません")
            st.session_state.embedding_model = None

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

        # バックエンドからモデル一覧を取得
        models = fetch_models()
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
            # QAテキストが空のままの場合は再処理を強制する（キャンセル済みは除外）
            if (
                upload_processed_once
                and not st.session_state.get("text")
                and not st.session_state.get("upload_error_message")
                and not st.session_state.get("upload_cancel_requested")
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
                    f"アップロード完了: {st.session_state['uploaded_file_name']}（クレンジング処理: {'あり' if cleanse_used else 'なし'}）"
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

            # ファイル内容をBytesIOで復元
            uploaded_file = io.BytesIO(st.session_state["uploaded_file_bytes"])
            uploaded_file.name = st.session_state["uploaded_file_name"]

            if upload_warning_message:
                st.warning(upload_warning_message)

            # エラー終了時のUI（リトライ・別PDF選択）
            if upload_processed_once and upload_error_message:
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

                # ジョブ未開始かつ未キャンセルなら開始
                if (upload_job_id is None) and not upload_cancel_requested:
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
                            st.info(f"PDF自動QA生成の状態: {label}")
                            if job_progress:
                                st.write(job_progress)

                            # 実行中・待機中: ローディング表示＋キャンセルボタンと自動ポーリング
                            if job_status in ("pending", "running"):
                                loader_placeholder.markdown(
                                    """
                                    <div class="pdf-upload-loader">
                                      <div class="pdf-loader-indicator">
                                        <div class="pdf-loader-dot"></div>
                                        <div class="pdf-loader-dot"></div>
                                        <div class="pdf-loader-dot"></div>
                                      </div>
                                      <div class="pdf-loader-text">PDFを処理中です… 少しお待ちください。</div>
                                    </div>
                                    <style>
                                    .pdf-upload-loader {
                                      display: flex;
                                      align-items: center;
                                      gap: 0.6rem;
                                      padding: 0.4rem 0.2rem 0.6rem 0.2rem;
                                    }
                                    .pdf-loader-indicator {
                                      display: inline-flex;
                                      gap: 0.3rem;
                                    }
                                    .pdf-loader-dot {
                                      width: 0.45rem;
                                      height: 0.45rem;
                                      border-radius: 999px;
                                      background: #4f8bf9;
                                      opacity: 0.5;
                                      animation: pdf-loader-bounce 1.2s infinite ease-in-out;
                                    }
                                    .pdf-loader-dot:nth-child(2) {
                                      animation-delay: 0.2s;
                                    }
                                    .pdf-loader-dot:nth-child(3) {
                                      animation-delay: 0.4s;
                                    }
                                    .pdf-loader-text {
                                      font-size: 0.85rem;
                                      color: #444;
                                    }
                                    @keyframes pdf-loader-bounce {
                                      0%, 80%, 100% {
                                        transform: scale(0.4);
                                        opacity: 0.3;
                                      }
                                      40% {
                                        transform: scale(1.0);
                                        opacity: 1.0;
                                      }
                                    }
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.session_state["upload_processing"] = True

                                # タイムアウト判定（長時間running/pendingが続く場合のガード）
                                started_at = st.session_state.get("upload_job_started_at")
                                now_ts = time.time()
                                if started_at is None:
                                    # 既存ジョブから再開した場合など、開始時刻が未設定ならここで補完
                                    st.session_state["upload_job_started_at"] = now_ts
                                    started_at = now_ts

                                elapsed = now_ts - started_at if started_at else 0.0
                                max_seconds = 600  # 約10分を上限とする

                                if elapsed > max_seconds:
                                    # タイムアウト扱いにして、自動ポーリングを停止
                                    st.session_state["upload_processing"] = False
                                    st.session_state["upload_job_id"] = None
                                    st.session_state["upload_job_status"] = "timeout"
                                    st.session_state["upload_job_progress"] = job_progress
                                    st.session_state["upload_error_message"] = (
                                        "PDF処理がタイムアウトしました。PDFのサイズや設定を見直して再試行してください。"
                                    )
                                    st.session_state["upload_job_started_at"] = None
                                    save_state_to_localstorage()
                                    st.error(
                                        "PDF処理が10分以上経過しても完了しなかったため、中断しました。"
                                    )
                                    # ここで明示的にrerunしないことで無限ループを防ぐ
                                    return

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
                                            st.rerun()

                                # 自動ポーリング（タイムアウトしていない場合のみ）
                                time.sleep(1)
                                st.rerun()

                            # 正常完了: ローディングを消して結果をセッションに反映
                            elif job_status == "completed":
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
                                    error_message = (
                                        "PDF処理結果に質問・回答が含まれていません。PDFの内容を確認してください。"
                                    )

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
        # has_uploaded_file の場合の処理ここまで
