from datetime import datetime
from pytz import timezone

def jst_now_str():
    return datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go  # レーダーチャート等で使用

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
        st.session_state.llm_model = "ollama_llama2" # Default to Ollama
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = "huggingface_bge_small" # Default to HuggingFace

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


def clear_database():
    try:
        response = requests.post(f"{BACKEND_URL}/clear_db/")
        if response.status_code == 200:
            st.success("データベースを正常にクリアしました！")
            # Clear session state related to old data
            st.session_state.text = ""
            st.session_state.chunks = []
            st.session_state.evaluation_results = None
            st.session_state.chat_history = []
        else:
            st.error(f"データベースのクリアに失敗しました: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"バックエンドに接続できませんでした: {e}")

# --- UI Layout ---
with st.sidebar:
    st.header("設定")
    
    # モデル・エンベディングモデルリストをAPI経由で取得
    import requests
    def fetch_models():
        try:
            resp = requests.get(f"{BACKEND_URL}/list_models")
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", [])
        except Exception as e:
            st.error(f"モデルリスト取得エラー: {e}")
            return []
    models = fetch_models()
    model_options = [m['display_name'] for m in models] if models else ["ollama_llama2", "openai"]
    model_names = [m['name'] for m in models] if models else ["ollama_llama2", "openai"]

    # デフォルト選択ロジック
    default_idx = 0
    if 'llm_model' in st.session_state and st.session_state.llm_model in model_names:
        default_idx = model_names.index(st.session_state.llm_model)

    st.session_state.llm_model = st.selectbox(
        "LLMモデル",
        model_options,
        index=default_idx
    )
    # 内部的にはnameで管理
    if models:
        st.session_state.llm_model = model_names[model_options.index(st.session_state.llm_model)]

    # Embeddingモデルも同様にAPI化（必要に応じて拡張可）
    embedding_options = {
        "OpenAI": "openai",
        "bge-small（ローカル高速）": "huggingface_bge_small",
        "MiniLM（軽量・多言語）": "huggingface_miniLM",
    }
    emb_default_idx = 0 if st.session_state.embedding_model == "huggingface_bge_small" else 1
    st.session_state.embedding_model = st.selectbox(
        "Embeddingモデル",
        embedding_options,
        index=emb_default_idx
    )

    uploaded_file = st.file_uploader("PDFをアップロード", type=["pdf"])
    if uploaded_file is not None:
        # 多重アップロード防止: ファイル名・サイズで判定
        if (
            "last_uploaded_filename" not in st.session_state
            or st.session_state.last_uploaded_filename != uploaded_file.name
            or st.session_state.last_uploaded_filesize != uploaded_file.size
        ):
            with st.spinner('PDFを処理中...'):
                import io
                files = {'file': ('uploaded.pdf', io.BytesIO(uploaded_file.getvalue()), 'application/pdf')}
                try:
                    response = requests.post(f"{BACKEND_URL}/uploadfile/", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        if 'questions' in data and 'answers' in data:
                            st.session_state.text = data['text']
                            st.session_state.qa_questions = data['questions']
                            st.session_state.qa_answers = data['answers']
                            st.success("PDFからテキスト・質問・回答セットを自動生成しました。以降の評価処理でこのセットが使われます。")
                            st.write("### 自動生成された質問:")
                            for i, q in enumerate(data['questions']):
                                st.write(f"Q{i+1}: {q}")
                            st.write("### 自動生成された回答:")
                            for i, a in enumerate(data['answers']):
                                st.write(f"A{i+1}: {a}")
                        else:
                            st.error(f"PDF処理APIの返却内容にquestions/answersが含まれていません: {data}")
                    else:
                        st.error(f"ファイル処理エラー: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"接続エラー: {e}")
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.last_uploaded_filesize = uploaded_file.size
        else:
            st.info("同じファイルは再アップロードされません。")

    if st.button("データベースを初期化してやり直す"):
        clear_database()
        st.rerun()

    # 以降の処理は正常なインデントで記述
    if st.session_state.text:
        st.subheader("チャンキング設定")
        chunk_method = st.radio("チャンク化方式", ["recursive", "semantic"], index=0, help="recursive: 文字数ベース, semantic: 意味ベース")
        chunk_size = st.slider("チャンクサイズ", 200, 4000, 1000, 100)
        chunk_overlap = st.slider("チャンクオーバーラップ", 50, 1000, 200, 50)
        embedding_model = st.selectbox("埋め込みモデル (semantic時必須)", ["huggingface_bge_small", "openai"], index=0)
        if st.button("チャンキングとベクトル化"):
            with st.spinner('チャンキングとベクトル化を実行中...'):
                # 1. Chunk
                payload = {
                    "text": st.session_state.text,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "chunk_method": chunk_method,
                }
                if chunk_method == "semantic":
                    payload["embedding_model"] = embedding_model

                chunk_response = requests.post(f"{BACKEND_URL}/chunk/", json=payload)
                if chunk_response.status_code == 200:
                    st.session_state.chunks = chunk_response.json()['chunks']
                    # 2. Embed
                    embed_payload = {"chunks": st.session_state.chunks, "embedding_model": st.session_state.embedding_model}
                    embed_response = requests.post(f"{BACKEND_URL}/embed_and_store/", json=embed_payload)
                    if embed_response.status_code == 200:
                        st.success(f"{len(st.session_state.chunks)}個のチャンクを生成し、ベクトル化しました。")
                    else:
                        st.error(f"ベクトル化失敗: {embed_response.text}")
                else:
                    st.error(f"チャンキング失敗: {chunk_response.text}")


# --- Main Content ---
if not st.session_state.text:
    st.info("サイドバーでPDFファイルをアップロードし、設定を行ってください。")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["チャット", "評価", "一括評価", "比較"])

    with tab1:
        st.header("ドキュメントとチャット")

        for author, message in st.session_state.chat_history:
            with st.chat_message(author):
                st.markdown(message)

        if prompt := st.chat_input("ドキュメントについて質問を入力してください"):
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("考え中..."):
                payload = {"query": prompt, "llm_model": st.session_state.llm_model, "embedding_model": st.session_state.embedding_model}
                try:
                    response = requests.post(f"{BACKEND_URL}/query/", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        answer = result['answer']
                        st.session_state.chat_history.append(("assistant", answer))
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                    else:
                        st.error(f"クエリ失敗: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"接続エラー: {e}")

    with tab2:
        st.header("RAG評価")
        st.markdown("RAGパイプラインの性能を評価するための質問セットを定義します。")

        # --- 自動生成QAセット優先で利用 ---
        auto_questions = st.session_state.get('qa_questions', None)
        auto_answers = st.session_state.get('qa_answers', None)
        if auto_questions:
            st.markdown("#### アップロードPDFから自動生成された質問セットを利用します")
            questions = auto_questions
            st.write("\n".join([f"Q{i+1}: {q}" for i, q in enumerate(questions)]))
        else:
            questions_input = st.text_area("1行に1つの質問を入力してください", height=150, 
                                           value="主なチャンキング技術には何がありますか？\nセマンティックチャンキングと再帰的文字分割の違いは何ですか？")
            questions = [q.strip() for q in questions_input.split('\n') if q.strip()]

        if st.button("評価を実行"):
            if not questions:
                st.warning("最低1つの質問を入力してください。")
            else:
                with st.spinner("評価を実行中... これには数分かかる場合があります。"):
                    try:
                        # 1. Get answers and contexts for all questions
                        answers = []
                        contexts = []
                        # --- アップロードPDFから自動生成された回答セットがあれば利用 ---
                        if auto_answers and len(auto_answers) == len(questions):
                            import concurrent.futures
                            from concurrent.futures import ThreadPoolExecutor
                            
                            # 評価用のヘルパー関数
                            def evaluate_combination(params):
                                embedding, strategy, size, overlap = params
                                try:
                                    response = requests.post(f"{BACKEND_URL}/evaluate/", json={"embedding_model": embedding, "strategy": strategy, "chunk_size": size, "chunk_overlap": overlap})
                                    return response.json()
                                except requests.exceptions.RequestException as e:
                                    st.error(f"接続エラー: {e}")
                                    return None
                            
                            # ここに並列処理の実装を追加
                            with ThreadPoolExecutor(max_workers=4) as executor:
                                futures = []
                                for embedding in ["huggingface_bge_small", "openai"]:
                                    for strategy in ["recursive", "semantic"]:
                                        for size in [500, 1000]:
                                            for overlap in [0, 100, 200]:
                                                if size > overlap:
                                                    futures.append(executor.submit(evaluate_combination, (embedding, strategy, size, overlap)))
                                                else:
                                                    st.warning(f"chunk_size <= overlap となる不正な組み合わせは自動的に除外しました: ({embedding}, {strategy}, {size}, {overlap})")
                                                # 進捗バーの設定
                                                progress_bar = st.progress(0)
                                                total_tasks = len(futures)
                                                completed_tasks = 0
                                                for future in concurrent.futures.as_completed(futures):
                                                    result = future.result()
                                                    if result is not None:
                                                        answers.append(result['answer'])
                                                        contexts.append(result['context'])
                                                    completed_tasks += 1
                                                    progress_bar.progress(min(completed_tasks / total_tasks, 1.0))
                                                progress_bar.empty()
                                                st.session_state.evaluation_results = {"answers": answers, "contexts": contexts}
                                                st.success("評価が完了しました！")
                        else:
                            st.warning("自動生成された回答が見つからないか、質問数と回答数が一致しません。")
                    except Exception as e:
                        st.error(f"評価中にエラーが発生しました: {str(e)}")

        if st.session_state.evaluation_results:
            st.subheader("評価指標")
            st.write("評価APIの返却内容:", st.session_state.evaluation_results)  # 返却内容を確認用に表示
            # evaluation_resultsがリスト形式の場合に対応
            eval_results = st.session_state.evaluation_results
            if isinstance(eval_results, list):
                results_df = pd.DataFrame(eval_results)
            else:
                results_df = pd.DataFrame([eval_results])
            # 英語→日本語の指標名マッピング
            METRIC_JA = {
                "faithfulness": "ファクト整合性",
                "answer_relevancy": "回答関連性",
                "context_recall": "文脈再現率",
                "context_precision": "文脈適合率"
            }

            # DataFrameのカラム名も日本語に変換して表示
            results_df_ja = results_df.rename(columns=METRIC_JA)
            st.dataframe(results_df_ja)

            # Plotting
            metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
            available_metrics = [m for m in metrics if m in results_df.columns]

            if available_metrics:
                plot_df = results_df[available_metrics].T.reset_index()
                plot_df['index'] = plot_df['index'].map(METRIC_JA)
                plot_df.columns = ['指標', 'スコア']

                # Performance Ranking風 横棒グラフ
                fig_bar = px.bar(
                    plot_df,
                    y='指標',
                    x='スコア',
                    orientation='h',
                    text='スコア',
                    title="Performance Ranking",
                    labels={"指標": "Metric", "スコア": "Score"},
                )
                fig_bar.update_traces(texttemplate='%{x:.3f}', textposition='outside')
                fig_bar.update_layout(
                    title={
                        'text': "RAG Strategy Performance Ranking<br><sup>Error bars show standard deviation • * indicates statistical significance</sup>",
                        'x':0.5,
                        'xanchor': 'center'
                    },
                    font=dict(size=16),
                    height=550,
                    margin=dict(l=40, r=40, t=80, b=40),
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                if 'eval_figs' not in st.session_state:
                    st.session_state['eval_figs'] = []
                if len(st.session_state['eval_figs']) == 0 or st.session_state['eval_figs'][-1] != fig_bar:
                    st.session_state['eval_figs'].append(fig_bar)
                st.markdown('<br>', unsafe_allow_html=True)

                # RAGAS Metrics Comparison風 レーダーチャート
                import plotly.graph_objects as go
                metrics_en = ['Faithfulness', 'Answer Relevancy', 'Context Recall', 'Context Precision']
                metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
                r_values = [results_df[m].iloc[0] if m in results_df.columns else 0 for m in metrics]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=metrics_en,
                    fill='toself',
                    name='Sample'
                ))
                fig_radar.update_layout(
                    title={
                        'text': "RAGAS Metrics Comparison<br><sup>All metrics normalized to 0-1 scale</sup>",
                        'x':0.5,
                        'xanchor': 'center'
                    },
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    font=dict(size=16),
                    height=600,
                    margin=dict(l=40, r=40, t=80, b=40)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                if 'eval_figs' not in st.session_state:
                    st.session_state['eval_figs'] = []
                if len(st.session_state['eval_figs']) == 0 or st.session_state['eval_figs'][-1] != fig_radar:
                    st.session_state['eval_figs'].append(fig_radar)
                st.markdown('<br>', unsafe_allow_html=True)
            else:
                st.warning("評価指標が見つかりません。評価APIの返却内容をご確認ください。")

    with tab3:
        st.header("一括評価")
        st.markdown("Embeddingモデル・チャンク分割方式・サイズ・オーバーラップの全組み合わせで一括自動評価を行います。")

        # Embeddingモデルの複数選択
        embedding_options = {
            "bge-small（ローカル高速）": "huggingface_bge_small",
            "MiniLM（軽量・多言語）": "huggingface_miniLM",
            "OpenAI": "openai",
        }
        embedding_labels = list(embedding_options.keys())
        embedding_values = list(embedding_options.values())
        # デフォルトでbge-smallを選択
        selected_labels = st.multiselect(
            "Embeddingモデルを選択",
            embedding_labels,
            default=[embedding_labels[0]],
            key="bulk_embeddings_tab3"
        )
        selected_embeddings = [embedding_options[label] for label in selected_labels]

        # チャンク分割方式・パラメータ複数選択
        chunk_methods = st.multiselect(
            "チャンク分割方式を選択",
            ["fixed", "recursive", "semantic", "sentence", "paragraph"],
            default=["fixed", "recursive", "semantic"]
        )
        chunk_sizes = st.multiselect(
            "チャンクサイズ（文字数）",
            [128, 256, 500, 1000, 1500, 2000],
            default=[500, 1000]
        )
        chunk_overlaps = st.multiselect(
            "オーバーラップ（文字数）",
            [0, 32, 64, 100, 200, 300],
            default=[0, 100, 200]
        )
        st.caption("※Embeddingモデル・チャンク分割方式・サイズ・オーバーラップの全組み合わせで自動一括評価を実行します")

        if st.button("一括評価を実行"):
            # テキストがアップロードされているか確認
            if not st.session_state.get("text"):
                st.error("評価を実行するには、まずドキュメントをアップロードしてください。")
                st.stop()
                
            with st.spinner("一括評価を実行中..."):
                import concurrent.futures
                from concurrent.futures import ThreadPoolExecutor
                import time
                
                bulk_results = []
                invalid_combinations = []
                valid_combinations = []
                
                # 有効な組み合わせのみ抽出
                for method in chunk_methods:
                    if method in ["fixed", "recursive", "semantic"]:
                        for size in chunk_sizes:
                            for overlap in chunk_overlaps:
                                if size > overlap:
                                    valid_combinations.append((method, size, overlap))
                                else:
                                    invalid_combinations.append((method, size, overlap))
                    else:
                        # size/overlapを使わない方式は一度だけNoneで追加
                        valid_combinations.append((method, None, None))
                
                if not valid_combinations:
                    st.error("有効なチャンク戦略の組み合わせがありません。chunk_size > overlap となるように選択してください。")
                    st.stop()
                
                if invalid_combinations:
                    st.warning(f"chunk_size <= overlap となる不正な組み合わせは自動的に除外しました: {invalid_combinations}")
                
                # 進捗バー、テキスト表示、ステータス表示の設定
                progress_bar = st.progress(0)
                progress_text = st.empty()
                status_display = st.empty()
                total_tasks = len(selected_embeddings) * len(valid_combinations)
                
                # 完了タスク数を追跡するためのリスト（ミュータブルなオブジェクト）
                completed_tasks = [0]
                
                # 評価用のヘルパー関数
                def evaluate_single(emb, method, size, overlap):
                    try:
                        # テキストの存在を再確認
                        text = st.session_state.get("text")
                        qa_questions = st.session_state.get("qa_questions", [])
                        qa_answers = st.session_state.get("qa_answers", [])
                        
                        if not text:
                            st.error("評価対象のテキストが見つかりません。")
                            return None
                            
                        if not qa_questions or not qa_answers:
                            st.error("評価を実行するには、Q&Aを設定してください。")
                            return None
                            
                        payload = {
                            "embedding_model": emb,
                            "chunk_methods": [method],
                            "chunk_sizes": [size] if size is not None else [1000],
                            "chunk_overlaps": [overlap] if overlap is not None else [200],
                            "text": text,
                            "questions": qa_questions,
                            "answers": qa_answers,
                        }
                        
                        response = requests.post(
                            f"{BACKEND_URL}/bulk_evaluate/", 
                            json=payload, 
                            timeout=300
                        )
                        
                        completed_tasks[0] += 1
                        progress_bar.progress(min(completed_tasks[0] / total_tasks, 1.0))
                        progress_text.text(f"完了: {completed_tasks[0]} / {total_tasks} 件")
                        
                        if response.status_code == 200:
                            return response.json()
                        else:
                            st.error("評価に失敗しました。詳細:")
                            st.json({
                                "status_code": response.status_code,
                                "error": response.text,
                                "request_payload": payload
                            })
                            return None
                            
                    except requests.exceptions.RequestException as e:
                        st.error(f"APIリクエストエラー: {str(e)}")
                        return None
                    except json.JSONDecodeError as e:
                        st.error(f"JSONデコードエラー: {str(e)}")
                        return None
                    except Exception as e:
                        st.error(f"予期せぬエラーが発生しました: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
                        return None
                
                # デバッグ用: 並列処理を一時的に無効化
                bulk_results = []
                
                try:
                    for emb in selected_embeddings:
                        for method, size, overlap in valid_combinations:
                            # 現在の評価ステータスを表示
                            status_display.info(f"評価中: {emb}, {method}, size={size}, overlap={overlap}")
                            
                            # 評価を実行
                            result = evaluate_single(emb, method, size, overlap)
                            
                            if result:
                                bulk_results.append(result)
                                status_display.success(f"完了: {emb}, {method}, size={size}, overlap={overlap}")
                            else:
                                status_display.warning(f"スキップ: {emb}, {method}, size={size}, overlap={overlap}")
                            
                            # 少し待機（UIの更新のため）
                            import time
                            time.sleep(0.1)
                    
                    # 最終的な進捗表示
                    progress_text.success(f"完了: {total_tasks} / {total_tasks} 件")
                    
                    # 進捗バーを100%に
                    progress_bar.progress(1.0)
                    
                    # 最終的なサマリを表示
                    if bulk_results:
                        status_display.success(f"評価が完了しました！ (成功: {len(bulk_results)}件)")
                    else:
                        status_display.error("評価が完了しましたが、有効な結果は得られませんでした。")
                except Exception as e:
                    status_display.error(f"評価中にエラーが発生しました: {str(e)}")
                
                # 結果をセッションに保存
                if bulk_results:
                    try:
                        # 結果がリストのリストの場合にフラット化
                        flat_results = []
                        for result in bulk_results:
                            if isinstance(result, list):
                                flat_results.extend(result)
                            else:
                                flat_results.append(result)
                        
                        st.session_state.bulk_evaluation_results = flat_results
                    except Exception as e:
                        status_display.error(f"結果の処理中にエラーが発生しました: {str(e)}")
                else:
                    status_display.error("一括評価に失敗しました。APIレスポンスをご確認ください。")
                    
                # 結果の詳細はデバッグ用にコンソールに出力
                if bulk_results:
                    print("\n=== 評価結果サマリ ===")
                    print(f"成功: {len(bulk_results)}件")
                    for i, result in enumerate(bulk_results, 1):
                        print(f"\n--- 結果 {i} ---")
                        print(json.dumps(result, ensure_ascii=False, indent=2))

        if st.session_state.bulk_evaluation_results:
            st.subheader("一括評価結果")
            st.write("一括評価APIの返却内容:", st.session_state.bulk_evaluation_results)  # 返却内容を確認用に表示

            # 結果をDataFrameに変換
            eval_results = st.session_state.bulk_evaluation_results
            if isinstance(eval_results, list):
                results_df = pd.DataFrame(eval_results)
            else:
                results_df = pd.DataFrame([eval_results])
            
            # 必要カラム補完・ラベル列追加
            required_cols = {
                'avg_chunk_len', 'num_chunks', 'overall_score', 'chunk_strategy', 'embedding_model',
                'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'answer_correctness'
            }
            missing_cols = required_cols - set(results_df.columns)
            
            if 'chunk_method' in results_df.columns and 'chunk_strategy' not in results_df.columns:
                results_df['chunk_strategy'] = results_df['chunk_method']
                st.info('chunk_strategy列をchunk_methodから補完しました')
            
            if len(results_df) > 0:
                # 不足カラムの補完
                if missing_cols:
                    st.info(f'バブルチャート用のカラムが不足しています: {missing_cols}。自動で仮値を補完します。')
                    for col in missing_cols:
                        if col == 'chunk_strategy':
                            results_df[col] = 'unknown'
                        else:
                            results_df[col] = 0.0
                
                # ラベル列を追加（chunk_sizeがあれば含める）
                if 'chunk_size' in results_df.columns:
                    results_df['label'] = results_df['chunk_strategy'] + '-' + results_df['chunk_size'].astype(str)
                else:
                    results_df['label'] = results_df['chunk_strategy']
                
                # バブルチャートの作成（色をパフォーマンススコアに基づいて設定）
                # バブルのサイズを調整するためのスケーリングファクター
                size_scale = 20  # バブルのサイズを調整するためのスケーリングファクター
                
                fig_bubble = px.scatter(
                    results_df,
                    x="num_chunks",  # カラム名をnum_chunksに修正
                    y="avg_chunk_len",
                    size=[min(s * size_scale, 50) for s in results_df["overall_score"]],  # バブルの最大サイズを制限
                    color="overall_score",  # 色をパフォーマンススコアに基づいて設定
                    hover_name="label",
                    text="label",
                    title="Chunk Distribution vs Performance (Bulk Evaluation)",
                    labels={
                        "num_chunks": "Number of Chunks",
                        "avg_chunk_len": "Average Chunk Size (characters)",
                        "overall_score": "Performance Score"
                    },
                    color_continuous_scale=px.colors.sequential.Viridis,
                    color_continuous_midpoint=0.5,  # 色の中心値を0.5に設定
                )
                
                # バブルチャートのスタイルを更新
                fig_bubble.update_traces(
                    textposition='middle center',  # テキストをバブルの中央に配置
                    textfont=dict(
                        size=12,  # フォントサイズを大きく
                        color='white',  # テキストの色を白に
                        family='Arial',  # フォントファミリーを指定
                    ),
                    marker=dict(
                        line=dict(width=1, color='DarkSlateGrey'),
                        opacity=0.8  # バブルの透明度を調整
                    ),
                    textfont_size=12,  # テキストのフォントサイズ
                    textfont_color='white',  # テキストの色
                    textfont_family='Arial',  # テキストのフォント
                    texttemplate='%{text}<br>Score: %{marker.color:.2f}',  # テキストのフォーマット
                    hovertemplate=
                    '<b>%{hovertext}</b><br>' +
                    'Chunks: %{x}<br>' +
                    'Avg Size: %{y}<br>' +
                    'Score: %{marker.color:.2f}<extra></extra>',  # ホバーテキストのフォーマット
                )
                
                fig_bubble.update_layout(
                    title={
                        'text': "Chunk Distribution vs Performance (Bulk Evaluation)<br><sup>Bubble size = performance</sup>",
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    coloraxis_colorbar=dict(title="Performance Score"),
                    font=dict(size=16),
                    height=600,
                    margin=dict(l=40, r=40, t=80, b=40)
                )
                
                # バブルチャートを表示
                st.plotly_chart(fig_bubble, use_container_width=True)
                st.markdown('<br>', unsafe_allow_html=True)
                
                # レーダーチャートの表示
                fig_radar = go.Figure()
                metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
                metrics_labels = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision", "Answer Correctness"]
                
                # モデルが1つの場合でもレーダーチャートを表示
                models = results_df['embedding_model'].unique() if 'embedding_model' in results_df.columns else ["default"]
                
                for model in models:
                    model_data = results_df[results_df['embedding_model'] == model] if 'embedding_model' in results_df.columns else results_df
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[model_data[m].mean() if m in model_data.columns else 0.5 for m in metrics],
                        theta=metrics_labels,
                        fill='toself',
                        name=f"{model}-{model_data['chunk_size'].iloc[0] if 'chunk_size' in model_data.columns else ''}" if len(models) > 1 else "Bulk Evaluation"
                    ))
                
                fig_radar.update_layout(
                    title={
                        'text': "RAGAS Metrics Comparison (Bulk Evaluation)<br><sup>All metrics normalized to 0-1 scale</sup>",
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=len(models) > 1,
                    font=dict(size=16),
                    height=600,
                    margin=dict(l=40, r=40, t=80, b=40)
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown('<br>', unsafe_allow_html=True)
                
                # 結果をDataFrameに変換
            results_df = pd.DataFrame(st.session_state.bulk_evaluation_results)

            import plotly.graph_objects as go  # レーダーチャート用
            # --- 評価方法（チャンク方式）ごとの違いをまとめて可視化 ---
            st.subheader("評価方法ごとの比較（チャンク戦略単位で集約）")
            # バックエンドの返却値カラム名に合わせて修正
            agg_cols = ["overall_score", "faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness", "avg_chunk_len", "num_chunks"]
            missing_cols = [col for col in agg_cols if col not in results_df.columns]
            if missing_cols:
                st.warning(f"集約グラフ描画に必要なカラムが不足しています: {missing_cols}。バックエンドの返却値・バージョンを確認してください。")
            else:
                agg_df = results_df.groupby("chunk_strategy")[agg_cols].mean().reset_index()

                # 1. 水平棒グラフ（総合スコア）
                fig_bar = px.bar(
                    agg_df,
                    y="chunk_strategy",
                    x="overall_score",
                    orientation="h",
                    text="overall_score",
                    title="Performance Ranking",
                    labels={"chunk_strategy": "Chunking Strategy", "overall_score": "Overall Performance Score"},
                )
                fig_bar.update_traces(texttemplate='%{x:.3f}', textposition='outside')
                fig_bar.update_layout(
                    title={
                        'text': "RAG Strategy Performance Ranking<br><sup>Error bars show standard deviation • * indicates statistical significance</sup>",
                        'x':0.5,
                        'xanchor': 'center'
                    },
                    font=dict(size=16),
                    height=550,
                    margin=dict(l=40, r=40, t=80, b=40),
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                if 'eval_figs' not in st.session_state:
                    st.session_state['eval_figs'] = []

# --- モデルごとの比較 ---
if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
    st.subheader("モデルごとの比較")
    
    # 評価結果をDataFrameに変換
    eval_results = st.session_state.evaluation_results
    if isinstance(eval_results, list):
        results_df = pd.DataFrame(eval_results)
    else:
        results_df = pd.DataFrame([eval_results])
    
    score_metrics = [
        ("overall_score", "総合スコア(加重平均)"),
        ("faithfulness", "ファクト整合性"),
        ("answer_relevancy", "回答関連性"),
        ("context_recall", "文脈再現率"),
        ("context_precision", "文脈適合率"),
        ("answer_correctness", "回答正確性")
    ]
    
    for metric, label in score_metrics:
        if metric in results_df.columns:
            st.markdown(f"#### {label}：モデル・チャンク戦略別の比較")
            # 2カラムの幅比率を調整しグラフが広くなるように
            col1, col2 = st.columns([3,2])  # 左を広めに
            
            with col1:
                st.markdown(f"##### モデルごとに評価方法別の比較")
                fig1 = px.bar(
                    results_df,
                    x="chunk_strategy",
                    y=metric,
                    color="embedding_model",
                    barmode="group",
                    title=f"{label}：モデルごとに評価方法別の比較",
                    labels={"chunk_strategy": "評価方法（チャンク方式）", "embedding_model": "モデル", metric: label}
                )
                fig1.update_layout(height=600, margin=dict(l=40, r=40, t=80, b=40), legend=dict(font=dict(size=16)))
                st.plotly_chart(fig1, use_container_width=True)
                
                if 'eval_figs' not in st.session_state:
                    st.session_state['eval_figs'] = []
                if len(st.session_state['eval_figs']) == 0 or st.session_state['eval_figs'][-1] != fig1:
                    st.session_state['eval_figs'].append(fig1)
                
                st.markdown('<br>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"##### 評価方法ごとにモデル別の比較")
                fig2 = px.bar(
                    results_df,
                    x="embedding_model",
                    y=metric,
                    color="chunk_strategy",
                    barmode="group",
                    title=f"{label}：評価方法ごとにモデル別の比較",
                    labels={"embedding_model": "モデル", "chunk_strategy": "評価方法（チャンク方式）", metric: label}
                )
                # グラフ2も同様にゆとりを持たせる
                fig2.update_layout(height=600, margin=dict(l=40, r=40, t=80, b=40), legend=dict(font=dict(size=16)))
                st.plotly_chart(fig2, use_container_width=True)
                if 'eval_figs' not in st.session_state:
                    st.session_state['eval_figs'] = []
                if len(st.session_state['eval_figs']) == 0 or st.session_state['eval_figs'][-1] != fig2:
                    st.session_state['eval_figs'].append(fig2)
                st.markdown('<br>', unsafe_allow_html=True)
            # レーダーチャートや表も既存通り表示
            fig = go.Figure()
            for model in results_df['embedding_model'].unique():
                model_data = results_df[results_df['embedding_model'] == model]
                fig.add_trace(go.Scatterpolar(
                    r=[model_data[m].mean() for m in score_metrics[1:]],
                    theta=[label for _, label in score_metrics[1:]],
                    fill='toself',
                    name=model
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title=f"{label}：モデルごとのレーダーチャート"
            )
            # グラフが潰れないように高さ・マージン・凡例サイズを調整
            fig.update_layout(height=600, margin=dict(l=40, r=40, t=80, b=40), legend=dict(font=dict(size=16)))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<br>', unsafe_allow_html=True)
            
            # --- 比較表（全指標） ---
            st.markdown("#### 全指標の比較表")
            # 日本語ラベル変換用マッピング
            METRIC_JA = {
                "overall_score": "総合スコア(加重平均)",
                "faithfulness": "ファクト整合性",
                "answer_relevancy": "回答関連性",
                "context_recall": "文脈再現率",
                "context_precision": "文脈適合率",
                "answer_correctness": "回答正確性",
                "embedding_model": "Embeddingモデル",
                "chunk_strategy": "チャンク戦略"
            }
            results_df_ja = results_df.rename(columns=METRIC_JA)
            st.dataframe(results_df_ja, use_container_width=True)

        # --- サイドバー：チャンク化のみ実行 ---
        st.sidebar.markdown("## チャンク分割（プレビュー専用）")
        chunk_method = st.sidebar.selectbox(
            "チャンク分割方式を選択",
            ["fixed", "recursive", "semantic", "sentence", "paragraph"],
            index=0
        )
        chunk_size = st.sidebar.selectbox(
            "チャンクサイズ（文字数）",
            [128, 256, 500, 1000, 1500, 2000],
            index=2
        ) if chunk_method in ["fixed", "recursive", "semantic"] else None
        chunk_overlap = st.sidebar.selectbox(
            "オーバーラップ（文字数）",
            [0, 32, 64, 100, 200, 300],
            index=0
        ) if chunk_method in ["fixed", "recursive", "semantic"] else None
        if st.sidebar.button("チャンク化を実行", help="選択したパターンでテキストをチャンク化"):
            # チャンク化API呼び出し
            import requests
            BACKEND_URL = st.secrets.get('BACKEND_URL', 'http://localhost:8000')
            payload = {
                "chunk_method": chunk_method,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "text": st.session_state.get('text', '')
            }
            try:
                response = requests.post(f"{BACKEND_URL}/chunk_text/", json=payload)
                if response.status_code == 200:
                    chunk_result = response.json()
                    st.session_state['sidebar_chunk_result'] = chunk_result
                    st.sidebar.success(f"チャンク化成功: {chunk_result.get('num_chunks', 0)}個")
                else:
                    st.sidebar.error(f"チャンク化失敗: {response.text}")
            except Exception as e:
                st.sidebar.error(f"チャンク化リクエストで例外: {e}")

        # チャンク化結果表示
        if 'sidebar_chunk_result' in st.session_state:
            chunk_result = st.session_state['sidebar_chunk_result']
            st.sidebar.markdown(f"- チャンク数: {chunk_result.get('num_chunks', 0)}")
            st.sidebar.markdown(f"- 平均チャンク長: {chunk_result.get('avg_chunk_len', 0)}")
            st.sidebar.markdown("---")
            st.sidebar.markdown("#### チャンクリスト（先頭5件）")
            for i, chunk in enumerate(chunk_result.get('chunks', [])[:5]):
                st.sidebar.markdown(f"**{i+1}**: {chunk[:80]}{'...' if len(chunk)>80 else ''}")

        # --- 一括評価ロジック ---
        if st.session_state.get('run_bulk_eval', False):
            with st.spinner("一括評価中...（全パターンを評価しています）"):
                import requests
                BACKEND_URL = st.secrets.get('BACKEND_URL', 'http://localhost:8000')
                payload = {
                    "embedding_model": st.session_state.get('selected_embedding_model', 'openai'),
                    "chunk_methods": chunk_methods,
                    "chunk_sizes": chunk_sizes,
                    "chunk_overlaps": chunk_overlaps,
                    # 他の必要なパラメータもここに追加
                }
                try:
                    response = requests.post(f"{BACKEND_URL}/bulk_evaluate/", json=payload)
                    if response.status_code == 200:
                        results = response.json()
                        st.session_state['bulk_eval_results'] = results
                        st.success("一括評価が完了しました！")
                    else:
                        st.error(f"一括評価に失敗: {response.text}")
                except Exception as e:
                    st.error(f"一括評価リクエストで例外: {e}")
                st.session_state['run_bulk_eval'] = False

        # --- 一括評価結果をグラフ・表に反映 ---
        # --- 一括評価結果の表示部（全指標網羅・日本語ラベル対応）---
        if 'bulk_eval_results' in st.session_state:
            import pandas as pd
            st.markdown("### 一括評価結果（全組み合わせ）")
            results_df = pd.DataFrame(st.session_state['bulk_eval_results'])

            # 必要な指標カラム一覧
            required_cols = [
                "embedding_model", "chunk_method", "chunk_size", "chunk_overlap",
                "overall_score", "faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness", "avg_chunk_len", "num_chunks"
            ]
            # 欠損カラムを0.0で補完
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            if missing_cols:
                st.warning(f"一括評価結果に不足している指標カラムがあります: {missing_cols}。0.0で補完します。バックエンドのバージョンやAPIレスポンスもご確認ください。")
                for col in missing_cols:
                    results_df[col] = 0.0
            # カラム順を統一
            results_df = results_df[required_cols]

            # --- 日本語ラベル変換 ---
            METRIC_JA = {
                "embedding_model": "Embeddingモデル",
                "chunk_method": "チャンク方式",
                "chunk_size": "チャンクサイズ",
                "chunk_overlap": "オーバーラップ",
                "overall_score": "総合スコア",
                "faithfulness": "ファクト整合性",
                "answer_relevancy": "回答関連性",
                "context_recall": "文脈再現率",
                "context_precision": "文脈適合率",
                "answer_correctness": "回答正確性",
                "avg_chunk_len": "平均チャンク長",
                "num_chunks": "チャンク数"
            }
            results_df_ja = results_df.rename(columns=METRIC_JA)

            # --- DataFrameを日本語ラベルで表示 ---
            st.dataframe(results_df_ja, use_container_width=True)

            # --- 以降の全グラフ・比較表・PDF出力のデータソースもresults_df_jaに統一 ---
            # 既存のscore_metricsや比較グラフもresults_df_jaを使う
            score_metrics = [
                (col, METRIC_JA.get(col, col)) for col in ["overall_score", "faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
            ]
            # グラフ描画

            for metric, label in score_metrics:
                if metric in results_df.columns:
                    st.markdown(f"#### {label}：モデル・チャンク戦略・サイズ・オーバーラップ別の比較")
                    # 2カラムの幅比率を調整しグラフが広くなるように
                    col1, col2 = st.columns([3,2])  # 左を広めに
                    with col1:
                        st.markdown(f"##### モデルごとに評価方法・サイズ・オーバーラップ別の比較")
                        fig1 = px.bar(
                            results_df,
                            x="chunk_method",
                            y=metric,
                            color="embedding_model",
                            barmode="group",
                            facet_col="chunk_size",
                            facet_row="chunk_overlap",
                            title=f"{label}：モデルごとに評価方法・サイズ・オーバーラップ別の比較",
                            labels={"chunk_method": "チャンク方式", "embedding_model": "モデル", metric: label}
                        )
                        # グラフ1も同様にゆとりを持たせる
                        fig1.update_layout(height=600, margin=dict(l=40, r=40, t=80, b=40), legend=dict(font=dict(size=16)))
                        st.plotly_chart(fig1, use_container_width=True)
                        st.markdown('<br>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"##### 評価方法ごとにモデル・サイズ・オーバーラップ別の比較")
                        fig2 = px.bar(
                            results_df,
                            x="embedding_model",
                            y=metric,
                            color="chunk_method",
                            barmode="group",
                            facet_col="chunk_size",
                            facet_row="chunk_overlap",
                            title=f"{label}：評価方法ごとにモデル・サイズ・オーバーラップ別の比較",
                            labels={"embedding_model": "モデル", "chunk_method": "チャンク方式", metric: label}
                        )
                        # グラフ2も同様にゆとりを持たせる
                        fig2.update_layout(height=600, margin=dict(l=40, r=40, t=80, b=40), legend=dict(font=dict(size=16)))
                        st.plotly_chart(fig2, use_container_width=True)
                        st.markdown('<br>', unsafe_allow_html=True)
            # 比較表もresults_dfで
            st.markdown("#### 全指標の比較表（網羅的）")
            st.dataframe(results_df, use_container_width=True)
            # PDF出力機能もresults_dfを利用して自動生成

        # --- 操作ボタン ---
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("再評価・グラフ再描画", help="最新のデータでグラフを再描画します"):
                st.experimental_rerun()
        with col_btn2:
            if st.button("全結果をPDF出力（比較レイアウト）", help="全グラフ・比較表をPDFで一括ダウンロード"):
                st.session_state['pdf_export'] = True
        with col_btn3:
            if st.button("比較API実行", help="バックエンド比較APIを呼び出して結果を表示"):
                st.session_state['run_compare_api'] = True

        # --- PDF一括出力機能 ---
        if st.session_state.get('pdf_export', False):
            from fpdf import FPDF
            import tempfile
            import plotly.io as pio
            import os
            from PIL import Image
            import io
            st.info("PDFを生成中...")
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.add_page()
            pdf.add_font('NotoSansJPVF', '', '/app/fonts/NotoSansJP-VariableFont_wght.ttf', uni=True)
            pdf.set_font('NotoSansJPVF', '', 14)
            pdf.cell(0, 10, txt="RAG評価結果 比較レポート", ln=1, align='C')
            pdf.ln(2)
            # --- 画面に表示したグラフ（fig）のみPDFへ ---
            eval_figs = st.session_state.get('eval_figs', [])
            img_paths = []
            for fig in eval_figs:
                # タイトル取得
                title = fig.layout.title.text if hasattr(fig.layout, 'title') and fig.layout.title.text else ""
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    fig.write_image(f.name, format='png', scale=2)
                    img_paths.append((f.name, None, title))

            for left, right, label in img_paths:
                pdf.add_page()  # 1ページ1枚配置
                pdf.set_font('NotoSansJPVF', '', 14)
                pdf.cell(0, 12, txt=f"{label}", ln=1, align='C')
                y = pdf.get_y()
                img_width = pdf.w - 40
                x_center = (pdf.w - img_width) / 2
                pdf.image(left, x=x_center, y=y, w=img_width)
                pdf.ln(img_width * 0.6 + 20)

            pdf.add_page()
            pdf.set_font('NotoSansJPVF', '', 12)
            pdf.cell(0, 10, txt="全指標の比較表", ln=1)
            table_cols = ["embedding_model", "chunk_strategy"] + [m for m, _ in score_metrics]
            table_df = results_df[table_cols].copy()
            col_width = (pdf.w - 20) / len(table_df.columns)
            for col in table_df.columns:
                pdf.cell(col_width, 8, str(col), border=1)
            pdf.ln()
            for _, row in table_df.iterrows():
                for val in row:
                    pdf.cell(col_width, 8, str(val), border=1)
                pdf.ln()
            for left, right, _ in img_paths:
                try:
                    os.remove(left)
                    os.remove(right)
                except Exception:
                    pass
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            st.success("PDF生成が完了しました！")
            st.download_button(
                label="PDFダウンロード",
                data=pdf_bytes,
                file_name="rag_evaluation_report.pdf",
                mime="application/pdf"
            )
            st.session_state['pdf_export'] = False

        # --- 比較タブ ---
        with tab4:
            st.header("比較")
            st.markdown("一括評価結果を比較します。")

            # 比較するモデルと戦略の選択
            models = ["ollama_llama2", "openai"]
            strategies = ["rag", "ragas"]

            selected_models = st.multiselect("モデルを選択", models, key="compare_models_tab4_main")
            selected_strategies = st.multiselect("戦略を選択", strategies, key="compare_strategies_tab4_main")
        selected_models = st.multiselect("モデルを選択", models, key="compare_models_tab4")
        selected_strategies = st.multiselect("戦略を選択", strategies, key="compare_strategies_tab4")

        if st.button("比較を実行"):
            with st.spinner("比較を実行中..."):
                # QAセット必須チェック
                if st.session_state.get("text") and st.session_state.get("qa_questions") and st.session_state.get("qa_answers"):
                    payload = {"models": selected_models, "strategies": selected_strategies,
                               "text": st.session_state.text,
                               "questions": st.session_state.qa_questions,
                               "answers": st.session_state.qa_answers}
                    response = requests.post(f"{BACKEND_URL}/compare/", json=payload)
                    if response.status_code == 200:
                        st.success("比較が完了しました！")
                        st.write("比較APIの返却内容:", response.json())  # 返却内容を確認用に表示
                    else:
                        st.error(f"比較に失敗しました: {response.text}")
                else:
                    st.warning("PDFアップロードとQA自動生成を先に実施してください。比較は行いません。")

