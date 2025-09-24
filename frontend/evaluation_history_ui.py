# 評価履歴表示UI（Streamlit）
import streamlit as st
from http_client import http_get, http_delete
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

def show_evaluation_history(backend_url: str):
    """
    評価履歴を表示するStreamlit UI
    """
    st.header("📊 評価履歴・実験管理")
    
    # タブで機能を分割
    tab1, tab2, tab3 = st.tabs(["実験一覧", "詳細分析", "統計情報"])
    
    with tab1:
        st.subheader("実験履歴一覧")
        
        # 実験一覧を取得
        try:
            response = http_get(f"{backend_url}/api/v1/experiments/")
            if response.status_code == 200:
                data = response.json()
                experiments = data.get("experiments", [])
                
                if experiments:
                    # データフレームに変換
                    df = pd.DataFrame(experiments)
                    
                    # 日時フォーマット
                    if 'created_at' in df.columns:
                        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    # 表示用カラム選択
                    display_columns = ['id', 'experiment_name', 'file_name', 'status', 
                                     'total_combinations', 'completed_combinations', 'created_at']
                    available_columns = [col for col in display_columns if col in df.columns]
                    
                    # 実験一覧表示
                    st.dataframe(
                        df[available_columns], 
                        use_container_width=True,
                        column_config={
                            "id": "実験ID",
                            "experiment_name": "実験名",
                            "file_name": "ファイル名",
                            "status": "ステータス",
                            "total_combinations": "総組み合わせ数",
                            "completed_combinations": "完了数",
                            "created_at": "作成日時"
                        }
                    )
                    
                    # 実験削除機能
                    st.subheader("実験削除")
                    delete_exp_id = st.selectbox(
                        "削除する実験を選択",
                        options=[None] + df['id'].tolist(),
                        format_func=lambda x: "選択してください" if x is None else f"ID:{x} - {df[df['id']==x]['experiment_name'].iloc[0] if len(df[df['id']==x]) > 0 else 'Unknown'}",
                        key="delete_experiment_selectbox"
                    )
                    
                    if delete_exp_id is not None:
                        st.warning(f"実験ID {delete_exp_id} を削除しますか？この操作は元に戻せません。")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("削除実行", type="primary", key="delete_confirm_button"):
                                try:
                                    delete_response = http_delete(f"{backend_url}/api/v1/experiments/{delete_exp_id}/")
                                    if delete_response.status_code == 200:
                                        st.success("実験を削除しました")
                                        st.rerun()  # ページをリロード
                                    else:
                                        st.error(f"削除エラー: {delete_response.status_code}")
                                except Exception as e:
                                    st.error(f"削除エラー: {str(e)}")
                        with col2:
                            if st.button("キャンセル", key="delete_cancel_button"):
                                st.rerun()  # ページをリロード
                    
                    # 実験詳細表示
                    st.subheader("実験詳細")
                    selected_exp_id = st.selectbox(
                        "詳細を表示する実験を選択",
                        options=df['id'].tolist(),
                        format_func=lambda x: f"ID:{x} - {df[df['id']==x]['experiment_name'].iloc[0] if len(df[df['id']==x]) > 0 else 'Unknown'}",
                        key="experiment_detail_selectbox"
                    )
                    
                    if selected_exp_id:
                        # 実験結果を取得
                        try:
                            result_response = http_get(f"{backend_url}/api/v1/experiments/{selected_exp_id}/detailed_results/")
                            if result_response.status_code == 200:
                                result_data = result_response.json()
                                results = result_data.get("results", [])
                                
                                if results:
                                    result_df = pd.DataFrame(results)
                                    
                                    # メトリクス表示
                                    metrics_cols = ['overall_score', 'faithfulness', 'answer_relevancy', 
                                                  'context_recall', 'context_precision', 'answer_correctness']
                                    available_metrics = [col for col in metrics_cols if col in result_df.columns]
                                    
                                    if available_metrics:
                                        st.write("**評価指標**")
                                        st.dataframe(
                                            result_df[['embedding_model', 'chunk_strategy'] + available_metrics],
                                            use_container_width=True
                                        )
                                    
                                    # チャンク詳細情報を表示
                                    st.write("**チャンク詳細情報**")
                                    
                                    # チャンク詳細情報がある結果をフィルタリング
                                    results_with_chunks = [r for r in results if r.get('chunks_details')]
                                    
                                    if results_with_chunks:
                                        # チャンク詳細を表示する結果を選択
                                        chunk_result_options = [
                                            f"{r['embedding_model']} - {r['chunk_strategy']} (ID: {r['id']})"
                                            for r in results_with_chunks
                                        ]
                                        
                                        selected_chunk_result = st.selectbox(
                                            "チャンク詳細を表示する結果を選択",
                                            options=range(len(results_with_chunks)),
                                            format_func=lambda i: chunk_result_options[i],
                                            key="chunk_result_selectbox"
                                        )
                                        
                                        if selected_chunk_result is not None:
                                            selected_result = results_with_chunks[selected_chunk_result]
                                            chunks_details = selected_result['chunks_details']
                                            
                                            # チャンク戦略情報を表示
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("チャンク戦略", chunks_details.get('chunk_strategy', 'N/A'))
                                            with col2:
                                                st.metric("総チャンク数", chunks_details.get('total_chunks', 0))
                                            with col3:
                                                avg_len = selected_result.get('avg_chunk_len', 0)
                                                st.metric("平均チャンク長", f"{avg_len:.1f}" if avg_len else "N/A")
                                            
                                            # パラメータ情報を表示
                                            params = chunks_details.get('parameters', {})
                                            if params:
                                                st.write("**チャンキングパラメータ**")
                                                param_cols = st.columns(len(params))
                                                for i, (key, value) in enumerate(params.items()):
                                                    if value is not None:
                                                        with param_cols[i % len(param_cols)]:
                                                            st.metric(key.replace('_', ' ').title(), str(value))
                                            
                                            # チャンクサンプルを表示
                                            chunks_list = chunks_details.get('chunks', [])
                                            if chunks_list:
                                                st.write(f"**チャンクサンプル** (最初の{len(chunks_list)}件を表示)")
                                                
                                                # チャンク情報をデータフレームで表示
                                                chunk_df = pd.DataFrame([
                                                    {
                                                        'チャンクID': chunk.get('index', i),
                                                        '長さ': chunk.get('length', 0),
                                                        '内容プレビュー': chunk.get('content', '')[:100] + '...' if len(chunk.get('content', '')) > 100 else chunk.get('content', '')
                                                    }
                                                    for i, chunk in enumerate(chunks_list)
                                                ])
                                                
                                                st.dataframe(
                                                    chunk_df,
                                                    use_container_width=True,
                                                    column_config={
                                                        'チャンクID': st.column_config.NumberColumn('チャンクID', width='small'),
                                                        '長さ': st.column_config.NumberColumn('長さ', width='small'),
                                                        '内容プレビュー': st.column_config.TextColumn('内容プレビュー', width='large')
                                                    }
                                                )
                                                
                                                # 選択したチャンクの詳細表示
                                                if len(chunks_list) > 0:
                                                    selected_chunk_idx = st.selectbox(
                                                        "詳細を表示するチャンクを選択",
                                                        options=range(len(chunks_list)),
                                                        format_func=lambda i: f"チャンク {chunks_list[i].get('index', i)} (長さ: {chunks_list[i].get('length', 0)})",
                                                        key="chunk_detail_selectbox"
                                                    )
                                                    
                                                    if selected_chunk_idx is not None:
                                                        selected_chunk = chunks_list[selected_chunk_idx]
                                                        st.write(f"**チャンク {selected_chunk.get('index', selected_chunk_idx)} の全内容**")
                                                        
                                                        # チャンク内容をコードブロックで表示
                                                        chunk_content = selected_chunk.get('content', '')
                                                        if chunk_content:
                                                            st.code(chunk_content, language='text')
                                                        else:
                                                            st.info("チャンク内容が空です")
                                                
                                                # チャンク長の分布をグラフで表示
                                                if len(chunks_list) > 1:
                                                    st.write("**チャンク長の分布**")
                                                    chunk_lengths = [chunk.get('length', 0) for chunk in chunks_list]
                                                    
                                                    fig = px.histogram(
                                                        x=chunk_lengths,
                                                        nbins=min(20, len(chunks_list)),
                                                        title="チャンク長の分布",
                                                        labels={'x': 'チャンク長', 'y': '频度'}
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.info("チャンクサンプルがありません")
                                    else:
                                        st.info("チャンク詳細情報が保存されている結果がありません")
                                        
                                        # グラフ表示
                                        if len(result_df) > 1:
                                            st.write("**スコア比較**")
                                            fig = px.bar(
                                                result_df, 
                                                x='chunk_strategy', 
                                                y='overall_score',
                                                color='embedding_model',
                                                title="チャンク戦略別総合スコア"
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("この実験の結果データがありません。")
                            else:
                                st.error(f"実験結果取得エラー: {result_response.status_code}")
                        except Exception as e:
                            st.error(f"実験結果取得エラー: {str(e)}")
                    
                    # 実験削除機能
                    st.subheader("実験削除")
                    with st.expander("実験削除（注意）"):
                        delete_exp_id = st.selectbox(
                            "削除する実験を選択",
                            options=[None] + df['id'].tolist(),
                            format_func=lambda x: "選択してください" if x is None else f"ID:{x} - {df[df['id']==x]['experiment_name'].iloc[0] if len(df[df['id']==x]) > 0 else 'Unknown'}"
                        )
                        
                        if delete_exp_id and st.button("実験を削除", type="secondary"):
                            try:
                                delete_response = http_delete(f"{backend_url}/api/v1/experiments/{delete_exp_id}/")
                                if delete_response.status_code == 200:
                                    st.success("実験を削除しました。")
                                    st.rerun()
                                else:
                                    st.error(f"削除エラー: {delete_response.status_code}")
                            except Exception as e:
                                st.error(f"削除エラー: {str(e)}")
                else:
                    st.info("実験履歴がありません。")
            else:
                st.error(f"実験履歴取得エラー: {response.status_code}")
        except Exception as e:
            st.error(f"実験履歴取得エラー: {str(e)}")
    
    with tab2:
        st.subheader("詳細分析")
        
        # 全実験の結果を統合分析
        try:
            response = http_get(f"{backend_url}/api/v1/experiments/")
            if response.status_code == 200:
                data = response.json()
                experiments = data.get("experiments", [])
                
                if experiments:
                    # 全実験の結果を取得
                    all_results = []
                    for exp in experiments:
                        try:
                            result_response = http_get(f"{backend_url}/api/v1/experiments/{exp['id']}/detailed_results/")
                            if result_response.status_code == 200:
                                result_data = result_response.json()
                                results = result_data.get("results", [])
                                for result in results:
                                    result['experiment_id'] = exp['id']
                                    result['experiment_name'] = exp['experiment_name']
                                all_results.extend(results)
                        except:
                            continue
                    
                    if all_results:
                        all_df = pd.DataFrame(all_results)
                        
                        # モデル別性能比較
                        if 'embedding_model' in all_df.columns and 'overall_score' in all_df.columns:
                            st.write("**モデル別性能比較**")
                            model_avg = all_df.groupby('embedding_model')['overall_score'].agg(['mean', 'std', 'count']).reset_index()
                            
                            fig = px.bar(
                                model_avg, 
                                x='embedding_model', 
                                y='mean',
                                error_y='std',
                                title="モデル別平均スコア（エラーバー：標準偏差）"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # チャンク戦略別性能比較
                        if 'chunk_strategy' in all_df.columns:
                            st.write("**チャンク戦略別性能比較**")
                            chunk_avg = all_df.groupby('chunk_strategy')['overall_score'].agg(['mean', 'std', 'count']).reset_index()
                            
                            fig = px.bar(
                                chunk_avg, 
                                x='chunk_strategy', 
                                y='mean',
                                error_y='std',
                                title="チャンク戦略別平均スコア（エラーバー：標準偏差）"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 相関分析
                        numeric_cols = ['overall_score', 'faithfulness', 'answer_relevancy', 
                                      'context_recall', 'context_precision', 'answer_correctness',
                                      'avg_chunk_len', 'num_chunks']
                        available_numeric = [col for col in numeric_cols if col in all_df.columns]
                        
                        if len(available_numeric) > 1:
                            st.write("**指標間相関分析**")
                            corr_matrix = all_df[available_numeric].corr()
                            
                            fig = px.imshow(
                                corr_matrix,
                                text_auto=True,
                                aspect="auto",
                                title="評価指標間の相関係数"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("分析対象の結果データがありません。")
                else:
                    st.info("実験履歴がありません。")
            else:
                st.error(f"実験履歴取得エラー: {response.status_code}")
        except Exception as e:
            st.error(f"詳細分析エラー: {str(e)}")
    
    with tab3:
        st.subheader("統計情報")
        
        # 統計情報を取得
        try:
            response = http_get(f"{backend_url}/api/v1/experiments/statistics/")
            if response.status_code == 200:
                stats = response.json()
                
                # 基本統計
                overall_stats = stats.get("overall", {})
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("総実験数", overall_stats.get("total_experiments", 0))
                with col2:
                    st.metric("総評価結果数", overall_stats.get("total_results", 0))
                with col3:
                    avg_score = overall_stats.get("avg_overall_score", 0)
                    st.metric("平均スコア", f"{avg_score:.3f}" if avg_score else "N/A")
                with col4:
                    max_score = overall_stats.get("max_overall_score", 0)
                    st.metric("最高スコア", f"{max_score:.3f}" if max_score else "N/A")
                with col5:
                    min_score = overall_stats.get("min_overall_score", 0)
                    st.metric("最低スコア", f"{min_score:.3f}" if min_score else "N/A")
                
                # モデル別統計
                model_stats = stats.get("by_model", [])
                if model_stats:
                    st.write("**モデル別統計**")
                    model_df = pd.DataFrame(model_stats)
                    st.dataframe(
                        model_df,
                        use_container_width=True,
                        column_config={
                            "model": "モデル",
                            "count": "実行回数",
                            "avg_score": st.column_config.NumberColumn("平均スコア", format="%.3f"),
                            "max_score": st.column_config.NumberColumn("最高スコア", format="%.3f"),
                            "min_score": st.column_config.NumberColumn("最低スコア", format="%.3f")
                        }
                    )
                
                # チャンク戦略別統計
                strategy_stats = stats.get("by_strategy", [])
                if strategy_stats:
                    st.write("**チャンク戦略別統計**")
                    strategy_df = pd.DataFrame(strategy_stats)
                    st.dataframe(
                        strategy_df,
                        use_container_width=True,
                        column_config={
                            "strategy": "戦略",
                            "count": "実行回数",
                            "avg_score": st.column_config.NumberColumn("平均スコア", format="%.3f"),
                            "max_score": st.column_config.NumberColumn("最高スコア", format="%.3f"),
                            "min_score": st.column_config.NumberColumn("最低スコア", format="%.3f")
                        }
                    )
            else:
                st.error(f"統計情報取得エラー: {response.status_code}")
        except Exception as e:
            st.error(f"統計情報取得エラー: {str(e)}")

if __name__ == "__main__":
    # テスト用
    show_evaluation_history("http://localhost:8000")
