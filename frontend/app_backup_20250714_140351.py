from datetime import datetime
from pytz import timezone
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import re
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv
import base64
import io
import zipfile
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import matplotlib.font_manager as fm
import tempfile
import shutil
import matplotlib.pyplot as plt
from openai import OpenAI

def jst_now_str():
    return datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')

# ---# 日本語フォントの設定
def get_japanese_font():
    """利用可能な日本語フォントを検出して返す"""
    try:
        # 一般的な日本語対応フォントの優先順位
        font_preferences = [
            'IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'Noto Sans JP',
            'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Meiryo', 'MS Gothic',
            'Yu Gothic', 'TakaoGothic', 'VL Gothic', 'Arial Unicode MS', 'sans-serif'
        ]
        
        # 利用可能なフォントを取得
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 利用可能なフォントから優先順に選択
        for font in font_preferences:
            if any(font.lower() in f.lower() for f in available_fonts):
                return font
        
        # デフォルトのサンセリフフォント
        return 'sans-serif'
    except Exception as e:
        print(f"フォント検出エラー: {e}")
        return 'sans-serif'

# グローバルな日本語フォントを設定
japanese_font = get_japanese_font()

def plot_overlap_comparison(results_df: pd.DataFrame) -> None:
    """
    オーバーラップサイズごとの評価指標を可視化する関数
    
    Args:
        results_df (pd.DataFrame): 評価結果が格納されたDataFrame
    """
    # セッション状態の初期化
    if 'download_charts' not in st.session_state:
        st.session_state.download_charts = False
        
    # 必要なメトリクスを定義
    required_columns = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'answer_correctness', 'overall_score']
    available_metrics = [col for col in required_columns if col in results_df.columns]
    
    if not available_metrics:
        st.warning("比較可能な評価指標が見つかりません。")
        return
        
        # オーバーラップ情報を準備
        if 'overlap' not in results_df.columns and 'chunk_overlap' in results_df.columns:
            results_df['overlap'] = results_df['chunk_overlap']
        elif 'overlap' not in results_df.columns and 'contexts' in results_df.columns:
            try:
                results_df['overlap'] = results_df['contexts'].apply(
                    lambda x: len(' '.join(x).split()) - len(set(' '.join(x).split())) if x and len(x) > 0 else 0
                )
            except Exception as e:
                st.warning(f"オーバーラップ情報の計算中にエラーが発生しました: {str(e)}")
                return
        
        if 'overlap' not in results_df.columns:
            st.warning("オーバーラップ情報が見つかりません。比較には'overlap'列または'chunk_overlap'列が必要です。")
            return
        
        # グループ化に使用するカラムを決定
        group_cols = ['overlap']
        if 'embedding_model' in results_df.columns:
            group_cols.append('embedding_model')
        if 'chunk_strategy' in results_df.columns:
            group_cols.append('chunk_strategy')
        if 'chunk_size' in results_df.columns:
            group_cols.append('chunk_size')
        
        # データを集計
        overlap_scores = results_df.groupby(group_cols)[available_metrics].mean().reset_index()
        
        if len(overlap_scores) <= 1:
            st.warning(f"オーバーラップの値が1種類しかありません（値: {results_df['overlap'].iloc[0]}）。比較には複数のオーバーラップ値が必要です。")
            return
        
    # ダウンロード用の一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_files = []
        
        # タブで複数の可視化を表示
        tab1, tab2, tab3 = st.tabs(["折れ線グラフ", "ヒートマップ", "最適値サマリー"])
        
        with tab1:
            # ダウンロードボタンを配置
            if st.button("📥 グラフをダウンロード", key="download_charts"):
                st.session_state.download_charts = True
                
                # メトリクスごとに個別のグラフを作成
            for metric in available_metrics:
                st.subheader(f"{metric} の比較")
                
                # モデルとチャンクサイズの両方の情報がある場合
                if 'embedding_model' in group_cols and 'chunk_size' in group_cols:
                    models = overlap_scores['embedding_model'].unique()
                    chunk_sizes = sorted(overlap_scores['chunk_size'].unique())
                    
                    # モデルごとのタブを作成
                    model_tabs = st.tabs([f"{model}" for model in models])
                    
                    for tab_idx, model in enumerate(models):
                        with model_tabs[tab_idx]:
                            model_data = overlap_scores[overlap_scores['embedding_model'] == model]
                            
                            # チャンクサイズごとに異なる色を割り当て
                            colors = px.colors.qualitative.Plotly
                            
                            # グラフを作成
                            fig = go.Figure()
                            
                            # 各チャンクサイズのデータを追加
                            for i, chunk_size in enumerate(chunk_sizes):
                                size_data = model_data[model_data['chunk_size'] == chunk_size]
                                if len(size_data) > 0:
                                    color_idx = i % len(colors)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=size_data['overlap'],
                                        y=size_data[metric],
                                        name=f"チャンクサイズ: {chunk_size}",
                                        mode='lines+markers',
                                        line=dict(width=3, color=colors[color_idx]),
                                        marker=dict(size=10, color=colors[color_idx]),
                                        hovertemplate=f'<b>{model} (チャンク: {chunk_size})</b><br>オーバーラップ: %{{x}}<br>スコア: %{{y:.3f}}<extra></extra>',
                                        showlegend=True
                                    ))
                            
                            # レイアウトを設定
                            fig.update_layout(
                                title=f"{model} - チャンクサイズ別比較",
                                xaxis_title="オーバーラップサイズ (トークン数)",
                                yaxis_title=f"{metric} スコア (0-1)",
                                template='plotly_white',
                                height=400,
                                margin=dict(l=50, r=50, t=80, b=50),
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1,
                                    bgcolor='rgba(255,255,255,0.9)',
                                    bordercolor='rgba(0,0,0,0.2)',
                                    borderwidth=1
                                ),
                                xaxis=dict(
                                    showgrid=True,
                                    gridwidth=1,
                                    gridcolor='rgba(0,0,0,0.1)'
                                ),
                                yaxis=dict(
                                    range=[0, 1.05],
                                    showgrid=True,
                                    gridwidth=1,
                                    gridcolor='rgba(0,0,0,0.1)'
                                )
                            )
                            
                            # グラフを表示
                            # グラフを表示
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # ダウンロード用に保存
                            if st.session_state.download_charts:
                                if 'embedding_model' in group_cols and 'chunk_size' in group_cols:
                                    model_name = model.replace(" ", "_").replace("/", "-")
                                    filename = f"{metric}_model_{model_name}.png"
                                else:
                                    filename = f"{metric}.png"
                                
                                img_bytes = fig.to_image(format="png")
                                filepath = os.path.join(tmp_dir, filename)
                                with open(filepath, "wb") as f:
                                    f.write(img_bytes)
                                download_files.append(filepath)
                
                # モデル情報のみある場合
                elif 'embedding_model' in group_cols:
                    # モデルごとに異なる色を割り当て
                    colors = px.colors.qualitative.Plotly
                    
                    # グラフを作成
                    fig = go.Figure()
                    
                    # 各モデルのデータを追加
                    for i, model in enumerate(overlap_scores['embedding_model'].unique()):
                        model_data = overlap_scores[overlap_scores['embedding_model'] == model]
                        color_idx = i % len(colors)
                        
                        fig.add_trace(go.Scatter(
                            x=model_data['overlap'],
                            y=model_data[metric],
                            name=model,
                            mode='lines+markers',
                            line=dict(width=3, color=colors[color_idx]),
                            marker=dict(size=10, color=colors[color_idx]),
                            hovertemplate=f'<b>{model}</b><br>オーバーラップ: %{{x}}<br>スコア: %{{y:.3f}}<extra></extra>'
                        ))
                    
                    # レイアウトを設定
                    fig.update_layout(
                        xaxis_title="オーバーラップサイズ (トークン数)",
                        yaxis_title=f"{metric} スコア (0-1)",
                        template='plotly_white',
                        height=400,
                        margin=dict(l=50, r=50, t=50, b=50),
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='right',
                            x=1,
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='rgba(0,0,0,0.2)',
                            borderwidth=1
                        ),
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(0,0,0,0.1)'
                        ),
                        yaxis=dict(
                            range=[0, 1.05],
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(0,0,0,0.1)',
                            showline=True,
                            linewidth=2,
                            linecolor='black',
                            mirror=True,
                            ticks='outside',
                            tickwidth=2,
                        )
                    )

                    # グラフを表示
                    st.plotly_chart(fig, use_container_width=True)

                # ヒートマップの保存処理
                if st.session_state.download_charts and 'heatmap_filename' in locals():
                    filepath = os.path.join(tmp_dir, heatmap_filename)
                    img_bytes = fig.to_image(format="png")
                    with open(filepath, "wb") as f:
                        f.write(img_bytes)
                    download_files.append(filepath)
                    
                    # ダウンロード用に保存
                    filename = f"{metric}_all_models.png"
                    img_bytes = fig.to_image(format="png")
                    filepath = os.path.join(tmp_dir, filename)
{{ ... }}
                        st.error("評価対象のテキストが見つかりません。")
                        return None
                        
                    if not qa_questions or not qa_answers:
                        st.error("評価を実行するには、Q&Aを設定してください。")
                        # サマリーテーブルを表示
                # サマリーテーブルを表示
        with tab3:
            if 'summary_df' in locals() and not summary_df.empty:
                st.dataframe(
                    summary_df,
                    column_config={
                        "embedding_model": "モデル",
                        "chunk_strategy": "チャンク戦略",
                        "chunk_size": "チャンクサイズ",
                        "overlap": "最適オーバーラップ",
                        **{col: st.column_config.NumberColumn(col, format="%.3f") for col in available_metrics}
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # サマリーテーブルをCSVでダウンロード
                csv = summary_df.to_csv(index=False, encoding='utf-8-sig')
                b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
                href = f'<a href="data:text/csv;charset=utf-8-sig;base64,{b64}" download="summary_table.csv">📥 サマリーテーブルをダウンロード (CSV)</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        # ダウンロード処理
        if st.session_state.download_charts and download_files:
            with st.spinner("ダウンロードファイルを準備中..."):
                # ZIPファイルを作成
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in download_files:
                        if os.path.exists(file_path):
                            zipf.write(file_path, os.path.basename(file_path))
                
                # ダウンロードリンクを表示
                if zip_buffer.getbuffer().nbytes > 0:
                    b64 = base64.b64encode(zip_buffer.getvalue()).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="overlap_comparison_charts.zip">📥 グラフをダウンロード (ZIP)</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.session_state.download_charts = False
                    
                    response = requests.post(
                        f"{BACKEND_URL}/bulk_evaluate/", 
                        json=payload, 
{{ ... }}
                        timeout=300
                    )
                    
                    completed_tasks[0] += 1
                    progress_bar.progress(min(completed_tasks[0] / total_tasks, 1.0))
                    progress_text.text(f"完了: {completed_tasks[0]} / {total_tasks} 件")
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # レスポンスがリストの場合は最初の要素を取得
                        if isinstance(result, list):
                            if len(result) > 0:
                                result = result[0]
                            else:
                                result = {}
                        
                        # 結果が辞書でない場合は辞書に変換
                        if not isinstance(result, dict):
                            result = {}
                        
                        # メタデータを追加
                        result['embedding_model'] = emb
                        result['chunk_method'] = method
                        result['chunk_size'] = size
                        result['chunk_overlap'] = overlap
                        
                        # オーバーラップ情報が含まれていない場合はデフォルト値を設定
                        if 'overlap' not in result:
                            result['overlap'] = overlap if overlap is not None else 0
                            
                        return result
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
                    
                    # 結果の詳細をデバッグ用にコンソールに出力
                    if bulk_results:
                        print("\n=== 評価結果サマリ ===")
                        print(f"成功: {len(bulk_results)}件")
                        for i, result in enumerate(bulk_results, 1):
                            print(f"\n--- 結果 {i} ---")
                            print(json.dumps(result, ensure_ascii=False, indent=2))
                except Exception as e:
                    if 'status_text' in locals():
                        status_text.error(f"結果の処理中にエラーが発生しました: {str(e)}")


    if st.session_state.bulk_evaluation_results:
        st.subheader("一括評価結果")
        st.write("一括評価APIの返却内容:", st.session_state.bulk_evaluation_results)  # 返却内容を確認用に表示
        
        # 結果をDataFrameに変換
        eval_results = st.session_state.bulk_evaluation_results
        
        # 既存のデータ処理ロジックを維持
        if isinstance(eval_results, list):
            results_df = pd.DataFrame(eval_results)
        else:
            results_df = pd.DataFrame([eval_results])
        
        # スコア情報を展開
        if 'scores' in results_df.columns:
            # スコアが辞書形式で格納されている場合、各スコアを個別のカラムに展開
            scores_df = pd.json_normalize(results_df['scores'])
            # 元のカラムと結合（接頭辞を付けて競合を避ける）
            results_df = pd.concat([results_df.drop('scores', axis=1), scores_df.add_prefix('score_')], axis=1)
        
        # 必要カラム補完・ラベル列追加
        required_cols = {
            'avg_chunk_len', 'num_chunks', 'overall_score', 'chunk_strategy', 'embedding_model',
            'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'answer_correctness'
        }
        
        # スコアカラムの補完（score_プレフィックスが付いている場合に対応）
        for col in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'answer_correctness', 'overall_score']:
            if f'score_{col}' in results_df.columns and col not in results_df.columns:
                results_df[col] = results_df[f'score_{col}']
        
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
            
            # オーバーラップ情報を追加（chunk_overlapをデフォルト値として使用）
            if 'overlap' not in results_df.columns and 'chunk_overlap' in results_df.columns:
                results_df['overlap'] = results_df['chunk_overlap']
            
            # ラベル列を追加（chunk_sizeがあれば含める）
            if 'chunk_size' in results_df.columns:
                results_df['label'] = results_df['chunk_strategy'] + '-' + results_df['chunk_size'].astype(str)
            else:
                results_df['label'] = results_df['chunk_strategy']
        
        # オーバーラップ比較を表示
        if 'overlap' in results_df.columns and len(results_df['overlap'].unique()) > 1:
            plot_overlap_comparison(results_df)
            
            # メトリクスとその日本語ラベルを定義
            metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
            metrics_jp = ["信頼性", "回答の関連性", "コンテキストの再現性", "コンテキストの正確性", "回答の正確性"]
            
            # モデルごとにデータをグループ化
            if 'embedding_model' in results_df.columns:
                model_groups = list(results_df.groupby('embedding_model'))
            else:
                model_groups = [('default', results_df)]

            # モデルごとにバブルチャートを表示
            for model_name, model_data in model_groups:
                if not model_data.empty and 'chunk_size' in model_data.columns and 'overall_score' in model_data.columns:
                    # バブルチャートの作成
                    # 必要なカラムが存在するか確認し、存在しない場合はデフォルト値を設定
                    if 'num_chunks' not in model_data.columns:
                        model_data['num_chunks'] = 0
                    if 'avg_chunk_len' not in model_data.columns:
                        model_data['avg_chunk_len'] = 0
                    if 'overall_score' not in model_data.columns:
                        model_data['overall_score'] = 0
                    
                    # バブルサイズを計算（0除算を防ぐ）
                    bubble_sizes = [min(s * 20, 50) if pd.notnull(s) else 5 for s in model_data["overall_score"]]
                    
                    # データフレームのコピーを作成（元のデータを変更しないように）
                    plot_data = model_data.copy()
                    plot_data['bubble_size'] = bubble_sizes
                    
                    fig_bubble = px.scatter(
                        data_frame=plot_data,
                        x="num_chunks",
                        y="avg_chunk_len",
                        size="bubble_size",
                        color="overall_score",
                        hover_data={
                            "chunk_size": True,
                            "chunk_strategy": True,
                            "num_chunks": True,
                            "avg_chunk_len": ":.1f",
                            "overall_score": ".3f"
                        },
                        labels={
                            "num_chunks": "チャンク数",
                            "avg_chunk_len": "平均チャンク長",
                            "overall_score": "総合スコア"
                        },
                        color_continuous_scale=px.colors.sequential.Viridis,
                        width=1200,  # グラフの幅を拡大
                        height=800,  # グラフの高さを拡大
                        color_continuous_midpoint=0.5,
                    )
                    
                    # バブルチャートのスタイルを更新
                    fig_bubble.update_traces(
                        textposition='middle center',
                        textfont=dict(size=12, color='white', family='Arial'),
                        marker=dict(line=dict(width=1, color='DarkSlateGrey'), opacity=0.8),
                        hovertemplate=
                        '<b>%{hovertext}</b><br>' +
                        'チャンク数: %{x}<br>' +
                        '平均サイズ: %{y}文字<br>' +
                        'スコア: %{marker.color:.2f}<extra></extra>',
                    )
                    
                    fig_bubble.update_layout(
                        title={
                            'text': f"{model_name} - チャンク分布とパフォーマンス",
                            'x': 0.5,
                            'xanchor': 'center'
                        },
                        coloraxis_colorbar=dict(title="スコア"),
                        font=dict(size=14),
                        height=500,
                        margin=dict(l=40, r=40, t=80, b=40)
                    )
                    
                    st.plotly_chart(fig_bubble, use_container_width=True)
                    st.markdown('<br>', unsafe_allow_html=True)
            
            # モデルごとにバーチャートを表示
            for model_name, model_data in model_groups:
                if not model_data.empty and 'chunk_strategy' in model_data.columns and 'overall_score' in model_data.columns:
                    # チャンク戦略ごとのパフォーマンスを集計
                    strategy_scores = model_data.groupby('chunk_strategy')['overall_score'].mean().sort_values(ascending=False)
                    
                    # データフレームの作成
                    bar_data = pd.DataFrame({
                        'strategy': strategy_scores.index,
                        'score': strategy_scores.values,
                        'score_text': [f"{x:.3f}" for x in strategy_scores.values]
                    })
                    
                    # バーチャートの作成
                    fig_bar = px.bar(
                        data_frame=bar_data,
                        x='score',
                        y='strategy',
                        orientation='h',
                        text='score_text',
                        title=f"{model_name} - チャンク戦略別パフォーマンス",
                        labels={'score': '平均スコア', 'strategy': 'チャンク戦略'},
                        color='score',
                        color_continuous_scale=px.colors.sequential.Viridis,
                        width=1200,  # グラフの幅を拡大
                        height=800,  # グラフの高さを拡大
                    )
                    
                    # バーの上にスコアを表示
                    fig_bar.update_traces(
                        texttemplate='%{x:.3f}',
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>スコア: %{x:.3f}<extra></extra>',
                    )
                    
                    # レイアウトの調整
                    fig_bar.update_layout(
                        title={
                            'text': f"{model_name} - チャンク戦略別パフォーマンス",
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'size': 18}
                        },
                        xaxis=dict(range=[0, 1.1]),
                        coloraxis_showscale=False,
                        height=400,
                        margin=dict(l=100, r=40, t=100, b=40),
                        yaxis=dict(autorange="reversed"),
                        font=dict(size=14)
                    )
                    
                    # バーチャートを表示
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown('<br>', unsafe_allow_html=True)
            
            # チャンク戦略ごとにレーダーチャートを表示
            if 'chunk_strategy' in results_df.columns:
                chunk_strategies = results_df['chunk_strategy'].unique()
                
                for strategy in chunk_strategies:
                    strategy_data = results_df[results_df['chunk_strategy'] == strategy]
                    
                    if not strategy_data.empty:
                        st.subheader(f"{strategy} - 評価メトリクスの比較")
                        fig_radar = go.Figure()
                        
                        # 各モデルのデータを追加
                        for model_name, model_data in model_groups:
                            model_strategy_data = strategy_data[strategy_data['embedding_model'] == model_name] if 'embedding_model' in strategy_data.columns else strategy_data
                            
                            if not model_strategy_data.empty:
                                # 各メトリクスの平均値を計算
                                r_values = [model_strategy_data[m].mean() if m in model_strategy_data.columns else 0.5 for m in metrics]
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=r_values,
                                    theta=metrics_jp,
                                    fill='toself',
                                    name=model_name,
                                    hovertemplate='%{theta}: %{r:.2f}<extra></extra>',
                                    line=dict(width=2)
                                ))
                        
                        # レイアウトの調整
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1],
                                    tickfont=dict(size=10),
                                    tickangle=0,
                                    tickformat='.1f',
                                    gridwidth=1
                                ),
                                angularaxis=dict(
                                    rotation=90,
                                    direction='clockwise',
                                    tickfont=dict(size=12),
                                    gridwidth=1
                                ),
                                bgcolor='rgba(0,0,0,0.02)'
                            ),
                            showlegend=True,
                            legend=dict(
                                orientation='h',
                                yanchor='bottom',
                                y=1.15,
                                xanchor='center',
                                x=0.5,
                                font=dict(size=12)
                            ),
                            margin=dict(l=60, r=60, t=30, b=60),  # 上部マージンを小さく調整
                            height=500,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                        st.markdown('<br>', unsafe_allow_html=True)
            
            # 結果をDataFrameに変換
        results_df = pd.DataFrame(st.session_state.bulk_evaluation_results)
        
        # グラフをZIPファイルとしてダウンロードするボタンを追加
        st.markdown("---")
        st.subheader("グラフのエクスポート")
        
        # ダウンロードボタンのスタイルをカスタマイズ
        st.markdown("""
        <style>
            .stDownloadButton button {
                width: 100%;
                height: 3em;
                font-size: 1.2em !important;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .stDownloadButton button:hover {
                background-color: #45a049;
            }
            .stDownloadButton button:active {
                background-color: #3e8e41;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # ダウンロードボタン
        if st.button("📥 すべてのグラフをZIPファイルでダウンロード", key="download_all_graphs"):
            with st.spinner("グラフを生成してZIPファイルを作成中..."):
                zip_data = create_zip_with_graphs(st.session_state.bulk_evaluation_results, "rag_evaluation_graphs")
                
                if zip_data:
                    # ダウンロード用のリンクを生成
                    b64 = base64.b64encode(zip_data).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="rag_evaluation_graphs.zip" class="download-link">ZIPファイルをダウンロード</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("グラフのZIPファイルを生成しました。上記のリンクをクリックしてダウンロードしてください。")
                else:
                    st.error("グラフの生成中にエラーが発生しました。もう一度お試しください。")

with tab4:
    if 'uploaded_file_bytes' not in st.session_state:
        st.warning("PDFをアップロードしてください。")
        st.stop()
        
    st.header("評価結果の比較")
    
    if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
        st.subheader("評価結果の比較")
        
        # 評価結果をDataFrameに変換
        eval_results = st.session_state.evaluation_results
        if 'results' in eval_results:
            df_data = []
            for i, result in enumerate(eval_results['results']):
                row = {
                    '質問番号': i+1,
                    '質問': result.get('question', ''),
                    '回答': result.get('answer', '')
                }
                if 'details' in result:
                    for k, v in result['details'].items():
                        if isinstance(v, (int, float)):
                            row[k] = v
                df_data.append(row)
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df)
                
                # スコアの可視化
                score_cols = [col for col in df.columns if col not in ['質問番号', '質問', '回答']]
                if score_cols:
                    st.subheader("スコアの比較")
                    fig = px.bar(df, x='質問番号', y=score_cols, 
                                title="質問ごとのスコア比較",
                                labels={'value': 'スコア', 'variable': '評価項目', '質問番号': '質問番号'})
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("比較する評価結果がありません。評価または一括評価を実行してください。")
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

        if st.button("評価を実行", key="evaluate_button_evaluation"):
            if not questions:
                st.warning("最低1つの質問を入力してください。")
            else:
                with st.spinner("評価を実行中... これには数分かかる場合があります。"):
                    try:
                        # 1. 質問に対する回答とコンテキストを取得
                        answers = []
                        contexts = []
                        # --- アップロードPDFから自動生成された回答セットがあれば利用 ---
                        if auto_answers and len(auto_answers) == len(questions):
                            answers = auto_answers
                            st.success("自動生成された回答セットを使用します。")
                        else:
                            # 回答を生成するコードをここに追加
                            st.warning("自動生成された回答セットがありません。")
                            st.stop()

                        # 2. 評価を実行
                        evaluation_payload = {
                            "questions": questions,
                            "answers": answers,
                            "contexts": ["" for _ in questions],  # コンテキストは空で仮設定
                            "model": st.session_state.llm_model
                        }
                        
                        eval_response = requests.post(f"{BACKEND_URL}/evaluate/", json=evaluation_payload)
                        if eval_response.status_code == 200:
                            st.session_state.evaluation_results = eval_response.json()
                            st.success("評価が完了しました！")
                            # 評価結果を表示するタブに移動
                            st.session_state.active_tab = "評価"
                            st.rerun()
                        else:
                            st.error(f"評価の実行中にエラーが発生しました: {eval_response.text}")
                    except Exception as e:
                        st.error(f"評価の実行中にエラーが発生しました: {str(e)}")
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

# チャットボットタブ
with tab_chatbot:
    st.header("チャットボット")
    
    # チャット履歴の初期化
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # チャットメッセージの表示
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # チャット入力
    if prompt := st.chat_input("メッセージを入力..."):
        # ユーザーメッセージを表示
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 選択されたモデルで応答を生成
        response_text = ""
        with st.chat_message("assistant"):
            try:
                if not os.getenv("OPENAI_API_KEY"):
                    st.error("APIキーが設定されていません。.envファイルにOPENAI_API_KEYを設定してください。")
                    response_text = "APIキーが設定されていません。設定を確認してください。"
                
                # プロンプトを準備
                messages = [{"role": "system", "content": "あなたは親切で役立つアシスタントです。"}]
                messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages])
                
                # モデルに応じてAPIを呼び出し
                if st.session_state.chat_model in ["gpt-4o-mini", "gpt-3.5-turbo"]:
                    # OpenAI APIを使用
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        response_text = "エラー: APIキーが設定されていません。"
                    else:
                        try:
                            client = OpenAI(api_key=api_key)
                            response = client.chat.completions.create(
                                model=st.session_state.chat_model,
                                messages=messages,
                                temperature=0.7,
                                max_tokens=1000
                            )
                            response_text = response.choices[0].message.content
                        except Exception as e:
                            response_text = f"APIエラーが発生しました: {str(e)}"
                else:
                    # 無料モデルの場合（例としての実装）
                    response_text = f"{st.session_state.chat_model} からの応答: あなたのメッセージ「{prompt}」を受け取りました。\n\n（注: 無料モデルの場合はダミー応答です）"
                
                st.markdown(response_text)
                
                # アシスタントのメッセージを履歴に追加
                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_msg = f"エラーが発生しました: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
        
        # 画面を更新
        st.rerun()