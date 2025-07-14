from datetime import datetime
from pytz import timezone

def jst_now_str():
    return datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')

import os
import json
import streamlit as st
import streamlit.components.v1 as components
import base64
import json
import pandas as pd
import plotly.express as px
import requests
from typing import List, Dict, Any, Optional, Tuple, Literal, Union
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go  # レーダーチャート等で使用
from openai import OpenAI
from dotenv import load_dotenv
import io
import zipfile
from datetime import datetime
import tempfile
import shutil
import traceback

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
    オーバーラップごとの評価指標を比較するグラフを表示する
    
    Args:
        results_df: 評価結果のDataFrame
    """
    # 必要なカラムが存在するか確認
    required_columns = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'overall_score']
    available_metrics = [col for col in required_columns if col in results_df.columns]
    
    if not available_metrics:
        st.warning("比較可能な評価指標が見つかりません。")
        return
    
    # オーバーラップ情報がなければコンテキストから計算を試みる
    if 'overlap' not in results_df.columns and 'contexts' in results_df.columns:
        try:
            results_df['overlap'] = results_df['contexts'].apply(
                lambda x: len(' '.join(x).split()) - len(set(' '.join(x).split())) 
                if x and len(x) > 0 else 0
            )
        except Exception as e:
            st.warning(f"オーバーラップ情報の計算中にエラーが発生しました: {str(e)}")
            return
    
    if 'overlap' not in results_df.columns:
        st.warning("オーバーラップ情報が見つかりません。比較には'overlap'列または'contexts'列が必要です。")
        return
    
    # オーバーラップの値でグループ化して平均を計算
    try:
        overlap_scores = results_df.groupby('overlap')[available_metrics].mean().reset_index()
        overlap_scores = overlap_scores.sort_values('overlap')
        
        if len(overlap_scores) <= 1:
            st.warning(f"オーバーラップの値が1種類しかありません（値: {results_df['overlap'].iloc[0]}）。比較には複数のオーバーラップ値が必要です。")
            return
            
        # 可視化
        st.subheader("オーバーラップ比較")
        
        # タブで複数の可視化を表示
        tab1, tab2 = st.tabs(["折れ線グラフ", "棒グラフ"])
        
        with tab1:
            fig_line = px.line(
                overlap_scores.melt(id_vars='overlap', var_name='metric', value_name='score'),
                x='overlap',
                y='score',
                color='metric',
                title='オーバーラップサイズごとの評価指標',
                labels={'overlap': 'オーバーラップサイズ', 'score': 'スコア', 'metric': '評価指標'},
                markers=True
            )
            fig_line.update_layout(
                xaxis_title='オーバーラップサイズ',
                yaxis_title='スコア',
                legend_title='評価指標',
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
        with tab2:
            # データをロングフォーマットに変換
            melted_data = overlap_scores.melt(id_vars='overlap', var_name='metric', value_name='score')
            
            fig_bar = px.bar(
                data_frame=melted_data,
                x='metric',
                y='score',
                color='overlap',
                barmode='group',
                title='評価指標ごとのオーバーラップ比較',
                labels={'metric': '評価指標', 'score': 'スコア', 'overlap': 'オーバーラップ'}
            )
            fig_bar.update_layout(
                xaxis_title='評価指標',
                yaxis_title='スコア',
                legend_title='オーバーラップサイズ',
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # データテーブルも表示
        with st.expander("詳細データを表示"):
            st.dataframe(overlap_scores.set_index('overlap').style.background_gradient(cmap='YlGnBu'))
            
    except Exception as e:
        st.error(f"オーバーラップ比較の表示中にエラーが発生しました: {str(e)}")
        st.error(f"エラーの詳細: {traceback.format_exc()}")

# --- グラフ保存用ユーティリティ関数 ---

def save_plotly_figure(fig, filename: str, width: int = 1200, height: int = 800, scale: float = 3.0) -> bytes:
    """
    Plotlyの図を画像データとして保存する
    
    Args:
        fig: Plotlyの図オブジェクト
        filename: 保存するファイル名（拡張子は不要）
        width: 画像の幅（ピクセル）
        height: 画像の高さ（ピクセル）
        scale: スケールファクター（解像度を上げる場合）
        
    Returns:
        bytes: 画像データ（PNG形式）
    """
    # グローバルな日本語フォントを使用して設定
    fig.update_layout(
        font_family=japanese_font,
        title_font_family=japanese_font,
        font=dict(family=f"{japanese_font}, Arial, sans-serif")
    )
    
    # 一時ファイルに保存してから読み込む（日本語文字化け対策）
    temp_dir = tempfile.mkdtemp()
    try:
        temp_file = os.path.join(temp_dir, f"{filename}.png")
        fig.write_image(temp_file, width=width, height=height, scale=scale)
        with open(temp_file, 'rb') as f:
            img_data = f.read()
        return img_data
    except Exception as e:
        st.error(f"画像の保存中にエラーが発生しました: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def create_zip_with_graphs(bulk_results: Union[dict, list], filename: str = "graphs") -> Optional[bytes]:
    """
    一括評価結果からグラフを生成し、ZIPファイルとして返す
    
    Args:
        bulk_results: 一括評価結果（辞書またはリスト）
        filename: 生成するZIPファイルのベース名（拡張子は不要）
        
    Returns:
        Optional[bytes]: 生成されたZIPファイルのバイナリデータ。エラー時はNoneを返す
    """
    # 進捗表示用のプレースホルダーを作成
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 進捗状況を更新する関数
    def update_progress(current: int, total: int, message: str) -> None:
        """進捗状況を更新する"""
        progress = int((current / total) * 100) if total > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"進捗: {current}/{total} - {message}")

    # 一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp()
    saved_files = []
    
    try:
        # 結果をDataFrameに変換
        if isinstance(bulk_results, list):
            results_df = pd.DataFrame(bulk_results)
        else:
            results_df = pd.DataFrame([bulk_results])
            
        # 処理するグラフの総数を計算
        total_graphs = 0
        if not results_df.empty:
            # バブルチャートとバーチャートの数
            total_graphs += len(results_df['embedding_model'].unique()) * 2
            # レーダーチャートの数
            if 'chunk_strategy' in results_df.columns:
                total_graphs += len(results_df['chunk_strategy'].unique())
                
        if total_graphs == 0:
            status_text.warning("生成するグラフが見つかりませんでした。")
            return None
            
        current_graph = 0
    
        # 必要なカラムが存在するか確認し、不足している場合はデフォルト値を設定
        required_cols = {
            'avg_chunk_len', 'num_chunks', 'overall_score', 'chunk_strategy', 'embedding_model',
            'faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'answer_correctness'
        }
        
        # 不足カラムの補完
        for col in required_cols:
            if col not in results_df.columns:
                if col == 'chunk_strategy':
                    results_df[col] = 'unknown'
                else:
                    results_df[col] = 0.5
    
        # メトリクスとその日本語ラベルを定義
        metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
        metrics_jp = ["信頼性", "回答の関連性", "コンテキストの再現性", "コンテキストの正確性", "回答の正確性"]
        
        # モデルごとにデータをグループ化
        if 'embedding_model' in results_df.columns:
            model_groups = list(results_df.groupby('embedding_model'))
        else:
            model_groups = [('default', results_df)]
    
        # 各グラフを一時ディレクトリに保存
        
        # 1. バブルチャートを保存
        for model_name, model_data in model_groups:
            if not model_data.empty and 'chunk_size' in model_data.columns and 'overall_score' in model_data.columns:
                # バブルチャートの作成
                fig_bubble = px.scatter(
                    model_data,
                    x="num_chunks",
                    y="avg_chunk_len",
                    size=[min(s * 8, 20) for s in model_data["overall_score"]],  # バブルのサイズをさらに小さく調整
                    color="overall_score",
                    hover_name=model_data['chunk_strategy'] + '-' + model_data['chunk_size'].astype(str),
                    text=model_data['chunk_strategy'],
                    title=f"{model_name} - チャンク分布とパフォーマンス",
                    labels={
                        "num_chunks": "チャンク数",
                        "avg_chunk_len": "平均チャンクサイズ (文字数)",
                        "overall_score": "総合スコア"
                    },
                    color_continuous_scale=px.colors.sequential.Viridis,
                    color_continuous_midpoint=0.5,
                )
                
                # バブルチャートのスタイルを更新
                fig_bubble.update_traces(
                    # テキストは別途注釈として配置するため、バブル内のテキストは非表示に
                    texttemplate='',  # 空のテンプレートでテキストを非表示に
                    marker=dict(
                        line=dict(width=1.5, color='rgba(0,0,0,0.7)'),
                        opacity=0.8,
                        sizemode='diameter',
                        sizemin=6,  # 最小サイズをさらに小さく
                        sizeref=0.1  # サイズの感度を調整
                    ),
                    hovertemplate=
                    '<b>%{hovertext}</b><br><br>' +
                    'チャンク数: <b>%{x}</b><br>' +
                    '平均サイズ: <b>%{y}文字</b><br>' +
                    'スコア: <b>%{marker.color:.2f}</b><extra></extra>',
                    hoverlabel=dict(
                        font_size=14,
                        font_family=japanese_font,
                        bgcolor='white',
                        bordercolor='#333',
                        font_color='#333'
                    )
                )
                
                # レイアウトの調整
                fig_bubble.update_layout(
                    title={
                        'text': f"{model_name} - チャンク分布とパフォーマンス",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {
                            'size': 20,
                            'family': japanese_font,
                            'color': '#333'
                        },
                        'y': 0.95,
                        'yanchor': 'top'
                    },
                    coloraxis_colorbar=dict(
                        title=dict(
                            text='スコア',
                            font=dict(
                                size=14,
                                family=japanese_font
                            )
                        ),
                        tickfont=dict(
                            family=japanese_font,
                            size=12
                        )
                    ),
                    font=dict(
                        size=14,
                        family=japanese_font,
                        color='#333'
                    ),
                    height=600,
                    margin=dict(l=80, r=50, t=100, b=120),  # 下部の余裕を増やして注釈用のスペースを確保
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        title=dict(
                            text='チャンク数',
                            font=dict(
                                size=14,
                                family=japanese_font
                            )
                        ),
                        tickfont=dict(
                            size=12,
                            family=japanese_font
                        ),
                        gridcolor='rgba(0,0,0,0.1)',
                        showline=True,
                        linewidth=1,
                        linecolor='#ddd',
                        mirror=True
                    ),
                    yaxis=dict(
                        title=dict(
                            text='平均チャンクサイズ (文字数)',
                            font=dict(
                                size=14,
                                family=japanese_font
                            )
                        ),
                        tickfont=dict(
                            size=12,
                            family=japanese_font
                        ),
                        gridcolor='rgba(0,0,0,0.1)',
                        showline=True,
                        linewidth=1,
                        linecolor='#ddd',
                        mirror=True
                    )
                )
                
                # バブルに注釈を追加（テキストをバブルの外側に配置）
                for i, row in model_data.iterrows():
                    fig_bubble.add_annotation(
                        x=row['num_chunks'],
                        y=row['avg_chunk_len'],
                        text=row['chunk_strategy'],
                        showarrow=False,
                        yshift=10,  # バブルの上に配置
                        font=dict(
                            size=10,
                            family=japanese_font,
                            color='#333333'
                        ),
                        xanchor='center',
                        yanchor='bottom',
                        opacity=0.9
                    )
                
                # 画像として保存（高解像度で）
                current_graph += 1
                update_progress(current_graph, total_graphs, f"バブルチャートを生成中: {model_name}")
                img_data = save_plotly_figure(fig_bubble, f"bubble_chart_{model_name}", width=1200, height=800, scale=3.0)
                if img_data:
                    filepath = os.path.join(temp_dir, f"bubble_chart_{model_name}.png")
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                    saved_files.append(filepath)
        
        # 2. バーチャートを保存
        for model_name, model_data in model_groups:
            if not model_data.empty and 'chunk_strategy' in model_data.columns and 'overall_score' in model_data.columns:
                # チャンク戦略ごとのパフォーマンスを集計
                strategy_scores = model_data.groupby('chunk_strategy')['overall_score'].mean().sort_values(ascending=False)
                
                # バーチャートの作成
                # データフレームを作成
                bar_data = pd.DataFrame({
                    'strategy': strategy_scores.index,
                    'score': strategy_scores.values
                })
                
                fig_bar = px.bar(
                    data_frame=bar_data,
                    x='score',
                    y='strategy',
                    orientation='h',
                    title=f"{model_name} - チャンク戦略別パフォーマンス",
                    labels={'score': '平均スコア', 'strategy': 'チャンク戦略'},
                    color='score',
                    color_continuous_scale=px.colors.sequential.Viridis,
                )
                
                # バーの上にスコアを表示
                fig_bar.update_traces(
                    texttemplate='%{x:.3f}',
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>スコア: %{x:.3f}<extra></extra>',
                    textfont=dict(size=12, family=japanese_font, color='#333333'),
                )
                
                # レイアウトの調整
                fig_bar.update_layout(
                    title={
                        'text': f"{model_name} - チャンク戦略別パフォーマンス",
                        'x': 0.5,
                        'xanchor': 'center',
                        'y': 0.95,
                        'yanchor': 'top',
                        'font': {
                            'size': 18,
                            'family': japanese_font,
                            'color': '#333333'
                        }
                    },
                    xaxis=dict(
                        range=[0, 1.1],
                        title=dict(
                            text='平均スコア',
                            font=dict(
                                size=14,
                                family=japanese_font
                            )
                        ),
                        tickfont=dict(
                            size=12,
                            family=japanese_font
                        ),
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(0, 0, 0, 0.1)',
                        showline=True,
                        linewidth=1,
                        linecolor='gray',
                        automargin=True
                    ),
                    yaxis=dict(
                        title=dict(
                            text='チャンク戦略',
                            font=dict(
                                size=14,
                                family=japanese_font
                            )
                        ),
                        tickfont=dict(
                            size=12,
                            family=japanese_font
                        ),
                        autorange="reversed",
                        automargin=True,
                        showline=True,
                        linewidth=1,
                        linecolor='gray'
                    ),
                    coloraxis_showscale=False,
                    height=500,
                    margin=dict(l=120, r=50, t=120, b=80),  # 左マージンを増やして縦軸ラベルのため
                    font=dict(
                        size=14,
                        family=japanese_font,
                        color='#333333'
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hoverlabel=dict(
                        font_size=12,
                        font_family=japanese_font
                    )
                )
                
                # 画像として保存（高解像度で）
                current_graph += 1
                update_progress(current_graph, total_graphs, f"バーチャートを生成中: {model_name}")
                img_data = save_plotly_figure(fig_bar, f"bar_chart_{model_name}", width=1200, height=800, scale=3.0)
                if img_data:
                    filepath = os.path.join(temp_dir, f"bar_chart_{model_name}.png")
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                    saved_files.append(filepath)
        
        # 3. レーダーチャートを保存
        if 'chunk_strategy' in results_df.columns:
            chunk_strategies = results_df['chunk_strategy'].unique()
            
            for strategy in chunk_strategies:
                strategy_data = results_df[results_df['chunk_strategy'] == strategy]
                
                if not strategy_data.empty:
                    fig_radar = go.Figure()
                    
                    # 各モデルのデータを追加
                    for model_name, model_data in model_groups:
                        model_strategy_data = strategy_data[strategy_data['embedding_model'] == model_name] if 'embedding_model' in strategy_data.columns else strategy_data
                        
                        if not model_strategy_data.empty:
                            # 各メトリクスの平均値を計算
                            r_values = [model_strategy_data[m].mean() if m in model_strategy_data.columns else 0.5 for m in metrics]
                            
                            # 各点に表示するテキストを準備（値とパーセンテージ）
                            text_values = [f'{v:.2f}' for v in r_values]
                            
                            # レーダーチャートにトレースを追加
                            fig_radar.add_trace(go.Scatterpolar(
                                r=r_values,
                                theta=metrics_jp,
                                fill='toself',
                                name=model_name,
                                text=text_values,
                                textposition='top center',
                                textfont=dict(
                                    size=11,
                                    color='black',
                                    family=japanese_font
                                ),
                                hovertemplate='<b>%{theta}</b><br>スコア: %{r:.2f}<extra></extra>',
                                line=dict(width=2),
                                mode='lines+markers+text',
                                marker=dict(
                                    size=6,
                                    opacity=0.8
                                )
                            ))
                    
                        # カラーパレットを定義（視認性の高い色を選択）
                    colors = [
                        '#1f77b4',  # 青
                        '#ff7f0e',  # オレンジ
                        '#2ca02c',  # 緑
                        '#d62728',  # 赤
                        '#9467bd',  # 紫
                        '#8c564b',  # 茶色
                        '#e377c2',  # ピンク
                        '#7f7f7f',  # グレー
                        '#bcbd22',  # オリーブ
                        '#17becf'   # シアン
                    ]
                    
                    # 各トレースに色を適用し、テキストの表示位置を調整
                    for i, trace in enumerate(fig_radar.data):
                        # 各点の角度に基づいてテキストの位置をずらす
                        text_positions = []
                        for j, theta in enumerate(trace.theta):
                            # 角度に基づいて位置を調整（0-360度を0-11のインデックスにマッピング）
                            angle = (j * 30) % 360  # 30度刻みで配置（12方向）
                            
                            # 角度に応じた位置を設定
                            if 15 <= angle < 165:  # 右側
                                pos = 'top center'
                            elif 195 <= angle < 345:  # 左側
                                pos = 'bottom center'
                            else:  # 上下
                                pos = 'middle right' if angle < 180 else 'middle left'
                                
                            text_positions.append(pos)
                        
                        trace.update(
                            line=dict(
                                width=2.5,
                                color=colors[i % len(colors)]
                            ),
                            marker=dict(
                                size=8,  # マーカーを少し大きく
                                color=colors[i % len(colors)],
                                line=dict(width=1, color='black'),
                                opacity=0.9
                            ),
                            textposition=text_positions,  # 動的に設定した位置を使用
                            textfont=dict(
                                size=10,  # テキストサイズを少し小さく
                                color='black',
                                family=japanese_font
                            ),
                            mode='lines+markers+text',
                            opacity=0.8
                        )
                    
                    # レイアウトの調整
                    fig_radar.update_layout(
                        title={
                            'text': f"{strategy} - 評価メトリクスの比較",
                            'x': 0.5,
                            'xanchor': 'center',
                            'y': 0.95,  # 上部に余白を確保
                            'yanchor': 'top',
                            'font': {
                                'size': 20,
                                'family': japanese_font,
                                'color': '#333333'
                            }
                        },
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1],
                                tickfont=dict(size=11, family=japanese_font, color='#555555'),
                                tickangle=0,
                                tickformat='.1f',
                                gridwidth=1,
                                gridcolor='lightgray',
                                linecolor='gray',
                                linewidth=1,
                                showline=True
                            ),
                            angularaxis=dict(
                                rotation=90,
                                direction='clockwise',
                                tickfont=dict(size=12, family=japanese_font, color='#333333'),
                                gridwidth=1,
                                gridcolor='lightgray',
                                linecolor='gray',
                                linewidth=1,
                                showline=True
                            ),
                            bgcolor='rgba(250, 250, 250, 0.8)'
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation='h',
                            yanchor='top',
                            y=-0.15,  # 下側に配置
                            xanchor='center',
                            x=0.5,
                            font=dict(size=12, family=japanese_font, color='#333333'),
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='#DDDDDD',
                            borderwidth=1,
                            itemclick=False,
                            itemdoubleclick=False
                        ),
                        margin=dict(l=80, r=80, t=120, b=150),  # 下部の余白を増やして凡例用のスペースを確保
                        height=600,  # 高さを少し増やして余裕を持たせる
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        font=dict(family=japanese_font, color='#333333'),
                        hoverlabel=dict(
                            font_size=12,
                            font_family=japanese_font
                        )
                    )
                    
                    # グリッド線を追加
                    fig_radar.update_polars(
                        radialaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            gridwidth=1,
                            showline=True,
                            linecolor='gray',
                            linewidth=1
                        ),
                        angularaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            gridwidth=1,
                            showline=True,
                            linecolor='gray',
                            linewidth=1
                        )
                    )
                    
                    # 画像として保存（高解像度で）
                    current_graph += 1
                    update_progress(current_graph, total_graphs, f"レーダーチャートを生成中: {strategy}")
                    img_data = save_plotly_figure(fig_radar, f"radar_chart_{strategy}", width=1200, height=800, scale=3.0)
                    if img_data:
                        filepath = os.path.join(temp_dir, f"radar_chart_{strategy}.png".replace("/", "_"))
                        with open(filepath, 'wb') as f:
                            f.write(img_data)
                        saved_files.append(filepath)
        
        # ZIPファイルを作成
        if saved_files:
            update_progress(total_graphs, total_graphs, "ZIPファイルを作成中...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"{filename}_{timestamp}.zip"
            zip_path = os.path.join(temp_dir, zip_filename)
            
            try:
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for i, file in enumerate(saved_files, 1):
                        zipf.write(file, os.path.basename(file))
                        update_progress(total_graphs, total_graphs, f"ZIPに追加中: {i}/{len(saved_files)}")
                
                # ZIPファイルを読み込んで返す
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                
                if 'status_text' in locals():
                    status_text.success(f"完了！ {len(saved_files)}個のファイルをZIPに保存しました。")
                progress_bar.empty()  # 完了したらプログレスバーを消す
                return zip_data
                
            except Exception as e:
                if 'status_text' in locals():
                    status_text.error(f"ZIPファイルの作成中にエラーが発生しました: {str(e)}")
                return None
        else:
            if 'status_text' in locals():
                status_text.warning("保存するグラフがありませんでした。")
            return None
            
    except Exception as e:
        if 'status_text' in locals():
            status_text.error(f"グラフの生成中にエラーが発生しました: {str(e)}")
        return None
        
    finally:
        # 一時ファイルを削除
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"一時ファイルの削除中にエラーが発生しました: {e}")
# 環境変数を読み込み
load_dotenv()

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

# --- localStorageユーティリティ ---
def save_state_to_localstorage():
    state_keys = [
        "file_id", "text", "qa_questions", "qa_answers", "uploaded_file_name", "uploaded_file_bytes",
        "evaluation_results", "bulk_evaluation_results", "chunks", "chat_history",
        "current_evaluation", "evaluation_history", "active_tab",
        "tab1_content", "tab2_content", "tab3_content", "tab4_content"
    ]
    state = {}
    for k in state_keys:
        v = st.session_state.get(k)
        if v is not None:
            # バイト列はbase64で保存
            if k == "uploaded_file_bytes" and isinstance(v, bytes):
                state[k] = base64.b64encode(v).decode('utf-8')
            # 評価履歴とタブ内容はJSON文字列として保存
            elif k in ["evaluation_history", "bulk_evaluation_results", "tab1_content", "tab2_content", "tab3_content", "tab4_content"]:
                state[k] = json.dumps(v)
            else:
                state[k] = v
    # JSON文字列をUTF-8でエンコードしてからbase64エンコード
    js = json.dumps(state, ensure_ascii=False)
    encoded_js = base64.b64encode(js.encode('utf-8')).decode('utf-8')
    components.html(f"""
    <script>
    localStorage.setItem('rag_app_state', '{encoded_js}');
    window.parent.postMessage({{streamlitMessage: 'localStorageSaved'}}, '*');
    </script>
    """, height=0)

# --- session_stateの初期化 ---
def init_session_state():
    default_state = {
        "file_id": None,
        "text": "",
        "qa_questions": [],
        "qa_answers": [],
        "uploaded_file_name": "",
        "uploaded_file_bytes": None,
        "evaluation_results": {},  # 評価結果
        "bulk_evaluation_results": {},  # バルク評価結果
        "chunks": [],  # チャンクデータ
        "chat_history": [],  # チャット履歴
        "current_evaluation": None,  # 現在の評価セッション
        "evaluation_history": [],  # 評価履歴
        "active_tab": "tab1",  # 現在のアクティブタブ
        "tab1_content": {"chat_history": []},  # タブ1の表示内容
        "tab2_content": {},  # タブ2の表示内容
        "tab3_content": {},  # タブ3の表示内容
        "tab4_content": {},  # タブ4の表示内容
        "_localstorage_loaded": False
    }
    for k, v in default_state.items():
        if k not in st.session_state:
            st.session_state[k] = v

# --- UI Layout ---
with st.sidebar:
    # --- リセットボタン ---
    # ローカルストレージやセッション状態にデータがある場合のみ表示
    from streamlit_js_eval import streamlit_js_eval
    local_state = streamlit_js_eval(js_expressions="localStorage.getItem('rag_app_state')", key="check_localstorage")
    has_data = local_state is not None or any(
        st.session_state.get(k) not in [None, "", [], {}]
        for k in ["text", "qa_questions", "qa_answers", "uploaded_file_name", "uploaded_file_bytes"]
    )
    
    if has_data:
        if st.button("リセット（すべてクリア）"):
            # 1. localStorageをクリア
            components.html("""
            <script>
            localStorage.removeItem('rag_app_state');
            window.parent.postMessage({streamlitMessage: 'localStorageCleared'}, '*');
            </script>
            """, height=0)
            
            # 2. session_stateを初期化
            init_session_state()
            
            # 3. バックエンドのデータベースをクリア
            try:
                response = requests.post(f"{BACKEND_URL}/clear_db/")
                if response.status_code == 200:
                    st.success("すべてのデータを正常にクリアしました！")
                else:
                    st.error(f"データベースのクリアに失敗しました: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"バックエンドに接続できませんでした: {e}")
            
            # 4. ページを再読み込み
            st.rerun()
    else:
        st.warning("""
        📝 データがありません
        
        PDFをアップロードしてからリセットできます。
        """)

    # --- データベース初期化ボタン ---
    if st.button("データベースのみ初期化"):
        try:
            response = requests.post(f"{BACKEND_URL}/clear_db/")
            if response.status_code == 200:
                st.success("データベースを正常にクリアしました！")
                st.session_state.text = ""
                st.session_state.chunks = []
                st.session_state.evaluation_results = None
                st.session_state.chat_history = []
            else:
                st.error(f"データベースのクリアに失敗しました: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"バックエンドに接続できませんでした: {e}")

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
    # モデル設定
    st.subheader("モデル設定")
    
    # 環境変数の読み込みを確認
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("警告: OPENAI_API_KEY が設定されていません。")
    else:
        st.sidebar.success("APIキーが設定されています")
        
    llm_model = st.selectbox(
        "LLMモデル",
        options=model_options,
        index=model_options.index(st.session_state.llm_model) if st.session_state.llm_model in model_options else 0,
        key="llm_model_select"
    )
    
    chat_model_options = ["gpt-4o-mini", "gpt-3.5-turbo", "llama3-70b-8192"]
    chat_model = st.selectbox(
        "チャットボットモデル",
        options=chat_model_options,
        index=0,
        key="chat_model_select"
    )
    
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

    import io
    # --- file_idがセッションにあれば状態を復元 ---
    # --- localStorageから復元 ---
    def load_state_from_localstorage():
        from streamlit_js_eval import streamlit_js_eval
        local_state = streamlit_js_eval(js_expressions="localStorage.getItem('rag_app_state')", key="load_localstorage")
        if local_state and not st.session_state.get("_localstorage_loaded", False):
            try:
                # 文字エンコーディングを確認
                if isinstance(local_state, str):
                    local_state = local_state.encode('utf-8')
                # base64デコードしてJSON文字列を復元
                js_bytes = base64.b64decode(local_state)
                js = js_bytes.decode("utf-8")
                state = json.loads(js)
                for k, v in state.items():
                    if k == "uploaded_file_bytes":
                        st.session_state[k] = base64.b64decode(v)
                    elif k in ["evaluation_history", "bulk_evaluation_results", "tab1_content", "tab2_content", "tab3_content", "tab4_content"]:
                        # 評価履歴とタブ内容はJSON文字列として保存されているので、デコード
                        st.session_state[k] = json.loads(v)
                    else:
                        st.session_state[k] = v
                st.session_state["_localstorage_loaded"] = True
            except Exception as e:
                st.warning(f"localStorage復元エラー: {e}")
                init_session_state()    # エラーが発生した場合でもsession_stateをクリア
                init_session_state()
        # localStorage取得時はwindow.postMessageで値が返るが、Streamlit標準では直接受け取れないため、
        # ここでは「アップロードや復元のたびにsave_state_to_localstorage()」を呼ぶことで永続化する

    load_state_from_localstorage()

    if "file_id" in st.session_state and not st.session_state.get("text"):
        try:
            resp = requests.get(f"{BACKEND_URL}/get_extracted/{st.session_state['file_id']}")
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["text"] = data.get("text", "")
                st.session_state["qa_questions"] = data.get("questions", [])
                st.session_state["qa_answers"] = data.get("answers", [])
                # --- ファイル名とバイト列も必ず復元 ---
                st.session_state["uploaded_file_name"] = data.get("file_name", f"{st.session_state['file_id']}.pdf")
                if "pdf_bytes_base64" in data:
                    st.session_state["uploaded_file_bytes"] = base64.b64decode(data["pdf_bytes_base64"])
                st.success(f"前回アップロード済みファイルID: {st.session_state['file_id']} のデータを復元しました")
                save_state_to_localstorage()
            else:
                st.warning("保存済みデータの復元に失敗しました。file_idをクリアします。")
                del st.session_state["file_id"]
        except Exception as e:
            st.warning(f"保存済みデータの復元に失敗: {e}")
            del st.session_state["file_id"]
    # すでにアップロード済みかどうかでUIを分岐
    if "uploaded_file_bytes" in st.session_state and "uploaded_file_name" in st.session_state:
        st.info(f"アップロード済: {st.session_state['uploaded_file_name']}")
        # 再アップロードしたい場合のリセットボタン
        if st.button("アップロードをやり直す"):
            for key in ["uploaded_file_bytes", "uploaded_file_name", "uploaded_file_size", "text", "qa_questions", "qa_answers", "file_id"]:
                if key in st.session_state:
                    del st.session_state[key]
            components.html("""
            <script>localStorage.removeItem('rag_app_state');</script>
            """, height=0)
            st.rerun()
        # ファイル内容をBytesIOで復元
        uploaded_file = io.BytesIO(st.session_state["uploaded_file_bytes"])
        uploaded_file.name = st.session_state["uploaded_file_name"]
        # まだテキストやQAがセッションに無ければPDF処理を実行
        if not st.session_state.get("text"):
            with st.spinner('PDFを処理中...'):
                files = {'file': (uploaded_file.name, uploaded_file, 'application/pdf')}
                try:
                    response = requests.post(f"{BACKEND_URL}/uploadfile/", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        if "file_id" in data:
                            st.session_state["file_id"] = data["file_id"]
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
                        save_state_to_localstorage()
                    else:
                        st.error(f"ファイル処理エラー: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"接続エラー: {e}")
        else:
            # 既にテキスト・QAがある場合は表示のみ
            st.success("PDFからテキスト・質問・回答セットは既に抽出済みです。")
            st.write("### 自動生成された質問:")
            for i, q in enumerate(st.session_state.qa_questions):
                st.write(f"Q{i+1}: {q}")
            st.write("### 自動生成された回答:")
            for i, a in enumerate(st.session_state.qa_answers):
                st.write(f"A{i+1}: {a}")
        save_state_to_localstorage()
    else:
        # まだアップロードされていない場合はfile_uploaderを表示
        uploaded_file = st.file_uploader("PDFをアップロード", type=["pdf"])
        if uploaded_file is not None:
            st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.session_state["uploaded_file_size"] = uploaded_file.size
            save_state_to_localstorage()
            st.rerun()

# メインコンテンツのタブ定義
tab1, tab2, tab3, tab4, tab_chatbot = st.tabs(["チャンキング設定", "評価", "一括評価", "比較", "チャットボット"])

# タブ1: チャンキング設定
with tab1:
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
                        # --- メインコンテンツ ---
                        if not st.session_state.text:
                            st.info("サイドバーでPDFファイルをアップロードし、設定を行ってください。")

# タブ2: 評価
# PDFがアップロードされていない場合はチャット画面のみ表示
if 'uploaded_file_bytes' not in st.session_state:
    st.warning("PDFをアップロードしてください。")
    st.stop()

with tab2:
    st.header("評価の実行と結果")
    
    # 評価実行セクション
    with st.expander("評価を実行", expanded=True):
        st.subheader("評価の実行")
        
        # 質問と回答の入力
        questions = st.text_area("評価する質問を入力（1行に1つ）", 
                              value="\n".join(st.session_state.get('qa_questions', [])), 
                              height=150,
                              help="評価したい質問を1行ずつ入力してください")
        
        answers = st.text_area("回答を入力（1行に1つ、質問と順番を合わせてください）", 
                             value="\n".join(st.session_state.get('qa_answers', [])), 
                             height=150,
                             help="質問に対する回答を1行ずつ入力してください")
        
        # 評価実行ボタン
        if st.button("評価を実行", key="evaluate_button_evaluation_tab"):
            questions = [q.strip() for q in questions.split('\n') if q.strip()]
            answers = [a.strip() for a in answers.split('\n') if a.strip()]
            
            if not questions:
                st.warning("評価する質問を入力してください。")
            elif len(questions) != len(answers):
                st.warning("質問と回答の数が一致しません。")
            else:
                with st.spinner("評価を実行中... しばらくお待ちください。"):
                    try:
                        evaluation_payload = {
                            "questions": questions,
                            "answers": answers,
                            "contexts": ["" for _ in questions],
                            "model": st.session_state.llm_model
                        }
                        
                        response = requests.post(
                            f"{BACKEND_URL}/evaluate/", 
                            json=evaluation_payload
                        )
                        
                        if response.status_code == 200:
                            st.session_state.evaluation_results = response.json()
                            st.session_state.qa_questions = questions
                            st.session_state.qa_answers = answers
                            st.success("評価が完了しました！")
                            st.rerun()
                        else:
                            st.error(f"評価の実行中にエラーが発生しました: {response.text}")
                    except Exception as e:
                        st.error(f"評価の実行中にエラーが発生しました: {str(e)}")
    
    # 評価結果表示セクション
    st.subheader("評価結果")
    if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
        eval_results = st.session_state.evaluation_results
        
        # 評価結果を表形式で表示
        st.subheader("評価結果サマリー")
        
        # スコアの表示
        if 'scores' in eval_results:
            scores = eval_results['scores']
            st.write("### スコア")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ファクト整合性", f"{scores.get('faithfulness', 0):.2f}")
            with col2:
                st.metric("回答関連性", f"{scores.get('answer_relevancy', 0):.2f}")
            with col3:
                st.metric("文脈再現率", f"{scores.get('context_recall', 0):.2f}")
            with col4:
                st.metric("文脈適合率", f"{scores.get('context_precision', 0):.2f}")
        
        # 詳細な評価結果を表示
        if 'results' in eval_results:
            st.write("### 質問ごとの詳細")
            for i, result in enumerate(eval_results['results']):
                with st.expander(f"質問 {i+1}: {result.get('question', '')}"):
                    st.write(f"**質問**: {result.get('question', '')}")
                    st.write(f"**回答**: {result.get('answer', '')}")
                    st.write(f"**スコア**: {result.get('score', 'N/A')}")
                    if 'details' in result:
                        st.json(result['details'])
    else:
        st.info("評価結果がありません。上記のフォームから評価を実行してください。")

# 一括評価タブ
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

    if st.button("一括評価を実行", key="bulk_evaluate_button_2"):
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