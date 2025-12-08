import os
import io
import shutil
import tempfile
from datetime import datetime
from typing import Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --- 日本語フォント関連 ---

def get_japanese_font() -> str:
    """利用可能な日本語フォントを検出して返す。"""
    try:
        font_preferences = [
            "IPAexGothic",
            "IPAGothic",
            "Noto Sans CJK JP",
            "Noto Sans JP",
            "Hiragino Sans",
            "Hiragino Kaku Gothic ProN",
            "Meiryo",
            "MS Gothic",
            "Yu Gothic",
            "TakaoGothic",
            "VL Gothic",
            "Arial Unicode MS",
            "sans-serif",
        ]
        import matplotlib.font_manager as fm

        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in font_preferences:
            if any(font.lower() in f.lower() for f in available_fonts):
                return font
        return "sans-serif"
    except Exception as e:
        print(f"フォント検出エラー: {e}")
        return "sans-serif"


japanese_font: str = get_japanese_font()


# --- オーバーラップ比較グラフ ---

def plot_overlap_comparison(results_df: pd.DataFrame) -> None:
    """オーバーラップごとの評価指標を比較するグラフを表示する。"""
    try:
        from evaluation_history_ui import create_label  # 循環参照を避けるためローカルimport

        required_columns = [
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "answer_correctness",
            "overall_score",
        ]
        available_metrics = [
            col for col in required_columns if col in results_df.columns
        ]
        if not available_metrics:
            st.warning("比較可能な評価指標が見つかりません。")
            return

        if "overlap" not in results_df.columns and "chunk_overlap" in results_df.columns:
            results_df["overlap"] = results_df["chunk_overlap"]
        elif "overlap" not in results_df.columns and "contexts" in results_df.columns:
            try:
                results_df["overlap"] = results_df["contexts"].apply(
                    lambda x: len(" ".join(x).split())
                    - len(set(" ".join(x).split()))
                    if x and len(x) > 0
                    else 0
                )
            except Exception as e:
                st.warning(f"オーバーラップ情報の計算中にエラーが発生しました: {e}")
                return

        if "overlap" not in results_df.columns:
            st.warning(
                "オーバーラップ情報が見つかりません。比較には'overlap'列または'chunk_overlap'列が必要です。"
            )
            return

        group_cols = ["overlap"]
        if "embedding_model" in results_df.columns:
            group_cols.append("embedding_model")
        if "chunk_strategy" in results_df.columns:
            group_cols.append("chunk_strategy")
        if "chunk_size" in results_df.columns:
            group_cols.append("chunk_size")

        overlap_scores = (
            results_df.groupby(group_cols)[available_metrics].mean().reset_index()
        )
        if len(overlap_scores) <= 1:
            st.warning(
                f"オーバーラップの値が1種類しかありません（値: {results_df['overlap'].iloc[0]}）。比較には複数のオーバーラップ値が必要です。"
            )
            return

        tab1, tab2, tab3 = st.tabs(["折れ線グラフ", "ヒートマップ", "最適値サマリー"])

        # --- 折れ線グラフ ---
        with tab1:
            all_figs = []
            all_tables = []

            for metric in available_metrics:
                st.subheader(f"{metric} の比較")

                if "embedding_model" in group_cols and "chunk_size" in group_cols:
                    models = overlap_scores["embedding_model"].unique()
                    chunk_sizes = sorted(overlap_scores["chunk_size"].unique())
                    model_tabs = st.tabs([f"{model}" for model in models])

                    for tab_idx, model in enumerate(models):
                        with model_tabs[tab_idx]:
                            model_data = overlap_scores[
                                overlap_scores["embedding_model"] == model
                            ]
                            colors = px.colors.qualitative.Plotly
                            fig = go.Figure()

                            for i, chunk_size in enumerate(chunk_sizes):
                                size_data_raw = model_data[
                                    model_data["chunk_size"] == chunk_size
                                ]
                                if len(size_data_raw) == 0:
                                    continue
                                size_data = (
                                    size_data_raw.groupby("overlap", as_index=False)[metric]
                                    .mean()
                                )
                                color_idx = i % len(colors)

                                if "label" in size_data_raw.columns:
                                    display_strategy = size_data_raw["label"].iloc[0]
                                else:
                                    strategy = size_data_raw["chunk_strategy"].iloc[0]
                                    if isinstance(strategy, str):
                                        base_strategy = strategy.split("-")[0].lower()
                                        if base_strategy in [
                                            "semantic",
                                            "sentence",
                                            "paragraph",
                                        ]:
                                            display_strategy = base_strategy
                                        else:
                                            display_strategy = (
                                                f"{base_strategy}-{chunk_size}"
                                            )
                                    else:
                                        display_strategy = str(strategy)

                                if isinstance(display_strategy, str) and any(
                                    s in display_strategy
                                    for s in ["semantic", "sentence", "paragraph"]
                                ):
                                    hover_text = (
                                        "<b>"
                                        + display_strategy
                                        + "</b><br>オーバーラップ: %{x}<br>スコア: %{y:.3f}<extra></extra>"
                                    )
                                else:
                                    hover_text = (
                                        "<b>"
                                        + display_strategy
                                        + f" (チャンク: {chunk_size})</b><br>オーバーラップ: %{{x}}<br>スコア: %{{y:.3f}}<extra></extra>"
                                    )

                                fig.add_trace(
                                    go.Scatter(
                                        x=size_data["overlap"],
                                        y=size_data[metric],
                                        name=display_strategy,
                                        mode="lines+markers",
                                        line=dict(width=3, color=colors[color_idx]),
                                        marker=dict(size=10, color=colors[color_idx]),
                                        hovertemplate=hover_text,
                                        showlegend=True,
                                    )
                                )

                            strategy_name = "チャンクサイズ別比較"
                            if "chunk_strategy" in model_data.columns:
                                row0 = model_data.iloc[0]
                                base_strategy = create_label(row0)
                                if base_strategy in [
                                    "semantic",
                                    "sentence",
                                    "paragraph",
                                ]:
                                    strategy_name = f"{base_strategy}戦略"
                                else:
                                    strategy_name = (
                                        f"{base_strategy}戦略 - チャンクサイズ別比較"
                                    )

                            fig.update_layout(
                                title=f"{model} - {strategy_name}",
                                xaxis_title="オーバーラップサイズ (トークン数)",
                                yaxis_title=f"{metric} スコア (0-1)",
                                template="plotly_white",
                                height=400,
                                margin=dict(l=50, r=50, t=80, b=50),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                    bgcolor="rgba(255,255,255,0.9)",
                                    bordercolor="rgba(0,0,0,0.2)",
                                    borderwidth=1,
                                ),
                                xaxis=dict(
                                    showgrid=True,
                                    gridwidth=1,
                                    gridcolor="rgba(0,0,0,0.1)",
                                ),
                                yaxis=dict(
                                    range=[0, 1.05],
                                    showgrid=True,
                                    gridwidth=1,
                                    gridcolor="rgba(0,0,0,0.1)",
                                ),
                            )

                            st.plotly_chart(fig, use_container_width=True)
                            all_figs.append((f"{metric}_{model}_chunk.png", fig))

                elif "embedding_model" in group_cols:
                    colors = px.colors.qualitative.Plotly
                    fig = go.Figure()
                    for i, model in enumerate(
                        overlap_scores["embedding_model"].unique()
                    ):
                        model_data = overlap_scores[
                            overlap_scores["embedding_model"] == model
                        ]
                        color_idx = i % len(colors)
                        fig.add_trace(
                            go.Scatter(
                                x=model_data["overlap"],
                                y=model_data[metric],
                                name=model,
                                mode="lines+markers",
                                line=dict(width=3, color=colors[color_idx]),
                                marker=dict(size=10, color=colors[color_idx]),
                                hovertemplate="<b>"
                                + model
                                + "</b><br>オーバーラップ: %{x}<br>スコア: %{y:.3f}<extra></extra>",
                            )
                        )

                    fig.update_layout(
                        xaxis_title="オーバーラップサイズ (トークン数)",
                        yaxis_title=f"{metric} スコア (0-1)",
                        template="plotly_white",
                        height=400,
                        margin=dict(l=50, r=50, t=50, b=50),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="rgba(0,0,0,0.2)",
                            borderwidth=1,
                        ),
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(0,0,0,0.1)",
                        ),
                        yaxis=dict(
                            range=[0, 1.05],
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(0,0,0,0.1)",
                            showline=True,
                            linewidth=2,
                            linecolor="black",
                            mirror=True,
                            ticks="outside",
                            tickwidth=2,
                            tickcolor="black",
                            ticklen=6,
                        ),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    fig = px.line(
                        overlap_scores,
                        x="overlap",
                        y=metric,
                        title=f"{metric} スコア",
                        labels={
                            "overlap": "オーバーラップサイズ (トークン数)",
                            metric: "スコア (0-1)",
                        },
                        markers=True,
                    )
                    fig.update_traces(line=dict(width=3), marker=dict(size=10))
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(0,0,0,0.1)",
                        ),
                        yaxis=dict(
                            range=[0, 1.05],
                            showgrid=True,
                            gridwidth=1,
                            gridcolor="rgba(0,0,0,0.1)",
                        ),
                        margin=dict(l=50, r=50, t=50, b=50),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)

            all_tables.append(("detail.csv", overlap_scores))

            with st.expander("詳細データを表示"):
                st.dataframe(
                    overlap_scores.style.background_gradient(
                        subset=available_metrics, cmap="YlGnBu"
                    ),
                    use_container_width=True,
                )

            import zipfile
            import plotly.io as pio  # noqa: F401 (for fig.to_image)

            _dl_cnt = st.session_state.get("_plot_overlap_download_cnt", 0)
            st.session_state["_plot_overlap_download_cnt"] = _dl_cnt + 1
            if st.button(
                "全グラフ・表を一括ダウンロード (zip)",
                key=f"download_zip_btn_{_dl_cnt}",
            ):
                progress_bar = st.progress(0)
                total_tasks = len(all_figs) + len(all_tables)
                current = 0
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for fname, fig in all_figs:
                        img_bytes = fig.to_image(format="png")
                        zf.writestr(fname, img_bytes)
                        current += 1
                        progress_bar.progress(current / total_tasks)
                    for tname, df in all_tables:
                        zf.writestr(
                            tname, df.to_csv(index=False, encoding="utf-8")
                        )
                        current += 1
                        progress_bar.progress(current / total_tasks)
                zip_buffer.seek(0)
                progress_bar.empty()
                st.download_button(
                    label="ダウンロード開始",
                    data=zip_buffer,
                    file_name=f"overlap_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                )

        # --- ヒートマップ ---
        with tab2:
            if "chunk_size" in group_cols and "embedding_model" in group_cols:
                for model in overlap_scores["embedding_model"].unique():
                    model_data = overlap_scores[
                        overlap_scores["embedding_model"] == model
                    ]
                    pivot_data = (
                        model_data.pivot_table(
                            index="chunk_size",
                            columns="overlap",
                            values="overall_score",
                            aggfunc="mean",
                        )
                        .sort_index(ascending=False)
                    )
                    if not pivot_data.empty:
                        fig = px.imshow(
                            pivot_data,
                            labels=dict(
                                x="オーバーラップサイズ (トークン数)",
                                y="チャンクサイズ (トークン数)",
                                color="スコア (0-1)",
                            ),
                            title=f"{model} - チャンクサイズとオーバーラップの関係",
                            color_continuous_scale="Viridis",
                            aspect="auto",
                        )
                        fig.update_layout(
                            xaxis_title="オーバーラップサイズ (トークン数)",
                            yaxis_title="チャンクサイズ (トークン数)",
                            coloraxis_colorbar_title="スコア (0-1)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # --- 最適値サマリー ---
        with tab3:
            if (
                "embedding_model" in group_cols
                and "chunk_strategy" in group_cols
                and "chunk_size" in group_cols
            ):
                best_overlaps = []
                for model in overlap_scores["embedding_model"].unique():
                    model_data = overlap_scores[
                        overlap_scores["embedding_model"] == model
                    ]
                    for strategy in model_data["chunk_strategy"].unique():
                        strategy_data = model_data[
                            model_data["chunk_strategy"] == strategy
                        ]
                        for size in strategy_data["chunk_size"].unique():
                            size_data = strategy_data[
                                strategy_data["chunk_size"] == size
                            ]
                            if not size_data.empty:
                                best_idx = size_data["overall_score"].idxmax()
                                best_overlaps.append(
                                    {
                                        "モデル": model,
                                        "チャンク化方法": strategy,
                                        "チャンクサイズ": size,
                                        "最適オーバーラップ": size_data.loc[
                                            best_idx, "overlap"
                                        ],
                                        "最高スコア": round(
                                            size_data.loc[best_idx, "overall_score"],
                                            3,
                                        ),
                                    }
                                )

                if best_overlaps:
                    summary_df = pd.DataFrame(best_overlaps)
                    st.dataframe(
                        summary_df.sort_values(
                            ["モデル", "チャンク化方法", "チャンクサイズ"]
                        ),
                        column_config={
                            "最高スコア": st.column_config.ProgressColumn(
                                "最高スコア",
                                format="%.3f",
                                min_value=0,
                                max_value=1.0,
                            )
                        },
                        use_container_width=True,
                    )
                else:
                    st.info("📄 PDFファイルをアップロードしてください")

    except Exception as e:
        import traceback

        st.error(f"オーバーラップ比較の表示中にエラーが発生しました: {e}")
        st.error(f"エラーの詳細: {traceback.format_exc()}")


# --- グラフ保存ユーティリティ ---

def save_plotly_figure(
    fig, filename: str, width: int = 1200, height: int = 800, scale: float = 3.0
) -> Optional[bytes]:
    """Plotlyの図を画像データとして保存する。"""
    fig.update_layout(
        font_family=japanese_font,
        title_font_family=japanese_font,
        font=dict(family=f"{japanese_font}, Arial, sans-serif"),
    )
    temp_dir = tempfile.mkdtemp()
    try:
        temp_file = os.path.join(temp_dir, f"{filename}.png")
        fig.write_image(temp_file, width=width, height=height, scale=scale)
        with open(temp_file, "rb") as f:
            img_data = f.read()
        return img_data
    except Exception as e:
        st.error(f"画像の保存中にエラーが発生しました: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_zip_with_graphs(
    bulk_results: Union[dict, list], filename: str = "graphs"
) -> Optional[bytes]:
    """一括評価結果からグラフを生成し、ZIPファイルとして返す。"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(current: int, total: int, message: str) -> None:
        progress = int((current / total) * 100) if total > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"進捗: {current}/{total} - {message}")

    temp_dir = tempfile.mkdtemp()
    saved_files: list[str] = []

    try:
        if isinstance(bulk_results, list):
            results_df = pd.DataFrame(bulk_results)
        else:
            results_df = pd.DataFrame([bulk_results])

        total_graphs = 0
        if not results_df.empty:
            total_graphs += len(results_df["embedding_model"].unique()) * 2
            if "chunk_strategy" in results_df.columns:
                total_graphs += len(results_df["chunk_strategy"].unique())
        if total_graphs == 0:
            status_text.warning("生成するグラフが見つかりませんでした。")
            return None

        current_graph = 0
        required_cols = {
            "avg_chunk_len",
            "num_chunks",
            "overall_score",
            "chunk_strategy",
            "embedding_model",
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "answer_correctness",
        }
        for col in required_cols:
            if col not in results_df.columns:
                if col == "chunk_strategy":
                    results_df[col] = "unknown"
                else:
                    results_df[col] = 0.5

        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "answer_correctness",
        ]
        metrics_jp = [
            "信頼性",
            "回答の関連性",
            "コンテキストの再現性",
            "コンテキストの正確性",
            "回答の正確性",
        ]

        if "embedding_model" in results_df.columns:
            model_groups = list(results_df.groupby("embedding_model"))
        else:
            model_groups = [("default", results_df)]

        # 1. バブルチャート
        for model_name, model_data in model_groups:
            if not model_data.empty and "chunk_size" in model_data.columns and "overall_score" in model_data.columns:
                fig_bubble = px.scatter(
                    model_data,
                    x="num_chunks",
                    y="avg_chunk_len",
                    size=[min(s * 8, 20) for s in model_data["overall_score"]],
                    color="overall_score",
                    hover_name=model_data["chunk_strategy"]
                    + "-"
                    + model_data["chunk_size"].astype(str),
                    text=model_data["chunk_strategy"],
                    title=f"{model_name} - チャンク分布とパフォーマンス",
                    labels={
                        "num_chunks": "チャンク数",
                        "avg_chunk_len": "平均チャンクサイズ (文字数)",
                        "overall_score": "総合スコア",
                    },
                    color_continuous_scale=px.colors.sequential.Viridis,
                    color_continuous_midpoint=0.5,
                )
                fig_bubble.update_traces(
                    texttemplate="",
                    marker=dict(
                        line=dict(width=1.5, color="rgba(0,0,0,0.7)"),
                        opacity=0.8,
                        sizemode="diameter",
                        sizemin=6,
                        sizeref=0.1,
                    ),
                    hovertemplate=(
                        "<b>%{hovertext}</b><br><br>"
                        "チャンク数: <b>%{x}</b><br>"
                        "平均サイズ: <b>%{y}文字</b><br>"
                        "スコア: <b>%{marker.color:.3f}</b><extra></extra>"
                    ),
                    hoverlabel=dict(
                        font_size=14,
                        font_family=japanese_font,
                        bgcolor="white",
                        bordercolor="#333",
                        font_color="#333",
                    ),
                )
                fig_bubble.update_layout(
                    title={
                        "text": f"{model_name} - チャンク分布とパフォーマンス",
                        "x": 0.5,
                        "xanchor": "center",
                        "font": {
                            "size": 20,
                            "family": japanese_font,
                            "color": "#333",
                        },
                        "y": 0.95,
                        "yanchor": "top",
                    },
                    coloraxis_colorbar=dict(
                        title=dict(
                            text="スコア",
                            font=dict(size=14, family=japanese_font),
                        ),
                        tickfont=dict(family=japanese_font, size=12),
                    ),
                    font=dict(
                        size=14,
                        family=japanese_font,
                        color="#333",
                    ),
                    height=600,
                    margin=dict(l=80, r=50, t=100, b=120),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(
                        title=dict(
                            text="チャンク数",
                            font=dict(size=14, family=japanese_font),
                        ),
                        tickfont=dict(size=12, family=japanese_font),
                        gridcolor="rgba(0,0,0,0.1)",
                        showline=True,
                        linewidth=1,
                        linecolor="#ddd",
                        mirror=True,
                    ),
                    yaxis=dict(
                        title=dict(
                            text="平均チャンクサイズ (文字数)",
                            font=dict(size=14, family=japanese_font),
                        ),
                        tickfont=dict(size=12, family=japanese_font),
                        gridcolor="rgba(0,0,0,0.1)",
                        showline=True,
                        linewidth=1,
                        linecolor="#ddd",
                        mirror=True,
                    ),
                )
                for _, row in model_data.iterrows():
                    fig_bubble.add_annotation(
                        x=row["num_chunks"],
                        y=row["avg_chunk_len"],
                        text=row["chunk_strategy"],
                        showarrow=False,
                        yshift=10,
                        font=dict(
                            size=10, family=japanese_font, color="#333333"
                        ),
                        xanchor="center",
                        yanchor="bottom",
                        opacity=0.9,
                    )
                current_graph += 1
                update_progress(
                    current_graph, total_graphs, f"バブルチャートを生成中: {model_name}"
                )
                img_data = save_plotly_figure(
                    fig_bubble,
                    f"bubble_chart_{model_name}",
                    width=1200,
                    height=800,
                    scale=3.0,
                )
                if img_data:
                    filepath = os.path.join(
                        temp_dir, f"bubble_chart_{model_name}.png"
                    )
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    saved_files.append(filepath)

        # 2. バーチャート
        for model_name, model_data in model_groups:
            if not model_data.empty and "chunk_strategy" in model_data.columns and "overall_score" in model_data.columns:
                strategy_scores = (
                    model_data.groupby("chunk_strategy")["overall_score"]
                    .mean()
                    .sort_values(ascending=False)
                )
                bar_data = pd.DataFrame(
                    {
                        "strategy": strategy_scores.index,
                        "score": strategy_scores.values,
                    }
                )
                fig_bar = px.bar(
                    data_frame=bar_data,
                    x="score",
                    y="strategy",
                    orientation="h",
                    title=f"{model_name} - チャンク戦略別パフォーマンス",
                    labels={"score": "平均スコア", "strategy": "チャンク戦略"},
                    color="score",
                    color_continuous_scale=px.colors.sequential.Viridis,
                )
                fig_bar.update_traces(
                    texttemplate="%{x:.3f}",
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>スコア: %{x:.3f}<extra></extra>",
                    textfont=dict(size=12, family=japanese_font, color="#333333"),
                )
                fig_bar.update_layout(
                    title={
                        "text": f"{model_name} - チャンク戦略別パフォーマンス",
                        "x": 0.5,
                        "xanchor": "center",
                        "y": 0.95,
                        "yanchor": "top",
                        "font": {
                            "size": 18,
                            "family": japanese_font,
                            "color": "#333333",
                        },
                    },
                    xaxis=dict(
                        range=[0, 1.1],
                        title=dict(
                            text="平均スコア",
                            font=dict(size=14, family=japanese_font),
                        ),
                        tickfont=dict(size=12, family=japanese_font),
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="rgba(0, 0, 0, 0.1)",
                        showline=True,
                        linewidth=1,
                        linecolor="gray",
                        automargin=True,
                    ),
                    yaxis=dict(
                        title=dict(
                            text="チャンク戦略",
                            font=dict(size=14, family=japanese_font),
                        ),
                        tickfont=dict(size=12, family=japanese_font),
                        autorange="reversed",
                        automargin=True,
                        showline=True,
                        linewidth=1,
                        linecolor="gray",
                    ),
                    coloraxis_showscale=False,
                    height=500,
                    margin=dict(l=120, r=50, t=120, b=80),
                    font=dict(
                        size=14,
                        family=japanese_font,
                        color="#333333",
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    hoverlabel=dict(
                        font_size=12,
                        font_family=japanese_font,
                    ),
                )
                current_graph += 1
                update_progress(
                    current_graph, total_graphs, f"バーチャートを生成中: {model_name}"
                )
                img_data = save_plotly_figure(
                    fig_bar,
                    f"bar_chart_{model_name}",
                    width=1200,
                    height=800,
                    scale=3.0,
                )
                if img_data:
                    filepath = os.path.join(
                        temp_dir, f"bar_chart_{model_name}.png"
                    )
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    saved_files.append(filepath)

        # 3. レーダーチャート
        if "chunk_strategy" in results_df.columns:
            chunk_strategies = results_df["chunk_strategy"].unique()
            for strategy in chunk_strategies:
                strategy_data = results_df[
                    results_df["chunk_strategy"] == strategy
                ]
                if not strategy_data.empty:
                    fig_radar = go.Figure()
                    for model_name, model_data in model_groups:
                        if "embedding_model" in strategy_data.columns:
                            model_strategy_data = strategy_data[
                                strategy_data["embedding_model"] == model_name
                            ]
                        else:
                            model_strategy_data = strategy_data
                        if not model_strategy_data.empty:
                            r_values = [
                                model_strategy_data[m].mean()
                                if m in model_strategy_data.columns
                                else 0.5
                                for m in metrics
                            ]
                            text_values = [f"{v:.2f}" for v in r_values]
                            fig_radar.add_trace(
                                go.Scatterpolar(
                                    r=r_values,
                                    theta=metrics_jp,
                                    fill="toself",
                                    name=model_name,
                                    text=text_values,
                                    textposition="top center",
                                    textfont=dict(
                                        size=11,
                                        color="black",
                                        family=japanese_font,
                                    ),
                                    hovertemplate="<b>%{theta}</b><br>スコア: %{r:.2f}<extra></extra>",
                                    line=dict(width=2),
                                    mode="lines+markers+text",
                                    marker=dict(size=6, opacity=0.8),
                                )
                            )

                    colors = [
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",
                        "#9467bd",
                        "#8c564b",
                        "#e377c2",
                        "#7f7f7f",
                        "#bcbd22",
                        "#17becf",
                    ]
                    for i, trace in enumerate(fig_radar.data):
                        text_positions = []
                        for j, theta in enumerate(trace.theta):
                            angle = (j * 30) % 360
                            if 15 <= angle < 165:
                                pos = "top center"
                            elif 195 <= angle < 345:
                                pos = "bottom center"
                            else:
                                pos = "middle right" if angle < 180 else "middle left"
                            text_positions.append(pos)
                        trace.update(
                            line=dict(
                                width=2.5, color=colors[i % len(colors)]
                            ),
                            marker=dict(
                                size=8,
                                color=colors[i % len(colors)],
                                line=dict(width=1, color="black"),
                                opacity=0.9,
                            ),
                            textposition=text_positions,
                            textfont=dict(
                                size=10,
                                color="black",
                                family=japanese_font,
                            ),
                            mode="lines+markers+text",
                            opacity=0.8,
                        )

                    display_strategy = str(strategy)
                    title_text = f"{display_strategy} - 評価メトリクスの比較"
                    fig_radar.update_layout(
                        title={
                            "text": title_text,
                            "x": 0.5,
                            "xanchor": "center",
                            "y": 0.95,
                            "yanchor": "top",
                            "font": {
                                "size": 20,
                                "family": japanese_font,
                                "color": "#333333",
                            },
                        },
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1],
                                tickfont=dict(
                                    size=11,
                                    family=japanese_font,
                                    color="#555555",
                                ),
                                tickangle=0,
                                tickformat=".1f",
                                gridwidth=1,
                                gridcolor="lightgray",
                                linecolor="gray",
                                linewidth=1,
                                showline=True,
                            ),
                            angularaxis=dict(
                                rotation=90,
                                direction="clockwise",
                                tickfont=dict(
                                    size=12,
                                    family=japanese_font,
                                    color="#333333",
                                ),
                                gridwidth=1,
                                gridcolor="lightgray",
                                linecolor="gray",
                                linewidth=1,
                                showline=True,
                            ),
                            bgcolor="rgba(250, 250, 250, 0.8)",
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.15,
                            xanchor="center",
                            x=0.5,
                            font=dict(
                                size=12,
                                family=japanese_font,
                                color="#333333",
                            ),
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="#DDDDDD",
                            borderwidth=1,
                            itemclick=False,
                            itemdoubleclick=False,
                        ),
                        margin=dict(l=80, r=80, t=120, b=150),
                        height=600,
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        font=dict(family=japanese_font, color="#333333"),
                        hoverlabel=dict(font_size=12, font_family=japanese_font),
                    )
                    fig_radar.update_polars(
                        radialaxis=dict(
                            showgrid=True,
                            gridcolor="lightgray",
                            gridwidth=1,
                            showline=True,
                            linecolor="gray",
                            linewidth=1,
                        ),
                        angularaxis=dict(
                            showgrid=True,
                            gridcolor="lightgray",
                            gridwidth=1,
                            showline=True,
                            linecolor="gray",
                            linewidth=1,
                        ),
                    )
                    current_graph += 1
                    update_progress(
                        current_graph,
                        total_graphs,
                        f"レーダーチャートを生成中: {strategy}",
                    )
                    img_data = save_plotly_figure(
                        fig_radar,
                        f"radar_chart_{strategy}",
                        width=1200,
                        height=800,
                        scale=3.0,
                    )
                    if img_data:
                        filepath = os.path.join(
                            temp_dir,
                            f"radar_chart_{strategy}.png".replace("/", "_"),
                        )
                        with open(filepath, "wb") as f:
                            f.write(img_data)
                        saved_files.append(filepath)

        # ZIP 作成
        if saved_files:
            update_progress(total_graphs, total_graphs, "ZIPファイルを作成中...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"{filename}_{timestamp}.zip"
            zip_path = os.path.join(temp_dir, zip_filename)
            import zipfile

            try:
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    for i, file in enumerate(saved_files, 1):
                        zipf.write(file, os.path.basename(file))
                        update_progress(
                            total_graphs,
                            total_graphs,
                            f"ZIPに追加中: {i}/{len(saved_files)}",
                        )
                with open(zip_path, "rb") as f:
                    zip_data = f.read()
                status_text.success(
                    f"完了！ {len(saved_files)}個のファイルをZIPに保存しました。"
                )
                progress_bar.empty()
                return zip_data
            except Exception as e:
                status_text.error(
                    f"ZIPファイルの作成中にエラーが発生しました: {e}"
                )
                return None
        else:
            status_text.warning("保存するグラフがありませんでした。")
            return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
