# 評価履歴表示UI（Streamlit）
import streamlit as st
from http_client import http_get, http_delete
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
from typing import Any, Mapping


def _create_label(row: pd.Series) -> str:
    """チャンク戦略とサイズからラベルを生成する."""
    chunk_strategy = str(row.get("chunk_strategy") or row.get("chunk_method") or "unknown").strip()
    chunk_size = row.get("chunk_size")
    if chunk_strategy.lower() in {"semantic", "sentence", "paragraph"}:
        return chunk_strategy.lower()
    # pandas.NA を含む値に対して直接 ==/!= を行うと "boolean value of NA is ambiguous" になるため、
    # 先に pd.isna で判定してからその他の条件をチェックする
    if chunk_size is not None and not pd.isna(chunk_size):
        if chunk_size != "":
            return f"{chunk_strategy}-{chunk_size}"
    return chunk_strategy or "unknown"


def _plot_overlap_comparison_for_history(results_df: pd.DataFrame, key_prefix: str = "") -> None:
    """一括評価タブと同様のオーバーラップ比較グラフを描画する."""
    required_columns = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
        "answer_correctness",
        "overall_score",
    ]
    available_metrics = [col for col in required_columns if col in results_df.columns]
    if not available_metrics:
        st.info("比較可能な評価指標が不足しているため、オーバーラップ比較グラフは表示できません。")
        return

    if "overlap" not in results_df.columns:
        if "chunk_overlap" in results_df.columns:
            results_df = results_df.assign(overlap=results_df["chunk_overlap"])
        else:
            st.info("overlap または chunk_overlap 列が存在しないため、オーバーラップ比較グラフは表示できません。")
            return
    else:
        results_df = results_df.copy()

    group_cols = ["overlap"]
    if "embedding_model" in results_df.columns:
        group_cols.append("embedding_model")
    if "chunk_strategy" in results_df.columns:
        group_cols.append("chunk_strategy")
    if "chunk_size" in results_df.columns:
        group_cols.append("chunk_size")

    overlap_scores = results_df.groupby(group_cols)[available_metrics].mean().reset_index()
    if len(overlap_scores) <= 1:
        st.info("オーバーラップ値が一種類のみのため、比較グラフは省略します。")
        return

    prefix = key_prefix or ""

    tab1, tab2, tab3 = st.tabs(["折れ線グラフ", "ヒートマップ", "最適値サマリー"])

    with tab1:
        for metric in available_metrics:
            st.subheader(f"{metric} の比較")
            if "embedding_model" in overlap_scores.columns and "chunk_size" in overlap_scores.columns:
                models = overlap_scores["embedding_model"].unique()
                chunk_sizes = sorted(overlap_scores["chunk_size"].dropna().unique())
                if len(models) == 0:
                    st.info("モデル情報が不足しているため、グラフは表示できません。")
                    continue
                model_tabs = st.tabs([str(model) for model in models])
                for idx, model in enumerate(models):
                    with model_tabs[idx]:
                        model_data = overlap_scores[overlap_scores["embedding_model"] == model]
                        fig = go.Figure()
                        colors = px.colors.qualitative.Plotly
                        for i, chunk_size in enumerate(chunk_sizes):
                            size_rows = model_data[model_data["chunk_size"] == chunk_size]
                            if size_rows.empty:
                                continue
                            mean_rows = size_rows.groupby("overlap", as_index=False)[metric].mean()
                            display_label = _create_label(size_rows.iloc[0]) if len(size_rows) > 0 else str(chunk_size)
                            fig.add_trace(
                                go.Scatter(
                                    x=mean_rows["overlap"],
                                    y=mean_rows[metric],
                                    mode="lines+markers",
                                    name=display_label,
                                    line=dict(color=colors[i % len(colors)], width=3),
                                    marker=dict(size=9, color=colors[i % len(colors)]),
                                    hovertemplate=
                                        f"<b>{display_label}</b><br>オーバーラップ: %{{x}}<br>スコア: %{{y:.3f}}<extra></extra>",
                                )
                            )
                        fig.update_layout(
                            title=f"{model} - チャンクサイズ別比較",
                            xaxis_title="オーバーラップサイズ (トークン数)",
                            yaxis_title=f"{metric} スコア (0-1)",
                            template="plotly_white",
                            height=400,
                            margin=dict(l=50, r=50, t=60, b=40),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        )
                        chart_key = f"{prefix}overlap_line_{metric}_{str(model)}"
                        st.plotly_chart(fig, use_container_width=True, key=chart_key)
            else:
                fig = px.line(
                    overlap_scores,
                    x="overlap",
                    y=metric,
                    color="embedding_model" if "embedding_model" in overlap_scores.columns else None,
                    title=f"{metric} の比較",
                    markers=True,
                )
                fig.update_layout(
                    xaxis_title="オーバーラップサイズ (トークン数)",
                    yaxis_title=f"{metric} スコア (0-1)",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{prefix}overlap_line_simple_{metric}")

    with tab2:
        if "chunk_size" in overlap_scores.columns and "embedding_model" in overlap_scores.columns:
            for model in overlap_scores["embedding_model"].unique():
                model_data = overlap_scores[overlap_scores["embedding_model"] == model]
                pivot = model_data.pivot_table(
                    index="chunk_size",
                    columns="overlap",
                    values="overall_score",
                    aggfunc="mean",
                )
                if pivot.empty:
                    continue
                fig = px.imshow(
                    pivot,
                    labels=dict(x="オーバーラップ", y="チャンクサイズ", color="総合スコア"),
                    title=f"{model} - チャンク×オーバーラップ ヒートマップ",
                    aspect="auto",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{prefix}overlap_heatmap_{str(model)}")
        else:
            st.info("ヒートマップを描画するためのチャンクサイズ/モデル情報が不足しています。")

    with tab3:
        if "chunk_strategy" in overlap_scores.columns:
            summary_df = (
                overlap_scores
                .groupby("chunk_strategy")[available_metrics]
                .mean()
                .sort_values("overall_score", ascending=False)
            )
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("chunk_strategy 列が存在しないため、サマリー表示を省略します。")


def _render_bulk_style_charts(results_df: pd.DataFrame, key_prefix: str = "") -> None:
    """一括評価タブと同じ構成のグラフ群を描画する.

    ユーザーが「簡易ビュー」と「詳細ビュー」を切り替えられるようにし、
    簡易ビューでは全体像を素早く把握するための最低限のグラフのみを表示する。
    """
    if results_df.empty:
        st.info("評価結果が空のため、グラフは表示できません。")
        return

    df = results_df.copy()

    # embedding_model 列が存在しない場合はダミー値で補完し、グラフ描画時の KeyError を防ぐ
    if "embedding_model" not in df.columns:
        df["embedding_model"] = "unknown"

    if "chunk_strategy" not in df.columns and "chunk_method" in df.columns:
        df["chunk_strategy"] = df["chunk_method"]
    if "overlap" not in df.columns and "chunk_overlap" in df.columns:
        df["overlap"] = df["chunk_overlap"]
    for col in [
        "overall_score",
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
        "answer_correctness",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    if "chunk_size" not in df.columns:
        df["chunk_size"] = pd.NA
    if "num_chunks" not in df.columns:
        df["num_chunks"] = 0
    if "avg_chunk_len" not in df.columns:
        df["avg_chunk_len"] = 0

    df["label"] = df.apply(_create_label, axis=1)

    # ビューモードの選択（簡易 / 詳細）
    view_mode = st.radio(
        "グラフ表示モード",
        ["簡易ビュー", "詳細ビュー"],
        index=0,
        key=f"{key_prefix}view_mode_radio",
        horizontal=True,
    )

    # --- 簡易ビュー: overall_score のチャンク戦略別バーのみ ---
    if view_mode == "簡易ビュー":
        if "chunk_strategy" in df.columns and "overall_score" in df.columns:
            st.write("### チャンク戦略別 平均総合スコア（簡易ビュー）")
            bar_df = (
                df.groupby("chunk_strategy")["overall_score"].mean().sort_values(ascending=False).reset_index()
            )
            fig = px.bar(
                bar_df,
                x="chunk_strategy",
                y="overall_score",
                title="チャンク戦略別 平均総合スコア",
                labels={"chunk_strategy": "チャンク戦略", "overall_score": "平均総合スコア"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}chunk_strategy_bar_simple")
        else:
            st.info("簡易ビューを表示するための列（chunk_strategy / overall_score）が不足しています。")
        return

    # --- 詳細ビュー: 既存のすべてのグラフを表示 ---
    st.write("### オーバーラップ比較")
    _plot_overlap_comparison_for_history(df, key_prefix)

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
        "answer_correctness",
    ]
    metrics_jp = ["信頼性", "回答の関連性", "コンテキスト再現", "コンテキスト精度", "回答正確性"]
    available_metrics = [m for m in metrics if m in df.columns]

    if available_metrics:
        st.write("### レーダーチャート")
        for label in df["label"].dropna().unique():
            subset = df[df["label"] == label]
            if subset.empty:
                continue
            fig = go.Figure()
            for model in subset["embedding_model"].dropna().unique():
                model_df = subset[subset["embedding_model"] == model]
                if model_df.empty:
                    continue
                r_values = [
                    float(model_df[m].mean()) if m in model_df.columns and pd.notna(model_df[m]).any() else 0
                    for m in metrics
                ]
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_values,
                        theta=metrics_jp,
                        fill="toself",
                        name=str(model),
                        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
                    )
                )
            fig.update_layout(
                title=f"{label} - モデル別メトリクス比較",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}radar_{label}_{time.time_ns()}")

    if {"num_chunks", "avg_chunk_len", "overall_score"}.issubset(df.columns):
        st.write("### バブルチャート (チャンク分布)")
        for model, model_data in df.groupby("embedding_model"):
            if model_data.empty:
                continue
            plot_data = model_data.copy()
            plot_data["bubble_size"] = plot_data["overall_score"].apply(
                lambda v: min(float(v) * 20, 50) if pd.notna(v) else 5
            )
            model_str = str(model).replace(" ", "_")
            fig = px.scatter(
                plot_data,
                x="num_chunks",
                y="avg_chunk_len",
                size="bubble_size",
                color="overall_score",
                hover_data={
                    "chunk_size": True,
                    "chunk_strategy": True,
                    "num_chunks": True,
                    "avg_chunk_len": ":.1f",
                    "overall_score": ".3f",
                },
                labels={
                    "num_chunks": "チャンク数",
                    "avg_chunk_len": "平均チャンク長",
                    "overall_score": "総合スコア",
                },
                color_continuous_scale=px.colors.sequential.Viridis,
                title=f"{model} - チャンク分布とスコア",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}bubble_{model_str}")

    if "chunk_strategy" in df.columns and "overall_score" in df.columns:
        st.write("### チャンク戦略別平均スコア")
        bar_df = (
            df.groupby("chunk_strategy")["overall_score"].mean().sort_values(ascending=False).reset_index()
        )
        fig = px.bar(
            bar_df,
            x="chunk_strategy",
            y="overall_score",
            title="チャンク戦略別 平均総合スコア",
            labels={"chunk_strategy": "チャンク戦略", "overall_score": "平均総合スコア"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}chunk_strategy_bar")


def apply_bulk_chunk_settings_from_history(row: Mapping[str, Any]) -> None:
    chunk_strategy = str(row.get("chunk_strategy") or row.get("chunk_method") or "").strip()
    method_candidates = ["semantic", "recursive", "fixed", "sentence", "paragraph"]
    chunk_method = None
    for m in method_candidates:
        if chunk_strategy.startswith(m):
            chunk_method = m
            break
    if not chunk_method and chunk_strategy:
        chunk_method = chunk_strategy.split("-", 1)[0]
    if not chunk_method:
        return

    st.session_state["bulk_chunk_methods"] = [chunk_method]

    if chunk_method != "semantic":
        size_val = row.get("chunk_size")
        overlap_val = row.get("chunk_overlap")
        # pandas.NA に対しては常にスキップする
        if size_val is not None and not pd.isna(size_val) and size_val != "":
            try:
                st.session_state["bulk_chunk_sizes_select"] = [int(size_val)]
            except Exception:
                pass
        if overlap_val is not None and not pd.isna(overlap_val) and overlap_val != "":
            try:
                st.session_state["bulk_chunk_overlaps_select"] = [int(overlap_val)]
            except Exception:
                pass

    embedding_model = row.get("embedding_model")
    if embedding_model:
        st.session_state["embedding_model"] = str(embedding_model)


def show_evaluation_history(backend_url: str):
    """
    評価履歴を表示するStreamlit UI
    """
    st.header("📊 評価履歴・実験管理")
    
    # タブで機能を分割
    tab1, tab2, tab3 = st.tabs(["実験一覧", "評価ダッシュボード", "統計情報"])
    
    # 履歴APIから実験一覧を取得
    experiments = []
    experiments_error = None
    try:
        response = http_get(f"{backend_url}/history/experiments")
        if response.status_code == 200:
            data = response.json() or {}
            experiments = data.get("items", data.get("experiments", []))
        else:
            experiments_error = f"実験履歴取得エラー: {response.status_code}"
    except Exception as e:
        experiments_error = f"実験履歴取得エラー: {str(e)}"

    with tab1:
        st.subheader("実験履歴一覧")

        if experiments_error:
            st.error(experiments_error)
        else:
            if experiments:
                df = pd.DataFrame(experiments)

                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                    df['created_date'] = df['created_at'].dt.date
                else:
                    df['created_at'] = pd.NaT
                    df['created_date'] = pd.NaT

                date_options = sorted(df['created_date'].dropna().unique().tolist(), reverse=True)
                if date_options:
                    selected_date = st.selectbox(
                        "表示する日付",
                        options=[None] + date_options,
                        format_func=lambda v: "すべて" if v is None else v.isoformat(),
                        key="history_date_filter_selectbox"
                    )
                else:
                    selected_date = None
                    st.info("作成日時が取得できていないため日付フィルタは利用できません。")

                filtered_df = df.copy()
                if selected_date is not None:
                    filtered_df = filtered_df[filtered_df['created_date'] == selected_date]

                # PDF ID での絞り込み
                if 'pdf_file_id' in filtered_df.columns:
                    pdf_id_values = (
                        filtered_df['pdf_file_id']
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    if pdf_id_values:
                        selected_pdf_id = st.selectbox(
                            "PDF ID で絞り込み",
                            options=[None] + sorted(pdf_id_values),
                            format_func=lambda v: "すべて" if v is None else str(v),
                            key="history_pdf_id_filter_selectbox",
                        )
                        if selected_pdf_id is not None:
                            filtered_df = filtered_df[
                                filtered_df['pdf_file_id'].astype(str) == str(selected_pdf_id)
                            ]

                # ステータスでの絞り込み
                if 'status' in filtered_df.columns:
                    status_values = (
                        filtered_df['status']
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    if status_values:
                        selected_status = st.selectbox(
                            "ステータスで絞り込み",
                            options=[None] + sorted(status_values),
                            format_func=lambda v: "すべて" if v is None else str(v),
                            key="history_status_filter_selectbox",
                        )
                        if selected_status is not None:
                            filtered_df = filtered_df[
                                filtered_df['status'].astype(str) == str(selected_status)
                            ]

                if filtered_df.empty:
                    st.info("選択された条件に一致する実験履歴がありません。")
                    return

                display_df = filtered_df.copy()
                if 'created_at' in display_df.columns:
                    display_df['created_at'] = display_df['created_at'].dt.strftime('%Y-%m-%d %H:%M')

                display_columns = [
                    'created_date',
                    'created_at',
                    'id',
                    'experiment_name',
                    'file_name',
                    'pdf_file_id',
                    'status',
                    'total_combinations',
                    'completed_combinations'
                ]
                available_columns = [col for col in display_columns if col in display_df.columns]

                st.write(f"**表示中の実験数**: {len(display_df)} 件")
                st.dataframe(
                    display_df[available_columns],
                    use_container_width=True,
                    column_config={
                        "created_date": "作成日",
                        "created_at": "作成日時",
                        "id": "実験ID",
                        "experiment_name": "実験名",
                        "file_name": "ファイル名",
                        "pdf_file_id": "PDFファイルID",
                        "status": "ステータス",
                        "total_combinations": "総組み合わせ数",
                        "completed_combinations": "完了数"
                    }
                )

                if filtered_df['created_date'].notna().any():
                    st.write("**日付別の履歴**")
                    for date_value in sorted(filtered_df['created_date'].dropna().unique(), reverse=True):
                        per_day_df = display_df[display_df['created_date'] == date_value]
                        with st.expander(f"{date_value} の実験 ({len(per_day_df)} 件)", expanded=(selected_date == date_value)):
                            st.dataframe(
                                per_day_df[available_columns],
                                use_container_width=True
                            )

                st.subheader("実験詳細")
                detail_df_source = filtered_df
                selected_exp_id = st.selectbox(
                    "詳細を表示する実験を選択",
                    options=detail_df_source['id'].tolist(),
                    format_func=lambda x: f"ID:{x} - {detail_df_source[detail_df_source['id']==x]['experiment_name'].iloc[0] if len(detail_df_source[detail_df_source['id']==x]) > 0 else 'Unknown'}",
                    key="experiment_detail_selectbox"
                )
                if selected_exp_id:
                    selected_exp_name_series = detail_df_source.loc[detail_df_source['id'] == selected_exp_id, 'experiment_name']
                    selected_exp_name = selected_exp_name_series.iloc[0] if not selected_exp_name_series.empty else "unknown"
                    key_safe_exp_name = str(selected_exp_name).replace(" ", "_") if selected_exp_name else "unknown"

                    # この実験に紐づくPDF ID を取得し、他タブへのショートカットを提供
                    pdf_id_series = detail_df_source.loc[detail_df_source['id'] == selected_exp_id, 'pdf_file_id']
                    selected_pdf_id = pdf_id_series.iloc[0] if not pdf_id_series.empty else None
                    if selected_pdf_id is not None:
                        st.caption(f"この実験のPDF ID: {selected_pdf_id}")
                        col_chat_shortcut, col_bulk_shortcut = st.columns(2)
                        with col_chat_shortcut:
                            if st.button(
                                "このPDFでチャットボットを開く",
                                key=f"history_open_chat_for_exp_{selected_exp_id}",
                            ):
                                # チャットボットタブの単一PDFスコープ用に選択
                                st.session_state["rag_pdf_file_id"] = str(selected_pdf_id)
                                st.session_state["rag_scope"] = "single"
                                st.info(
                                    "チャットボットタブでこのPDFが選択されるようになりました。上部の『チャットボット』タブを開いてください。"
                                )
                        with col_bulk_shortcut:
                            if st.button(
                                "このPDFを一括評価タブで選択",
                                key=f"history_open_bulk_for_exp_{selected_exp_id}",
                            ):
                                # 一括評価タブの評価対象PDFとして選択
                                st.session_state["file_id"] = str(selected_pdf_id)
                                st.info(
                                    "RAGAS一括評価タブでこのPDFがデフォルト選択されます。上部の『RAGAS一括評価』タブを開いてください。"
                                )

                    # 実験結果を取得
                    try:
                        result_response = http_get(f"{backend_url}/history/experiments/{selected_exp_id}/results")
                        if result_response.status_code == 200:
                            result_data = result_response.json() or {}
                            results = result_data.get("items", result_data.get("results", []))
                            if not isinstance(results, list):
                                results = []

                            for res in results:
                                details_dict = {}
                                details_raw = res.get('details')
                                if isinstance(details_raw, str):
                                    try:
                                        details_dict = json.loads(details_raw)
                                    except json.JSONDecodeError:
                                        details_dict = {}
                                elif isinstance(details_raw, dict):
                                    details_dict = details_raw

                                if isinstance(details_dict, dict):
                                    res.setdefault('metrics', details_dict.get('metrics', []))
                                    res['details_dict'] = details_dict
                                else:
                                    res.setdefault('metrics', [])
                                    res['details_dict'] = {}

                            if results:
                                result_df = pd.DataFrame(results)

                                # experiment_results テーブル由来の ID 情報を明示的に表示する
                                if 'experiment_id' not in result_df.columns:
                                    result_df['experiment_id'] = selected_exp_id

                                result_columns = [
                                    'id',  # experiment_results.id
                                    'experiment_id',
                                    'embedding_model',
                                    'chunk_strategy',
                                    'chunk_size',
                                    'chunk_overlap',
                                    'num_chunks',
                                    'avg_chunk_len',
                                    'overall_score',
                                    'faithfulness',
                                    'answer_relevancy',
                                    'context_recall',
                                    'context_precision',
                                    'answer_correctness',
                                    'answer_similarity',
                                ]
                                available_result_columns = [
                                    col for col in result_columns if col in result_df.columns
                                ]

                                # --- CSVエクスポート用のフィルタ＆ダウンロード ---
                                with st.expander("この実験の結果をCSVダウンロード", expanded=False):
                                    export_df = result_df.copy()
                                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

                                    # Embeddingモデルでフィルタ
                                    if 'embedding_model' in export_df.columns:
                                        with col_f1:
                                            emb_choices = (
                                                export_df['embedding_model']
                                                .dropna()
                                                .astype(str)
                                                .unique()
                                                .tolist()
                                            )
                                            emb_choices = sorted(emb_choices)
                                            selected_embs = st.multiselect(
                                                "Embeddingモデル",
                                                emb_choices,
                                                default=emb_choices,
                                                key=f"csv_emb_{selected_exp_id}",
                                            )
                                            if selected_embs:
                                                export_df = export_df[export_df['embedding_model'].astype(str).isin(selected_embs)]

                                    # チャンク戦略でフィルタ
                                    if 'chunk_strategy' in export_df.columns:
                                        with col_f2:
                                            strat_choices = (
                                                export_df['chunk_strategy']
                                                .dropna()
                                                .astype(str)
                                                .unique()
                                                .tolist()
                                            )
                                            strat_choices = sorted(strat_choices)
                                            selected_strats = st.multiselect(
                                                "チャンク戦略",
                                                strat_choices,
                                                default=strat_choices,
                                                key=f"csv_chunk_strategy_{selected_exp_id}",
                                            )
                                            if selected_strats:
                                                export_df = export_df[export_df['chunk_strategy'].astype(str).isin(selected_strats)]

                                    # チャンクサイズでフィルタ
                                    if 'chunk_size' in export_df.columns:
                                        with col_f3:
                                            size_choices = (
                                                export_df['chunk_size']
                                                .dropna()
                                                .unique()
                                                .tolist()
                                            )
                                            try:
                                                size_choices = sorted({int(v) for v in size_choices})
                                            except Exception:
                                                size_choices = sorted(size_choices)
                                            selected_sizes = st.multiselect(
                                                "チャンクサイズ",
                                                size_choices,
                                                default=size_choices,
                                                key=f"csv_chunk_size_{selected_exp_id}",
                                            )
                                            if selected_sizes:
                                                export_df = export_df[export_df['chunk_size'].isin(selected_sizes)]

                                    # オーバーラップでフィルタ
                                    if 'chunk_overlap' in export_df.columns:
                                        with col_f4:
                                            ov_choices = (
                                                export_df['chunk_overlap']
                                                .dropna()
                                                .unique()
                                                .tolist()
                                            )
                                            try:
                                                ov_choices = sorted({int(v) for v in ov_choices})
                                            except Exception:
                                                ov_choices = sorted(ov_choices)
                                            selected_ovs = st.multiselect(
                                                "オーバーラップ",
                                                ov_choices,
                                                default=ov_choices,
                                                key=f"csv_chunk_overlap_{selected_exp_id}",
                                            )
                                            if selected_ovs:
                                                export_df = export_df[export_df['chunk_overlap'].isin(selected_ovs)]

                                    st.caption(f"CSV対象レコード数: {len(export_df)} 件")
                                    if available_result_columns and not export_df.empty:
                                        csv_bytes = export_df[available_result_columns].to_csv(index=False).encode("utf-8-sig")
                                        st.download_button(
                                            "この条件でCSVダウンロード",
                                            data=csv_bytes,
                                            file_name=f"experiment_{selected_exp_id}_results.csv",
                                            mime="text/csv",
                                            key=f"csv_download_{selected_exp_id}",
                                        )

                                if available_result_columns:
                                    st.write("**評価結果一覧**")
                                    st.dataframe(
                                        result_df[available_result_columns],
                                        use_container_width=True
                                    )

                                metrics_cols = [
                                    'overall_score',
                                    'faithfulness',
                                    'answer_relevancy',
                                    'context_recall',
                                    'context_precision',
                                    'answer_correctness',
                                    'answer_similarity',
                                ]
                                available_metrics = [col for col in metrics_cols if col in result_df.columns]

                                if available_metrics:
                                    st.write("**評価指標**")
                                    st.dataframe(
                                        result_df[['embedding_model', 'chunk_strategy'] + available_metrics],
                                        use_container_width=True
                                    )

                                    # グラフ用にメトリクスを整形（組み合わせごとに棒グラフ表示）
                                    chart_df = result_df[['embedding_model', 'chunk_strategy'] + available_metrics].copy()
                                    chart_df['combination'] = chart_df.apply(
                                        lambda row: f"{row.get('embedding_model', 'unknown')} / {row.get('chunk_strategy', 'unknown')}",
                                        axis=1
                                    )
                                    metric_chart = chart_df.melt(
                                        id_vars='combination',
                                        value_vars=available_metrics,
                                        var_name='metric',
                                        value_name='score'
                                    )
                                    st.write("**評価指標グラフ**")
                                    metric_fig = px.bar(
                                        metric_chart,
                                        x='combination',
                                        y='score',
                                        color='metric',
                                        barmode='group',
                                        title="メトリクス別スコア比較"
                                    )
                                    metric_fig.update_layout(xaxis_title="モデル / チャンク戦略", yaxis_title="スコア")
                                    st.plotly_chart(
                                        metric_fig,
                                        use_container_width=True,
                                        key=f"detail_metric_bar_{selected_exp_id}_{key_safe_exp_name}"
                                    )

                                    st.write("**一括評価スタイルのグラフ**")
                                    _render_bulk_style_charts(
                                        result_df,
                                        key_prefix=f"history_exp_{selected_exp_id}_{key_safe_exp_name}_"
                                    )

                                    row_options = list(range(len(result_df)))
                                    if row_options:
                                        labels_for_rows = []
                                        for idx_row in row_options:
                                            r = result_df.iloc[idx_row]
                                            label = (
                                                f"{r.get('embedding_model', 'unknown')} / "
                                                f"{r.get('chunk_strategy', r.get('chunk_method', 'unknown'))} / "
                                                f"size={r.get('chunk_size', '-')}, overlap={r.get('chunk_overlap', '-')}"
                                            )
                                            labels_for_rows.append(label)

                                        selected_result_idx_for_apply = st.selectbox(
                                            "一括評価タブに反映する設定",
                                            options=row_options,
                                            format_func=lambda i: labels_for_rows[i],
                                            key=f"history_result_select_for_apply_{selected_exp_id}",
                                        )
                                        if st.button(
                                            "この設定を一括評価タブに反映",
                                            key=f"history_apply_to_bulk_tab_{selected_exp_id}",
                                        ):
                                            apply_bulk_chunk_settings_from_history(result_df.iloc[selected_result_idx_for_apply])
                                            st.success("一括評価タブのEmbedding/チャンク設定に反映しました。『RAGAS一括評価』タブを開いて確認してください。")

                                st.write("**質問別メトリクス**")
                                results_with_metrics = [
                                    r for r in results if isinstance(r.get('metrics'), list) and r['metrics']
                                ]

                                if results_with_metrics:
                                    metrics_options = [
                                        f"{r.get('embedding_model', 'unknown')} / {r.get('chunk_strategy', 'unknown')}"
                                        for r in results_with_metrics
                                    ]
                                    selected_metric_idx = st.selectbox(
                                        "質問別メトリクスを表示する結果を選択",
                                        options=range(len(results_with_metrics)),
                                        format_func=lambda i: metrics_options[i],
                                        key="metrics_per_qa_selectbox"
                                    )

                                    detailed_rows = []
                                    for item in results_with_metrics[selected_metric_idx].get('metrics', []):
                                        if not isinstance(item, dict):
                                            continue
                                        row = {
                                            '質問': item.get('question', ''),
                                            '生成回答': item.get('pred_answer', ''),
                                            '正解': item.get('ground_truth', ''),
                                        }
                                        metric_values = item.get('metrics', {})
                                        if isinstance(metric_values, dict):
                                            for key, value in metric_values.items():
                                                row[key] = value
                                        detailed_rows.append(row)

                                    if detailed_rows:
                                        detail_df = pd.DataFrame(detailed_rows)
                                        metric_columns = [
                                            col for col in detail_df.columns if col not in ['質問', '生成回答', '正解']
                                        ]
                                        for col in metric_columns:
                                            detail_df[col] = pd.to_numeric(detail_df[col], errors='coerce')
                                        st.dataframe(detail_df, use_container_width=True)

                                        # 質問ごとのメトリクス推移を折れ線グラフで表示
                                        numeric_columns = [col for col in metric_columns if detail_df[col].notna().any()]
                                        if numeric_columns:
                                            question_chart = detail_df.melt(
                                                id_vars='質問',
                                                value_vars=numeric_columns,
                                                var_name='メトリクス',
                                                value_name='スコア'
                                            )
                                            question_chart.sort_values('質問', inplace=True)
                                            question_fig = px.line(
                                                question_chart,
                                                x='質問',
                                                y='スコア',
                                                color='メトリクス',
                                                markers=True,
                                                title="質問別メトリクス推移"
                                            )
                                            question_fig.update_layout(xaxis_title="質問", yaxis_title="スコア")
                                            st.plotly_chart(
                                                question_fig,
                                                use_container_width=True,
                                                key=f"detail_question_line_{selected_exp_id}_{key_safe_exp_name}"
                                            )
                                    else:
                                        st.info("質問別メトリクスは取得できませんでした。")
                                else:
                                    st.info("質問別メトリクスが保存されている結果がありません。")
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
                            options=[None] + detail_df_source['id'].tolist(),
                            format_func=lambda x: "選択してください" if x is None else f"ID:{x} - {detail_df_source[detail_df_source['id']==x]['experiment_name'].iloc[0] if len(detail_df_source[detail_df_source['id']==x]) > 0 else 'Unknown'}"
                        )
                        
                        if delete_exp_id and st.button("実験を削除", type="secondary"):
                            try:
                                delete_response = http_delete(f"{backend_url}/history/experiments/{delete_exp_id}")
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
                st.info("実験履歴がありません。")
    
    with tab2:
        st.subheader("評価ダッシュボード")
        
        # 全実験の結果を統合分析
        try:
            response = http_get(f"{backend_url}/history/experiments")
            if response.status_code == 200:
                data = response.json() or {}
                experiments = data.get("items", data.get("experiments", []))
                
                if experiments:
                    # --- ダッシュボード用フィルタ（期間 / PDF ID / 実験名） ---
                    exp_df = pd.DataFrame(experiments)

                    # 作成日時から日付列を生成
                    if "created_at" in exp_df.columns:
                        try:
                            exp_df["created_at"] = pd.to_datetime(exp_df["created_at"], errors="coerce")
                            exp_df["created_date"] = exp_df["created_at"].dt.date
                        except Exception:
                            exp_df["created_date"] = pd.NaT
                    else:
                        exp_df["created_date"] = pd.NaT

                    col_f1, col_f2, col_f3 = st.columns(3)

                    # 期間フィルタ（From〜To）
                    with col_f1:
                        min_date = exp_df["created_date"].dropna().min()
                        max_date = exp_df["created_date"].dropna().max()
                        if pd.notna(min_date) and pd.notna(max_date):
                            from_date = st.date_input(
                                "作成日 From",
                                value=min_date,
                                key="dash_date_from",
                            )
                            to_date = st.date_input(
                                "作成日 To",
                                value=max_date,
                                key="dash_date_to",
                            )
                            if from_date:
                                exp_df = exp_df[exp_df["created_date"] >= from_date]
                            if to_date:
                                exp_df = exp_df[exp_df["created_date"] <= to_date]

                    # PDF ID フィルタ
                    with col_f2:
                        if "pdf_file_id" in exp_df.columns:
                            pdf_ids = (
                                exp_df["pdf_file_id"]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )
                            pdf_ids = sorted(pdf_ids)
                            selected_pdf = st.selectbox(
                                "PDF ID",
                                options=[None] + pdf_ids,
                                format_func=lambda v: "すべて" if v is None else str(v),
                                key="dash_pdf_id_filter",
                            )
                            if selected_pdf is not None:
                                exp_df = exp_df[exp_df["pdf_file_id"].astype(str) == str(selected_pdf)]

                    # 実験名フィルタ（部分一致）
                    with col_f3:
                        name_keyword = st.text_input(
                            "実験名に含まれる文字列",
                            value="",
                            key="dash_exp_name_filter",
                            help="部分一致でフィルタします（大文字小文字は区別しません）",
                        ).strip()
                        if name_keyword:
                            exp_df = exp_df[
                                exp_df.get("experiment_name", "")
                                .astype(str)
                                .str.contains(name_keyword, case=False, na=False)
                            ]

                    if exp_df.empty:
                        st.info("選択された条件に一致する実験がありません。フィルタ条件を見直してください。")
                        return

                    filtered_experiments = exp_df.to_dict("records")

                    # 全実験の結果を取得（フィルタ後の実験のみ）
                    all_results = []
                    for exp in filtered_experiments:
                        try:
                            result_response = http_get(f"{backend_url}/history/experiments/{exp['id']}/results")
                            if result_response.status_code == 200:
                                result_data = result_response.json()
                                results = result_data.get("items", result_data.get("results", []))
                                for result in results:
                                    result['experiment_id'] = exp['id']
                                    result['experiment_name'] = exp['experiment_name']
                                all_results.extend(results)
                        except:
                            continue
                    
                    if all_results:
                        all_df = pd.DataFrame(all_results)
                        if 'experiment_id' in all_df.columns:
                            suffix_source = tuple(sorted(all_df['experiment_id'].dropna().unique().tolist()))
                        else:
                            suffix_source = tuple(range(len(all_df)))
                        analysis_suffix = f"{len(all_df)}_{abs(hash(suffix_source))}"

                        # --- ダッシュボード用のサマリー指標 ---
                        total_experiments = len(filtered_experiments)
                        total_results = len(all_df)

                        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
                        with col_kpi1:
                            st.metric("総実験数", total_experiments)
                        with col_kpi2:
                            st.metric("総評価結果数", total_results)

                        best_model_label = "-"
                        best_strategy_label = "-"
                        if 'embedding_model' in all_df.columns and 'overall_score' in all_df.columns:
                            try:
                                model_avg = (
                                    all_df.groupby('embedding_model')['overall_score']
                                    .mean()
                                    .sort_values(ascending=False)
                                )
                                if not model_avg.empty:
                                    best_model_label = f"{model_avg.index[0]} ({model_avg.iloc[0]:.3f})"
                            except Exception:
                                best_model_label = "-"

                        if 'chunk_strategy' in all_df.columns and 'overall_score' in all_df.columns:
                            try:
                                strategy_avg = (
                                    all_df.groupby('chunk_strategy')['overall_score']
                                    .mean()
                                    .sort_values(ascending=False)
                                )
                                if not strategy_avg.empty:
                                    best_strategy_label = f"{strategy_avg.index[0]} ({strategy_avg.iloc[0]:.3f})"
                            except Exception:
                                best_strategy_label = "-"

                        with col_kpi3:
                            st.metric("ベストEmbeddingモデル", best_model_label)
                        with col_kpi4:
                            st.metric("ベストチャンク戦略", best_strategy_label)
                        
                        # Embeddingモデル上位3件（平均総合スコア）
                        if "embedding_model" in all_df.columns and "overall_score" in all_df.columns:
                            try:
                                top_model_df = (
                                    all_df.groupby("embedding_model")["overall_score"]
                                    .agg(["mean", "count"])
                                    .reset_index()
                                    .sort_values("mean", ascending=False)
                                    .head(3)
                                )
                                top_model_df.rename(
                                    columns={"embedding_model": "Embeddingモデル", "mean": "平均スコア", "count": "件数"},
                                    inplace=True,
                                )
                                st.write("**Embeddingモデル 上位3件（平均総合スコア）**")
                                st.dataframe(top_model_df, use_container_width=True)
                            except Exception:
                                pass

                        # チャンク戦略上位3件（平均総合スコア）
                        if "chunk_strategy" in all_df.columns and "overall_score" in all_df.columns:
                            try:
                                top_strategy_df = (
                                    all_df.groupby("chunk_strategy")["overall_score"]
                                    .agg(["mean", "count"])
                                    .reset_index()
                                    .sort_values("mean", ascending=False)
                                    .head(3)
                                )
                                top_strategy_df.rename(
                                    columns={"chunk_strategy": "チャンク戦略", "mean": "平均スコア", "count": "件数"},
                                    inplace=True,
                                )
                                st.write("**チャンク戦略 上位3件（平均総合スコア）**")
                                st.dataframe(top_strategy_df, use_container_width=True)
                            except Exception:
                                pass
                        
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
                            st.plotly_chart(fig, use_container_width=True, key=f"analysis_model_bar_{analysis_suffix}")

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
                            st.plotly_chart(fig, use_container_width=True, key=f"analysis_chunk_bar_{analysis_suffix}")
                        
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
                            st.plotly_chart(fig, use_container_width=True, key=f"analysis_corr_heatmap_{analysis_suffix}")
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
            response = http_get(f"{backend_url}/history/experiments")
            if response.status_code == 200:
                data = response.json() or {}
                experiments_detail = data.get("items", data.get("experiments", []))
                total_experiments = len(experiments_detail)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("総実験数", total_experiments)

                if experiments_detail:
                    try:
                        all_results = []
                        for exp in experiments_detail:
                            exp_id = exp.get('id')
                            if exp_id is None:
                                continue
                            resp = http_get(f"{backend_url}/history/experiments/{exp_id}/results")
                            if resp.status_code != 200:
                                continue
                            res_json = resp.json() or {}
                            items = res_json.get("items", res_json.get("results", []))
                            for item in items:
                                item['experiment_id'] = exp_id
                                item['experiment_name'] = exp.get('experiment_name')
                            all_results.extend(items)

                        if all_results:
                            all_df = pd.DataFrame(all_results)
                            if 'experiment_id' in all_df.columns:
                                stats_suffix_src = tuple(sorted(all_df['experiment_id'].dropna().unique().tolist()))
                            else:
                                stats_suffix_src = tuple(range(len(all_df)))
                            stats_suffix = f"{len(all_df)}_{abs(hash(stats_suffix_src))}"

                            if 'embedding_model' in all_df.columns and 'overall_score' in all_df.columns:
                                st.write("**モデル別性能比較**")
                                model_avg = all_df.groupby('embedding_model')['overall_score'].agg(['mean', 'std', 'count']).reset_index()
                                fig = px.bar(model_avg, x='embedding_model', y='mean', error_y='std', title="モデル別平均スコア（エラーバー：標準偏差）")
                                st.plotly_chart(fig, use_container_width=True, key=f"stats_model_bar_{stats_suffix}")

                            if 'chunk_strategy' in all_df.columns and 'overall_score' in all_df.columns:
                                st.write("**チャンク戦略別性能比較**")
                                chunk_avg = all_df.groupby('chunk_strategy')['overall_score'].agg(['mean', 'std', 'count']).reset_index()
                                fig = px.bar(chunk_avg, x='chunk_strategy', y='mean', error_y='std', title="チャンク戦略別平均スコア（エラーバー：標準偏差）")
                                st.plotly_chart(fig, use_container_width=True, key=f"stats_chunk_bar_{stats_suffix}")

                            numeric_cols = ['overall_score', 'faithfulness', 'answer_relevancy', 'context_recall',
                                            'context_precision', 'answer_correctness', 'answer_similarity',
                                            'avg_chunk_len', 'num_chunks']
                            available_numeric = [col for col in numeric_cols if col in all_df.columns]
                            if len(available_numeric) > 1:
                                st.write("**指標間相関分析**")
                                corr_matrix = all_df[available_numeric].corr()
                                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="評価指標間の相関係数")
                                st.plotly_chart(fig, use_container_width=True, key=f"stats_corr_heatmap_{stats_suffix}")
                        else:
                            st.info("統計を計算するための結果データがありません。")
                    except Exception as e:
                        st.error(f"統計情報取得中にエラー: {str(e)}")
            else:
                st.error(f"統計情報取得エラー: {response.status_code}")
        except Exception as e:
            st.error(f"統計情報取得エラー: {str(e)}")

if __name__ == "__main__":
    # テスト用
    show_evaluation_history("http://localhost:8000")
