from __future__ import annotations

from typing import Any

import streamlit as st


def render_overview_tab(tab_overview: Any) -> None:
    """システム説明タブ（概要）の簡易実装。"""
    with tab_overview:
        st.subheader("システム概要")
        st.markdown(
            """\
このアプリは、PDF文書を対象とした RAG (Retrieval-Augmented Generation) システムです。\\

- PDFアップロードと自動QA生成（サイドバー）\\
- チャンキング設定タブでの分割パラメータ検討（復旧中）\\
- 一括評価タブでのRAGAS評価（復旧中）\\
- チャットボットタブでの対話型RAG\\
- 履歴タブでのPDF・QA・チャンク・チャットログの参照\\

現在、一部タブは git clean の影響から復旧中ですが、\\
PDFアップロード〜チャット〜履歴閲覧の基本フローは利用できます。"""
        )
