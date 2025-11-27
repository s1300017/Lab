from __future__ import annotations

from typing import Any

import streamlit as st


def render_thesis_tab(tab_thesis: Any, BACKEND_URL: str) -> None:
    """卒論向け分析タブのプレースホルダ実装。"""
    with tab_thesis:
        st.subheader("卒論向け分析（復旧中）")
        st.info(
            "卒論向け分析タブの詳細な分析機能は現在復旧作業中です。\n"
            "PDFの要約やチャットを用いた確認は、チャットボットタブから行えます。"
        )
