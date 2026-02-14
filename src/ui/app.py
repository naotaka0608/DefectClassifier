"""Streamlit メインアプリケーション"""

import sys
from pathlib import Path

# Streamlit creates a new process, but running via `uv run` or `python -m streamlit` sets up the path correctly.
# However, to ensure `src` is importable when running `streamlit run src/ui/app.py`, we explicitly add the root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.core.category_manager import CategoryManager
from src.core.constants import CATEGORIES_CONFIG_PATH

# ページ設定
st.set_page_config(
    page_title="傷分類システム",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS（外部ファイルから読み込み）
_css_path = Path(__file__).parent / "styles" / "main.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _get_category_manager() -> CategoryManager:
    """CategoryManagerをセッションから取得（未初期化なら初期化）"""
    if "category_manager" not in st.session_state:
        st.session_state.category_manager = CategoryManager(CATEGORIES_CONFIG_PATH)
    return st.session_state.category_manager


def main():
    """メインエントリポイント"""
    # ヘッダー
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">🔍 傷分類システム</div>
            <div class="app-subtitle">機械学習による製品検品支援システム</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # CategoryManager を事前初期化
    _get_category_manager()

    # サイドバー
    with st.sidebar:
        st.markdown("### 📋 メニュー")
        page = st.radio(
            "ページを選択",
            ["📥 受信トレイ", "📂 データセット", "📚 学習", "📊 評価", "🎯 分類", "⚙️ 設定"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 📌 システム情報")
        st.info("Version: 1.2.0")

    # ページルーティング
    if page == "📥 受信トレイ":
        from src.ui.views.inbox import show_inbox_page

        show_inbox_page()
    elif page == "📂 データセット":
        from src.ui.views.dataset import show_dataset_page

        show_dataset_page()
    elif page == "📚 学習":
        from src.ui.views.training import show_training_page

        show_training_page()
    elif page == "📊 評価":
        from src.ui.views.evaluation import show_evaluation_page

        show_evaluation_page()
    elif page == "🎯 分類":
        from src.ui.views.classify import show_classify_page

        show_classify_page()
    elif page == "⚙️ 設定":
        from src.ui.views.settings import show_settings_page

        show_settings_page()


if __name__ == "__main__":
    main()
