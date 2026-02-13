"""Streamlit メインアプリケーション"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

# ページ設定
st.set_page_config(
    page_title="傷分類システム",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS
st.markdown(
    """
    <style>
    /* メインコンテナ */
    .main {
        padding: 1rem 2rem;
    }

    /* サイドバー */
    .css-1d391kg {
        padding-top: 2rem;
    }

    /* メトリクスカード */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* 結果カード */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }

    /* ボタンスタイル */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: transform 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }

    /* タブスタイル */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
    }

    /* 進捗バー */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* ヘッダー */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }

    .app-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .app-subtitle {
        font-size: 1.1rem;
        opacity: 0.8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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

    # サイドバー
    with st.sidebar:
        st.markdown("### 📋 メニュー")
        page = st.radio(
            "ページを選択",
            ["🎯 分類", "📥 受信トレイ", "📂 データセット", "📚 学習", "📊 評価", "⚙️ 設定"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 📌 システム情報")
        st.info("Version: 1.2.0")

    # ページルーティング
    if page == "🎯 分類":
        from src.ui.views.classify import show_classify_page

        show_classify_page()
    elif page == "📥 受信トレイ":
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
    elif page == "⚙️ 設定":
        from src.ui.views.settings import show_settings_page

        show_settings_page()


if __name__ == "__main__":
    main()
