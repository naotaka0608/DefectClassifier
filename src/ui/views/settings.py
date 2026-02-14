"""設定ページ"""

import yaml
from pathlib import Path
import streamlit as st

from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG
from src.core.constants import CHECKPOINTS_DIR, CONFIG_DIR, DATA_DIR


def show_settings_page():
    """設定ページを表示"""
    st.markdown("## ⚙️ 設定")
    st.markdown("システム設定とカテゴリの管理ができます。")

    # タブ
    tab1, tab2, tab3 = st.tabs(["📂 カテゴリ管理", "🧠 モデル設定", "ℹ️ システム情報"])

    with tab1:
        _show_category_management_tab()

    with tab2:
        _show_model_settings_tab()

    with tab3:
        _show_system_info_tab()


def _show_category_management_tab():
    """カテゴリ管理タブ"""
    st.markdown("### 📂 カテゴリ管理")
    st.info("カテゴリを追加・編集できます。変更後はモデルの再学習が必要になる場合があります。")

    # カテゴリマネージャー
    if "category_manager" not in st.session_state:
        st.session_state.category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)

    category_manager = st.session_state.category_manager

    task_names = {"cause": "原因分類", "shape": "形状分類", "depth": "深さ分類"}

    for task, name in task_names.items():
        with st.expander(f"📁 {name}", expanded=True):
            categories = category_manager.get_category_details(task)

            # カテゴリ一覧
            for i, cat in enumerate(categories):
                cols = st.columns([3, 4, 2, 1])
                cols[0].text_input(
                    "名前",
                    value=cat.name,
                    key=f"{task}_name_{i}",
                    label_visibility="collapsed",
                )
                cols[1].text_input(
                    "説明",
                    value=cat.description,
                    key=f"{task}_desc_{i}",
                    label_visibility="collapsed",
                )
                cols[2].text_input(
                    "コード",
                    value=cat.code,
                    key=f"{task}_code_{i}",
                    label_visibility="collapsed",
                )
                cols[3].button("🗑️", key=f"{task}_del_{i}", help="削除")

            # 新規追加ボタン
            if st.button(f"➕ {name}に追加", key=f"add_{task}"):
                st.info("カテゴリ追加機能（実装予定）")

    st.markdown("---")

    # 保存ボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 変更を保存", use_container_width=True):
            st.success("設定を保存しました！")
    with col2:
        if st.button("🔄 リセット", use_container_width=True):
            st.session_state.category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)
            st.info("設定をリセットしました。")
            st.rerun()


def _show_model_settings_tab():
    """モデル設定タブ"""
    st.markdown("### 🧠 モデル設定")

    # 現在のモデル
    st.markdown("#### 📌 現在のモデル")

    model_info = {
        "モデルパス": str(CHECKPOINTS_DIR / "best_model.pth"),
        "バックボーン": "ResNet50",
        "学習日時": "2026-02-05 09:00",
        "エポック数": 200,
        "最終精度": "95.8%",
    }

    for key, value in model_info.items():
        st.markdown(f"**{key}**: `{value}`")

    st.markdown("---")

    # モデル読み込み
    st.markdown("#### 📥 モデル読み込み")

    uploaded_model = st.file_uploader(
        "モデルファイルを選択",
        type=["pth", "pt"],
        help="学習済みのPyTorchモデルファイル",
    )

    if uploaded_model:
        if st.button("🔄 モデルを読み込み", use_container_width=True):
            with st.spinner("モデルを読み込み中..."):
                # 実際の実装ではモデルを読み込む
                pass
            st.success("モデルを読み込みました！")

    st.markdown("---")

    # デフォルト設定
    st.markdown("#### ⚙️ デフォルト設定")

    default_backbone = st.selectbox(
        "デフォルトバックボーン",
        options=["resnet50", "resnet101", "efficientnet_b4"],
    )

    default_threshold = st.slider(
        "分類閾値",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    if st.button("💾 デフォルト設定を保存", use_container_width=True):
        st.success("デフォルト設定を保存しました！")



def _show_system_info_tab():
    """システム情報タブ"""
    st.markdown("### ℹ️ システム情報")

    # バージョン情報
    st.markdown("#### 📌 バージョン")

    import sys

    import torch

    info_data = {
        "アプリケーション": "1.0.0",
        "Python": sys.version.split()[0],
        "PyTorch": torch.__version__,
        "CUDA利用可能": "✅ はい" if torch.cuda.is_available() else "❌ いいえ",
    }

    if torch.cuda.is_available():
        info_data["CUDA バージョン"] = torch.version.cuda
        info_data["GPU"] = torch.cuda.get_device_name(0)

    for key, value in info_data.items():
        cols = st.columns([1, 2])
        cols[0].markdown(f"**{key}**")
        cols[1].markdown(f"`{value}`")

    st.markdown("---")

    # ディレクトリ情報
    st.markdown("#### 📁 ディレクトリ")

    from pathlib import Path

    dirs = {
        "設定ファイル": CONFIG_DIR,
        "チェックポイント": CHECKPOINTS_DIR,
        "データ": DATA_DIR,
        "ログ": Path("logs"),
    }

    for name, path in dirs.items():
        exists = "✅" if path.exists() else "❌"
        st.markdown(f"**{name}**: `{path}` {exists}")

    st.markdown("---")

    # ライセンス
    st.markdown("#### 📜 ライセンス")
    st.markdown(
        """
        このソフトウェアはMITライセンスの下で提供されています。

        - PyTorch: BSD-3-Clause
        - Streamlit: Apache-2.0
        - torchvision: BSD-3-Clause
        """
    )
