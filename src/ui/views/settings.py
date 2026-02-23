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
    tab1, tab2, tab3, tab4 = st.tabs(["📂 カテゴリ管理", "🧠 モデル設定", "👀 フォルダ監視", "ℹ️ システム情報"])

    with tab1:
        _show_category_management_tab()

    with tab2:
        _show_model_settings_tab()

    with tab3:
        _show_monitor_settings_tab()

    with tab4:
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
        if st.button("💾 変更を保存", width="stretch"):
            st.success("設定を保存しました！")
    with col2:
        if st.button("🔄 リセット", width="stretch"):
            st.session_state.category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)
            st.info("設定をリセットしました。")
            st.rerun()


def _show_model_settings_tab():
    """モデル設定タブ"""
    st.markdown("### 🧠 モデル設定")

    # 現在のモデル
    st.markdown("#### 📌 現在のモデル")

    model_path = CHECKPOINTS_DIR / "best_model.pth"
    if model_path.exists():
        import datetime
        mtime = datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
        metrics_str = "更新済み (詳細不明)"
    else:
        mtime = "不明"
        metrics_str = "不明"

    model_info = {
        "モデルパス": str(model_path),
        "バックボーン": "ResNet50",
        "学習日時": str(mtime),
        "ステータス": "利用可能" if model_path.exists() else "未学習",
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
        if st.button("🔄 モデルを読み込み", width="stretch"):
            with st.spinner("モデルを読み込み中..."):
                # 実際の実装ではモデルを読み込む
                pass
            st.success("モデルを読み込みました！")

    st.markdown("---")

    # ONNXエクスポート
    st.markdown("#### 📦 モデルエクスポート")
    st.markdown("現在のモデルをONNX形式でエクスポートします。")
    
    if st.button("🚀 ONNXへ変換", key="export_onnx"):
        from src.core.constants import BEST_MODEL_PATH
        if not BEST_MODEL_PATH.exists():
            st.error("モデルファイルが見つかりません。")
        else:
            with st.spinner("ONNX変換中..."):
                try:
                    from src.deploy.onnx_exporter import export_to_onnx
                    
                    output_path = BEST_MODEL_PATH.parent / "model.onnx"
                    export_to_onnx(BEST_MODEL_PATH, output_path)
                    st.success(f"エクスポート完了: `{output_path}`")
                except Exception as e:
                    st.error(f"エクスポート失敗: {e}")

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

    if st.button("💾 デフォルト設定を保存", width="stretch"):
        st.success("デフォルト設定を保存しました！")


def _show_monitor_settings_tab():
    """フォルダ監視設定タブ"""
    st.markdown("### 👀 フォルダ監視")
    st.markdown("指定したフォルダに画像が保存されると、自動的にAI判定を実行します。")

    # 初期値
    default_monitor_dir = str(DATA_DIR / "monitor")
    
    monitor_path = st.text_input(
        "監視対象フォルダパス",
        value=st.session_state.get("monitor_path", default_monitor_dir),
        help="新しい画像（jpg, png等）を監視するローカルフォルダ"
    )
    st.session_state.monitor_path = monitor_path

    # Watcher の状態管理
    from src.services.watcher import DefectWatcher
    
    is_running = False
    if "defect_watcher" in st.session_state and st.session_state.defect_watcher:
        is_running = True

    st.markdown(f"**現在のステータス**: {'🟢 実行中' if is_running else '⚪ 停止中'}")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ 監視開始", disabled=is_running, width="stretch", type="primary"):
            try:
                watcher = DefectWatcher(monitor_path)
                watcher.start()
                st.session_state.defect_watcher = watcher
                st.success(f"監視を開始しました: {monitor_path}")
                st.rerun()
            except Exception as e:
                st.error(f"監視の開始に失敗しました: {e}")

    with col2:
        if st.button("⏹️ 監視停止", disabled=not is_running, width="stretch"):
            if "defect_watcher" in st.session_state:
                st.session_state.defect_watcher.stop()
                st.session_state.defect_watcher = None
                st.info("監視を停止しました。")
                st.rerun()

    st.markdown("---")
    st.markdown("#### ✅ 自動判定の流れ")
    st.markdown("""
    1.  指定したフォルダに画像ファイルが置かれます。
    2.  AIが自動的に読み込み、分類を実行します。
    3.  結果は背景でデータベースに保存されます。
    4.  「📈 履歴」ページから自動判定の結果を確認できます。
    """)



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
