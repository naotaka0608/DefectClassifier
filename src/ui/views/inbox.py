"""受信トレイ画面"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG
from src.core.constants import ANNOTATIONS_FILE, INBOX_DIR, TRAIN_IMAGES_DIR


def show_inbox_page():
    """受信トレイページを表示"""
    st.markdown("## 📥 受信トレイ")
    st.markdown("API経由で受信した画像を確認・修正して学習データに追加します。")

    # ディレクトリ作成（念のため）
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # カテゴリマネージャー初期化
    if "category_manager" not in st.session_state:
        st.session_state.category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)
    category_manager = st.session_state.category_manager

    # 画像リスト取得
    # jsonファイルとペアになっているjpgを探す
    json_files = sorted(list(INBOX_DIR.glob("*.json")), reverse=True)
    
    if not json_files:
        st.info("受信トレイは空です。")
        return

    # サイドバーにリスト表示
    st.sidebar.markdown("### 📨 受信リスト")
    selected_json_path = st.sidebar.radio(
        "画像を選択",
        json_files,
        format_func=lambda p: p.stem,
        key="inbox_selection"
    )

    if selected_json_path:
        _show_detail_view(selected_json_path, category_manager)


def _show_detail_view(json_path: Path, category_manager: CategoryManager):
    """詳細ビューを表示"""
    # データの読み込み
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        # 画像パスの解決（絶対パスが変わっている可能性を考慮してファイル名から再構築）
        image_filename = Path(metadata["image_path"]).name
        image_path = INBOX_DIR / image_filename
        
        if not image_path.exists():
            st.error(f"画像ファイルが見つかりません: {image_filename}")
            return
            
        image = Image.open(image_path)
        
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption=f"{json_path.stem}", use_container_width=True)
        
        st.markdown("### 📋 メタデータ")
        st.json(metadata["prediction"], expanded=False)
        st.text(f"Timestamp: {metadata.get('timestamp')}")
        st.text(f"Request ID: {metadata.get('request_id')}")

    with col2:
        st.markdown("### ✏️ ラベル修正・登録")
        
        # 現在の予測値をデフォルトにする
        pred = metadata.get("prediction", {}).get("cause", {})
        # cause, shape, depth構造がAPIレスポンスによって違うかも。
        # PredictResponseは:
        # success, cause={label, confidence...}, shape=..., depth=.
        
        current_cause = metadata.get("prediction", {}).get("cause", {}).get("label")
        current_shape = metadata.get("prediction", {}).get("shape", {}).get("label")
        current_depth = metadata.get("prediction", {}).get("depth", {}).get("label")
        
        # フォーム
        with st.form(key=f"label_form_{json_path.stem}"):
            new_cause = st.selectbox(
                "原因 (Cause)",
                category_manager.get_categories("cause"),
                index=_get_index(category_manager.get_categories("cause"), current_cause)
            )
            
            new_shape = st.selectbox(
                "形状 (Shape)",
                category_manager.get_categories("shape"),
                index=_get_index(category_manager.get_categories("shape"), current_shape)
            )
            
            new_depth = st.selectbox(
                "深さ (Depth)",
                category_manager.get_categories("depth"),
                index=_get_index(category_manager.get_categories("depth"), current_depth)
            )
            
            submitted = st.form_submit_button("✅ データセットに追加")
            
        if submitted:
            _add_to_dataset(image_path, json_path, new_cause, new_shape, new_depth)
            
        st.markdown("---")
        if st.button("🗑️ 削除", type="primary"):
            _delete_item(image_path, json_path)


def _get_index(options, value):
    try:
        return options.index(value)
    except (ValueError, IndexError):
        return 0


def _add_to_dataset(image_path: Path, json_path: Path, cause, shape, depth):
    """データセットに追加処理"""
    try:
        # 1. 画像の移動（リネームして衝突回避）
        new_filename = image_path.name
        # もし同名ファイルがあればタイムスタンプなどを付与するなどすべきだが、
        # 今回はUUID付きなので基本大丈夫。念のためチェック
        target_image_path = TRAIN_IMAGES_DIR / new_filename
        if target_image_path.exists():
            st.warning("同名のファイルが存在します。")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            target_image_path = TRAIN_IMAGES_DIR / f"{timestamp}_{new_filename}"
            
        shutil.move(str(image_path), str(target_image_path))
        
        # 2. アノテーションの追記
        annotation = {
            "file_name": target_image_path.name,
            "cause": cause,
            "shape": shape,
            "depth": depth,
            "added_at": datetime.now().isoformat(),
            "source": "inbox"
        }
        
        annotations = []
        if ANNOTATIONS_FILE.exists():
            with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
                try:
                    annotations = json.load(f)
                    if not isinstance(annotations, list): # 古い形式対応
                        annotations = []
                except json.JSONDecodeError:
                    annotations = []
                    
        annotations.append(annotation)
        
        with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
            
        # 3. 元のJSON削除
        json_path.unlink()
        
        st.success(f"データセットに追加しました: {target_image_path.name}")
        st.rerun()
        
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")


def _delete_item(image_path: Path, json_path: Path):
    """アイテムを削除"""
    try:
        if image_path.exists():
            image_path.unlink()
        if json_path.exists():
            json_path.unlink()
        st.success("削除しました")
        st.rerun()
    except Exception as e:
        st.error(f"削除に失敗しました: {e}")
