"""データセット閲覧画面"""

import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG
from src.core.constants import ANNOTATIONS_FILE, TRAIN_IMAGES_DIR


def show_dataset_page():
    """データセットページを表示"""
    st.markdown("## 📂 データセット")
    st.markdown("学習に使用するデータセットを確認します。")

    # カテゴリマネージャー初期化
    if "category_manager" not in st.session_state:
        st.session_state.category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)
    category_manager = st.session_state.category_manager

    # 画像アップロードセクション
    _show_upload_section(category_manager)

    st.markdown("---")

    # アノテーションデータの読み込み
    annotations = _load_annotations()
    
    if not annotations:
        st.info("データセットは空です。画像をアップロードしてください。")
        return

    # DataFrameに変換して操作しやすくする
    df = pd.DataFrame(annotations)
    
    # サイドバーでフィルタリング
    st.sidebar.markdown("### 🔍 フィルター")
    
    # フィルタ条件の保存
    filters = {}
    
    # 各タスクのフィルタ作成
    task_names = {"cause": "原因", "shape": "形状", "depth": "深さ"}
    for task, name in task_names.items():
        categories = ["すべて"] + category_manager.get_categories(task)
        selected = st.sidebar.selectbox(f"{name}", categories, key=f"filter_{task}")
        if selected != "すべて":
            filters[task] = selected

    # フィルタリング実行
    filtered_df = df.copy()
    for task, value in filters.items():
        filtered_df = filtered_df[filtered_df[task] == value]
        
    st.sidebar.markdown(f"**該当件数:** {len(filtered_df)} / {len(df)}")

    if len(filtered_df) == 0:
        st.warning("条件に一致するデータはありません。")
        return

    # メインエリア表示
    # 2カラムレイアウト: リストと詳細
    
    # リスト表示（画像選択用）
    # 画像ファイル名とラベル情報を結合して表示用テキストを作成
    filtered_df["display_label"] = filtered_df.apply(
        lambda x: f"{x['cause']} / {x['shape']} / {x['depth']} ({x['file_name']})", axis=1
    )
    
    # 選択ボックス
    selected_index = st.selectbox(
        "画像を選択",
        filtered_df.index,
        format_func=lambda i: filtered_df.loc[i, "display_label"]
    )
    
    # 詳細表示
    if selected_index is not None:
        row = filtered_df.loc[selected_index]
        _show_image_detail(row)


def _show_upload_section(category_manager: CategoryManager):
    """画像アップロードセクションを表示"""
    with st.expander("📤 画像をアップロードして追加", expanded=False):
        uploaded_files = st.file_uploader(
            "画像ファイルを選択 (複数可)", 
            type=["jpg", "jpeg", "png", "bmp"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            st.markdown("#### ラベルの一括指定")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cause = st.selectbox("原因 (Cause)", category_manager.get_categories("cause"), key="upload_cause")
            with col2:
                shape = st.selectbox("形状 (Shape)", category_manager.get_categories("shape"), key="upload_shape")
            with col3:
                depth = st.selectbox("深さ (Depth)", category_manager.get_categories("depth"), key="upload_depth")

            if st.button(f"💾 {len(uploaded_files)}枚の画像をデータセットに追加", type="primary"):
                _save_uploaded_images(uploaded_files, cause, shape, depth)
                st.success(f"{len(uploaded_files)}枚の画像を追加しました！")
                st.rerun()


def _save_uploaded_images(uploaded_files, cause, shape, depth):
    """アップロードされた画像を保存し、アノテーションを更新"""
    # ディレクトリ作成
    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 既存アノテーション読み込み
    annotations = _load_annotations()
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for file in uploaded_files:
        # 一意なファイル名を生成
        file_ext = Path(file.name).suffix
        unique_name = f"{uuid.uuid4().hex}{file_ext}"
        save_path = TRAIN_IMAGES_DIR / unique_name
        
        # 画像保存
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
            
        # アノテーションデータ作成
        # DefectDatasetに合わせて image_path を保存 (dataディレクトリからの相対パス)
        relative_image_path = f"processed/train/images/{unique_name}"
        
        annotation = {
            "image_path": relative_image_path,
            "file_name": unique_name, # UI表示用に保持
            "original_name": file.name,
            "cause": cause,
            "shape": shape,
            "depth": depth,
            "source": "upload",
            "added_at": current_time
        }
        annotations.append(annotation)
    
    # アノテーション保存 (辞書形式で保存)
    with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump({"samples": annotations}, f, ensure_ascii=False, indent=2)


def _load_annotations():
    """アノテーションファイルを読み込む"""
    if not ANNOTATIONS_FILE.exists():
        return []
    
    try:
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # リスト形式（旧形式）の場合
            if isinstance(data, list):
                # マイグレーション：image_pathがない場合は追加
                for item in data:
                    if "image_path" not in item and "file_name" in item:
                        item["image_path"] = f"processed/train/images/{item['file_name']}"
                return data
                
            # 辞書形式（新形式）の場合
            if isinstance(data, dict) and "samples" in data:
                return data["samples"]
                
            return []
    except Exception:
        return []


def _show_image_detail(row):
    """画像詳細を表示"""
    image_path = TRAIN_IMAGES_DIR / row["file_name"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if image_path.exists():
            image = Image.open(image_path)
            st.image(image, caption=row["file_name"], use_container_width=True)
        else:
            st.error(f"画像ファイルが見つかりません: {row['file_name']}")
            
    with col2:
        st.markdown("### 🏷️ ラベル情報")
        
        # 見やすいカード形式で表示
        _info_card("原因 (Cause)", row["cause"], "#667eea")
        _info_card("形状 (Shape)", row["shape"], "#764ba2")
        _info_card("深さ (Depth)", row["depth"], "#f093fb")
        
        st.markdown("---")
        st.markdown("### ℹ️ メタデータ")
        st.text(f"追加日時: {row.get('added_at', '不明')}")
        st.text(f"ソース: {row.get('source', '不明')}")
        if "original_name" in row:
             st.text(f"元ファイル名: {row['original_name']}")


def _info_card(title, value, color):
    """情報カードを表示"""
    st.markdown(
        f"""
        <div style="
            background-color: {color}20;
            border-left: 5px solid {color};
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        ">
            <div style="font-size: 0.8em; color: gray;">{title}</div>
            <div style="font-size: 1.2em; font-weight: bold;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
