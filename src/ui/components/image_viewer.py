import streamlit as st
from PIL import Image
from pathlib import Path

def image_viewer(image_path: Path | str, caption: str = "", use_container_width: bool = True):
    """
    画像を表示するコンポーネント（エラーハンドリング付き）
    
    Args:
        image_path (Path | str): 画像ファイルのパス
        caption (str): 画像のキャプション
        use_container_width (bool): コンテナ幅いっぱいに表示するかどうか
    """
    path_obj = Path(image_path)
    
    if not path_obj.exists():
        st.error(f"画像ファイルが見つかりません: {path_obj.name}")
        return

    try:
        image = Image.open(path_obj)
        st.image(image, caption=caption, use_container_width=use_container_width)
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
