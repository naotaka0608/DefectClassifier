"""画像表示コンポーネント"""
from pathlib import Path
import streamlit as st
from PIL import Image

def image_viewer(image_path: Path | str, caption: str = "", stretch: bool = True):
    """
    画像を表示するコンポーネント（エラーハンドリング付き）
    
    Args:
        image_path (Path | str): 画像ファイルのパス
        caption (str): 画像のキャプション
        stretch (bool): コンテナ幅いっぱいに表示するかどうか
    """
    path_obj = Path(image_path)
    
    if not path_obj.exists():
        st.error(f"画像ファイルが見つかりません: {path_obj.name}")
        return

    try:
        image = Image.open(path_obj)
        width = "stretch" if stretch else "content"
        st.image(image, caption=caption, width=width)
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
