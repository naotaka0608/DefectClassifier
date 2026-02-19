from typing import Union
from pathlib import Path
import streamlit as st
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

def image_viewer(image_source: Union[Path, str, Image.Image, UploadedFile], caption: str = "", stretch: bool = True, width: int | None = None):
    """
    画像を表示するコンポーネント（エラーハンドリング付き）
    
    Args:
        image_source: 画像ソース (パス, PIL Image, UploadedFile)
        caption (str): 画像のキャプション
        stretch (bool): コンテナ幅いっぱいに表示するかどうか (widthが指定された場合は無視)
        width (int): 画像の表示幅 (ピクセル)
    """
    
    try:
        # widthが指定されている場合はそれを使用
        # stretch=Trueかつwidth=Noneの場合はuse_container_width=True
        use_container_width = False
        if width is None:
            if stretch:
                use_container_width = True
        
        if isinstance(image_source, (str, Path)):
            path_obj = Path(image_source)
            if not path_obj.exists():
                st.error(f"画像ファイルが見つかりません: {path_obj.name}")
                return
            image = Image.open(path_obj)
            st.image(image, caption=caption, use_container_width=use_container_width, width=width)
            
        elif isinstance(image_source, Image.Image):
            st.image(image_source, caption=caption, use_container_width=use_container_width, width=width)
            
        elif isinstance(image_source, UploadedFile):
            # UploadedFileはseek(0)が必要かもしれないが、PIL.openが処理する
            # ただし、既にreadされている場合はseek(0)が必要
            image_source.seek(0)
            image = Image.open(image_source)
            st.image(image, caption=caption, use_container_width=use_container_width, width=width)
            
        else:
            st.error(f"サポートされていない画像ソースです: {type(image_source)}")
            
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
