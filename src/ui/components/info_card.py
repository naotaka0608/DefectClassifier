import streamlit as st

def info_card(title: str, value: str, color: str):
    """
    情報カードを表示するコンポーネント
    
    Args:
        title (str): カードのタイトル（ラベル名など）
        value (str): 表示する値
        color (str): 左側のボーダーと背景に使用するカラーコード
    """
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
