"""分類ページ"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG
from src.ui.components.image_viewer import image_viewer


def show_classify_page():
    """分類ページを表示"""
    st.markdown("## 🎯 画像分類")
    st.markdown("傷の画像をアップロードして分類結果を確認できます。")

    # カテゴリマネージャー初期化
    if "category_manager" not in st.session_state:
        st.session_state.category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)

    category_manager = st.session_state.category_manager

    # レイアウト
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 画像アップロード")

        uploaded_file = st.file_uploader(
            "画像を選択してください",
            type=["jpg", "jpeg", "png", "bmp"],
            help="対応形式: JPG, PNG, BMP",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_viewer(uploaded_file, caption="アップロード画像")

            # 分類実行ボタン
            if st.button("🔍 分類を実行", width="stretch"):
                _run_classification(image, category_manager)

    with col2:
        st.markdown("### 📊 分類結果")

        if "classification_result" in st.session_state:
            result = st.session_state.classification_result
            probs = st.session_state.classification_probs

            # 結果表示
            _display_results(result, probs, category_manager)
        else:
            st.info("画像をアップロードして分類を実行してください。")


def _run_classification(image: Image.Image, category_manager: CategoryManager):
    """分類を実行（デモ用のダミー結果）"""
    with st.spinner("分類中..."):
        # 実際の実装ではモデルを使用
        # ここではデモ用のダミーデータを生成
        import random

        result = {
            "cause": random.choice(category_manager.get_categories("cause")),
            "shape": random.choice(category_manager.get_categories("shape")),
            "depth": random.choice(category_manager.get_categories("depth")),
        }

        probs = {}
        for task in ["cause", "shape", "depth"]:
            categories = category_manager.get_categories(task)
            raw_probs = np.random.dirichlet(np.ones(len(categories)))
            probs[task] = {cat: float(p) for cat, p in zip(categories, raw_probs)}
            # 選択されたカテゴリの確率を高く
            max_cat = max(probs[task], key=probs[task].get)
            result[task] = max_cat

        st.session_state.classification_result = result
        st.session_state.classification_probs = probs

    st.success("分類が完了しました！")
    st.rerun()


def _display_results(result: dict, probs: dict, category_manager: CategoryManager):
    """結果を表示"""
    # メトリクスカード
    cols = st.columns(3)

    task_names = {"cause": "原因", "shape": "形状", "depth": "深さ"}
    task_icons = {"cause": "⚡", "shape": "📐", "depth": "📏"}
    task_colors = {"cause": "#667eea", "shape": "#764ba2", "depth": "#f093fb"}

    for i, (task, name) in enumerate(task_names.items()):
        with cols[i]:
            label = result[task]
            confidence = probs[task][label] * 100

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {task_colors[task]} 0%, {task_colors[task]}99 100%);
                    padding: 1.5rem;
                    border-radius: 1rem;
                    color: white;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                ">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{task_icons[task]}</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">{name}</div>
                    <div style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{label}</div>
                    <div style="font-size: 0.85rem; opacity: 0.8;">{confidence:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # 確率分布グラフ
    st.markdown("### 📈 確率分布")

    tabs = st.tabs(["原因", "形状", "深さ"])

    for tab, (task, name) in zip(tabs, task_names.items()):
        with tab:
            categories = list(probs[task].keys())
            values = list(probs[task].values())

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=categories,
                        y=[v * 100 for v in values],
                        marker_color=[
                            task_colors[task] if cat == result[task] else "#e0e0e0"
                            for cat in categories
                        ],
                        text=[f"{v * 100:.1f}%" for v in values],
                        textposition="auto",
                    )
                ]
            )

            fig.update_layout(
                title=f"{name}分類の確率分布",
                xaxis_title="カテゴリ",
                yaxis_title="確率 (%)",
                yaxis_range=[0, 100],
                showlegend=False,
                height=300,
                margin=dict(l=40, r=40, t=60, b=40),
            )

            st.plotly_chart(fig, width="stretch")
