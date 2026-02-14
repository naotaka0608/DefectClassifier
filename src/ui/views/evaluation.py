"""評価ページ"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG


def show_evaluation_page():
    """評価ページを表示"""
    st.markdown("## 📊 モデル評価")
    st.markdown("学習済みモデルの性能を評価できます。")

    # カテゴリマネージャー
    if "category_manager" not in st.session_state:
        st.session_state.category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)

    category_manager = st.session_state.category_manager

    # タブ
    tab1, tab2, tab3 = st.tabs(["📈 サマリー", "🎯 混同行列", "📋 詳細分析"])

    with tab1:
        _show_summary_tab(category_manager)

    with tab2:
        _show_confusion_matrix_tab(category_manager)

    with tab3:
        _show_detailed_analysis_tab(category_manager)


def _show_summary_tab(category_manager: CategoryManager):
    """サマリータブ"""
    st.markdown("### 📈 評価サマリー")

    # メトリクスカード
    cols = st.columns(4)

    metrics = [
        ("全体精度", "95.2%", "↑ 2.1%", "#667eea"),
        ("原因分類", "94.8%", "↑ 1.5%", "#764ba2"),
        ("形状分類", "96.1%", "↑ 2.8%", "#f093fb"),
        ("深さ分類", "94.7%", "↑ 1.9%", "#5ee7df"),
    ]

    for col, (name, value, delta, color) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {color} 0%, {color}99 100%);
                    padding: 1.5rem;
                    border-radius: 1rem;
                    color: white;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                ">
                    <div style="font-size: 0.9rem; opacity: 0.9;">{name}</div>
                    <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{value}</div>
                    <div style="font-size: 0.85rem; color: #a0ffa0;">{delta}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # クラス別精度
    st.markdown("### 🎯 クラス別精度")

    task_names = {"cause": "原因", "shape": "形状", "depth": "深さ"}

    for task, name in task_names.items():
        categories = category_manager.get_categories(task)
        accuracies = np.random.uniform(0.85, 0.98, len(categories))

        fig = go.Figure(
            data=[
                go.Bar(
                    x=categories,
                    y=[a * 100 for a in accuracies],
                    marker_color=[
                        f"hsl({i * 360 / len(categories)}, 70%, 60%)"
                        for i in range(len(categories))
                    ],
                    text=[f"{a * 100:.1f}%" for a in accuracies],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title=f"{name}分類のクラス別精度",
            xaxis_title="カテゴリ",
            yaxis_title="精度 (%)",
            yaxis_range=[0, 100],
            height=300,
            margin=dict(l=40, r=40, t=60, b=40),
        )

        st.plotly_chart(fig, width="stretch")


def _show_confusion_matrix_tab(category_manager: CategoryManager):
    """混同行列タブ"""
    st.markdown("### 🎯 混同行列")

    task_names = {"cause": "原因", "shape": "形状", "depth": "深さ"}
    selected_task = st.selectbox(
        "分類タスクを選択",
        options=list(task_names.keys()),
        format_func=lambda x: task_names[x],
    )

    categories = category_manager.get_categories(selected_task)
    n_classes = len(categories)

    # ダミーの混同行列
    np.random.seed(42)
    cm = np.random.randint(5, 100, size=(n_classes, n_classes))
    np.fill_diagonal(cm, np.random.randint(150, 250, size=n_classes))

    # 正規化
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # ヒートマップ
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_normalized,
            x=categories,
            y=categories,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="実際: %{y}<br>予測: %{x}<br>件数: %{text}<br>割合: %{z:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{task_names[selected_task]}分類の混同行列",
        xaxis_title="予測ラベル",
        yaxis_title="実際のラベル",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, width="stretch")

    # 統計情報
    st.markdown("### 📊 統計情報")

    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total

    cols = st.columns(3)
    cols[0].metric("総サンプル数", f"{total:,}")
    cols[1].metric("正解数", f"{correct:,}")
    cols[2].metric("全体精度", f"{accuracy * 100:.1f}%")


def _show_detailed_analysis_tab(category_manager: CategoryManager):
    """詳細分析タブ"""
    st.markdown("### 📋 詳細分析")

    task_names = {"cause": "原因", "shape": "形状", "depth": "深さ"}

    for task, name in task_names.items():
        categories = category_manager.get_categories(task)

        st.markdown(f"#### {name}分類")

        # ダミーデータ
        data = []
        for cat in categories:
            precision = np.random.uniform(0.85, 0.98)
            recall = np.random.uniform(0.82, 0.97)
            f1 = 2 * precision * recall / (precision + recall)
            support = np.random.randint(100, 500)
            data.append(
                {
                    "カテゴリ": cat,
                    "Precision": f"{precision * 100:.1f}%",
                    "Recall": f"{recall * 100:.1f}%",
                    "F1 Score": f"{f1 * 100:.1f}%",
                    "サンプル数": support,
                }
            )

        import pandas as pd

        df = pd.DataFrame(data)
        st.dataframe(df, width="stretch", hide_index=True)

        st.markdown("---")

    # エクスポートボタン
    st.markdown("### 📥 レポートエクスポート")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📄 CSVでエクスポート",
            data="dummy,csv,data",
            file_name="evaluation_report.csv",
            mime="text/csv",
            width="stretch",
        )

    with col2:
        st.download_button(
            "📊 PDFでエクスポート",
            data=b"dummy_pdf_data",
            file_name="evaluation_report.pdf",
            mime="application/pdf",
            width="stretch",
        )
