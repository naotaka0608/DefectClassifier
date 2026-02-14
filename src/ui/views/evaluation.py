"""評価ページ"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG
from src.core.constants import ANNOTATIONS_FILE, CHECKPOINTS_DIR, DATA_DIR, BEST_MODEL_PATH, FINAL_MODEL_PATH
from src.core.data_manager import DataManager
from src.core.types import TaskType
from src.inference.predictor import DefectPredictor
from src.training.runner import train_model  # for type hint if needed, or remove


def show_evaluation_page():
    """評価ページを表示"""
    st.markdown("## 📊 モデル評価")
    st.markdown("学習済みモデルの性能を評価できます。")

    # カテゴリマネージャー
    if "category_manager" not in st.session_state:
        st.session_state.category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)

    category_manager = st.session_state.category_manager

    # 評価実行ボタン
    if "evaluation_results" not in st.session_state:
        if st.button("評価を実行する", type="primary"):
            with st.spinner("評価を実行中... (これには時間がかかる場合があります)"):
                try:
                    results = _run_evaluation(category_manager)
                    st.session_state.evaluation_results = results
                    st.success("評価が完了しました！")
                    st.rerun()
                except Exception as e:
                    st.error(f"評価中にエラーが発生しました: {e}")
            return
    else:
        if st.button("再評価する"):
            del st.session_state.evaluation_results
            st.rerun()

    if "evaluation_results" in st.session_state:
        results = st.session_state.evaluation_results
        
        # タブ
        tab1, tab2, tab3 = st.tabs(["📈 サマリー", "🎯 混同行列", "📋 詳細分析"])

        with tab1:
            _show_summary_tab(results, category_manager)

        with tab2:
            _show_confusion_matrix_tab(results, category_manager)

        with tab3:
            _show_detailed_analysis_tab(results, category_manager)


def _run_evaluation(category_manager: CategoryManager) -> dict[str, Any]:
    """評価を実行して結果を返す"""
    
    # データ読み込みと分割 (runner.pyと同じロジック)
    data_manager = DataManager(ANNOTATIONS_FILE)
    all_samples = data_manager.load_annotations()
    
    if len(all_samples) == 0:
        raise ValueError("データがありません。")

    from src.core.data_utils import split_dataset
    _, val_samples = split_dataset(all_samples, train_ratio=0.8, seed=42)

    if not val_samples:
         raise ValueError("検証用データが不足しています。")

    # モデル読み込み
    model_path = BEST_MODEL_PATH
    if not model_path.exists():
         model_path = FINAL_MODEL_PATH
         if not model_path.exists():
             raise FileNotFoundError("学習済みモデルが見つかりません。")

    predictor = DefectPredictor(model_path=model_path, category_manager=category_manager)
    
    # 推論実行
    # 推論実行
    images = []
    true_labels = {TaskType.CAUSE: [], TaskType.SHAPE: [], TaskType.DEPTH: []}
    
    for sample in val_samples:
        # 画像パス解決
        if "image_path" in sample:
            img_path = DATA_DIR / sample["image_path"]
        else:
             # フォールバック (dataset.pyと同様)
             rel_path = DATA_DIR / "train_images" # 仮
             # 実際の構成に合わせて調整が必要だが、dataset.pyのロジックを見ると
             # TRAIN_IMAGES_DIR.relative_to(DATA_DIR) を使っている。
             # 簡略化のため、絶対パスを構築して存在確認
             # DataManagerが保存したパスは相対パスのはず
             img_path = Path(DATA_DIR) / sample.get("image_path", "")
             
        if not img_path.exists():
            continue
            
        try:
            from PIL import Image
            img = np.array(Image.open(img_path).convert("RGB"))
            images.append(img)
            true_labels[TaskType.CAUSE].append(sample["cause"])
            true_labels[TaskType.SHAPE].append(sample["shape"])
            true_labels[TaskType.DEPTH].append(sample["depth"])
        except Exception:
            continue

    if not images:
        raise ValueError("有効な検証用画像がありません。")

    # バッチ推論
    # メモリ節約のため小分けにする
    batch_size = 16
    predictions = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        batch_preds = predictor.predict_batch(batch_images)
        predictions.extend(batch_preds)

    pred_labels = {
        TaskType.CAUSE: [p.cause.label for p in predictions],
        TaskType.SHAPE: [p.shape.label for p in predictions],
        TaskType.DEPTH: [p.depth.label for p in predictions],
    }
    
    return {
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "total_samples": len(images)
    }


def _show_summary_tab(results: dict, category_manager: CategoryManager):
    """サマリータブ"""
    st.markdown("### 📈 評価サマリー")
    
    true_labels = results["true_labels"]
    pred_labels = results["pred_labels"]
    
    # 各タスクの精度計算
    accuracies = {}
    for task in [TaskType.CAUSE, TaskType.SHAPE, TaskType.DEPTH]:
        acc = accuracy_score(true_labels[task], pred_labels[task])
        accuracies[task] = acc

    # 全体精度（全タスク正解）
    all_correct = 0
    total = results["total_samples"]
    for i in range(total):
        if (true_labels[TaskType.CAUSE][i] == pred_labels[TaskType.CAUSE][i] and
            true_labels[TaskType.SHAPE][i] == pred_labels[TaskType.SHAPE][i] and
            true_labels[TaskType.DEPTH][i] == pred_labels[TaskType.DEPTH][i]):
            all_correct += 1
    overall_acc = all_correct / total if total > 0 else 0

    # メトリクスカード
    cols = st.columns(4)
    metrics = [
        ("全体完全一致", f"{overall_acc:.1%}", "#667eea"),
        ("原因分類", f"{accuracies[TaskType.CAUSE]:.1%}", "#764ba2"),
        ("形状分類", f"{accuracies[TaskType.SHAPE]:.1%}", "#f093fb"),
        ("深さ分類", f"{accuracies[TaskType.DEPTH]:.1%}", "#5ee7df"),
    ]

    for col, (name, value, color) in zip(cols, metrics):
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
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # クラス別精度 (F1 Score)
    st.markdown("### 🎯 クラス別 F1スコア")
    task_names = {TaskType.CAUSE: "原因", TaskType.SHAPE: "形状", TaskType.DEPTH: "深さ"}

    for task, name in task_names.items():
        labels = category_manager.get_categories(task)
        # scikit-learn で計算
        p, r, f1, s = precision_recall_fscore_support(
            true_labels[task], 
            pred_labels[task], 
            labels=labels, 
            zero_division=0
        )
        
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=[score * 100 for score in f1],
                    marker_color=[
                        f"hsl({i * 360 / len(labels)}, 70%, 60%)"
                        for i in range(len(labels))
                    ],
                    text=[f"{score * 100:.1f}%" for score in f1],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title=f"{name}分類 (F1 Score)",
            xaxis_title="カテゴリ",
            yaxis_title="F1 Score (%)",
            yaxis_range=[0, 100],
            height=300,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig, width="stretch")


def _show_confusion_matrix_tab(results: dict, category_manager: CategoryManager):
    """混同行列タブ"""
    st.markdown("### 🎯 混同行列")

    task_names = {TaskType.CAUSE: "原因", TaskType.SHAPE: "形状", TaskType.DEPTH: "深さ"}
    selected_task = st.selectbox(
        "分類タスクを選択",
        options=list(task_names.keys()),
        format_func=lambda x: task_names[x],
    )

    categories = category_manager.get_categories(selected_task)
    y_true = results["true_labels"][selected_task]
    y_pred = results["pred_labels"][selected_task]

    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    # 正規化 (行方向の和で割る = Recall的な視点)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

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
        yaxis=dict(autorange="reversed") # y軸を上から下の順に
    )

    st.plotly_chart(fig, width="stretch")
    
    # 統計情報
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0
    
    st.metric(f"{task_names[selected_task]}分類の精度", f"{accuracy:.1%}")


def _show_detailed_analysis_tab(results: dict, category_manager: CategoryManager):
    """詳細分析タブ"""
    st.markdown("### 📋 詳細分析")

    task_names = {TaskType.CAUSE: "原因", TaskType.SHAPE: "形状", TaskType.DEPTH: "深さ"}
    
    report_data = []

    for task, name in task_names.items():
        st.markdown(f"#### {name}分類")
        categories = category_manager.get_categories(task)
        
        y_true = results["true_labels"][task]
        y_pred = results["pred_labels"][task]
        
        # scikit-learn で詳細レポート計算
        p, r, f1, s = precision_recall_fscore_support(
            y_true, y_pred, labels=categories, zero_division=0
        )
        
        task_data = []
        for i, cat in enumerate(categories):
            row = {
                "タスク": name,
                "カテゴリ": cat,
                "Precision": f"{p[i]:.1%}",
                "Recall": f"{r[i]:.1%}",
                "F1 Score": f"{f1[i]:.1%}",
                "サンプル数": int(s[i]),
            }
            task_data.append(row)
            report_data.append(row)

        df = pd.DataFrame(task_data).drop(columns=["タスク"])
        st.dataframe(df, width="stretch", hide_index=True)
        st.markdown("---")
        
    # ダウンロード用データ
    if report_data:
        full_df = pd.DataFrame(report_data)
        csv = full_df.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            "📄 詳細レポートをCSVでダウンロード",
            data=csv,
            file_name="evaluation_report.csv",
            mime="text/csv",
        )
