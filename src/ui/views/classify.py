"""分類ページ"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG
from src.core.types import TaskType
from src.ui.components.image_viewer import image_viewer
from src.core.database import db
from src.core.constants import HISTORY_DIR

if TYPE_CHECKING:
    from src.inference.predictor import DefectPredictor


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
            
            # ヒートマップ表示切り替え
            if "classification_heatmaps" in st.session_state:
                heatmaps = st.session_state.classification_heatmaps
                
                # 表示モード選択
                task_name_map = {TaskType.CAUSE: "原因", TaskType.SHAPE: "形状", TaskType.DEPTH: "深さ"}
                view_options = ["オリジナル"] + [f"ヒートマップ: {task_name_map[t]}" for t in [TaskType.CAUSE, TaskType.SHAPE, TaskType.DEPTH]]
                selected_view = st.radio("表示画像", view_options, horizontal=True, label_visibility="collapsed")
                
                if selected_view == "オリジナル":
                    image_viewer(uploaded_file, caption="アップロード画像", width=350)
                else:
                    # 'ヒートマップ: 原因' -> '原因' -> TaskType.CAUSE
                    selected_label = selected_view.split(": ")[1]
                    target_task = next(t for t, name in task_name_map.items() if name == selected_label)
                    
                    if target_task in heatmaps:
                        st.image(heatmaps[target_task], caption=f"Grad-CAM: {task_name_map[target_task]}", width=350)
            else:
                image_viewer(uploaded_file, caption="アップロード画像", width=350)

            show_heatmap = st.checkbox("🔍 判断根拠(ヒートマップ)を表示", value=False, help="AIが注目した領域を可視化します")

            # 分類実行ボタン
            if st.button("🔍 分類を実行", width="stretch"):
                _run_classification(image, category_manager, show_heatmap)
            
            if "classification_error" in st.session_state:
                st.error(st.session_state.classification_error)
                del st.session_state.classification_error

    with col2:
        st.markdown("### 📊 分類結果")

        if "classification_result" in st.session_state:
            result = st.session_state.classification_result
            probs = st.session_state.classification_probs

            # 結果表示
            _display_results(result, probs, category_manager)
        else:
            st.info("画像をアップロードして分類を実行してください。")

@st.cache_resource
def _get_predictor() -> "DefectPredictor":
    """推論器をロード・キャッシュ"""
    from src.inference.predictor import DefectPredictor
    from src.core.constants import CHECKPOINTS_DIR, BEST_MODEL_PATH
    
    # モデルパス解決
    model_path = BEST_MODEL_PATH
    if not model_path.exists():
        # best_modelがない場合はcheckpoints以下の最新を使用
        checkpoints = sorted(list(CHECKPOINTS_DIR.glob("*.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
        if checkpoints:
            model_path = checkpoints[0]
        else:
            st.error("モデルファイルが見つかりません。")
            return None
            
    try:
        # PredictorにはCategoryManagerが必要
        from src.core.category_manager import CategoryManager
        from src.core.config import DEFAULT_CATEGORIES_CONFIG
        
        category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)
        predictor = DefectPredictor(model_path=model_path, category_manager=category_manager)
        return predictor
    except Exception as e:
        st.error(f"モデルのロードに失敗しました: {e}")
        return None


def _run_classification(image: Image.Image, category_manager: CategoryManager, show_heatmap: bool = False):
    """分類を実行"""
    
    predictor = _get_predictor()
    if predictor is None:
        return

    with st.spinner(f"モデル '{predictor.model_version}' で分類中..."):
        try:
            # 推論実行
            import numpy as np
            image_np = np.array(image)
            result = predictor.predict(image_np)
            
            # 結果をセッションに保存
            st.session_state.classification_result = {
                TaskType.CAUSE: result.cause.label,
                TaskType.SHAPE: result.shape.label,
                TaskType.DEPTH: result.depth.label,
                "is_anomaly": result.is_anomaly,
                "anomaly_score": result.anomaly_score,
            }
            
            probs = {}
            for task in [TaskType.CAUSE, TaskType.SHAPE, TaskType.DEPTH]:
                categories = category_manager.get_categories(task)
                task_res = getattr(result, task)
                
                if hasattr(task_res, "probabilities") and task_res.probabilities:
                    probs[task] = task_res.probabilities
                else:
                    conf = task_res.confidence
                    other_prob = (1.0 - conf) / (len(categories) - 1) if len(categories) > 1 else 0.0
                    
                    task_probs = {}
                    for cat in categories:
                        if cat == task_res.label:
                            task_probs[cat] = conf
                        else:
                            task_probs[cat] = other_prob
                    
                    probs[task] = task_probs

            st.session_state.classification_probs = probs
            
            # Grad-CAM (オプション)
            if show_heatmap:
                from src.analysis.gradcam import GradCAM, overlay_heatmap
                from src.training.dataset import DefectDataset
                
                # Transform (推論時と同じ前処理)
                # FIX: image_sizeをkwargsとして渡す
                transform = DefectDataset.get_inference_transform(image_size=[224, 224])
                
                img_np = np.array(image)
                augmented = transform(image=img_np)
                input_tensor = augmented["image"].unsqueeze(0).to(predictor.device) # (1, C, H, W)
                
                gradcam = GradCAM(predictor.model)
                
                # 各タスクについてヒートマップ生成
                heatmaps = {}
                for task in [TaskType.CAUSE, TaskType.SHAPE, TaskType.DEPTH]:
                    # 予測されたクラスに対するヒートマップ
                    cam, _ = gradcam(input_tensor, task_type=task)
                    
                    # 重ね合わせ
                    # overlay_heatmapはPIL Imageを返す
                    overlay = overlay_heatmap(image, cam, alpha=0.6)
                    heatmaps[task] = overlay
                
                gradcam.remove_hooks()
                st.session_state.classification_heatmaps = heatmaps
            else:
                if "classification_heatmaps" in st.session_state:
                    del st.session_state.classification_heatmaps

            # Database に保存
            try:
                # 画像を history フォルダに保存
                import uuid
                from datetime import datetime
                HISTORY_DIR.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
                image_path = HISTORY_DIR / image_filename
                image.save(image_path, quality=95)
                
                # DB 保存
                db.save_result(
                    image_path=str(image_path),
                    cause={"label": result.cause.label, "confidence": result.cause.confidence},
                    shape={"label": result.shape.label, "confidence": result.shape.confidence},
                    depth={"label": result.depth.label, "confidence": result.depth.confidence},
                    inference_time_ms=getattr(result, "inference_time_ms", 0.0),
                    model_version=predictor.model_version,
                    is_anomaly=result.is_anomaly,
                    anomaly_score=result.anomaly_score,
                    details=probs
                )
            except Exception as e:
                import logger
                logger.error(f"Failed to save history: {e}")

            st.success("分類が完了しました！")
            st.rerun()
            
        except Exception as e:
            # エラーメッセージをセッションに保存して表示
            # st.errorだとrerunで消える可能性があるため
            st.session_state.classification_error = f"推論中にエラーが発生しました: {e}"
            st.rerun()


def _display_results(result: dict, probs: dict, category_manager: CategoryManager):
    """結果を表示"""
    # メトリクスカード
    cols = st.columns(3)

    task_names = {TaskType.CAUSE: "原因", TaskType.SHAPE: "形状", TaskType.DEPTH: "深さ"}
    task_icons = {TaskType.CAUSE: "⚡", TaskType.SHAPE: "📐", TaskType.DEPTH: "📏"}
    task_colors = {TaskType.CAUSE: "#667eea", TaskType.SHAPE: "#764ba2", TaskType.DEPTH: "#f093fb"}

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
    
    # 信頼度チェック
    if result.get("is_anomaly"):
        st.error(f"⚠️ **判定不能（未知のデータ）**: 入力画像が学習データと大きく異なっている可能性があります (信頼度スコア: {result.get('anomaly_score', 0):.1%})。結果が不正確な可能性があります。")

    elif result.get(TaskType.CAUSE) and probs.get(TaskType.CAUSE):
        cause_conf = probs[TaskType.CAUSE][result[TaskType.CAUSE]]
        if cause_conf < 0.4:  # 40%未満は警告
            st.warning(f"⚠️ 原因分類の確信度が低いです ({cause_conf:.1%})。判定結果は信頼できない可能性があります。")
            
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
                height=250,
                margin=dict(l=40, r=40, t=40, b=40),
            )

            st.plotly_chart(fig, width='stretch')
