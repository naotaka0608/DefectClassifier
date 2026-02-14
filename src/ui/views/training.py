"""学習ページ"""

import time
from pathlib import Path
import yaml

import plotly.graph_objects as go
import streamlit as st

from src.core.config import DEFAULT_MODEL_CONFIG, load_config


def show_training_page():
    """学習ページを表示"""
    st.markdown("## 📚 モデル学習")
    st.markdown("傷分類モデルの学習を実行できます。")

    # タブ
    tab1, tab2, tab3 = st.tabs(["🚀 学習実行", "🖼️ データ拡張", "📜 学習履歴"])

    with tab1:
        _show_training_tab()
    
    with tab2:
        _show_augmentation_settings_tab()

    with tab3:
        _show_history_tab()


from src.training.runner import train_model

def _show_training_tab():
    """学習実行タブ"""
    col1, col2 = st.columns([1, 1])

    # 設定値の保持用辞書
    current_settings = {}

    with col1:
        st.markdown("### ⚙️ 学習設定")

        # データセット設定
        st.markdown("#### 📁 データセット")
        data_dir = st.text_input(
            "データディレクトリ",
            value="data/processed",
            help="学習データが格納されているディレクトリ",
        )

        # ハイパーパラメータ
        st.markdown("#### 🎛️ ハイパーパラメータ")

        epochs = st.slider("エポック数", min_value=1, max_value=500, value=10, step=1)
        current_settings["epochs"] = epochs

        batch_size = st.select_slider(
            "バッチサイズ", options=[4, 8, 16, 32, 64, 128], value=32
        )
        current_settings["batch_size"] = batch_size

        learning_rate = st.select_slider(
            "学習率",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}",
        )
        current_settings["learning_rate"] = learning_rate

        # モデル設定
        st.markdown("#### 🧠 モデル設定")

        backbone = st.selectbox(
            "バックボーン",
            options=["resnet50", "resnet101", "efficientnet_b4"],
            index=0,
        )
        current_settings["backbone"] = backbone

        pretrained = st.checkbox("事前学習済み重みを使用", value=True)
        current_settings["pretrained"] = pretrained

        # GPU設定
        st.markdown("#### 💻 計算リソース")
        use_gpu = st.checkbox("GPUを使用", value=True)
        mixed_precision = st.checkbox("混合精度学習", value=True)
        current_settings["mixed_precision"] = mixed_precision

    with col2:
        st.markdown("### 📊 学習モニター")

        # プレースホルダー作成
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_container = st.empty()

        if st.button("🚀 学習開始", use_container_width=True, type="primary"):
            # 設定を保存
            _save_training_config(current_settings)
            
            # 履歴初期化
            st.session_state.training_history = {
                "train_loss": [],
                "val_loss": [],
                "accuracy": [],
            }
            
            try:
                # コールバック関数
                def progress_callback(metrics):
                    # 進捗更新
                    current_epoch = metrics["epoch"]
                    total = metrics["total_epochs"]
                    progress = current_epoch / total
                    progress_bar.progress(progress)
                    
                    # 履歴更新
                    st.session_state.training_history["train_loss"].append(metrics["train_loss"])
                    st.session_state.training_history["val_loss"].append(metrics["val_loss"])
                    st.session_state.training_history["accuracy"].append(metrics["metrics"]["mean_accuracy"])
                    
                    # テキスト更新
                    status_text.markdown(
                        f"""
                        **Epoch {current_epoch}/{total}**
                        - Train Loss: `{metrics['train_loss']:.4f}`
                        - Val Loss: `{metrics['val_loss']:.4f}`
                        - Accuracy: `{metrics['metrics']['mean_accuracy'] * 100:.1f}%`
                        """
                    )
                    
                    # グラフ更新
                    with chart_container.container():
                        _plot_training_history(st.session_state.training_history)

                # 学習実行
                with st.spinner("学習を実行中... (これには時間がかかります)"):
                    history = train_model(progress_callback=progress_callback)
                
                st.success("✅ 学習が完了しました！")
                
            except Exception as e:
                st.error(f"学習中にエラーが発生しました: {e}")
                import traceback
                st.code(traceback.format_exc())

        # 学習完了後の表示（履歴がある場合）
        if "training_history" in st.session_state and st.session_state.training_history["train_loss"]:
             with chart_container.container():
                _plot_training_history(st.session_state.training_history)


def _save_training_config(settings):
    """学習設定を保存"""
    config_path = Path("config/model_config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {"model": {}, "training": {}}

    # モデル設定更新
    if "model" not in config: config["model"] = {}
    config["model"]["backbone"] = settings["backbone"]
    config["model"]["pretrained"] = settings["pretrained"]

    # 学習設定更新
    if "training" not in config: config["training"] = {}
    config["training"]["epochs"] = settings["epochs"]
    config["training"]["batch_size"] = settings["batch_size"]
    config["training"]["learning_rate"] = settings["learning_rate"]
    config["training"]["mixed_precision"] = settings["mixed_precision"]
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)


def _plot_training_history(history: dict):
    """学習履歴をプロット"""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    if not epochs:
        return

    # 損失グラフ
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=epochs,
            y=history["train_loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="#667eea", width=2),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=epochs,
            y=history["val_loss"],
            mode="lines+markers",
            name="Val Loss",
            line=dict(color="#f093fb", width=2),
        )
    )
    fig1.update_layout(
        title="損失の推移",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=250,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig1, use_container_width=True, key=f"loss_chart_{len(epochs)}")

    # 精度グラフ
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=epochs,
            y=[a * 100 for a in history["accuracy"]],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#764ba2", width=2),
            fill="tozeroy",
            fillcolor="rgba(118, 75, 162, 0.1)",
        )
    )
    fig2.update_layout(
        title="精度の推移",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100],
        height=250,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True, key=f"acc_chart_{len(epochs)}")


def _show_history_tab():
    """学習履歴タブ"""
    st.markdown("### 📜 過去の学習履歴")

    # デモ用のダミーデータ
    history_data = [
        {
            "id": "train_001",
            "date": "2026-02-01 10:30",
            "epochs": 100,
            "best_accuracy": 92.5,
            "model_path": "checkpoints/model_001.pth",
        },
        {
            "id": "train_002",
            "date": "2026-02-03 14:15",
            "epochs": 150,
            "best_accuracy": 94.2,
            "model_path": "checkpoints/model_002.pth",
        },
        {
            "id": "train_003",
            "date": "2026-02-05 09:00",
            "epochs": 200,
            "best_accuracy": 95.8,
            "model_path": "checkpoints/model_003.pth",
        },
    ]

    for item in reversed(history_data):
        with st.expander(f"📁 {item['id']} - {item['date']}", expanded=False):
            cols = st.columns(4)
            cols[0].metric("エポック数", item["epochs"])
            cols[1].metric("最高精度", f"{item['best_accuracy']}%")
            cols[2].markdown(f"**モデルパス**  \n`{item['model_path']}`")
            cols[3].button("📥 読み込み", key=f"load_{item['id']}")
