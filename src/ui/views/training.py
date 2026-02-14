"""学習ページ"""

import time
from pathlib import Path
import yaml

import plotly.graph_objects as go
import streamlit as st

from src.core.config import DEFAULT_MODEL_CONFIG, AppConfig, load_config, save_config
from src.core.constants import CHECKPOINTS_DIR, MODEL_CONFIG_PATH, PROCESSED_DIR


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
            value=str(PROCESSED_DIR),
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


def _show_augmentation_settings_tab():
    """データ拡張設定タブ"""
    st.markdown("### 🖼️ データ拡張設定")
    st.info("学習時のデータ拡張パラメータを設定します。")

    col1, col2 = st.columns(2)

    # 現在の設定を読み込み
    try:
        config = load_config(MODEL_CONFIG_PATH)
        aug_config = config.augmentation
    except Exception:
        aug_config = AppConfig().augmentation

    current_settings = {}

    with col1:
        st.markdown("#### 📏 変形・サイズ")
        
        # リサイズ
        resize_h = st.number_input("リサイズ (高さ)", value=aug_config.resize[0])
        resize_w = st.number_input("リサイズ (幅)", value=aug_config.resize[1])
        current_settings["resize"] = [resize_h, resize_w]

        # クロップ
        crop_h = st.number_input("クロップ (高さ)", value=aug_config.crop_size[0])
        crop_w = st.number_input("クロップ (幅)", value=aug_config.crop_size[1])
        current_settings["crop_size"] = [crop_h, crop_w]

        st.markdown("#### 🔄 回転・反転")
        
        # フリップ
        h_flip = st.slider("水平反転確率", 0.0, 1.0, float(aug_config.horizontal_flip))
        current_settings["horizontal_flip"] = h_flip
        
        v_flip = st.slider("垂直反転確率", 0.0, 1.0, float(aug_config.vertical_flip))
        current_settings["vertical_flip"] = v_flip
        
        rotate = st.slider("90度回転確率", 0.0, 1.0, float(aug_config.random_rotate90))
        current_settings["random_rotate90"] = rotate

    with col2:
        st.markdown("#### 🎨 色彩変換")
        
        brightness = st.slider("明るさ変化", 0.0, 1.0, float(aug_config.color_jitter["brightness"]))
        contrast = st.slider("コントラスト変化", 0.0, 1.0, float(aug_config.color_jitter["contrast"]))
        saturation = st.slider("彩度変化", 0.0, 1.0, float(aug_config.color_jitter["saturation"]))
        hue = st.slider("色相変化", 0.0, 0.5, float(aug_config.color_jitter["hue"]))
        
        current_settings["color_jitter"] = {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue
        }

        st.markdown("#### 🌫️ ノイズ")
        
        noise_prob = st.slider("ガウシアンノイズ確率", 0.0, 1.0, float(aug_config.gaussian_noise["probability"]))
        noise_limit = st.slider("ノイズ強度上限", 0, 100, int(aug_config.gaussian_noise["var_limit"][1]))
        
        current_settings["gaussian_noise"] = {
            "probability": noise_prob,
            "var_limit": [10, noise_limit]
        }

    st.markdown("---")
    if st.button("💾 データ拡張設定を保存", use_container_width=True):
        _save_augmentation_config(current_settings)
        st.success("データ拡張設定を保存しました！")


def _save_augmentation_config(settings):
    """データ拡張設定を保存"""
    # 既存設定読み込み
    try:
        config = load_config(MODEL_CONFIG_PATH)
    except Exception:
        config = AppConfig()

    # Configオブジェクト更新
    config.augmentation.resize = settings["resize"]
    config.augmentation.crop_size = settings["crop_size"]
    config.augmentation.horizontal_flip = settings["horizontal_flip"]
    config.augmentation.vertical_flip = settings["vertical_flip"]
    config.augmentation.random_rotate90 = settings["random_rotate90"]
    config.augmentation.color_jitter = settings["color_jitter"]
    config.augmentation.gaussian_noise = settings["gaussian_noise"]

    # 保存
    save_config(config, MODEL_CONFIG_PATH)
    """学習設定を保存"""
    # 既存設定を読み込み（なければデフォルト）
    AppConfig()
    try:
        config = load_config(MODEL_CONFIG_PATH)
    except Exception:
        config = AppConfig()

    # パラメータ更新
    # model
    config.model.backbone = settings["backbone"]
    config.model.pretrained = settings["pretrained"]
    
    # training
    config.training.epochs = settings["epochs"]
    config.training.batch_size = settings["batch_size"]
    config.training.learning_rate = settings["learning_rate"]
    config.training.mixed_precision = settings["mixed_precision"]
    
    # 保存
    save_config(config, MODEL_CONFIG_PATH)


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
