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


def _show_training_tab():
    """学習実行タブ"""
    col1, col2 = st.columns([1, 1])

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

        epochs = st.slider("エポック数", min_value=10, max_value=500, value=100, step=10)

        batch_size = st.select_slider(
            "バッチサイズ", options=[8, 16, 32, 64, 128], value=32
        )

        learning_rate = st.select_slider(
            "学習率",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}",
        )

        # モデル設定
        st.markdown("#### 🧠 モデル設定")

        backbone = st.selectbox(
            "バックボーン",
            options=["resnet50", "resnet101", "efficientnet_b4"],
            index=0,
        )

        pretrained = st.checkbox("事前学習済み重みを使用", value=True)

        # GPU設定
        st.markdown("#### 💻 計算リソース")
        use_gpu = st.checkbox("GPUを使用", value=True)
        mixed_precision = st.checkbox("混合精度学習", value=True)

    with col2:
        st.markdown("### 📊 学習モニター")

        # 学習状態
        if "training_state" not in st.session_state:
            st.session_state.training_state = "idle"

        if st.session_state.training_state == "idle":
            if st.button("🚀 学習開始", use_container_width=True):
                st.session_state.training_state = "running"
                st.session_state.training_history = {
                    "train_loss": [],
                    "val_loss": [],
                    "accuracy": [],
                }
                st.rerun()

        elif st.session_state.training_state == "running":
            _run_training_demo(epochs)

        elif st.session_state.training_state == "completed":
            st.success("✅ 学習が完了しました！")

            if "training_history" in st.session_state:
                _plot_training_history(st.session_state.training_history)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 新規学習", use_container_width=True):
                    st.session_state.training_state = "idle"
                    st.rerun()
            with col_b:
                st.download_button(
                    "📥 モデルをダウンロード",
                    data=b"dummy_model_data",
                    file_name="best_model.pth",
                    mime="application/octet-stream",
                    use_container_width=True,
                )


def _show_augmentation_settings_tab():
    """データ拡張設定タブ"""
    st.markdown("### 🖼️ データ拡張設定")
    st.info("学習時に適用されるデータ拡張（Augmentation）パラメータを設定します。過学習を防ぐために重要です。")

    # 設定ファイルパス
    config_path = Path("config/model_config.yaml")

    # 設定読み込み
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        st.error(f"設定ファイルが見つかりません: {config_path}")
        return

    aug_config = config.get("augmentation", {})

    with st.form("augmentation_form"):
        # 1. サイズ設定
        st.markdown("#### 📏 サイズ設定")
        col1, col2 = st.columns(2)
        with col1:
            resize_w = st.number_input("リサイズ幅 (px)", value=aug_config.get("resize", [256, 256])[0])
            crop_w = st.number_input("切り出し幅 (px)", value=aug_config.get("crop_size", [224, 224])[0])
        with col2:
            resize_h = st.number_input("リサイズ高さ (px)", value=aug_config.get("resize", [256, 256])[1])
            crop_h = st.number_input("切り出し高さ (px)", value=aug_config.get("crop_size", [224, 224])[1])

        st.markdown("---")

        # 2. 幾何学的変換
        st.markdown("#### 🔄 幾何学的変換")
        col1, col2 = st.columns(2)
        with col1:
            h_flip = st.slider("水平反転の確率", 0.0, 1.0, aug_config.get("horizontal_flip", 0.5))
            rotate = st.slider("90度回転の確率", 0.0, 1.0, aug_config.get("random_rotate90", 0.5))
        with col2:
            v_flip = st.slider("垂直反転の確率", 0.0, 1.0, aug_config.get("vertical_flip", 0.5))

        st.markdown("---")

        # 3. 色調変化
        st.markdown("#### 🎨 色調変化 (Color Jitter)")
        jitter = aug_config.get("color_jitter", {})
        col1, col2 = st.columns(2)
        with col1:
            brightness = st.slider("明るさ変化", 0.0, 1.0, jitter.get("brightness", 0.2))
            saturation = st.slider("彩度変化", 0.0, 1.0, jitter.get("saturation", 0.2))
        with col2:
            contrast = st.slider("コントラスト変化", 0.0, 1.0, jitter.get("contrast", 0.2))
            hue = st.slider("色相変化", 0.0, 0.5, jitter.get("hue", 0.1))

        st.markdown("---")

        # 4. ノイズ
        st.markdown("#### 🌫️ ノイズ (Gaussian Noise)")
        noise = aug_config.get("gaussian_noise", {})
        noise_prob = st.slider("ノイズ付加確率", 0.0, 1.0, noise.get("probability", 0.3))

        limit = noise.get("var_limit", [10, 50])
        # var_limit can be int or list. Handle list safely
        if isinstance(limit, int):
            limit = [limit, limit]

        noise_limit = st.slider(
            "ノイズ分散範囲",
            0, 100,
            (int(limit[0]), int(limit[1]))
        )

        submitted = st.form_submit_button("💾 設定を保存", use_container_width=True, type="primary")

        if submitted:
            # 設定更新
            new_aug = {
                "resize": [int(resize_w), int(resize_h)],
                "crop_size": [int(crop_w), int(crop_h)],
                "horizontal_flip": float(h_flip),
                "vertical_flip": float(v_flip),
                "random_rotate90": float(rotate),
                "color_jitter": {
                    "brightness": float(brightness),
                    "contrast": float(contrast),
                    "saturation": float(saturation),
                    "hue": float(hue)
                },
                "gaussian_noise": {
                    "var_limit": [int(noise_limit[0]), int(noise_limit[1])],
                    "probability": float(noise_prob)
                }
            }

            config["augmentation"] = new_aug

            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, allow_unicode=True, sort_keys=False)
                st.success("データ拡張設定を保存しました！")
            except Exception as e:
                st.error(f"保存に失敗しました: {e}")


def _run_training_demo(total_epochs: int):
    """学習デモ"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.container()

    # デモ用の学習シミュレーション
    history = st.session_state.training_history
    current_epoch = len(history["train_loss"])

    if current_epoch < total_epochs:
        # 1エポック分の進捗
        import numpy as np

        train_loss = 2.0 * np.exp(-current_epoch / 30) + np.random.normal(0, 0.05)
        val_loss = 2.2 * np.exp(-current_epoch / 35) + np.random.normal(0, 0.08)
        accuracy = 1 - np.exp(-current_epoch / 25) + np.random.normal(0, 0.02)
        accuracy = min(max(accuracy, 0), 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(accuracy)

        progress = (current_epoch + 1) / total_epochs
        progress_bar.progress(progress)

        status_text.markdown(
            f"""
            **Epoch {current_epoch + 1}/{total_epochs}**
            - Train Loss: `{train_loss:.4f}`
            - Val Loss: `{val_loss:.4f}`
            - Accuracy: `{accuracy * 100:.1f}%`
            """
        )

        with metrics_container:
            _plot_training_history(history)

        time.sleep(0.1)  # シミュレーション用の遅延
        st.rerun()
    else:
        st.session_state.training_state = "completed"
        st.rerun()


def _plot_training_history(history: dict):
    """学習履歴をプロット"""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # 損失グラフ
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=epochs,
            y=history["train_loss"],
            mode="lines",
            name="Train Loss",
            line=dict(color="#667eea", width=2),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=epochs,
            y=history["val_loss"],
            mode="lines",
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
    st.plotly_chart(fig1, use_container_width=True)

    # 精度グラフ
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=epochs,
            y=[a * 100 for a in history["accuracy"]],
            mode="lines",
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
    st.plotly_chart(fig2, use_container_width=True)


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
