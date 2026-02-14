"""チャートコンポーネント"""

import plotly.graph_objects as go
import streamlit as st


def plot_training_history(history: dict, key_suffix: str = ""):
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
    st.plotly_chart(fig1, width="stretch", key=f"loss_chart_{len(epochs)}{key_suffix}")

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
    st.plotly_chart(fig2, width="stretch", key=f"acc_chart_{len(epochs)}{key_suffix}")
