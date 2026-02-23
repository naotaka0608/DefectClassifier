"""判定履歴ページ"""

import streamlit as st
from PIL import Image
from pathlib import Path
from src.core.database import db
from src.ui.components.image_viewer import image_viewer

def show_history_page():
    """判定履歴ページを表示"""
    st.markdown("## 📊 判定履歴")
    st.markdown("過去の分類結果を確認できます。")

    # 履歴取得
    history = db.get_history(limit=50)

    if not history:
        st.info("履歴がありません。画像分類を実行するとここに表示されます。")
        return

    # 統計情報の表示
    _show_stats(history)

    # 履歴テーブル
    st.markdown("### 📋 実行ログ")
    
    # フィルタ/検索 (簡易)
    search = st.text_input("🔍 ラベルで検索", "").lower()
    
    filtered_history = history
    if search:
        filtered_history = [
            item for item in history 
            if search in str(item.get("cause_label", "")).lower() or 
               search in str(item.get("shape_label", "")).lower() or 
               search in str(item.get("depth_label", "")).lower()
        ]

    for item in filtered_history:
        with st.expander(f"🕒 {item['timestamp']} - {item['cause_label']} / {item['shape_label']} / {item['depth_label']}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                img_path = Path(item["image_path"])
                if img_path.exists():
                    image_viewer(img_path, width=250)
                else:
                    st.error("画像ファイルが見つかりません")
                    st.caption(f"Path: {item['image_path']}")
            
            with col2:
                # 詳細情報
                st.markdown(f"**原因**: `{item['cause_label']}` ({item['cause_confidence']:.1%})")
                st.markdown(f"**形状**: `{item['shape_label']}` ({item['shape_confidence']:.1%})")
                st.markdown(f"**深さ**: `{item['depth_label']}` ({item['depth_confidence']:.1%})")
                
                status_color = "red" if item.get("is_anomaly") else "green"
                status_text = "判定不能 (未知)" if item.get("is_anomaly") else "正常"
                st.markdown(f"**ステータス**: :{status_color}[{status_text}]")
                
                # 確率分布の表示
                details_json = item.get("details_json")
                if details_json:
                    import json
                    try:
                        details = json.loads(details_json)
                        with st.expander("📊 確率分布の詳細"):
                            for task, probs in details.items():
                                st.markdown(f"**{task.capitalize()}**")
                                # 上位5つをフォーマットして表示
                                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
                                for label, val in sorted_probs:
                                    st.write(f"- {label}: **{val:.2%}** ({val})")
                    except:
                        pass
                
                st.markdown("---")
                st.caption(f"推論時間: {item['inference_time_ms']:.1f}ms | モデル: {item['model_version']}")
                
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    # PDFレポート生成
                    pdf_key = f"pdf_ready_{item['id']}"
                    if st.button("📄 PDF生成", key=f"btn_gen_{item['id']}"):
                        from src.utils.report_generator import ReportGenerator
                        repo_gen = ReportGenerator()
                        st.session_state[pdf_key] = repo_gen.generate(item).getvalue()
                    
                    if pdf_key in st.session_state:
                        st.download_button(
                            label="📥 ダウンロード",
                            data=st.session_state[pdf_key],
                            file_name=f"report_{item['id']}_{item['timestamp'][:10]}.pdf",
                            mime="application/pdf",
                            key=f"btn_dl_{item['id']}"
                        )

                with btn_cols[1]:
                    if st.button("🗑️ 削除", key=f"del_{item['id']}"):
                        db.delete_history_item(item['id'])
                        if pdf_key in st.session_state:
                            del st.session_state[pdf_key]
                        st.success("削除しました")
                        st.rerun()

def _show_stats(history):
    """簡単な統計を表示"""
    cols = st.columns(4)
    
    total = len(history)
    avg_inf = sum(item["inference_time_ms"] for item in history) / total if total > 0 else 0
    
    # 原因別のカウント (上位)
    causes = [item["cause_label"] for item in history if item["cause_label"]]
    from collections import Counter
    most_common_cause = Counter(causes).most_common(1)
    top_cause = most_common_cause[0][0] if most_common_cause else "N/A"

    with cols[0]:
        st.metric("総判定数", f"{total}件")
    with cols[1]:
        st.metric("平均推論時間", f"{avg_inf:.1f}ms")
    with cols[2]:
        st.metric("最多原因", top_cause)
    with cols[3]:
        st.metric("最新ステータス", "正常" if total > 0 else "-")
