"""PDFレポート生成モジュール"""

import io
from pathlib import Path
from typing import Dict, Any
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

class ReportGenerator:
    """判定結果をPDFレポートとして出力するクラス"""

    def __init__(self):
        # 日本語フォントの設定 (MS UI Gothic などが一般的だが、環境依存を避けるため最低限の処理)
        # Windows標準のフォントパスを試行
        font_paths = [
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/meiryo.ttc",
        ]
        self.font_name = "Helvetica" # フォントがない場合のフォールバック
        for p in font_paths:
            if Path(p).exists():
                try:
                    pdfmetrics.registerFont(TTFont("MS-Gothic", p))
                    self.font_name = "MS-Gothic"
                    break
                except:
                    continue

    def generate(self, result: Dict[str, Any], output_path: str | Path = None) -> io.BytesIO:
        """
        判定結果からPDFを生成
        
        Args:
            result: DBから取得した1レコード分の辞書
            output_path: ファイルに保存する場合のパス
            
        Returns:
            BytesIO: PDFデータ
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # タイトル
        title_style = ParagraphStyle(
            name='TitleStyle',
            fontName=self.font_name,
            fontSize=18,
            leading=22,
            alignment=1, # Center
            spaceAfter=20
        )
        elements.append(Paragraph("傷分類判定レポート", title_style))
        
        # 基本情報テーブル
        data = [
            ["項目", "内容"],
            ["判定日時", result.get("timestamp", "-")],
            ["モデルバージョン", result.get("model_version", "-")],
            ["推論時間", f"{result.get('inference_time_ms', 0):.1f} ms"],
            ["判定ステータス", "判定不能（未知）" if result.get("is_anomaly") else "正常"],
        ]
        
        t = Table(data, colWidths=[150, 300])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))
        
        # 判定結果
        elements.append(Paragraph("■ 分類結果", styles['Heading2']))
        res_data = [
            ["タスク", "第1候補ラベル", "確信度"],
            ["原因", result.get("cause_label", "-"), f"{result.get('cause_confidence', 0):.1%}"],
            ["形状", result.get("shape_label", "-"), f"{result.get('shape_confidence', 0):.1%}"],
            ["深さ", result.get("depth_label", "-"), f"{result.get('depth_confidence', 0):.1%}"],
        ]
        rt = Table(res_data, colWidths=[100, 200, 150])
        rt.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(rt)
        
        # 他の候補の表示
        details_json = result.get("details_json")
        if details_json:
            import json
            try:
                details = json.loads(details_json)
                elements.append(Spacer(1, 10))
                elements.append(Paragraph("■ 他の候補 (確率分布)", styles['Heading3']))
                
                other_rows = [["タスク", "候補ラベル", "確信度"]]
                for task, probs in details.items():
                    # 上位3つ（自分以外）を取得しようとするが、正規化されているので上位3つを出す
                    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    for i, (label, conf) in enumerate(sorted_items):
                        task_name = f"{task} ({i+1})" if i > 0 else task
                        other_rows.append([task_name, label, f"{conf:.2%}"])
                
                ot = Table(other_rows, colWidths=[100, 200, 150])
                ot.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), self.font_name),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('SIZE', (0, 0), (-1, -1), 8),
                ]))
                elements.append(ot)
            except:
                pass

        elements.append(Spacer(1, 20))
        
        # 画像表示
        elements.append(Paragraph("■ 判定画像", styles['Heading2']))
        img_path = Path(result.get("image_path", ""))
        if img_path.exists():
            # 画像リサイズ (幅をA4に合わせる)
            img = Image(str(img_path), width=400, height=300, kind='proportional')
            elements.append(img)
        else:
            elements.append(Paragraph("画像ファイルが見つかりません", styles['Normal']))
            
        # PDF構築
        doc.build(elements)
        
        # ファイル保存
        pdf_data = buffer.getvalue()
        if output_path:
            with open(output_path, "wb") as f:
                f.write(pdf_data)
                
        buffer.seek(0)
        return buffer
