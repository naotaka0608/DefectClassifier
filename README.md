# 傷分類システム (Hantei)

製品検品用の傷分類機械学習システムです。PyTorch + Streamlit + FastAPI で構築されています。

## 機能

- **傷の3種類分類**: 原因・形状・深さを同時に分類
- **モダンなUI**: Streamlitによる直感的な操作画面
- **REST API**: C#など外部システムとの連携
- **カスタマイズ可能**: カテゴリを自由に追加・変更

---

## セットアップ

### 1. 依存関係のインストール

```bash
# uvを使用（推奨）
uv sync

# または pip を使用
pip install -e .
```

### 2. GPU対応（オプション）

CUDA対応のPyTorchを使用する場合:

```bash
# CUDA 12.x の場合
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 使い方

### Streamlit UI を起動

```bash
uv run streamlit run src/ui/app.py
```

ブラウザで http://localhost:8501 を開きます。

### FastAPI を起動

```bash
uv run uvicorn src.api.main:app --reload
```

API ドキュメント: http://localhost:8000/docs

---

## UI画面の説明

### 🎯 分類画面

1. 左側の「画像アップロード」エリアに傷の画像をドラッグ＆ドロップ
2. 「分類を実行」ボタンをクリック
3. 右側に分類結果（原因・形状・深さ）と確率分布が表示されます

### 📚 学習画面

1. 「学習実行」タブでハイパーパラメータを設定
   - エポック数、バッチサイズ、学習率など
2. 「学習開始」ボタンをクリック
3. リアルタイムで損失・精度のグラフが更新されます
4. 学習完了後、モデルをダウンロード可能

### 📊 評価画面

- **サマリー**: 全体精度とクラス別精度
- **混同行列**: 予測と実際のラベルの対応
- **詳細分析**: Precision/Recall/F1スコア

### ⚙️ 設定画面

- **カテゴリ管理**: 原因・形状・深さの分類カテゴリを編集
- **モデル設定**: 使用するモデルの切り替え
- **システム情報**: バージョン、GPU状態など

---

## データ準備

### アノテーションファイルの形式

`data/processed/train/annotations.json`:

```json
{
  "version": "1.0",
  "samples": [
    {
      "id": "00001",
      "image_path": "images/00001.jpg",
      "cause": "擦り傷",
      "shape": "線状",
      "depth": "表層"
    },
    {
      "id": "00002",
      "image_path": "images/00002.jpg",
      "cause": "打痕",
      "shape": "点状",
      "depth": "中層"
    }
  ]
}
```

### ディレクトリ構造

```
data/
├── processed/
│   ├── train/
│   │   ├── images/
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   └── annotations.json
│   └── val/
│       ├── images/
│       └── annotations.json
```

---

## 外部連携API

詳細なAPI仕様（JSONフォーマット、エンドポイント一覧）については、以下のドキュメントを参照してください：

[外部連携API仕様書 (docs/external_api.md)](docs/external_api.md)

---

## カテゴリのカスタマイズ

`config/categories.yaml` を編集:

```yaml
categories:
  cause:
    - name: "擦り傷"
      description: "表面を擦ることで発生した傷"
      code: "SCRATCH"
    - name: "打痕"
      description: "衝撃による凹み"
      code: "DENT"
    # 新しいカテゴリを追加
    - name: "汚れ"
      description: "表面の汚染"
      code: "STAIN"
```

カテゴリを変更した場合は、モデルの再学習が必要です。

---

## C#連携（ONNX変換）

学習済みモデルをONNX形式に変換してC#から利用:

```bash
uv run python scripts/export_onnx.py \
  --checkpoint checkpoints/best_model.pth \
  --output model.onnx
```

C#側では ONNX Runtime を使用して推論できます。

---

## プロジェクト構成

```
hantei/
├── config/                  # 設定ファイル
│   ├── categories.yaml      # カテゴリ定義
│   └── model_config.yaml    # モデル設定
├── src/
│   ├── api/                 # FastAPI
│   ├── models/              # 機械学習モデル
│   ├── training/            # 学習パイプライン
│   ├── inference/           # 推論
│   ├── core/                # コア機能
│   └── ui/                  # Streamlit UI
├── scripts/                 # ユーティリティスクリプト
├── data/                    # データディレクトリ
├── checkpoints/             # モデル保存先
└── pyproject.toml           # プロジェクト設定
```

---

## トラブルシューティング

### GPUが認識されない

```python
import torch
print(torch.cuda.is_available())  # False の場合はCUDA版PyTorchを再インストール
```

### モデルが読み込めない

- チェックポイントファイルのパスを確認
- カテゴリ数がモデル作成時と一致しているか確認

### 学習が遅い

- `mixed_precision: true` を有効化（model_config.yaml）
- バッチサイズを増やす（GPUメモリに余裕がある場合）

---

## ライセンス

MIT License
