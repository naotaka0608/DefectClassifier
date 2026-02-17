# 外部連携API仕様書

Hanteiシステムの外部連携用APIのドキュメントです。
本システムはREST APIを提供しており、C#等の外部プログラムからHTTPリクエストを通じて傷分類機能を利用できます。

## ベースURL

デフォルト設定の場合: `http://localhost:8000`

## エンドポイント一覧

### 1. ヘルスチェック

サーバーの状態とモデルの読み込み状況を確認します。

- **URL**: `/health`
- **Method**: `GET`

#### レスポンス (JSON)

```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "version": "1.0.0"
}
```

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `status` | string | サーバーの状態 ("healthy" など) |
| `model_loaded` | boolean | モデルが正常にロードされているか |
| `gpu_available` | boolean | GPUが利用可能か |
| `version` | string | APIのバージョン |

---

### 2. 単一画像分類 (Predict)

1枚の画像を送信し、傷の分類（原因・形状・深さ）を行います。

- **URL**: `/api/v1/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### リクエスト (JSON)

```json
{
  "image_base64": "/9j/4AAQSkZJRg...",  // Base64エンコードされた画像データ
  "return_confidence": true,            // (任意) 信頼度スコアを返すか。デフォルト: true
  "model_name": "best_model"            // (任意) 使用するモデル名。指定なしでデフォルトモデル
}
```

| フィールド | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `image_base64` | string | 〇 | 画像ファイルをBase64文字列に変換したもの |
| `return_confidence` | boolean | - | `true`の場合、確信度(confidence)を含める |
| `model_name` | string | - | 特定のモデルを指定する場合に使用 |

#### レスポンス (JSON)

```json
{
  "success": true,
  "cause": {
    "label": "擦り傷",
    "confidence": 0.92,
    "class_id": 0
  },
  "shape": {
    "label": "線状",
    "confidence": 0.88,
    "class_id": 0
  },
  "depth": {
    "label": "表層",
    "confidence": 0.85,
    "class_id": 0
  },
  "inference_time_ms": 45.2,
  "model_version": "best_model.pth"
}
```

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `success` | boolean | 処理が成功したか |
| `cause` | object | **原因**の分類結果 |
| `shape` | object | **形状**の分類結果 |
| `depth` | object | **深さ**の分類結果 |
| `inference_time_ms` | float | 推論にかかった時間（ミリ秒） |
| `model_version` | string | 使用されたモデルのファイル名 |

**分類結果オブジェクト (`cause`, `shape`, `depth`) の詳細:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `label` | string | 分類ラベル（例: "擦り傷"） |
| `confidence` | float | 確信度 (0.0 〜 1.0) |
| `class_id` | integer | クラスID |

---

### 3. バッチ画像分類 (Batch Predict)

複数の画像を一度に送信し、まとめて分類を行います。

- **URL**: `/api/v1/predict/batch`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### リクエスト (JSON)

```json
{
  "images": [
    "/9j/4AAQSkZJRg...",
    "/9j/4AAQSkZJRg..."
  ],
  "return_confidence": true,
  "model_name": null
}
```

| フィールド | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `images` | list[string] | 〇 | Base64画像文字列のリスト（最大32枚推奨） |
| `return_confidence` | boolean | - | `true`の場合、確信度を含める |
| `model_name` | string | - | モデル指定 |

#### レスポンス (JSON)

```json
{
  "success": true,
  "results": [
    {
      "success": true,
      "cause": { "label": "擦り傷", "confidence": 0.95, "class_id": 0 },
      "shape": { "label": "線状", "confidence": 0.90, "class_id": 1 },
      "depth": { "label": "表層", "confidence": 0.85, "class_id": 2 },
      "inference_time_ms": 15.5,
      "model_version": "best_model.pth"
    },
    {
      "success": true,
      "cause": { "label": "打痕", "confidence": 0.92, "class_id": 1 },
      "shape": { "label": "点状", "confidence": 0.88, "class_id": 0 },
      "depth": { "label": "中層", "confidence": 0.80, "class_id": 1 },
      "inference_time_ms": 15.5,
      "model_version": "best_model.pth"
    }
  ],
  "total_inference_time_ms": 35.2
}
```

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `success` | boolean | バッチ処理全体の成功可否 |
| `results` | list[object] | 各画像の分類結果リスト（順序はリクエストと同じ） |
| `total_inference_time_ms` | float | 全体の推論処理時間 |

---

### 4. カテゴリ一覧取得

現在設定されている分類カテゴリの一覧を取得します。

- **URL**: `/api/v1/categories`
- **Method**: `GET`

#### レスポンス (JSON)

```json
{
  "cause_categories": [
    {
      "id": 0,
      "name": "擦り傷",
      "code": "SCRATCH"
    },
    {
      "id": 1,
      "name": "打痕",
      "code": "DENT"
    }
  ],
  "shape_categories": [
    {
      "id": 0,
      "name": "線状",
      "code": "LINEAR"
    },
    ...
  ],
  "depth_categories": [
    ...
  ]
}
```

---

### 5. カテゴリ設定のリロード

`config/categories.yaml` を変更した後、APIを再起動せずに設定を反映させます。

- **URL**: `/api/v1/categories/reload`
- **Method**: `POST`

#### レスポンス (JSON)

```json
{
  "success": true,
  "message": "Categories reloaded"
}
```
