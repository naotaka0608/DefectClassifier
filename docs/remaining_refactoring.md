# 残リファクタリング項目

> **Note:** #5, #8, #11, #12, #13, #14 すべて完了しました！ 🎉

## 🟢 Easy（簡単）

### #5 `sys.path` ハック削除
**ファイル**: `src/ui/app.py`, `src/api/main.py`

`sys.path.insert(0, str(project_root))` によるパスハックが残っている。`uv run` 経由で実行する限り不要だが、直接 `python src/ui/app.py` で起動する場合は必要。削除する場合は起動方法を `uv run` に統一すること。

---

## 🟡 Medium（中程度）

### #8 `predict.py` グローバル変数を FastAPI DI に置き換え
**ファイル**: `src/api/routes/predict.py`

`_predictor`, `_category_manager`, `_config` がグローバル変数で管理されている。FastAPI の依存性注入（`Depends`）パターンに統一することで、テスタビリティが向上する。

**修正イメージ**:
```python
# 現状
_predictor: Optional[DefectPredictor] = None
def get_predictor():
    if _predictor is None: raise ...
    return _predictor

# 改善案: app.state を使用
@router.post("/predict")
async def predict(request: Request):
    predictor = request.app.state.predictor
```

### #11 `dataset.py` Augmentation パラメータを config から読む
**ファイル**: `src/training/dataset.py`, `src/training/runner.py`

`DefectDataset._default_transform()` のパラメータ（resize, crop_size, flip確率など）がハードコードされている。UIの「データ拡張設定」で保存した `AugmentationConfig` の値が学習時に反映されていない。

**修正方針**:
1. `runner.py` で `AugmentationConfig` を読み込む
2. `DefectDataset` のコンストラクタに `AugmentationConfig` を渡す
3. `_default_transform()` を `_build_transform(config)` に変更

---

## 🔴 Hard（大変）

### #12 Train/Val 分割での Augmentation 問題
**ファイル**: `src/training/runner.py`, `src/training/dataset.py`

`random_split` で分割すると、Validation データにも Training 用の Augmentation（回転・反転・ノイズ等）が適用されてしまう。本来 Validation は Resize + Normalize のみであるべき。

**修正方針**:
- `DefectDataset` にサンプルリストを直接渡せるようにする
- Train用とVal用で別々の `DefectDataset` インスタンスを作成
- それぞれ異なる Transform を適用

### #13 評価ページの実データ対応
**ファイル**: `src/ui/views/evaluation.py`

現在の評価ページはダミーデータ（ランダム生成）で混同行列やメトリクスを表示している。実際のモデルで検証データに対して推論を実行し、リアルな評価結果を算出するロジックに置き換える必要がある。

**修正方針**:
1. 検証データセットを読み込む
2. 学習済みモデルで推論を実行
3. 実際の混同行列・精度・F1スコアを算出
4. 結果を表示

### #14 テストコードの整備
**ディレクトリ**: `tests/`（新規作成）

ユニットテストがほぼ存在しない。以下のコンポーネントに対してテストを追加すべき：

| 対象 | テスト内容 |
|------|-----------|
| `DataManager` | アノテーション読み書き、サンプル追加 |
| `CategoryManager` | カテゴリ読み込み、名前⇔インデックス変換 |
| `DefectPredictor` | 推論結果の形式、バッチ推論 |
| API エンドポイント | `/predict`, `/predict/batch`, `/health` |
| `config.py` | 設定の読み書き、デフォルト値 |
