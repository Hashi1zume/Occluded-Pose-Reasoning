# Configuration Guide

このドキュメントでは、学習・評価用のYAML設定ファイルの各パラメータについて説明します。

## 目次

- [全般設定](#全般設定)
- [データセット設定](#データセット設定)
- [モデル設定](#モデル設定)
- [損失関数設定](#損失関数設定)
- [訓練設定](#訓練設定)
- [テスト設定](#テスト設定)
- [デバッグ設定](#デバッグ設定)
- [設定例](#設定例)

---

## 全般設定

基本的な実行環境とハードウェア設定です。

### パラメータ一覧

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `AUTO_RESUME` | bool | `false` | チェックポイントからの自動再開。`true`にすると`OUTPUT_DIR/checkpoint.pth`から学習を再開 |
| `CUDNN.BENCHMARK` | bool | `true` | cuDNNのベンチマークモードを有効化。入力サイズが固定の場合に高速化 |
| `CUDNN.DETERMINISTIC` | bool | `false` | 決定論的アルゴリズムの使用。再現性を重視する場合は`true` |
| `CUDNN.ENABLED` | bool | `true` | cuDNNを使用するかどうか |
| `DATA_DIR` | str | `''` | データディレクトリのベースパス（空の場合はデフォルト） |
| `GPUS` | tuple | `(0,)` | 使用するGPU IDのリスト（例: `(0, 1, 2, 3)`） |
| `OUTPUT_DIR` | str | `'output'` | モデルチェックポイントと結果の保存先ディレクトリ |
| `LOG_DIR` | str | `'log'` | TensorBoardログとテキストログの保存先 |
| `WORKERS` | int | `24` | DataLoaderのワーカープロセス数（CPUコア数に応じて調整） |
| `PRINT_FREQ` | int | `100` | ログ出力頻度（イテレーション単位） |

### 設定例

```yaml
AUTO_RESUME: false
CUDNN:
  BENCHMARK: true
  DETERMINISTIC: false
  ENABLED: true
DATA_DIR: ''
GPUS: (0,)
OUTPUT_DIR: 'output'
LOG_DIR: 'log'
WORKERS: 24
PRINT_FREQ: 100
```

---

## データセット設定

データの読み込み方法とData Augmentation（データ拡張）の設定です。

### パラメータ一覧

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `DATASET.COLOR_RGB` | bool | `true` | BGRからRGBへの色変換を行う |
| `DATASET.DATASET` | str | `'coco'` | 使用するデータセット（`'coco'` または `'mpii'`） |
| `DATASET.DATA_FORMAT` | str | `'jpg'` | 画像ファイルのフォーマット（`'jpg'`, `'zip'`） |
| `DATASET.FLIP` | bool | `true` | 水平フリップによるData Augmentation |
| `DATASET.NUM_JOINTS_HALF_BODY` | int | `8` | Half-body変換を適用する最小可視ジョイント数 |
| `DATASET.PROB_HALF_BODY` | float | `0.3` | Half-body変換を適用する確率 |
| `DATASET.ROOT` | str | `'data/coco'` | データセットのルートディレクトリ |
| `DATASET.ROT_FACTOR` | int | `45` | 回転Augmentationの角度範囲（±度） |
| `DATASET.SCALE_FACTOR` | float | `0.35` | スケールAugmentationの変動範囲（±比率） |
| `DATASET.TEST_SET` | str | `'val2017'` | テスト/検証用データセット名 |
| `DATASET.TRAIN_SET` | str | `'train2017'` | 訓練用データセット名 |

### Data Augmentation詳細

**回転（Rotation）**:
- `ROT_FACTOR: 45` → 画像を±45度の範囲でランダムに回転
- 確率: 60%で適用

**スケール（Scale）**:
- `SCALE_FACTOR: 0.35` → 画像サイズを65%～135%の範囲でランダムに変動
- 正規分布に基づいてスケーリング

**水平フリップ（Horizontal Flip）**:
- `FLIP: true` → 50%の確率で水平反転
- キーポイントの左右も適切にスワップ

**Half-body Transform**:
- 上半身または下半身のみを切り出して学習
- 小さな人物への対応力を向上

### 設定例

```yaml
DATASET:
  COLOR_RGB: true
  DATASET: 'coco'
  DATA_FORMAT: jpg
  FLIP: true
  NUM_JOINTS_HALF_BODY: 8
  PROB_HALF_BODY: 0.3
  ROOT: 'data/coco'
  ROT_FACTOR: 45
  SCALE_FACTOR: 0.35
  TEST_SET: 'val2017'
  TRAIN_SET: 'train2017'
```

---

## モデル設定

ネットワークアーキテクチャと座標表現方式の設定です。

### 基本パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `MODEL.INIT_WEIGHTS` | bool | `true` | 事前学習済み重みで初期化するかどうか |
| `MODEL.NAME` | str | `'pose_hrnet'` | バックボーンモデル名（`'pose_hrnet'`, `'pose_resnet'`など） |
| `MODEL.PRETRAINED` | str | `''` | ImageNet事前学習済みモデルのパス |
| `MODEL.NUM_JOINTS` | int | `17` | キーポイント数（COCO: 17, MPII: 16） |
| `MODEL.IMAGE_SIZE` | list | `[192, 256]` | 入力画像サイズ `[width, height]` |
| `MODEL.HEATMAP_SIZE` | list | `[192, 256]` | 出力ヒートマップサイズ `[width, height]` |

### 座標表現方式

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `MODEL.COORD_REPRESENTATION` | str | `'sa-simdr'` | 座標表現方式（`'heatmap'`, `'simdr'`, `'sa-simdr'`） |
| `MODEL.SIMDR_SPLIT_RATIO` | float | `2.0` | SimDRの解像度分割比率 |
| `MODEL.HEAD_INPUT` | int | `3072` | Transformerへの入力特徴量次元数 |
| `MODEL.SIGMA` | int | `4` | ヒートマップ生成時のガウシアン標準偏差 |

**座標表現方式の違い**:

- **`heatmap`**: 従来のヒートマップベース手法
- **`simdr`**: SimDR（Simple Discrete Representation）
- **`sa-simdr`**: SA-SimDR（Spatially-Aware SimDR、本研究で使用）

### HRNetアーキテクチャ設定

HRNetは複数のステージ（STAGE2, STAGE3, STAGE4）から構成されます。

#### STAGE2設定

```yaml
MODEL:
  EXTRA:
    STAGE2:
      NUM_MODULES: 1        # モジュールの反復回数
      NUM_BRANCHES: 2       # 並列ブランチ数
      BLOCK: BASIC          # Residualブロックタイプ
      NUM_BLOCKS: [4, 4]    # 各ブランチのブロック数
      NUM_CHANNELS: [32, 64] # 各ブランチのチャネル数
      FUSE_METHOD: SUM      # 特徴量融合方法
```

#### STAGE3設定

```yaml
    STAGE3:
      NUM_MODULES: 4        # モジュールを4回反復
      NUM_BRANCHES: 3       # 3つの解像度でマルチスケール処理
      BLOCK: BASIC
      NUM_BLOCKS: [4, 4, 4]
      NUM_CHANNELS: [32, 64, 128]
      FUSE_METHOD: SUM
```

#### STAGE4設定

```yaml
    STAGE4:
      NUM_MODULES: 3        # モジュールを3回反復
      NUM_BRANCHES: 4       # 4つの解像度
      BLOCK: BASIC
      NUM_BLOCKS: [4, 4, 4, 4]
      NUM_CHANNELS: [32, 64, 128, 256]
      FUSE_METHOD: SUM
```

#### その他のEXTRA設定

```yaml
    FINAL_CONV_KERNEL: 7    # 最終畳み込み層のカーネルサイズ
    PRETRAINED_LAYERS:      # 事前学習済み重みをロードするレイヤー
    - 'conv1'
    - 'bn1'
    - 'conv2'
    - 'bn2'
    - 'layer1'
    - 'transition1'
    - 'stage2'
    - 'transition2'
    - 'stage3'
    - 'transition3'
    - 'stage4'
```

### 設定例（HRNet-W32）

```yaml
MODEL:
  INIT_WEIGHTS: true
  NAME: pose_hrnet
  SIMDR_SPLIT_RATIO: 2.0
  HEAD_INPUT: 3072
  NUM_JOINTS: 17
  PRETRAINED: 'models/pytorch/imagenet/hrnet_w32-36af842e.pth'
  COORD_REPRESENTATION: 'sa-simdr'
  IMAGE_SIZE: [192, 256]
  HEATMAP_SIZE: [192, 256]
  SIGMA: 4
  EXTRA:
    PRETRAINED_LAYERS:
    - 'conv1'
    - 'bn1'
    # ... (上記参照)
    FINAL_CONV_KERNEL: 7
    STAGE2:
      NUM_MODULES: 1
      NUM_BRANCHES: 2
      BLOCK: BASIC
      NUM_BLOCKS: [4, 4]
      NUM_CHANNELS: [32, 64]
      FUSE_METHOD: SUM
    # STAGE3, STAGE4も同様
```

---

## 損失関数設定

モデルの学習に使用する損失関数の設定です。

### パラメータ一覧

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `LOSS.USE_TARGET_WEIGHT` | bool | `true` | ターゲット重み（可視性）を損失計算に使用 |
| `LOSS.TYPE` | str | `'KLDiscretLoss'` | 損失関数の種類 |
| `LOSS.LABEL_SMOOTHING` | float | `0.0` | ラベルスムージング係数（NMTCritierion使用時） |

### 損失関数の種類

- **`JointsMSELoss`**: ヒートマップベースのMSE損失
- **`NMTCritierion`**: SimDR用のクロスエントロピー損失
- **`NMTNORMCritierion`**: 正規化版NMT損失
- **`KLDiscretLoss`**: SA-SimDR用のKLダイバージェンス損失（推奨）

### 設定例

```yaml
LOSS:
  USE_TARGET_WEIGHT: true
  TYPE: 'KLDiscretLoss'
```

---

## 訓練設定

学習率、最適化手法、エポック数などの訓練パラメータです。

### パラメータ一覧

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `TRAIN.BATCH_SIZE_PER_GPU` | int | `32` | GPU毎のバッチサイズ |
| `TRAIN.SHUFFLE` | bool | `true` | 訓練データをシャッフル |
| `TRAIN.BEGIN_EPOCH` | int | `0` | 開始エポック番号 |
| `TRAIN.END_EPOCH` | int | `210` | 終了エポック番号 |
| `TRAIN.OPTIMIZER` | str | `'adam'` | 最適化手法（`'adam'`, `'sgd'`） |
| `TRAIN.LR` | float | `0.001` | 初期学習率 |
| `TRAIN.LR_FACTOR` | float | `0.1` | 学習率減衰係数 |
| `TRAIN.LR_STEP` | list | `[170, 200]` | 学習率を減衰させるエポック |
| `TRAIN.WD` | float | `0.0001` | Weight Decay（L2正則化係数） |
| `TRAIN.GAMMA1` | float | `0.99` | Cosine Annealing用パラメータ |
| `TRAIN.GAMMA2` | float | `0.0` | Cosine Annealing用パラメータ |
| `TRAIN.MOMENTUM` | float | `0.9` | モーメンタム（SGD使用時） |
| `TRAIN.NESTEROV` | bool | `false` | Nesterovモーメンタムの使用 |

### 学習率スケジュール

**MultiStepLR方式**（従来）:
```yaml
LR: 0.001
LR_FACTOR: 0.1
LR_STEP: [170, 200]
```
- Epoch 0-169: LR = 0.001
- Epoch 170-199: LR = 0.001 × 0.1 = 0.0001
- Epoch 200-210: LR = 0.0001 × 0.1 = 0.00001

**Cosine Annealing方式**（本実装）:
- `tools/train.py`内で実装
- `GAMMA1`と`GAMMA2`でコサインカーブを調整
- Warm-upとCosine Annealingを組み合わせ

### バッチサイズと実効バッチサイズ

実効バッチサイズ = `BATCH_SIZE_PER_GPU` × `len(GPUS)`

例:
- `BATCH_SIZE_PER_GPU: 32`, `GPUS: (0,)` → 実効バッチサイズ = 32
- `BATCH_SIZE_PER_GPU: 32`, `GPUS: (0, 1, 2, 3)` → 実効バッチサイズ = 128

### 設定例

```yaml
TRAIN:
  BATCH_SIZE_PER_GPU: 32
  SHUFFLE: true
  BEGIN_EPOCH: 0
  END_EPOCH: 210
  OPTIMIZER: adam
  LR: 0.001
  LR_FACTOR: 0.1
  LR_STEP: [170, 200]
  WD: 0.0001
  GAMMA1: 0.99
  GAMMA2: 0.0
  MOMENTUM: 0.9
  NESTEROV: false
```

---

## テスト設定

モデルの評価とテスト時の設定です。

### パラメータ一覧

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `TEST.BATCH_SIZE_PER_GPU` | int | `32` | テスト時のGPU毎バッチサイズ |
| `TEST.COCO_BBOX_FILE` | str | `''` | 外部人物検出器の結果ファイルパス |
| `TEST.USE_GT_BBOX` | bool | `true` | Ground Truthバウンディングボックスを使用 |
| `TEST.MODEL_FILE` | str | `''` | 評価に使用するモデルチェックポイントパス |
| `TEST.FLIP_TEST` | bool | `true` | Test Time Augmentation（TTA）で水平フリップ |
| `TEST.POST_PROCESS` | bool | `false` | ポスト処理（調整・補正）の有効化 |
| `TEST.SHIFT_HEATMAP` | bool | `true` | ヒートマップシフト補正 |
| `TEST.BBOX_THRE` | float | `1.0` | バウンディングボックス信頼度閾値 |
| `TEST.IMAGE_THRE` | float | `0.0` | 画像レベル信頼度閾値 |
| `TEST.IN_VIS_THRE` | float | `0.01` | 可視性閾値 |
| `TEST.NMS_THRE` | float | `1.0` | Non-Maximum Suppression閾値 |
| `TEST.OKS_THRE` | float | `0.9` | OKS（Object Keypoint Similarity）閾値 |

### Test Time Augmentation（TTA）

`FLIP_TEST: true`の場合:
1. オリジナル画像で推論
2. 水平フリップした画像で推論
3. 両者の結果を平均化

→ 精度向上（約0.5-1.0 AP改善）

### バウンディングボックスの扱い

**Ground Truth使用時** (`USE_GT_BBOX: true`):
- アノテーションのバウンディングボックスを使用
- 姿勢推定性能の純粋な評価に適している

**検出器使用時** (`USE_GT_BBOX: false`):
- 外部人物検出器（例: Faster R-CNN）の結果を使用
- `COCO_BBOX_FILE`で検出結果ファイルを指定
- より実用的な評価

### 設定例

```yaml
TEST:
  BATCH_SIZE_PER_GPU: 32
  COCO_BBOX_FILE: '/path/to/COCO_val2017_detections_AP_H_56_person.json'
  BBOX_THRE: 1.0
  IMAGE_THRE: 0.0
  IN_VIS_THRE: 0.01
  MODEL_FILE: '/path/to/checkpoint.pth'
  NMS_THRE: 1.0
  OKS_THRE: 0.9
  USE_GT_BBOX: true
  FLIP_TEST: true
  POST_PROCESS: false
  SHIFT_HEATMAP: true
```

---

## デバッグ設定

学習・評価時のビジュアライゼーション設定です。

### パラメータ一覧

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `DEBUG.DEBUG` | bool | `false` | デバッグモードの有効化 |
| `DEBUG.SAVE_BATCH_IMAGES_GT` | bool | `false` | Ground Truth画像の保存 |
| `DEBUG.SAVE_BATCH_IMAGES_PRED` | bool | `false` | 予測結果画像の保存 |
| `DEBUG.SAVE_HEATMAPS_GT` | bool | `false` | Ground Truthヒートマップの保存 |
| `DEBUG.SAVE_HEATMAPS_PRED` | bool | `false` | 予測ヒートマップの保存 |

### 出力ファイル

デバッグモードを有効にすると、以下が`OUTPUT_DIR/`以下に保存されます:

- `batch_*.jpg`: 入力画像とキーポイントの可視化
- `heatmap_gt_*.jpg`: Ground Truthヒートマップ
- `heatmap_pred_*.jpg`: 予測ヒートマップ

**注意**: デバッグモードは学習速度を低下させるため、通常は`false`に設定してください。

### 設定例

```yaml
DEBUG:
  DEBUG: true
  SAVE_BATCH_IMAGES_GT: true
  SAVE_BATCH_IMAGES_PRED: true
  SAVE_HEATMAPS_GT: true
  SAVE_HEATMAPS_PRED: true
```

---

## 設定例

### 提案手法（Transformer + SA-SimDR + Visibility）

```yaml
# experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml

AUTO_RESUME: false
CUDNN:
  BENCHMARK: true
  DETERMINISTIC: false
  ENABLED: true
GPUS: (0,)
OUTPUT_DIR: 'output'
LOG_DIR: 'log'
WORKERS: 24
PRINT_FREQ: 100

DATASET:
  COLOR_RGB: true
  DATASET: 'coco'
  DATA_FORMAT: jpg
  FLIP: true
  NUM_JOINTS_HALF_BODY: 8
  PROB_HALF_BODY: 0.3
  ROOT: 'data/coco'
  ROT_FACTOR: 45
  SCALE_FACTOR: 0.35
  TEST_SET: 'val2017'
  TRAIN_SET: 'train2017'

MODEL:
  INIT_WEIGHTS: true
  NAME: pose_hrnet
  SIMDR_SPLIT_RATIO: 2.0
  HEAD_INPUT: 3072
  NUM_JOINTS: 17
  PRETRAINED: 'models/pytorch/imagenet/hrnet_w32-36af842e.pth'
  COORD_REPRESENTATION: 'sa-simdr'
  IMAGE_SIZE: [192, 256]
  HEATMAP_SIZE: [192, 256]
  SIGMA: 4
  EXTRA:
    PRETRAINED_LAYERS:
    - 'conv1'
    - 'bn1'
    - 'conv2'
    - 'bn2'
    - 'layer1'
    - 'transition1'
    - 'stage2'
    - 'transition2'
    - 'stage3'
    - 'transition3'
    - 'stage4'
    FINAL_CONV_KERNEL: 7
    STAGE2:
      NUM_MODULES: 1
      NUM_BRANCHES: 2
      BLOCK: BASIC
      NUM_BLOCKS: [4, 4]
      NUM_CHANNELS: [32, 64]
      FUSE_METHOD: SUM
    STAGE3:
      NUM_MODULES: 4
      NUM_BRANCHES: 3
      BLOCK: BASIC
      NUM_BLOCKS: [4, 4, 4]
      NUM_CHANNELS: [32, 64, 128]
      FUSE_METHOD: SUM
    STAGE4:
      NUM_MODULES: 3
      NUM_BRANCHES: 4
      BLOCK: BASIC
      NUM_BLOCKS: [4, 4, 4, 4]
      NUM_CHANNELS: [32, 64, 128, 256]
      FUSE_METHOD: SUM

LOSS:
  USE_TARGET_WEIGHT: true
  TYPE: 'KLDiscretLoss'

TRAIN:
  BATCH_SIZE_PER_GPU: 32
  SHUFFLE: true
  BEGIN_EPOCH: 0
  END_EPOCH: 210
  OPTIMIZER: adam
  LR: 0.001
  LR_FACTOR: 0.1
  LR_STEP: [170, 200]
  WD: 0.0001
  GAMMA1: 0.99
  GAMMA2: 0.0
  MOMENTUM: 0.9
  NESTEROV: false

TEST:
  BATCH_SIZE_PER_GPU: 32
  USE_GT_BBOX: true
  FLIP_TEST: true
  POST_PROCESS: false
  SHIFT_HEATMAP: true
  BBOX_THRE: 1.0
  NMS_THRE: 1.0
  OKS_THRE: 0.9

DEBUG:
  DEBUG: false
  SAVE_BATCH_IMAGES_GT: false
  SAVE_BATCH_IMAGES_PRED: false
  SAVE_HEATMAPS_GT: false
  SAVE_HEATMAPS_PRED: false
```

**実行コマンド**:
```bash
python tools/train.py \
  --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml \
  --transformer --occlusion_mask_strategy
```

### ベースライン（Transformerなし）

```yaml
# experiments/coco/hrnet/sa_simdr/w32_256x192_adam_lr1e-3_split2_sigma4_210.yaml

# ほとんどの設定は上記と同じ
# 以下の点が異なる:

MODEL:
  HEAD_INPUT: 3072  # Transformerなしでも定義されているが未使用

# 実行時に --transformer フラグを指定しない
```

**実行コマンド**:
```bash
python tools/train.py \
  --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_adam_lr1e-3_split2_sigma4_210.yaml
```

---

## トラブルシューティング

### よくある設定ミス

**1. GPU数とバッチサイズの不整合**
```yaml
GPUS: (0, 1, 2, 3)  # 4 GPUs
BATCH_SIZE_PER_GPU: 64  # メモリ不足の可能性
```
→ GPU毎のメモリを確認し、バッチサイズを調整

**2. 事前学習済みモデルのパスミス**
```yaml
PRETRAINED: 'models/pytorch/imagenet/hrnet_w32-36af842e.pth'
```
→ ファイルが存在することを確認

**3. データセットパスの誤り**
```yaml
ROOT: 'data/coco'  # このディレクトリに images/ が必要
```
→ `data/coco/images/train2017/`, `data/coco/annotations/` が存在することを確認

**4. 座標表現方式と損失関数の不整合**
```yaml
COORD_REPRESENTATION: 'sa-simdr'
LOSS:
  TYPE: 'JointsMSELoss'  # ❌ 不整合
```
→ SA-SimDRには`KLDiscretLoss`を使用

### 推奨設定

**GPU数に応じたバッチサイズ**:
- 1 GPU (24GB): `BATCH_SIZE_PER_GPU: 32`
- 2 GPUs: `BATCH_SIZE_PER_GPU: 32` (実効64)
- 4 GPUs: `BATCH_SIZE_PER_GPU: 32` (実効128)

**ワーカー数**:
- CPUコア数の50-75%程度
- 例: 32コアCPU → `WORKERS: 16-24`

**学習時間の目安**（COCO train2017, 1 GPU）:
- 1エポック: 約2-3時間
- 210エポック: 約420-630時間（17-26日）

---

## 参考資料

- [HRNet論文](https://arxiv.org/abs/1902.09212)
- [SimDR論文](https://arxiv.org/abs/2107.03332)
- [本手法論文（WACV 2024）](https://openaccess.thecvf.com/content/WACV2024/html/Rajasegaran_Rethinking_Visibility_in_Human_Pose_Estimation_Occluded_Pose_Reasoning_via_WACV_2024_paper.html)
- [COCO Keypoints Dataset](https://cocodataset.org/#keypoints-2020)
