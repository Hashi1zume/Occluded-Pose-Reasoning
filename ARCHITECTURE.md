# Occluded Pose Reasoning via Transformers - アーキテクチャドキュメント

このドキュメントは、論文「Rethinking Visibility in Human Pose Estimation: Occluded Pose Reasoning via Transformers (WACV 2024)」のPyTorch実装の詳細なアーキテクチャと使用方法を説明します。

---

## 目次

1. [プロジェクト概要](#プロジェクト概要)
2. [アーキテクチャ全体図](#アーキテクチャ全体図)
3. [コンポーネント詳細](#コンポーネント詳細)
4. [データセット配置](#データセット配置)
5. [学習方法](#学習方法)
6. [設定ファイル（YAML）の種類](#設定ファイルyamlの種類)
7. [モデルチェックポイント](#モデルチェックポイント)
8. [評価方法](#評価方法)

---

## プロジェクト概要

### 研究背景

従来の姿勢推定手法は、オクルージョン（遮蔽）が存在する環境で精度が大幅に低下します。本研究では、Transformerを用いて関節間の依存関係を学習し、遮蔽された関節の位置を推論する手法を提案します。

### 主要な貢献

1. **Transformer based Joint Reasoning**: 関節間のSelf-Attentionによる依存関係学習
2. **Visibility-Aware Training**: 可視性情報を明示的に活用した訓練戦略
3. **Occlusion Mask Strategy**: 遮蔽された関節の特徴を意図的に減衰させる学習手法

---

## アーキテクチャ全体図

### ベースライン（Transformerなし）

```
入力画像 (256x192)
    ↓
┌─────────────────────┐
│   HRNet Backbone    │  ← ImageNet事前学習済み
│   (pose_hrnet.py)   │
└─────────────────────┘
    ↓
特徴マップ (17 x H x W)
    ↓
┌─────────────────────┐
│   Output Layer      │
│   SA-SimDR Head     │
└─────────────────────┘
    ↓
座標 (x, y) × 17関節
```

### 提案手法（Transformer + VisibilityNet）

```
入力画像 (256x192)
    ↓
┌──────────────────────────────────────────────────────┐
│              HRNet Backbone (pose_hrnet.py)          │
│              ImageNet事前学習済み                      │
└──────────────────────────────────────────────────────┘
    ↓
特徴マップ (17 x 64 x 48)
    ↓ ─────────────┐
    │              │
    ↓              ↓
┌─────────────┐  ┌──────────────────────────────┐
│ VisibilityNet│  │   TransformerOPORTEncoder   │
│ (HRNetJoint- │  │   (transformer.py)           │
│  VisibilityNet)│  │                              │
└─────────────┘  │  ┌────────────────────────┐  │
    ↓            │  │ 1. conv_learn_tokens   │  │
可視性予測       │  │    (特徴→トークン変換)  │  │
(17関節)         │  ├────────────────────────┤  │
    ↓            │  │ 2. get_visibility_mask │  │
    │            │  │    (マスク生成)         │  │
    │            │  ├────────────────────────┤  │
    │            │  │ 3. Masked Embedding    │  │
    │────────────┼─→│    (要素積適用)         │  │
    │            │  ├────────────────────────┤  │
    │            │  │ 4. Multi-Head          │  │
    │            │  │    Self-Attention × N  │  │
    │            │  │    (関節間依存関係学習) │  │
    │            │  └────────────────────────┘  │
    │            └──────────────────────────────┘
    │                           ↓
    │            Transformer特徴 (17 x 3072)
    │                           ↓
    │            ┌─────────────────────┐
    │            │   Output Layer      │
    │            │   SA-SimDR Head     │
    │            └─────────────────────┘
    │                           ↓
    │            座標 (x, y) × 17関節
    │                           │
    └───────────────────────────┘
            (訓練時の損失計算)
```

---

## コンポーネント詳細

### 1. HRNet Backbone

**ファイル**: `lib/models/pose_hrnet.py`

**役割**: 入力画像から高解像度の特徴マップを抽出

**特徴**:
- マルチスケール特徴を並列に処理
- 高解像度表現を維持
- ImageNet事前学習済みモデルを使用

**出力**: `[Batch, 17, 64, 48]` の特徴マップ（各関節ごと）

---

### 2. VisibilityNet（可視性予測ネットワーク）

**ファイル**: `lib/models/transformer.py:591-610`

**クラス**: `HRNetJointVisibilityNet`

**アーキテクチャ**:
```python
Input: [B, 17, 3072]  # バックボーンからの特徴
  ↓
Conv1D (3072 → 64)
  ↓ ReLU
Conv1D (64 → 32)
  ↓ ReLU
Conv1D (32 → 1)
  ↓ Sigmoid
Output: [B, 17]  # 各関節の可視性スコア (0~1)
```

**損失関数**: `CrossEntropyLoss`（2値分類: 可視 vs 不可視）

**役割**:
- 各関節の可視性を予測（0.5を閾値として2値化）
- テスト時はGround Truthの可視性情報不要

---

### 3. TransformerOPORTEncoder

**ファイル**: `lib/models/transformer.py:375-461`

**クラス**: `TransformerOPORTEncoder`

#### 処理フロー

```python
def forward(x, visibility_state, occlusion_mask_strategy):
    # Step 1: 特徴マップをジョイントトークンに変換
    x = conv_learn_tokens(x)  # [B, 256, H*W] → [B, 17, H*W]

    # Step 2: 可視性マスク生成
    if occlusion_mask_strategy:
        mask = get_visibility_mask(visibility_state)
        # 可視: 1.0, 遮蔽: 0.01
        x_masked = x.mul(mask.unsqueeze(2))
        out = embedding_layer(x_masked)
    else:
        out = embedding_layer(x)

    # Step 3: 位置エンコーディング（オプション）
    if self.pe:
        out = positional_encoder(out)

    # Step 4: Transformerレイヤー（Self-Attention）
    for layer in self.layers:
        out = layer(out, out, out)  # Q, K, V全て同じ

    # Step 5: 最終変換
    out = fc(out)  # [B, 17, 3072]

    return out
```

#### 可視性マスクの仕組み

`get_visibility_mask()`の動作:

```python
# visibility_state: [B, 17]
# 値: 0=見えない, 1=部分的, 2=完全に可視

is_visible = (visibility_state == 2).float()      # [B, 17]
is_occluded = (visibility_state != 2).float()    # [B, 17]

mask = is_visible * 1.0 + is_occluded * 0.01
# 可視な関節: 1.0
# 遮蔽された関節: 0.01
```

**設計意図**:
- 遮蔽された関節の特徴を完全にゼロにしない（0.01を残す）
- Transformerに「この関節は信頼できない」というシグナルを与える
- Self-Attentionで可視な関節から情報を借りて推論

---

### 4. Multi-Head Self-Attention

**ファイル**: `lib/models/transformer.py:88-180`

**クラス**: `MultiHeadAttention`

**パラメータ**:
- ヘッド数: 8
- 埋め込み次元: 512
- 各ヘッドの次元: 64

**計算式**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**役割**: 関節間の依存関係を学習（例: 肩と手首が見えていれば、隠れた肘を推論）

---

### 5. Output Layer（SA-SimDR Head）

**ファイル**: `lib/models/transformer.py:464-567`

**役割**: Transformer特徴から関節座標を直接回帰

**出力**:
- `output_x`: [B, 17, 192*2] → X座標の分布
- `output_y`: [B, 17, 256*2] → Y座標の分布

**損失関数**: `KLDiscretLoss`（分布間のKLダイバージェンス）

---

## データセット配置

### COCO Keypointsデータセット

#### ディレクトリ構造

```
<プロジェクトルート>/
├── data/
│   └── coco/
│       ├── annotations/
│       │   ├── person_keypoints_train2017.json
│       │   └── person_keypoints_val2017.json
│       ├── train2017/
│       │   └── [118,287枚の画像]
│       └── val2017/
│           └── [5,000枚の画像]
```

#### ダウンロード方法

```bash
cd <プロジェクトルート>
mkdir -p data/coco && cd data/coco

# 訓練画像（19GB）
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# 検証画像（1GB）
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# アノテーション（241MB）
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### 事前学習済みモデル

```bash
mkdir -p models/pytorch/imagenet
cd models/pytorch/imagenet

# HRNet-W32のImageNet事前学習モデル
wget https://github.com/HRNet/HRNet-Image-Classification/releases/download/v1.0/hrnet_w32-36af842e.pth
```

---

## 学習方法

### 1. ベースライン訓練（Transformerなし）

```bash
python tools/train.py \
    --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_adam_lr1e-3_split2_sigma4_210.yaml
```

**構成**: HRNet → SA-SimDR Output Layer

**訓練対象**:
- バックボーンのパラメータ
- 出力層のパラメータ

**用途**: 比較実験のベースライン

---

### 2. 提案手法訓練（Transformer + VisibilityNet）

```bash
python tools/train.py \
    --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml \
    --transformer \
    --occlusion_mask_strategy
```

**構成**: HRNet → Transformer → SA-SimDR Output Layer + VisibilityNet

**訓練対象**:
- バックボーンのパラメータ
- Transformerのパラメータ（学習率: 1e-4）
- 出力層のパラメータ
- VisibilityNetのパラメータ

**用途**: 論文の提案手法

---

### オプティマイザの構成

`tools/train.py:230-234` より:

```python
optimizer = torch.optim.AdamW([
    {'params': model.parameters()},              # バックボーン
    {'params': output_layer.parameters()},       # 出力層
    {'params': visibility_branch.parameters()},  # VisibilityNet
    {'params': transformer.parameters(), 'lr': 1e-4}  # Transformer
], lr=cfg.TRAIN.LR)  # デフォルト: 1e-3
```

**注意**: Transformerのみ低い学習率（1e-4）を使用

---

### 訓練の流れ

1. **Forward Pass**:
   - 画像 → HRNet → 特徴マップ
   - 特徴マップ → VisibilityNet → 可視性予測
   - 特徴マップ + 可視性マスク → Transformer → 拡張特徴
   - 拡張特徴 → Output Layer → 座標予測

2. **Loss Calculation**:
   - 座標損失: `KLDiscretLoss(pred_coords, target_coords)`
   - 可視性損失: `CrossEntropyLoss(pred_visibility, gt_visibility)`

3. **Backward Pass**:
   ```python
   optimizer.zero_grad()
   loss.backward()              # 座標損失の逆伝播
   loss_visibility.backward()   # 可視性損失の逆伝播
   optimizer.step()
   ```

---

## 設定ファイル（YAML）の種類

### 1. ベースライン設定

**ファイル**: `experiments/coco/hrnet/sa_simdr/w32_256x192_adam_lr1e-3_split2_sigma4_210.yaml`

#### 主要な設定

```yaml
GPUS: (0,)
DATASET:
  DATASET: 'coco'
  ROOT: 'data/coco'
  TRAIN_SET: 'train2017'
  TEST_SET: 'val2017'

MODEL:
  NAME: pose_hrnet
  COORD_REPRESENTATION: 'sa-simdr'
  PRETRAINED: 'models/pytorch/imagenet/hrnet_w32-36af842e.pth'
  IMAGE_SIZE: [192, 256]
  NUM_JOINTS: 17

TRAIN:
  BATCH_SIZE_PER_GPU: 32
  END_EPOCH: 210
  LR: 0.001
  LR_STEP: [170, 200]

LOSS:
  TYPE: 'KLDiscretLoss'

TEST:
  USE_GT_BBOX: true
  FLIP_TEST: true
```

#### 使用コマンド

```bash
python tools/train.py \
    --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_adam_lr1e-3_split2_sigma4_210.yaml
```

**フラグ**: なし

---

### 2. 提案手法設定

**ファイル**: `experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml`

#### ベースラインとの違い

| 項目 | ベースライン | 提案手法 |
|------|------------|---------|
| **必須フラグ** | なし | `--transformer`<br>`--occlusion_mask_strategy` |
| **アーキテクチャ** | HRNet → Output | HRNet → Transformer → Output<br>+ VisibilityNet |
| **可視性推論** | なし | あり |
| **学習対象** | 2モジュール | 4モジュール |
| **パラメータ数** | ~28M | ~32M |

#### 使用コマンド

```bash
python tools/train.py \
    --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml \
    --transformer \
    --occlusion_mask_strategy
```

**必須フラグ**:
- `--transformer`: Transformerを有効化
- `--occlusion_mask_strategy`: 可視性マスク戦略を使用

---

### 設定ファイルのカスタマイズ

#### GPU数の変更

```yaml
GPUS: (0,1,2,3)  # 4GPUを使用
```

#### バッチサイズの調整

```yaml
TRAIN:
  BATCH_SIZE_PER_GPU: 16  # GPU毎のバッチサイズ
```

実効バッチサイズ = `BATCH_SIZE_PER_GPU` × GPU数

#### エポック数の変更

```yaml
TRAIN:
  END_EPOCH: 100  # 210 → 100に短縮
```

---

## モデルチェックポイント

### 保存される内容

#### `checkpoint.pth`（完全版）

```python
{
    'epoch': 210,
    'model': 'pose_hrnet',
    'model_state_dict': {...},         # バックボーンの重み
    'transformer_state_dict': {...},   # Transformerの重み
    'output_state_dict': {...},        # 出力層の重み
    'visibility_state_dict': {...},    # VisibilityNetの重み
    'best_state_dict': {...},          # 最良モデル
    'perf': 0.756,                     # 性能指標（AP）
    'optimizer': {...}                 # Optimizerの状態
}
```

**用途**: 訓練の再開、全コンポーネントの復元

---

#### `final_state.pth`（バックボーンのみ）

```python
{
    'conv1.weight': Tensor(...),
    'bn1.weight': Tensor(...),
    # ... (バックボーンのパラメータのみ)
}
```

**用途**: 他のタスクへの転移学習、推論専用

---

### チェックポイントの読み込み

```python
import torch

# 完全なチェックポイント
checkpoint = torch.load('checkpoint.pth', map_location='cpu')

# 各コンポーネントの復元
model.load_state_dict(checkpoint['model_state_dict'])
transformer.load_state_dict(checkpoint['transformer_state_dict'])
output_layer.load_state_dict(checkpoint['output_state_dict'])
visibility_branch.load_state_dict(checkpoint['visibility_state_dict'])
```

---

## 評価方法

### テストコマンド

```bash
python tools/test.py \
    --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml \
    --transformer
```

### 評価指標

COCOデータセットでは以下の指標を使用:

- **AP (Average Precision)**: 主要指標
- **AP50**: IoU=0.50での精度
- **AP75**: IoU=0.75での精度
- **APM**: 中サイズの人物での精度
- **APL**: 大サイズの人物での精度

### 期待される性能

論文の報告値（COCO val2017）:

| モデル | AP | AP50 | AP75 |
|--------|-----|------|------|
| ベースライン（HRNet-W32） | 74.9 | 90.2 | 82.1 |
| + Transformer | 75.3 | 90.4 | 82.5 |
| + Transformer + VisNet | **75.6** | 90.5 | 82.8 |

---

## ディレクトリ構造

```
Occluded-Pose-Reasoning/
├── data/                       # データセット
│   └── coco/
│       ├── annotations/
│       ├── train2017/
│       └── val2017/
├── experiments/                # 実験設定ファイル
│   └── coco/hrnet/sa_simdr/
│       ├── w32_256x192_adam_lr1e-3_split2_sigma4_210.yaml
│       └── tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml
├── lib/                        # ライブラリコード
│   ├── config/                 # 設定管理
│   ├── core/                   # 訓練・評価ループ
│   │   ├── function.py         # 訓練関数
│   │   └── loss.py             # 損失関数
│   ├── dataset/                # データローダー
│   │   └── coco.py
│   └── models/                 # モデル定義
│       ├── pose_hrnet.py       # バックボーン
│       └── transformer.py      # Transformer、VisNet
├── models/                     # 事前学習済みモデル
│   └── pytorch/imagenet/
│       └── hrnet_w32-36af842e.pth
├── output/                     # 訓練済みモデル（自動生成）
├── tools/                      # 実行スクリプト
│   ├── train.py                # 訓練
│   ├── test.py                 # テスト
│   └── evaluate.py             # 評価
└── README.md
```

---

## トラブルシューティング

### Q1: CUDA out of memory

**解決策**: バッチサイズを削減

```yaml
TRAIN:
  BATCH_SIZE_PER_GPU: 16  # 32 → 16
```

### Q2: データセットが見つからない

**原因**: `DATASET.ROOT`のパスが正しくない

**解決策**: 絶対パスまたは相対パスを確認

```yaml
DATASET:
  ROOT: '/absolute/path/to/data/coco'
```

### Q3: 事前学習モデルが読み込めない

**原因**: `MODEL.PRETRAINED`のパスが正しくない

**解決策**: ダウンロードして正しい位置に配置

```bash
mkdir -p models/pytorch/imagenet
cd models/pytorch/imagenet
wget https://github.com/HRNet/HRNet-Image-Classification/releases/download/v1.0/hrnet_w32-36af842e.pth
```

---

## 参考文献

```bibtex
@inproceedings{sun2024rethinking,
  title={Rethinking Visibility in Human Pose Estimation: Occluded Pose Reasoning via Transformers},
  author={Sun, Pengzhan and Gu, Kerui and Wang, Yunsong and Yang, Linlin and Yao, Angela},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5903--5912},
  year={2024}
}
```

---

## 連絡先

- Email: pengzhansun6@gmail.com
- Google Drive (事前学習モデル): [リンク](https://drive.google.com/drive/folders/1WlbhN_3FVBkcQijuoTRbzFYB7UUMLuMc?usp=sharing)
