# Training Documentation - 論文提案手法の学習ガイド

このドキュメントでは、論文「Rethinking Visibility in Human Pose Estimation: Occluded Pose Reasoning via Transformers (WACV 2024)」の**提案手法**を使用した学習方法を説明します。

# VisNet(MLP)を利用してオクルージョンの有無を分類して学習する際の実行コマンド

```
python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml  --transformer  --occlusion_mask_strategy  --modelDir ./output/  --logDir ./log
```

---

## 目次

- [提案手法の概要](#提案手法の概要)
- [実装されている手法の違い](#実装されている手法の違い)
- [学習の前提条件](#学習の前提条件)
- [提案手法での学習手順](#提案手法での学習手順)
- [学習設定の詳細](#学習設定の詳細)
- [学習の監視方法](#学習の監視方法)
- [トラブルシューティング](#トラブルシューティング)

---

## 提案手法の概要

### アーキテクチャ

```
入力画像 (256×192)
    ↓
HRNet Backbone
特徴抽出 (17, 3072)
    ├──────────────────┐
    ↓                  ↓
VisibilityNet      Feature Copy
(可視性予測MLP)        ↓
    ↓              Masking
Visibility         (visible: 1.0
Scores (0-1)        occluded: 0.01)
    ↓                  ↓
二値化(閾値0.5)    Masked Features
    └──────────────────┘
            ↓
    Transformer Encoder
    (Self-Attention × 3層)
            ↓
      Output Layer
            ↓
    SA-SimDR 座標予測
    (X: 384次元, Y: 512次元)
```

### コア技術

1. **VisibilityNet**: ジョイント毎の可視性を予測（3層1D CNN）
2. **Occlusion-aware Masking**: 予測可視性に基づいて特徴量を抑制
3. **Transformer**: 可視ジョイントから遮蔽ジョイントを推論
4. **SA-SimDR**: 高解像度座標表現（従来の2倍）

---

## 実装されている手法の違い

このリポジトリには**3つの手法**が実装されています：

### 1. ベースライン（Baseline）

**YAMLファイル**: `w32_256x192_adam_lr1e-3_split2_sigma4_210.yaml`

**コマンド**:
```bash
python tools/train.py \
  --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_adam_lr1e-3_split2_sigma4_210.yaml
```

**構成**:
- HRNet-W32
- SA-SimDR
- Transformer: ✗
- VisibilityNet: ✗

**用途**: ベースライン性能の評価

---

### 2. Pseudo手法（GT可視性使用）

**YAMLファイル**: `tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml`

**コマンド**:
```bash
python tools/train.py \
  --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml \
  --transformer --occlusion_mask_strategy
```

**構成**:
- HRNet-W32
- SA-SimDR
- Transformer: ✓
- VisibilityNet: ✗（Ground Truth可視性を使用）
- Occlusion Masking: ✓

**用途**: 理想的な可視性情報での**上限性能評価**

**注意**:
- 訓練時: GT可視性でマスク作成
- テスト時: GT可視性が必要（実用的でない）

---

### 3. 提案手法（VisibilityNet使用）⭐

**YAMLファイル**: `tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml`（同じ）

**コマンド**:
```bash
python tools/train.py \
  --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml \
  --transformer --occlusion_mask_strategy \
  --modelDir ./output/ --logDir ./log
```

**構成**:
- HRNet-W32
- SA-SimDR
- Transformer: ✓
- VisibilityNet: ✓（**可視性を予測**）
- Occlusion Masking: ✓

**用途**: **論文で提案する実用的な手法**

**重要**: `tools/train.py`の315行目と322行目で`visibility_branch=visibility_branch`が設定されていることを確認

---

## 学習の前提条件

### 1. 環境構築

```bash
# Conda環境のアクティブ化
conda activate occluded_pose_reasoning

# 環境が未構築の場合
./setup_environment.sh
```

### 2. データセットの準備

```
data/coco/
├── images/
│   ├── train2017/  (シンボリックリンクまたは実体)
│   ├── val2017/
│   └── test2017/
└── annotations/
    ├── person_keypoints_train2017.json
    └── person_keypoints_val2017.json
```

**確認コマンド**:
```bash
ls -la data/coco/images/train2017/ | head
ls data/coco/annotations/
```

### 3. 事前学習済みモデル

```bash
# HRNet-W32 ImageNet事前学習済みモデル
mkdir -p models/pytorch/imagenet
wget -P models/pytorch/imagenet/ \
  https://github.com/HRNet/HRNet-Image-Classification/releases/download/v1.0/hrnet_w32-36af842e.pth

# 確認
ls -lh models/pytorch/imagenet/hrnet_w32-36af842e.pth
```

### 4. コード修正の確認

`tools/train.py`の315行目と322行目を確認：

```python
# 315行目（訓練）
visibility_branch=visibility_branch  # ← Noneでないこと

# 322行目（検証）
visibility_branch=visibility_branch  # ← Noneでないこと
```

---

## 提案手法での学習手順

### ステップ1: YAMLファイルの設定確認

`experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml`

**重要なパラメータ**:

```yaml
GPUS: (0,)                    # 使用するGPU（複数GPUの場合: (0, 1, 2, 3)）
WORKERS: 16                   # データローダーワーカー数（CPUコア数の50-75%）
PRINT_FREQ: 100               # ログ出力頻度

MODEL:
  COORD_REPRESENTATION: 'sa-simdr'  # SA-SimDRを使用
  SIMDR_SPLIT_RATIO: 2.0            # 2倍解像度
  HEAD_INPUT: 3072                  # Transformer入力次元

TRAIN:
  BATCH_SIZE_PER_GPU: 128     # バッチサイズ（GPUメモリに応じて調整）
  END_EPOCH: 210              # 総エポック数
  LR: 0.001                   # 学習率
  LR_STEP: [170, 200]         # 学習率減衰エポック

DEBUG:
  DEBUG: false                # 本番学習ではfalse推奨
```

### ステップ2: 学習の開始

```bash
# プロジェクトルートに移動
cd /path/to/Occluded-Pose-Reasoning

# Conda環境をアクティブ化
conda activate occluded_pose_reasoning

# 学習開始（提案手法）
python tools/train.py \
  --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml \
  --transformer \
  --occlusion_mask_strategy \
  --modelDir ./output/ \
  --logDir ./log
```

**フラグの説明**:
- `--cfg`: 設定ファイルのパス
- `--transformer`: Transformerを有効化
- `--occlusion_mask_strategy`: オクルージョンマスキング戦略を有効化
- `--modelDir`: モデルチェックポイントの保存先
- `--logDir`: ログファイルの保存先

### ステップ3: 学習開始の確認

正常に開始されると、以下のログが表示されます：

```
=> creating output/coco/pose_hrnet/tilsiter_...
=> creating log/coco/pose_hrnet/tilsiter_..._2025-10-01-XX-XX
=> init weights from normal distribution
=> loading pretrained model models/pytorch/imagenet/hrnet_w32-36af842e.pth
Total number of parameters: 29332209

loading annotations into memory...
Done (t=3.70s)
=> num_images: 118287
=> load 149813 samples

=> num_images: 5000
=> load 6352 samples
```

**⚠️ この警告が表示されないことを確認**:
```
Please note you are not using visibility_branch...  # ← 出てはいけない
```

この警告が出る場合、VisNetが使われていません（Pseudo手法になっている）。

---

## 学習設定の詳細

### 損失関数

提案手法では**2つの損失**を同時に最適化します：

#### 1. 姿勢推定損失（KL Divergence Loss）

```python
L_pose = KLDiscretLoss(pred_x, pred_y, target_x, target_y, target_weight)
```

- SA-SimDRの座標分布とGround Truthとの差異
- 主要な損失

#### 2. 可視性予測損失（Cross Entropy Loss）

```python
L_vis = CrossEntropyLoss(pred_visibility, gt_visibility)
```

- VisibilityNetの予測とGround Truth可視性との差異
- VisibilityNetの学習に使用

**総合損失**（コード内で自動的に計算）:
```python
loss.backward()            # 姿勢推定損失
loss_visibility.backward() # 可視性損失
optimizer.step()           # 両方の勾配で更新
```

### Optimizer設定

```python
optimizer = torch.optim.AdamW([
    {'params': model.parameters()},              # HRNet: LR=0.001
    {'params': output_layer.parameters()},       # Output層: LR=0.001
    {'params': visibility_branch.parameters()},  # VisNet: LR=0.001
    {'params': transformer.parameters(), 'lr': 1e-4}  # Transformer: LR=0.0001
], lr=cfg.TRAIN.LR)
```

**学習率の違い**:
- HRNet, Output層, VisNet: 0.001（高速学習）
- Transformer: 0.0001（慎重に学習）

### 学習率スケジューラ

**Cosine Annealing with Warm-up**:

```python
# Transformerのみ1000イテレーションのWarm-up
# その後Cosine Annealingで徐々に減衰
# 最小学習率: 1e-5
```

**スケジュール**:
- Epoch 0-169: 初期学習率
- Epoch 170-199: 学習率×0.1
- Epoch 200-210: 学習率×0.01

---

## 学習の監視方法

### 1. リアルタイムログ監視

```bash
# ターミナルでの出力
Epoch: [0][0/1171] Time 2.345s Loss 12.3456 (12.3456) Acc 0.000 (0.000) Lr 0.001000
Epoch: [0][100/1171] Time 0.345s Loss 8.2345 (9.1234) Acc 0.123 (0.098) Lr 0.001000
...
```

**見方**:
- `[0][100/1171]`: エポック0, イテレーション100/1171
- `Loss 8.2345 (9.1234)`: 現在のバッチ損失 (平均損失)
- `Acc 0.123 (0.098)`: 現在の精度 (平均精度)
- `Lr 0.001000`: 現在の学習率

### 2. GPU使用率の監視

```bash
# 別ターミナルで実行
watch -n 1 nvidia-smi

# または
nvidia-smi -l 1
```

**理想的な状態**:
- GPU使用率: 95-100%
- メモリ使用率: 80-95%
- 温度: 70-85°C

### 3. TensorBoard監視

```bash
# TensorBoard起動
tensorboard --logdir=log/coco/pose_hrnet/ --port=6006

# ブラウザでアクセス
# http://localhost:6006
```

**監視項目**:
- `train_loss`: 訓練損失の推移
- 学習率の推移
- 検証精度（APなど）

### 4. ログファイル確認

```bash
# 最新のログファイルを確認
tail -f log/coco/pose_hrnet/tilsiter_.../train_*.log
```

---

## チェックポイントとモデル保存

### 保存されるファイル

```
output/coco/pose_hrnet/tilsiter_.../
├── checkpoint.pth       # 最新のチェックポイント
├── model_best.pth       # 最高性能のモデル
└── final_state.pth      # 最終エポックのHRNet重み
```

### チェックポイントの内容

```python
checkpoint.pth = {
    'epoch': 10,
    'model_state_dict': ...,           # HRNet重み
    'transformer_state_dict': ...,     # Transformer重み
    'output_state_dict': ...,          # Output層重み
    'visibility_state_dict': ...,      # VisibilityNet重み ⭐
    'optimizer': ...,                  # Optimizer状態
    'perf': 0.723                      # 検証AP
}
```

### 学習の再開

```yaml
# YAMLファイルで設定
AUTO_RESUME: true
```

または

```bash
# checkpoint.pthが存在する場合、自動的に再開
python tools/train.py --cfg ... --transformer --occlusion_mask_strategy
```

---

## 学習時間の目安

### GPU構成別の学習時間（210エポック）

| GPU構成 | バッチサイズ | 1エポック | 総時間 |
|---------|------------|---------|--------|
| 1×RTX 4090 | 64 | 2.5時間 | **22日** |
| 1×RTX 4090 | 128 | 1.8時間 | **16日** |
| 2×RTX 4090 | 128 (実効256) | 1.0時間 | **9日** |
| 4×RTX 4090 | 128 (実効512) | 0.6時間 | **5日** |

**計算根拠**:
- 訓練サンプル数: 149,813
- イテレーション/エポック: 149,813 ÷ バッチサイズ
- イテレーション時間: 約0.35秒（Transformer + VisNet込み）

---

## トラブルシューティング

### 1. Out of Memory (OOM)エラー

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解決策**:
```yaml
TRAIN:
  BATCH_SIZE_PER_GPU: 64  # 128 → 64に削減
```

または

```yaml
WORKERS: 8  # 16 → 8に削減（メモリ節約）
```

### 2. VisNetが学習されていない

**確認方法**:
```bash
# ログに以下が表示される
"Please note you are not using visibility_branch..."
```

**原因**: `tools/train.py`が修正されていない

**解決策**:
```python
# tools/train.py: 315行目と322行目
visibility_branch=visibility_branch  # Noneになっていないか確認
```

### 3. GPU使用率が低い（<70%）

**原因**: データローディングがボトルネック

**解決策**:
```yaml
WORKERS: 24  # 16 → 24に増加
```

### 4. 損失がNaNになる

**原因**: 学習率が高すぎる、または勾配爆発

**解決策**:
```yaml
TRAIN:
  LR: 0.0005  # 0.001 → 0.0005に削減
```

または勾配クリッピングを追加（コード修正が必要）

### 5. 可視性予測精度が上がらない

**原因**: VisNetの学習が不安定

**対処法**:
1. VisNetの学習率を調整
2. 可視性損失の重みを調整（コード修正が必要）
3. より多くのエポックで学習

---

## ベストプラクティス

### 1. デバッグモードの無効化

学習開始前に必ず確認：

```yaml
DEBUG:
  DEBUG: false                    # true → false
  SAVE_BATCH_IMAGES_GT: false
  SAVE_BATCH_IMAGES_PRED: false
  SAVE_HEATMAPS_GT: false
  SAVE_HEATMAPS_PRED: false
```

デバッグモードは**学習速度を30-50%低下**させます。

### 2. ハードウェアに応じた設定

**高性能GPU（RTX 4090 24GB）**:
```yaml
BATCH_SIZE_PER_GPU: 128
WORKERS: 16
```

**中性能GPU（RTX 3090 24GB）**:
```yaml
BATCH_SIZE_PER_GPU: 64
WORKERS: 12
```

**低性能GPU（RTX 3080 12GB）**:
```yaml
BATCH_SIZE_PER_GPU: 32
WORKERS: 8
```

### 3. 定期的なバックアップ

```bash
# 重要なチェックポイントをバックアップ
cp output/coco/pose_hrnet/tilsiter_.../model_best.pth \
   ~/backups/model_best_epoch100.pth
```

### 4. 検証頻度の調整

デフォルトでは1エポック毎に検証が実行されますが、時間短縮のため調整可能（コード修正が必要）。

---

## 評価の実行

学習完了後、モデルを評価：

```bash
python tools/test.py \
  --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml \
  --transformer \
  --modelDir ./output/ \
  --logDir ./log
```

**評価指標**:
- AP (Average Precision): 全体精度
- AP@0.5: IoU 0.5での精度
- AP@0.75: IoU 0.75での精度
- APM, APL: 中・大サイズ人物での精度

---

## 参考情報

### 論文

- **タイトル**: Rethinking Visibility in Human Pose Estimation: Occluded Pose Reasoning via Transformers
- **会議**: WACV 2024 (Oral)
- **リンク**: [OpenAccess](https://openaccess.thecvf.com/content/WACV2024/html/Rajasegaran_Rethinking_Visibility_in_Human_Pose_Estimation_Occluded_Pose_Reasoning_via_WACV_2024_paper.html)

### 関連ドキュメント

- `ARCHITECTURE.md`: アーキテクチャの詳細
- `CONFIG_GUIDE.md`: YAMLファイルの全パラメータ説明
- `README.md`: プロジェクト概要

### 学習済みモデル

論文著者が提供する学習済みモデル:
- [Google Drive](https://drive.google.com/drive/folders/1WlbhN_3FVBkcQijuoTRbzFYB7UUMLuMc?usp=sharing)

---

## まとめ

**提案手法（VisibilityNet使用）の学習**には以下が必要です：

1. ✅ `tools/train.py`で`visibility_branch=visibility_branch`が設定されている
2. ✅ コマンドラインで`--transformer --occlusion_mask_strategy`を指定
3. ✅ YAMLファイルで`COORD_REPRESENTATION: 'sa-simdr'`
4. ✅ COCOデータセットと事前学習済みモデルの準備

**学習時間**: 1GPU（RTX 4090）で約16-22日

**期待される性能**: COCO val2017で AP 70-75% 程度（オクルージョン環境で特に強い）
