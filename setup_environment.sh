#!/bin/bash
# Occluded-Pose-Reasoning 学習環境構築スクリプト
#
# 前提条件:
# - Conda/Miniconda がインストール済み
# - CUDA 12.1以上がインストール済み
# - COCOデータセットが data/coco/ に配置済み（images/ディレクトリ構造を含む）

set -e  # エラーで即座に停止

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Occluded-Pose-Reasoning 環境構築"
echo "========================================"
echo ""

# 環境名の設定
ENV_NAME="occluded_pose_reasoning"

# Conda環境の存在確認
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "[確認] Conda環境 '${ENV_NAME}' は既に存在します"
    read -p "既存環境を削除して再作成しますか？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[1/7] 既存環境を削除中..."
        conda deactivate 2>/dev/null || true
        conda env remove -n ${ENV_NAME} -y
    else
        echo "既存環境を使用します"
        eval "$(conda shell.bash hook)"
        conda activate ${ENV_NAME}
        skip_conda_create=true
    fi
fi

# Conda環境の作成
if [ -z "$skip_conda_create" ]; then
    echo "[1/7] Conda環境を作成中..."
    conda create -n ${ENV_NAME} python=3.11 -y
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
    echo "✓ Conda環境 '${ENV_NAME}' を作成しました"
fi

# PyTorchのインストール
echo ""
echo "[2/7] PyTorch (CUDA 12.1) をインストール中..."
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
echo "✓ PyTorch インストール完了"

# 依存パッケージのインストール
echo ""
echo "[3/7] 依存パッケージをインストール中..."
pip install --no-cache-dir -r requirement.txt
echo "✓ 依存パッケージ インストール完了"

# NumPyバージョンの確認と固定
echo ""
echo "[4/7] NumPyバージョンを確認中..."
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)")
if [[ ! "$NUMPY_VERSION" == "1.24.3" ]]; then
    echo "⚠ NumPy ${NUMPY_VERSION} が検出されました。1.24.3に修正します..."
    pip uninstall -y numpy scipy pandas scikit-image matplotlib 2>/dev/null || true
    pip install --no-cache-dir numpy==1.24.3
    pip install --no-cache-dir 'scipy<1.12' 'pandas<2.0' 'scikit-image<0.23' 'matplotlib<3.8'
fi
echo "✓ NumPy 1.24.3 を確認しました"

# NMSモジュールのコンパイル準備
echo ""
echo "[5/7] NMS Cythonモジュールをビルド中..."
cd lib/nms

# 古いコンパイル済みファイルを削除
rm -f *.so *.c *.cpp
rm -rf build/

# Cythonソースコードの修正 (np.int_t -> np.intp_t)
echo "  - Cythonソースコードを修正中..."
sed -i.bak 's/np\.int_t/np.intp_t/g' cpu_nms.pyx
sed -i.bak 's/np\.int_t/np.intp_t/g' gpu_nms.pyx

# CUDA architectureフラグの更新 (sm_35 -> sm_60 for CUDA 12.x)
if grep -q "arch=sm_35" setup_linux.py; then
    sed -i.bak "s/-arch=sm_35/-arch=sm_60/g" setup_linux.py
    echo "  - CUDA architecture sm_35 → sm_60 に更新"
fi

# NMSモジュールのビルド
echo "  - NMSモジュールをコンパイル中..."
python setup_linux.py build_ext --inplace

if [ ! -f "cpu_nms.*.so" ] && [ ! -f "gpu_nms.*.so" ]; then
    echo "✗ NMSモジュールのビルドに失敗しました"
    exit 1
fi

cd ../..
echo "✓ NMSモジュール ビルド完了"

# 事前学習済みモデルディレクトリの作成
echo ""
echo "[6/7] 事前学習済みモデルディレクトリを準備中..."
mkdir -p models/pytorch/imagenet

if [ ! -f "models/pytorch/imagenet/hrnet_w32-36af842e.pth" ]; then
    echo "  ⚠ HRNet事前学習済みモデルが見つかりません"
    echo "    以下のコマンドでダウンロードしてください:"
    echo "    wget -P models/pytorch/imagenet/ https://github.com/HRNet/HRNet-Image-Classification/releases/download/v1.0/hrnet_w32-36af842e.pth"
else
    echo "  ✓ HRNet事前学習済みモデルを確認しました"
fi

# 出力ディレクトリの作成
echo ""
echo "[7/7] 出力ディレクトリを作成中..."
mkdir -p output
mkdir -p log
echo "✓ 出力ディレクトリを作成しました"

# インポートテスト
echo ""
echo "[テスト] インポートテストを実行中..."
python test_imports.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 環境構築が完了しました！"
    echo "========================================"
    echo ""
    echo "次のステップ:"
    echo ""
    echo "1. Conda環境のアクティブ化:"
    echo "   conda activate ${ENV_NAME}"
    echo ""
    echo "2. 事前学習済みモデルのダウンロード (未実施の場合):"
    echo "   wget -P models/pytorch/imagenet/ https://github.com/HRNet/HRNet-Image-Classification/releases/download/v1.0/hrnet_w32-36af842e.pth"
    echo ""
    echo "3. 学習の開始:"
    echo "   python tools/train.py \\"
    echo "     --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml \\"
    echo "     --transformer --occlusion_mask_strategy"
    echo ""
    echo "4. ベースライン学習 (Transformerなし):"
    echo "   python tools/train.py \\"
    echo "     --cfg experiments/coco/hrnet/sa_simdr/w32_256x192_adam_lr1e-3_split2_sigma4_210.yaml"
    echo ""
else
    echo ""
    echo "✗ インポートテストが失敗しました"
    echo "エラーメッセージを確認してください"
    exit 1
fi
