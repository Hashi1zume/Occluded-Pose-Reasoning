#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NMSモジュールと依存関係のインポートテストスクリプト
"""

import sys
import os

# プロジェクトのlibディレクトリをパスに追加
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

def test_imports():
    """各モジュールのインポートをテストする"""

    print("=" * 60)
    print("インポートテスト開始")
    print("=" * 60)
    print()

    # 基本パッケージのテスト
    print("[1/8] 基本パッケージのテスト...")
    try:
        import numpy as np
        import torch
        import scipy
        print(f"  ✓ NumPy {np.__version__}")
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ SciPy {scipy.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        sys.exit(1)
    print()

    # その他の依存パッケージ
    print("[2/8] その他の依存パッケージのテスト...")
    try:
        import cv2
        import pandas
        import yaml
        import skimage
        print(f"  ✓ OpenCV {cv2.__version__}")
        print(f"  ✓ Pandas {pandas.__version__}")
        print(f"  ✓ PyYAML (インポート成功)")
        print(f"  ✓ scikit-image {skimage.__version__}")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        sys.exit(1)
    print()

    # NMSモジュールのテスト
    print("[3/8] NMSモジュールのテスト...")
    try:
        from nms.nms import oks_nms, oks_iou, soft_oks_nms
        print("  ✓ oks_nms インポート成功")
        print("  ✓ oks_iou インポート成功")
        print("  ✓ soft_oks_nms インポート成功")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # データセットモジュールのテスト
    print("[4/8] データセットモジュールのテスト...")
    try:
        import dataset
        from dataset.coco import COCODataset
        from dataset.mpii import MPIIDataset
        print("  ✓ dataset パッケージインポート成功")
        print("  ✓ COCODataset インポート成功")
        print("  ✓ MPIIDataset インポート成功")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # モデルモジュールのテスト
    print("[5/8] モデルモジュールのテスト...")
    try:
        import models
        from models import pose_hrnet
        from models import transformer
        print("  ✓ models パッケージインポート成功")
        print("  ✓ pose_hrnet インポート成功")
        print("  ✓ transformer インポート成功")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # コア機能のテスト
    print("[6/8] コア機能モジュールのテスト...")
    try:
        from core.loss import JointsMSELoss, KLDiscretLoss
        from core.function import train_sa_simdr, validate_sa_simdr
        from core.inference import get_final_preds
        print("  ✓ loss モジュールインポート成功")
        print("  ✓ function モジュールインポート成功")
        print("  ✓ inference モジュールインポート成功")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # 設定モジュールのテスト
    print("[7/8] 設定モジュールのテスト...")
    try:
        from config import cfg, update_config
        print("  ✓ config インポート成功")
        print(f"  ✓ デフォルトモデル名: {cfg.MODEL.NAME}")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        sys.exit(1)
    print()

    # ユーティリティのテスト
    print("[8/8] ユーティリティモジュールのテスト...")
    try:
        from utils.utils import get_optimizer, save_checkpoint, create_logger
        from utils.transforms import flip_back
        print("  ✓ utils.utils インポート成功")
        print("  ✓ utils.transforms インポート成功")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        sys.exit(1)
    print()

    # 成功メッセージ
    print("=" * 60)
    print("✓ すべてのインポートテストが成功しました！")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("1. COCOデータセットをダウンロード (data/coco/)")
    print("2. HRNet事前学習済みモデルをダウンロード (models/pytorch/imagenet/)")
    print("3. 学習を開始:")
    print("   python tools/train.py --cfg experiments/coco/hrnet/sa_simdr/tilsiter_w32_256x192_adam_lr1e-3_split2_sigma4_210_transformer_visibility_pseudo.yaml --transformer --occlusion_mask_strategy")
    print()

    return True


if __name__ == '__main__':
    try:
        success = test_imports()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nテストが中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
