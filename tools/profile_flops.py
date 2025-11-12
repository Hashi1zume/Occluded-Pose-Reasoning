#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Occluded-Pose-Reasoning 向けに FLOPs/パラメータ計測用途へ追加
# ------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import pprint
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count

import _init_paths  # noqa: F401
from config import cfg, update_config
import models


def _get_tensor_numel(arg: Union[torch.Tensor, Tuple, list]) -> int:
    """fvcore のカスタムハンドラ内で Tensor サイズを安全に取得するヘルパ."""
    if isinstance(arg, torch.Tensor):
        return int(arg.numel())
    if isinstance(arg, (list, tuple)) and arg:
        return _get_tensor_numel(arg[0])
    return 0


def _elemwise_flop(_inputs, outputs) -> int:
    """add/mul/div など要素単位演算の FLOPs を出力要素数ベースで概算."""
    if not outputs:
        return 0
    return _get_tensor_numel(outputs[0])


def _softmax_flop(_inputs, outputs) -> int:
    """softmax は exp + sum + div を含むため約 5 FLOPs/要素とみなす."""
    if not outputs:
        return 0
    return 5 * _get_tensor_numel(outputs[0])


def _silu_flop(_inputs, outputs) -> int:
    """SiLU は sigmoid + mul 相当なので 4 FLOPs/要素で近似."""
    if not outputs:
        return 0
    return 4 * _get_tensor_numel(outputs[0])


def _zero_flop(*_args, **_kwargs) -> int:
    """メタ演算 (lift_fresh など) は FLOPs 0 とみなす."""
    return 0


CUSTOM_FVCORE_HANDLES: dict[str, Callable[..., int]] = {
    "aten::add": _elemwise_flop,
    "aten::add_": _elemwise_flop,
    "aten::mul": _elemwise_flop,
    "aten::div": _elemwise_flop,
    "aten::softmax": _softmax_flop,
    "aten::silu": _silu_flop,
    "aten::lift_fresh": _zero_flop,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='姿勢推定モデルの FLOPs/Params を計測する補助スクリプト')
    parser.add_argument(
        '--cfg',
        help='実験設定 YAML へのパス',
        required=True,
        type=str,
    )
    parser.add_argument(
        'opts',
        help='cfg の任意パラメータを上書きしたい場合に追加',
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--modelDir', help='モデル出力ディレクトリ', type=str, default='')
    parser.add_argument('--logDir', help='ログ出力ディレクトリ', type=str, default='')
    parser.add_argument('--dataDir', help='データルート', type=str, default='')
    parser.add_argument('--prevModelDir', help='旧モデルの配置先', type=str, default='')
    parser.add_argument(
        '--transformer',
        help='Transformer ブランチを有効化',
        action='store_true',
    )
    parser.add_argument(
        '--occlusion_mask_strategy',
        help='Transformer 内部で Occlusion Mask 戦略を使用',
        action='store_true',
    )
    parser.add_argument(
        '--low',
        help='低解像度ヘッドを使用 (学習時引数と揃える)',
        action='store_true',
    )
    parser.add_argument(
        '--print_breakdown',
        help='モジュール単位の FLOPs 内訳を表示',
        action='store_true',
    )
    return parser.parse_args()


class PoseProfileModel(nn.Module):
    def __init__(self, cfg, use_transformer: bool, occlusion_mask_strategy: bool):
        super().__init__()
        self.cfg = cfg
        self.use_transformer = use_transformer
        self.occlusion_mask_strategy = occlusion_mask_strategy
        self.pose_net = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
        self.transformer: Union[nn.Module, None] = None
        self.output_layer: Union[nn.Module, None] = None
        self.visibility_branch: Union[nn.Module, None] = None

        if self.use_transformer:
            vocab_size = 48 if cfg.low else cfg.MODEL.HEAD_INPUT
            # TransformerEncoder 生成時のハイパラは train.py と揃えておかないと
            # 実際の forward と FLOPs が乖離してしまう
            self.transformer = eval('models.transformer.TransformerEncoder')(
                cfg,
                seq_len=cfg.MODEL.NUM_JOINTS,
                vocab_size=vocab_size,
                embed_dim=512,
                output_dim=1,
                num_layers=3,
                pe=False,
                n_heads=2,
                expansion_factor=2,
            )
            self.output_layer = eval('models.transformer.Output')(cfg)
            self.visibility_branch = eval('models.transformer.HRNetJointVisibilityNet')()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        features = self.pose_net(x)

        if not self.use_transformer:
            return features

        assert isinstance(features, torch.Tensor), 'Transformer 併用時はテンソル出力が前提'
        # visibility_branch で視認性を推定し Transformer のマスクに流すことで
        # 学習時と同じ演算を再現し FLOPs を正しく数える
        visibility_state = torch.ones(features.size(0), features.size(1), device=features.device)
        if self.visibility_branch is not None:
            pred_visibility = self.visibility_branch(features.detach())
            visibility_state = (pred_visibility >= 0.5).float()

        assert self.transformer is not None and self.output_layer is not None
        transformed = self.transformer(features, visibility_state, self.occlusion_mask_strategy)
        return self.output_layer(transformed)


def main():
    args = parse_args()
    update_config(cfg, args)
    print('Effective config:\n{}'.format(pprint.pformat(cfg)))

    if cfg.transformer and not torch.cuda.is_available():
        raise RuntimeError('Transformer ブランチは GPU 実行を前提としているため CUDA 環境で実行してください')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseProfileModel(cfg, use_transformer=cfg.transformer, occlusion_mask_strategy=args.occlusion_mask_strategy)
    model = model.to(device)
    model.eval()

    dummy = torch.randn(
        1,
        3,
        cfg.MODEL.IMAGE_SIZE[1],
        cfg.MODEL.IMAGE_SIZE[0],
        device=device,
    )

    with torch.no_grad():
        flops_analyzer = FlopCountAnalysis(model, dummy)
        # fvcore が未サポートの素朴演算を登録し、警告なしで厳密なカウントを得る
        for op_name, handler in CUSTOM_FVCORE_HANDLES.items():
            flops_analyzer.set_op_handle(op_name, handler)
        total_flops = flops_analyzer.total()
        params = parameter_count(model)['']

    print('Total FLOPs: {:.3f} G'.format(total_flops / 1e9))
    print('Total Params: {:.3f} M'.format(params / 1e6))

    if args.print_breakdown:
        print('--- FLOPs by module (G) ---')
        module_flops = flops_analyzer.by_module()
        for name, value in sorted(module_flops.items(), key=lambda item: item[1], reverse=True):
            readable_name = name or 'model'
            print(f'{readable_name}: {value / 1e9:.3f} G')


if __name__ == '__main__':
    main()
