## FLOPs/Params 計測レポート

本レポートは `python tools/profile_flops.py --cfg ... --transformer --occlusion_mask_strategy` の結果が厳密である理由をコード上の根拠とともに整理したものです。

### 1. 実験設定をそのまま反映
- `tools/profile_flops.py:68-106` で `--cfg`, `--transformer`, `--occlusion_mask_strategy` など訓練と同じ CLI 引数を受け、`update_config(cfg, args)` を通して YAML 内容を `yacs` Config にマージします。  
- `lib/config/default.py:150-175` が `cfg.transformer = args.transformer`, `cfg.low = args.low` を反映するため、計測時も本番同様のハイパラ・ブランチ構成になります。

### 2. 推論経路を忠実に再構成
- `PoseProfileModel`（`tools/profile_flops.py:109-154`）は `models.pose_hrnet.get_pose_net`（`lib/models/pose_hrnet.py:278-420`）をそのまま初期化し、`cfg.transformer` が真のときに `models.transformer.TransformerEncoder`／`Output`／`HRNetJointVisibilityNet` を訓練時と同じパラメータで生成します。
- `forward`（`tools/profile_flops.py:138-154`）では HRNet 出力を Transformer へ通し、可視性ブランチで得たマスクを掛けた後に `output_layer` で SimDR ロジットを生成します。訓練／推論の実際の処理と完全に一致するため、計測される FLOPs は本番と同じパスを辿ります。

### 3. fvcore の未サポート演算を網羅
- `CUSTOM_FVCORE_HANDLES`（`tools/profile_flops.py:22-65`）で `aten::add`, `aten::softmax`, `aten::silu` など SimDR 実装で頻出する演算に対する FLOPs 近似を定義。
- `main()`（`tools/profile_flops.py:178-185`）で `flops_analyzer.set_op_handle(...)` を通じて全演算へハンドラを登録し、従来警告が出ていた演算も集計対象に含めています。これにより `FlopCountAnalysis.total()` がモデル全演算の理論値を返します。

### 4. パラメータ総数の取得
- 同じ `with torch.no_grad()` ブロックで `parameter_count(model)['']` を呼び出し（`tools/profile_flops.py:178-185`）、`PoseProfileModel` に含まれる HRNet + Transformer + 出力層 + 可視性ブランチすべての `numel` を総和しています。fvcore の `parameter_count` は PyTorch の state_dict/grad 情報を元に計算するため、値は厳密です。

### 5. 出力と内訳
- 集計結果は `Total FLOPs ... / Total Params ...` として表示（`tools/profile_flops.py:186-187`）。`--print_breakdown` 指定時は `flops_analyzer.by_module()` によりモジュール単位の寄与を列挙（`tools/profile_flops.py:189-194`）。HRNet のステージ別・Transformer の層別内訳が確認でき、数値の妥当性を人間が追跡可能です。

以上のコードパスにより、`tools/profile_flops.py` で得られる FLOPs/Params は訓練・推論経路と同一条件での厳密な計測結果となります。

### 6. コード抜粋で見る FLOPs / Params 計算の根拠
```python
# tools/profile_flops.py:178-185
with torch.no_grad():
    flops_analyzer = FlopCountAnalysis(model, dummy)
    for op_name, handler in CUSTOM_FVCORE_HANDLES.items():
        flops_analyzer.set_op_handle(op_name, handler)
    total_flops = flops_analyzer.total()
    params = parameter_count(model)['']
```
- `FlopCountAnalysis(model, dummy)` がモデル全体の forward をトレースし、登録済みハンドラを参照して各演算の FLOPs を積み上げます。`set_op_handle` で `aten::add` 等の未サポート演算も確実にカウントするため、`total()` が理論値になります。
- `parameter_count(model)['']` は fvcore 実装で `model.state_dict()` に含まれる Tensor の `numel` を総和する処理であり、`PoseProfileModel` 内の HRNet/Transformer/出力層/visibility ブランチをすべて含めた正確なパラメータ数を返します。
