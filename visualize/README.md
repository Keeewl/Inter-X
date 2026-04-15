# 可视化结构与功能

本仓库的可视化可以按「用途 + 数据来源 + 渲染增强」分成三层，便于理解与维护。

## 1) 结构概览

- **GT/原始数据可视化**
  - 目标：看原始 SMPL-X 参数、真实高度/相对位置关系。
  - 数据：`datasets/interx/motions/*/P1.npz`、`P2.npz`。
  - 入口：`visualize/smplx_viewer_tool/data_viewer.py`。
- **模型输出可视化（含转换）**
  - 目标：可视化 ReGenNet 结果，支持 `canonical / restored / restored_shape_height`。
  - 数据：`outputs/<run>/results.npy` + `results_meta.npz`。
  - 入口：`outputs/convert_results_to_motions.py` + `visualize/smplx_viewer_tool/data_viewer.py`。
- **可视化增强层（可选叠加）**
  - 目标：部位分割上色、关节级调试、颜色映射等。
  - 入口：`visualize/smplx_viewer_tool/part_segm/`、`visualize/joint_viewer_tool/`。

## 2) 目录与入口

- `visualize/smplx_viewer_tool/`
  - 主视图：`data_viewer.py`
  - 分部位可视化：`part_segm/`（pkl + 颜色 json）
- `visualize/joint_viewer_tool/`
  - 关节/调试可视化（如果需要）
- `outputs/convert_results_to_motions.py`
  - ReGenNet 输出转换入口（含三种模式）
- `commands/datasets/visual_dataset.md`
  - 原始数据可视化示例
- `commands/gen/visual_gen.md`
  - 模型输出可视化示例

## 3) 工作流

### 3.1 原始 GT 可视化（Inter-X）
```bash
cd visualize/smplx_viewer_tool
python data_viewer.py --dataset interx --interaction_order ../../datasets/interx/annots/interaction_order.pkl
```
- 会读取 `datasets/interx/motions` 原始 `P1.npz / P2.npz`。
- `--interaction_order` 用于按 actor/reactor 上色（actor=蓝，reactor=红）。

### 3.2 模型输出可视化（ReGenNet → Inter-X）
1) 转换输出为 `P1.npz / P2.npz`
```bash
python outputs/convert_results_to_motions.py \
  --outputs_root outputs \
  --shape_mode restored_shape_height \
  --raw_motions_root datasets/interx/motions \
  --interaction_order datasets/interx/annots/interaction_order.pkl \
  --overwrite
```
2) 可视化
```bash
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/<run>/motions --texts_dir '' --dataset interx
```

模式说明：
- `canonical`：中性体型，便于调试动作模式。
- `restored`：恢复原始 betas/gender，保留预测 transl。
- `restored_shape_height`：恢复原始 betas/gender + raw transl，对齐高度/全局位置。

### 3.3 可视化增强（分部位上色）
```bash
cd visualize/smplx_viewer_tool
python data_viewer.py \
  --dataset interx \
  --data_dir ./interx_data \
  --texts_dir ./interx_texts \
  --part_segm part_segm/6_parts/part_segm.pkl \
  --part_colors part_segm/6_parts/colors.json
```

## 4) 说明与注意事项

- `restored` / `restored_shape_height` 需要 `results_meta.npz`。
- `restored_shape_height` 依赖 `frame_ix + downsample` 对齐 raw transl。
- 文本窗口会显示 metadata（如 `dataset_key / frame range / actor mapping`），用于对齐分析。
- 颜色映射默认按 actor/reactor（若无 `source_role`，会用 `interaction_order.pkl` 推断）。
