# Conda Environment
```bash
# Conda env
conda activate inter-x
```



# Model results visualization

当前 outputs 目录已有的 run：
- `cmdm_chi3d_offline_200K`
- `cmdm_interx_online_200K`
- `cnetv5_interx_online_200K`

Convert specific runs only (canonical)：
```bash
python outputs/convert_results_to_motions.py --runs cmdm_chi3d_offline_200K --overwrite
python outputs/convert_results_to_motions.py --runs cmdm_interx_online_200K --overwrite
python outputs/convert_results_to_motions.py --runs cnetv5_interx_online_200K --overwrite
```

Convert with restored body shape (Inter-X only, requires `results_meta.npz`)：
```bash
python outputs/convert_results_to_motions.py \
  --runs cmdm_interx_online_200K \
  --shape_mode restored \
  --raw_motions_root datasets/interx/motions \
  --interaction_order datasets/interx/annots/interaction_order.pkl \
  --overwrite
```

Convert with restored shape + raw height/global alignment (Inter-X only)：
```bash
python outputs/convert_results_to_motions.py \
  --runs cmdm_interx_handshake_online_200K \
  --shape_mode restored_shape_height \
  --raw_motions_root datasets/interx/motions \
  --interaction_order datasets/interx/annots/interaction_order.pkl \
  --overwrite
```

Visualizing model outputs:
```bash
# Visualizing cmdm chi3d outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/cmdm_chi3d_offline_200K/motions --texts_dir '' --dataset chi3d --title "cmdm-chi3d"

# Visualizing cmdm interx outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/cmdm_interx_online_200K/motions --texts_dir '' --dataset interx --title "cmdm-interx"

# Visualizing cnetv5 interx outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/cnetv5_interx_online_200K/motions --texts_dir '' --dataset interx --title "cnetv5-interx"

# Visualizing cnetv5 interx handshake outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/cnetv5_interx_handshake_online_200K/motions --texts_dir '' --dataset interx --title "cnetv5-interx-handshake"

# Visualizing cmdm interx handshake outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/cmdm_interx_handshake_online_200K/motions --texts_dir '' --dataset interx --title "cmdm-interx-handshake"

```

说明：
- 对于 Inter-X outputs，可视化面板会自动显示 `raw_index: xx/11388`。
- 该索引来自 `dataset_key` 在 `datasets/interx/motions/` 中按名称排序后的 1-based 位置。
- 如需手动指定原始 motions 根目录，可额外传 `--raw_index_root <path>`.
