# Conda Environment
```bash
# Conda env
conda activate inter-x
```



# Part segmentation

```bash
# Visual part segmentation
cd visualize/smplx_viewer_tool
python data_viewer.py \
  --dataset interx \
  --part_segm part_segm/hand_foot_body.pkl \
  --part_colors part_segm/hand_foot_body_colors.json
```