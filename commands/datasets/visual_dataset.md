# Conda Environment
```bash
# Conda env
conda activate inter-x
```



# Inter-X Dataset
```bash
# Create soft link (execute once)
cd visualize/smplx_viewer_tool
ln -s /Users/keweiou/Desktop/Project/Inter-X/datasets/interx/motions ./interx_data
ln -s /Users/keweiou/Desktop/Project/Inter-X/datasets/interx/texts ./interx_texts

# Visualizing inter-x datasets
cd visualize/smplx_viewer_tool
python data_viewer.py --dataset interx \
  --interaction_order ../../datasets/interx/annots/interaction_order.pkl

# Load the SMPL-X motion parameters
cd datasets
python load_smplx_para.py
```

说明：
- viewer 会按 `datasets/interx/motions/` 的排序显示页码 `xx/11388`。
- 对于模型 outputs，只要 metadata 中带有 `dataset_key`，viewer 也会自动显示对应的 `raw_index: xx/11388`。



# Inter-X Reaction Generation Preprocess
```bash
# 1. Build inter-x.h5 from raw motions
cd preprocess
python 1_prepare_data.py

# 2. Align actor->reactor order into inter-x_regen.h5
python 4_reaction_generation.py

# 3. Split into train/val/test for reaction generation
python 2_split_train_val.py
```



# Chi3d Dataset
```bash
# Convert h5 file to motions and fix the problem of body flip
python datasets/chi3d/check/convert_chi3d_to_aitviewer.py \
  --h5_dir datasets/chi3d/smplx/conditioned \
  --out_dir datasets/chi3d/motions \
  --body_has_root \
  --flip_x

# Create soft link (execute once)
cd visualize/smplx_viewer_tool
ln -s /Users/keweiou/Desktop/Project/Inter-X/datasets/chi3d/motions ./chi3d_data

# Visualizing chi3d datasets
cd visualize/smplx_viewer_tool
python data_viewer.py --dataset chi3d
```
