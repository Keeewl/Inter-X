# Conda Environment
```bash
# conda env
conda activate inter-x
```

# Inter-X Dataset
```bash
# create soft link (execute once)
cd visualize/smplx_viewer_tool
ln -s /Users/keweiou/Desktop/Project/Inter-X/datasets/interx/motions ./interx_data
ln -s /Users/keweiou/Desktop/Project/Inter-X/datasets/interx/texts ./interx_texts

# vasual inter-x datasets
python data_viewer.py --dataset interx

# load the SMPL-X motion parameters
python load_smplx_para.py
```

# Chi3d Dataset
```bash
# convert h5 file to motions
python datasets/chi3d/check/convert_chi3d_to_aitviewer.py \
  --h5_dir datasets/chi3d/smplx/conditioned \
  --out_dir datasets/chi3d/motions \
  --body_has_root

# create soft link (execute once)
cd visualize/smplx_viewer_tool
ln -s /Users/keweiou/Desktop/Project/Inter-X/datasets/chi3d/motions ./chi3d_data

# vasual chi3d datasets
python data_viewer.py --dataset chi3d
```