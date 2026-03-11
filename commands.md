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
python data_viewer.py --dataset interx

# Load the SMPL-X motion parameters
python load_smplx_para.py
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



# Model results visualization

Convert all ReGenNet outputs under `outputs/`:
```bash
# Convert results to motions for all runs that contain results.npy
python outputs/convert_results_to_motions.py --outputs_root outputs --overwrite
```

Convert specific runs only:
```bash
# Convert only selected runs
python outputs/convert_results_to_motions.py --runs cnet_v3_chi3d_train_200K cmdm_chi3d_offline_200K --overwrite
```

Visualizing model outputs:
```bash
# Visualizing cmdm outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/cmdm_chi3d_offline_200K/motions --texts_dir '' --dataset chi3d --title "cmdm-chi3d"

# Visualizing cnet_v3 outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/cnet_v3_chi3d_train_200K/motions --texts_dir '' --dataset chi3d --title "cnet_v3-chi3d"
```
