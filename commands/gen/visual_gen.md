# Conda Environment
```bash
# Conda env
conda activate inter-x
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
python outputs/convert_results_to_motions.py --runs rnetv1_1K_interx_exp1 --overwrite
```

Visualizing model outputs:
```bash
# Visualizing cmdm outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/cmdm_chi3d_offline_200K/motions --texts_dir '' --dataset chi3d --title "cmdm-chi3d"

# Visualizing cnet_v3 outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/cnet_v3_chi3d_train_200K/motions --texts_dir '' --dataset chi3d --title "cnet_v3-chi3d"

# Visualizing cnet_v5 outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/cnetv5_interx_online_exp1_200K/motions --texts_dir '' --dataset interx --title "cnet_v5-interx"

# Visualizing rnet_v1 outputs
cd visualize/smplx_viewer_tool
python data_viewer.py --data_dir ../../outputs/rnetv1_1K_interx_exp1/motions --texts_dir '' --dataset interx --title "rnet_v1-interx"
```
