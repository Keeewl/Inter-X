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
