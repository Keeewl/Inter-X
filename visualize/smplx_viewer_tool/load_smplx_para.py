import os
import numpy as np

# load the motion data
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
motion_path = os.path.join(repo_root, 'datasets', 'motions', 'G001T000A000R000', 'P1.npz')
motion = np.load(motion_path)
motion_parms = {
            'root_orient': motion['root_orient'],  # controls the global root orientation
            'pose_body': motion['pose_body'],  # controls the body
            'pose_lhand': motion['pose_lhand'],  # controls the left hand articulation
            'pose_rhand': motion['pose_rhand'],  # controls the right hand articulation
            'trans': motion['trans'],  # controls the global body position
            'betas': motion['betas'],  # controls the body shape
            'gender': motion['gender'],  # controls the gender
        }
print(motion_parms)