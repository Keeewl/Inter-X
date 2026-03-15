import os
import numpy as np

# load the motion data
motion = np.load('interx/motions/G001T000A000R000/P1.npz')
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