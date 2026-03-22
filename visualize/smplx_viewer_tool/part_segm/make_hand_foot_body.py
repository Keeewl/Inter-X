import argparse
import os
import pickle

import numpy as np


def load_joint_map(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    joint2num = data["joint2num"]
    if isinstance(joint2num, np.ndarray):
        joint2num = joint2num.item()
    if not isinstance(joint2num, dict):
        raise ValueError("joint2num must be a dict in the SMPL-X model file")
    return data["weights"], joint2num


def build_part_indices(weights, joint2num):
    hand_tokens = ("Wrist", "Thumb", "Index", "Middle", "Ring", "Pinky", "Hand")
    foot_tokens = ("Ankle", "Foot", "Toes")

    hand_joints = {j for j in joint2num if any(t in j for t in hand_tokens)}
    foot_joints = {j for j in joint2num if any(t in j for t in foot_tokens)}

    hand_ids = [joint2num[j] for j in sorted(hand_joints)]
    foot_ids = [joint2num[j] for j in sorted(foot_joints)]

    if not hand_ids:
        raise ValueError("No hand joints found in joint2num")
    if not foot_ids:
        raise ValueError("No foot joints found in joint2num")

    vertex_to_joint = np.argmax(weights, axis=1)
    hand_verts = np.where(np.isin(vertex_to_joint, hand_ids))[0]
    foot_verts = np.where(np.isin(vertex_to_joint, foot_ids))[0]

    all_verts = np.arange(weights.shape[0])
    body_mask = ~(np.isin(all_verts, hand_verts) | np.isin(all_verts, foot_verts))
    body_verts = all_verts[body_mask]

    return {
        "hand": hand_verts.astype(np.int64).tolist(),
        "foot": foot_verts.astype(np.int64).tolist(),
        "body": body_verts.astype(np.int64).tolist(),
    }


def main():
    base_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Generate hand/foot/body parts segmentation for SMPL-X.")
    parser.add_argument(
        "--model",
        default=os.path.join(base_dir, "..", "body_models", "smplx", "SMPLX_NEUTRAL.npz"),
        help="Path to SMPL-X model .npz file",
    )
    parser.add_argument(
        "--out",
        default="hand_foot_body.pkl",
        help="Output .pkl path (part_name -> vertex indices)",
    )
    args = parser.parse_args()

    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.normpath(os.path.join(base_dir, model_path))
    weights, joint2num = load_joint_map(model_path)
    segm = build_part_indices(weights, joint2num)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(__file__), out_path)
    with open(out_path, "wb") as f:
        pickle.dump(segm, f)

    print(f"Saved parts segmentation to: {out_path}")


if __name__ == "__main__":
    main()
