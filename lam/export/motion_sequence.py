import os
import json
import glob
import numpy as np
import torch


def export_motion_sequence(source, output_path, fps=30):
    """
    Export FLAME motion parameters as a single JSON file.

    Args:
        source: either a directory path (str) with per-frame .npz files,
                or a dict/list of tensors with FLAME params
        output_path: path for the output JSON file
        fps: frames per second (default 30)

    Returns:
        dict with metadata about the exported sequence

    Output JSON format:
    {
      "fps": 30,
      "frameCount": N,
      "frames": {
        "expr": [[100 floats], ...],
        "rotation": [[3 floats], ...],
        "jaw_pose": [[3 floats], ...],
        "neck_pose": [[3 floats], ...],
        "eyes_pose": [[6 floats], ...],
        "translation": [[3 floats], ...]
      }
    }
    """
    if isinstance(source, str):
        frames = _load_from_directory(source)
    elif isinstance(source, list):
        frames = _load_from_list(source)
    elif isinstance(source, dict):
        frames = _load_from_dict(source)
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    frame_count = _get_frame_count(frames)

    output = {
        "fps": fps,
        "frameCount": frame_count,
        "frames": frames,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)

    return {
        "output_path": output_path,
        "frame_count": frame_count,
        "fps": fps,
        "keys": list(frames.keys()),
    }


def _to_list(val):
    """Convert tensor/ndarray to nested Python list."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().float().numpy().tolist()
    elif isinstance(val, np.ndarray):
        return val.astype(float).tolist()
    elif isinstance(val, list):
        return val
    return val


def _load_from_directory(dir_path):
    """
    Load FLAME params from a directory of .npz files.
    Follows the same pattern as load_flame_params in head_utils.py.
    """
    npz_files = sorted(glob.glob(os.path.join(dir_path, "*.npz")))
    if not npz_files:
        # Try looking in a flame_param subdirectory
        npz_files = sorted(glob.glob(os.path.join(dir_path, "flame_param", "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {dir_path}")

    keys = ["expr", "rotation", "jaw_pose", "neck_pose", "eyes_pose", "translation"]
    frames = {k: [] for k in keys}

    for npz_path in npz_files:
        data = dict(np.load(npz_path, allow_pickle=True))
        for key in keys:
            if key in data:
                val = data[key]
                if isinstance(val, np.ndarray):
                    # .npz files from VHAP often have shape [1, N], squeeze batch dim
                    if val.ndim >= 2 and val.shape[0] == 1:
                        val = val[0]
                    frames[key].append(val.tolist())
                else:
                    frames[key].append(val)

    return frames


def _load_from_list(frame_list):
    """Load from a list of per-frame dicts (each dict has tensor/ndarray values)."""
    keys = ["expr", "rotation", "jaw_pose", "neck_pose", "eyes_pose", "translation"]
    frames = {k: [] for k in keys}

    for frame_data in frame_list:
        for key in keys:
            if key in frame_data:
                val = _to_list(frame_data[key])
                if isinstance(val[0], list):
                    # Already batched, take first
                    frames[key].append(val[0])
                else:
                    frames[key].append(val)

    return frames


def _load_from_dict(param_dict):
    """
    Load from a dict of tensors where each value has shape [N, ...].
    N is the number of frames.
    """
    keys = ["expr", "rotation", "jaw_pose", "neck_pose", "eyes_pose", "translation"]
    frames = {}

    for key in keys:
        if key in param_dict:
            val = _to_list(param_dict[key])
            if isinstance(val, list) and len(val) > 0:
                # If val is [N, D], it's already per-frame
                if isinstance(val[0], list):
                    frames[key] = val
                else:
                    # Single frame, wrap in list
                    frames[key] = [val]
            else:
                frames[key] = val

    return frames


def _get_frame_count(frames):
    """Get the number of frames from the frames dict."""
    for key in frames:
        if isinstance(frames[key], list) and len(frames[key]) > 0:
            return len(frames[key])
    return 0
