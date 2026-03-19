"""
ARKit delta extraction and serialization from FLAME expression shapedirs.

Extracts 52 ARKit position deltas from FLAME's upsampled expression shapedirs
using the [52, 100] ARKit-to-FLAME mapping matrix, and serializes them as a
WebGPU-ready fp16 binary with manifest.
"""

import json
import os

import torch

from lam.export.arkit_flame_mapping import (
    ARKIT_BLENDSHAPE_NAMES,
    load_mapping,
)


def extract_arkit_deltas(flame_model, mapping_path):
    """
    Extract 52 ARKit position deltas from FLAME's upsampled expression shapedirs.

    Uses the ARKit-to-FLAME mapping matrix M [52, 100] and the expression portion
    of shapedirs_up to compute per-Gaussian position deltas for each ARKit
    blendshape via a single einsum.

    Args:
        flame_model: FlameHeadSubdivided instance with shapedirs_up buffer
        mapping_path: path to ARKit-to-FLAME mapping JSON (52x100 matrix)

    Returns:
        arkit_deltas: torch.Tensor [52, V_up, 3] float32 canonical-space position deltas
        vertex_count: int — actual upsampled vertex count
    """
    # Slice expression portion of shapedirs_up: [V_up, 3, n_expr]
    # Must use explicit start:end to avoid teeth_bs dims at the end
    expr_start = flame_model.n_shape_params
    expr_end = expr_start + flame_model.n_expr_params
    shapedirs_expr = flame_model.shapedirs_up[:, :, expr_start:expr_end]

    # Load mapping matrix [52, 100] and move to same device
    M = load_mapping(mapping_path)
    M_tensor = torch.from_numpy(M).to(shapedirs_expr.device)

    # Compute all 52 ARKit deltas in one operation
    # M_tensor[a, l] * shapedirs_expr[v, k, l] -> arkit_deltas[a, v, k]
    arkit_deltas = torch.einsum("al,vkl->avk", M_tensor, shapedirs_expr)

    return arkit_deltas, flame_model.vertex_num_upsampled


def serialize_arkit_deltas(arkit_deltas, output_path):
    """
    Serialize [52, V, 3] deltas to fp16 Gaussian-major vec4 binary.

    Output layout: [V, 52, 4] fp16 little-endian.
    The 4th component (w) is zero padding for vec4 alignment in WebGPU.

    Args:
        arkit_deltas: torch.Tensor [52, V, 3] float32
        output_path: path to write binary file

    Returns:
        vertex_count: int
        channel_count: int (52)
    """
    # Transpose to Gaussian-major: [V, 52, 3]
    deltas_gmajor = arkit_deltas.permute(1, 0, 2).contiguous()

    V, N, _ = deltas_gmajor.shape

    # Pad xyz to vec4 with zero w: [V, 52, 4]
    pad = torch.zeros(V, N, 1, device=deltas_gmajor.device)
    deltas_vec4 = torch.cat([deltas_gmajor, pad], dim=2)

    # Convert to fp16 and write raw bytes
    deltas_fp16 = deltas_vec4.half().cpu().contiguous()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(deltas_fp16.numpy().tobytes())

    return V, N


def generate_manifest(vertex_count, output_path, arkit_names=None):
    """
    Generate au_manifest.json with ARKit channel info and bone bindings.

    Args:
        vertex_count: number of Gaussians (upsampled FLAME vertices)
        output_path: path to write JSON manifest
        arkit_names: optional list of 52 ARKit blendshape names
                     (defaults to ARKIT_BLENDSHAPE_NAMES)
    """
    if arkit_names is None:
        arkit_names = list(ARKIT_BLENDSHAPE_NAMES)

    manifest = {
        "version": 2,
        "format": "arkit",
        "channel_count": 52,
        "channel_names": arkit_names,
        "gaussian_count": vertex_count,
        "delta_types": ["position"],
        "precision": "fp16",
        "layout": "gaussian_major_vec4",
        "bone_bindings": {
            "jawOpen": {
                "bone": "jaw",
                "axis_angle": [0.35, 0.0, 0.0],
                "note": "max jaw rotation at blendshape weight 1.0",
            },
            "jawForward": {
                "bone": "jaw",
                "axis_angle": [0.0, 0.0, 0.05],
                "note": "slight forward translation approximated as rotation",
            },
            "jawLeft": {
                "bone": "jaw",
                "axis_angle": [0.0, 0.15, 0.0],
                "note": "jaw lateral rotation left at weight 1.0",
            },
            "jawRight": {
                "bone": "jaw",
                "axis_angle": [0.0, -0.15, 0.0],
                "note": "jaw lateral rotation right at weight 1.0",
            },
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
