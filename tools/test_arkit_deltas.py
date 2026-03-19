#!/usr/bin/env python3
"""
Integration test for ARKit delta extraction from FLAME model.

Requires: GPU + FLAME model weights at model_zoo/human_parametric_models/
Run inside Docker container or conda env with PyTorch + FLAME model access:

    python tools/test_arkit_deltas.py

Optional args:
    --human-model-path  Path to human_parametric_models dir
                        (default: ./model_zoo/human_parametric_models)
    --output-dir        Directory for test output files (default: /tmp)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lam.export.arkit_deltas import (
    extract_arkit_deltas,
    generate_manifest,
    serialize_arkit_deltas,
)
from lam.export.arkit_flame_mapping import (
    ARKIT_BLENDSHAPE_NAMES,
    get_default_mapping_path,
)


def build_flame_model(human_model_path):
    """Load FlameHeadSubdivided the same way gs_renderer.py does."""
    from lam.models.rendering.flame_model.flame import FlameHeadSubdivided

    flame_model = FlameHeadSubdivided(
        300,
        100,
        add_teeth=True,
        add_shoulder=False,
        flame_model_path=os.path.join(
            human_model_path, "flame_assets/flame/flame2023.pkl"
        ),
        flame_lmk_embedding_path=os.path.join(
            human_model_path, "flame_assets/flame/landmark_embedding_with_eyes.npy"
        ),
        flame_template_mesh_path=os.path.join(
            human_model_path, "flame_assets/flame/head_template_mesh.obj"
        ),
        flame_parts_path=os.path.join(
            human_model_path, "flame_assets/flame/FLAME_masks.pkl"
        ),
        subdivide_num=1,
        teeth_bs_flag=False,
        oral_mesh_flag=False,
    )
    flame_model.to("cuda")
    flame_model.eval()
    return flame_model


def main():
    parser = argparse.ArgumentParser(description="Test ARKit delta extraction")
    parser.add_argument(
        "--human-model-path",
        default="./model_zoo/human_parametric_models",
        help="Path to human_parametric_models directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp",
        help="Directory for test output files",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ARKit Delta Extraction — Integration Test")
    print("=" * 70)

    # --- 1. Load FLAME model ---
    print("\n[1/7] Loading FlameHeadSubdivided ...")
    flame_model = build_flame_model(args.human_model_path)
    print(f"  vertex_num_upsampled = {flame_model.vertex_num_upsampled}")
    print(f"  n_shape_params       = {flame_model.n_shape_params}")
    print(f"  n_expr_params        = {flame_model.n_expr_params}")
    print(f"  teeth_bs_flag        = {flame_model.teeth_bs_flag}")

    # --- 2. Print shapedirs_up info ---
    print("\n[2/7] shapedirs_up diagnostics ...")
    shapedirs_up = flame_model.shapedirs_up
    print(f"  shapedirs_up.shape   = {list(shapedirs_up.shape)}")
    print(f"  shapedirs_up.dtype   = {shapedirs_up.dtype}")
    print(f"  shapedirs_up.device  = {shapedirs_up.device}")

    expr_start = flame_model.n_shape_params
    expr_end = expr_start + flame_model.n_expr_params
    shapedirs_expr = shapedirs_up[:, :, expr_start:expr_end]
    print(f"  expression portion   = [:, :, {expr_start}:{expr_end}]")
    print(f"  expression shape     = {list(shapedirs_expr.shape)}")

    # --- 3. Extract deltas ---
    print("\n[3/7] Extracting ARKit deltas ...")
    mapping_path = get_default_mapping_path()
    print(f"  mapping_path = {mapping_path}")

    with torch.no_grad():
        arkit_deltas, vertex_count = extract_arkit_deltas(flame_model, mapping_path)

    print(f"  arkit_deltas.shape   = {list(arkit_deltas.shape)}")
    print(f"  vertex_count         = {vertex_count}")

    expected_shape = [52, vertex_count, 3]
    assert list(arkit_deltas.shape) == expected_shape, (
        f"Shape mismatch: {list(arkit_deltas.shape)} != {expected_shape}"
    )
    print("  Shape check PASSED")

    # --- 4. Per-channel statistics ---
    print("\n[4/7] Per-channel delta statistics:")
    print(f"  {'Channel':<25s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s} {'L2max':>10s}")
    print("  " + "-" * 75)

    channel_magnitudes = []
    for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES):
        ch = arkit_deltas[i]  # [V, 3]
        ch_min = ch.min().item()
        ch_max = ch.max().item()
        ch_mean = ch.mean().item()
        ch_std = ch.std().item()
        l2_max = ch.norm(dim=1).max().item()
        channel_magnitudes.append((name, l2_max))
        print(f"  {name:<25s} {ch_min:>10.6f} {ch_max:>10.6f} {ch_mean:>10.6f} {ch_std:>10.6f} {l2_max:>10.6f}")

    # Sort by magnitude
    channel_magnitudes.sort(key=lambda x: x[1], reverse=True)
    print("\n  Top 10 channels by max L2 displacement:")
    for i, (name, mag) in enumerate(channel_magnitudes[:10]):
        print(f"    {i+1:2d}. {name:<25s} {mag:.6f}")

    near_zero = [(n, m) for n, m in channel_magnitudes if m < 1e-6]
    if near_zero:
        print(f"\n  WARNING: {len(near_zero)} channels with near-zero deltas:")
        for name, mag in near_zero:
            print(f"    - {name} (L2max={mag:.2e})")
    else:
        print("\n  All channels have nonzero deltas")

    # --- 5. Serialize binary ---
    print("\n[5/7] Serializing to fp16 binary ...")
    bin_path = os.path.join(args.output_dir, "test_arkit_deltas.bin")
    v_count, ch_count = serialize_arkit_deltas(arkit_deltas, bin_path)
    print(f"  Output: {bin_path}")
    print(f"  vertex_count  = {v_count}")
    print(f"  channel_count = {ch_count}")

    expected_size = v_count * 52 * 4 * 2  # V * channels * vec4 * fp16
    actual_size = os.path.getsize(bin_path)
    print(f"  Expected size = {expected_size} bytes ({expected_size / 1024 / 1024:.2f} MB)")
    print(f"  Actual size   = {actual_size} bytes ({actual_size / 1024 / 1024:.2f} MB)")
    assert actual_size == expected_size, (
        f"File size mismatch: {actual_size} != {expected_size}"
    )
    print("  File size check PASSED")

    # --- 6. Generate manifest ---
    print("\n[6/7] Generating manifest ...")
    manifest_path = os.path.join(args.output_dir, "test_au_manifest.json")
    generate_manifest(vertex_count, manifest_path)
    print(f"  Output: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    print(f"  version        = {manifest['version']}")
    print(f"  channel_count  = {manifest['channel_count']}")
    print(f"  gaussian_count = {manifest['gaussian_count']}")
    print(f"  precision      = {manifest['precision']}")
    print(f"  layout         = {manifest['layout']}")
    print(f"  bone_bindings  = {list(manifest['bone_bindings'].keys())}")
    assert manifest["gaussian_count"] == vertex_count
    assert manifest["channel_count"] == 52
    assert len(manifest["channel_names"]) == 52
    print("  Manifest check PASSED")

    # --- 7. Round-trip verification ---
    print("\n[7/7] Round-trip binary verification ...")
    with open(bin_path, "rb") as f:
        raw = f.read()
    loaded = np.frombuffer(raw, dtype=np.float16).reshape(v_count, 52, 4)
    loaded_tensor = torch.from_numpy(loaded.copy()).float()

    # Compare xyz components (ignore w padding)
    loaded_xyz = loaded_tensor[:, :, :3]  # [V, 52, 3]
    original_gmajor = arkit_deltas.permute(1, 0, 2).cpu().float()  # [V, 52, 3]

    # fp16 introduces quantization error — check relative tolerance
    abs_diff = (loaded_xyz - original_gmajor).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    print(f"  Max abs diff (fp16 quantization) = {max_abs_diff:.2e}")
    print(f"  Mean abs diff                    = {mean_abs_diff:.2e}")

    # w padding should be exactly zero
    w_values = loaded_tensor[:, :, 3]
    assert (w_values == 0).all(), "w padding contains nonzero values!"
    print("  w-padding zero check PASSED")

    # fp16 relative error should be < 0.1% for values in typical range
    assert max_abs_diff < 0.01, f"fp16 round-trip error too large: {max_abs_diff}"
    print("  Round-trip check PASSED")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
