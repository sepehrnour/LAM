# Project: LAM Animatable Gaussian Avatar Pipeline

## Core Value

Generate animatable Gaussian splat avatars from a single image that can be driven in real-time by the Omote/Muse SDK's FACS-based expression system.

## Problem Statement

LAM (Large Animatable Model) generates 20K Gaussian splats from a single portrait image, but its current export pipeline only produces static Gaussians with 5-bone skeletal skinning. This is insufficient for facial expression animation — smiles, brow raises, blinks, etc. cannot deform the Gaussians. The Omote character platform (Muse SDK) needs animatable Gaussian avatars driven by FACS Action Units, not just bone transforms.

## Stakeholders

- **Consumer**: Muse SDK / OmoteRuntime (TypeScript, client-side, WebGPU)
- **Producer**: LAM microservice (Python, CUDA, server-side one-shot generation)

## Requirements

### Must Have
- AU-based Gaussian deformation deltas exported alongside SPZ (per-Gaussian position offsets for each AU)
- GLB mesh avatar in the same export bundle (existing Blender pipeline, already works)
- Compatible with Muse's 14→20 AU system (AU1,2,4,5,6,7,9,10,12,15,20,23,25,26 + AU43,17,27,22,28,16)
- 16GB VRAM budget (RTX 5080) for generation; animation is client-side
- 32GB RAM budget for the Python process

### Should Have
- AU manifest describing which AUs are included, their indices, and names
- Compressed delta format (fp16, delta encoding) — target under 3MB for 20 AUs
- Gaussian-to-mesh vertex binding data (for future refinement)

### Won't Have (This Milestone)
- Multi-view input (separate milestone — architecture change + retraining)
- A2E model retraining to output AUs natively (separate effort in Muse repo)
- Client-side Muse integration (separate repo, separate milestone)

## Architecture

```
Single Image
    ↓
LAM Inference (infer_single_view)
    ↓
┌──────────────────────────────────────┐
│ GaussianModel (20K Gaussians)        │
│ + FLAME params (shape, expression)   │
│ + Camera matrices                    │
└──────────────────────────────────────┘
    ↓
Export Pipeline (lam/export/)
    ├── canonical.spz          (existing)
    ├── lbs_weights.json       (existing)
    ├── bone_tree.json         (existing)
    ├── flame_identity.json    (existing)
    ├── arkit_to_flame.json    (existing)
    ├── au_deltas.bin          (NEW — per-Gaussian AU position deltas)
    ├── au_manifest.json       (NEW — AU names, indices, count)
    ├── avatar.glb             (NEW in bundle — existing Blender pipeline)
    └── metadata.json          (existing, extended)
```

**AU Delta Generation Flow:**
```
For each AU (20 total):
  1. AU → FLAME expression params via [AU, 100] mapping matrix
  2. animation_forward(expression) → deformed FLAME mesh vertices
  3. Compute vertex deltas: deformed - rest_pose
  4. Transfer deltas to Gaussians via nearest-vertex binding
  5. Store as au_delta[au_idx] = [20K, 3] fp16
```

## Key Technical Decisions

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | FACS AUs as deformation basis (not ARKit) | AUs are atomic, composable, fewer basis shapes (20 vs 52), matches Muse's native expression system | 2026-03-19 |
| 2 | Post-hoc ARKit→AU projection for A2E lip sync (interim) | Zero ML work, ships now. Dual-head AU+ARKit A2E model is the long-term plan | 2026-03-19 |
| 3 | GLB + Gaussian in same export bundle | One inference pass produces both; Muse picks format by client capability | 2026-03-19 |
| 4 | Vertex-transfer for Gaussian deformation (not learned) | FLAME mesh deformation is well-understood; transferring via nearest-vertex is deterministic and debuggable | 2026-03-19 |

## Constraints

- **VRAM**: 16GB (RTX 5080 sm_120). Model inference + AU delta generation must fit.
- **XFORMERS_DISABLED=1**: No xformers for sm_120, must use native attention.
- **FLAME model**: 300 shape params, 100 expression params, 5-bone LBS.
- **Upstream dependency**: Fork of aigc3d/LAM (SIGGRAPH 2025). Must not break upstream merge path.
- **Docker**: CUDA 12.8 runtime, PyTorch 2.10.0+cu128, 4 CUDA extensions.

## Repository

- Fork: https://github.com/sepehrnour/LAM (private)
- Upstream: https://github.com/aigc3d/LAM
- Branch: `omote/main`
- Muse SDK: `C:\Users\Sepehr\Desktop\Dev\muse` (consumer, separate repo)
