# Phase 1: AU Deformation Basis + ARKit Delta Export - Research

**Researched:** 2026-03-19
**Domain:** FLAME parametric model expression extraction, Gaussian splat deformation, binary serialization
**Confidence:** HIGH

<research_summary>
## Summary

Researched the complete data pipeline from FLAME's upsampled expression shapedirs to serialized ARKit position deltas for WebGPU consumption. The critical discovery: **all 52 ARKit blendshapes are already mapped** in the existing `arkit_to_flame.json` matrix — no new AU-to-FLAME mapping work is needed. The pipeline is a pure linear algebra operation: matrix multiply the [52, 100] ARKit→FLAME mapping against `shapedirs_up[:, :, n_shape:]` to produce [52, V, 3] position deltas.

The implementation avoids `animation_forward` entirely (canonical-space deltas compose linearly). Jaw rotation is handled separately via bone_bindings metadata. Teeth vertices are included in `shapedirs_up` when `teeth_bs_flag=True` — must verify whether the model uses this flag.

**Primary recommendation:** Single `torch.einsum` to compute all 52 ARKit deltas from `shapedirs_up` expression portion + the transposed mapping matrix. Serialize as fp16 Gaussian-major binary. Total new code: ~150 lines Python.
</research_summary>

<standard_stack>
## Standard Stack

### Core (already in codebase)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0+cu128 | Tensor ops, `shapedirs_up` access | Already loaded for inference |
| NumPy | (bundled) | Matrix analysis, serialization | Already available |
| json | stdlib | Manifest generation | Zero dependency |

### Supporting (already in codebase)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| struct | stdlib | fp16 binary packing | Serialization to raw binary |
| torch.half() | PyTorch | fp16 conversion | Convert float32 deltas to fp16 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Raw fp16 binary | NumPy .npy | .npy has 128-byte header, client needs parser — raw binary loads directly into WebGPU buffer |
| torch.einsum | Manual loops | einsum is 10-100x faster, single GPU kernel launch |
| JSON manifest | Protobuf | Protobuf adds build dependency for a 20-line config — JSON is fine |

**No new dependencies needed.** Everything required is already loaded in the inference pipeline.
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Data Flow
```
flame_model.shapedirs_up            [V_up, 3, n_shape + n_expr (+ teeth_bs)]
    ↓ slice expression portion
shapedirs_expr                      [V_up, 3, n_expr]  (n_expr=100)
    ↓ einsum with mapping matrix M^T
arkit_deltas                        [52, V_up, 3]
    ↓ convert to fp16, pad to vec4, transpose to Gaussian-major
arkit_deltas_gpu_layout             [V_up, 52, 4] fp16
    ↓ serialize
arkit_deltas.bin                    V_up * 52 * 4 * 2 bytes = ~8.3MB
```

### Pattern 1: Direct shapedirs extraction (no animation_forward)
**What:** Extract expression deltas from the registered buffer without calling the forward pass
**When to use:** Always — this is the correct approach for canonical-space deltas
**Why:** `shapedirs_up` is already on GPU as a registered buffer. The expression portion starts at index `n_shape_params` along dim 2.

```python
# Access expression shapedirs (already on GPU)
shapedirs_expr = flame_model.shapedirs_up[:, :, flame_model.n_shape_params:]
# Shape: [V_up, 3, n_expr]  where n_expr = 100

# Load ARKit→FLAME mapping [52, 100]
M = load_mapping("assets/default_arkit_to_flame.json")  # [52, 100]
M_tensor = torch.from_numpy(M).to(shapedirs_expr.device)

# Compute all 52 ARKit deltas in one operation
# M_tensor: [52, 100], shapedirs_expr: [V, 3, 100]
# Result: [52, V, 3]
arkit_deltas = torch.einsum("al,vkl->avk", M_tensor, shapedirs_expr)
```

### Pattern 2: Gaussian-major vec4 serialization for WebGPU
**What:** Transpose and pad for cache-coherent GPU reads
**When to use:** For the binary export file

```python
# Transpose to Gaussian-major: [V, 52, 3]
deltas_gmajor = arkit_deltas.permute(1, 0, 2)

# Pad xyz to vec4 (16-byte alignment for WebGPU)
pad = torch.zeros(deltas_gmajor.shape[0], deltas_gmajor.shape[1], 1,
                  device=deltas_gmajor.device)
deltas_vec4 = torch.cat([deltas_gmajor, pad], dim=2)  # [V, 52, 4]

# Convert to fp16 and serialize
deltas_fp16 = deltas_vec4.half().cpu().contiguous()
raw_bytes = deltas_fp16.numpy().tobytes()  # little-endian by default on x86
```

### Anti-Patterns to Avoid
- **Calling animation_forward for each blendshape:** 52 forward passes when a single einsum does it. Also introduces nonlinear bone-rotation artifacts.
- **Extracting from full shapedirs (including shape dims):** The first `n_shape_params` dims are identity shape, not expression. Only slice `[:, :, n_shape_params:]`.
- **Forgetting teeth_bs dimensions:** If `teeth_bs_flag=True`, `shapedirs_up` has 4 extra dims at the end (shape: [V, 3, 300+100+4]). The expression portion is `[:, :, 300:400]`, NOT `[:, :, 300:]`.
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ARKit→FLAME mapping | New AU→FLAME matrix from scratch | Existing `arkit_to_flame.json` [52, 100] | Already hand-tuned and validated, covers all 52 shapes |
| Expression blendshape math | Manual per-vertex loops | `torch.einsum("al,vkl->avk", ...)` | Single GPU kernel, 10-100x faster |
| Mesh subdivision | Custom vertex interpolation | `flame_model.shapedirs_up` (pre-computed) | Subdivision already done at model init, includes all buffers |
| fp16 conversion | Manual bit-packing | `tensor.half().numpy().tobytes()` | PyTorch handles IEEE 754 fp16 correctly |
| Vertex count discovery | Hardcoding 20000 | `flame_model.vertex_num_upsampled` | Actual count depends on subdivision level and teeth |

**Key insight:** The entire delta computation is a single matrix multiply. The existing `arkit_flame_mapping.py` + `shapedirs_up` buffer contain everything needed. New code is pure glue — read tensor, multiply, serialize.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: shapedirs dimension ordering
**What goes wrong:** Indexing expression dims incorrectly when teeth_bs is enabled
**Why it happens:** `shapedirs_up` shape is `[V, 3, n_shape + n_expr + n_teeth_bs]`. With teeth_bs_flag=True, the last 4 dims are teeth blendshapes, not expression.
**How to avoid:** Always use `n_shape_params` and `n_expr_params` from the model, never hardcode:
```python
expr_start = flame_model.n_shape_params
expr_end = expr_start + flame_model.n_expr_params  # NOT shapedirs.shape[2]
shapedirs_expr = flame_model.shapedirs_up[:, :, expr_start:expr_end]
```
**Warning signs:** Delta magnitudes are wrong (too small or too large), teeth moving independently

### Pitfall 2: Mapping matrix direction
**What goes wrong:** Using M as FLAME→ARKit instead of ARKit→FLAME
**Why it happens:** The matrix `M` is [52, 100] meaning `flame_expr = arkit_weights @ M`. For our use case we need `delta_i = M[i, :] @ shapedirs_expr` (the i-th ARKit shape's expression weights applied to the expression basis).
**How to avoid:** Read the docstring in `arkit_flame_mapping.py` line 68: "flame_expr = arkit_weights @ M"
**Warning signs:** All deltas look similar or nonsensical

### Pitfall 3: Vertex count mismatch
**What goes wrong:** Assuming 20K vertices, getting a different count
**Why it happens:** Actual vertex count depends on: base FLAME vertices (~5023) × subdivision factor (2 levels) + teeth vertices + optional oral mesh vertices. The subdivision produces ~20K but the exact count varies.
**How to avoid:** Always read `flame_model.vertex_num_upsampled` and include it in the manifest.
**Warning signs:** Buffer size doesn't match expected, WebGPU validation errors

### Pitfall 4: FLAME expression dims are entangled
**What goes wrong:** Expecting each FLAME dim to correspond to one facial movement
**Why it happens:** FLAME expression is PCA — each dim is a blend of movements. The ARKit mapping matrix handles this by using weighted combinations.
**How to avoid:** Trust the mapping matrix. Don't try to "improve" it by zeroing out small weights or making it sparser — those small weights provide important corrections.
**Warning signs:** Expressions look subtly wrong (asymmetric when they shouldn't be, missing secondary movements)

### Pitfall 5: jawOpen expression delta is insufficient alone
**What goes wrong:** Jaw appears to open but chin doesn't rotate properly
**Why it happens:** FLAME's jawOpen expression (dim 15) captures soft tissue deformation but NOT the rigid jaw bone rotation. The bone rotation is a separate pose parameter in `animation_forward`.
**How to avoid:** Export `bone_bindings` in the manifest. The jawOpen ARKit delta handles soft tissue; the client also applies jaw bone rotation via LBS.
**Warning signs:** Jaw opens but the lower lip/chin don't move down far enough
</common_pitfalls>

<code_examples>
## Code Examples

### Complete delta extraction function
```python
# Source: Derived from flame.py animation_forward + arkit_flame_mapping.py
import torch
import numpy as np
import json
from lam.export.arkit_flame_mapping import load_mapping, ARKIT_BLENDSHAPE_NAMES

def extract_arkit_deltas(flame_model, mapping_path):
    """
    Extract 52 ARKit position deltas from FLAME's upsampled expression shapedirs.

    Returns:
        deltas: torch.Tensor [52, V_up, 3] float32 — canonical-space position deltas
        vertex_count: int — actual upsampled vertex count
    """
    # 1. Get expression portion of shapedirs_up
    expr_start = flame_model.n_shape_params
    expr_end = expr_start + flame_model.n_expr_params
    shapedirs_expr = flame_model.shapedirs_up[:, :, expr_start:expr_end]
    # Shape: [V_up, 3, 100]

    # 2. Load mapping matrix [52, 100]
    M = load_mapping(mapping_path)
    M_tensor = torch.from_numpy(M).to(shapedirs_expr.device)

    # 3. Compute all 52 ARKit deltas
    # einsum: M[a,l] * shapedirs[v,k,l] -> deltas[a,v,k]
    arkit_deltas = torch.einsum("al,vkl->avk", M_tensor, shapedirs_expr)

    return arkit_deltas, flame_model.vertex_num_upsampled
```

### Serialization to WebGPU-ready binary
```python
def serialize_arkit_deltas(arkit_deltas, output_path):
    """
    Serialize [52, V, 3] deltas to fp16 Gaussian-major vec4 binary.
    Layout: [V, 52, 4] fp16 little-endian — w component is zero padding.
    """
    # Transpose to Gaussian-major: [V, 52, 3]
    deltas_gmajor = arkit_deltas.permute(1, 0, 2).contiguous()

    # Pad to vec4: [V, 52, 4]
    V, N, _ = deltas_gmajor.shape
    pad = torch.zeros(V, N, 1, device=deltas_gmajor.device)
    deltas_vec4 = torch.cat([deltas_gmajor, pad], dim=2)

    # Convert to fp16
    deltas_fp16 = deltas_vec4.half().cpu().contiguous()

    # Write raw binary
    with open(output_path, "wb") as f:
        f.write(deltas_fp16.numpy().tobytes())

    return V, N
```

### Manifest generation
```python
def generate_manifest(vertex_count, output_path):
    """Generate au_manifest.json with ARKit channel info and bone bindings."""
    manifest = {
        "version": 2,
        "format": "arkit",
        "channel_count": 52,
        "channel_names": ARKIT_BLENDSHAPE_NAMES,
        "gaussian_count": vertex_count,
        "delta_types": ["position"],
        "precision": "fp16",
        "layout": "gaussian_major_vec4",
        "bone_bindings": {
            "jawOpen": {
                "bone": "jaw",
                "axis_angle": [0.35, 0.0, 0.0],
                "note": "max jaw rotation at blendshape weight 1.0"
            },
            "jawForward": {
                "bone": "jaw",
                "axis_angle": [0.0, 0.0, 0.05],
                "note": "slight forward translation approximated as rotation"
            }
        }
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
```
</code_examples>

<sota_updates>
## State of the Art (2025-2026)

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-vertex nearest-mesh binding for Gaussian deformation | Direct shapedirs_up (Gaussians ARE mesh vertices) | LAM architecture | Eliminates binding step entirely |
| animation_forward per blendshape | Single einsum on shapedirs_up | Design review finding | 52x faster, linear composition |
| AU-major buffer layout [N_AU, V, 3] | Gaussian-major [V, N_AU, 4] with vec4 padding | WebGPU cache analysis | Cache-coherent reads per workgroup |
| float32 deltas | fp16 sufficient | Facial movements are 0.001-0.05 range | 50% size reduction, no visible quality loss |

**Key finding from codebase analysis:**
- All 52 ARKit blendshapes are mapped to FLAME expression dims 1-46 (dim 0 unused, dims 47-99 unused)
- The mapping is sparse: 63 nonzero entries out of 5200 (1.2%)
- Some FLAME dims are shared with opposing signs (e.g., dim 5: eyeBlinkLeft +0.9 / eyeWideLeft -0.6)
- Mapping matrix `M` direction: `flame_expr[100] = arkit_weights[52] @ M[52, 100]`
</sota_updates>

<open_questions>
## Open Questions

1. **teeth_bs_flag value in LAM-20K model**
   - What we know: `teeth_bs_flag` adds 4 extra dims to shapedirs. FlameHeadSubdivided passes it through from config.
   - What's unclear: Whether the LAM-20K checkpoint (`step_045500/config.json`) enables this flag.
   - Recommendation: Check `config.json` at model load time. If enabled, expression slice is `[:, :, 300:400]` not `[:, :, 300:]`.

2. **Jaw bone rotation calibration**
   - What we know: `jawOpen` maps to FLAME dim 15 with weight 0.9. The jaw bone rotation should match.
   - What's unclear: The exact axis-angle value that produces "full open" jaw matching dim 15 at weight 1.0. The `0.35` in the code example is estimated.
   - Recommendation: Calibrate empirically during Phase 3 validation — render with dim 15 = 1.0, measure the resulting jaw angle.

3. **Vertex count with teeth and oral mesh**
   - What we know: Base FLAME ~5023 vertices, 2 subdivisions → ~20K. Teeth and oral mesh add extra vertices.
   - What's unclear: Whether the SPZ export includes teeth/oral vertices or just the face.
   - Recommendation: Check `gs.xyz.shape[0]` against `flame_model.vertex_num_upsampled` during export. They should match if Gaussians correspond to upsampled FLAME vertices.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `lam/models/rendering/flame_model/flame.py` lines 690-741: FlameHeadSubdivided init, shapedirs_up construction
- `lam/models/rendering/flame_model/flame.py` lines 816-895: animation_forward implementation
- `lam/models/rendering/gs_renderer.py` lines 570-625: animate_gs_model (Gaussians = upsampled mesh vertices)
- `lam/export/arkit_flame_mapping.py`: Complete [52, 100] mapping matrix with comments
- `assets/default_arkit_to_flame.json`: Serialized mapping matrix
- Design review analysis from flame-specialist, muse-architect, systems-engineer (2026-03-19)

### Secondary (MEDIUM confidence)
- FLAME 2023 model documentation (expression PCA, LBS skinning)
- Muse SDK `packages/expression/src/au.ts`, `packages/character/src/face/FaceCompositor.ts`: Consumer-side architecture

### Tertiary (LOW confidence - needs validation)
- Jaw bone rotation calibration value (0.35 rad estimated, needs empirical validation)
- Higher FLAME dims (47-99) quality impact (flagged for Phase 3 visual comparison)
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: FLAME shapedirs_up tensor, expression blendshape extraction
- Ecosystem: PyTorch einsum, fp16 serialization, JSON manifest
- Patterns: Direct shapedirs extraction, Gaussian-major buffer layout
- Pitfalls: Dimension indexing, teeth_bs flag, jaw rotation, mapping direction

**Confidence breakdown:**
- Standard stack: HIGH — all tools already in codebase, verified
- Architecture: HIGH — verified against actual flame.py code, validated by design review
- Pitfalls: HIGH — derived from code analysis, confirmed by FLAME specialist
- Code examples: HIGH — derived directly from flame.py patterns, verified tensor shapes

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (30 days — FLAME model is stable, mapping matrix is hand-tuned)
</metadata>

---

*Phase: 01-au-deformation-basis*
*Research completed: 2026-03-19*
*Ready for planning: yes*
